use super::*;
use clap::Parser;

#[test]
#[ignore = "requires exclusive RK3588 NPU access"]
fn lossless_w8a8_matches_raw_across_batches_and_partitions() -> Result<()> {
    let _lock = crate::validation::BoardLock::acquire()?;
    let cli = crate::cli::Cli::try_parse_from(["infer", "-m", "unused", "run"])?;
    let mut options = cli.model.options()?;
    options.tune = false;
    options.tune_cache.clear();
    configure_cpu(&mut options)?;
    for mask in [1, 3, 7] {
        options.npu_mask = mask;
        // Tiny non-shrinking tile, mixed entropy with tails, and split-K.
        for (k, n) in [(32, 32), (257, 288), (4353, 257)] {
            let mut state = 0x1234_5678u32;
            let matrix = QuantMatrix {
                bits: 8,
                k,
                n,
                codes: (0..k * n)
                    .map(|i| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        if i % n < 128 && k > 32 {
                            (state % 5) as i8 - 2
                        } else {
                            state as i8
                        }
                    })
                    .collect(),
                scales: (0..n).map(|i| 0.001 + (i % 11) as f32 * 0.0001).collect(),
            };
            options.compress_w8a8 = false;
            let raw = Backend::new(&options, 14)?;
            let ordinary = raw.load(matrix.clone(), None)?;
            options.compress_w8a8 = true;
            let compressed = Backend::new(&options, 15)?;
            let encoded = compressed.load(matrix, None)?;
            assert!(!ordinary.compression_enabled());
            assert!(encoded.compression_enabled());
            assert_eq!(
                unsafe {
                    rk_npu_f32i8f32_weights_compressed_layout_generation(
                        ordinary.w8.as_ref().unwrap().raw,
                    )
                },
                0
            );
            assert!(!release_compression_scratch([&raw]));
            assert!(release_compression_scratch([&compressed]));
            assert_eq!(raw.stats()["weight_payload_bytes_estimated"], false);
            assert_eq!(compressed.stats()["compress_w8a8"], true);
            assert_eq!(compressed.stats()["weight_payload_bytes_estimated"], true);
            let first_tile = compressed.0.lock().unwrap().w8_strategies[&(encoded.key, 1)].n_tile;
            let seed_matches = first_tile as usize == n.div_ceil(32) * 32;
            for (index, rows) in [1, 3, 128, 129, 256, 1].into_iter().enumerate() {
                let input: Vec<_> = (0..rows * k)
                    .map(|i| ((i * 13 % 251) as f32 - 125.) * 0.007)
                    .collect();
                let mut expected = vec![0.; rows * n];
                let mut actual = expected.clone();
                ordinary.run(&input, &mut expected, rows)?;
                encoded.run(&input, &mut actual, rows)?;
                assert_eq!(
                    release_compression_scratch([&compressed]),
                    index == 0 && !seed_matches,
                    "different M buckets sharing a cached N layout must not request cleanup"
                );
                assert_eq!(
                    unsafe {
                        rk_npu_f32i8f32_weights_compressed_layout_generation(
                            encoded.w8.as_ref().unwrap().raw,
                        )
                    },
                    if seed_matches { 1 } else { 2 }
                );
                assert!(!release_compression_scratch([&raw]));
                assert!(expected.iter().any(|v| *v != 0.));
                assert!(
                    expected
                        .iter()
                        .zip(&actual)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "raw/DCOMP mismatch: mask={mask} M={rows} K={k} N={n}"
                );
            }
            eprintln!("PASS raw/DCOMP bit-exact: mask={mask} K={k} N={n}");
        }
    }
    Ok(())
}

fn set_n_tile(backend: &Backend, linear: &Linear, rows: usize, n_tile: i32) -> Result<()> {
    let m = bucket(rows)?;
    let mut b = backend.0.lock().unwrap();
    let mut strategy = b.w8_strategy(linear.key, m)?;
    strategy.n_tile = n_tile;
    strategy.n_groups = (linear.n().div_ceil(32) * 32).div_ceil(n_tile as usize) as i32;
    b.work.remove(&(linear.key, m));
    b.w8_strategies.insert((linear.key, m), strategy);
    Ok(())
}

#[test]
#[ignore = "requires exclusive RK3588 NPU access"]
fn compression_cleanup_follows_native_layout_creation() -> Result<()> {
    let _lock = crate::validation::BoardLock::acquire()?;
    let cli = crate::cli::Cli::try_parse_from(["infer", "-m", "unused", "--no-tune", "run"])?;
    let mut options = cli.model.options()?;
    options.npu_mask = 1;
    options.tune_cache.clear();
    configure_cpu(&mut options)?;
    let (k, n) = (96, 192);
    let matrix = QuantMatrix {
        bits: 8,
        k,
        n,
        codes: (0..k * n).map(|i| (i % 7) as i8 - 3).collect(),
        scales: vec![0.01; n],
    };
    let raw = Backend::new(&options, 14)?;
    let ordinary = raw.load(matrix.clone(), None)?;
    options.compress_w8a8 = true;
    let compressed = Backend::new(&options, 15)?;
    let encoded = compressed.load(matrix, None)?;
    assert!(release_compression_scratch([&compressed]));
    let input: Vec<_> = (0..128 * k)
        .map(|i| ((i % 97) as f32 - 48.) * 0.02)
        .collect();
    // First use retires the unused full-N seed. Returning to Ntile192 must
    // create it again, even though that partition existed at weight creation.
    for (rows, tile, generation, reclaim) in [
        (1, 64, 2, true),
        (3, 64, 2, false),
        (1, 64, 2, false),
        (3, 96, 3, true),
        (128, 96, 3, false),
        (1, 192, 4, true),
        (128, 192, 4, false),
    ] {
        set_n_tile(&compressed, &encoded, rows, tile)?;
        let mut expected = vec![0.; rows * n];
        let mut actual = expected.clone();
        ordinary.run(&input, &mut expected, rows)?;
        encoded.run(&input, &mut actual, rows)?;
        assert_eq!(expected, actual);
        assert_eq!(
            unsafe {
                rk_npu_f32i8f32_weights_compressed_layout_generation(
                    encoded.w8.as_ref().unwrap().raw,
                )
            },
            generation
        );
        assert_eq!(release_compression_scratch([&compressed]), reclaim);
    }
    Ok(())
}
