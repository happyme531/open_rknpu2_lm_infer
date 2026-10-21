use super::*;

#[test]
#[ignore = "requires exclusive RK3588 NPU access"]
fn narrow_w8_gdn_gates_match_integer_oracle() -> Result<()> {
    let _lock = crate::validation::BoardLock::acquire()?;
    let cache = tempfile::tempdir()?;
    let mut options = ModelOptions {
        path: cache.path().into(),
        context: 128,
        chunk: 32,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: false,
        tune_search: false,
        tune_cache: cache.path().into(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        experts: Default::default(),
        speculation: Default::default(),
    };
    configure_cpu(&mut options)?;
    let backend = Backend::new(&options, 15)?;
    for n in [32, 64] {
        let k = 2560;
        let codes: Vec<i8> = (0..k * n)
            .map(|i| ((i * 13 + i / n * 7) % 255) as i16 - 127)
            .map(|v| v as i8)
            .collect();
        let scales: Vec<f32> = (0..n).map(|i| 0.001 + (i % 11) as f32 * 0.0001).collect();
        let linear = backend.load(
            QuantMatrix {
                bits: 8,
                k,
                n,
                codes: codes.clone(),
                scales: scales.clone(),
            },
            None,
        )?;
        for rows in [1, 2, 3, 5, 7] {
            backend.prepare(rows)?;
            let input: Vec<f32> = (0..rows * k)
                .map(|i| ((i * 17 + i / k * 3) % 127) as f32 / 63.0 - 1.0)
                .collect();
            let mut output = vec![0.0; rows * n];
            linear.run(&input, &mut output, rows)?;
            for r in 0..rows {
                let row = &input[r * k..(r + 1) * k];
                let scale = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / 127.0;
                let quant: Vec<i32> = row
                    .iter()
                    .map(|v| (v * scale.recip()).round().clamp(-127.0, 127.0) as i32)
                    .collect();
                for col in 0..n {
                    let sum: i32 = (0..k).map(|i| quant[i] * codes[i * n + col] as i32).sum();
                    let expected = (sum as f32 * scale) * scales[col];
                    let actual = output[r * n + col];
                    ensure!(
                        (actual - expected).abs() <= 2e-6 * expected.abs().max(1.0),
                        "N={n} M={rows}: {actual} != {expected}"
                    );
                }
            }
        }
    }
    Ok(())
}
