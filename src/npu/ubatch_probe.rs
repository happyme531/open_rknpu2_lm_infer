//! Ignored board experiments for selecting a physical prefill batch size.
use super::*;
use crate::model::checkpoint::{QuantLoader, TensorStore, concatenate};
use std::{path::PathBuf, time::Instant};

#[test]
#[ignore = "requires RKNPU_MODEL and exclusive RK3588 NPU access"]
fn real_weight_ubatch_probe() -> Result<()> {
    let _lock = crate::validation::BoardLock::acquire()?;
    let batches = std::env::var("RKNPU_PROBE_BATCHES")
        .unwrap_or_else(|_| "64,128,256".into())
        .split(',')
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(
        !batches.is_empty() && batches.iter().all(|m| (1..=512).contains(m)),
        "probe batches must be within 1..512"
    );
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let rows = batches.iter().try_fold(1usize, |rows, &m| {
        rows.checked_mul(m / gcd(rows, m))
            .context("probe row count overflow")
    })?;
    ensure!(rows <= 4096, "probe common row count exceeds 4096");
    let order: Vec<_> = batches
        .iter()
        .chain(batches.iter().rev())
        .copied()
        .collect();
    let path = PathBuf::from(std::env::var_os("RKNPU_MODEL").context("set RKNPU_MODEL")?);
    let config: Value = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    let dim = |key: &str| config[key].as_u64().unwrap() as usize;
    let h = dim("hidden_size");
    let f = dim("intermediate_size");
    let d = config["head_dim"]
        .as_u64()
        .map_or(h / dim("num_attention_heads"), |v| v as usize);
    let q = dim("num_attention_heads") * d;
    let kv = dim("num_key_value_heads") * d;
    let loader = QuantLoader::new(TensorStore::open(&path)?, &config)?;
    let mut options = ModelOptions {
        path: path.clone(),
        context: 4096,
        chunk: 64,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: false,
        tune_search: false,
        tune_cache: PathBuf::new(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        experts: Default::default(),
        speculation: Default::default(),
    };
    configure_cpu(&mut options)?;
    let ctx = Handle::new(unsafe { rk_npu_open(ptr::null()) }, rk_npu_close, "NPU")?;
    let domain = Handle::new(
        unsafe { rk_npu_iommu_domain_create(ctx.raw, 15) },
        rk_npu_iommu_domain_free,
        "domain",
    )?;
    let groups = [
        (
            "qkv",
            h,
            vec![
                ("self_attn.q_proj", q),
                ("self_attn.k_proj", kv),
                ("self_attn.v_proj", kv),
            ],
        ),
        ("o", q, vec![("self_attn.o_proj", h)]),
        ("gate_up", h, vec![("mlp.gate_proj", f), ("mlp.up_proj", f)]),
        ("down", f, vec![("mlp.down_proj", h)]),
    ];
    for (name, k, modules) in groups {
        let layer = (0..dim("num_hidden_layers"))
            .find(|&i| {
                modules
                    .iter()
                    .all(|(m, _)| loader.bits(&format!("model.layers.{i}.{m}")).ok() == Some(4))
            })
            .with_context(|| format!("no all-W4 representative for {name}"))?;
        let matrix = concatenate(
            modules
                .iter()
                .map(|(m, n)| loader.load(&format!("model.layers.{layer}.{m}"), *n, k))
                .collect::<Result<_>>()?,
        )?;
        let n = matrix.n;
        let wc = rk_npu_i4_weight_config {
            K: k as i32,
            N: n as i32,
            k_tile: 0,
        };
        let weights = Handle::new(
            unsafe {
                rk_npu_w4a8_weights_create(
                    domain.raw,
                    &wc,
                    matrix.codes.as_ptr(),
                    matrix.scales.as_ptr(),
                )
            },
            rk_npu_w4a8_weights_free,
            "W4",
        )?;
        let mut info = rk_npu_w4a8_weight_info::default();
        check(
            unsafe { rk_npu_w4a8_weights_query(weights.raw, &mut info) },
            "weight query",
        )?;
        ensure!(
            info.bound_relaxed == 0 && info.multiplier == 1.0,
            "uncertified weights"
        );
        let mut strategies = BTreeMap::new();
        for &m in &batches {
            let mut cfg = rk_npu_w4a8_autotune_config::default();
            unsafe { rk_npu_w4a8_autotune_config_init(&mut cfg, m as i32, n as i32, k as i32) };
            cfg.required_k_tile = info.config.k_tile;
            cfg.k_tile_multiplier = 1.0;
            cfg.allowed_cpu_core_mask = 0xf0;
            cfg.allowed_npu_core_mask = 7;
            cfg.warmup = 1;
            cfg.loops = 2;
            cfg.repeats = 2;
            let mut fast = rk_npu_w4a8_strategy_ex::default();
            let mut stable = rk_npu_w4a8_strategy_ex::default();
            eprintln!(
                "Probe {name} layer={layer} M={m} K={k} N={n} Ktile={}",
                info.config.k_tile
            );
            check(
                unsafe {
                    rk_npu_w4a8_autotune_ex(
                        ctx.raw,
                        &cfg,
                        matrix.codes.as_ptr(),
                        matrix.scales.as_ptr(),
                        &mut fast,
                        &mut stable,
                    )
                },
                "operator tune",
            )?;
            strategies.insert(m, fast);
        }
        drop(matrix);
        let input: Vec<f32> = (0..rows * k)
            .map(|i| ((i * 17 + i / k * 3) % 127) as f32 / 63.0 - 1.0)
            .collect();
        let mut work = BTreeMap::new();
        for (&m, s) in &strategies {
            work.insert(
                m,
                Handle::new(
                    unsafe {
                        rk_npu_w4a8_workspace_create_ex(domain.raw, &s.base.config, &s.execution)
                    },
                    rk_npu_w4a8_workspace_free,
                    "probe workspace",
                )?,
            );
        }
        // Compare equal work, using a common multiple of the requested batches.
        // All outputs are checked against the first batch; timings include
        // activation quantization, packing, submits, waits and CPU output.
        for fused in [false, true] {
            if fused && name != "gate_up" {
                continue;
            }
            let width = if fused { n / 2 } else { n };
            let mut output = vec![0.0; rows * width];
            let mut reference = vec![];
            for round in 0..3 {
                for &m in &order {
                    let run = |out: &mut [f32]| -> Result<()> {
                        for row in (0..rows).step_by(m) {
                            let rc = unsafe {
                                if fused {
                                    rk_npu_w4a8_run_swiglu_f32(
                                        work[&m].raw,
                                        weights.raw,
                                        input.as_ptr().add(row * k),
                                        out.as_mut_ptr().add(row * width),
                                        ptr::null_mut(),
                                    )
                                } else {
                                    rk_npu_w4a8_run_f32(
                                        work[&m].raw,
                                        weights.raw,
                                        input.as_ptr().add(row * k),
                                        out.as_mut_ptr().add(row * width),
                                        ptr::null_mut(),
                                    )
                                }
                            };
                            check(rc, "probe run")?;
                        }
                        Ok(())
                    };
                    run(&mut output)?;
                    let start = Instant::now();
                    run(&mut output)?;
                    let us = start.elapsed().as_secs_f64() * 1e6;
                    if reference.is_empty() {
                        reference = output.clone();
                    }
                    let max_error = output
                        .iter()
                        .zip(&reference)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    ensure!(output.iter().zip(&reference).all(|(a,b)|a.is_finite()&&(a-b).abs()<=3e-5*b.abs().max(1.0)),"batch output mismatch");
                    let s = &strategies[&m];
                    println!(
                        "UBATCH_PROBE {}",
                        json!({"operator":name,"layer":layer,"K":k,"N":n,"M":m,"rows":rows,"fused_swiglu":fused,"round":round,"total_us":us,"us_per_token":us/rows as f64,"max_abs_error":max_error,"m_tile":s.base.config.m_tile,"k_tile":s.base.config.k_tile,"n_tile":s.base.config.n_tile,"layout":s.execution.input_layout})
                    );
                }
            }
        }
        loader.store.release_pages()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires RKNPU_MODEL and exclusive RK3588 NPU access"]
fn large_ubatch_model_sweep() -> Result<()> {
    let _lock = crate::validation::BoardLock::acquire()?;
    let path = PathBuf::from(std::env::var_os("RKNPU_MODEL").context("set RKNPU_MODEL")?);
    let report =
        PathBuf::from(std::env::var_os("RKNPU_UBATCH_REPORT").context("set RKNPU_UBATCH_REPORT")?);
    let cache =
        PathBuf::from(std::env::var_os("RKNPU_TUNE_CACHE").context("set RKNPU_TUNE_CACHE")?);
    let prompts = [768, 1024];
    let tokens: Vec<u32> = (1000..2024).collect();
    let continuation: Vec<u32> = (700..708).collect();
    let mut reference = BTreeMap::<usize, Vec<f32>>::new();
    let mut results = vec![];
    let mut all_exact = true;
    for chunk in [128, 192, 256] {
        eprintln!("UBATCH_MODEL loading batch={chunk}");
        let started = Instant::now();
        // This explicit ignored experiment prepares missing strategies offline,
        // then verifies that preparation and search stay outside timed calls.
        let mut model = crate::model::load(ModelOptions {
            path: path.clone(),
            context: 8192,
            chunk,
            threads: 4,
            npu_mask: 7,
            cpu_mask: 0xf0,
            tune: true,
            tune_search: true,
            tune_cache: cache.clone(),
            attention_tile: 512,
            vision_dtype: Default::default(),
            experts: Default::default(),
            speculation: Default::default(),
        })?;
        model.reserve(tokens.len() + continuation.len())?;
        let loaded_ms = started.elapsed().as_secs_f64() * 1000.0;
        for rows in prompts {
            eprintln!("UBATCH_MODEL preparing batch={chunk} prompt={rows}");
            let prepare = Instant::now();
            model.prepare(rows)?;
            let prepare_ms = prepare.elapsed().as_secs_f64() * 1000.0;
            let before = model.stats();
            let mut samples = vec![];
            let mut logits = vec![];
            for sample in 0..4 {
                model.reset()?;
                model.prepare(rows)?;
                let timed = Instant::now();
                let values = model.forward(&tokens[..rows])?;
                let ms = timed.elapsed().as_secs_f64() * 1000.0;
                if sample == 3 {
                    logits.extend_from_slice(values);
                }
                if sample > 0 {
                    samples.push(ms);
                }
                eprintln!("UBATCH_MODEL batch={chunk} prompt={rows} sample={sample} ms={ms:.3}");
            }
            let decode = Instant::now();
            for &token in &continuation {
                logits.extend_from_slice(model.forward(&[token])?);
            }
            let decode_ms = decode.elapsed().as_secs_f64() * 1000.0;
            let after = model.stats();
            for counter in [
                "tune_calls",
                "weight_reload_calls",
                "strategy_fallback_calls",
            ] {
                ensure!(
                    before[counter] == after[counter],
                    "timed path changed {counter}"
                );
            }
            let baseline = reference.entry(rows).or_insert_with(|| logits.clone());
            ensure!(baseline.len() == logits.len(), "logit shape mismatch");
            let exact = baseline
                .iter()
                .zip(&logits)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            all_exact &= exact;
            let max_error = baseline
                .iter()
                .zip(&logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let squared_error: f64 = baseline
                .iter()
                .zip(&logits)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum();
            let squared_norm: f64 = baseline.iter().map(|v| f64::from(*v).powi(2)).sum();
            let quality = json!({"values":logits.len(),"bitwise_equal":exact,"max_abs_error":max_error,"relative_l2":(squared_error/squared_norm.max(f64::MIN_POSITIVE)).sqrt()});
            let dump = report.with_file_name(format!("ubatch-{chunk}-pp{rows}.f32"));
            let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(dump, bytes)?;
            let speeds: Vec<_> = samples.iter().map(|ms| rows as f64 * 1000.0 / ms).collect();
            let result = json!({"batch":chunk,"prompt":rows,"model_load_and_initial_tuning_ms":loaded_ms,"prepare_ms":prepare_ms,"samples_ms":samples,"samples_tps":speeds,"mean_tps":speeds.iter().sum::<f64>()/speeds.len() as f64,"decode_calls":continuation.len(),"decode_ms":decode_ms,"quality":quality,"stats_before":before,"stats_after":after});
            println!("UBATCH_MODEL_RESULT {result}");
            results.push(result);
            std::fs::write(
                &report,
                serde_json::to_vec_pretty(
                    &json!({"context":8192,"batches":[128,192,256],"order":"128,192,256","warmup":1,"repetitions":3,"token_ids":tokens,"continuation":continuation,"timing":"complete synchronous model forward including output head; excludes offline preparation and weight loading","results":results}),
                )?,
            )?;
        }
    }
    ensure!(
        all_exact,
        "batch changes altered logits; retain raw dumps for analysis"
    );
    Ok(())
}

#[test]
#[ignore = "requires exclusive RK3588 NPU access"]
fn large_ubatch_attention_splitting() -> Result<()> {
    let _lock = crate::validation::BoardLock::acquire()?;
    let mut options = ModelOptions {
        path: PathBuf::new(),
        context: 512,
        chunk: 256,
        threads: 2,
        npu_mask: 1,
        cpu_mask: 0x30,
        tune: false,
        tune_search: false,
        tune_cache: PathBuf::new(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        experts: Default::default(),
        speculation: Default::default(),
    };
    configure_cpu(&mut options)?;
    let full_backend = Backend::new(&options, 15)?;
    let full = full_backend.attention(4, 1, 128, 1.0 / 128.0f32.sqrt())?;
    options.chunk = 128;
    let baseline_backend = Backend::new(&options, 14)?;
    let baseline = baseline_backend.attention(4, 1, 128, 1.0 / 128.0f32.sqrt())?;
    for rows in [129, 192, 193, 256] {
        full.truncate(0)?;
        baseline.truncate(0)?;
        let q: Vec<f32> = (0..rows * 4 * 128)
            .map(|i| ((i * 19) % 101) as f32 * 0.006 - 0.3)
            .collect();
        let k: Vec<f32> = (0..rows * 128)
            .map(|i| ((i * 7) % 53) as f32 * 0.009 - 0.2)
            .collect();
        let v: Vec<f32> = (0..rows * 128)
            .map(|i| ((i * 11) % 71) as f32 * 0.012 - 0.4)
            .collect();
        let mut actual = vec![0.0; q.len()];
        let mut expected = vec![0.0; q.len()];
        full.run(&q, &k, &v, &mut actual, rows, 0)?;
        baseline.run(
            &q[..128 * 4 * 128],
            &k[..128 * 128],
            &v[..128 * 128],
            &mut expected[..128 * 4 * 128],
            128,
            0,
        )?;
        baseline.run(
            &q[128 * 4 * 128..],
            &k[128 * 128..],
            &v[128 * 128..],
            &mut expected[128 * 4 * 128..],
            rows - 128,
            128,
        )?;
        assert_eq!(actual, expected, "Attention split mismatch at {rows} rows");
        println!("UBATCH_ATTENTION rows={rows} bitwise_equal=true");
    }
    Ok(())
}
