use super::*;

#[test]
#[ignore = "requires exclusive RK3588 NPU access"]
fn fp16_linear_and_noncausal_attention_match_cpu() -> Result<()> {
    let _lock = crate::validation::BoardLock::acquire()?;
    let mut options = ModelOptions {
        path: std::path::PathBuf::new(),
        context: 256,
        chunk: 128,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: false,
        tune_search: false,
        tune_cache: std::path::PathBuf::new(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        experts: Default::default(),
        speculation: Default::default(),
    };
    configure_cpu(&mut options)?;
    let device = Backend::new(&options, 11)?;
    let floats = FloatBackend::new(device.clone());
    let round = |x: f32| f16::from_f32(x).to_f32();
    let (k, n) = (96, 160);
    let weights: Vec<_> = (0..k * n)
        .map(|i| round(((i * 13 % 71) as f32 - 35.0) * 0.007))
        .collect();
    let bias: Vec<_> = (0..n).map(|i| i as f32 * 0.0001).collect();
    let linear = floats.load(&weights, &bias, k, n)?;
    let half_linear = FloatBackend::fp16_storage(device.clone()).load(&weights, &bias, k, n)?;
    for rows in [3, 129] {
        let input: Vec<_> = (0..rows * k)
            .map(|i| round(((i * 7 % 83) as f32 - 41.0) * 0.01))
            .collect();
        let mut output = vec![0.0; rows * n];
        linear.run(&input, &mut output, rows)?;
        let input_half: Vec<_> = input.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        let mut output_half = vec![0; output.len()];
        half_linear.run_half(&input_half, &mut output_half, rows)?;
        assert_eq!(
            output_half,
            output
                .iter()
                .map(|&x| f16::from_f32(x).to_bits())
                .collect::<Vec<_>>()
        );
        let mut max = 0f32;
        for r in 0..rows {
            for j in 0..n {
                let expected = (0..k)
                    .map(|i| input[r * k + i] * weights[j * k + i])
                    .sum::<f32>()
                    + bias[j];
                max = max.max((output[r * n + j] - expected).abs());
            }
        }
        eprintln!("FP16 Linear M={rows}: max_abs={max}");
        assert!(max < 0.004);
    }
    let rows = 129;
    let heads = 2;
    let d = 64;
    let width = heads * d;
    let q: Vec<_> = (0..rows * width)
        .map(|i| round(((i * 19 % 101) as f32 - 50.0) * 0.01))
        .collect();
    let k: Vec<_> = (0..q.len())
        .map(|i| round(((i * 11 % 83) as f32 - 41.0) * 0.01))
        .collect();
    let v: Vec<_> = (0..q.len())
        .map(|i| round(((i * 7 % 73) as f32 - 36.0) * 0.01))
        .collect();
    let attention = device.noncausal_attention(heads, d)?;
    let mut output = vec![0.0; q.len()];
    attention.run_full(&q, &k, &v, &mut output, rows)?;
    let mut max = 0f64;
    for h in 0..heads {
        for r in 0..rows {
            let scores: Vec<_> = (0..rows)
                .map(|s| {
                    (0..d)
                        .map(|i| q[r * width + h * d + i] as f64 * k[s * width + h * d + i] as f64)
                        .sum::<f64>()
                        / (d as f64).sqrt()
                })
                .collect();
            let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let probabilities: Vec<_> = scores.iter().map(|x| (x - peak).exp()).collect();
            let sum: f64 = probabilities.iter().sum();
            for i in 0..d {
                let expected = probabilities
                    .iter()
                    .enumerate()
                    .map(|(s, p)| p * v[s * width + h * d + i] as f64)
                    .sum::<f64>()
                    / sum;
                max = max.max((output[r * width + h * d + i] as f64 - expected).abs());
            }
        }
    }
    eprintln!("noncausal Attention: max_abs={max}");
    assert!(max < 0.01);
    let previous = output.clone();
    attention.run_full(&q, &k, &v, &mut output, rows)?;
    assert_eq!(previous, output);
    let half_values = |values: &[f32]| {
        values
            .iter()
            .map(|&x| f16::from_f32(x).to_bits())
            .collect::<Vec<_>>()
    };
    let mut half_output = vec![0; output.len()];
    attention.run_full_half(
        &half_values(&q),
        &half_values(&k),
        &half_values(&v),
        &mut half_output,
        rows,
    )?;
    assert_eq!(half_output, half_values(&output));
    Ok(())
}
