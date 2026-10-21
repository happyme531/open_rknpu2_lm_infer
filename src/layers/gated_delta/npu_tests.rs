use super::*;
use crate::{model::ModelOptions, npu::Backend};
use serde_json::json;

fn relative(actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut error = 0.;
    let mut energy = 0.;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite());
        error += (f64::from(a) - f64::from(b)).powi(2);
        energy += f64::from(b).powi(2);
    }
    (error / energy.max(1e-30)).sqrt()
}
fn series(n: usize, offset: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 17 + offset) % 251) as f32 - 125.) * scale / 125.)
        .collect()
}
#[test]
#[ignore = "requires exclusive RK3588 NPU access"]
fn npu_prefill_threshold_state_handoff_and_layer_reuse() -> Result<()> {
    let _lock = crate::validation::BoardLock::acquire()?;
    let mut options = ModelOptions {
        path: Default::default(),
        context: 512,
        chunk: 128,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: false,
        tune_search: false,
        tune_cache: Default::default(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        experts: Default::default(),
        speculation: Default::default(),
    };
    crate::npu::configure_cpu(&mut options)?;
    let device = Backend::new(&options, 12)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let dim = GdnDimensions::new(16, 32, 128, 128)?;
    let mut accelerated = Scratch::with_npu(device.clone());
    let mut merged_scratch = Scratch::with_npu(device);
    let mut merged_states = [LayerState::new(dim, 4, 8)?, LayerState::new(dim, 4, 8)?];
    accelerated.prepare(&[1, 12])?;
    assert_eq!(
        accelerated.backend_stats()["gdn_npu_workspace_rows"],
        json!([])
    );
    let mut scalar = Scratch::default();
    let mut actual = [LayerState::new(dim, 4, 8)?, LayerState::new(dim, 4, 8)?];
    let mut expected = [LayerState::new(dim, 4, 8)?, LayerState::new(dim, 4, 8)?];
    let weights: Vec<_> = (0..2)
        .map(|layer| {
            MixerWeights::new(
                dim,
                4,
                series(dim.conv_width() * 4, layer * 31, 0.2),
                vec![-2.; 32],
                series(32, layer * 13, 0.2),
                vec![1.; 128],
                1e-6,
            )
        })
        .collect::<Result<_>>()?;
    let mut cpu_calls = 0;
    let mut npu_calls = 0;
    for (step, rows) in [12, 13, 1, 32, 128, 7, 64, 35, 1].into_iter().enumerate() {
        let qkv = series(rows * dim.conv_width(), step * 29, 0.7);
        let z = series(rows * dim.value_width(), step * 19, 1.);
        let a = series(rows * 32, step * 11, 0.3);
        let b = series(rows * 32, step * 7, 0.5);
        let input = Projected::new(rows, &qkv, &z, &a, &b);
        let mut merged = Vec::new();
        for t in 0..rows {
            merged.extend_from_slice(&qkv[t * dim.conv_width()..(t + 1) * dim.conv_width()]);
            merged.extend_from_slice(&z[t * dim.value_width()..(t + 1) * dim.value_width()]);
            merged.extend_from_slice(&a[t * 32..(t + 1) * 32]);
            merged.extend_from_slice(&b[t * 32..(t + 1) * 32]);
        }
        for layer in 0..2 {
            let output = actual[layer]
                .forward(&pool, &weights[layer], input, &mut accelerated)?
                .to_vec();
            let reference = expected[layer].forward(&pool, &weights[layer], input, &mut scalar)?;
            let mut direct = vec![0.; rows * dim.value_width()];
            merged_states[layer].forward_into(
                &pool,
                &weights[layer],
                Projected::from_merged(dim, rows, &merged)?,
                &mut merged_scratch,
                &mut direct,
            )?;
            assert_eq!(direct, output);
            assert_eq!(merged_states[layer].canonical(), actual[layer].canonical());
            assert_eq!(
                merged_states[layer].convolution(),
                actual[layer].convolution()
            );
            if step == 0 {
                assert_eq!(output, reference);
                assert_eq!(actual[layer].state, expected[layer].state);
            }
            let output_error = relative(&output, reference);
            let state_error = relative(actual[layer].canonical(), expected[layer].canonical());
            assert!(
                output_error < 0.01 && state_error < 0.01,
                "M={rows} layer={layer}: output={output_error} state={state_error}"
            );
            assert_eq!(actual[layer].convolution(), expected[layer].convolution());
            assert_eq!(actual[layer].position(), expected[layer].position());
            if rows > 12 {
                npu_calls += 1;
            } else {
                cpu_calls += 1;
            }
        }
        let stats = accelerated.backend_stats();
        assert_eq!(stats["gdn_npu_calls"], json!(npu_calls));
        assert_eq!(stats["gdn_cpu_calls"], json!(cpu_calls));
        assert!(stats["gdn_npu_workspace_rows"].as_array().unwrap().len() <= 2);
    }
    // Continue directly from exported NPU state. CPU decode must be bit-exact
    // against CPU execution starting from those same canonical state bits.
    let mut copied = LayerState::new(dim, 4, 8)?;
    copied.state.clone_from(&actual[0].state);
    copied.conv = actual[0].conv.clone();
    copied.position = actual[0].position;
    let qkv = series(7 * dim.conv_width(), 17, 0.7);
    let z = series(7 * dim.value_width(), 29, 1.);
    let a = series(7 * 32, 31, 0.3);
    let b = series(7 * 32, 37, 0.5);
    let input = Projected::new(7, &qkv, &z, &a, &b);
    let before = actual[0].canonical().to_vec();
    let history = actual[0].convolution().to_vec();
    let count = accelerated.backend_stats()["gdn_npu_calls"].clone();
    let output = actual[0]
        .verify(&pool, &weights[0], input, &mut accelerated)?
        .to_vec();
    let reference = copied.verify(&pool, &weights[0], input, &mut scalar)?;
    assert_eq!(output, reference);
    assert_eq!(actual[0].canonical(), before);
    assert_eq!(actual[0].convolution(), history);
    assert_eq!(accelerated.backend_stats()["gdn_npu_calls"], count);
    actual[0].finish_round(&pool, 5)?;
    copied.finish_round(&pool, 5)?;
    assert_eq!(actual[0].canonical(), copied.canonical());
    assert_eq!(actual[0].convolution(), copied.convolution());
    let one = Projected::new(
        1,
        &qkv[..dim.conv_width()],
        &z[..dim.value_width()],
        &a[..32],
        &b[..32],
    );
    let output = actual[0]
        .forward(&pool, &weights[0], one, &mut accelerated)?
        .to_vec();
    assert_eq!(
        output,
        copied.forward(&pool, &weights[0], one, &mut scalar)?
    );
    assert_eq!(actual[0].canonical(), copied.canonical());
    // Derived invalid gates are rejected without committing state or history.
    let bad = MixerWeights::new(
        dim,
        4,
        vec![0.1; dim.conv_width() * 4],
        vec![-1000.; 32],
        vec![f32::MAX; 32],
        vec![1.; 128],
        1e-6,
    )?;
    let qkv = series(13 * dim.conv_width(), 17, 0.7);
    let z = series(13 * dim.value_width(), 29, 1.);
    let a = vec![f32::MAX; 13 * 32];
    let b = vec![0.; 13 * 32];
    let input = Projected::new(13, &qkv, &z, &a, &b);
    let before = actual[0].canonical().to_vec();
    let history = actual[0].convolution().to_vec();
    let position = actual[0].position();
    assert!(
        actual[0]
            .forward(&pool, &bad, input, &mut accelerated)
            .is_err()
    );
    assert_eq!(actual[0].canonical(), before);
    assert_eq!(actual[0].convolution(), history);
    assert_eq!(actual[0].position(), position);
    actual[0].reset();
    accelerated.reset_statistics();
    assert!(actual[0].canonical().iter().all(|&v| v == 0.));
    assert_eq!(accelerated.backend_stats()["gdn_npu_calls"], 0);
    Ok(())
}
