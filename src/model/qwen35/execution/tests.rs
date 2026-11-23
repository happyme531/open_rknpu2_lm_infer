use super::*;
use crate::{
    cli::Cli,
    model::{LanguageModel, SpeculativeType},
    validation::BoardLock,
};
use anyhow::Context;
use clap::Parser;

fn state(model: &Qwen35) -> Vec<f32> {
    model
        .sequences
        .single()
        .mixers
        .iter()
        .flat_map(|layer| match &*layer {
            crate::model::qwen35::decoder::LayerSeqState::Recurrent(state) => state
                .canonical()
                .iter()
                .chain(state.convolution())
                .copied()
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}
fn exact(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|v| v.is_finite()));
    assert!(
        a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
        "pipeline changed FP32 values"
    );
}

#[test]
#[ignore = "requires RK3588 and RKNPU_MTP_MODEL"]
fn paired_prefill_preserves_recurrence_features_and_decode() -> Result<()> {
    let _lock = BoardLock::acquire()?;
    let path = std::env::var("RKNPU_MTP_MODEL").context("set RKNPU_MTP_MODEL")?;
    let cli = Cli::try_parse_from([
        "infer",
        "-m",
        &path,
        "--ctx-size",
        "1536",
        "--prefill-ubatches",
        "2",
        "--no-tune",
        "run",
    ])?;
    let mut options = cli.model.options()?;
    let mtp = std::env::var("RKNPU_PIPELINE_MTP").as_deref() == Ok("1");
    if mtp {
        options.speculation.kind = SpeculativeType::DraftMtp;
    }
    crate::npu::configure_cpu(&mut options)?;
    let mut model = Qwen35::load(options)?;
    if mtp {
        model.enable_mtp(3)?;
    }
    model.capture_features(&[0, model.weights.layers.len() - 1])?;
    model.reserve(1536)?;
    model.prepare(128)?;
    let tokens: Vec<u32> = (1000..2536).collect();
    for (history, prompt) in [
        (0, 256),
        (0, 257),
        (0, 384),
        (0, 513),
        (0, 1024),
        (1, 256),
        (128, 256),
    ] {
        let mut reference = Vec::new();
        for lanes in [1, 2] {
            model.options.prefill_ubatches = lanes;
            model.reset()?;
            if history > 0 {
                model.prefill(&tokens[..history])?;
            }
            let mut values = vec![model.forward(&tokens[history..history + prompt])?.to_vec()];
            for token in [700, 701] {
                values.push(model.forward(&[token])?.to_vec());
            }
            values.push(state(&model));
            values.push(model.take_features());
            if mtp {
                values.push(model.draft_logits(42, &AtomicBool::new(false))?.to_vec());
                model.abort_verification()?;
            }
            assert_eq!(model.position(), history + prompt + 2);
            if lanes == 1 {
                reference = values;
            } else {
                for (a, b) in reference.iter().zip(&values) {
                    exact(a, b);
                }
                assert!(model.dual_groups > 0);
            }
        }
        eprintln!(
            "PASS Qwen3.5 history={history} prompt={prompt}: logits, recurrence, convolution, features and continuation bit-exact; mtp={mtp}"
        );
    }
    let positions: Vec<_> = (0..257).map(|p| [p, p / 2, p / 3]).collect();
    let h = model.weights.config.hidden_size;
    let features = [(120, std::sync::Arc::<[f32]>::from(vec![0.125; 16 * h]))];
    let mut expected = Vec::new();
    for lanes in [1, 2] {
        model.reset()?;
        let steps = crate::model::PrefillStep::chunks(257, 128);
        for group in crate::model::prefill::groups(&steps, 128, lanes) {
            let inputs: Vec<_> = group
                .iter()
                .map(|s| {
                    (
                        &tokens[s.range.clone()],
                        Some(EmbeddedChunk {
                            positions: &positions[s.range.clone()],
                            features: &features,
                        }),
                    )
                })
                .collect();
            model.execute(&inputs, None, false)?;
        }
        model.output_head(1)?;
        let values = vec![
            model.logits.clone(),
            state(&model),
            model.take_features(),
            model.forward(&[700])?.to_vec(),
        ];
        if lanes == 1 {
            expected = values;
        } else {
            for (a, b) in expected.iter().zip(&values) {
                exact(a, b);
            }
        }
    }
    eprintln!(
        "PASS Qwen3.5 prepared three-axis positions and feature span crossing a ubatch boundary; mtp={mtp}"
    );
    let cancel = AtomicBool::new(true);
    assert!(model.forward_cancellable(&tokens[..256], &cancel).is_err());
    assert_eq!(model.position(), 0);
    Ok(())
}
