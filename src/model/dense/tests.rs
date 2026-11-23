use super::*;
use crate::validation::BoardLock;
use clap::Parser;

fn equal(expected: &[f32], actual: &[f32], label: &str) -> Result<()> {
    ensure!(expected.len() == actual.len(), "{label}: logits shape");
    ensure!(
        expected.iter().chain(actual).all(|v| v.is_finite()),
        "{label}: nonfinite logits"
    );
    if let Some((index, (a, b))) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits())
    {
        anyhow::bail!("{label}: logit {index}: {a} != {b}");
    }
    Ok(())
}

#[test]
#[ignore = "requires RK3588, RKNPU_MODEL=Qwen3-4B and RKNPU_TUNE_CACHE"]
fn paired_prefill_matches_serial_logits_and_cache() -> Result<()> {
    let _lock = BoardLock::acquire()?;
    let cli = crate::cli::Cli::try_parse_from([
        "infer",
        "--ctx-size",
        "4224",
        "--ubatch-size",
        "128",
        "--prefill-ubatches",
        "2",
        "run",
    ])?;
    let mut options = cli.model.options()?;
    options.tune_cache = std::env::var_os("RKNPU_TUNE_CACHE")
        .context("set RKNPU_TUNE_CACHE")?
        .into();
    crate::npu::configure_cpu(&mut options)?;
    let mut model = Dense::load(options)?;
    model.reserve(4224)?;
    model.prepare(128)?;
    let tokens: Vec<u32> = (1000..5224).collect();
    let continuation = [700, 701, 702, 703, 704, 705];
    for (history, prompt) in [
        (0, 256),
        (0, 257),
        (0, 384),
        (0, 512),
        (0, 513),
        (0, 1024),
        (0, 4096),
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
            for token in continuation {
                values.push(model.forward(&[token])?.to_vec());
            }
            ensure!(
                model.position() == history + prompt + continuation.len(),
                "position mismatch"
            );
            if lanes == 1 {
                reference = values;
            } else {
                for (step, (a, b)) in reference.iter().zip(&values).enumerate() {
                    equal(
                        a,
                        b,
                        &format!("history={history}, prompt={prompt}, step={step}"),
                    )?;
                }
                ensure!(
                    model.work.dual_groups > 0,
                    "paired prefill was not exercised"
                );
            }
        }
        eprintln!(
            "PASS paired prefill history={history}, prompt={prompt}, seven full logit rows bit-exact"
        );
    }
    model.options.prefill_ubatches = 1;
    model.reset()?;
    model.prefill(&tokens[..129])?;
    let expected = model.forward(&tokens[129..385])?.to_vec();
    model.options.prefill_ubatches = 2;
    model.reset()?;
    model.prefill(&tokens[..512])?;
    ensure!(model.truncate(129)?, "truncate unsupported");
    equal(
        &expected,
        model.forward(&tokens[129..385])?,
        "truncate and paired suffix",
    )?;
    let stats = model.stats();
    for name in [
        "tune_cache_misses",
        "tune_calls",
        "strategy_fallback_calls",
        "weight_reload_calls",
    ] {
        ensure!(
            stats[name].as_u64() == Some(0),
            "unexpected {name}: {}",
            stats[name]
        );
    }
    eprintln!("PASS paired prefill reset/truncate and cached strategies");
    Ok(())
}
