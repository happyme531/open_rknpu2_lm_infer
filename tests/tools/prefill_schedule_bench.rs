//! Shared fixed-clock benchmark for model adapters; excluded from normal builds.
use crate::{
    cli::Cli,
    model::{LanguageModel, ModelOptions},
    validation::BoardLock,
};
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::json;
use std::{io::Write, path::PathBuf};
fn clock(id: libc::clockid_t) -> u64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    assert_eq!(unsafe { libc::clock_gettime(id, &mut t) }, 0);
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}
pub(crate) fn run<M: LanguageModel>(
    load: impl FnOnce(ModelOptions) -> Result<M>,
    set_lanes: impl Fn(&mut M, usize),
) -> Result<()> {
    let _lock = BoardLock::acquire()?;
    let cli = Cli::try_parse_from([
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
    if let Some(cache) = std::env::var_os("RKNPU_TUNE_CACHE") {
        options.tune_cache = cache.into();
    }
    crate::npu::configure_cpu(&mut options)?;
    let mut model = load(options)?;
    model.reserve(4224)?;
    let tokens: Vec<_> = (1000..5224).collect();
    let out = PathBuf::from(
        std::env::var_os("RKNPU_SCHEDULE_REPORT").context("set RKNPU_SCHEDULE_REPORT")?,
    );
    std::fs::create_dir_all(&out)?;
    let mut stream = std::fs::File::create(out.join("samples.jsonl"))?;
    let mut samples = Vec::new();
    for prompt in [512, 4096] {
        model.prepare(prompt)?;
        let mut reference = Vec::new();
        for lanes in [1, 2] {
            set_lanes(&mut model, lanes);
            model.reset()?;
            let mut values = vec![model.forward(&tokens[..prompt])?.to_vec()];
            for token in [700, 701] {
                values.push(model.forward(&[token])?.to_vec());
            }
            if lanes == 1 {
                reference = values;
            } else {
                for (a, b) in reference.iter().zip(&values) {
                    ensure!(
                        a.len() == b.len()
                            && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "benchmark warmup logits changed"
                    );
                }
            }
        }
        eprintln!("PASS benchmark pp{prompt} plus two decode rows bit-exact");
        for cycle in 0..2 {
            for (order, lanes) in [1, 2, 2, 1].into_iter().enumerate() {
                set_lanes(&mut model, lanes);
                model.reset()?;
                let cpu = clock(libc::CLOCK_PROCESS_CPUTIME_ID);
                let start = clock(libc::CLOCK_MONOTONIC);
                model.prefill(&tokens[..prompt])?;
                let end = clock(libc::CLOCK_MONOTONIC);
                let cpu = clock(libc::CLOCK_PROCESS_CPUTIME_ID) - cpu;
                let stats = model.stats();
                ensure!(
                    stats["head_ms"].as_f64() == Some(0.),
                    "prefill evaluated output head"
                );
                let sample = json!({"prompt":prompt,"lanes":lanes,"cycle":cycle,"order":order,
                    "start_ns":start,"end_ns":end,"wall_ms":(end-start)as f64/1e6,
                    "cpu_seconds":cpu as f64/1e9,"stats":stats});
                writeln!(stream, "{sample}")?;
                stream.flush()?;
                eprintln!(
                    "SCHEDULE pp{prompt} lanes={lanes} cycle={cycle} order={order}: {} ms",
                    sample["wall_ms"]
                );
                samples.push(sample);
            }
        }
    }
    std::fs::write(
        out.join("samples.json"),
        serde_json::to_vec_pretty(&samples)?,
    )?;
    Ok(())
}
