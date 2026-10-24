//! Fixed-token logits, chunking, cancellation and compressed-cache validation.
use anyhow::{Result, ensure};
use clap::Parser;
use open_rknpu2_lm_infer::{
    model::{self, LanguageModel, ModelOptions, experts},
    validation::BoardLock,
};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::atomic::AtomicBool};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_delimiter = ',', required = true)]
    tokens: Vec<u32>,
    #[arg(long, value_delimiter = ',', default_value = "42,123,89")]
    continuation: Vec<u32>,
    #[arg(long)]
    report: PathBuf,
    /// FP32 little-endian logits: last prompt row, followed by continuation rows.
    #[arg(long)]
    reference: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "resident")]
    storage: experts::Storage,
    #[arg(long)]
    expert_store: Option<PathBuf>,
    #[arg(long, default_value_t = 8.)]
    expert_cache_percent: f64,
    #[arg(long, default_value_t = 128)]
    chunk: usize,
    #[arg(long, default_value_t = 0.01)]
    max_nrmse: f64,
}
fn evaluate(model: &mut dyn LanguageModel, args: &Args, chunk: usize) -> Result<Vec<f32>> {
    model.reset()?;
    let mut result = Vec::new();
    for tokens in args.tokens.chunks(chunk) {
        result = model.forward(tokens)?.to_vec();
    }
    for &token in &args.continuation {
        result.extend_from_slice(model.forward(&[token])?);
    }
    ensure!(
        model.position() == args.tokens.len() + args.continuation.len(),
        "Ling position mismatch"
    );
    Ok(result)
}
fn compare(expected: &[f32], actual: &[f32], vocab: usize, limit: f64) -> Result<Value> {
    ensure!(
        expected.len() == actual.len() && actual.iter().all(|x| x.is_finite()),
        "invalid logits"
    );
    let error = expected
        .iter()
        .zip(actual)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum::<f64>();
    let energy = expected.iter().map(|&a| (a as f64).powi(2)).sum::<f64>();
    let nrmse = (error / energy.max(1e-30)).sqrt();
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let matches = expected
        .chunks(vocab)
        .zip(actual.chunks(vocab))
        .filter(|(a, b)| argmax(a) == argmax(b))
        .count();
    Ok(
        json!({"nrmse":nrmse,"max_nrmse":limit,"top1_matches":matches,"rows":actual.len()/vocab,
        "passed":nrmse<=limit && matches==actual.len()/vocab}),
    )
}
fn main() -> Result<()> {
    let args = Args::parse();
    let _lock = BoardLock::acquire()?;
    let context = (args.tokens.len() + args.continuation.len() + 1)
        .max(args.chunk)
        .max(256);
    let mut model = model::load(ModelOptions {
        path: args.model.clone(),
        context,
        chunk: args.chunk,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: false,
        tune_search: false,
        tune_cache: Default::default(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        checkpoints: Default::default(),
        speculation: Default::default(),
        experts: experts::Options {
            storage: args.storage,
            cache: args.expert_store.clone(),
            cache_percent: args.expert_cache_percent,
            prefetch: false,
            direct: false,
        },
    })?;
    model.reserve(context)?;
    model.prepare(args.tokens.len())?;
    let vocab = model.info().vocab_size;
    let baseline = evaluate(model.as_mut(), &args, args.chunk)?;
    let reference = args
        .reference
        .as_ref()
        .map(|path| -> Result<Value> {
            let bytes = std::fs::read(path)?;
            ensure!(bytes.len().is_multiple_of(4), "invalid reference bytes");
            let expected: Vec<_> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            compare(&expected, &baseline, vocab, args.max_nrmse)
        })
        .transpose()?;
    let mut checks = Vec::new();
    for chunk in [1, 3, 7] {
        let actual = evaluate(model.as_mut(), &args, chunk)?;
        checks.push(
            json!({"chunk":chunk,"comparison":compare(&baseline,&actual,vocab,args.max_nrmse)?}),
        );
    }
    model.reset()?;
    if args.tokens.len() > 1 {
        model.prefill(&args.tokens[..args.tokens.len() - 1])?;
    }
    let mut head_free = model
        .forward(&args.tokens[args.tokens.len() - 1..])?
        .to_vec();
    for &token in &args.continuation {
        head_free.extend_from_slice(model.forward(&[token])?);
    }
    let prefill = compare(&baseline, &head_free, vocab, args.max_nrmse)?;
    ensure!(
        !model.truncate(1)?,
        "hybrid state claimed arbitrary rollback"
    );
    let cancelled = AtomicBool::new(true);
    ensure!(
        model
            .forward_cancellable(&args.tokens[..1], &cancelled)
            .is_err()
            && model.position() == 0,
        "cancellation failed to reset state"
    );
    let recovered = evaluate(model.as_mut(), &args, args.chunk)?;
    let recovery = compare(&baseline, &recovered, vocab, 0.)?;
    let passed = reference.as_ref().is_none_or(|r| r["passed"] == true)
        && checks.iter().all(|r| r["comparison"]["passed"] == true)
        && prefill["passed"] == true
        && recovery["passed"] == true;
    let report = json!({"model":args.model,"tokens":args.tokens,"continuation":args.continuation,"reference":reference,
        "chunking":checks,"head_free_prefill":prefill,"cancellation_reset":recovery,"stats":model.stats(),"passed":passed});
    if let Some(parent) = args.report.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.report, serde_json::to_vec_pretty(&report)?)?;
    let bytes: Vec<_> = baseline.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(args.report.with_extension("f32"), bytes)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    ensure!(passed, "Ling logits validation failed");
    Ok(())
}
