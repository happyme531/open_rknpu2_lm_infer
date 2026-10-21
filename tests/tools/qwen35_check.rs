//! Same-weight logits, chunking, cancellation and multi-row scoring on RK3588.
use anyhow::{Result, ensure};
use clap::Parser;
use open_rknpu2_lm_infer::{
    model::{self, LanguageModel, ModelOptions},
    validation::BoardLock,
};
use serde::Deserialize;
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    report: PathBuf,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    prompt: Vec<u32>,
    continuation: Vec<u32>,
    logits: PathBuf,
}
#[derive(Deserialize)]
struct Manifest {
    cases: Vec<Case>,
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "invalid FP32 reference length"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}

fn compare(expected: &[f32], actual: &[f32], vocab: usize, threshold: f64) -> serde_json::Value {
    assert_eq!(expected.len(), actual.len());
    let mut squared = 0.0;
    let mut energy = 0.0;
    let mut maximum = 0.0f32;
    for (&a, &b) in expected.iter().zip(actual) {
        squared += (a as f64 - b as f64).powi(2);
        energy += (a as f64).powi(2);
        maximum = maximum.max((a - b).abs());
    }
    let nrmse = (squared / energy.max(1e-30)).sqrt();
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let matching = expected
        .chunks(vocab)
        .zip(actual.chunks(vocab))
        .filter(|(a, b)| argmax(a) == argmax(b))
        .count();
    let rows = actual.len() / vocab;
    let finite = actual.iter().all(|v| v.is_finite());
    json!({"rows":rows,"nrmse":nrmse,"max_abs":maximum,"top1_matches":matching,"max_nrmse":threshold,
        "passed":finite && nrmse<=threshold && matching==rows})
}

fn evaluate(model: &mut dyn LanguageModel, case: &Case, chunk: usize) -> Result<Vec<f32>> {
    model.reset()?;
    model.reserve(case.prompt.len() + case.continuation.len())?;
    model.prepare(case.prompt.len())?;
    let mut logits = Vec::new();
    for tokens in case.prompt.chunks(chunk) {
        logits = model.forward(tokens)?.to_vec();
    }
    for &token in &case.continuation {
        logits.extend_from_slice(model.forward(&[token])?);
    }
    ensure!(
        model.position() == case.prompt.len() + case.continuation.len(),
        "consumed position mismatch"
    );
    Ok(logits)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let references: Manifest =
        serde_json::from_slice(&std::fs::read(args.reference.join("cases.json"))?)?;
    let _lock = BoardLock::acquire()?;
    let mut model = model::load(ModelOptions {
        path: args.model,
        context: 256,
        chunk: 32,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: false,
        tune_search: false,
        tune_cache: args.report.with_extension("cache"),
        attention_tile: 512,
        vision_dtype: Default::default(),
        experts: Default::default(),
        speculation: Default::default(),
    })?;
    let mut checks = Vec::new();
    let mut passed = true;
    let vocab = model.info().vocab_size;
    for case in &references.cases {
        let baseline = evaluate(model.as_mut(), case, usize::MAX)?;
        let expected = read_f32(&args.reference.join(&case.logits))?;
        let reference = compare(&expected, &baseline, vocab, 0.01);
        passed &= reference["passed"] == true;
        let mut chunks = Vec::new();
        for chunk in [1, 3, 7] {
            let actual = evaluate(model.as_mut(), case, chunk)?;
            let comparison = compare(&baseline, &actual, vocab, 0.001);
            passed &= comparison["passed"] == true;
            chunks.push(json!({"chunk":chunk,"comparison":comparison}));
        }
        ensure!(
            !model.truncate(1)?,
            "hybrid state must not claim arbitrary prefix rollback"
        );
        let cancelled = AtomicBool::new(true);
        ensure!(
            model
                .forward_cancellable(&[case.prompt[0]], &cancelled)
                .is_err(),
            "cancellation was ignored"
        );
        ensure!(
            model.position() == 0,
            "cancelled hybrid request retained partial state"
        );
        let after_cancel = evaluate(model.as_mut(), case, usize::MAX)?;
        let cancel_comparison = compare(&baseline, &after_cancel, vocab, 0.0);
        passed &= cancel_comparison["passed"] == true;
        checks.push(json!({"case":case.name,"reference":reference,"chunking":chunks,"cancellation_reset":cancel_comparison}));
        println!("{}", checks.last().unwrap());
    }
    let prompt = &references.cases[0].prompt;
    let mut scoring = Vec::new();
    for rows in [1, 2, 3, 5, 7] {
        model.reset()?;
        let batch = model.token_nll(&prompt[..rows], &prompt[1..rows + 1])?;
        model.reset()?;
        let mut sequential = Vec::new();
        for i in 0..rows {
            sequential.push(open_rknpu2_lm_infer::ppl::negative_log_likelihood(
                model.forward(&prompt[i..i + 1])?,
                prompt[i + 1],
            )?);
        }
        let max_error = batch
            .iter()
            .zip(&sequential)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let ok = max_error <= 0.01;
        passed &= ok;
        scoring.push(json!({"rows":rows,"max_nll_difference":max_error,"passed":ok}));
    }
    let report =
        json!({"passed":passed,"cases":checks,"batch_scoring":scoring,"stats":model.stats()});
    std::fs::write(args.report, serde_json::to_string_pretty(&report)? + "\n")?;
    ensure!(
        passed,
        "Qwen3.5 real-checkpoint correctness gate failed; see report"
    );
    Ok(())
}
