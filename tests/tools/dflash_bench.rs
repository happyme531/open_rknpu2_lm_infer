//! Real-text AR/MTP1/DFlash comparison, including all shared-head and repair work.
use anyhow::{Result, ensure};
use clap::Parser;
use open_rknpu2_lm_infer::{
    model::{LanguageModel, ModelOptions, dflash::DFlash, qwen35::Qwen35},
    sampling::{Sampler, SamplingParams},
    speculative,
    text::TextCodec,
    validation::BoardLock,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    mode: String,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    draft: PathBuf,
    #[arg(long)]
    inputs: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    tune_cache: PathBuf,
    #[arg(long, default_value_t = 128)]
    steps: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, value_delimiter = ',', default_value = "512,1024,3800")]
    lengths: Vec<usize>,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "ar,mtp1,dflash4,dflash8,dflash8,dflash4,mtp1,ar"
    )]
    schedule: Vec<String>,
    #[arg(long)]
    no_tune: bool,
    #[arg(long)]
    fixture: Option<PathBuf>,
}
fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.
}
fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &x)| {
            if x > best.1 { (i, x) } else { best }
        })
        .0 as u32
}
fn generate(
    model: &mut Qwen35,
    draft: &mut DFlash,
    prompt: &[u32],
    mode: &str,
    steps: usize,
    fixture: Option<&Path>,
) -> Result<Value> {
    let setup = Instant::now();
    model.reset()?;
    model.enable_mtp(usize::from(mode == "mtp1"))?;
    let block = match mode {
        "ar" | "mtp1" => 0,
        "dflash4" => 4,
        "dflash8" => 8,
        _ => anyhow::bail!("unknown mode {mode}"),
    };
    model.capture_features(if block > 0 {
        &draft.config.dflash_config.target_layer_ids
    } else {
        &[]
    })?;
    model.reserve(prompt.len() + steps)?;
    model.prepare(prompt.len())?;
    draft.reset()?;
    let setup_ms = ms(setup);
    let stats_before = model.stats();
    let draft_stats_before = draft.stats();
    let total = Instant::now();
    let first = argmax(model.forward(prompt)?);
    let target_prefill_ms = ms(total);
    let mut features = model.take_features();
    if let Some(path) = fixture.filter(|_| block > 0) {
        std::fs::create_dir_all(path)?;
        let bytes: Vec<u8> = features.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path.join(format!("{mode}-prefix.f32")), bytes)?;
    }
    let context_start = Instant::now();
    if block > 0 {
        draft.prepare_context(&features, None)?;
        features.clear();
    }
    let context_prefill_ms = ms(context_start);
    let prefill_ms = ms(total);
    let mut output = vec![first];
    let params = SamplingParams {
        temperature: 0.,
        seed: Some(42),
        ..Default::default()
    };
    let mut sampler = Sampler::new(params, model.info().vocab_size, prompt)?;
    sampler.observe(first);
    let cancel = AtomicBool::new(false);
    let mut rounds = Vec::new();
    let decode = Instant::now();
    while output.len() < steps && !model.info().eos_tokens.contains(output.last().unwrap()) {
        let anchor = *output.last().unwrap();
        let remaining = steps - output.len();
        if mode == "ar" {
            let next = argmax(model.forward(&[anchor])?);
            output.push(next);
            continue;
        }
        if mode == "mtp1" {
            let r = speculative::run_round_with_options(
                model,
                &mut sampler,
                anchor,
                remaining,
                &cancel,
                Default::default(),
            )?;
            let take = r
                .output
                .iter()
                .position(|t| model.info().eos_tokens.contains(t))
                .map_or(r.output.len(), |i| i + 1);
            let commit = Instant::now();
            model.finish_verification(take)?;
            let commit_ms = ms(commit);
            for &t in &r.output[..take] {
                output.push(t);
                sampler.observe(t);
            }
            rounds.push(json!({"proposed":r.inputs.len()-1,"accepted":r.accepted.min(take),"consumed":take,"draft_ms":r.draft_ms,"head_ms":null,"verify_ms":r.verify_ms,"commit_ms":commit_ms}));
            continue;
        }
        let rows = block.min(remaining);
        let mut proposed = vec![anchor; rows];
        let mut backbone_ms = 0.;
        let mut head_ms = 0.;
        if rows > 1 {
            proposed[1..].fill(draft.config.dflash_config.mask_token_id);
            let embeddings = model.token_embeddings(&proposed)?;
            let start = Instant::now();
            let hidden = draft.draft(&features, &embeddings, model.position(), &cancel)?;
            backbone_ms = ms(start);
            if let Some(path) = fixture.filter(|_| rounds.is_empty()) {
                std::fs::create_dir_all(path)?;
                let data = json!({"mode":mode,"position":model.position(),"tokens":proposed,"new_features":features,"embeddings":embeddings,"draft_hidden":hidden});
                std::fs::write(
                    path.join(format!("{mode}.json")),
                    serde_json::to_vec(&data)?,
                )?;
            }
            let start = Instant::now();
            let h = draft.config.hidden_size;
            let logits = model.project_hidden(&hidden[h..], rows - 1)?;
            let vocab = logits.len() / (rows - 1);
            for (token, row) in proposed[1..].iter_mut().zip(logits.chunks_exact(vocab)) {
                *token = argmax(row);
            }
            head_ms = ms(start);
        }
        let start = Instant::now();
        let logits = model.verify_tokens(&proposed, &cancel)?;
        let posterior: Vec<_> = logits
            .chunks_exact(logits.len() / rows)
            .map(argmax)
            .collect();
        let accepted = proposed[1..]
            .iter()
            .zip(&posterior)
            .take_while(|(a, b)| a == b)
            .count();
        let verify_ms = ms(start);
        let mut emitted = proposed[1..accepted + 1].to_vec();
        emitted.push(posterior[accepted]);
        let take = emitted
            .iter()
            .position(|t| model.info().eos_tokens.contains(t))
            .map_or(emitted.len(), |i| i + 1);
        let start = Instant::now();
        model.finish_verification(take)?;
        features = model.take_features();
        let commit_ms = ms(start);
        output.extend_from_slice(&emitted[..take]);
        rounds.push(json!({"proposed":rows-1,"accepted":accepted.min(take),"consumed":take,"draft_ms":backbone_ms,"head_ms":head_ms,"verify_ms":verify_ms,"commit_ms":commit_ms}));
    }
    let decode_ms = ms(decode);
    ensure!(
        output.len() == steps,
        "request stopped early: {} tokens",
        output.len()
    );
    ensure!(
        model.position() == prompt.len() + output.len() - 1,
        "pending anchor mismatch"
    );
    Ok(
        json!({"mode":mode,"prompt_tokens":prompt.len(),"tokens":output,"setup_ms":setup_ms,"target_prefill_ms":target_prefill_ms,"context_prefill_ms":context_prefill_ms,"prefill_ms":prefill_ms,"decode_ms":decode_ms,"decode_tokens":steps-1,"decode_tps":(steps-1) as f64*1000./decode_ms,"request_ms":setup_ms+ms(total),"rounds":rounds,"stats_before":stats_before,"stats":model.stats(),"draft_stats_before":draft_stats_before,"draft_stats":draft.stats()}),
    )
}
fn main() -> Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output)?;
    let _lock = BoardLock::acquire()?;
    let mut options = ModelOptions {
        path: args.model.clone(),
        context: 4096,
        chunk: 128,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: !args.no_tune,
        tune_search: args.mode == "target-tune",
        tune_cache: args.tune_cache.clone(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        experts: Default::default(),
        speculation: Default::default(),
    };
    open_rknpu2_lm_infer::npu::configure_cpu(&mut options)?;
    let mut model = Qwen35::load(options.clone())?;
    if args.mode == "target-tune" {
        for rows in [1, 2, 4, 8, 88, 128] {
            model.prepare(rows)?;
        }
        std::fs::write(
            args.output.join("target-tune.json"),
            serde_json::to_vec_pretty(&model.stats())?,
        )?;
        return Ok(());
    }
    let mut draft = DFlash::load(&options, &args.draft)?;
    ensure!(args.mode == "run", "mode must be target-tune or run");
    let inputs: Value = serde_json::from_slice(&std::fs::read(&args.inputs)?)?;
    let codec = TextCodec::load(&args.model)?;
    let mut reference: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    let mut report = json!({"model":args.model,"draft":args.draft,"draft_dtype":"fp16","steps":args.steps,"schedule":args.schedule,"completed":false,"runs":[]});
    for (block, mode) in args.schedule.iter().enumerate() {
        for &length in &args.lengths {
            let p = inputs["prompts"]
                .as_array()
                .unwrap()
                .iter()
                .find(|p| p["prompt_tokens"] == length)
                .ok_or_else(|| anyhow::anyhow!("missing input {length}"))?;
            let tokens: Vec<u32> = serde_json::from_value(p["tokens"].clone())?;
            ensure!(tokens.len() == length, "input length mismatch");
            for iteration in 0..=args.warmup {
                let mut r = generate(
                    &mut model,
                    &mut draft,
                    &tokens,
                    mode,
                    args.steps,
                    args.fixture.as_deref(),
                )?;
                let output: Vec<u32> = serde_json::from_value(r["tokens"].clone())?;
                if let Some(prior) = reference.get(&length) {
                    ensure!(
                        prior == &output,
                        "{mode} differs from greedy AR at {length}"
                    );
                } else {
                    ensure!(mode == "ar", "schedule must begin with AR");
                    reference.insert(length, output.clone());
                }
                r["text"] = codec.decode(&output, true)?.into();
                r["warmup"] = (iteration < args.warmup).into();
                r["block"] = block.into();
                eprintln!(
                    "DFLASH {mode} pp={length} iteration={iteration} decode={:.3}",
                    r["decode_tps"].as_f64().unwrap()
                );
                report["runs"].as_array_mut().unwrap().push(r);
                std::fs::write(
                    args.output.join("samples.json"),
                    serde_json::to_vec_pretty(&report)?,
                )?;
            }
        }
    }
    report["completed"] = true.into();
    std::fs::write(
        args.output.join("samples.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}
