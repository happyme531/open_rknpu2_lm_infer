//! Real generation benchmark: includes complete draft/target/replay/head calls.
use anyhow::{Result, ensure};
use clap::Parser;
use open_rknpu2_lm_infer::{
    engine::Prompt,
    model::{self, LanguageModel, ModelOptions},
    sampling::{DraftSampling, Sampler, SamplingParams},
    speculative,
    text::TextCodec,
    validation::BoardLock,
};
use serde_json::json;
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(
        long = "spec-draft-n-max",
        visible_alias = "draft",
        default_value_t = 0
    )]
    draft: usize,
    /// Ordered configurations, e.g. 0,1,1,0 for a paired ABBA block.
    #[arg(long, value_delimiter = ',')]
    schedule: Vec<usize>,
    #[arg(
        long = "spec-draft-p-min",
        visible_alias = "draft-p-min",
        default_value_t = 0.
    )]
    draft_p_min: f64,
    #[arg(long = "spec-draft-sampling", value_enum, default_value = "greedy")]
    draft_sampling: DraftSampling,
    /// Candidate count for confidence; zero uses the full vocabulary.
    #[arg(long, default_value_t = 10)]
    draft_confidence_top_k: usize,
    /// Threshold per ordered configuration; defaults to --draft-p-min.
    #[arg(long, value_delimiter = ',')]
    p_min_schedule: Vec<f64>,
    #[arg(long)]
    draft_keep_low_confidence: bool,
    #[arg(long, value_delimiter = ',')]
    keep_low_confidence_schedule: Vec<bool>,
    #[arg(long)]
    trace_draft_confidence: bool,
    #[arg(long, default_value_t = 64)]
    steps: usize,
    #[arg(long, default_value_t = 1)]
    samples: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 128)]
    chunk: usize,
    #[arg(long, default_value_t = 1024)]
    context: usize,
    #[arg(long)]
    tune_cache: Option<PathBuf>,
    #[arg(long)]
    prompts: Option<PathBuf>,
    #[arg(long, default_value_t = open_rknpu2_lm_infer::sampling::DEFAULT_TEMPERATURE)]
    temperature: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}
fn generate(
    m: &mut dyn LanguageModel,
    tokens: &[u32],
    steps: usize,
    params: SamplingParams,
    draft_options: speculative::DraftOptions,
) -> Result<serde_json::Value> {
    let setup = Instant::now();
    m.reset()?;
    m.reserve(tokens.len() + steps)?;
    m.prepare(tokens.len())?;
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.;
    let stats_before = m.stats();
    let total = Instant::now();
    let logits = m.forward(tokens)?.to_vec();
    let prefill_ms = total.elapsed().as_secs_f64() * 1000.;
    let mut sampler = Sampler::new(params, m.info().vocab_size, tokens)?;
    let cancel = AtomicBool::new(false);
    let mut output = Vec::new();
    if steps > 0 {
        let first = sampler.sample(&logits)?;
        sampler.observe(first);
        output.push(first);
    }
    let first_ms = total.elapsed().as_secs_f64() * 1000.;
    let mut rounds = Vec::new();
    let decode = Instant::now();
    while output.len() < steps && !m.info().eos_tokens.contains(output.last().unwrap()) {
        let anchor = *output.last().unwrap();
        if m.speculation_window() > 0 {
            let r = speculative::run_round_with_options(
                m,
                &mut sampler,
                anchor,
                steps - output.len(),
                &cancel,
                draft_options,
            )?;
            let take = r
                .output
                .iter()
                .position(|t| m.info().eos_tokens.contains(t))
                .map_or(r.output.len(), |i| i + 1);
            let repair = Instant::now();
            m.finish_verification(take)?;
            let repair_ms = repair.elapsed().as_secs_f64() * 1000.;
            for &token in &r.output[..take] {
                output.push(token);
                sampler.observe(token);
            }
            rounds.push(
                json!({"proposed":r.inputs.len()-1,"accepted":r.accepted.min(take),
                "consumed":take,"draft_ms":r.draft_ms,"verify_ms":r.verify_ms,"repair_ms":repair_ms,
                "draft_attempts":r.draft_attempts,"draft_confidence":r.draft_confidence,
                "confidence_stopped":r.confidence_stopped}),
            );
        } else {
            let token = sampler.sample(m.forward(&[anchor])?)?;
            output.push(token);
            sampler.observe(token);
        }
    }
    let decode_ms = decode.elapsed().as_secs_f64() * 1000.;
    ensure!(
        m.position() == tokens.len() + output.len().saturating_sub(1),
        "pending-anchor position mismatch"
    );
    Ok(
        json!({"prompt":tokens,"tokens":output,"setup_ms":setup_ms,"prefill_ms":prefill_ms,
        "ttft_ms":setup_ms+first_ms,
        "decode_ms":decode_ms,"decode_tokens":output.len().saturating_sub(1),
        "decode_tps":if output.len()>1 {Some((output.len()-1) as f64*1000./decode_ms)} else {None},
        "generation_ms":total.elapsed().as_secs_f64()*1000.,
        "request_ms":setup_ms+total.elapsed().as_secs_f64()*1000.,"rounds":rounds,
        "stats_before":stats_before,"stats":m.stats()}),
    )
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.steps > 0 && args.samples > 0,
        "steps/samples must be positive"
    );
    let prompts: Vec<String> = if let Some(path) = &args.prompts {
        serde_json::from_slice(&std::fs::read(path)?)?
    } else {
        vec![
            "Explain the differences between TCP and UDP, with practical examples.".into(),
            "请详细解释归并排序的原理，并给出 Python 实现。".into(),
        ]
    };
    let codec = TextCodec::load(&args.model)?;
    let encoded: Vec<_> = prompts
        .iter()
        .map(|text| {
            codec.encode(&Prompt::Chat {
                messages: vec![json!({"role":"user","content":text})],
                tools: None,
                enable_thinking: false,
            })
        })
        .collect::<Result<_>>()?;
    let _lock = BoardLock::acquire()?;
    let schedule = if args.schedule.is_empty() {
        vec![args.draft]
    } else {
        args.schedule.clone()
    };
    ensure!(schedule.iter().all(|&k| k <= 7), "draft length exceeds 7");
    let p_min_schedule = if args.p_min_schedule.is_empty() {
        vec![args.draft_p_min; schedule.len()]
    } else {
        args.p_min_schedule.clone()
    };
    ensure!(
        p_min_schedule.len() == schedule.len()
            && p_min_schedule
                .iter()
                .all(|p| p.is_finite() && (0. ..=1.).contains(p)),
        "provide one p_min in [0,1] per configuration"
    );
    let keep_schedule = if args.keep_low_confidence_schedule.is_empty() {
        vec![args.draft_keep_low_confidence; schedule.len()]
    } else {
        args.keep_low_confidence_schedule.clone()
    };
    ensure!(
        keep_schedule.len() == schedule.len(),
        "provide one keep policy per configuration"
    );
    let mut m = model::load(ModelOptions {
        path: args.model,
        context: args.context,
        chunk: args.chunk,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: args.tune_cache.is_some(),
        tune_search: false,
        tune_cache: args.tune_cache.unwrap_or_default(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        speculation: open_rknpu2_lm_infer::model::SpeculativeConfig {
            kind: open_rknpu2_lm_infer::model::SpeculativeType::DraftMtp,
            max_draft: schedule[0],
            p_min: 0.,
            sampling: Default::default(),
            ..Default::default()
        },
    })?;
    let params = SamplingParams {
        temperature: args.temperature,
        seed: Some(args.seed),
        ..Default::default()
    };
    let mut samples = Vec::new();
    let mut greedy_reference = std::collections::BTreeMap::new();
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    for (block, &draft) in schedule.iter().enumerate() {
        let draft_options = speculative::DraftOptions {
            p_min: p_min_schedule[block],
            keep_low_confidence: keep_schedule[block],
            collect_confidence: args.trace_draft_confidence,
            sampling: args.draft_sampling,
            confidence_top_k: args.draft_confidence_top_k,
        };
        m.reset()?;
        if m.speculation_window() != draft {
            m.enable_mtp(draft)?;
        }
        for (i, tokens) in encoded.iter().enumerate() {
            for _ in 0..args.warmup {
                generate(
                    m.as_mut(),
                    tokens,
                    args.steps,
                    params.clone(),
                    draft_options,
                )?;
            }
            for sample in 0..args.samples {
                let mut record = generate(
                    m.as_mut(),
                    tokens,
                    args.steps,
                    params.clone(),
                    draft_options,
                )?;
                let generated = serde_json::from_value::<Vec<u32>>(record["tokens"].clone())?;
                if args.temperature == 0. {
                    if let Some(reference) = greedy_reference.get(&i) {
                        ensure!(
                            &generated == reference,
                            "greedy token mismatch at block {block}, draft {draft}, prompt {i}"
                        );
                    } else {
                        greedy_reference.insert(i, generated.clone());
                    }
                }
                record["text"] = codec
                    .tokenizer
                    .decode(&generated, true)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .into();
                record["prompt_index"] = i.into();
                record["sample"] = sample.into();
                record["block"] = block.into();
                record["draft"] = draft.into();
                record["draft_p_min"] = draft_options.p_min.into();
                record["keep_low_confidence"] = draft_options.keep_low_confidence.into();
                record["draft_sampling"] = draft_options.sampling.as_str().into();
                record["draft_confidence_top_k"] = draft_options.confidence_top_k.into();
                eprintln!(
                    "block={block} draft={draft} p_min={} prompt={i} sample={sample} tps={}",
                    draft_options.p_min, record["decode_tps"]
                );
                samples.push(record);
                let report = json!({"draft":args.draft,"schedule":schedule,
                    "p_min_schedule":p_min_schedule,"keep_low_confidence":args.draft_keep_low_confidence,
                    "keep_low_confidence_schedule":keep_schedule,
                    "trace_draft_confidence":args.trace_draft_confidence,
                    "draft_sampling":args.draft_sampling.as_str(),"draft_confidence_top_k":args.draft_confidence_top_k,
                    "chunk":args.chunk,"context":args.context,"steps":args.steps,"warmup":args.warmup,
                    "temperature":args.temperature,"seed":args.seed,"samples":samples});
                std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
            }
        }
    }
    Ok(())
}
