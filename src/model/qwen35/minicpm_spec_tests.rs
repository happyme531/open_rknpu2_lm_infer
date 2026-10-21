//! Real target/draft round-trip, using the production sampling and commit driver.
use super::Qwen35;
use crate::{
    engine::Prompt,
    model::{LanguageModel, ModelOptions, SpeculativeConfig, SpeculativeType},
    sampling::{Sampler, SamplingParams},
    text::TextCodec,
    validation::BoardLock,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};

fn generate(model: &mut Qwen35, prompt: &[u32], limit: usize) -> Result<(Vec<u32>, Value)> {
    model.reset()?;
    model.reserve(prompt.len() + limit)?;
    model.prepare(prompt.len())?;
    let mut sampler = Sampler::new(SamplingParams::default(), model.info().vocab_size, prompt)?;
    let begin = Instant::now();
    let first = sampler.sample(model.forward(prompt)?)?;
    let prefill_ms = begin.elapsed().as_secs_f64() * 1000.;
    let mut output = vec![first];
    sampler.observe(first);
    let cancel = AtomicBool::new(false);
    let mut rounds = Vec::new();
    let decode = Instant::now();
    while output.len() < limit && !model.info().eos_tokens.contains(output.last().unwrap()) {
        let anchor = *output.last().unwrap();
        if model.speculation_window() == 0 {
            let next = sampler.sample(model.forward(&[anchor])?)?;
            output.push(next);
            sampler.observe(next);
            continue;
        }
        let round = crate::speculative::run_round(
            model,
            &mut sampler,
            anchor,
            limit - output.len(),
            &cancel,
        )?;
        let take = round
            .output
            .iter()
            .position(|t| model.info().eos_tokens.contains(t))
            .map_or(round.output.len(), |i| i + 1);
        let commit = Instant::now();
        model.finish_verification(take)?;
        let commit_ms = commit.elapsed().as_secs_f64() * 1000.;
        for &token in &round.output[..take] {
            output.push(token);
            sampler.observe(token);
        }
        rounds.push(
            json!({"proposed":round.inputs.len()-1,"accepted":round.accepted,"emitted":take,
            "draft_ms":round.draft_ms,"verify_ms":round.verify_ms,"commit_ms":commit_ms}),
        );
    }
    let decode_ms = decode.elapsed().as_secs_f64() * 1000.;
    let stats = json!({"prefill_ms":prefill_ms,"decode_ms":decode_ms,"decode_tokens":output.len()-1,
        "decode_tps":(output.len()-1) as f64*1000./decode_ms,"rounds":rounds,"stats":model.stats()});
    Ok((output, stats))
}

#[test]
#[ignore = "requires MiniCPM target, prepared expert store, DSpark checkpoint and RK3588"]
fn minicpm_dspark_generation_matches_ar() -> Result<()> {
    let _lock = BoardLock::acquire()?;
    let path = PathBuf::from(std::env::var("MINICPM_TEST_MODEL")?);
    let codec = TextCodec::load(&path)?;
    let mut options = ModelOptions {
        path,
        context: 512,
        chunk: 128,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: false,
        tune_search: false,
        tune_cache: PathBuf::new(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        experts: crate::model::experts::Options {
            cache: Some(std::env::var("MINICPM_TEST_STORE")?.into()),
            ..Default::default()
        },
        speculation: SpeculativeConfig {
            kind: SpeculativeType::DraftDspark,
            model: Some(std::env::var("MINICPM_TEST_DSPARK")?.into()),
            max_draft: 15,
            ..Default::default()
        },
    };
    crate::npu::configure_cpu(&mut options)?;
    let mut model = Qwen35::load(options)?;
    let draft = model.dflash.take();
    model.capture_features(&[])?;
    let text = "列出从1到20的整数，用逗号分隔，不要解释。";
    let prompt = codec.encode(&Prompt::Chat {
        messages: vec![json!({"role":"user","content":text})],
        tools: None,
        enable_thinking: false,
    })?;
    let (reference, ar) = generate(&mut model, &prompt, 64)?;
    let mut report = json!({"prompt":text,"ar":{"tokens":reference,"text":codec.decode(&reference,true)?,"timing":ar},"dspark":[]});
    model.reset()?;
    model.dflash = draft;
    let layers = model
        .dflash
        .as_ref()
        .unwrap()
        .config
        .dflash_config
        .target_layer_ids
        .clone();
    model.capture_features(&layers)?;
    for maximum in [1, 3, 7, 15] {
        model.options.speculation.max_draft = maximum;
        let (tokens, timing) = generate(&mut model, &prompt, 64)?;
        let identical = tokens == reference;
        report["dspark"].as_array_mut().unwrap().push(
            json!({"max_draft":maximum,"identical":identical,"tokens":tokens,"timing":timing}),
        );
        if let Ok(path) = std::env::var("MINICPM_TEST_REPORT") {
            std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        }
        ensure!(
            identical,
            "DSpark {maximum} output differs from greedy autoregressive output"
        );
        println!("DSpark max_draft={maximum} matches AR");
    }
    Ok(())
}
