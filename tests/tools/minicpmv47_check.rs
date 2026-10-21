//! Real MiniCPM checkpoint: held-out NLL, batch consistency and greedy chat.
use anyhow::{Result, ensure};
use clap::Parser;
use open_rknpu2_lm_infer::{
    engine::Prompt,
    model::{self, ModelOptions},
    text::TextCodec,
    validation::BoardLock,
};
use serde_json::json;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    windows: PathBuf,
    #[arg(long)]
    report: PathBuf,
    #[arg(
        long,
        value_name = "DIR",
        help = "Directory for prepared native expert weight records on disk"
    )]
    expert_store: Option<PathBuf>,
}
fn top(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}
fn save(path: &std::path::Path, report: &serde_json::Value) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(report)? + "\n")?;
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    let windows: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args.windows)?)?;
    ensure!(
        windows.len() == 8 && windows.iter().all(|w| w.len() == 512),
        "expected eight 512-token windows"
    );
    let _lock = BoardLock::acquire()?;
    let codec = TextCodec::load(&args.model)?;
    let mut model = model::load(ModelOptions {
        path: args.model.clone(),
        context: 1024,
        chunk: 128,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: false,
        tune_search: false,
        tune_cache: PathBuf::new(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        experts: model::experts::Options {
            cache: args.expert_store,
            ..Default::default()
        },
        speculation: Default::default(),
    })?;
    model.reserve(1024)?;
    model.prepare(511)?;
    let mut report = json!({"model":args.model,"windows":args.windows,"completed":false,
        "score_target_positions":[257,511],"chunk":128,"nll":[],"generation":[]});
    let mut total = 0.;
    for (i, window) in windows.iter().enumerate() {
        model.reset()?;
        model.prefill(&window[..256])?;
        let nll = model.token_nll(&window[256..511], &window[257..512])?;
        let sum: f64 = nll.iter().sum();
        total += sum;
        report["nll"]
            .as_array_mut()
            .unwrap()
            .push(json!({"window":i,"mean_nll":sum/255.,"token_nll":nll}));
        report["ppl"] = json!((total / ((i + 1) * 255) as f64).exp());
        save(&args.report, &report)?;
        println!("window={} ppl={}", i, report["ppl"]);
    }
    let prefix = &windows[0][..128];
    model.reset()?;
    let batched = model.forward(prefix)?.to_vec();
    model.reset()?;
    let mut scalar = Vec::new();
    for token in prefix {
        scalar = model.forward(&[*token])?.to_vec();
    }
    let delta: f64 = scalar
        .iter()
        .zip(&batched)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum();
    let energy: f64 = batched.iter().map(|&x| (x as f64).powi(2)).sum();
    report["chunk_consistency"] = json!({"rows":128,"nrmse":(delta/energy).sqrt(),
        "batched_top1":top(&batched),"scalar_top1":top(&scalar)});
    save(&args.report, &report)?;
    for text in [
        "小明有12个苹果，送出5个，又买了8个，现在有几个？请简短回答。",
        "用一句话解释为什么天空是蓝色的。",
    ] {
        let prompt = Prompt::Chat {
            messages: vec![json!({"role":"user","content":text})],
            tools: None,
            enable_thinking: false,
        };
        let input = codec.encode(&prompt)?;
        model.reset()?;
        let mut logits = model.forward(&input)?.to_vec();
        let mut answer = Vec::new();
        for _ in 0..48 {
            let token = top(&logits);
            if model.info().eos_tokens.contains(&token) {
                break;
            }
            answer.push(token);
            logits = model.forward(&[token])?.to_vec();
        }
        let output = codec.decode(&answer, true)?;
        println!("{text}\n{output}");
        report["generation"]
            .as_array_mut()
            .unwrap()
            .push(json!({"prompt":text,"tokens":answer,"answer":output}));
        save(&args.report, &report)?;
    }
    report["stats"] = model.stats();
    report["completed"] = json!(true);
    save(&args.report, &report)?;
    Ok(())
}
