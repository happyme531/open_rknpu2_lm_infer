//! Fixed real-text pp/tg comparison with phase-local expert and process I/O counters.
use anyhow::{Result, ensure};
use clap::Parser;
use open_rknpu2_lm_infer::{
    engine::Prompt,
    model::{self, ModelOptions, experts},
    text::TextCodec,
    validation::BoardLock,
};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    text: PathBuf,
    #[arg(long)]
    report: PathBuf,
    #[arg(
        long,
        value_name = "DIR",
        help = "Directory for prepared native expert weight records on disk"
    )]
    expert_store: PathBuf,
    #[arg(long, value_enum, default_value = "resident")]
    storage: experts::Storage,
    #[arg(long)]
    no_prefetch: bool,
    #[arg(
        long,
        default_value_t = experts::DEFAULT_CACHE_PERCENT,
        value_parser = experts::parse_cache_percent,
        help = "Percentage of routed experts kept in NPU cache per layer (at least top-k)"
    )]
    expert_cache_percent: f64,
    #[arg(long, default_value_t = 512)]
    prompt: usize,
    #[arg(long, default_value_t = 128)]
    generate: usize,
}

fn storage_io() -> Result<Value> {
    if let Ok(text) = std::fs::read_to_string("/proc/self/io") {
        let mut counters = serde_json::Map::new();
        counters.insert("source".into(), json!("/proc/self/io"));
        for line in text.lines() {
            if let Some((name, value)) = line.split_once(':') {
                counters.insert(name.into(), json!(value.trim().parse::<u64>()?));
            }
        }
        return Ok(Value::Object(counters));
    }
    let text = std::fs::read_to_string("/sys/class/block/mmcblk0/stat")?;
    let fields = text
        .split_whitespace()
        .map(str::parse::<u64>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(fields.len() >= 11, "incomplete eMMC device counters");
    Ok(
        json!({"source":"/sys/class/block/mmcblk0/stat","scope":"whole eMMC device",
        "read_bytes":fields[2]*512,"write_bytes":fields[6]*512,
        "read_ms":fields[3],"write_ms":fields[7],"busy_ms":fields[9]}),
    )
}

fn save(args: &Args, report: &Value) -> Result<()> {
    std::fs::write(&args.report, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.prompt > 0 && args.generate > 0,
        "empty benchmark phase"
    );
    let _lock = BoardLock::acquire()?;
    let codec = TextCodec::load(&args.model)?;
    let text = std::fs::read_to_string(&args.text)?;
    let tokens = codec.encode(&Prompt::Text(text))?;
    let capacity = args.prompt + args.generate;
    ensure!(tokens.len() >= capacity, "insufficient real-text tokens");
    let tokens = &tokens[..capacity];
    let mut report = json!({"model":args.model,"text":args.text,
        "storage":format!("{:?}",args.storage),"prefetch":!args.no_prefetch,
        "expert_cache_percent":args.expert_cache_percent,
        "prompt":args.prompt,"generate":args.generate,"ubatch":128,
        "tokens":tokens,"completed":false,"steps":[],
        "protocol":"continuous real-text teacher-forced IDs; head-free pp; one LM head per tg token; excludes tokenization, load, reserve, preparation, statistics and output writes"});
    save(&args, &report)?;
    let start = Instant::now();
    let mut model = model::load(ModelOptions {
        path: args.model.clone(),
        context: capacity.max(1024),
        chunk: 128,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: false,
        tune_search: false,
        tune_cache: PathBuf::new(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        speculation: Default::default(),
        experts: experts::Options {
            storage: args.storage,
            cache: Some(args.expert_store.clone()),
            cache_percent: args.expert_cache_percent,
            prefetch: !args.no_prefetch,
            ..Default::default()
        },
    })?;
    report["load_seconds"] = json!(start.elapsed().as_secs_f64());
    let start = Instant::now();
    model.reserve(capacity)?;
    model.prepare(args.prompt)?;
    report["prepare_seconds"] = json!(start.elapsed().as_secs_f64());
    model.reset()?;
    report["before"] = model.stats();
    report["expert_slots"] = report["before"]["expert_paging"]["slots_per_layer"].clone();
    report["io_before"] = storage_io()?;
    save(&args, &report)?;
    let start = Instant::now();
    model.prefill(&tokens[..args.prompt])?;
    let pp = start.elapsed().as_secs_f64();
    report["prefill_seconds"] = json!(pp);
    report["prefill_tokens_per_second"] = json!(args.prompt as f64 / pp);
    report["after_prefill"] = model.stats();
    report["io_after_prefill"] = storage_io()?;
    println!("prefill {} tokens: {pp:.6} s", args.prompt);
    save(&args, &report)?;
    let mut outputs = Vec::new();
    let mut total = 0.;
    let decode_start = Instant::now();
    for (step, &token) in tokens[args.prompt..].iter().enumerate() {
        let start = Instant::now();
        let logits = model.forward(&[token])?;
        let elapsed = start.elapsed().as_secs_f64();
        total += elapsed;
        let top = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i);
        outputs.extend(logits.iter().flat_map(|x| x.to_le_bytes()));
        report["steps"].as_array_mut().unwrap().push(
            json!({"step":step,"position":args.prompt+step,"token":token,"top1":top,"seconds":elapsed}),
        );
        if (step + 1) % 16 == 0 {
            println!("decode {}/{}: {total:.6} s", step + 1, args.generate);
        }
    }
    let decode = decode_start.elapsed().as_secs_f64();
    report["decode_seconds"] = json!(decode);
    report["decode_forward_seconds"] = json!(total);
    report["decode_tokens_per_second"] = json!(args.generate as f64 / decode);
    report["prompt_gen_seconds"] = json!(pp + decode);
    report["after_decode"] = model.stats();
    report["io_after_decode"] = storage_io()?;
    std::fs::write(args.report.with_extension("logits.f32"), outputs)?;
    report["completed"] = json!(true);
    save(&args, &report)?;
    println!("complete: pp={pp:.6} s tg={total:.6} s");
    Ok(())
}
