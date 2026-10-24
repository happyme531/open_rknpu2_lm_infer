//! Reproduce processor, ViT or teacher-forced multimodal logits without HTTP.
use anyhow::{Context, Result};
use clap::{ArgGroup, Parser};
use open_rknpu2_lm_infer::{
    cli::ModelArgs,
    engine::Prompt,
    model::{self, checkpoint::TensorStore, qwen35::vision::Encoder},
    text::TextCodec,
    validation::BoardLock,
    vision,
};
use std::{io::Write, path::PathBuf, sync::atomic::AtomicBool};

#[derive(Parser)]
#[command(group(ArgGroup::new("mode").required(true).args(["image", "messages", "cases"])))]
struct Args {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long)]
    image: Option<PathBuf>,
    #[arg(long)]
    messages: Option<PathBuf>,
    /// JSON array with name, messages and optional continuation token IDs.
    #[arg(long)]
    cases: Option<PathBuf>,
    #[arg(long, requires = "messages", conflicts_with_all = ["image", "cases"])]
    continuation: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, conflicts_with = "cases")]
    processor_only: bool,
}
fn dump(path: PathBuf, values: &[f32]) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    for x in values {
        f.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    let mut options = args.model.options()?;
    std::fs::create_dir_all(&args.output)?;
    if let Some(image) = args.image {
        let processor =
            vision::ImageProcessor::load(&options.path)?.context("no image processor")?;
        let image = processor.prepare(&std::fs::read(image)?)?;
        dump(args.output.join("patches.f32"), &image.patches)?;
        std::fs::write(
            args.output.join("image.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"grid":image.grid,"identity":image.identity,"visual_tokens":image.tokens()}),
            )?,
        )?;
        if !args.processor_only {
            let _lock = BoardLock::acquire()?;
            open_rknpu2_lm_infer::npu::configure_cpu(&mut options)?;
            let store = TensorStore::open(&options.path)?;
            let mut encoder = Encoder::load(&options, &store, 11)?;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(options.threads)
                .build()?;
            let features = encoder.encode(&image, &pool, &AtomicBool::new(false))?;
            dump(args.output.join("features.f32"), &features)?;
            // A repeated identical image must use the bounded feature cache.
            let repeated = encoder.encode(&image, &pool, &AtomicBool::new(false))?;
            assert_eq!(&*features, &*repeated);
            assert_eq!(encoder.calls, 1);
            assert_eq!(encoder.hits, 1);
            std::fs::write(
                args.output.join("encoder.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"calls":encoder.calls,"hits":encoder.hits,"encode_ms":encoder.elapsed_ms,"cache_bytes":encoder.cache_bytes(),"vision_dtype":options.vision_dtype.as_str()}),
                )?,
            )?;
        }
    } else if let Some(path) = args.cases {
        let cases: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(path)?)?;
        let codec = TextCodec::load(&options.path)?;
        let _lock = BoardLock::acquire()?;
        let mut model = model::load(options)?;
        for case in cases {
            let name = case["name"].as_str().context("case name")?;
            anyhow::ensure!(
                !name.is_empty() && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-'),
                "invalid case name"
            );
            let directory = args.output.join(name);
            std::fs::create_dir_all(&directory)?;
            let messages = serde_json::from_value(case["messages"].clone())?;
            let input = codec.prepare(
                &Prompt::Chat {
                    messages,
                    tools: None,
                    enable_thinking: false,
                },
                model.uses_user_boundaries(),
            )?;
            let continuation: Vec<u32> = case
                .get("continuation")
                .map(|x| serde_json::from_value(x.clone()))
                .transpose()?
                .unwrap_or_default();
            model.reset()?;
            model.reserve(input.tokens.len() + continuation.len() + 1)?;
            let prepared = model.prepare_prepared(&input, 0)?;
            let mut logits = model
                .forward_prepared(prepared, &AtomicBool::new(false))?
                .to_vec();
            for token in continuation {
                logits.extend_from_slice(model.forward(&[token])?);
            }
            dump(directory.join("logits.f32"), &logits)?;
            std::fs::write(
                directory.join("stats.json"),
                serde_json::to_vec_pretty(&model.stats())?,
            )?;
        }
    } else {
        let messages: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(
            args.messages.context("provide --image or --messages")?,
        )?)?;
        let input = TextCodec::load(&options.path)?.prepare(
            &Prompt::Chat {
                messages,
                tools: None,
                enable_thinking: false,
            },
            options.checkpoints.capacity > 0,
        )?;
        std::fs::write(
            args.output.join("input.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"tokens":input.tokens,"positions":input.positions,"next_position":input.next_position,"images":input.images.iter().map(|s|serde_json::json!({"start":s.start,"grid":s.image.grid,"tokens":s.image.tokens()})).collect::<Vec<_>>()}),
            )?,
        )?;
        if args.processor_only {
            return Ok(());
        }
        let _lock = BoardLock::acquire()?;
        let mut model = model::load(options)?;
        model.reserve(input.tokens.len() + 8)?;
        let prepared = model.prepare_prepared(&input, 0)?;
        let mut logits = model
            .forward_prepared(prepared, &AtomicBool::new(false))?
            .to_vec();
        if let Some(path) = args.continuation {
            let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(path)?)?;
            for token in tokens {
                logits.extend_from_slice(model.forward(&[token])?);
            }
        }
        dump(args.output.join("logits.f32"), &logits)?;
        std::fs::write(
            args.output.join("stats.json"),
            serde_json::to_vec_pretty(&model.stats())?,
        )?;
    }
    Ok(())
}
