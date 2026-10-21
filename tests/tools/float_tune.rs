//! Explicit floating MM profiles, independent of a model checkpoint.
use anyhow::{Result, ensure};
use clap::Parser;
use open_rknpu2_lm_infer::{
    model::ModelOptions,
    npu::{self, Backend, FloatBackend},
    validation::BoardLock,
};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Logical M values; vision projections use these rows, merger uses M/4.
    #[arg(long, value_delimiter = ',', default_value = "256,1024")]
    rows: Vec<usize>,
    /// Tune the seven Qwen3.5 C7 vision projection shapes.
    #[arg(long)]
    vision: bool,
    #[arg(long, requires = "n")]
    k: Option<usize>,
    #[arg(long, requires = "k")]
    n: Option<usize>,
    #[arg(long)]
    bf16: bool,
    /// FP16 activation storage, including FP16 output after bias.
    #[arg(long, conflicts_with = "bf16")]
    fp16_storage: bool,
    #[arg(long, default_value = ".cache")]
    tune_cache: PathBuf,
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long, default_value = "7")]
    npu_mask: u32,
    /// Read profiles and report baseline on misses, without searching.
    #[arg(long)]
    load_only: bool,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.vision != args.k.is_some(),
        "choose --vision or --k K --n N"
    );
    let _lock = BoardLock::acquire()?;
    let mut options = ModelOptions {
        path: PathBuf::new(),
        context: 1024,
        chunk: 128,
        threads: 4,
        npu_mask: args.npu_mask,
        cpu_mask: 0xf0,
        tune: true,
        tune_search: !args.load_only,
        tune_cache: args.tune_cache,
        attention_tile: 512,
        vision_dtype: Default::default(),
        experts: Default::default(),
        speculation: Default::default(),
    };
    npu::configure_cpu(&mut options)?;
    let device = Backend::new(&options, 12)?;
    let backend = if args.fp16_storage {
        FloatBackend::fp16_storage(device.clone())
    } else if args.bf16 {
        FloatBackend::bf16(device.clone())
    } else {
        FloatBackend::new(device.clone())
    };
    let mut profiles = Vec::new();
    for rows in args.rows {
        ensure!(
            rows > 0 && (!args.vision || rows.is_multiple_of(4)),
            "vision rows must be a positive multiple of four"
        );
        let shapes = if args.vision {
            vec![
                (rows, 1536, 1024),
                (rows, 1024, 3072),
                (rows, 1024, 1024),
                (rows, 1024, 4096),
                (rows, 4096, 1024),
                (rows / 4, 4096, 4096),
                (rows / 4, 4096, 2560),
            ]
        } else {
            vec![(rows, args.k.unwrap(), args.n.unwrap())]
        };
        for (m, k, n) in shapes {
            backend.prepare(k, n, m)?;
            profiles.push(backend.profile(k, n)?);
        }
    }
    let report = serde_json::json!({"profiles": profiles, "stats": device.stats()});
    if let Some(path) = args.report {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
