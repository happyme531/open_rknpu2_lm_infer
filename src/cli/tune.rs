use crate::model::ModelOptions;
use crate::{
    model::VisionDtype,
    npu::{self, Backend, FloatBackend},
    vision::ImageProcessor,
};
use anyhow::Context;
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Instant};

pub(super) fn run(
    mut options: ModelOptions,
    vision: bool,
    images: Vec<PathBuf>,
    batches: Vec<usize>,
    report: Option<PathBuf>,
) -> Result<()> {
    ensure!(options.tune, "tune cannot be combined with --no-tune");
    options.tune_search = true;
    let start = Instant::now();
    let mut result = if vision {
        vision_profiles(options, images)?
    } else {
        text_profiles(options, batches)?
    };
    result["elapsed_seconds"] = json!(start.elapsed().as_secs_f64());
    eprintln!(
        "Tuning complete: {:.1}s; cache hits={}, searched={}",
        start.elapsed().as_secs_f64(),
        result["stats"]["tune_cache_hits"],
        result["stats"]["tune_calls"]
    );
    if let Some(path) = report {
        std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
    }
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn text_profiles(options: ModelOptions, mut batches: Vec<usize>) -> Result<Value> {
    if batches.is_empty() {
        batches = vec![1, options.chunk];
    }
    ensure!(
        batches.iter().all(|&m| m > 0 && m <= options.chunk),
        "tuning batches must be within 1..ubatch-size"
    );
    batches.sort_unstable();
    batches.dedup();
    eprintln!(
        "Tuning: loading model; cache={} (existing entries are reused)",
        options.tune_cache.display()
    );
    let mut model = crate::model::load(options)?;
    for (index, &rows) in batches.iter().enumerate() {
        eprintln!("Tuning batch {}/{}: M={rows}", index + 1, batches.len());
        model.prepare(rows)?;
    }
    Ok(json!({"mode":"text", "batches":batches, "stats":model.stats()}))
}

fn vision_profiles(mut options: ModelOptions, images: Vec<PathBuf>) -> Result<Value> {
    crate::model::ensure_enabled(&options.path)?;
    let processor =
        ImageProcessor::load(&options.path)?.context("model has no supported vision encoder")?;
    let mut inputs = Vec::new();
    let mut rows = Vec::new();
    for path in images {
        let image = processor
            .prepare(&std::fs::read(&path).with_context(|| format!("read {}", path.display()))?)?;
        let m = image.grid[1] * image.grid[2];
        eprintln!(
            "Vision input {}: {}x{}, {} patches, {} image tokens",
            path.display(),
            image.grid[2] * 16,
            image.grid[1] * 16,
            m,
            image.tokens()
        );
        inputs.push(
            json!({"path":path, "width":image.grid[2]*16, "height":image.grid[1]*16,
            "patches":m, "image_tokens":image.tokens()}),
        );
        rows.push(m);
    }
    if rows.is_empty() {
        eprintln!(
            "Vision input presets: 256x256, 384x256, 512x512; use --image PATH for your inputs"
        );
        rows = vec![256, 384, 1024];
    }
    rows.sort_unstable();
    rows.dedup();
    let shapes = crate::model::vision_tuning_shapes(&options.path, &rows)?;
    npu::configure_cpu(&mut options)?;
    eprintln!(
        "Tuning ViT {}: {} shapes; cache={}",
        options.vision_dtype.as_str(),
        shapes.len(),
        options.tune_cache.join("floating").display()
    );
    let device = Backend::new(&options, 11)?;
    let backend = match options.vision_dtype {
        VisionDtype::Fp16 => FloatBackend::fp16_storage(device.clone()),
        VisionDtype::Fp32 => FloatBackend::new(device.clone()),
    };
    let mut profiles = Vec::new();
    for (index, shape) in shapes.iter().enumerate() {
        eprintln!(
            "Vision {}/{}: {} M={} K={} N={}",
            index + 1,
            shapes.len(),
            shape.name,
            shape.rows,
            shape.k,
            shape.n
        );
        backend.prepare(shape.k, shape.n, shape.rows)?;
        let mut profile = backend.profile(shape.k, shape.n)?;
        profile["projection"] = json!(shape.name);
        profiles.push(profile);
    }
    Ok(
        json!({"mode":"vision", "vision_dtype":options.vision_dtype.as_str(),
        "rows":rows, "images":inputs, "profiles":profiles, "stats":device.stats()}),
    )
}
