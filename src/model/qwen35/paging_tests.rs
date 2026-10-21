//! Teacher-forced paging runs with full logits and a cross-thread I/O timeline.
use super::Qwen35;
use crate::{
    model::{
        LanguageModel, ModelOptions,
        experts::{Options, Storage},
    },
    validation::BoardLock,
};
use anyhow::{Context, Result};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

#[test]
#[ignore = "requires MiniCPM checkpoint, prepared expert store, token windows and RK3588"]
fn minicpm_prefetch_timeline() -> Result<()> {
    let _lock = BoardLock::acquire()?;
    let windows: Vec<Vec<u32>> =
        serde_json::from_slice(&std::fs::read(std::env::var("MINICPM_TEST_WINDOWS")?)?)?;
    let report_path = PathBuf::from(std::env::var("MINICPM_TEST_REPORT")?);
    let mut options = ModelOptions {
        path: std::env::var("MINICPM_TEST_MODEL")?.into(),
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
        speculation: Default::default(),
        experts: Options {
            storage: Storage::Stream,
            cache: Some(std::env::var("MINICPM_TEST_STORE")?.into()),
            prefetch: std::env::var("MINICPM_TEST_PREFETCH").as_deref() != Ok("0"),
            ..Default::default()
        },
    };
    crate::npu::configure_cpu(&mut options)?;
    let mut model = Qwen35::load(options)?;
    model.reserve(512)?;
    model.prepare(128)?;
    let mut report = json!({"model":model.options.path,"store":model.options.experts.cache,
        "prefetch":model.options.experts.prefetch,"steps":[],"completed":false});
    let mut outputs = Vec::new();
    for step in 0..=8 {
        let tokens = if step == 0 {
            &windows[0][..128]
        } else {
            &windows[0][127 + step..128 + step]
        };
        let position = model.position();
        let paging = model
            .paging
            .as_ref()
            .context("expected streaming model")?
            .clone();
        paging.forward_marker(position, tokens.len(), true);
        let start = Instant::now();
        let logits = model.forward(tokens)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
        paging.forward_marker(position, tokens.len(), false);
        let top = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        outputs.extend(logits.iter().flat_map(|value| value.to_le_bytes()));
        report["steps"]
            .as_array_mut()
            .unwrap()
            .push(json!({"position":position,"rows":tokens.len(),
            "elapsed_ms":elapsed_ms,"top1":top,"paging":paging.stats()}));
        println!(
            "paging position={position} rows={} elapsed_ms={elapsed_ms:.3}",
            tokens.len()
        );
    }
    std::fs::write(report_path.with_extension("logits.f32"), outputs)?;
    model.paging.as_ref().unwrap().save_timeline()?;
    report["completed"] = json!(true);
    report["stats"] = model.stats();
    let status = std::fs::read_to_string("/proc/self/status")?;
    report["process_memory"] = json!(
        status
            .lines()
            .filter(
                |line| ["VmHWM:", "VmRSS:", "RssAnon:", "RssFile:", "RssShmem:"]
                    .iter()
                    .any(|key| line.starts_with(key))
            )
            .collect::<Vec<_>>()
    );
    std::fs::write(report_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
