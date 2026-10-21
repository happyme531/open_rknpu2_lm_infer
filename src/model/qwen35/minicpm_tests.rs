//! Real-checkpoint board checks; no device access in the default test suite.
use super::{Qwen35, decoder::FeedForward};
use crate::{
    model::{
        LanguageModel, ModelOptions,
        experts::store::{self, Geometry, Store, Writer},
    },
    validation::BoardLock,
};
use anyhow::{Context, Result, ensure};
use serde_json::json;
use std::path::PathBuf;

fn logits(model: &mut Qwen35, prefix: &[u32], batch: usize, trace: &str) -> Result<Vec<f32>> {
    if let Ok(directory) = std::env::var("MINICPM_TEST_TRACE") {
        model.trace =
            crate::model::trace::Trace::in_directory(PathBuf::from(directory).join(trace))?
                .at_position(if batch == 1 { prefix.len() - 1 } else { 0 });
    }
    model.reset()?;
    let mut last = Vec::new();
    for tokens in prefix.chunks(batch) {
        last = model.forward(tokens)?.to_vec();
    }
    Ok(last)
}
fn compare(a: &[f32], b: &[f32]) -> serde_json::Value {
    let delta: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum();
    let energy: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
    let top = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .unwrap()
            .0
    };
    json!({"nrmse":(delta/energy).sqrt(),"batch_top1":top(a),"scalar_top1":top(b)})
}
fn read_f32(path: &std::path::Path) -> Result<Vec<f32>> {
    Ok(std::fs::read(path)?
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn trace_differences(directory: &std::path::Path, layers: usize) -> Result<serde_json::Value> {
    let mut result = Vec::new();
    for mode in ["cpu-batch", "tf32-batch", "compensated-batch"] {
        for layer in 0..layers {
            for stage in [
                "input",
                "gdn_qkv",
                "attn",
                "o",
                "post",
                "router_ids",
                "router_weights",
                "ffn",
                "output",
            ] {
                let batch = directory
                    .join(mode)
                    .join(format!("p0-l{layer}-{stage}.f32"));
                let scalar = directory
                    .join("cpu-scalar")
                    .join(format!("p127-l{layer}-{stage}.f32"));
                if !batch.exists() || !scalar.exists() {
                    continue;
                }
                let (b, s) = (read_f32(&batch)?, read_f32(&scalar)?);
                ensure!(b.len() == 128 * s.len(), "trace row shape mismatch");
                let b = &b[b.len() - s.len()..];
                let max_abs = b
                    .iter()
                    .zip(&s)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                result.push(json!({"mode":mode,"layer":layer,"stage":stage,"max_abs":max_abs,"comparison":compare(b,&s)}));
            }
        }
    }
    Ok(json!(result))
}

#[test]
#[ignore = "requires MINICPM_TEST_MODEL, MINICPM_TEST_WINDOWS and RK3588"]
fn minicpm_gdn_batch_reference_and_expert_export() -> Result<()> {
    let _lock = BoardLock::acquire()?;
    let path = PathBuf::from(std::env::var("MINICPM_TEST_MODEL").context("MINICPM_TEST_MODEL")?);
    let windows: Vec<Vec<u32>> =
        serde_json::from_slice(&std::fs::read(std::env::var("MINICPM_TEST_WINDOWS")?)?)?;
    let prefix = &windows[0][..128];
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
        experts: Default::default(),
        speculation: Default::default(),
    };
    crate::npu::configure_cpu(&mut options)?;
    let mut model = Qwen35::load(options)?;
    let stamps = model.loader.source_stamps()?;
    model.reserve(512)?;
    model.prepare(128)?;
    let npu_scratch =
        crate::layers::gated_delta::Scratch::with_npu(model.backends.last().unwrap().clone());
    model.work.recurrent = Default::default();
    let cpu_batch = logits(&mut model, prefix, 128, "cpu-batch")?;
    let cpu_scalar = logits(&mut model, prefix, 1, "cpu-scalar")?;
    model.work.recurrent = npu_scratch;
    let npu_batch = logits(&mut model, prefix, 128, "tf32-batch")?;
    model.work.recurrent = crate::layers::gated_delta::Scratch::with_compensated_npu(
        model.backends.last().unwrap().clone(),
    );
    let compensated_batch = logits(&mut model, prefix, 128, "compensated-batch")?;
    let mut report = json!({"cpu_batch_scalar":compare(&cpu_batch,&cpu_scalar),
        "npu_batch_scalar":compare(&npu_batch,&cpu_scalar),"npu_cpu_batch":compare(&npu_batch,&cpu_batch),
        "compensated_batch_scalar":compare(&compensated_batch,&cpu_scalar),"compensated_cpu_batch":compare(&compensated_batch,&cpu_batch)});
    println!("GDN comparison: {report}");
    if let Ok(directory) = std::env::var("MINICPM_TEST_TRACE") {
        report["trace_differences"] =
            trace_differences(std::path::Path::new(&directory), model.layers.len())?;
    }
    if let Ok(path) = std::env::var("MINICPM_TEST_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    if let Ok(directory) = std::env::var("MINICPM_TEST_STORE") {
        let directory = PathBuf::from(directory);
        let _writer_lock = store::lock(&directory)?;
        let m = model.config.moe.as_ref().context("expected MoE model")?;
        let first = match &model.layers[0].ffn {
            FeedForward::Routed(m) => m,
            _ => anyhow::bail!("expected routed layer"),
        };
        let geometry = Geometry {
            native_version: crate::npu::Moe::blob_version(),
            layers: model.layers.len(),
            experts: m.experts,
            hidden: model.config.hidden_size,
            intermediate: model.config.intermediate_size,
            down_bits: 4,
        };
        ensure!(
            model.loader.source_stamps()? == stamps,
            "checkpoint changed before expert export"
        );
        if Store::open(&directory, &geometry, &stamps, false).is_err() {
            let mut writer = Writer::create(
                &directory,
                geometry.clone(),
                stamps.clone(),
                first.export(0)?.len(),
            )?;
            for (layer, entry) in model.layers.iter().enumerate() {
                let FeedForward::Routed(routed) = &entry.ffn else {
                    anyhow::bail!("expected routed layer")
                };
                for expert in 0..geometry.experts {
                    writer.write(layer, expert, &routed.export(expert)?)?;
                }
                println!("exported expert layer {}/{}", layer + 1, geometry.layers);
            }
            ensure!(
                model.loader.source_stamps()? == stamps,
                "checkpoint changed during expert export"
            );
            writer.finish()?;
        }
        let store = Store::open(&directory, &geometry, &stamps, true)?;
        ensure!(
            store.read(0, 0)?.as_bytes() == first.export(0)?,
            "native expert file byte mismatch"
        );
        report["native_expert_store"] =
            json!({"directory":directory,"geometry":geometry,"record_bytes":store.record_bytes()});
    }
    if let Ok(path) = std::env::var("MINICPM_TEST_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires MINICPM_TEST_MODEL, MINICPM_TEST_STORE, MINICPM_TEST_WINDOWS and RK3588"]
fn minicpm_stream_resident_equivalence() -> Result<()> {
    let _lock = BoardLock::acquire()?;
    let windows: Vec<Vec<u32>> =
        serde_json::from_slice(&std::fs::read(std::env::var("MINICPM_TEST_WINDOWS")?)?)?;
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
        experts: crate::model::experts::Options {
            cache: Some(std::env::var("MINICPM_TEST_STORE")?.into()),
            ..Default::default()
        },
    };
    crate::npu::configure_cpu(&mut options)?;
    let mut reference = Vec::new();
    {
        let mut resident = Qwen35::load(options.clone())?;
        resident.reserve(512)?;
        resident.prepare(128)?;
        reference.push(resident.forward(&windows[0][..128])?.to_vec());
        for token in &windows[0][128..132] {
            reference.push(resident.forward(&[*token])?.to_vec());
        }
    }
    options.experts.storage = crate::model::experts::Storage::Stream;
    let mut streamed = Qwen35::load(options)?;
    streamed.reserve(512)?;
    streamed.prepare(128)?;
    let mut report = Vec::new();
    for (step, expected) in reference.iter().enumerate() {
        let tokens = if step == 0 {
            &windows[0][..128]
        } else {
            &windows[0][127 + step..128 + step]
        };
        let actual = streamed.forward(tokens)?;
        report.push(compare(expected, actual));
        ensure!(
            expected == actual,
            "resident/streamed expert arithmetic differs at step {step}: {}",
            report.last().unwrap()
        );
    }
    let report = json!({"comparisons":report,"stats":streamed.stats()});
    println!("streamed equivalence: {report}");
    if let Ok(path) = std::env::var("MINICPM_TEST_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    Ok(())
}
