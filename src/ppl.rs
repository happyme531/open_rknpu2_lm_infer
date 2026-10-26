use crate::{
    model::{self, ModelOptions},
    validation::BoardLock,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

/// llama.cpp-compatible FP32 shifted exp and FP64 sum/log reduction.
pub fn negative_log_likelihood(logits: &[f32], target: u32) -> Result<f64> {
    ensure!(
        (target as usize) < logits.len() && logits.iter().all(|v| v.is_finite()),
        "invalid logits or target for NLL"
    );
    let largest = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&v| (v - largest).exp() as f64).sum();
    Ok(sum.ln() - (logits[target as usize] - largest) as f64)
}

fn save(path: &Path, value: &Value) -> Result<()> {
    let temporary = path.with_extension("tmp");
    std::fs::write(
        &temporary,
        format!("{}\n", serde_json::to_string_pretty(value)?),
    )?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

pub fn run(
    options: ModelOptions,
    tokens_file: PathBuf,
    score_start: usize,
    report_path: PathBuf,
) -> Result<()> {
    let windows: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&tokens_file)?)?;
    ensure!(!windows.is_empty(), "no PPL windows");
    let length = windows[0].len();
    ensure!(
        score_start > 0
            && score_start < length
            && length <= options.context
            && windows.iter().all(|w| w.len() == length),
        "invalid PPL window/score range"
    );
    let _lock = BoardLock::acquire()?;
    let mut model = model::load(options.clone())?;
    ensure!(
        windows
            .iter()
            .flatten()
            .all(|&t| (t as usize) < model.info().vocab_size),
        "PPL token outside model vocabulary"
    );
    let identity = json!({"model":options.path,"inputs":tokens_file,"windows":windows.len(),
        "window_size":length,"scored_target_positions":[score_start,length - 1],
        "chunk":options.chunk,"attention_tile":options.attention_tile,
        "context":options.context,"cpu_threads":options.threads,
        "expert_min_weight":options.experts.min_weight,
        "cpu_core_mask":options.cpu_mask,"npu_core_mask":options.npu_mask,
        "kv_reset_per_window":true,"chat_template":false,"added_bos_eos":false,
        "arithmetic":"INT32 W4/W8 Linear, native FP16 GQA, FP32 exp and FP64 NLL reduction"});
    let mut report = if report_path.exists() {
        let previous: Value = serde_json::from_slice(&std::fs::read(&report_path)?)?;
        ensure!(
            previous["protocol"] == identity,
            "existing PPL report has a different protocol"
        );
        previous
    } else {
        json!({"protocol":identity,"completed":false,"chunks":[],"scored_tokens":0,"nll_sum":0.0})
    };
    if report["completed"] == true {
        println!("{report}");
        return Ok(());
    }
    let completed = report["chunks"].as_array().unwrap().len();
    ensure!(completed <= windows.len(), "invalid PPL report progress");
    let mut nll_sum = report["nll_sum"].as_f64().unwrap();
    let mut count = report["scored_tokens"].as_u64().unwrap();
    model.reserve(length)?;
    model.prepare(length - 1)?;
    let start = Instant::now();
    let previous_seconds = report["seconds"].as_f64().unwrap_or(0.0);
    for (i, window) in windows.iter().enumerate().skip(completed) {
        let tick = Instant::now();
        model.reset()?;
        if score_start > 1 {
            model.prefill(&window[..score_start - 1])?;
        }
        let values =
            model.token_nll(&window[score_start - 1..length - 1], &window[score_start..])?;
        let sum: f64 = values.iter().sum();
        nll_sum += sum;
        count += values.len() as u64;
        report["chunks"]
            .as_array_mut()
            .unwrap()
            .push(json!({"chunk":i+1,"nll":sum/values.len() as f64,
            "token_nll":values,"seconds":tick.elapsed().as_secs_f64()}));
        report["scored_tokens"] = json!(count);
        report["nll_sum"] = json!(nll_sum);
        report["mean_nll"] = json!(nll_sum / count as f64);
        report["ppl"] = json!((nll_sum / count as f64).exp());
        report["seconds"] = json!(previous_seconds + start.elapsed().as_secs_f64());
        report["completed"] = json!(i + 1 == windows.len());
        if i + 1 == windows.len() {
            report["stats"] = model.stats();
        }
        save(&report_path, &report)?;
        eprintln!(
            "PPL {}/{}: {:.6}, {} targets, {:.2}s/window",
            i + 1,
            windows.len(),
            report["ppl"].as_f64().unwrap(),
            count,
            tick.elapsed().as_secs_f64()
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"ppl":report["ppl"],"scored_tokens":count,
        "windows":windows.len(),"completed":report["completed"],"seconds":report["seconds"]}))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_nll_matches_shifted_probabilities() {
        let logits = [1000.0, 1001.0, 999.0];
        let expected = (1.0f64 + (-1.0f32).exp() as f64 + (-2.0f32).exp() as f64).ln();
        assert!((negative_log_likelihood(&logits, 1).unwrap() - expected).abs() < 1e-12);
        assert!(
            (negative_log_likelihood(&[2.0, 2.0, 2.0], 0).unwrap() - 3.0f64.ln()).abs() < 1e-12
        );
        assert!(negative_log_likelihood(&[f32::NAN], 0).is_err());
        assert!(negative_log_likelihood(&logits, 3).is_err());
    }
}
