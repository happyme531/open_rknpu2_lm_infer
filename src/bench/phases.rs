use super::Case;
use crate::model::LanguageModel;
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::time::Instant;

pub(super) struct Measurement {
    pub prefill: f64,
    pub decode: f64,
    pub stages: Value,
}

fn delta(begin: &Value, end: &Value) -> Value {
    let mut result = json!({});
    for key in [
        "linear_ms",
        "attention_ms",
        "gdn_ms",
        "cpu_ms",
        "head_ms",
        "decoder_ms",
        "gdn_split_ms",
        "gdn_layer_prepare_ms",
        "gdn_buffers_ms",
        "gdn_gates_ms",
        "gdn_output_norm_ms",
        "gdn_output_copy_ms",
        "gdn_npu_ms",
        "gdn_npu_hardware_ms",
        "gdn_npu_prepare_ms",
        "gdn_npu_coefficients_ms",
        "gdn_npu_solve_ms",
        "gdn_npu_collect_ms",
    ] {
        if let (Some(a), Some(b)) = (begin[key].as_f64(), end[key].as_f64()) {
            result[key] = json!(b - a);
        }
    }
    result
}

pub(super) fn measure_phases(
    model: &mut dyn LanguageModel,
    tokens: &[u32],
    case: Case,
) -> Result<Measurement> {
    let Case {
        prompt,
        generate,
        depth,
    } = case;
    model.reset()?;
    if depth > 0 {
        model.prepare(depth)?;
        model.prefill(&tokens[..depth])?;
    }
    model.prepare(prompt.max(1))?;
    let before = model.stats();
    let mut pp = 0.0;
    if prompt > 0 {
        let _span = crate::model::trace::Span::new(c"llm.prefill", -1, prompt, depth);
        let start = Instant::now();
        model.prefill(&tokens[depth..depth + prompt])?;
        pp = start.elapsed().as_secs_f64() * 1000.0;
    }
    let after_prefill = model.stats();
    let prefill_stages = delta(&before, &after_prefill);
    ensure!(
        prefill_stages["head_ms"]
            .as_f64()
            .is_none_or(|value| value == 0.0),
        "prefill unexpectedly evaluated lm_head"
    );
    let mut tg = 0.0;
    if generate > 0 {
        let _span = crate::model::trace::Span::new(c"llm.decode", -1, generate, depth + prompt);
        let start = Instant::now();
        for &token in &tokens[depth + prompt..depth + prompt + generate] {
            model.forward(&[token])?;
        }
        tg = start.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(Measurement {
        prefill: pp,
        decode: tg,
        stages: json!({
        "prefill":prefill_stages, "decode":delta(&after_prefill, &model.stats())}),
    })
}

/// Check the real backend before timing: omitting a prompt head must not change
/// the next consumed token's logits or the cache position. Uses tuned buckets.
pub(super) fn verify_head_free(
    model: &mut dyn LanguageModel,
    tokens: &[u32],
    chunk: usize,
) -> Result<Value> {
    let rows = chunk.min(tokens.len().saturating_sub(1));
    if rows == 0 {
        return Ok(json!({"checked":false,"reason":"one-token benchmark capacity"}));
    }
    model.reserve(rows + 1)?;
    model.prepare(rows)?;
    model.reset()?;
    model.forward(&tokens[..rows])?;
    let expected = model.forward(&tokens[rows..rows + 1])?.to_vec();
    model.reset()?;
    model.prefill(&tokens[..rows])?;
    ensure!(
        model.position() == rows,
        "head-free prefill position mismatch"
    );
    let head_ms = model.stats()["head_ms"].as_f64();
    ensure!(
        head_ms.is_none_or(|value| value == 0.0),
        "head-free prefill ran lm_head"
    );
    let actual = model.forward(&tokens[rows..rows + 1])?;
    ensure!(
        expected == actual,
        "head-free prefill changed continuation logits"
    );
    model.reset()?;
    Ok(
        json!({"checked":true,"prompt_tokens":rows,"continuation_logits_exact":true,
              "prefill_head_ms":head_ms}),
    )
}
