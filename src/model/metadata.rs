use anyhow::ensure;
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::Path;

fn text_config(config: &Value) -> &Value {
    config.get("text_config").unwrap_or(config)
}

pub(crate) fn context_limit(config: &Value) -> Result<usize> {
    let tokens = text_config(config)["max_position_embeddings"]
        .as_u64()
        .filter(|v| *v > 0)
        .context("model config requires positive max_position_embeddings")?;
    Ok(tokens.min(32768) as usize)
}

pub(crate) fn inspect(path: &Path, requested_context: usize) -> Result<Value> {
    let config: Value = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    let tokenizer_path = path.join("tokenizer_config.json");
    let tokenizer: Value = if tokenizer_path.exists() {
        serde_json::from_slice(&std::fs::read(tokenizer_path)?)?
    } else {
        Value::Null
    };
    let text = text_config(&config);
    Ok(
        json!({"path":path,"model_type":config["model_type"],"architectures":config["architectures"],
        "layers":text["num_hidden_layers"],"hidden_size":text["hidden_size"],"attention_heads":text["num_attention_heads"],
        "kv_heads":text["num_key_value_heads"],"vocab_size":text["vocab_size"],"max_position_embeddings":text["max_position_embeddings"],
        "quantization_config":config.get("quantization_config").or_else(|| text.get("quantization_config")),
        "vision_config":config.get("vision_config"),
        "chat_template":path.join("chat_template.jinja").exists() || tokenizer.get("chat_template").is_some(),"requested_context":requested_context}),
    )
}

/// Text configs may list end-of-text while the native chat tokenizer uses a
/// distinct end-of-turn token. Both terminate generation; neither is emitted.
#[allow(
    dead_code,
    reason = "Shared EOS metadata is available to every model family."
)]
pub(crate) fn eos_tokens(path: &Path, configured: &[u32], vocab: usize) -> Result<Vec<u32>> {
    let mut result = configured.to_vec();
    let generation = path.join("generation_config.json");
    if generation.exists() {
        let raw: Value = serde_json::from_slice(&std::fs::read(generation)?)?;
        if let Some(value) = raw.get("eos_token_id").filter(|v| !v.is_null()) {
            if value.is_array() {
                result.extend(serde_json::from_value::<Vec<u32>>(value.clone())?);
            } else {
                result.push(serde_json::from_value(value.clone())?);
            }
        }
    }
    let tokenizer = path.join("tokenizer_config.json");
    if tokenizer.exists() {
        let raw: Value = serde_json::from_slice(&std::fs::read(tokenizer)?)?;
        if let Some(eos) = raw["eos_token"]
            .as_str()
            .or_else(|| raw["eos_token"]["content"].as_str())
            && let Some(added) = raw["added_tokens_decoder"].as_object()
        {
            for (id, token) in added {
                if token["content"].as_str() == Some(eos) {
                    result.push(id.parse()?);
                }
            }
        }
    }
    result.sort_unstable();
    result.dedup();
    ensure!(
        result.iter().all(|v| (*v as usize) < vocab),
        "EOS outside model vocabulary"
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn context_zero_uses_text_limit_and_runtime_cap() {
        assert_eq!(
            context_limit(&json!({"max_position_embeddings":8192})).unwrap(),
            8192
        );
        assert_eq!(
            context_limit(&json!({"text_config":{"max_position_embeddings":262144}})).unwrap(),
            32768
        );
        assert!(context_limit(&json!({"text_config":{"max_position_embeddings":0}})).is_err());
    }

    #[test]
    fn configured_end_of_text_and_tokenizer_end_of_turn_both_stop() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokenizer_config.json"),r#"{"eos_token":"<|im_end|>","added_tokens_decoder":{"248044":{"content":"<|endoftext|>"},"248046":{"content":"<|im_end|>"}}}"#).unwrap();
        assert_eq!(
            eos_tokens(dir.path(), &[248044], 248320).unwrap(),
            [248044, 248046]
        );
    }
}
