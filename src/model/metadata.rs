use crate::sampling::SamplingParams;
use anyhow::ensure;
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;

fn text_config(config: &Value) -> &Value {
    config.get("text_config").unwrap_or(config)
}

fn optional_config(path: &Path) -> Result<Option<Value>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
    };
    let config: Value =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    ensure!(
        config.is_object(),
        "{} must contain a JSON object",
        path.display()
    );
    Ok(Some(config))
}

pub(crate) fn sampling_defaults(path: &Path) -> Result<SamplingParams> {
    #[derive(Deserialize)]
    struct GenerationSampling {
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i64>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
    }

    let defaults = SamplingParams::default();
    let mut config_path = path.join("generation_config.json");
    let config = if let Some(config) = optional_config(&config_path)? {
        config
    } else {
        config_path = path.join("config.json");
        let Some(Value::Object(mut config)) = optional_config(&config_path)? else {
            return Ok(defaults);
        };
        // Older checkpoints may store generation parameters in the text config.
        if let Some(Value::Object(text)) = config.remove("text_config") {
            for (key, value) in text {
                let field = config.entry(key).or_insert(Value::Null);
                if field.is_null() {
                    *field = value;
                }
            }
        }
        Value::Object(config)
    };
    let source = || format!("sampling defaults in {}", config_path.display());
    let config: GenerationSampling = serde_json::from_value(config).with_context(source)?;
    let top_k = match config.top_k {
        None | Some(-1) => defaults.top_k,
        Some(k) => usize::try_from(k)
            .context("top_k must be -1 or nonnegative")
            .with_context(source)?,
    };
    let sampling = SamplingParams {
        temperature: config.temperature.unwrap_or(defaults.temperature),
        top_p: config.top_p.unwrap_or(defaults.top_p),
        top_k,
        min_p: config.min_p.unwrap_or(defaults.min_p),
        repeat_penalty: config.repetition_penalty.unwrap_or(defaults.repeat_penalty),
        presence_penalty: config.presence_penalty.unwrap_or(defaults.presence_penalty),
        frequency_penalty: config
            .frequency_penalty
            .unwrap_or(defaults.frequency_penalty),
        ..defaults
    };
    sampling.validate().with_context(source)?;
    Ok(sampling)
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
    if let Some(raw) = optional_config(&path.join("generation_config.json"))?
        && let Some(value) = raw.get("eos_token_id").filter(|v| !v.is_null())
    {
        if value.is_array() {
            result.extend(serde_json::from_value::<Vec<u32>>(value.clone())?);
        } else {
            result.push(serde_json::from_value(value.clone())?);
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
    fn model_sampling_defaults_use_optional_fields_and_validate_values() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("generation_config.json");
        let defaults = sampling_defaults(dir.path())?;
        assert_eq!(defaults.top_k, 0);
        assert_eq!(defaults.temperature, 1.);
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"temperature":0.7,"text_config":{"temperature":0.8,"top_k":10}}"#,
        )?;
        let legacy = sampling_defaults(dir.path())?;
        assert_eq!(legacy.temperature, 0.7);
        assert_eq!(legacy.top_k, 10);
        std::fs::write(
            &path,
            r#"{"temperature":0.6,"top_k":20,"top_p":0.95,"min_p":null,"repetition_penalty":1.1,"presence_penalty":0.3,"frequency_penalty":0.2,"eos_token_id":[2,3]}"#,
        )?;
        let sampling = sampling_defaults(dir.path())?;
        assert_eq!(sampling.temperature, 0.6);
        assert_eq!(sampling.top_k, 20);
        assert_eq!(sampling.top_p, 0.95);
        assert_eq!(sampling.min_p, 0.);
        assert_eq!(sampling.repeat_penalty, 1.1);
        assert_eq!(sampling.presence_penalty, 0.3);
        assert_eq!(sampling.frequency_penalty, 0.2);
        assert_eq!(eos_tokens(dir.path(), &[1], 4)?, [1, 2, 3]);

        std::fs::write(&path, r#"{"top_k":-1}"#)?;
        let unbounded = sampling_defaults(dir.path())?;
        assert_eq!(unbounded.top_k, 0);
        assert_eq!(unbounded.temperature, 1.);
        for invalid in [r#"{"top_k":-2}"#, r#"{"temperature":-1}"#, "[]"] {
            std::fs::write(&path, invalid)?;
            assert!(sampling_defaults(dir.path()).is_err());
        }
        Ok(())
    }

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
