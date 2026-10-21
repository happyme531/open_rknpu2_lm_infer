use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ops::{gated_delta::GdnDimensions, rope::MRope};

pub const MAX_CONTEXT: usize = 32768;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub experts: usize,
    pub top_k: usize,
    pub shared_intermediate: usize,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub layers: Vec<LayerType>,
    pub gdn: GdnDimensions,
    pub conv_kernel: usize,
    pub epsilon: f32,
    pub max_position_embeddings: usize,
    pub rope: MRope,
    pub mtp_layers: usize,
    pub mtp_dedicated_embeddings: bool,
    pub tie_word_embeddings: bool,
    pub eos_tokens: Vec<u32>,
    /// Quantization is resolved explicitly, independently of tensor storage.
    pub quantization: Option<Value>,
    pub text_prefix: &'static str,
    pub moe: Option<MoeConfig>,
}

fn dimension(config: &Value, name: &str) -> Result<usize> {
    let value = config[name]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .filter(|v| *v > 0)
        .with_context(|| format!("missing or invalid Qwen3.5 {name}"))?;
    Ok(value)
}

fn flag(config: &Value, name: &str, default: bool) -> Result<bool> {
    config.get(name).map_or(Ok(default), |v| {
        v.as_bool().with_context(|| format!("invalid {name}"))
    })
}

impl Config {
    pub fn parse(root: &Value) -> Result<Self> {
        let nested = match root["model_type"].as_str() {
            Some("qwen3_5" | "qwen3_5_moe" | "minicpmv4_7") => true,
            Some("qwen3_5_text" | "qwen3_5_moe_text") => false,
            other => anyhow::bail!("expected Qwen3.5/MiniCPM-V 4.7 configuration, got {other:?}"),
        };
        let text = if nested {
            root.get("text_config")
                .filter(|v| v.is_object())
                .context("missing Qwen3.5 text_config")?
        } else {
            root
        };
        ensure!(
            text["model_type"]
                .as_str()
                .is_none_or(|s| matches!(s, "qwen3_5_text" | "qwen3_5_moe_text")),
            "unsupported Qwen3.5 text model type"
        );
        ensure!(
            text["hidden_act"] == "silu"
                && !flag(text, "attention_bias", false)?
                && !flag(text, "mlp_bias", false)?
                && flag(text, "attn_output_gate", true)?
                && !flag(text, "use_sliding_window", false)?
                && text["mlp_only_layers"].as_array().is_none_or(Vec::is_empty),
            "Qwen3.5 requires SiLU, attention output gate and no bias/sliding window"
        );
        let hidden_size = dimension(text, "hidden_size")?;
        let moe = if !text["num_experts"].is_null() {
            let m = MoeConfig {
                experts: dimension(text, "num_experts")?,
                top_k: dimension(text, "num_experts_per_tok")?,
                shared_intermediate: dimension(text, "shared_expert_intermediate_size")?,
            };
            ensure!(
                m.top_k <= m.experts && m.top_k <= 8,
                "invalid routed expert count"
            );
            Some(m)
        } else {
            None
        };
        let intermediate_size = dimension(
            text,
            if moe.is_some() {
                "moe_intermediate_size"
            } else {
                "intermediate_size"
            },
        )?;
        let vocab_size = dimension(text, "vocab_size")?;
        ensure!(
            vocab_size <= u32::MAX as usize,
            "vocabulary exceeds token ID range"
        );
        let num_attention_heads = dimension(text, "num_attention_heads")?;
        let num_key_value_heads = dimension(text, "num_key_value_heads")?;
        let head_dim = dimension(text, "head_dim")?;
        ensure!(
            num_attention_heads.is_multiple_of(num_key_value_heads),
            "attention heads must be divisible by KV heads"
        );
        let count = dimension(text, "num_hidden_layers")?;
        let layers: Vec<LayerType> =
            if let Some(value) = text.get("layer_types").filter(|v| !v.is_null()) {
                serde_json::from_value(value.clone()).context("invalid Qwen3.5 layer_types")?
            } else {
                let interval = text
                    .get("full_attention_interval")
                    .map_or(Ok(4), |_| dimension(text, "full_attention_interval"))?;
                (0..count)
                    .map(|i| {
                        if (i + 1) % interval == 0 {
                            LayerType::FullAttention
                        } else {
                            LayerType::LinearAttention
                        }
                    })
                    .collect()
            };
        ensure!(
            layers.len() == count,
            "layer_types length does not match num_hidden_layers"
        );
        let gdn = GdnDimensions::new(
            dimension(text, "linear_num_key_heads")?,
            dimension(text, "linear_num_value_heads")?,
            dimension(text, "linear_key_head_dim")?,
            dimension(text, "linear_value_head_dim")?,
        )?;
        let conv_kernel = dimension(text, "linear_conv_kernel_dim")?;
        let epsilon = text["rms_norm_eps"]
            .as_f64()
            .context("missing rms_norm_eps")? as f32;
        ensure!(epsilon.is_finite() && epsilon > 0.0, "invalid rms_norm_eps");
        let max_position_embeddings = dimension(text, "max_position_embeddings")?;
        let rope = super::rope::from_config(text, head_dim)?;
        let mtp_layers = text.get("mtp_num_hidden_layers").map_or(Ok(0), |v| {
            v.as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .context("invalid mtp_num_hidden_layers")
        })?;
        let outer_quant = root.get("quantization_config").filter(|v| !v.is_null());
        let inner_quant = text.get("quantization_config").filter(|v| !v.is_null());
        if let (Some(a), Some(b)) = (outer_quant, inner_quant) {
            ensure!(a == b, "conflicting outer/text quantization_config");
        }
        let eos = text
            .get("eos_token_id")
            .or_else(|| root.get("eos_token_id"));
        let eos_tokens: Vec<u32> = match eos {
            Some(Value::Array(_)) => serde_json::from_value(eos.unwrap().clone())?,
            Some(Value::Number(_)) => vec![serde_json::from_value(eos.unwrap().clone())?],
            None | Some(Value::Null) => Vec::new(),
            _ => anyhow::bail!("invalid eos_token_id"),
        };
        ensure!(
            eos_tokens.iter().all(|v| (*v as usize) < vocab_size),
            "EOS outside vocabulary"
        );
        Ok(Self {
            hidden_size,
            intermediate_size,
            vocab_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            layers,
            gdn,
            conv_kernel,
            epsilon,
            max_position_embeddings,
            rope,
            mtp_layers,
            mtp_dedicated_embeddings: flag(text, "mtp_use_dedicated_embeddings", false)?,
            tie_word_embeddings: flag(
                text,
                "tie_word_embeddings",
                flag(root, "tie_word_embeddings", false)?,
            )?,
            eos_tokens,
            quantization: outer_quant.or(inner_quant).cloned(),
            moe,
            text_prefix: if nested {
                "model.language_model"
            } else {
                "model"
            },
        })
    }

    pub fn validate_context(&self, context: usize) -> Result<()> {
        ensure!(
            context > 0 && context <= MAX_CONTEXT.min(self.max_position_embeddings),
            "Qwen3.5 context exceeds runtime/model limit ({} tokens)",
            MAX_CONTEXT.min(self.max_position_embeddings)
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minicpmv47_hybrid_moe_configuration() {
        let raw = serde_json::from_str(include_str!(
            "../../../tests/fixtures/minicpmv47/config.json"
        ))
        .unwrap();
        let c = Config::parse(&raw).unwrap();
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.intermediate_size, 512);
        assert_eq!(c.layers.len(), 40);
        assert_eq!(
            c.layers
                .iter()
                .filter(|&&l| l == LayerType::FullAttention)
                .count(),
            10
        );
        let m = c.moe.unwrap();
        assert_eq!((m.experts, m.top_k, m.shared_intermediate), (256, 8, 512));
        assert_eq!(c.text_prefix, "model.language_model");
    }

    fn official() -> Value {
        serde_json::from_str(include_str!(
            "../../../llm_generated_docs/evidence/qwen35_mtp/official_config.json"
        ))
        .unwrap()
    }

    #[test]
    fn official_config_and_all_text_mtp_tensor_names() {
        let c = Config::parse(&official()).unwrap();
        assert_eq!(c.hidden_size, 2560);
        assert_eq!(
            c.layers
                .iter()
                .filter(|v| **v == LayerType::LinearAttention)
                .count(),
            24
        );
        assert_eq!(c.gdn.state_elements() * 24 * 4, 48 * 1024 * 1024);
        assert_eq!(c.rope.rotary_dim(), 64);
        assert_eq!(c.mtp_layers, 1);
        assert!(c.validate_context(32768).is_ok());
        assert!(c.validate_context(32769).is_err());
        let names: Vec<String> = serde_json::from_str(include_str!(
            "../../../llm_generated_docs/evidence/qwen35_mtp/official_tensor_names.json"
        ))
        .unwrap();
        let actual: std::collections::BTreeSet<_> = names
            .into_iter()
            .filter(|s| !s.starts_with("model.visual."))
            .collect();
        let expected = c.text_tensors(false).into_iter().map(|v| v.name).collect();
        assert_eq!(actual, expected);
        assert!(
            c.text_tensors(true)
                .iter()
                .any(|v| v.name == "lm_head.weight")
        );
    }

    #[test]
    fn standalone_and_explicit_layer_order_and_quantization() {
        let mut root = official();
        root["text_config"]["layer_types"]
            .as_array_mut()
            .unwrap()
            .swap(0, 3);
        root["quantization_config"] = serde_json::json!({"quant_method": "compressed-tensors"});
        let c = Config::parse(&root).unwrap();
        assert_eq!(c.layers[0], LayerType::FullAttention);
        assert!(c.quantization.is_some());
        root["text_config"]["quantization_config"] = root["quantization_config"].clone();
        assert!(Config::parse(&root).is_ok());
        root["text_config"]["quantization_config"]["quant_method"] = "other".into();
        assert!(Config::parse(&root).is_err());
        let mut standalone = official()["text_config"].clone();
        standalone.as_object_mut().unwrap().remove("layer_types");
        let c = Config::parse(&standalone).unwrap();
        assert_eq!(c.text_prefix, "model");
        assert_eq!(c.layers[3], LayerType::FullAttention);
        standalone["full_attention_interval"] = 0.into();
        assert!(Config::parse(&standalone).is_err());
    }
}
