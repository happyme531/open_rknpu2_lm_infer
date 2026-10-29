use anyhow::{Context, Result, ensure};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub vocab: usize,
    pub layers: usize,
    pub group: usize,
    pub dense_layers: usize,
    pub dense_intermediate: usize,
    pub intermediate: usize,
    pub shared_intermediate: usize,
    pub experts: usize,
    pub top: usize,
    pub groups: usize,
    pub top_groups: usize,
    pub conv: usize,
    pub epsilon: f32,
    pub lower_bound: f32,
    pub routing_scale: f32,
    pub rope_theta: f32,
    pub max_context: usize,
    pub eos: Vec<u32>,
}

impl Config {
    pub fn parse(v: &Value) -> Result<Self> {
        let n = |key: &str| -> Result<usize> {
            v[key]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n > 0)
                .with_context(|| format!("invalid Ling {key}"))
        };
        let f = |key: &str| -> Result<f32> {
            v[key]
                .as_f64()
                .map(|x| x as f32)
                .filter(|x| x.is_finite())
                .with_context(|| format!("invalid Ling {key}"))
        };
        ensure!(
            v["model_type"] == "bailing_hybrid",
            "expected Ling bailing_hybrid checkpoint"
        );
        for (key, expected) in [
            ("num_attention_heads", 16),
            ("num_key_value_heads", 16),
            ("head_dim", 128),
            ("q_lora_rank", 256),
            ("kv_lora_rank", 512),
            ("qk_nope_head_dim", 128),
            ("qk_rope_head_dim", 64),
            ("qk_head_dim", 192),
            ("v_head_dim", 128),
        ] {
            ensure!(n(key)? == expected, "unsupported Ling {key}");
        }
        ensure!(
            v["hidden_act"] == "silu"
                && v["no_kda_lora"] == true
                && v["kda_safe_gate"] == true
                && v["rope_interleave"] == true
                && v["rope_scaling"].is_null()
                && v["gated_attention_proj_granularity_type"] == "head_wise"
                && v["norm_topk_prob"] == true
                && v["moe_router_enable_expert_bias"] == true
                && v["topk_method"] == "noaux_tc"
                && v["scoring_func"] == "sigmoid"
                && v["use_qkv_bias"] != true
                && v["use_bias"] != true
                && v["up_proj_norm"] != true
                && v["value_norm"] != true
                && v["scale_router_input"] != true
                && v["expert_swiglu_limit_list"].is_null()
                && v["share_expert_swiglu_limit_list"].is_null(),
            "unsupported Ling attention/router/FFN configuration"
        );
        let dense_layers = v["first_k_dense_replace"]
            .as_u64()
            .context("missing first_k_dense_replace")? as usize;
        let result = Self {
            hidden: n("hidden_size")?,
            vocab: n("vocab_size")?,
            layers: n("num_hidden_layers")?,
            group: n("layer_group_size")?,
            dense_layers,
            dense_intermediate: n("intermediate_size")?,
            intermediate: n("moe_intermediate_size")?,
            shared_intermediate: n("moe_shared_expert_intermediate_size")?
                * n("num_shared_experts")?,
            experts: n("num_experts")?,
            top: n("num_experts_per_tok")?,
            groups: n("n_group")?,
            top_groups: n("topk_group")?,
            conv: n("short_conv_kernel_size")?,
            epsilon: f("rms_norm_eps")?,
            lower_bound: f("kda_lower_bound")?,
            routing_scale: f("routed_scaling_factor")?,
            rope_theta: f("rope_theta")?,
            max_context: n("max_position_embeddings")?.min(32768),
            eos: match &v["eos_token_id"] {
                Value::Array(a) => a
                    .iter()
                    .map(|x| serde_json::from_value(x.clone()))
                    .collect::<std::result::Result<_, _>>()?,
                Value::Null => Vec::new(),
                x => vec![serde_json::from_value(x.clone())?],
            },
        };
        ensure!(
            result.dense_layers < result.layers
                && result.top <= 8
                && result.top <= result.experts
                && result.experts.is_multiple_of(result.groups)
                && result.experts / result.groups >= 2
                && result.top_groups <= result.groups
                && result.top <= result.top_groups * (result.experts / result.groups)
                && result.epsilon > 0.
                && result.lower_bound < 0.
                && result.routing_scale > 0.
                && result.rope_theta > 0.,
            "invalid Ling layer/router dimensions"
        );
        Ok(result)
    }
    pub fn mla(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.group) || layer >= self.layers / self.group * self.group
    }
}

#[cfg(test)]
pub(super) fn official() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/ling3tiny/config.json"
    ))
    .unwrap()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_hybrid_order_and_router_geometry() -> Result<()> {
        let c = Config::parse(&official())?;
        assert_eq!((c.hidden, c.layers, c.experts, c.top), (1536, 24, 128, 8));
        assert_eq!(
            (
                c.dense_layers,
                c.dense_intermediate,
                c.intermediate,
                c.shared_intermediate
            ),
            (1, 4608, 512, 512)
        );
        assert_eq!(
            (0..c.layers).filter(|&i| c.mla(i)).collect::<Vec<_>>(),
            [3, 7, 11, 15, 19, 23]
        );
        assert_eq!(
            (c.groups, c.top_groups, c.lower_bound, c.routing_scale),
            (8, 4, -5., 2.5)
        );
        let mut raw = official();
        raw["kv_lora_rank"] = 256.into();
        assert!(Config::parse(&raw).is_err());
        Ok(())
    }
}
