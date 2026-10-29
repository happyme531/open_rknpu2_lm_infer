//! Logical checkpoint tensor mapping, independent of storage bit width.
use super::config::{Config, LayerType};
use serde::Serialize;

impl Config {
    /// Logical tensors before compressed-tensors storage expansion. A supplied
    /// lm_head takes precedence over embedding sharing, including quantized heads.
    pub fn text_tensors(&self, has_lm_head: bool) -> Vec<TensorSpec> {
        let mut specs = Vec::new();
        let mut add = |name: String, shape: Vec<usize>, linear: bool| {
            specs.push(TensorSpec {
                name,
                shape,
                linear,
            });
        };
        let h = self.hidden_size;
        add(
            format!("{}.embed_tokens.weight", self.text_prefix),
            vec![self.vocab_size, h],
            false,
        );
        add(format!("{}.norm.weight", self.text_prefix), vec![h], false);
        if has_lm_head || !self.tie_word_embeddings {
            add("lm_head.weight".into(), vec![self.vocab_size, h], true);
        }
        for (i, kind) in self.layers.iter().enumerate() {
            self.layer_tensors(&format!("{}.layers.{i}", self.text_prefix), *kind, &mut add);
        }
        if self.mtp_layers > 0 {
            for name in ["norm", "pre_fc_norm_hidden", "pre_fc_norm_embedding"] {
                add(format!("mtp.{name}.weight"), vec![h], false);
            }
            add("mtp.fc.weight".into(), vec![h, 2 * h], true);
            if self.mtp_dedicated_embeddings {
                add(
                    "mtp.embed_tokens.weight".into(),
                    vec![self.vocab_size, h],
                    false,
                );
            }
            for i in 0..self.mtp_layers {
                self.layer_tensors(
                    &format!("mtp.layers.{i}"),
                    LayerType::FullAttention,
                    &mut add,
                );
            }
        }
        specs
    }

    fn layer_tensors(
        &self,
        prefix: &str,
        kind: LayerType,
        add: &mut impl FnMut(String, Vec<usize>, bool),
    ) {
        let h = self.hidden_size;
        for name in ["input_layernorm", "post_attention_layernorm"] {
            add(format!("{prefix}.{name}.weight"), vec![h], false);
        }
        if let Some(m) = &self.moe {
            add(
                format!("{prefix}.mlp.gate.weight"),
                vec![m.experts, h],
                false,
            );
            add(
                format!("{prefix}.mlp.shared_expert_gate.weight"),
                vec![1, h],
                false,
            );
            add(
                format!("{prefix}.mlp.experts.gate_up_proj"),
                vec![m.experts, 2 * self.intermediate_size, h],
                true,
            );
            add(
                format!("{prefix}.mlp.experts.down_proj"),
                vec![m.experts, h, self.intermediate_size],
                true,
            );
            for (name, n, k) in [
                ("gate_proj", m.shared_intermediate, h),
                ("up_proj", m.shared_intermediate, h),
                ("down_proj", h, m.shared_intermediate),
            ] {
                add(
                    format!("{prefix}.mlp.shared_expert.{name}.weight"),
                    vec![n, k],
                    true,
                );
            }
        } else {
            for (name, n, k) in [
                ("gate_proj", self.intermediate_size, h),
                ("up_proj", self.intermediate_size, h),
                ("down_proj", h, self.intermediate_size),
            ] {
                add(format!("{prefix}.mlp.{name}.weight"), vec![n, k], true);
            }
        }
        match kind {
            LayerType::FullAttention => {
                for (name, n, k) in [
                    ("q_proj", self.num_attention_heads * self.head_dim * 2, h),
                    ("k_proj", self.num_key_value_heads * self.head_dim, h),
                    ("v_proj", self.num_key_value_heads * self.head_dim, h),
                    ("o_proj", h, self.num_attention_heads * self.head_dim),
                ] {
                    add(
                        format!("{prefix}.self_attn.{name}.weight"),
                        vec![n, k],
                        true,
                    );
                }
                for name in ["q_norm", "k_norm"] {
                    add(
                        format!("{prefix}.self_attn.{name}.weight"),
                        vec![self.head_dim],
                        false,
                    );
                }
            }
            LayerType::LinearAttention => {
                let p = format!("{prefix}.linear_attn");
                for (name, n, k) in [
                    ("in_proj_qkv", self.gdn.conv_width(), h),
                    ("in_proj_z", self.gdn.value_width(), h),
                    ("in_proj_a", self.gdn.value_heads, h),
                    ("in_proj_b", self.gdn.value_heads, h),
                    ("out_proj", h, self.gdn.value_width()),
                ] {
                    add(format!("{p}.{name}.weight"), vec![n, k], true);
                }
                add(
                    format!("{p}.conv1d.weight"),
                    vec![self.gdn.conv_width(), 1, self.conv_kernel],
                    false,
                );
                add(format!("{p}.norm.weight"), vec![self.gdn.value_dim], false);
                for name in ["A_log", "dt_bias"] {
                    add(format!("{p}.{name}"), vec![self.gdn.value_heads], false);
                }
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    /// Linear modules may be quantized or explicitly retained as floating point.
    pub linear: bool,
}
