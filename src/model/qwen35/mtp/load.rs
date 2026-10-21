use super::*;
use crate::{
    model::{ModelOptions, checkpoint::TensorStore, projection::Projection},
    npu::{Backend, FloatBackend},
};
use anyhow::ensure;

fn vector(store: &TensorStore, name: &str, n: usize) -> Result<Vec<f32>> {
    let t = store.get(&format!("{name}.weight"))?;
    t.require_shape(&[n], name)?;
    t.floats()
}
impl Predictor {
    pub fn load(
        options: &ModelOptions,
        c: &Config,
        store: &TensorStore,
        domain: u32,
        max_draft: usize,
    ) -> Result<Self> {
        ensure!(
            c.mtp_layers == 1 && !c.mtp_dedicated_embeddings,
            "MTP currently requires one layer and shared embedding/head"
        );
        let device = Backend::new(options, domain)?;
        let cache = device.attention(
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            (c.head_dim as f32).sqrt().recip(),
        )?;
        cache.reserve(options.context)?;
        // Preserve all supplied BF16 values; FP16 GEMM uses FP32 host buffers.
        let backend = FloatBackend::for_decoder(device.clone());
        let (h, f, q, kv) = (
            c.hidden_size,
            c.intermediate_size,
            c.num_attention_heads * c.head_dim,
            c.num_key_value_heads * c.head_dim,
        );
        let p = "mtp.layers.0";
        let layer = Layer {
            input_norm: vector(store, &format!("{p}.input_layernorm"), h)?,
            post_norm: vector(store, &format!("{p}.post_attention_layernorm"), h)?,
            inputs: Projection::floating(
                &backend,
                store,
                vec![
                    (format!("{p}.self_attn.q_proj"), 2 * q),
                    (format!("{p}.self_attn.k_proj"), kv),
                    (format!("{p}.self_attn.v_proj"), kv),
                ],
                h,
            )?,
            output: Linear::floating(&backend, store, &format!("{p}.self_attn.o_proj"), h, q)?,
            ffn: crate::model::qwen35::decoder::FeedForward::Dense {
                gate_up: Projection::floating(
                    &backend,
                    store,
                    vec![
                        (format!("{p}.mlp.gate_proj"), f),
                        (format!("{p}.mlp.up_proj"), f),
                    ],
                    h,
                )?,
                down: Linear::floating(&backend, store, &format!("{p}.mlp.down_proj"), h, f)?,
            },
            mixer: Mixer::Attention {
                cache,
                q_norm: vector(store, &format!("{p}.self_attn.q_norm"), c.head_dim)?,
                k_norm: vector(store, &format!("{p}.self_attn.k_norm"), c.head_dim)?,
            },
        };
        Ok(Self {
            backend: device,
            layer,
            fc: Linear::floating(&backend, store, "mtp.fc", h, 2 * h)?,
            hidden_norm: vector(store, "mtp.pre_fc_norm_hidden", h)?,
            embedding_norm: vector(store, "mtp.pre_fc_norm_embedding", h)?,
            final_norm: vector(store, "mtp.norm", h)?,
            adapter: adapter::Adapter::load(&options.path, h)?,
            work: Buffers::new(c, options.chunk),
            concat: Vec::new(),
            input: Vec::new(),
            normalized: Vec::new(),
            seed: Vec::new(),
            recursive: Vec::new(),
            seed_rope: [0; 3],
            position: 0,
            canonical: 0,
            max_draft,
            elapsed_ms: 0.,
        })
    }
}
