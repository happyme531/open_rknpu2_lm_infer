use super::{Config, layer::Layer, runtime::DFlash};
use crate::{
    model::{
        ModelOptions,
        checkpoint::TensorStore,
        cpu::Cpu,
        projection::{Linear, Projection},
    },
    npu::{Backend, FloatBackend},
    ops::rope::MRope,
};
use anyhow::Result;
use std::path::Path;

fn vector(store: &TensorStore, name: &str, width: usize) -> Result<Vec<f32>> {
    let t = store.get(&format!("{name}.weight"))?;
    t.require_shape(&[width], name)?;
    t.floats()
}

impl DFlash {
    pub fn load(options: &ModelOptions, path: &Path) -> Result<Self> {
        let c = Config::load(path, options.context)?;
        let store = TensorStore::open(path)?;
        // The streaming target reserves domain 1 for cold expert staging.
        let domain = if c.dspark() { 2 } else { 1 };
        let devices = vec![
            Backend::new(options, domain)?,
            Backend::new(options, domain + 1)?,
        ];
        let floating: Vec<_> = devices
            .iter()
            .cloned()
            .map(FloatBackend::for_decoder)
            .collect();
        let h = c.hidden_size;
        let f = c.intermediate_size;
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        let cpu = Cpu::new(options.threads)?;
        let mut layers = Vec::new();
        for i in 0..c.num_hidden_layers {
            let p = format!("layers.{i}");
            let noncausal = c.layer_types[i] == "full_attention";
            let index = usize::from(noncausal);
            let device = &devices[index];
            let backend = &floating[index];
            let cache = if noncausal {
                device.noncausal_gqa(c.num_attention_heads, c.num_key_value_heads, c.head_dim)?
            } else {
                device.attention(
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    c.head_dim,
                    (c.head_dim as f32).sqrt().recip(),
                )?
            };
            cache.reserve(options.context)?;
            layers.push(Layer {
                input_norm: vector(&store, &format!("{p}.input_layernorm"), h)?,
                post_norm: vector(&store, &format!("{p}.post_attention_layernorm"), h)?,
                q_norm: vector(&store, &format!("{p}.self_attn.q_norm"), c.head_dim)?,
                k_norm: vector(&store, &format!("{p}.self_attn.k_norm"), c.head_dim)?,
                query: Linear::floating(backend, &store, &format!("{p}.self_attn.q_proj"), q, h)?,
                kv: Projection::floating(
                    backend,
                    &store,
                    vec![
                        (format!("{p}.self_attn.k_proj"), kv),
                        (format!("{p}.self_attn.v_proj"), kv),
                    ],
                    h,
                )?,
                output: Linear::floating(backend, &store, &format!("{p}.self_attn.o_proj"), h, q)?,
                gate_up: Projection::floating(
                    backend,
                    &store,
                    vec![
                        (format!("{p}.mlp.gate_proj"), f),
                        (format!("{p}.mlp.up_proj"), f),
                    ],
                    h,
                )?,
                down: Linear::floating(backend, &store, &format!("{p}.mlp.down_proj"), h, f)?,
                cache,
            });
            eprintln!("loaded DFlash layer {}/{}", i + 1, c.num_hidden_layers);
        }
        let fc = Linear::floating(&floating[0], &store, "fc", h, c.context_width())?;
        let hidden_norm = vector(&store, "hidden_norm", h)?;
        let final_norm = vector(&store, "norm", h)?;
        let markov = if c.dspark() {
            Some(super::markov::Markov::load(&c, &store, &floating[0])?)
        } else {
            None
        };
        let rope = MRope::new(
            c.head_dim,
            c.head_dim,
            c.rope_parameters.rope_theta,
            [c.head_dim / 2, 0, 0],
            false,
        )?;
        Ok(Self {
            config: c,
            layers,
            fc,
            hidden_norm,
            final_norm,
            rope,
            cpu,
            backends: devices,
            floating,
            context: options.context,
            chunk: options.chunk,
            position: 0,
            markov,
        })
    }
}
