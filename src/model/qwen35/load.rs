use super::{
    config::{Config, LayerType},
    decoder::{Buffers, Layer, Mixer},
    head,
    runtime::Qwen35,
};
use crate::layers::gated_delta::{LayerState, MixerWeights};
use crate::model::trace::Trace;
use crate::{
    model::{
        ModelInfo, ModelOptions,
        checkpoint::{QuantLoader, TensorStore},
        cpu::Cpu,
        projection::Projection,
    },
    npu::{Backend, Linear},
};
use anyhow::{Result, ensure};
use serde_json::Value;
use std::{sync::Arc, time::Instant};

fn vector(loader: &QuantLoader, name: &str, width: usize) -> Result<Vec<f32>> {
    let tensor = loader.store.get(name)?;
    tensor.require_shape(&[width], name)?;
    tensor.floats()
}

fn linear(
    backend: &Backend,
    loader: &Arc<QuantLoader>,
    cpu: &Cpu,
    name: String,
    n: usize,
    k: usize,
) -> Result<Linear> {
    let matrix = cpu.pool.install(|| loader.load(&name, n, k))?;
    let reload = loader.clone();
    let pool = cpu.pool.clone();
    backend.load(
        matrix,
        Some(Box::new(move || pool.install(|| reload.load(&name, n, k)))),
    )
}

pub(super) fn load(options: ModelOptions) -> Result<Qwen35> {
    let start = Instant::now();
    let raw: Value = serde_json::from_slice(&std::fs::read(options.path.join("config.json"))?)?;
    let config = Config::parse(&raw)?;
    config.validate_context(options.context)?;
    ensure!(
        options.chunk > 0 && options.chunk <= options.context,
        "invalid Qwen3.5 chunk size"
    );
    ensure!(
        config.head_dim.is_multiple_of(32) && config.num_key_value_heads <= 32,
        "Qwen3.5 native attention requires D divisible by 32 and Hkv<=32"
    );
    let cpu = Cpu::new(options.threads)?;
    let quant_config = serde_json::json!({"quantization_config": config.quantization});
    let loader = Arc::new(QuantLoader::new(
        TensorStore::open(&options.path)?,
        &quant_config,
    )?);
    let prefix = config.text_prefix;
    let embedding = loader.store.get(&format!("{prefix}.embed_tokens.weight"))?;
    embedding.require_shape(&[config.vocab_size, config.hidden_size], "embedding")?;
    ensure!(
        embedding.floating(),
        "Qwen3.5 embedding must be BF16/F16/F32"
    );
    let final_norm = vector(
        &loader,
        &format!("{prefix}.norm.weight"),
        config.hidden_size,
    )?;
    let scale = (config.head_dim as f32).sqrt().recip();
    let kv_bytes = Backend::attention_cache_bytes(
        &options,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        scale,
    )? as usize;
    let mut backends = vec![Backend::new(&options, 15)?];
    let mut reserved_bytes = 0;
    let mut pending = Vec::with_capacity(config.layers.len());
    for (i, kind) in config.layers.iter().enumerate() {
        let p = format!("{prefix}.layers.{i}");
        let h = config.hidden_size;
        let f = config.intermediate_size;
        let q = config.num_attention_heads * config.head_dim;
        let kv = config.num_key_value_heads * config.head_dim;
        let inputs = match kind {
            LayerType::LinearAttention => vec![
                (
                    format!("{p}.linear_attn.in_proj_qkv"),
                    config.gdn.conv_width(),
                ),
                (
                    format!("{p}.linear_attn.in_proj_z"),
                    config.gdn.value_width(),
                ),
                (
                    format!("{p}.linear_attn.in_proj_a"),
                    config.gdn.value_heads(),
                ),
                (
                    format!("{p}.linear_attn.in_proj_b"),
                    config.gdn.value_heads(),
                ),
            ],
            LayerType::FullAttention => vec![
                (format!("{p}.self_attn.q_proj"), 2 * q),
                (format!("{p}.self_attn.k_proj"), kv),
                (format!("{p}.self_attn.v_proj"), kv),
            ],
        };
        let (out, out_k) = match kind {
            LayerType::LinearAttention => (
                format!("{p}.linear_attn.out_proj"),
                config.gdn.value_width(),
            ),
            LayerType::FullAttention => (format!("{p}.self_attn.o_proj"), q),
        };
        let gu = vec![
            (format!("{p}.mlp.gate_proj"), f),
            (format!("{p}.mlp.up_proj"), f),
        ];
        let down = format!("{p}.mlp.down_proj");
        let mut estimate = 0usize;
        for (name, n, k) in inputs
            .iter()
            .chain(gu.iter())
            .map(|(name, n)| (name, *n, h))
            .chain([(&out, h, out_k), (&down, h, f)])
        {
            let bits = loader.bits(name)? as usize;
            estimate += n.div_ceil(64) * 64 * k.div_ceil(32) * 32 * bits / 8 + n * 8 + 65536;
        }
        let kv = if *kind == LayerType::FullAttention {
            kv_bytes
        } else {
            0
        };
        // Payload is only part of the 32-bit IOVA footprint: shared projection,
        // GDN and Attention workspaces and allocation alignment also need room.
        let budget = 1usize << 30;
        ensure!(
            estimate + kv < budget,
            "Qwen3.5 layer exceeds IOMMU domain budget"
        );
        if reserved_bytes + estimate + kv > budget {
            ensure!(
                backends.len() < 15,
                "Qwen3.5 exceeds available weight domains"
            );
            backends.push(Backend::new(&options, 15 - backends.len() as u32)?);
            reserved_bytes = 0;
        }
        let backend = backends.last().unwrap();
        let mixer = match kind {
            LayerType::LinearAttention => {
                let name = format!("{p}.linear_attn");
                let conv = loader.store.get(&format!("{name}.conv1d.weight"))?;
                conv.require_shape(
                    &[config.gdn.conv_width(), 1, config.conv_kernel],
                    "GDN conv1d",
                )?;
                let weights = MixerWeights::new(
                    config.gdn,
                    config.conv_kernel,
                    conv.floats()?,
                    vector(&loader, &format!("{name}.A_log"), config.gdn.value_heads())?,
                    vector(
                        &loader,
                        &format!("{name}.dt_bias"),
                        config.gdn.value_heads(),
                    )?,
                    vector(
                        &loader,
                        &format!("{name}.norm.weight"),
                        config.gdn.value_dim(),
                    )?,
                    config.epsilon,
                )?;
                Mixer::Recurrent {
                    weights,
                    state: Box::new(LayerState::new(config.gdn, config.conv_kernel, 8)?),
                }
            }
            LayerType::FullAttention => {
                // Reserve all large KV ranges before loading any projection
                // weights/plans into these domains.
                let cache = backend.attention(
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    scale,
                )?;
                cache.reserve(options.context)?;
                Mixer::Attention {
                    cache,
                    q_norm: vector(
                        &loader,
                        &format!("{p}.self_attn.q_norm.weight"),
                        config.head_dim,
                    )?,
                    k_norm: vector(
                        &loader,
                        &format!("{p}.self_attn.k_norm.weight"),
                        config.head_dim,
                    )?,
                }
            }
        };
        reserved_bytes += estimate + kv;
        pending.push((
            p,
            inputs,
            out,
            out_k,
            gu,
            down,
            mixer,
            backends.len() - 1,
            *kind,
        ));
    }
    let mut layers = Vec::with_capacity(config.layers.len());
    for (i, (p, inputs, out, out_k, gu, down, mixer, domain, kind)) in
        pending.into_iter().enumerate()
    {
        let backend = &backends[domain];
        let h = config.hidden_size;
        let f = config.intermediate_size;
        layers.push(Layer {
            input_norm: vector(&loader, &format!("{p}.input_layernorm.weight"), h)?,
            post_norm: vector(&loader, &format!("{p}.post_attention_layernorm.weight"), h)?,
            inputs: Projection::load(backend, &loader, &cpu, inputs, h)?,
            output: linear(backend, &loader, &cpu, out, h, out_k)?.into(),
            gate_up: Projection::load(backend, &loader, &cpu, gu, h)?,
            down: linear(backend, &loader, &cpu, down, h, f)?.into(),
            mixer,
        });
        eprintln!(
            "loaded Qwen3.5 layer {}/{} ({kind:?})",
            i + 1,
            config.layers.len()
        );
    }
    let head = head::load(
        &options,
        15 - backends.len() as u32,
        &loader,
        &cpu,
        &embedding,
        config.tie_word_embeddings,
    )?;
    let mut work = Buffers::new(&config, options.chunk);
    if crate::npu::GatedDelta::supports(config.gdn, config.conv_kernel) {
        work.recurrent =
            crate::layers::gated_delta::Scratch::with_npu(backends.last().unwrap().clone());
    }
    let secondary = if options.prefill_ubatches == 2 {
        Some(crate::npu::ExecutionLane::new(&backends)?)
    } else {
        None
    };
    let secondary_work = secondary.as_ref().map(|lane| {
        let mut work = Buffers::new(&config, options.chunk);
        if crate::npu::GatedDelta::supports(config.gdn, config.conv_kernel) {
            work.recurrent = crate::layers::gated_delta::Scratch::with_npu(
                lane.backend(backends.last().unwrap()),
            );
        }
        work
    });
    let secondary_features = secondary
        .as_ref()
        .map(|_| super::features::Features::default());
    let info = ModelInfo {
        id: options
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        architecture: "qwen3_5".into(),
        context: options.context,
        vocab_size: config.vocab_size,
        eos_tokens: crate::model::metadata::eos_tokens(
            &options.path,
            &config.eos_tokens,
            config.vocab_size,
        )?,
    };
    let mtp_available =
        loader.store.contains("mtp.fc.weight") || loader.store.contains("mtp.fc.weight_packed");
    let trace = Trace::from_environment()?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let prefix_cache = super::cache::PrefixCache::new(options.checkpoints);
    let reserved = options.context.div_ceil(32) * 32;
    Ok(Qwen35 {
        options,
        info,
        config,
        loader,
        cpu,
        backends,
        layers: layers.into(),
        embedding,
        final_norm,
        head,
        work,
        secondary,
        secondary_work,
        dual_groups: 0,
        position: 0,
        reserved,
        logits: Vec::new(),
        trace,
        load_ms,
        forward_ms: 0.0,
        head_ms: 0.0,
        phases: Default::default(),
        mtp_available,
        vision: None,
        next_rope_position: 0,
        round: None,
        mtp: None,
        features: Default::default(),
        secondary_features,
        dflash: None,
        prefix_cache,
    })
}
