use super::{
    config::{Config, LayerType},
    decoder::{Buffers, FeedForward, Layer, Mixer},
    head,
    runtime::Qwen35,
};
use crate::layers::gated_delta::{LayerState, MixerWeights};
use crate::model::trace::Trace;
use crate::{
    model::{
        ModelInfo, ModelOptions,
        checkpoint::QuantLoader,
        cpu::Cpu,
        projection::{Linear, Projection},
    },
    npu::Backend,
};
use anyhow::{Result, ensure};
use serde_json::Value;
use std::{sync::Arc, time::Instant};

fn vector(loader: &QuantLoader, name: &str, width: usize) -> Result<Vec<f32>> {
    let tensor = loader.store.get(name)?;
    tensor.require_shape(&[width], name)?;
    tensor.floats()
}

pub(super) fn load(options: ModelOptions) -> Result<Qwen35> {
    let start = Instant::now();
    let raw: Value = serde_json::from_slice(&std::fs::read(options.path.join("config.json"))?)?;
    let config = Config::parse(&raw)?;
    let expert_slots = config
        .moe
        .as_ref()
        .map(|m| options.experts.slots(m.experts, m.top_k))
        .transpose()?
        .unwrap_or(0);
    if options.experts.storage == crate::model::experts::Storage::Stream
        && let Some(m) = &config.moe
    {
        eprintln!(
            "expert cache: {}% -> {expert_slots}/{} routed experts per layer ({:.3}% effective)",
            options.experts.cache_percent,
            m.experts,
            100. * expert_slots as f64 / m.experts as f64
        );
    }
    ensure!(
        config.moe.is_none() || [1, 3, 7].contains(&options.npu_mask),
        "MoE requires NPU mask 1/3/7"
    );
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
    let mut quant_config = raw.clone();
    quant_config["quantization_config"] = config.quantization.clone().unwrap_or(Value::Null);
    let loader = Arc::new(QuantLoader::open(&options.path, &quant_config)?);
    let expert_store = if let Some(m) = &config.moe {
        if let Some(directory) = &options.experts.cache {
            Some(crate::model::experts::prepare::open(
                directory,
                crate::model::experts::store::Geometry {
                    native_version: crate::npu::Moe::blob_version(),
                    layers: config.layers.len(),
                    experts: m.experts,
                    hidden: config.hidden_size,
                    intermediate: config.intermediate_size,
                    down_bits: 4,
                },
                &options,
                || loader.source_stamps(),
                |layer, expert| {
                    let result = cpu.pool.install(|| {
                        loader.expert(
                            &format!("{}.layers.{layer}.mlp.experts", config.text_prefix),
                            expert,
                            config.hidden_size,
                            config.intermediate_size,
                        )
                    })?;
                    if expert + 1 == m.experts {
                        loader.store.release_pages()?;
                    }
                    Ok(result)
                },
            )?)
        } else {
            None
        }
    } else {
        None
    };
    let paging = if options.experts.storage == crate::model::experts::Storage::Stream {
        let store = expert_store.as_ref().ok_or_else(|| {
            anyhow::anyhow!("streaming requires a MoE checkpoint and --expert-store")
        })?;
        Some(crate::model::experts::Paging::new(
            &options,
            store.clone(),
            expert_slots,
        )?)
    } else {
        None
    };
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
    let mut reserved_kv = 0;
    let mut layers = Vec::with_capacity(config.layers.len());
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
            .chain(gu.iter().filter(|_| config.moe.is_none()))
            .map(|(name, n)| (name, *n, h))
            .chain([(&out, h, out_k)])
            .chain(config.moe.is_none().then_some((&down, h, f)))
        {
            if !loader.quantized(name) {
                continue;
            }
            let bits = loader.bits(name)? as usize;
            estimate += n.div_ceil(64) * 64 * k.div_ceil(32) * 32 * bits / 8 + n * 8 + 65536;
        }
        if let Some(m) = &config.moe {
            let experts = if paging.is_some() {
                expert_slots
            } else {
                m.experts
            };
            estimate += experts * h * f * 3 / 2 + experts * h * 4 + m.shared_intermediate * h * 3;
        }
        let kv = if *kind == LayerType::FullAttention {
            kv_bytes
        } else {
            0
        };
        let budget = (2usize << 30) - (128usize << 20);
        ensure!(
            estimate + kv < budget,
            "Qwen3.5 layer exceeds IOMMU domain budget"
        );
        if backends.last().unwrap().weight_bytes() as usize + reserved_kv + estimate + kv > budget {
            ensure!(
                backends.len() < 15,
                "Qwen3.5 exceeds available weight domains"
            );
            backends.push(Backend::new(&options, 15 - backends.len() as u32)?);
            ensure!(
                paging.is_none() || backends.len() < 13,
                "no IOMMU domain for cold expert staging"
            );
            reserved_kv = 0;
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
                    state: Box::new(LayerState::new(
                        config.gdn,
                        config.conv_kernel,
                        8.max(options.speculation.window() + 1),
                    )?),
                }
            }
            LayerType::FullAttention => Mixer::Attention {
                cache: backend.attention(
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    scale,
                )?,
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
            },
        };
        layers.push(Layer {
            input_norm: vector(&loader, &format!("{p}.input_layernorm.weight"), h)?,
            post_norm: vector(&loader, &format!("{p}.post_attention_layernorm.weight"), h)?,
            inputs: Projection::load(backend, &loader, &cpu, inputs, h)?,
            output: Linear::load(backend, &loader, &cpu, &out, h, out_k)?,
            ffn: if config.moe.is_some() {
                FeedForward::Routed(Box::new(super::moe::Routed::load(
                    backend,
                    &loader,
                    &cpu,
                    &config,
                    &p,
                    i,
                    super::moe::Storage {
                        options: &options,
                        store: expert_store.as_ref(),
                        paging: paging.as_ref(),
                    },
                )?))
            } else {
                FeedForward::Dense {
                    gate_up: Projection::load(backend, &loader, &cpu, gu, h)?,
                    down: Linear::load(backend, &loader, &cpu, &down, h, f)?,
                }
            },
            mixer,
        });
        reserved_kv += kv;
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
    if config.moe.is_none() && crate::npu::GatedDelta::supports(config.gdn, config.conv_kernel) {
        work.recurrent =
            crate::layers::gated_delta::Scratch::with_npu(backends.last().unwrap().clone());
    }
    let info = ModelInfo {
        id: options
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        architecture: raw["model_type"].as_str().unwrap_or("qwen3_5").into(),
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
    Ok(Qwen35 {
        options,
        info,
        config,
        loader,
        cpu,
        backends,
        layers,
        paging,
        embedding,
        final_norm,
        head,
        work,
        position: 0,
        reserved: 0,
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
        dflash: None,
    })
}
