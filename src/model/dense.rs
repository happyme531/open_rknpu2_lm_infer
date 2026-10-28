use super::{
    LanguageModel, ModelInfo, ModelOptions,
    checkpoint::{QuantLoader, Tensor, TensorStore},
    cpu::Cpu,
    output_head::OutputHead,
    projection::Projection,
    trace::Trace,
};
use crate::npu::{Attention, Backend, Linear};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
mod head;
struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    qkv: Projection,
    gate_up: Projection,
    o: Linear,
    down: Linear,
    attention: Attention,
}

#[derive(Default)]
struct Timings {
    load: f64,
    prepare: f64,
    linear: f64,
    attention: f64,
    cpu: f64,
    head: f64,
}
fn elapsed(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}
fn dimension(config: &Value, name: &str) -> Result<usize> {
    let value = config[name]
        .as_u64()
        .with_context(|| format!("missing or invalid {name}"))?;
    ensure!(value > 0, "invalid {name}");
    usize::try_from(value).with_context(|| format!("invalid {name}"))
}
fn norm(loader: &QuantLoader, name: &str, width: usize) -> Result<Vec<f32>> {
    let tensor = loader.store.get(name)?;
    tensor.require_shape(&[width], name)?;
    tensor.floats()
}

pub struct Dense {
    trace: Trace,
    options: ModelOptions,
    info: ModelInfo,
    loader: Arc<QuantLoader>,
    cpu: Cpu,
    backends: Vec<Backend>,
    layers: Vec<Layer>,
    embedding: Tensor,
    head: OutputHead,
    final_norm: Vec<f32>,
    cosine: Vec<f32>,
    sine: Vec<f32>,
    hidden: Vec<f32>,
    normalized: Vec<f32>,
    projected: Vec<f32>,
    qkv: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    attended: Vec<f32>,
    gate_up: Vec<f32>,
    middle: Vec<f32>,
    scatter: Vec<f32>,
    logits: Vec<f32>,
    batch_logits: Vec<f32>,
    h: usize,
    f: usize,
    qh: usize,
    kh: usize,
    d: usize,
    position: usize,
    reserved_tokens: usize,
    epsilon: f32,
    embedding_multiplier: f32,
    residual_multiplier: f32,
    logits_scaling: f32,
    timings: Timings,
    workspaces: super::workspace::Workspaces,
}

impl Dense {
    pub fn load(options: ModelOptions) -> Result<Self> {
        let start = Instant::now();
        let trace = Trace::from_environment()?;
        let config: Value =
            serde_json::from_slice(&std::fs::read(options.path.join("config.json"))?)?;
        super::registry::require(&config)?;
        let architecture = config["model_type"]
            .as_str()
            .context("missing model_type")?;
        ensure!(
            matches!(architecture, "qwen3" | "llama" | "granite")
                && config["hidden_act"] == "silu"
                && !config["attention_bias"].as_bool().unwrap_or(false)
                && !config["mlp_bias"].as_bool().unwrap_or(false)
                && !config["use_sliding_window"].as_bool().unwrap_or(false)
                && config["rope_scaling"].is_null()
                && config["rope_parameters"]["rope_type"]
                    .as_str()
                    .is_none_or(|s| s == "default")
                && config["num_experts"].is_null(),
            "dense Qwen3/Llama/Granite require standard RoPE, SiLU and no projection bias/sliding window"
        );
        let h = dimension(&config, "hidden_size")?;
        let f = dimension(&config, "intermediate_size")?;
        let qh = dimension(&config, "num_attention_heads")?;
        let kh = dimension(&config, "num_key_value_heads")?;
        let d = config["head_dim"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(h / qh);
        let vocab = dimension(&config, "vocab_size")?;
        let count = dimension(&config, "num_hidden_layers")?;
        let epsilon = config["rms_norm_eps"]
            .as_f64()
            .context("missing rms_norm_eps")? as f32;
        let theta = config["rope_parameters"]["rope_theta"]
            .as_f64()
            .or_else(|| config["rope_theta"].as_f64())
            .context("missing rope_theta")? as f32;
        let multiplier = |name: &str| -> Result<f32> {
            let value = if architecture == "granite" {
                config[name]
                    .as_f64()
                    .with_context(|| format!("missing {name}"))? as f32
            } else {
                1.0
            };
            ensure!(value.is_finite() && value > 0.0, "invalid {name}");
            Ok(value)
        };
        let embedding_multiplier = multiplier("embedding_multiplier")?;
        let residual_multiplier = multiplier("residual_multiplier")?;
        let logits_scaling = multiplier("logits_scaling")?;
        let attention_scale = if architecture == "granite" {
            config["attention_multiplier"]
                .as_f64()
                .context("missing attention_multiplier")? as f32
        } else {
            1.0 / (d as f32).sqrt()
        };
        ensure!(
            attention_scale.is_finite() && attention_scale > 0.0,
            "invalid attention_multiplier"
        );
        ensure!(
            options.context > 0
                && options.chunk > 0
                && options.threads > 0
                && options.chunk <= options.context
                && options.context <= dimension(&config, "max_position_embeddings")?
                && epsilon.is_finite()
                && epsilon > 0.0
                && theta.is_finite()
                && theta > 0.0
                && qh % kh == 0
                && d >= 2
                && d.is_multiple_of(2),
            "invalid dense model dimensions/context"
        );
        ensure!(
            d.is_multiple_of(32) && kh <= 32 && options.context <= 32768,
            "native GQA requires D divisible by 32, Hkv<=32 and context<=32768"
        );
        let cpu = Cpu::new(options.threads)?;
        let loader = Arc::new(QuantLoader::new(
            TensorStore::open(&options.path)?,
            &config,
        )?);
        let embedding = loader.store.get("model.embed_tokens.weight")?;
        embedding.require_shape(&[vocab, h], "embedding")?;
        ensure!(embedding.floating(), "embedding must be BF16/F16/F32");
        let quantized = loader.store.contains("lm_head.weight_scale")
            || loader.store.contains("lm_head.weight_packed");
        let tied = config["tie_word_embeddings"].as_bool().unwrap_or(false);
        ensure!(
            !(quantized && tied),
            "quantized lm_head requires tie_word_embeddings=false"
        );
        let floating_head = if quantized {
            None
        } else {
            let head = if tied {
                embedding.clone()
            } else {
                loader.store.get("lm_head.weight")?
            };
            head.require_shape(&[vocab, h], "lm_head")?;
            ensure!(head.floating(), "unquantized lm_head must be BF16/F16/F32");
            Some(head)
        };
        let kv_per_layer =
            Backend::attention_cache_bytes(&options, qh, kh, d, attention_scale)? as usize;
        let mut backends = vec![Backend::new(&options, 15)?];
        let mut reserved = 0usize;
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let prefix = format!("model.layers.{i}.");
            let qkv_modules = vec![
                (format!("{prefix}self_attn.q_proj"), qh * d),
                (format!("{prefix}self_attn.k_proj"), kh * d),
                (format!("{prefix}self_attn.v_proj"), kh * d),
            ];
            let gu_modules = vec![
                (format!("{prefix}mlp.gate_proj"), f),
                (format!("{prefix}mlp.up_proj"), f),
            ];
            let o_name = format!("{prefix}self_attn.o_proj");
            let down_name = format!("{prefix}mlp.down_proj");
            let modules = qkv_modules
                .iter()
                .chain(gu_modules.iter())
                .map(|(name, n)| (name.clone(), *n, h))
                .chain([(o_name.clone(), h, qh * d), (down_name.clone(), h, f)]);
            let mut estimate = 0usize;
            for (name, n, k) in modules {
                let bits = loader.bits(&name)? as usize;
                estimate += n.div_ceil(64) * 64 * k.div_ceil(32) * 32 * bits / 8 + n * 8 + 65536;
            }
            let budget = (2usize << 30) - (128usize << 20);
            ensure!(
                estimate + kv_per_layer < budget,
                "layer exceeds NPU IOMMU domain budget"
            );
            if backends.last().unwrap().weight_bytes() as usize + reserved + estimate + kv_per_layer
                > budget
            {
                ensure!(
                    backends.len() < 16,
                    "model exceeds available NPU IOMMU domains"
                );
                backends.push(Backend::new(&options, 15 - backends.len() as u32)?);
                reserved = 0;
            }
            let backend = backends.last().unwrap();
            let load_linear = |name: String, n: usize, k: usize| -> Result<Linear> {
                let matrix = cpu.pool.install(|| loader.load(&name, n, k))?;
                let loader = loader.clone();
                let pool = cpu.pool.clone();
                backend.load(
                    matrix,
                    Some(Box::new(move || pool.install(|| loader.load(&name, n, k)))),
                )
            };
            let layer = Layer {
                input_norm: norm(&loader, &format!("{prefix}input_layernorm.weight"), h)?,
                post_norm: norm(
                    &loader,
                    &format!("{prefix}post_attention_layernorm.weight"),
                    h,
                )?,
                q_norm: if architecture == "qwen3" {
                    norm(&loader, &format!("{prefix}self_attn.q_norm.weight"), d)?
                } else {
                    Vec::new()
                },
                k_norm: if architecture == "qwen3" {
                    norm(&loader, &format!("{prefix}self_attn.k_norm.weight"), d)?
                } else {
                    Vec::new()
                },
                qkv: Projection::load(backend, &loader, &cpu, qkv_modules, h)?,
                gate_up: Projection::load(backend, &loader, &cpu, gu_modules, h)?,
                o: load_linear(o_name, h, qh * d)?,
                down: load_linear(down_name, h, f)?,
                attention: backend.attention(qh, kh, d, attention_scale)?,
            };
            layers.push(layer);
            reserved += kv_per_layer;
            eprintln!("loaded layer {}/{count}", i + 1);
        }
        let head = if let Some(tensor) = floating_head {
            OutputHead::floating(tensor, vocab, h)?
        } else {
            ensure!(
                backends.len() < 16,
                "no NPU IOMMU domain available for lm_head"
            );
            let mut head_options = options.clone();
            head_options.chunk = 1;
            OutputHead::quantized(
                &head_options,
                15 - backends.len() as u32,
                &loader,
                &cpu,
                vocab,
                h,
            )?
        };
        let final_norm = norm(&loader, "model.norm.weight", h)?;
        let mut cosine = vec![0.0; options.context * d / 2];
        let mut sine = cosine.clone();
        for p in 0..options.context {
            for i in 0..d / 2 {
                let phase = p as f32 * theta.powf(-((2 * i) as f32) / d as f32);
                cosine[p * d / 2 + i] = phase.cos();
                sine[p * d / 2 + i] = phase.sin();
            }
        }
        let chunk = options.chunk;
        let reserved_tokens = (options.context.div_ceil(32) * 32).min(128);
        let scatter_width = layers
            .iter()
            .flat_map(|l| [&l.qkv, &l.gate_up])
            .map(Projection::scratch_width)
            .max()
            .unwrap_or(0);
        let eos_value = config.get("eos_token_id").cloned().unwrap_or(Value::Null);
        let mut eos_tokens = Vec::new();
        if let Some(id) = eos_value.as_u64() {
            ensure!(id < vocab as u64, "invalid EOS token");
            eos_tokens.push(id as u32);
        } else if let Some(ids) = eos_value.as_array() {
            for id in ids {
                let id = id.as_u64().context("invalid EOS token")?;
                ensure!(id < vocab as u64, "invalid EOS token");
                eos_tokens.push(id as u32);
            }
        } else {
            ensure!(eos_value.is_null(), "invalid EOS token");
        }
        let info = ModelInfo {
            id: options
                .path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("qwen3")
                .to_string(),
            architecture: architecture.into(),
            context: options.context,
            vocab_size: vocab,
            eos_tokens,
        };
        loader.store.release_pages()?;
        Ok(Self {
            trace,
            options,
            info,
            loader,
            cpu,
            backends,
            layers,
            embedding,
            head,
            final_norm,
            cosine,
            sine,
            hidden: vec![0.0; chunk * h],
            normalized: vec![0.0; chunk * h],
            projected: vec![0.0; chunk * h],
            qkv: vec![0.0; chunk * (qh + 2 * kh) * d],
            query: vec![0.0; chunk * qh * d],
            key: vec![0.0; chunk * kh * d],
            value: vec![0.0; chunk * kh * d],
            attended: vec![0.0; chunk * qh * d],
            gate_up: vec![0.0; chunk * 2 * f],
            middle: vec![0.0; chunk * f],
            scatter: vec![0.0; chunk * scatter_width],
            logits: vec![0.0; vocab],
            batch_logits: Vec::new(),
            h,
            f,
            qh,
            kh,
            d,
            position: 0,
            reserved_tokens,
            workspaces: Default::default(),
            epsilon,
            embedding_multiplier,
            residual_multiplier,
            logits_scaling,
            timings: Timings {
                load: elapsed(start),
                ..Default::default()
            },
        })
    }

    fn trace(&self, layer: i32, stage: &str, values: &[f32]) -> Result<()> {
        self.trace.save(self.position, layer, stage, values)
    }

    fn chunk(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.len() > 1 {
            self.workspaces.reset();
        }
        let rows = tokens.len();
        let h = self.h;
        let d = self.d;
        let q_width = self.qh * d;
        let kv_width = self.kh * d;
        let qkv_width = q_width + 2 * kv_width;
        let mut start = Instant::now();
        for (r, token) in tokens.iter().enumerate() {
            self.embedding
                .row(*token as usize, &mut self.hidden[r * h..(r + 1) * h]);
        }
        self.cpu
            .scale(&mut self.hidden[..rows * h], self.embedding_multiplier);
        self.timings.cpu += elapsed(start);
        for (index, layer) in self.layers.iter().enumerate() {
            let index = index as i32;
            start = Instant::now();
            self.cpu.norm(
                &self.hidden[..rows * h],
                &mut self.normalized[..rows * h],
                &layer.input_norm,
                self.epsilon,
            );
            self.timings.cpu += elapsed(start);
            self.trace(index, "input", &self.normalized[..rows * h])?;
            start = Instant::now();
            layer.qkv.run(
                &self.normalized[..rows * h],
                &mut self.qkv[..rows * qkv_width],
                rows,
                &mut self.scatter,
            )?;
            self.timings.linear += elapsed(start);
            self.trace(index, "qkv", &self.qkv[..rows * qkv_width])?;
            start = Instant::now();
            for r in 0..rows {
                let source = &self.qkv[r * qkv_width..(r + 1) * qkv_width];
                self.query[r * q_width..(r + 1) * q_width].copy_from_slice(&source[..q_width]);
                self.key[r * kv_width..(r + 1) * kv_width]
                    .copy_from_slice(&source[q_width..q_width + kv_width]);
                self.value[r * kv_width..(r + 1) * kv_width]
                    .copy_from_slice(&source[q_width + kv_width..]);
            }
            self.cpu.rope(
                &mut self.query[..rows * q_width],
                self.qh,
                d,
                &layer.q_norm,
                self.epsilon,
                &self.cosine,
                &self.sine,
                self.position,
            );
            self.cpu.rope(
                &mut self.key[..rows * kv_width],
                self.kh,
                d,
                &layer.k_norm,
                self.epsilon,
                &self.cosine,
                &self.sine,
                self.position,
            );
            self.timings.cpu += elapsed(start);
            self.trace(index, "query", &self.query[..rows * q_width])?;
            self.trace(index, "key", &self.key[..rows * kv_width])?;
            self.trace(index, "value", &self.value[..rows * kv_width])?;
            start = Instant::now();
            layer.attention.run(
                &self.query[..rows * q_width],
                &self.key[..rows * kv_width],
                &self.value[..rows * kv_width],
                &mut self.attended[..rows * q_width],
                rows,
                self.position,
            )?;
            self.timings.attention += elapsed(start);
            self.trace(index, "attn", &self.attended[..rows * q_width])?;
            start = Instant::now();
            layer.o.run(
                &self.attended[..rows * q_width],
                &mut self.projected[..rows * h],
                rows,
            )?;
            self.timings.linear += elapsed(start);
            self.trace(index, "o", &self.projected[..rows * h])?;
            start = Instant::now();
            self.cpu
                .scale(&mut self.projected[..rows * h], self.residual_multiplier);
            self.cpu
                .add(&mut self.hidden[..rows * h], &self.projected[..rows * h]);
            self.cpu.norm(
                &self.hidden[..rows * h],
                &mut self.normalized[..rows * h],
                &layer.post_norm,
                self.epsilon,
            );
            self.timings.cpu += elapsed(start);
            self.trace(index, "post", &self.normalized[..rows * h])?;
            start = Instant::now();
            if !self.trace.enabled()
                && layer.gate_up.run_swiglu(
                    &self.normalized[..rows * h],
                    &mut self.middle[..rows * self.f],
                    rows,
                )?
            {
                self.timings.linear += elapsed(start);
            } else {
                layer.gate_up.run(
                    &self.normalized[..rows * h],
                    &mut self.gate_up[..rows * 2 * self.f],
                    rows,
                    &mut self.scatter,
                )?;
                self.timings.linear += elapsed(start);
                self.trace(index, "gate_up", &self.gate_up[..rows * 2 * self.f])?;
                start = Instant::now();
                self.cpu.swiglu(
                    &self.gate_up[..rows * 2 * self.f],
                    &mut self.middle[..rows * self.f],
                    self.f,
                );
                self.timings.cpu += elapsed(start);
            }
            self.trace(index, "middle", &self.middle[..rows * self.f])?;
            start = Instant::now();
            layer.down.run(
                &self.middle[..rows * self.f],
                &mut self.projected[..rows * h],
                rows,
            )?;
            self.timings.linear += elapsed(start);
            self.trace(index, "down", &self.projected[..rows * h])?;
            start = Instant::now();
            self.cpu
                .scale(&mut self.projected[..rows * h], self.residual_multiplier);
            self.cpu
                .add(&mut self.hidden[..rows * h], &self.projected[..rows * h]);
            self.timings.cpu += elapsed(start);
            self.trace(index, "hidden", &self.hidden[..rows * h])?;
        }
        self.position += rows;
        Ok(())
    }
    fn advance_impl(&mut self, tokens: &[u32], cancelled: Option<&AtomicBool>) -> Result<usize> {
        ensure!(
            !tokens.is_empty() && tokens.len() <= self.options.context - self.position,
            "empty input or context capacity exceeded"
        );
        ensure!(
            tokens.iter().all(|t| (*t as usize) < self.info.vocab_size),
            "token outside model vocabulary"
        );
        self.reserve(self.position + tokens.len())?;
        let mut last_rows = 0;
        for chunk in tokens.chunks(self.options.chunk) {
            ensure!(
                !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
                "generation cancelled"
            );
            self.chunk(chunk)?;
            last_rows = chunk.len();
        }
        ensure!(
            !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
            "generation cancelled"
        );
        Ok(last_rows)
    }
    fn forward_impl(&mut self, tokens: &[u32], cancelled: Option<&AtomicBool>) -> Result<&[f32]> {
        let last_rows = self.advance_impl(tokens, cancelled)?;
        self.output_head(last_rows)?;
        Ok(&self.logits)
    }
}

impl LanguageModel for Dense {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn reset(&mut self) -> Result<()> {
        for layer in &self.layers {
            layer.attention.reset()?;
        }
        self.position = 0;
        self.workspaces.reset();
        self.timings.linear = 0.0;
        self.timings.attention = 0.0;
        self.timings.cpu = 0.0;
        self.timings.head = 0.0;
        Ok(())
    }
    fn reserve(&mut self, tokens: usize) -> Result<()> {
        ensure!(
            tokens > 0 && tokens <= self.options.context,
            "KV reservation exceeds context"
        );
        if tokens <= self.reserved_tokens {
            return Ok(());
        }
        for layer in &self.layers {
            layer.attention.reserve(tokens)?;
        }
        self.reserved_tokens = tokens.div_ceil(32) * 32;
        Ok(())
    }
    fn truncate(&mut self, position: usize) -> Result<bool> {
        ensure!(
            position <= self.position,
            "cannot extend KV cache by truncation"
        );
        for layer in &self.layers {
            layer.attention.truncate(position)?;
        }
        self.position = position;
        Ok(true)
    }
    fn prepare(&mut self, prompt_tokens: usize) -> Result<()> {
        ensure!(
            prompt_tokens > 0 && prompt_tokens <= self.options.context,
            "prompt exceeds context capacity"
        );
        let start = Instant::now();
        let previous_tunes: usize = self
            .backends
            .iter()
            .chain(self.head.backend())
            .map(Backend::tune_operations)
            .sum();
        let mut rows = super::workspace::physical_rows([prompt_tokens], self.options.chunk);
        self.workspaces.prepare(&mut rows, &[1]);
        for backend in &self.backends {
            backend.retain_workspaces(&rows)?;
            for &rows in &rows {
                backend.prepare(rows)?;
            }
        }
        if let Some(head) = self.head.backend() {
            head.retain_workspaces(&[1])?;
            head.prepare(1)?;
        }
        self.timings.prepare += elapsed(start);
        if self
            .backends
            .iter()
            .chain(self.head.backend())
            .map(Backend::tune_operations)
            .sum::<usize>()
            != previous_tunes
        {
            self.loader.store.release_pages()?;
        }
        Ok(())
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.advance_impl(tokens, None).map(|_| ())
    }
    fn forward(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.forward_impl(tokens, None)
    }
    fn forward_cancellable(&mut self, tokens: &[u32], cancelled: &AtomicBool) -> Result<&[f32]> {
        self.forward_impl(tokens, Some(cancelled))
    }
    fn position(&self) -> usize {
        self.position
    }
    fn token_nll(&mut self, tokens: &[u32], targets: &[u32]) -> Result<Vec<f64>> {
        use rayon::prelude::*;
        ensure!(
            !tokens.is_empty()
                && tokens.len() == targets.len()
                && tokens.len() <= self.options.context - self.position,
            "invalid scoring input/target length"
        );
        ensure!(
            tokens
                .iter()
                .chain(targets)
                .all(|&t| (t as usize) < self.info.vocab_size),
            "scoring token outside model vocabulary"
        );
        self.reserve(self.position + tokens.len())?;
        let mut result = Vec::with_capacity(tokens.len());
        for (tokens, targets) in tokens
            .chunks(self.options.chunk)
            .zip(targets.chunks(self.options.chunk))
        {
            self.chunk(tokens)?;
            let rows = tokens.len();
            let start = Instant::now();
            self.cpu.norm(
                &self.hidden[..rows * self.h],
                &mut self.normalized[..rows * self.h],
                &self.final_norm,
                self.epsilon,
            );
            let vocab = self.info.vocab_size;
            self.batch_logits.resize(rows * vocab, 0.0);
            self.head.prepare(rows)?;
            self.head.run(
                &self.cpu,
                &self.normalized[..rows * self.h],
                &mut self.batch_logits,
                rows,
            )?;
            if self.logits_scaling != 1.0 {
                for value in &mut self.batch_logits {
                    *value /= self.logits_scaling;
                }
            }
            self.timings.head += elapsed(start);
            let start = Instant::now();
            let values = self.cpu.pool.install(|| {
                self.batch_logits
                    .par_chunks(vocab)
                    .zip(targets.par_iter())
                    .map(|(logits, &target)| crate::ppl::negative_log_likelihood(logits, target))
                    .collect::<Result<Vec<_>>>()
            })?;
            result.extend(values);
            self.timings.cpu += elapsed(start);
        }
        Ok(result)
    }
    fn stats(&self) -> Value {
        let mut result = super::metrics::backend_stats(&self.backends, self.head.backend());
        let buffers = [
            &self.hidden,
            &self.normalized,
            &self.projected,
            &self.qkv,
            &self.query,
            &self.key,
            &self.value,
            &self.attended,
            &self.gate_up,
            &self.middle,
            &self.scatter,
            &self.logits,
            &self.batch_logits,
            &self.cosine,
            &self.sine,
        ];
        let norm_bytes = self
            .layers
            .iter()
            .map(|l| {
                l.input_norm.capacity()
                    + l.post_norm.capacity()
                    + l.q_norm.capacity()
                    + l.k_norm.capacity()
            })
            .sum::<usize>()
            * 4
            + self.final_norm.capacity() * 4;
        result["cpu_scratch_bytes"] =
            json!(buffers.iter().map(|b| b.capacity() * 4).sum::<usize>());
        result["cpu_norm_weight_bytes"] = json!(norm_bytes);
        result["attention_cache_payload_bytes"] = json!(
            self.layers
                .iter()
                .map(|layer| layer.attention.bytes().expect("live native KV cache"))
                .sum::<u64>()
        );
        result["mapped_weight_bytes"] = json!(self.loader.store.mapped_bytes());
        result["lm_head_weight_bits"] = json!(self.head.bits());
        result["lm_head_backend"] = json!(
            self.head
                .bits()
                .map(|b| format!("open-rknpu2-w{b}a8"))
                .unwrap_or_else(|| "cpu-floating".into())
        );
        result["position"] = json!(self.position);
        result["trace_directory"] = json!(self.trace.path());
        result["layers"] = json!(self.layers.len());
        result["attention_backend"] = json!("native-npu-gqa-stable-softmax");
        result["activation_quantization"] = json!("full-K maxabs/127, round-away, clamp[-127,127]");
        result["kv_dtype"] = json!("float16");
        result["attention_tile"] = json!(self.options.attention_tile);
        result["model_load_ms"] = json!(self.timings.load);
        result["prepare_ms"] = json!(self.timings.prepare);
        result["linear_ms"] = json!(self.timings.linear);
        result["attention_ms"] = json!(self.timings.attention);
        result["cpu_ms"] = json!(self.timings.cpu);
        result["head_ms"] = json!(self.timings.head);
        result["tuning_enabled"] = json!(self.options.tune);
        result["tuning_search_enabled"] = json!(self.options.tune_search);
        result["tune_cache"] = json!(self.options.tune_cache);
        result
    }
}
