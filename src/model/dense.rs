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
mod batch;
use super::pipeline::{self, Layers};
use crate::npu::ExecutionLane;
mod head;
#[cfg(test)]
mod tests;
use batch::{Batch, Geometry, LayerPass};
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
    secondary: Option<ExecutionLane>,
    layers: Layers<Layer>,
    embedding: Tensor,
    head: OutputHead,
    final_norm: Vec<f32>,
    cosine: Vec<f32>,
    sine: Vec<f32>,
    batches: Vec<Batch>,
    logits: Vec<f32>,
    batch_logits: Vec<f32>,
    geometry: Geometry,
    position: usize,
    reserved_tokens: usize,
    logits_scaling: f32,
    timings: Timings,
    dual_groups: usize,
}

impl Dense {
    pub fn load(options: ModelOptions) -> Result<Self> {
        ensure!(
            (1..=2).contains(&options.prefill_ubatches),
            "prefill-ubatches must be 1 or 2"
        );
        let start = Instant::now();
        let trace = Trace::from_environment()?;
        let config: Value =
            serde_json::from_slice(&std::fs::read(options.path.join("config.json"))?)?;
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
            // Reserve device address space for both cached workspace lanes.
            let budget = if options.prefill_ubatches == 2 {
                1usize << 30
            } else {
                (2usize << 30) - (128usize << 20)
            };
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
        let geometry = Geometry {
            h,
            f,
            qh,
            kh,
            d,
            epsilon,
            embedding_multiplier,
            residual_multiplier,
        };
        let secondary = if options.prefill_ubatches == 2 {
            Some(ExecutionLane::new(&backends)?)
        } else {
            None
        };
        let batches = (0..options.prefill_ubatches)
            .map(|_| Batch::new(chunk, geometry, scatter_width))
            .collect();
        let mut model = Self {
            trace,
            options,
            info,
            loader,
            cpu,
            backends,
            secondary,
            layers: layers.into(),
            embedding,
            head,
            final_norm,
            cosine,
            sine,
            batches,
            logits: vec![0.0; vocab],
            batch_logits: Vec::new(),
            geometry,
            position: 0,
            reserved_tokens,
            logits_scaling,
            dual_groups: 0,
            timings: Timings {
                load: elapsed(start),
                ..Default::default()
            },
        };
        if model.options.compress_w8a8 {
            crate::npu::release_compression_scratch(
                model.backends.iter().chain(model.head.backend()),
            );
            model.timings.load = elapsed(start);
        }
        Ok(model)
    }

    fn trace(&self, layer: i32, stage: &str, values: &[f32]) -> Result<()> {
        self.trace.save(self.position, layer, stage, values)
    }

    fn chunk(&mut self, tokens: &[u32]) -> Result<()> {
        self.group(&[tokens], None)
    }

    fn group(&mut self, tokens: &[&[u32]], cancelled: Option<&AtomicBool>) -> Result<()> {
        let start = Instant::now();
        let mut position = self.position;
        for (batch, tokens) in self.batches.iter_mut().zip(tokens) {
            batch.embed(tokens, position, &self.embedding, &self.cpu, self.geometry);
            position += tokens.len();
        }
        self.timings.cpu += elapsed(start);
        let pass = LayerPass {
            cpu: &self.cpu,
            trace: &self.trace,
            cosine: &self.cosine,
            sine: &self.sine,
            geometry: self.geometry,
        };
        if tokens.len() == 2 {
            let secondary = self
                .secondary
                .as_mut()
                .expect("two prepared execution lanes");
            let (first, second) = self.batches.split_at_mut(1);
            let mut a = Timings::default();
            let mut b = Timings::default();
            pipeline::run(
                &self.layers,
                cancelled,
                |index, layer| pass.run(layer, index as i32, &mut first[0], &mut a),
                |index, layer| {
                    secondary.run(|| pass.run(layer, index as i32, &mut second[0], &mut b))
                },
            )?;
            self.timings.linear += a.linear + b.linear;
            self.timings.attention += a.attention + b.attention;
            self.timings.cpu += a.cpu + b.cpu;
            self.dual_groups += 1;
        } else {
            for (index, layer) in self.layers.iter_mut().enumerate() {
                pass.run(layer, index as i32, &mut self.batches[0], &mut self.timings)?;
            }
        }
        self.position = position;
        if tokens.len() == 2 {
            self.batches.swap(0, 1);
        }
        if self.options.compress_w8a8 {
            crate::npu::release_compression_scratch(&self.backends);
        }
        Ok(())
    }
    fn advance_impl(
        &mut self,
        tokens: &[u32],
        cancelled: Option<&AtomicBool>,
        mut progress: Option<&mut dyn FnMut(crate::progress::ModelProgress)>,
    ) -> Result<usize> {
        ensure!(
            !tokens.is_empty() && tokens.len() <= self.options.context - self.position,
            "empty input or context capacity exceeded"
        );
        ensure!(
            tokens.iter().all(|t| (*t as usize) < self.info.vocab_size),
            "token outside model vocabulary"
        );
        self.reserve(self.position + tokens.len())?;
        let start = self.position;
        if let Some(progress) = progress.as_mut() {
            progress(crate::progress::ModelProgress::Prefill {
                completed: 0,
                total: tokens.len(),
            });
        }
        let mut last_rows = 0;
        for (first, second) in
            super::prefill::batches(tokens, self.options.chunk, self.options.prefill_ubatches)
        {
            ensure!(
                !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
                "generation cancelled"
            );
            let batches = [first, second.unwrap_or_default()];
            let count = if second.is_some() { 2 } else { 1 };
            crate::progress::prefill_shape(first.len() != self.options.chunk, || {
                self.group(&batches[..count], cancelled)
            })?;
            last_rows = batches[count - 1].len();
            let completed = self.position - start;
            if completed < tokens.len() {
                if let Some(progress) = progress.as_mut() {
                    progress(crate::progress::ModelProgress::Prefill {
                        completed,
                        total: tokens.len(),
                    });
                }
            }
        }
        ensure!(
            !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
            "generation cancelled"
        );
        Ok(last_rows)
    }
    fn forward_impl(&mut self, tokens: &[u32], cancelled: Option<&AtomicBool>) -> Result<&[f32]> {
        let last_rows = self.advance_impl(tokens, cancelled, None)?;
        self.output_head(last_rows)?;
        Ok(&self.logits)
    }
}

impl LanguageModel for Dense {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn reset(&mut self) -> Result<()> {
        for layer in self.layers.iter() {
            layer.attention.reset()?;
        }
        self.position = 0;
        self.dual_groups = 0;
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
        for layer in self.layers.iter() {
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
        for layer in self.layers.iter() {
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
        let mut rows = vec![1, self.options.chunk];
        if !prompt_tokens.is_multiple_of(self.options.chunk) {
            rows.push(prompt_tokens % self.options.chunk);
        }
        rows.sort_unstable();
        rows.dedup();
        for backend in &self.backends {
            backend.retain_workspaces(&rows)?;
            backend.prepare(1)?;
            crate::progress::prefill_shape(prompt_tokens < self.options.chunk, || {
                backend.prepare(prompt_tokens.min(self.options.chunk))
            })?;
            if !prompt_tokens.is_multiple_of(self.options.chunk) {
                crate::progress::prefill_shape(true, || {
                    backend.prepare(prompt_tokens % self.options.chunk)
                })?;
            }
        }
        if let Some(secondary) = &self.secondary {
            secondary.prepare(self.options.chunk)?;
        }
        if let Some(head) = self.head.backend() {
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
        self.advance_impl(tokens, None, None).map(|_| ())
    }
    fn forward(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.forward_impl(tokens, None)
    }
    fn forward_cancellable(&mut self, tokens: &[u32], cancelled: &AtomicBool) -> Result<&[f32]> {
        self.forward_impl(tokens, Some(cancelled))
    }
    fn forward_prepared_with_progress(
        &mut self,
        prepared: crate::model::PreparedPrefill<'_>,
        cancelled: &AtomicBool,
        progress: &mut dyn FnMut(crate::progress::ModelProgress),
    ) -> Result<&[f32]> {
        ensure!(
            prepared.input.images.is_empty(),
            "this model does not support images"
        );
        let tokens = &prepared.input.tokens[prepared.start..];
        let rows = self.advance_impl(tokens, Some(cancelled), Some(&mut *progress))?;
        self.output_head(rows)?;
        progress(crate::progress::ModelProgress::Prefill {
            completed: tokens.len(),
            total: tokens.len(),
        });
        Ok(&self.logits)
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
            let batch = &mut self.batches[0];
            self.cpu.norm(
                &batch.hidden[..rows * self.geometry.h],
                &mut batch.normalized[..rows * self.geometry.h],
                &self.final_norm,
                self.geometry.epsilon,
            );
            let vocab = self.info.vocab_size;
            self.batch_logits.resize(rows * vocab, 0.0);
            self.head.prepare(rows)?;
            self.head.run(
                &self.cpu,
                &self.batches[0].normalized[..rows * self.geometry.h],
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
        let mut result = super::metrics::backend_stats(
            self.backends
                .iter()
                .chain(self.secondary.iter().flat_map(|lane| lane.backends())),
            self.head.backend(),
        );
        let buffers = [&self.logits, &self.batch_logits, &self.cosine, &self.sine];
        let norm_bytes = self
            .layers
            .iter()
            .map(|l| l.input_norm.len() + l.post_norm.len() + l.q_norm.len() + l.k_norm.len())
            .sum::<usize>()
            * 4
            + self.final_norm.capacity() * 4;
        result["cpu_scratch_bytes"] = json!(
            buffers.iter().map(|b| b.capacity() * 4).sum::<usize>()
                + self.batches.iter().map(Batch::bytes).sum::<usize>()
        );
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
        result["prefill_ubatches"] = json!(self.options.prefill_ubatches);
        result["dual_prefill_groups"] = json!(self.dual_groups);
        result["stage_timing_scope"] =
            json!("sum of per-ubatch elapsed stages; concurrent lanes overlap");
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
