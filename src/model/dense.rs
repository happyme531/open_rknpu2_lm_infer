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
mod step;
use super::batch::{BatchConfig, SeqPool};
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

struct Weights {
    layers: Layers<Layer>,
    embedding: Tensor,
    head: OutputHead,
    final_norm: Vec<f32>,
    cosine: Vec<f32>,
    sine: Vec<f32>,
    geometry: Geometry,
    logits_scaling: f32,
    trace: Trace,
}

struct Buffers {
    cpu: Cpu,
    batches: Vec<Batch>,
    logits: Vec<f32>,
    batch_logits: Vec<f32>,
    timings: Timings,
    dual_groups: usize,
    workspaces: super::workspace::Workspaces,
}

struct DenseSeqState {
    caches: Vec<Attention>,
    position: usize,
    reserved_tokens: usize,
}

impl DenseSeqState {
    fn reset(&mut self, reset_attention: impl Fn(&Attention) -> Result<()>) -> Result<()> {
        for cache in &self.caches {
            reset_attention(cache)?;
        }
        self.position = 0;
        Ok(())
    }
}

pub struct Dense {
    options: ModelOptions,
    info: ModelInfo,
    loader: Arc<QuantLoader>,
    backends: Vec<Backend>,
    secondary: Option<ExecutionLane>,
    weights: Weights,
    work: Buffers,
    sequences: SeqPool<DenseSeqState>,
}

impl Dense {
    pub fn load(options: ModelOptions) -> Result<Self> {
        Self::load_mode(options, None)
    }

    pub(super) fn load_batch(options: ModelOptions, config: BatchConfig) -> Result<Self> {
        Self::load_mode(options, Some(config))
    }

    fn load_mode(options: ModelOptions, batch_config: Option<BatchConfig>) -> Result<Self> {
        ensure!(
            (1..=2).contains(&options.prefill_ubatches),
            "prefill-ubatches must be 1 or 2"
        );
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
        let slots = batch_config.map_or(1, |config| config.slots);
        let kv_for_layer = if batch_config.is_some() {
            (kv_per_layer + super::batch::kv_tail_budget(&options, kh, d)) * slots
        } else {
            kv_per_layer
        };
        let mut caches: Vec<Vec<Attention>> =
            (0..slots).map(|_| Vec::with_capacity(count)).collect();
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
            let budget = if options.prefill_ubatches == 2 || batch_config.is_some() {
                1usize << 30
            } else {
                (2usize << 30) - (128usize << 20)
            };
            ensure!(
                estimate + kv_for_layer < budget,
                "layer {i} exceeds IOMMU domain budget: weights={estimate}, KV={kv_for_layer}, budget={budget}"
            );
            if backends.last().unwrap().weight_bytes() as usize + reserved + estimate + kv_for_layer
                > budget
            {
                ensure!(
                    backends.len() < 16,
                    "no IOMMU domain for layer {i}: weights={estimate}, KV={kv_for_layer}, budget={budget}"
                );
                backends.push(Backend::new(&options, 15 - backends.len() as u32)?);
                reserved = 0;
            }
            let backend = backends.last().unwrap();
            if batch_config.is_some() {
                for (slot, seq) in caches.iter_mut().enumerate() {
                    seq.push(super::batch::reserve_cache(
                        backend,
                        slot,
                        i,
                        qh,
                        kh,
                        d,
                        attention_scale,
                    )?);
                }
            }
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
            };
            if batch_config.is_none() {
                caches[0].push(backend.attention(qh, kh, d, attention_scale)?);
            }
            layers.push(layer);
            reserved += kv_for_layer;
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
        let chunk = batch_config.map_or(options.chunk, |config| config.max_rows);
        let reserved_tokens = options.context.div_ceil(32) * 32;
        let reserved_tokens = if batch_config.is_some() {
            reserved_tokens
        } else {
            reserved_tokens.min(128)
        };
        let states = caches
            .into_iter()
            .map(|caches| DenseSeqState {
                caches,
                position: 0,
                reserved_tokens,
            })
            .collect();
        let sequences = SeqPool::new(states, options.context, batch_config);
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
        let workspaces = super::workspace::Workspaces::for_generation(!options.tune_search)?;
        let mut model = Self {
            options,
            info,
            loader,
            backends,
            secondary,
            weights: Weights {
                layers: layers.into(),
                embedding,
                head,
                final_norm,
                cosine,
                sine,
                geometry,
                logits_scaling,
                trace,
            },
            work: Buffers {
                cpu,
                batches,
                logits: if batch_config.is_none() {
                    vec![0.0; vocab]
                } else {
                    Vec::new()
                },
                batch_logits: Vec::with_capacity(batch_config.map_or(0, |c| c.slots * vocab)),
                dual_groups: 0,
                workspaces,
                timings: Timings {
                    load: elapsed(start),
                    ..Default::default()
                },
            },
            sequences,
        };
        if let Some(config) = batch_config {
            super::batch::prepare(config, &model.backends, model.weights.head.backend())?;
            if !model.weights.trace.enabled() {
                let buckets = config.decoder_buckets();
                for layer in model.weights.layers.iter() {
                    layer.gate_up.prepare_swiglu(&buckets)?;
                }
            }
            model.loader.store.release_pages()?;
            super::batch::report(&model)?;
        }
        if model.options.compress_w8a8 {
            crate::npu::release_compression_scratch(
                model.backends.iter().chain(model.weights.head.backend()),
            );
            model.work.timings.load = elapsed(start);
        }
        Ok(model)
    }

    fn workspace_targets(
        &mut self,
    ) -> (
        &mut super::workspace::Workspaces,
        super::workspace::Targets<'_>,
    ) {
        (
            &mut self.work.workspaces,
            super::workspace::Targets {
                primary: &self.backends,
                secondary: self.secondary.as_ref(),
                head: self.weights.head.backend(),
                recurrent: [None, None],
                routers: None,
                floating: &[],
                chunk: self.options.chunk,
                decode_rows: &[1],
            },
        )
    }

    fn chunk(&mut self, tokens: &[u32]) -> Result<()> {
        self.group(&[tokens], None)
    }

    fn group(&mut self, tokens: &[&[u32]], cancelled: Option<&AtomicBool>) -> Result<()> {
        Self::forward_layers(
            &self.weights,
            &mut self.work,
            self.sequences.single_mut(),
            self.secondary.as_mut(),
            tokens,
            cancelled,
        )?;
        if self.options.compress_w8a8 {
            crate::npu::release_compression_scratch(&self.backends);
        }
        Ok(())
    }

    fn forward_layers(
        weights: &Weights,
        work: &mut Buffers,
        seq: &mut DenseSeqState,
        secondary: Option<&mut ExecutionLane>,
        tokens: &[&[u32]],
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        let start = Instant::now();
        let mut position = seq.position;
        for (batch, tokens) in work.batches.iter_mut().zip(tokens) {
            batch.embed(
                tokens,
                position,
                &weights.embedding,
                &work.cpu,
                weights.geometry,
            );
            position += tokens.len();
        }
        work.timings.cpu += elapsed(start);
        let pass = LayerPass {
            cpu: &work.cpu,
            trace: &weights.trace,
            cosine: &weights.cosine,
            sine: &weights.sine,
            geometry: weights.geometry,
        };
        let caches = &seq.caches;
        if tokens.len() == 2 {
            let secondary = secondary.expect("two prepared execution lanes");
            let (first, second) = work.batches.split_at_mut(1);
            let mut a = Timings::default();
            let mut b = Timings::default();
            pipeline::run(
                &weights.layers,
                cancelled,
                |index, layer| {
                    pass.single(layer, &caches[index], index as i32, &mut first[0], &mut a)
                },
                |index, layer| {
                    secondary.run(|| {
                        pass.single(layer, &caches[index], index as i32, &mut second[0], &mut b)
                    })
                },
            )?;
            work.timings.linear += a.linear + b.linear;
            work.timings.attention += a.attention + b.attention;
            work.timings.cpu += a.cpu + b.cpu;
            work.dual_groups += 1;
            work.batches.swap(0, 1);
        } else {
            for (index, (layer, cache)) in weights.layers.iter().zip(caches).enumerate() {
                pass.single(
                    &layer,
                    cache,
                    index as i32,
                    &mut work.batches[0],
                    &mut work.timings,
                )?;
            }
        }
        seq.position = position;
        Ok(())
    }
    fn advance_impl(
        &mut self,
        tokens: &[u32],
        cancelled: Option<&AtomicBool>,
        mut progress: Option<&mut dyn FnMut(crate::progress::ModelProgress)>,
    ) -> Result<usize> {
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= self.options.context - self.sequences.single().position,
            "empty input or context capacity exceeded"
        );
        ensure!(
            tokens.iter().all(|t| (*t as usize) < self.info.vocab_size),
            "token outside model vocabulary"
        );
        self.reserve(self.sequences.single().position + tokens.len())?;
        let start = self.sequences.single().position;
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
            let completed = self.sequences.single().position - start;
            if completed < tokens.len()
                && let Some(progress) = progress.as_mut()
            {
                progress(crate::progress::ModelProgress::Prefill {
                    completed,
                    total: tokens.len(),
                });
            }
        }
        ensure!(
            !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
            "generation cancelled"
        );
        Ok(last_rows)
    }
    fn forward_impl(&mut self, tokens: &[u32], cancelled: Option<&AtomicBool>) -> Result<&[f32]> {
        self.work.workspaces.begin_input(
            self.sequences.single().position,
            tokens.len(),
            &self.backends,
        );
        self.run_input(tokens, cancelled, None, true)?;
        Ok(&self.work.logits)
    }

    fn run_input(
        &mut self,
        tokens: &[u32],
        cancelled: Option<&AtomicBool>,
        progress: Option<&mut dyn FnMut(crate::progress::ModelProgress)>,
        logits: bool,
    ) -> Result<()> {
        let result = self
            .advance_impl(tokens, cancelled, progress)
            .and_then(|rows| {
                if logits {
                    self.output_head(rows)?;
                } else {
                    self.work
                        .workspaces
                        .finish_head_free(self.sequences.single().position);
                }
                Ok(())
            });
        if result.is_err() {
            let (workspaces, targets) = self.workspace_targets();
            workspaces.abort(targets)?;
        }
        result
    }

    fn prepare_input(
        &mut self,
        prompt_tokens: usize,
        id: Option<super::prefill::PlanId>,
    ) -> Result<()> {
        ensure!(
            prompt_tokens > 0 && prompt_tokens <= self.options.context,
            "prompt exceeds context capacity"
        );
        let start = Instant::now();
        let previous_tunes: usize = self
            .backends
            .iter()
            .chain(self.weights.head.backend())
            .map(Backend::tune_operations)
            .sum();
        let rows = super::workspace::physical_rows([prompt_tokens], self.options.chunk);
        let head_rows = super::workspace::head_rows(&rows, &[1], self.options.tune_search);
        let start_position = self.sequences.single().position;
        let plan = super::workspace::PrefillPlan {
            span: start_position..start_position + prompt_tokens,
            id,
            rows: &rows,
            head_rows: &head_rows,
            paired: super::prefill::raw_uses_secondary(
                prompt_tokens,
                self.options.chunk,
                self.options.prefill_ubatches,
            ),
        };
        let loader = self.loader.clone();
        let (workspaces, targets) = self.workspace_targets();
        workspaces.prepare(targets, plan, |targets, _| {
            if targets
                .primary
                .iter()
                .chain(targets.head)
                .map(Backend::tune_operations)
                .sum::<usize>()
                != previous_tunes
            {
                loader.store.release_pages()?;
            }
            Ok(())
        })?;
        self.work.timings.prepare += elapsed(start);
        Ok(())
    }

    fn run_prepared(
        &mut self,
        prepared: crate::model::PreparedPrefill<'_>,
        cancelled: &AtomicBool,
        progress: Option<&mut dyn FnMut(crate::progress::ModelProgress)>,
    ) -> Result<()> {
        ensure!(
            prepared.input.images.is_empty(),
            "this model does not support images"
        );
        ensure!(
            prepared.start == self.sequences.single().position
                && prepared.start < prepared.input.tokens.len(),
            "prepared input position mismatch"
        );
        self.work.workspaces.begin_prepared(
            prepared.start..prepared.input.tokens.len(),
            prepared.plan_id,
        )?;
        self.run_input(
            &prepared.input.tokens[prepared.start..],
            Some(cancelled),
            progress,
            true,
        )
    }
}

impl LanguageModel for Dense {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn reset(&mut self) -> Result<()> {
        self.sequences.single_mut().reset(Attention::reset)?;
        self.work.dual_groups = 0;
        self.work.workspaces.reset();
        self.work.timings.linear = 0.0;
        self.work.timings.attention = 0.0;
        self.work.timings.cpu = 0.0;
        self.work.timings.head = 0.0;
        Ok(())
    }
    fn reserve(&mut self, tokens: usize) -> Result<()> {
        ensure!(
            tokens > 0 && tokens <= self.options.context,
            "KV reservation exceeds context"
        );
        if tokens <= self.sequences.single().reserved_tokens {
            return Ok(());
        }
        for cache in &self.sequences.single().caches {
            cache.reserve(tokens)?;
        }
        self.sequences.single_mut().reserved_tokens = tokens.div_ceil(32) * 32;
        Ok(())
    }
    fn truncate(&mut self, position: usize) -> Result<bool> {
        ensure!(
            position <= self.sequences.single().position,
            "cannot extend KV cache by truncation"
        );
        for cache in &self.sequences.single().caches {
            cache.truncate(position)?;
        }
        self.sequences.single_mut().position = position;
        Ok(true)
    }
    fn prepare(&mut self, prompt_tokens: usize) -> Result<()> {
        self.prepare_input(prompt_tokens, None)
    }
    fn prepare_prepared<'a>(
        &mut self,
        input: &'a crate::input::PreparedInput,
        start: usize,
    ) -> Result<crate::model::PreparedPrefill<'a>> {
        ensure!(
            input.images.is_empty(),
            "this model does not support images"
        );
        ensure!(
            start == self.sequences.single().position && start < input.tokens.len(),
            "invalid prepared prompt"
        );
        let plan_id = Some(super::prefill::PlanId::new());
        self.prepare_input(input.tokens.len() - start, plan_id)?;
        Ok(crate::model::PreparedPrefill {
            input,
            start,
            steps: Vec::new(),
            checkpoint: None,
            plan_id,
        })
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.work.workspaces.begin_input(
            self.sequences.single().position,
            tokens.len(),
            &self.backends,
        );
        self.run_input(tokens, None, None, false)
    }
    fn forward(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.forward_impl(tokens, None)
    }
    fn forward_cancellable(&mut self, tokens: &[u32], cancelled: &AtomicBool) -> Result<&[f32]> {
        self.forward_impl(tokens, Some(cancelled))
    }
    fn forward_prepared(
        &mut self,
        prepared: crate::model::PreparedPrefill<'_>,
        cancelled: &AtomicBool,
    ) -> Result<&[f32]> {
        self.run_prepared(prepared, cancelled, None)?;
        Ok(&self.work.logits)
    }
    fn forward_prepared_with_progress(
        &mut self,
        prepared: crate::model::PreparedPrefill<'_>,
        cancelled: &AtomicBool,
        progress: &mut dyn FnMut(crate::progress::ModelProgress),
    ) -> Result<&[f32]> {
        let total = prepared.input.tokens.len().saturating_sub(prepared.start);
        self.run_prepared(prepared, cancelled, Some(&mut *progress))?;
        progress(crate::progress::ModelProgress::Prefill {
            completed: total,
            total,
        });
        Ok(&self.work.logits)
    }
    fn position(&self) -> usize {
        self.sequences.single().position
    }
    fn token_nll(&mut self, tokens: &[u32], targets: &[u32]) -> Result<Vec<f64>> {
        use rayon::prelude::*;
        ensure!(
            !tokens.is_empty()
                && tokens.len() == targets.len()
                && tokens.len() <= self.options.context - self.sequences.single().position,
            "invalid scoring input/target length"
        );
        ensure!(
            tokens
                .iter()
                .chain(targets)
                .all(|&t| (t as usize) < self.info.vocab_size),
            "scoring token outside model vocabulary"
        );
        let position = self.sequences.single().position;
        self.work
            .workspaces
            .begin_input(position, tokens.len(), &self.backends);
        self.reserve(position + tokens.len())?;
        let mut result = Vec::with_capacity(tokens.len());
        for (tokens, targets) in tokens
            .chunks(self.options.chunk)
            .zip(targets.chunks(self.options.chunk))
        {
            self.chunk(tokens)?;
            let rows = tokens.len();
            let start = Instant::now();
            let batch = &mut self.work.batches[0];
            self.work.cpu.norm(
                &batch.hidden[..rows * self.weights.geometry.h],
                &mut batch.normalized[..rows * self.weights.geometry.h],
                &self.weights.final_norm,
                self.weights.geometry.epsilon,
            );
            let vocab = self.info.vocab_size;
            self.work.batch_logits.resize(rows * vocab, 0.0);
            self.weights.head.prepare(rows)?;
            self.weights.head.run(
                &self.work.cpu,
                &self.work.batches[0].normalized[..rows * self.weights.geometry.h],
                &mut self.work.batch_logits,
                rows,
            )?;
            if self.weights.logits_scaling != 1.0 {
                for value in &mut self.work.batch_logits {
                    *value /= self.weights.logits_scaling;
                }
            }
            self.work.timings.head += elapsed(start);
            let start = Instant::now();
            let values = self.work.cpu.pool.install(|| {
                self.work
                    .batch_logits
                    .par_chunks(vocab)
                    .zip(targets.par_iter())
                    .map(|(logits, &target)| crate::ppl::negative_log_likelihood(logits, target))
                    .collect::<Result<Vec<_>>>()
            })?;
            result.extend(values);
            self.work.timings.cpu += elapsed(start);
        }
        Ok(result)
    }
    fn stats(&self) -> Value {
        let mut result = super::metrics::backend_stats(
            self.backends
                .iter()
                .chain(self.secondary.iter().flat_map(|lane| lane.backends())),
            self.weights.head.backend(),
        );
        let buffers = [
            &self.work.logits,
            &self.work.batch_logits,
            &self.weights.cosine,
            &self.weights.sine,
        ];
        let norm_bytes = self
            .weights
            .layers
            .iter()
            .map(|l| l.input_norm.len() + l.post_norm.len() + l.q_norm.len() + l.k_norm.len())
            .sum::<usize>()
            * 4
            + self.weights.final_norm.capacity() * 4;
        result["cpu_scratch_bytes"] = json!(
            buffers.iter().map(|b| b.capacity() * 4).sum::<usize>()
                + self.work.batches.iter().map(Batch::bytes).sum::<usize>()
        );
        result["cpu_norm_weight_bytes"] = json!(norm_bytes);
        result["attention_cache_payload_bytes"] = json!(
            self.sequences
                .single()
                .caches
                .iter()
                .map(|cache| cache.bytes().expect("live native KV cache"))
                .sum::<u64>()
        );
        result["mapped_weight_bytes"] = json!(self.loader.store.mapped_bytes());
        result["lm_head_weight_bits"] = json!(self.weights.head.bits());
        result["lm_head_backend"] = json!(
            self.weights
                .head
                .bits()
                .map(|b| format!("open-rknpu2-w{b}a8"))
                .unwrap_or_else(|| "cpu-floating".into())
        );
        result["prefill_workspace_cached_domains"] =
            json!(super::workspace::cached_domains(&self.backends));
        result["prefill_ubatches"] = json!(self.options.prefill_ubatches);
        result["dual_prefill_groups"] = json!(self.work.dual_groups);
        result["stage_timing_scope"] =
            json!("sum of per-ubatch elapsed stages; concurrent lanes overlap");
        result["position"] = json!(self.sequences.single().position);
        result["trace_directory"] = json!(self.weights.trace.path());
        result["layers"] = json!(self.weights.layers.len());
        result["attention_backend"] = json!("native-npu-gqa-stable-softmax");
        result["activation_quantization"] = json!("full-K maxabs/127, round-away, clamp[-127,127]");
        result["kv_dtype"] = json!("float16");
        result["attention_tile"] = json!(self.options.attention_tile);
        result["model_load_ms"] = json!(self.work.timings.load);
        result["prepare_ms"] = json!(self.work.timings.prepare);
        result["linear_ms"] = json!(self.work.timings.linear);
        result["attention_ms"] = json!(self.work.timings.attention);
        result["cpu_ms"] = json!(self.work.timings.cpu);
        result["head_ms"] = json!(self.work.timings.head);
        result["tuning_enabled"] = json!(self.options.tune);
        result["tuning_search_enabled"] = json!(self.options.tune_search);
        result["tune_cache"] = json!(self.options.tune_cache);
        result
    }
}
