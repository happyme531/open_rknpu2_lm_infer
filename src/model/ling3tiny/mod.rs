//! Ling-3.0-tiny hybrid KDA/MLA MoE decoder.
use crate::model::trace;
use crate::{
    model::{
        LanguageModel, ModelInfo, ModelOptions,
        checkpoint::{QuantLoader, Tensor},
        cpu::Cpu,
        experts::Paging,
        output_head::OutputHead,
        trace::Trace,
    },
    npu::{Backend, Tf32Backend},
    ops::norm,
};
use anyhow::{Result, ensure};
use std::{
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
mod config;
mod load;
mod mixer;
mod moe;

struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mixer: mixer::Mixer,
    ffn: moe::FeedForward,
}
pub struct Ling3Tiny {
    options: ModelOptions,
    info: ModelInfo,
    config: config::Config,
    weights: Arc<QuantLoader>,
    cpu: Cpu,
    backends: Vec<Backend>,
    router_backends: Vec<Tf32Backend>,
    layers: Vec<Layer>,
    paging: Option<Rc<Paging>>,
    embedding: Tensor,
    final_norm: Vec<f32>,
    head: OutputHead,
    position: usize,
    logits: Vec<f32>,
    last_hidden: Vec<f32>,
    trace: Trace,
    load_ms: f64,
    forward_ms: f64,
    head_ms: f64,
}
impl Ling3Tiny {
    pub fn load(options: ModelOptions) -> Result<Self> {
        load::load(options)
    }
    fn consume(&mut self, tokens: &[u32], cancel: Option<&AtomicBool>) -> Result<()> {
        ensure!(
            !tokens.is_empty() && self.position + tokens.len() <= self.options.context,
            "empty Ling input or context exceeded"
        );
        ensure!(
            tokens
                .iter()
                .all(|&id| (id as usize) < self.info.vocab_size),
            "token outside Ling vocabulary"
        );
        for batch in tokens.chunks(self.options.chunk) {
            let result = self.chunk(batch, cancel);
            if let Err(error) = result {
                self.reset()?;
                return Err(error);
            }
        }
        Ok(())
    }
    fn chunk(&mut self, tokens: &[u32], cancel: Option<&AtomicBool>) -> Result<()> {
        let start = Instant::now();
        let rows = tokens.len();
        let c = &self.config;
        let h = c.hidden;
        let (mut hidden, mut normalized) = trace::measure(c"ling.embedding", || {
            let mut hidden = vec![0.; rows * h];
            for (&id, row) in tokens.iter().zip(hidden.chunks_mut(h)) {
                self.embedding.row(id as usize, row);
            }
            (hidden, vec![0.; rows * h])
        });
        #[cfg(test)]
        if let Some(p) = &self.paging {
            p.forward_marker(self.position, rows, true);
        }
        for (index, layer) in self.layers.iter_mut().enumerate() {
            ensure!(
                !cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)),
                "generation cancelled"
            );
            let _layer = trace::Span::new(c"ling.layer", index as i32, rows, self.position);
            trace::measure(c"ling.layer_norm_residual", || {
                norm::rms_norm(
                    &hidden,
                    &layer.input_norm,
                    c.epsilon,
                    false,
                    &mut normalized,
                )
            })?;
            self.trace
                .save(self.position, index as i32, "input", &normalized)?;
            let attention = layer
                .mixer
                .forward(&normalized, rows, self.position, c, &self.cpu)?;
            self.trace
                .save(self.position, index as i32, "o", &attention)?;
            trace::measure(c"ling.layer_norm_residual", || {
                self.cpu.add(&mut hidden, &attention);
                norm::rms_norm(&hidden, &layer.post_norm, c.epsilon, false, &mut normalized)
            })?;
            self.trace
                .save(self.position, index as i32, "post", &normalized)?;
            let ffn = layer
                .ffn
                .run(&normalized, rows, self.position, c, &self.cpu)?;
            self.trace.save(self.position, index as i32, "ffn", &ffn)?;
            trace::measure(c"ling.layer_norm_residual", || {
                self.cpu.add(&mut hidden, &ffn)
            });
            self.trace
                .save(self.position, index as i32, "hidden", &hidden)?;
        }
        #[cfg(test)]
        if let Some(p) = &self.paging {
            p.forward_marker(self.position, rows, false);
        }
        self.last_hidden.clear();
        self.last_hidden
            .extend_from_slice(&hidden[(rows - 1) * h..]);
        self.position += rows;
        self.forward_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
    fn output(&mut self) -> Result<&[f32]> {
        let _head = trace::Span::new(c"ling.output_head", -1, 1, self.position);
        let start = Instant::now();
        let mut input = vec![0.; self.config.hidden];
        norm::rms_norm(
            &self.last_hidden,
            &self.final_norm,
            self.config.epsilon,
            false,
            &mut input,
        )?;
        self.logits.resize(self.config.vocab, 0.);
        self.head.run(&self.cpu, &input, &mut self.logits, 1)?;
        debug_assert!(
            self.logits.iter().all(|x| x.is_finite()),
            "nonfinite Ling logits"
        );
        self.trace.save(self.position, -1, "logits", &self.logits)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        self.head_ms += elapsed;
        Ok(&self.logits)
    }
}
impl LanguageModel for Ling3Tiny {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn position(&self) -> usize {
        self.position
    }
    fn reset(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.mixer.reset()?;
        }
        if let Some(p) = &self.paging {
            p.reset();
        }
        self.position = 0;
        self.forward_ms = 0.;
        self.head_ms = 0.;
        self.last_hidden.clear();
        self.logits.clear();
        Ok(())
    }
    fn reserve(&mut self, tokens: usize) -> Result<()> {
        ensure!(
            tokens <= self.options.context,
            "Ling reserve exceeds context"
        );
        for layer in &self.layers {
            layer.mixer.reserve(tokens)?;
        }
        Ok(())
    }
    fn prepare(&mut self, prompt_tokens: usize) -> Result<()> {
        let mut rows = vec![1, self.options.chunk.min(prompt_tokens.max(1))];
        let tail = prompt_tokens % self.options.chunk;
        if tail > 0 {
            rows.push(tail);
        }
        rows.sort_unstable();
        rows.dedup();
        for backend in &self.backends {
            for &n in &rows {
                backend.prepare(n)?;
            }
            backend.retain_workspaces(&rows)?;
        }
        for backend in &self.router_backends {
            backend.retain_workspaces(&rows);
            for &n in &rows {
                backend.prepare(self.config.hidden, self.config.experts, n)?;
            }
        }
        // Floating backends retain the current batch workspace per geometry;
        // smaller tail batches and decode switch lazily when consumed.
        let floating_rows = self.options.chunk.min(prompt_tokens.max(1));
        for layer in &self.layers {
            layer.mixer.prepare(floating_rows)?;
        }
        self.head.prepare(1)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.consume(tokens, None)
    }
    fn forward(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.consume(tokens, None)?;
        self.output()
    }
    fn forward_cancellable(&mut self, tokens: &[u32], cancel: &AtomicBool) -> Result<&[f32]> {
        self.consume(tokens, Some(cancel))?;
        self.output()
    }
    fn truncate(&mut self, position: usize) -> Result<bool> {
        if position == self.position {
            return Ok(true);
        }
        if position == 0 {
            self.reset()?;
            return Ok(true);
        }
        Ok(false)
    }
    fn stats(&self) -> serde_json::Value {
        let mut result = crate::model::metrics::backend_stats(&self.backends, self.head.backend());
        let mixers: Vec<_> = self
            .layers
            .iter()
            .map(|l| l.mixer.stats().expect("live Ling mixer state"))
            .collect();
        let kv_bytes = mixers
            .iter()
            .map(|m| {
                ["key_bytes", "value_bytes", "tail_bytes"]
                    .iter()
                    .map(|key| m[key].as_u64().unwrap_or(0))
                    .sum::<u64>()
            })
            .sum::<u64>();
        let fields = serde_json::json!({"architecture":"bailing_hybrid","position":self.position,
            "model_load_ms":self.load_ms,"decoder_ms":self.forward_ms,"head_ms":self.head_ms,
            "mapped_weight_bytes":self.weights.store.mapped_bytes(),"mixers":mixers,
            "attention_backend":"native-npu-mla-absorbed-stable-softmax","attention_cache_payload_bytes":kv_bytes,
            "mla_projection_backend":"native-npu-fp16","router_projection_backend":"native-npu-tf32",
            "kda_backend":"cpu-fp32","kv_dtype":"float16","expert_storage":if self.paging.is_some() {"stream"} else {"resident"},
            "joint_moe_shared_layers":self.layers.iter().filter(|l| l.ffn.joint_shared()).count(),
            "expert_paging":self.paging.as_ref().map(|p| p.stats()),"lm_head_weight_bits":self.head.bits(),
            "tuning_enabled":self.options.tune,"tuning_search_enabled":self.options.tune_search,
            "trace_directory":self.trace.path()});
        result
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        result
    }
}
