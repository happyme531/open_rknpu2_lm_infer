//! Ling-3.0-tiny hybrid KDA/MLA MoE decoder.
use crate::model::trace;
use crate::{
    model::{
        LanguageModel, ModelInfo, ModelOptions,
        checkpoint::{QuantLoader, Tensor},
        cpu::Cpu,
        experts::paged::Paging,
        output_head::OutputHead,
        trace::Trace,
        workspace::{PrefillPlan, RouterTargets, Targets, Workspaces},
    },
    npu::{Backend, FloatBackend, Tf32Backend},
    ops::norm,
};
use anyhow::{Result, ensure};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
mod buffers;
mod cache;
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
    floating_backends: Vec<FloatBackend>,
    layers: Vec<Layer>,
    paging: Option<Paging>,
    embedding: Tensor,
    final_norm: Vec<f32>,
    head: OutputHead,
    work: buffers::Buffers,
    position: usize,
    logits: Vec<f32>,
    last_hidden: Vec<f32>,
    prefix_cache: cache::PrefixCache,
    trace: Trace,
    load_ms: f64,
    forward_ms: f64,
    head_ms: f64,
    workspaces: Workspaces,
}
impl Ling3Tiny {
    pub fn load(options: ModelOptions) -> Result<Self> {
        load::load(options)
    }
    fn check_input(&self, tokens: &[u32]) -> Result<()> {
        ensure!(
            !tokens.is_empty() && tokens.len() <= self.options.context - self.position,
            "empty Ling input or context exceeded"
        );
        ensure!(
            tokens
                .iter()
                .all(|&id| (id as usize) < self.info.vocab_size),
            "token outside Ling vocabulary"
        );
        Ok(())
    }
    fn consume(&mut self, tokens: &[u32], cancel: Option<&AtomicBool>, logits: bool) -> Result<()> {
        self.check_input(tokens)?;
        self.workspaces
            .begin_input(self.position, tokens.len(), &self.backends);
        let result = (|| {
            for batch in tokens.chunks(self.options.chunk) {
                self.chunk(batch, cancel)?;
            }
            ensure!(
                !cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)),
                "generation cancelled"
            );
            if logits {
                self.output()?;
            } else {
                self.workspaces.finish_head_free(self.position);
            }
            Ok(())
        })();
        self.complete_input(result)
    }
    fn chunk(&mut self, tokens: &[u32], cancel: Option<&AtomicBool>) -> Result<()> {
        let start = Instant::now();
        let rows = tokens.len();
        let c = &self.config;
        let h = c.hidden;
        self.work.resize(rows, h);
        let w = &mut self.work;
        trace::measure(c"ling.embedding", || {
            for (&id, row) in tokens.iter().zip(w.hidden.chunks_mut(h)) {
                self.embedding.row(id as usize, row);
            }
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
                    &w.hidden,
                    &layer.input_norm,
                    c.epsilon,
                    false,
                    &mut w.normalized,
                )
            })?;
            self.trace
                .save(self.position, index as i32, "input", &w.normalized)?;
            layer.mixer.forward(
                &w.normalized,
                &mut w.output,
                self.position,
                c,
                &self.cpu,
                &mut w.mixer,
            )?;
            self.trace
                .save(self.position, index as i32, "o", &w.output)?;
            trace::measure(c"ling.layer_norm_residual", || {
                self.cpu.add(&mut w.hidden, &w.output);
                norm::rms_norm(
                    &w.hidden,
                    &layer.post_norm,
                    c.epsilon,
                    false,
                    &mut w.normalized,
                )
            })?;
            self.trace
                .save(self.position, index as i32, "post", &w.normalized)?;
            layer.ffn.run(
                &w.normalized,
                &mut w.output,
                self.position,
                c,
                &self.cpu,
                &mut w.ffn,
                self.paging.as_ref(),
            )?;
            self.trace
                .save(self.position, index as i32, "ffn", &w.output)?;
            trace::measure(c"ling.layer_norm_residual", || {
                self.cpu.add(&mut w.hidden, &w.output)
            });
            self.trace
                .save(self.position, index as i32, "hidden", &w.hidden)?;
        }
        #[cfg(test)]
        if let Some(p) = &self.paging {
            p.forward_marker(self.position, rows, false);
        }
        self.last_hidden.clear();
        self.last_hidden
            .extend_from_slice(&w.hidden[(rows - 1) * h..]);
        self.position += rows;
        if self.options.compress_w8a8 {
            crate::npu::release_compression_scratch(&self.backends);
        }
        self.forward_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
    fn prepare_rows(
        &mut self,
        span: std::ops::Range<usize>,
        id: Option<super::prefill::PlanId>,
        rows: Vec<usize>,
    ) -> Result<()> {
        let floating_rows = rows[0].min(self.options.chunk);
        let rows = crate::model::workspace::physical_rows(rows, self.options.chunk);
        let head_rows = crate::model::workspace::head_rows(&rows, &[1], self.options.tune_search);
        let (workspaces, targets, layers) = self.workspace_targets();
        // FP16 retains one row shape per geometry. Prepare the first actual
        // batch; later batches and decode switch lazily when consumed.
        workspaces.prepare(
            targets,
            PrefillPlan {
                span,
                id,
                rows: &rows,
                head_rows: &head_rows,
                paired: false,
            },
            |_, _| {
                for layer in layers {
                    layer.mixer.prepare(floating_rows)?;
                }
                Ok(())
            },
        )
    }
    fn workspace_targets(&mut self) -> (&mut Workspaces, Targets<'_>, &[Layer]) {
        (
            &mut self.workspaces,
            Targets {
                primary: &self.backends,
                secondary: None,
                head: self.head.backend(),
                recurrent: [None, None],
                routers: Some(RouterTargets {
                    backends: &self.router_backends,
                    hidden: self.config.hidden,
                    experts: self.config.experts,
                }),
                floating: &self.floating_backends,
                chunk: self.options.chunk,
                decode_rows: &[1],
            },
            &self.layers,
        )
    }
    fn complete_input(&mut self, result: Result<()>) -> Result<()> {
        if result.is_err() {
            let (workspaces, targets, _) = self.workspace_targets();
            workspaces.abort(targets)?;
            self.reset()?;
        }
        result
    }
    fn output(&mut self) -> Result<()> {
        if !self.workspaces.is_decoding() {
            let (workspaces, targets, _) = self.workspace_targets();
            workspaces.finish_prefill(targets)?;
        }
        let _head = trace::Span::new(c"ling.output_head", -1, 1, self.position);
        let start = Instant::now();
        norm::rms_norm(
            &self.last_hidden,
            &self.final_norm,
            self.config.epsilon,
            false,
            &mut self.work.head,
        )?;
        self.logits.resize(self.config.vocab, 0.);
        self.head
            .run(&self.cpu, &self.work.head, &mut self.logits, 1)?;
        debug_assert!(
            self.logits.iter().all(|x| x.is_finite()),
            "nonfinite Ling logits"
        );
        self.trace.save(self.position, -1, "logits", &self.logits)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        self.head_ms += elapsed;
        Ok(())
    }
}
impl LanguageModel for Ling3Tiny {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn uses_user_boundaries(&self) -> bool {
        self.prefix_cache.settings.capacity > 0
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
        self.workspaces.reset();
        self.forward_ms = 0.;
        self.head_ms = 0.;
        self.last_hidden.clear();
        self.logits.clear();
        self.prefix_cache.clear();
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
        ensure!(
            prompt_tokens > 0 && prompt_tokens <= self.options.context,
            "invalid Ling prompt length"
        );
        self.prepare_rows(
            self.position..self.position + prompt_tokens,
            None,
            vec![prompt_tokens],
        )
    }
    fn prepare_prepared<'a>(
        &mut self,
        input: &'a crate::input::PreparedInput,
        start: usize,
    ) -> Result<crate::model::PreparedPrefill<'a>> {
        self.prepare_prompt(input, start)
    }
    fn forward_prepared(
        &mut self,
        prepared: crate::model::PreparedPrefill<'_>,
        cancel: &AtomicBool,
    ) -> Result<&[f32]> {
        self.consume_prepared(prepared, cancel)?;
        Ok(&self.logits)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.consume(tokens, None, false)
    }
    fn forward(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.consume(tokens, None, true)?;
        Ok(&self.logits)
    }
    fn forward_cancellable(&mut self, tokens: &[u32], cancel: &AtomicBool) -> Result<&[f32]> {
        self.consume(tokens, Some(cancel), true)?;
        Ok(&self.logits)
    }
    fn truncate(&mut self, position: usize) -> Result<bool> {
        ensure!(
            position <= self.position,
            "cannot extend Ling prefix by truncation"
        );
        if position == self.position {
            return Ok(true);
        }
        if position == 0 {
            self.reset()?;
            return Ok(true);
        }
        if self.prefix_cache.has(position) {
            self.restore_cached_prefix(position)?;
            return Ok(true);
        }
        Ok(false)
    }
    fn restore_prefix(&mut self, maximum: usize) -> Result<usize> {
        self.restore_cached_prefix(maximum)
    }
    fn stats(&self) -> serde_json::Value {
        let mut result = crate::model::metrics::backend_stats(&self.backends, self.head.backend());
        result["prefill_workspace_cached_domains"] =
            serde_json::json!(crate::model::workspace::cached_domains(&self.backends));
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
        let mla_scratch_bytes: u64 = mixers
            .iter()
            .map(|m| m["host_scratch_bytes"].as_u64().unwrap_or(0))
            .sum();
        let fields = serde_json::json!({"architecture":"bailing_hybrid","position":self.position,
            "model_load_ms":self.load_ms,"decoder_ms":self.forward_ms,"head_ms":self.head_ms,
            "mapped_weight_bytes":self.weights.store.mapped_bytes(),"mixers":mixers,
            "cpu_scratch_bytes":self.work.allocated_bytes() as u64+mla_scratch_bytes+(self.last_hidden.capacity()+self.logits.capacity()) as u64*4,
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
            .as_object_mut()
            .unwrap()
            .extend(self.prefix_cache.statistics().as_object().unwrap().clone());
        result
    }
}
