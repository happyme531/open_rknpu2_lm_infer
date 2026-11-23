use super::{
    config::Config,
    decoder::{Buffers, Layer, LayerSeqState},
    multimodal::EmbeddedChunk,
    timings::Timings,
};
use crate::model::trace::Trace;
use crate::ops::norm;
use crate::{
    model::{
        LanguageModel, ModelInfo, ModelOptions, PreparedPrefill,
        batch::SeqPool,
        checkpoint::{QuantLoader, Tensor, TensorStore},
        cpu::Cpu,
        dflash::DFlash,
        workspace::{PrefillPlan, Targets, Workspaces},
    },
    npu::{Attention, Backend},
};
use anyhow::{Result, ensure};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

pub(super) struct SeqState {
    pub mixers: crate::model::pipeline::Layers<LayerSeqState>,
    pub position: usize,
    pub next_rope_position: usize,
    pub reserved: usize,
}

impl SeqState {
    pub(super) fn reset(
        &mut self,
        reset_attention: impl Fn(&Attention) -> Result<()>,
    ) -> Result<()> {
        for mixer in self.mixers.iter_mut() {
            match mixer {
                LayerSeqState::Recurrent(state) => state.reset(),
                LayerSeqState::Attention(cache) => reset_attention(cache)?,
            }
        }
        self.position = 0;
        self.next_rope_position = 0;
        Ok(())
    }

    pub(super) fn advance(&mut self, rope: &[[usize; 3]]) {
        self.position += rope.len();
        self.next_rope_position = rope
            .iter()
            .flatten()
            .copied()
            .max()
            .unwrap_or(self.next_rope_position)
            + 1;
    }
}

pub(super) struct Weights {
    pub config: Config,
    pub cpu: Cpu,
    pub layers: crate::model::pipeline::Layers<Layer>,
    pub paging: Option<crate::model::experts::paged::Paging>,
    pub embedding: Tensor,
    pub final_norm: Vec<f32>,
    pub head: crate::model::output_head::OutputHead,
    pub trace: Trace,
}

pub struct Qwen35 {
    pub(super) options: ModelOptions,
    pub(super) info: ModelInfo,
    pub(super) loader: Arc<QuantLoader>,
    pub(super) backends: Vec<Backend>,
    pub(super) weights: Weights,
    pub(super) sequences: SeqPool<SeqState>,
    pub(super) work: Buffers,
    pub(super) secondary: Option<crate::npu::ExecutionLane>,
    pub(super) secondary_work: Option<Buffers>,
    pub(super) dual_groups: usize,
    pub(super) logits: Vec<f32>,
    pub(super) load_ms: f64,
    pub(super) forward_ms: f64,
    pub(super) head_ms: f64,
    pub(super) phases: Timings,
    pub(super) mtp_available: bool,
    pub(super) vision: Option<super::vision::Encoder>,
    pub(super) round: Option<super::transaction::Round>,
    pub(super) mtp: Option<super::mtp::Predictor>,
    pub(super) features: super::features::Features,
    pub(super) secondary_features: Option<super::features::Features>,
    pub(super) dflash: Option<DFlash>,
    pub(super) workspaces: Workspaces,
    pub(super) prefix_cache: super::cache::PrefixCache,
}

impl Qwen35 {
    fn prepare_rows(
        &mut self,
        span: std::ops::Range<usize>,
        id: Option<crate::model::prefill::PlanId>,
        rows: impl IntoIterator<Item = usize>,
        paired: bool,
    ) -> Result<()> {
        let prefill_rows = crate::model::workspace::physical_rows(rows, self.options.chunk);
        let decode_rows: Vec<_> = (1..=self.speculation_window() + 1).collect();
        let head_rows = crate::model::workspace::head_rows(
            &prefill_rows,
            &decode_rows,
            self.options.tune_search,
        );
        let plan = PrefillPlan {
            span,
            id,
            rows: &prefill_rows,
            head_rows: &head_rows,
            paired,
        };
        let draft_window = self.options.speculation.window();
        let (workspaces, targets, draft, store) = self.workspace_targets(&decode_rows);
        workspaces.prepare(targets, plan, |_, rows| {
            if let Some(draft) = draft {
                draft.prepare(rows, draft_window)?;
            }
            store.release_pages()
        })
    }

    pub fn load(options: ModelOptions) -> Result<Self> {
        ensure!(
            (1..=2).contains(&options.prefill_ubatches),
            "prefill-ubatches must be 1 or 2"
        );
        options.validate_speculation()?;
        let mut model = super::load::load(options, None)?;
        if matches!(
            model.options.speculation.kind,
            crate::model::SpeculativeType::DraftDflash | crate::model::SpeculativeType::DraftDspark
        ) && model.options.speculation.window() > 0
        {
            model.load_dflash()?;
        }
        if model.options.compress_w8a8 {
            let start = Instant::now();
            crate::npu::release_compression_scratch(
                model.backends.iter().chain(model.weights.head.backend()),
            );
            model.load_ms += start.elapsed().as_secs_f64() * 1000.;
        }
        Ok(model)
    }

    #[cfg(feature = "model-qwen35")]
    pub(in crate::model) fn load_batch(
        options: ModelOptions,
        config: crate::model::batch::BatchConfig,
    ) -> Result<Self> {
        let mut model = super::load::load(options, Some(config))?;
        crate::model::batch::prepare(config, &model.backends, model.weights.head.backend())?;
        if !model.weights.trace.enabled() {
            let buckets = config.decoder_buckets();
            for layer in model.weights.layers.iter() {
                if let super::decoder::FeedForward::Dense { gate_up, .. } = &layer.ffn {
                    gate_up.prepare_swiglu(&buckets)?;
                }
            }
        }
        model
            .work
            .recurrent
            .prepare(&[model.options.chunk.min(config.max_rows)])?;
        model.loader.store.release_pages()?;
        if model.options.compress_w8a8 {
            crate::npu::release_compression_scratch(
                model.backends.iter().chain(model.weights.head.backend()),
            );
        }
        crate::model::batch::report(&model)?;
        Ok(model)
    }

    pub(super) fn check_input(&self, tokens: &[u32]) -> Result<()> {
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= self.options.context - self.sequences.single().position,
            "empty Qwen3.5 input or context capacity exceeded"
        );
        ensure!(
            tokens.iter().all(|v| (*v as usize) < self.info.vocab_size),
            "token outside Qwen3.5 vocabulary"
        );
        Ok(())
    }

    pub(super) fn chunk(
        &mut self,
        tokens: &[u32],
        cancelled: Option<&AtomicBool>,
        prepared: Option<EmbeddedChunk<'_>>,
    ) -> Result<()> {
        self.chunk_mode(tokens, cancelled, prepared, false)
    }

    pub(super) fn chunk_mode(
        &mut self,
        tokens: &[u32],
        cancelled: Option<&AtomicBool>,
        prepared: Option<EmbeddedChunk<'_>>,
        verify: bool,
    ) -> Result<()> {
        self.execute(&[(tokens, prepared)], cancelled, verify)
    }

    fn forward_impl(
        &mut self,
        tokens: &[u32],
        cancelled: Option<&AtomicBool>,
        logits: bool,
    ) -> Result<()> {
        self.check_input(tokens)?;
        ensure!(
            self.round.is_none(),
            "Qwen3.5 verification transaction is active"
        );
        let position = self.sequences.single().position;
        self.workspaces
            .begin_input(position, tokens.len(), &self.backends);
        let result = (|| {
            self.reserve(position + tokens.len())?;
            let mut rows = 0;
            for (first, second) in crate::model::prefill::batches(
                tokens,
                self.prompt_chunk_rows(),
                self.options.prefill_ubatches,
            ) {
                let inputs = [(first, None), (second.unwrap_or_default(), None)];
                let count = if second.is_some() { 2 } else { 1 };
                crate::progress::prefill_shape(first.len() != self.options.chunk, || {
                    if self.weights.paging.is_some() {
                        self.prompt_chunk(first, cancelled, None)
                    } else {
                        self.execute(&inputs[..count], cancelled, false)
                    }
                })?;
                rows = inputs[count - 1].0.len();
            }
            rows = (rows - 1) % self.options.chunk + 1;
            ensure!(
                !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
                "generation cancelled"
            );
            if logits {
                self.output_head(rows)?;
            } else {
                self.workspaces
                    .finish_head_free(self.sequences.single().position);
            }
            Ok(())
        })();
        self.complete_input(result)
    }

    fn score(&mut self, tokens: &[u32], targets: &[u32]) -> Result<Vec<f64>> {
        ensure!(
            self.round.is_none(),
            "Qwen3.5 verification transaction is active"
        );
        self.check_input(tokens)?;
        ensure!(
            tokens.len() == targets.len()
                && targets.iter().all(|v| (*v as usize) < self.info.vocab_size),
            "invalid Qwen3.5 scoring targets"
        );
        let position = self.sequences.single().position;
        self.workspaces
            .begin_input(position, tokens.len(), &self.backends);
        self.reserve(position + tokens.len())?;
        let mut values = Vec::with_capacity(tokens.len());
        for (tokens, targets) in tokens
            .chunks(self.options.chunk)
            .zip(targets.chunks(self.options.chunk))
        {
            self.chunk(tokens, None, None)?;
            let start = Instant::now();
            let h = self.weights.config.hidden_size;
            let rows = tokens.len();
            norm::rms_norm(
                &self.work.hidden[..rows * h],
                &self.weights.final_norm,
                self.weights.config.epsilon,
                true,
                &mut self.work.normalized[..rows * h],
            )?;
            let vocab = self.info.vocab_size;
            self.logits.resize(rows * vocab, 0.0);
            self.weights.head.prepare(rows)?;
            self.weights.head.run(
                &self.weights.cpu,
                &self.work.normalized[..rows * h],
                &mut self.logits,
                rows,
            )?;
            for (logits, target) in self.logits.chunks(vocab).zip(targets) {
                values.push(crate::ppl::negative_log_likelihood(logits, *target)?);
            }
            self.head_ms += start.elapsed().as_secs_f64() * 1000.0;
        }
        Ok(values)
    }

    pub(super) fn release_prefill_workspaces(&mut self) -> Result<()> {
        if self.workspaces.is_decoding() {
            return Ok(());
        }
        let rows: Vec<_> = (1..=self.speculation_window() + 1).collect();
        let (workspaces, targets, ..) = self.workspace_targets(&rows);
        workspaces.finish_prefill(targets)
    }

    fn workspace_targets<'a>(
        &'a mut self,
        rows: &'a [usize],
    ) -> (
        &'a mut Workspaces,
        Targets<'a>,
        Option<&'a DFlash>,
        &'a TensorStore,
    ) {
        (
            &mut self.workspaces,
            Targets {
                primary: &self.backends,
                secondary: self.secondary.as_ref(),
                head: self.weights.head.backend(),
                recurrent: [
                    Some(&mut self.work.recurrent),
                    self.secondary_work.as_mut().map(|work| &mut work.recurrent),
                ],
                routers: None,
                floating: &[],
                chunk: self.options.chunk,
                decode_rows: rows,
            },
            self.dflash.as_ref(),
            &self.loader.store,
        )
    }

    pub(super) fn complete_input<T>(&mut self, result: Result<T>) -> Result<T> {
        if result.is_err() {
            let rows: Vec<_> = (1..=self.speculation_window() + 1).collect();
            let (workspaces, targets, ..) = self.workspace_targets(&rows);
            workspaces.abort(targets)?;
            // A failed or cancelled hybrid chunk may have changed earlier layers.
            self.reset()?;
        }
        result
    }
}

impl LanguageModel for Qwen35 {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn uses_user_boundaries(&self) -> bool {
        self.prefix_cache.settings.capacity > 0
    }
    fn position(&self) -> usize {
        self.sequences.single().position
    }
    fn reset(&mut self) -> Result<()> {
        if let Some(paging) = &self.weights.paging {
            paging.reset();
        }
        self.sequences.single_mut().reset(Attention::reset)?;
        self.workspaces.reset();
        self.round = None;
        self.features.reset();
        if let Some(features) = &mut self.secondary_features {
            features.reset();
        }
        if let Some(mtp) = &mut self.mtp {
            mtp.reset()?;
        }
        if let Some(draft) = &mut self.dflash {
            draft.reset()?;
        }
        self.prefix_cache.clear();
        self.forward_ms = 0.0;
        self.head_ms = 0.0;
        self.phases.reset();
        self.work.recurrent.reset_statistics();
        if let Some(work) = &mut self.secondary_work {
            work.recurrent.reset_statistics();
        }
        self.dual_groups = 0;
        Ok(())
    }
    fn reserve(&mut self, tokens: usize) -> Result<()> {
        ensure!(
            tokens > 0 && tokens <= self.options.context,
            "Qwen3.5 reservation exceeds context"
        );
        if tokens > self.sequences.single().reserved {
            for mixer in self.sequences.single().mixers.iter() {
                if let LayerSeqState::Attention(cache) = &*mixer {
                    cache.reserve(tokens)?;
                }
            }
            self.sequences.single_mut().reserved = tokens.div_ceil(32) * 32;
        }
        Ok(())
    }
    fn truncate(&mut self, position: usize) -> Result<bool> {
        ensure!(
            position <= self.sequences.single().position,
            "cannot extend Qwen3.5 state by truncation"
        );
        if position == self.sequences.single().position {
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
    fn prepare(&mut self, prompt_tokens: usize) -> Result<()> {
        ensure!(
            prompt_tokens > 0 && prompt_tokens <= self.options.context,
            "Qwen3.5 prompt exceeds context"
        );
        let paired = self.weights.paging.is_none()
            && crate::model::prefill::raw_uses_secondary(
                prompt_tokens,
                self.prompt_chunk_rows(),
                self.options.prefill_ubatches,
            );
        let position = self.sequences.single().position;
        self.prepare_rows(
            position..position + prompt_tokens,
            None,
            [prompt_tokens],
            paired,
        )
    }
    fn prepare_prepared<'a>(
        &mut self,
        input: &'a crate::input::PreparedInput,
        start: usize,
    ) -> Result<PreparedPrefill<'a>> {
        ensure!(
            start < input.tokens.len() && input.tokens.len() <= self.options.context,
            "invalid prepared prompt"
        );
        let steps = self.prefix_cache.plan(input, start, self.options.chunk);
        let logical_rows = self.prompt_chunk_rows();
        let steps = if logical_rows > self.options.chunk {
            crate::model::PrefillStep::coalesce(steps, logical_rows)
        } else {
            steps
        };
        let prepared = PreparedPrefill {
            input,
            start,
            steps,
            checkpoint: self.prefix_cache.last_position(),
            plan_id: Some(crate::model::prefill::PlanId::new()),
        };
        let paired = self.weights.paging.is_none()
            && crate::model::prefill::groups(
                &prepared.steps,
                self.options.chunk,
                self.options.prefill_ubatches,
            )
            .any(|group| group.len() == 2);
        self.prepare_rows(
            start..input.tokens.len(),
            prepared.plan_id,
            prepared.steps.iter().map(|s| s.range.len()),
            paired,
        )?;
        Ok(prepared)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.forward_impl(tokens, None, false)
    }
    fn forward(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.forward_impl(tokens, None, true)?;
        Ok(&self.logits)
    }
    fn forward_prepared(
        &mut self,
        prepared: PreparedPrefill<'_>,
        cancelled: &AtomicBool,
    ) -> Result<&[f32]> {
        self.consume_prepared(prepared, cancelled, None)?;
        Ok(&self.logits)
    }
    fn forward_prepared_with_progress(
        &mut self,
        prepared: PreparedPrefill<'_>,
        cancelled: &AtomicBool,
        progress: &mut dyn FnMut(crate::progress::ModelProgress),
    ) -> Result<&[f32]> {
        self.consume_prepared(prepared, cancelled, Some(progress))?;
        Ok(&self.logits)
    }
    fn forward_cancellable(&mut self, tokens: &[u32], cancelled: &AtomicBool) -> Result<&[f32]> {
        self.forward_impl(tokens, Some(cancelled), true)?;
        Ok(&self.logits)
    }
    fn token_nll(&mut self, tokens: &[u32], targets: &[u32]) -> Result<Vec<f64>> {
        let result = self.score(tokens, targets);
        if result.is_err() {
            self.reset()?;
        }
        result
    }
    fn stats(&self) -> serde_json::Value {
        self.statistics()
    }
    fn enable_mtp(&mut self, max_draft: usize) -> Result<()> {
        let start = Instant::now();
        ensure!(
            self.sequences.single().position == 0 && self.round.is_none(),
            "configure MTP before prefill"
        );
        if max_draft == 0 {
            self.mtp = None;
            return Ok(());
        }
        ensure!(
            !self.has_dflash(),
            "MTP and DFlash cannot be enabled together"
        );
        ensure!(
            (1..=7).contains(&max_draft) && max_draft < self.options.chunk,
            "MTP draft length must be 1..7 and smaller than ubatch"
        );
        if let Some(mtp) = &mut self.mtp {
            mtp.max_draft = max_draft;
            return Ok(());
        }
        let domain = 13u32
            .checked_sub(self.backends.len() as u32)
            .ok_or_else(|| anyhow::anyhow!("no free IOMMU domain for MTP"))?;
        self.mtp = Some(super::mtp::Predictor::load(
            &self.options,
            &self.weights.config,
            &self.loader.store,
            domain,
            max_draft,
        )?);
        self.load_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
    fn speculation_window(&self) -> usize {
        if self.has_dflash() {
            self.options.speculation.window()
        } else {
            self.mtp.as_ref().map_or(0, |m| m.max_draft)
        }
    }
    fn speculation_p_min(&self) -> f64 {
        self.options.speculation.p_min
    }
    fn speculation_sampling(&self) -> crate::sampling::DraftSampling {
        self.options.speculation.sampling
    }
    fn draft_logits(&mut self, token: u32, cancel: &AtomicBool) -> Result<&[f32]> {
        ensure!(!cancel.load(Ordering::Relaxed), "generation cancelled");
        ensure!(
            self.round.is_none() && (token as usize) < self.info.vocab_size,
            "invalid MTP draft token or active target verification"
        );
        let mtp = self
            .mtp
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("MTP is disabled"))?;
        let input = mtp.draft(
            &self.weights.config,
            &self.weights.cpu,
            &self.weights.embedding,
            token,
            &self.weights.trace,
        )?;
        let head_start = Instant::now();
        self.logits.resize(self.info.vocab_size, 0.);
        self.weights
            .head
            .run(&self.weights.cpu, &input, &mut self.logits, 1)?;
        self.head_ms += head_start.elapsed().as_secs_f64() * 1000.;
        Ok(&self.logits)
    }
    fn draft_block_logits(
        &mut self,
        anchor: u32,
        drafts: usize,
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<f32>>> {
        self.dflash_logits(anchor, drafts, cancel)
    }
    fn adjust_draft_logits(
        &self,
        row: usize,
        previous: u32,
        logits: &mut [f32],
        cutoff: f64,
    ) -> Result<Option<f64>> {
        match &self.dflash {
            Some(draft) => self.phases.trace("draft Markov", || {
                draft.adjust(row, previous, logits, cutoff)
            }),
            None => Ok(None),
        }
    }
    fn verify_tokens(&mut self, tokens: &[u32], cancel: &AtomicBool) -> Result<&[f32]> {
        Qwen35::verify_tokens(self, tokens, cancel)
    }
    fn finish_verification(&mut self, consumed: usize) -> Result<()> {
        Qwen35::finish_verification(self, consumed)
    }
    fn abort_verification(&mut self) -> Result<()> {
        Qwen35::abort_verification(self)
    }
}
