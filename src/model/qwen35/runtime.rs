use super::{
    config::Config,
    decoder::{Buffers, Execution, Layer, Mixer},
    multimodal::EmbeddedChunk,
    timings::{Phase, Timings},
};
use crate::model::trace::Trace;
use crate::ops::norm;
use crate::{
    model::{
        LanguageModel, ModelInfo, ModelOptions, PreparedPrefill,
        checkpoint::{QuantLoader, Tensor},
        cpu::Cpu,
    },
    npu::Backend,
};
use anyhow::{Result, ensure};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

pub struct Qwen35 {
    pub(super) options: ModelOptions,
    pub(super) info: ModelInfo,
    pub(super) config: Config,
    pub(super) loader: Arc<QuantLoader>,
    pub(super) cpu: Cpu,
    pub(super) backends: Vec<Backend>,
    pub(super) layers: Vec<Layer>,
    pub(super) paging: Option<std::rc::Rc<crate::model::experts::paged::Paging>>,
    pub(super) embedding: Tensor,
    pub(super) final_norm: Vec<f32>,
    pub(super) head: crate::model::output_head::OutputHead,
    pub(super) work: Buffers,
    pub(super) position: usize,
    pub(super) reserved: usize,
    pub(super) logits: Vec<f32>,
    pub(super) trace: Trace,
    pub(super) load_ms: f64,
    pub(super) forward_ms: f64,
    pub(super) head_ms: f64,
    pub(super) phases: Timings,
    pub(super) mtp_available: bool,
    pub(super) vision: Option<super::vision::Encoder>,
    pub(super) next_rope_position: usize,
    pub(super) round: Option<super::transaction::Round>,
    pub(super) mtp: Option<super::mtp::Predictor>,
    pub(super) features: super::features::Features,
    pub(super) dflash: Option<crate::model::dflash::DFlash>,
    pub(super) workspaces: crate::model::workspace::Workspaces,
    pub(super) prefix_cache: super::cache::PrefixCache,
}

impl Qwen35 {
    fn prepare_rows(&mut self, rows: Vec<usize>) -> Result<()> {
        let prefill_rows = crate::model::workspace::physical_rows(rows, self.options.chunk);
        let mut rows = prefill_rows.clone();
        let head_rows: Vec<_> = (1..=self.speculation_window() + 1).collect();
        self.workspaces.prepare(&mut rows, &head_rows);
        for backend in &self.backends {
            backend.retain_workspaces(&rows)?;
            for rows in &rows {
                backend.prepare(*rows)?;
            }
        }
        // Generation projects only the prompt's final row; full-row scoring
        // prepares its own head workspace when token_nll is requested.
        if let Some(backend) = self.head.backend() {
            backend.retain_workspaces(&head_rows)?;
            for rows in head_rows {
                backend.prepare(rows)?;
            }
        }
        self.work.recurrent.prepare(&prefill_rows)?;
        if let Some(draft) = &self.dflash {
            draft.prepare(&rows, self.options.speculation.window())?;
        }
        self.loader.store.release_pages()?;
        Ok(())
    }
    pub fn load(options: ModelOptions) -> Result<Self> {
        options.validate_speculation()?;
        let mut model = super::load::load(options)?;
        if matches!(
            model.options.speculation.kind,
            crate::model::SpeculativeType::DraftDflash | crate::model::SpeculativeType::DraftDspark
        ) && model.options.speculation.window() > 0
        {
            model.load_dflash()?;
        }
        Ok(model)
    }

    pub(super) fn check_input(&self, tokens: &[u32]) -> Result<()> {
        ensure!(
            !tokens.is_empty() && tokens.len() <= self.options.context - self.position,
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
        let start = Instant::now();
        if !verify && tokens.len() > 1 {
            self.workspaces.reset();
        }
        let h = self.config.hidden_size;
        self.phases.context(self.position, tokens.len(), -1, verify);
        self.phases.stage(Phase::Cpu, "target embedding", || {
            for (row, token) in tokens.iter().enumerate() {
                self.embedding.row(
                    *token as usize,
                    &mut self.work.hidden[row * h..(row + 1) * h],
                );
            }
        });
        let positions: Vec<_>;
        let rope_positions = if let Some(EmbeddedChunk {
            positions: p,
            features,
        }) = prepared
        {
            ensure!(p.len() == tokens.len(), "prepared position shape mismatch");
            for (start, embedding) in features {
                let end = start + embedding.len() / h;
                let first = self.position.max(*start);
                let last = (self.position + tokens.len()).min(end);
                if first < last {
                    self.work.hidden[(first - self.position) * h..(last - self.position) * h]
                        .copy_from_slice(&embedding[(first - start) * h..(last - start) * h]);
                }
            }
            p
        } else {
            positions = (self.next_rope_position..self.next_rope_position + tokens.len())
                .map(|p| [p; 3])
                .collect();
            &positions
        };
        let embeddings = self
            .mtp
            .as_ref()
            .filter(|_| !verify)
            .map(|_| self.work.hidden[..tokens.len() * h].to_vec());
        self.features.begin(tokens.len());
        for (i, layer) in self.layers.iter_mut().enumerate() {
            ensure!(
                !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
                "generation cancelled"
            );
            layer.forward(
                &mut self.work,
                &Execution {
                    config: &self.config,
                    cpu: &self.cpu,
                    rows: tokens.len(),
                    position: self.position,
                    index: i as i32,
                    trace: &self.trace,
                    rope_positions,
                    verify,
                },
                &mut self.phases,
            )?;
            self.phases.trace("target feature record", || {
                self.features
                    .record(i, &self.work.hidden[..tokens.len() * h]);
            });
        }
        if let (Some(mtp), Some(embeddings)) = (&mut self.mtp, embeddings) {
            norm::rms_norm(
                &self.work.hidden[..tokens.len() * h],
                &self.final_norm,
                self.config.epsilon,
                true,
                &mut self.work.normalized[..tokens.len() * h],
            )?;
            mtp.catch_up(
                &self.config,
                &self.cpu,
                &embeddings,
                &self.work.normalized[..tokens.len() * h],
                rope_positions,
                &self.trace,
            )?;
        }
        if !verify {
            self.features.commit(tokens.len());
            self.flush_features(cancelled)?;
            self.position += tokens.len();
            self.next_rope_position = rope_positions
                .iter()
                .flatten()
                .copied()
                .max()
                .unwrap_or(self.next_rope_position)
                + 1;
        }
        self.forward_ms += start.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    fn forward_impl(
        &mut self,
        tokens: &[u32],
        cancelled: Option<&AtomicBool>,
        logits: bool,
    ) -> Result<()> {
        if tokens.len() > 1 {
            self.workspaces.reset();
        }
        self.check_input(tokens)?;
        ensure!(
            self.round.is_none(),
            "Qwen3.5 verification transaction is active"
        );
        self.reserve(self.position + tokens.len())?;
        let result = (|| {
            let mut rows = 0;
            for chunk in tokens.chunks(self.prompt_chunk_rows()) {
                self.prompt_chunk(chunk, cancelled, None)?;
                rows = chunk.len();
            }
            rows = (rows - 1) % self.options.chunk + 1;
            ensure!(
                !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
                "generation cancelled"
            );
            if logits {
                self.output_head(rows)?;
            }
            Ok(())
        })();
        // A failed/cancelled hybrid chunk may have modified earlier layers.
        if result.is_err() {
            self.reset()?;
        }
        result
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
        self.reserve(self.position + tokens.len())?;
        let mut values = Vec::with_capacity(tokens.len());
        for (tokens, targets) in tokens
            .chunks(self.options.chunk)
            .zip(targets.chunks(self.options.chunk))
        {
            self.chunk(tokens, None, None)?;
            let start = Instant::now();
            let h = self.config.hidden_size;
            let rows = tokens.len();
            norm::rms_norm(
                &self.work.hidden[..rows * h],
                &self.final_norm,
                self.config.epsilon,
                true,
                &mut self.work.normalized[..rows * h],
            )?;
            let vocab = self.info.vocab_size;
            self.logits.resize(rows * vocab, 0.0);
            self.head.prepare(rows)?;
            self.head.run(
                &self.cpu,
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
        if self
            .workspaces
            .enter_decode(self.backends.iter().chain(self.head.backend()), &rows)?
        {
            // GDN verification and one-row AR execute on CPU.
            self.work.recurrent.retain_workspaces(&[1]);
        }
        Ok(())
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
        self.position
    }
    fn reset(&mut self) -> Result<()> {
        if let Some(paging) = &self.paging {
            paging.reset();
        }
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Mixer::Recurrent { state, .. } => state.reset(),
                Mixer::Attention { cache, .. } => cache.reset()?,
            }
        }
        self.position = 0;
        self.workspaces.reset();
        self.round = None;
        self.features.reset();
        if let Some(mtp) = &mut self.mtp {
            mtp.reset()?;
        }
        if let Some(draft) = &mut self.dflash {
            draft.reset()?;
        }
        self.next_rope_position = 0;
        self.prefix_cache.clear();
        self.forward_ms = 0.0;
        self.head_ms = 0.0;
        self.phases.reset();
        self.work.recurrent.reset_statistics();
        Ok(())
    }
    fn reserve(&mut self, tokens: usize) -> Result<()> {
        ensure!(
            tokens > 0 && tokens <= self.options.context,
            "Qwen3.5 reservation exceeds context"
        );
        if tokens > self.reserved {
            for layer in &self.layers {
                if let Mixer::Attention { cache, .. } = &layer.mixer {
                    cache.reserve(tokens)?;
                }
            }
            self.reserved = tokens.div_ceil(32) * 32;
        }
        Ok(())
    }
    fn truncate(&mut self, position: usize) -> Result<bool> {
        ensure!(
            position <= self.position,
            "cannot extend Qwen3.5 state by truncation"
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
    fn prepare(&mut self, prompt_tokens: usize) -> Result<()> {
        ensure!(
            prompt_tokens > 0 && prompt_tokens <= self.options.context,
            "Qwen3.5 prompt exceeds context"
        );
        self.prepare_rows(vec![prompt_tokens])
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
        };
        let rows = prepared.steps.iter().map(|s| s.range.len()).collect();
        self.prepare_rows(rows)?;
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
        self.consume_prepared(prepared, cancelled)?;
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
            self.position == 0 && self.round.is_none(),
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
            &self.config,
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
        let input = mtp.draft(&self.config, &self.cpu, &self.embedding, token, &self.trace)?;
        let head_start = Instant::now();
        self.logits.resize(self.info.vocab_size, 0.);
        self.head.run(&self.cpu, &input, &mut self.logits, 1)?;
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
