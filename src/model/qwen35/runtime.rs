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
        LanguageModel, ModelInfo, ModelOptions,
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
}

impl Qwen35 {
    pub fn load(options: ModelOptions) -> Result<Self> {
        options.validate_speculation()?;
        let mut model = super::load::load(options)?;
        if model.options.speculation.kind == crate::model::SpeculativeType::DraftDflash
            && model.options.speculation.window() > 0
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
        let h = self.config.hidden_size;
        self.phases.measure(Phase::Cpu, || {
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
            self.features
                .record(i, &self.work.hidden[..tokens.len() * h]);
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
        self.check_input(tokens)?;
        ensure!(
            self.round.is_none(),
            "Qwen3.5 verification transaction is active"
        );
        self.reserve(self.position + tokens.len())?;
        let result = (|| {
            let mut rows = 0;
            for chunk in tokens.chunks(self.options.chunk) {
                self.chunk(chunk, cancelled, None)?;
                rows = chunk.len();
            }
            ensure!(
                !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
                "generation cancelled"
            );
            if logits {
                self.output_head(rows)?;
            }
            Ok(())
        })();
        // A failed/cancelled hybrid chunk may have modified earlier layers. No
        // arbitrary GDN rollback is exposed until prefix checkpoints exist.
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
}

impl LanguageModel for Qwen35 {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn position(&self) -> usize {
        self.position
    }
    fn reset(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Mixer::Recurrent { state, .. } => state.reset(),
                Mixer::Attention { cache, .. } => cache.reset()?,
            }
        }
        self.position = 0;
        self.round = None;
        self.features.reset();
        if let Some(mtp) = &mut self.mtp {
            mtp.reset()?;
        }
        if let Some(draft) = &mut self.dflash {
            draft.reset()?;
        }
        self.next_rope_position = 0;
        self.forward_ms = 0.0;
        self.head_ms = 0.0;
        self.phases = Timings::default();
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
        Ok(false)
    }
    fn prepare(&mut self, prompt_tokens: usize) -> Result<()> {
        ensure!(
            prompt_tokens > 0 && prompt_tokens <= self.options.context,
            "Qwen3.5 prompt exceeds context"
        );
        let mut rows = vec![1, self.options.chunk, prompt_tokens.min(self.options.chunk)];
        if !prompt_tokens.is_multiple_of(self.options.chunk) {
            rows.push(prompt_tokens % self.options.chunk);
        }
        rows.sort_unstable();
        rows.dedup();
        if self.speculation_window() > 0 {
            rows.extend(1..=self.speculation_window() + 1);
            rows.sort_unstable();
            rows.dedup();
        }
        for backend in self.backends.iter().chain(self.head.backend()) {
            backend.retain_workspaces(&rows)?;
            for rows in &rows {
                backend.prepare(*rows)?;
            }
        }
        self.work.recurrent.prepare(&rows)?;
        if let Some(draft) = &self.dflash {
            draft.prepare(&rows, self.options.speculation.window())?;
        }
        self.loader.store.release_pages()?;
        Ok(())
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
        input: &crate::input::PreparedInput,
        start: usize,
        cancelled: &AtomicBool,
    ) -> Result<&[f32]> {
        self.consume_prepared(input, start, cancelled)?;
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
            self.dflash.is_none(),
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
        if self.dflash.is_some() {
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
