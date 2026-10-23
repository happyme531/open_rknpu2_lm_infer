//! Atomic hybrid target verification. GDN retains compact inputs; attention
//! appends tentative KV and discards the unaccepted suffix at commit/abort.
use super::{decoder::Mixer, runtime::Qwen35};
use crate::{model::LanguageModel, ops::norm};
use anyhow::{Context, Result, ensure};
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

pub(super) struct Round {
    pub tokens: Vec<u32>,
    pub hidden: Vec<f32>,
}

#[cfg(test)]
#[path = "transaction_tests.rs"]
mod tests;

impl Qwen35 {
    /// Return one next-token logit row per real input, without committing state.
    /// Rows are [pending anchor, proposals...], with no NPU padding in this ABI.
    pub fn verify_tokens(&mut self, tokens: &[u32], cancel: &AtomicBool) -> Result<&[f32]> {
        ensure!(
            self.round.is_none(),
            "Qwen3.5 verification transaction is active"
        );
        self.check_input(tokens)?;
        ensure!(
            tokens.len() <= 8 && tokens.len() <= self.options.chunk,
            "verification requires at most min(8, ubatch) real rows"
        );
        self.reserve(self.position + tokens.len())?;
        self.round = Some(Round {
            tokens: tokens.to_vec(),
            hidden: Vec::new(),
        });
        let result = (|| -> Result<()> {
            self.chunk_mode(tokens, Some(cancel), None, true)?;
            ensure!(!cancel.load(Ordering::Relaxed), "generation cancelled");
            let head_start = Instant::now();
            let rows = tokens.len();
            let h = self.config.hidden_size;
            norm::rms_norm(
                &self.work.hidden[..rows * h],
                &self.final_norm,
                self.config.epsilon,
                true,
                &mut self.work.normalized[..rows * h],
            )?;
            self.round
                .as_mut()
                .unwrap()
                .hidden
                .extend_from_slice(&self.work.normalized[..rows * h]);
            self.logits.resize(rows * self.info.vocab_size, 0.);
            self.head.run(
                &self.cpu,
                &self.work.normalized[..rows * h],
                &mut self.logits,
                rows,
            )?;
            self.head_ms += head_start.elapsed().as_secs_f64() * 1000.;
            debug_assert!(
                self.logits.iter().all(|v| v.is_finite()),
                "nonfinite verification logits"
            );
            Ok(())
        })();
        if let Err(error) = result {
            self.abort_verification()?;
            return Err(error);
        }
        Ok(&self.logits)
    }

    /// Commit anchor plus the accepted proposal prefix. On device failure all
    /// model state is invalidated, so a partially committed cache cannot escape.
    pub fn finish_verification(&mut self, consumed: usize) -> Result<()> {
        let round = self
            .round
            .as_ref()
            .context("no active target verification")?;
        ensure!(
            (1..=round.tokens.len()).contains(&consumed),
            "invalid target commit boundary"
        );
        let result = (|| -> Result<()> {
            for layer in &mut self.layers {
                match &mut layer.mixer {
                    Mixer::Recurrent { state, .. } => {
                        state.finish_round(&self.cpu.pool, consumed)?
                    }
                    Mixer::Attention { cache, .. } => cache.truncate(self.position + consumed)?,
                }
            }
            if let Some(mtp) = &mut self.mtp {
                let round = self.round.as_ref().unwrap();
                let h = self.config.hidden_size;
                let mut embeddings = vec![0.; consumed * h];
                for (r, &token) in round.tokens[..consumed].iter().enumerate() {
                    self.embedding
                        .row(token as usize, &mut embeddings[r * h..(r + 1) * h]);
                }
                let rope: Vec<_> = (self.next_rope_position..self.next_rope_position + consumed)
                    .map(|p| [p; 3])
                    .collect();
                mtp.catch_up(
                    &self.config,
                    &self.cpu,
                    &embeddings,
                    &round.hidden[..consumed * h],
                    &rope,
                    &self.trace,
                )?;
            }
            self.position += consumed;
            self.next_rope_position += consumed;
            self.features.commit(consumed);
            self.flush_features(None)?;
            let vocab = self.info.vocab_size;
            self.logits
                .copy_within((consumed - 1) * vocab..consumed * vocab, 0);
            self.logits.truncate(vocab);
            self.round = None;
            Ok(())
        })();
        if result.is_err() {
            self.reset()?;
        }
        result
    }

    pub fn abort_verification(&mut self) -> Result<()> {
        let verified = self.round.take().is_some();
        self.features.abort();
        let result = (|| -> Result<()> {
            if let Some(mtp) = &mut self.mtp {
                mtp.abort()?;
            }
            if let Some(draft) = &self.dflash {
                draft.abort()?;
            }
            if !verified {
                return Ok(());
            }
            for layer in &mut self.layers {
                match &mut layer.mixer {
                    Mixer::Recurrent { state, .. } => state.abort_round(),
                    Mixer::Attention { cache, .. } => cache.truncate(self.position)?,
                }
            }
            // The last committed logits may have been overwritten by verify.
            self.logits.clear();
            Ok(())
        })();
        if result.is_err() {
            self.reset()?;
        }
        result
    }
}
