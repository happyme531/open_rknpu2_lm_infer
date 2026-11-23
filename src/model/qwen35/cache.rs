//! Resident prefix checkpoints, following llama.cpp's partial-state cache.
//! Ordinary attention KV stays in place and is truncated on restore.
use super::{decoder::LayerSeqState, mtp, runtime::Qwen35};
use crate::{layers::gated_delta::Snapshot, model::LanguageModel};
use anyhow::{Result, ensure};
use std::time::Instant;

pub(super) struct Checkpoint {
    next_rope: usize,
    recurrent: Vec<Snapshot>,
    mtp: Option<mtp::Snapshot>,
}

pub(super) type PrefixCache = crate::model::prefix_cache::PrefixCache<Checkpoint>;

impl Qwen35 {
    pub(super) fn save_prefix_checkpoint(&mut self) -> Result<()> {
        ensure!(
            self.round.is_none(),
            "cannot checkpoint tentative target state"
        );
        if self.sequences.single().position == 0 || self.prefix_cache.settings.capacity == 0 {
            return Ok(());
        }
        if self.prefix_cache.reuse(self.sequences.single().position) {
            return Ok(());
        }
        let start = Instant::now();
        self.prefix_cache.make_room();
        let mtp = self.mtp.as_ref().map(|m| m.snapshot());
        let recurrent = self
            .sequences
            .single()
            .mixers
            .iter()
            .filter_map(|l| match &*l {
                LayerSeqState::Recurrent(state) => Some(state.snapshot()),
                _ => None,
            })
            .collect::<Result<Vec<_>>>()?;
        let bytes = recurrent.iter().map(Snapshot::bytes).sum::<usize>()
            + mtp.as_ref().map_or(0, mtp::Snapshot::bytes);
        self.prefix_cache.insert(
            self.sequences.single().position,
            Checkpoint {
                next_rope: self.sequences.single().next_rope_position,
                recurrent,
                mtp,
            },
            bytes,
            start.elapsed().as_secs_f64() * 1000.,
        );
        Ok(())
    }

    pub(super) fn restore_cached_prefix(&mut self, maximum: usize) -> Result<usize> {
        ensure!(
            self.round.is_none() && maximum <= self.sequences.single().position,
            "invalid target prefix restore"
        );
        if maximum == self.sequences.single().position {
            return Ok(maximum);
        }
        let Some(checkpoint) = self
            .prefix_cache
            .entries
            .iter()
            .rev()
            .find(|c| c.position <= maximum)
        else {
            return Ok(0);
        };
        let position = checkpoint.position;
        let result = (|| -> Result<()> {
            let mut snapshots = checkpoint.state.recurrent.iter();
            for mixer in self.sequences.single_mut().mixers.iter_mut() {
                match mixer {
                    LayerSeqState::Recurrent(state) => state.restore(snapshots.next().unwrap())?,
                    LayerSeqState::Attention(cache) => cache.truncate(position)?,
                }
            }
            if let (Some(mtp), Some(snapshot)) = (&mut self.mtp, &checkpoint.state.mtp) {
                mtp.restore(snapshot)?;
            }
            if let Some(draft) = &mut self.dflash {
                draft.restore_prefix(position)?;
            }
            self.features.reset();
            self.sequences.single_mut().position = position;
            self.sequences.single_mut().next_rope_position = checkpoint.state.next_rope;
            self.logits.clear();
            Ok(())
        })();
        if let Err(error) = result {
            self.reset()?;
            return Err(error);
        }
        // Future KV rows may be overwritten by the new branch.
        self.prefix_cache.entries.retain(|c| c.position <= position);
        self.prefix_cache.hits += 1;
        Ok(position)
    }
}

#[cfg(test)]
mod tests;
