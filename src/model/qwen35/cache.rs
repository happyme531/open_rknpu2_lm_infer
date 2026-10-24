//! Resident prefix checkpoints, following llama.cpp's partial-state cache.
//! Ordinary attention KV stays in place and is truncated on restore.
use super::{decoder::Mixer, mtp, runtime::Qwen35};
use crate::{
    input::PreparedInput,
    layers::gated_delta::Snapshot,
    model::{CheckpointOptions, LanguageModel, PrefillStep},
};
use anyhow::{Result, ensure};
use std::time::Instant;
mod policy;

struct Checkpoint {
    position: usize,
    next_rope: usize,
    recurrent: Vec<Snapshot>,
    mtp: Option<mtp::Snapshot>,
    task: usize,
    bytes: usize,
}

pub(super) struct PrefixCache {
    pub settings: CheckpointOptions,
    task: usize,
    entries: Vec<Checkpoint>,
    pub saves: usize,
    pub hits: usize,
    pub save_ms: f64,
}

impl PrefixCache {
    pub fn new(settings: CheckpointOptions) -> Self {
        Self {
            settings,
            task: 0,
            entries: Vec::new(),
            saves: 0,
            hits: 0,
            save_ms: 0.,
        }
    }
    pub fn begin_request(&mut self) {
        self.task += 1;
    }
    pub fn plan(&self, input: &PreparedInput, start: usize, ubatch: usize) -> Vec<PrefillStep> {
        policy::plan(input, start, ubatch, self.settings, self.last_position())
    }
    pub fn last_position(&self) -> Option<usize> {
        self.entries.last().map(|c| c.position)
    }
    pub fn positions(&self) -> Vec<usize> {
        self.entries.iter().map(|c| c.position).collect()
    }
    fn reuse(&mut self, position: usize) -> bool {
        let Some(entry) = self.entries.iter_mut().find(|c| c.position == position) else {
            return false;
        };
        // Restore removes snapshots beyond the retained prefix. An existing
        // snapshot at this position therefore belongs to the live branch.
        entry.task = self.task;
        true
    }
    fn make_room(&mut self) {
        let mut previous = None;
        let mut i = 0;
        while self.entries.len() + 1 >= self.settings.capacity && i < self.entries.len() {
            let entry = &self.entries[i];
            if entry.task != self.task
                && previous.is_some_and(|p| entry.position <= p + self.settings.min_step)
            {
                self.entries.remove(i);
            } else {
                previous = Some(entry.position);
                i += 1;
            }
        }
        while self.entries.len() >= self.settings.capacity {
            self.entries.remove(0);
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
    pub fn has(&self, position: usize) -> bool {
        self.entries.iter().any(|c| c.position == position)
    }
    pub fn bytes(&self) -> usize {
        self.entries.iter().map(|c| c.bytes).sum()
    }
    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

impl Qwen35 {
    pub(super) fn save_prefix_checkpoint(&mut self) -> Result<()> {
        ensure!(
            self.round.is_none(),
            "cannot checkpoint tentative target state"
        );
        if self.position == 0 || self.prefix_cache.settings.capacity == 0 {
            return Ok(());
        }
        if self.prefix_cache.reuse(self.position) {
            return Ok(());
        }
        let start = Instant::now();
        self.prefix_cache.make_room();
        let mtp = self.mtp.as_ref().map(|m| m.snapshot());
        let recurrent = self
            .layers
            .iter()
            .filter_map(|l| match &l.mixer {
                Mixer::Recurrent { state, .. } => Some(state.snapshot()),
                _ => None,
            })
            .collect::<Result<Vec<_>>>()?;
        let bytes = recurrent.iter().map(Snapshot::bytes).sum::<usize>()
            + mtp.as_ref().map_or(0, mtp::Snapshot::bytes);
        self.prefix_cache.entries.push(Checkpoint {
            position: self.position,
            next_rope: self.next_rope_position,
            recurrent,
            mtp,
            task: self.prefix_cache.task,
            bytes,
        });
        self.prefix_cache.saves += 1;
        self.prefix_cache.save_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }

    pub(super) fn restore_cached_prefix(&mut self, maximum: usize) -> Result<usize> {
        ensure!(
            self.round.is_none() && maximum <= self.position,
            "invalid target prefix restore"
        );
        if maximum == self.position {
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
            let mut snapshots = checkpoint.recurrent.iter();
            for layer in &mut self.layers {
                match &mut layer.mixer {
                    Mixer::Recurrent { state, .. } => state.restore(snapshots.next().unwrap())?,
                    Mixer::Attention { cache, .. } => cache.truncate(position)?,
                }
            }
            if let (Some(mtp), Some(snapshot)) = (&mut self.mtp, &checkpoint.mtp) {
                mtp.restore(snapshot)?;
            }
            #[cfg(feature = "draft-dflash")]
            if let Some(draft) = &mut self.dflash {
                draft.restore_prefix(position)?;
            }
            self.features.reset();
            self.position = position;
            self.next_rope_position = checkpoint.next_rope;
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
