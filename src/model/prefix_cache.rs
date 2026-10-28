//! Resident SSM prefix checkpoints shared by recurrent model families.
//! Attention KV stays resident; callers restore the SSM payload and truncate KV.
use crate::{
    input::PreparedInput,
    model::{CheckpointOptions, PrefillStep},
};
mod policy;

pub(crate) struct Entry<T> {
    pub position: usize,
    pub state: T,
    pub task: usize,
    pub bytes: usize,
}

pub(crate) struct PrefixCache<T> {
    pub settings: CheckpointOptions,
    pub task: usize,
    pub entries: Vec<Entry<T>>,
    pub saves: usize,
    pub hits: usize,
    pub save_ms: f64,
}

impl<T> PrefixCache<T> {
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
    pub fn reuse(&mut self, position: usize) -> bool {
        let Some(entry) = self.entries.iter_mut().find(|c| c.position == position) else {
            return false;
        };
        // Restore removes snapshots beyond the retained prefix. An existing
        // snapshot at this position therefore belongs to the live branch.
        entry.task = self.task;
        true
    }
    pub fn make_room(&mut self) {
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

    pub fn insert(&mut self, position: usize, state: T, bytes: usize, elapsed_ms: f64) {
        self.entries.push(Entry {
            position,
            state,
            task: self.task,
            bytes,
        });
        self.saves += 1;
        self.save_ms += elapsed_ms;
    }
    pub fn statistics(&self) -> serde_json::Value {
        serde_json::json!({
            "prefix_truncate": "resident recurrent checkpoints",
            "prefix_checkpoint_limit": self.settings.capacity,
            "checkpoint_min_step": self.settings.min_step,
            "prefix_checkpoint_positions": self.positions(),
            "prefix_checkpoint_count": self.count(),
            "prefix_checkpoint_bytes": self.bytes(),
            "prefix_checkpoint_saves": self.saves,
            "prefix_checkpoint_hits": self.hits,
            "prefix_checkpoint_save_ms": self.save_ms,
        })
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

#[cfg(test)]
mod tests;
