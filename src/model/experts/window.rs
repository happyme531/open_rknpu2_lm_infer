//! Per-layer LRU window adapted from OpenBMB/llama.cpp-omni's
//! `src/llama-moe-slot.cpp` (a64932ec615e1463158129feaf2f0752d0c07a2e).
//! Requested experts are pinned; eviction first avoids the prediction set.
//! The main thread publishes a slot only after its NPU import succeeds.
//! See Omni-LICENSE for the upstream MIT notice.
use anyhow::{Result, ensure};
use std::collections::HashSet;

pub(super) struct Window {
    contents: Vec<Option<usize>>,
    age: Vec<u64>,
    clock: u64,
}
impl Window {
    pub fn new(slots: usize) -> Self {
        Self {
            contents: vec![None; slots],
            age: vec![0; slots],
            clock: 0,
        }
    }
    pub fn contents(&self) -> &[Option<usize>] {
        &self.contents
    }
    pub fn slot(&self, expert: usize) -> Option<usize> {
        self.contents.iter().position(|&id| id == Some(expert))
    }
    pub fn touch(&mut self, expert: usize) {
        if let Some(slot) = self.slot(expert) {
            self.clock += 1;
            self.age[slot] = self.clock;
        }
    }
    pub fn publish(&mut self, slot: usize, expert: usize) {
        self.contents[slot] = Some(expert);
        self.touch(expert);
    }
    pub fn invalidate(&mut self, slot: usize) {
        self.contents[slot] = None;
        self.age[slot] = 0;
    }
    pub fn victim(
        &self,
        requested: &HashSet<usize>,
        predicted: &HashSet<usize>,
        reserved: &HashSet<usize>,
    ) -> Result<usize> {
        if let Some(slot) = self
            .contents
            .iter()
            .enumerate()
            .position(|(slot, id)| id.is_none() && !reserved.contains(&slot))
        {
            return Ok(slot);
        }
        for protect_prediction in [true, false] {
            if let Some(slot) = self
                .contents
                .iter()
                .enumerate()
                .filter(|(slot, id)| {
                    !reserved.contains(slot)
                        && id.is_some_and(|id| {
                            !requested.contains(&id)
                                && (!protect_prediction || !predicted.contains(&id))
                        })
                })
                .min_by_key(|&(slot, _)| self.age[slot])
                .map(|(slot, _)| slot)
            {
                return Ok(slot);
            }
        }
        anyhow::bail!("expert window cannot hold this token's requested experts")
    }
    pub fn plan(
        &self,
        requested: &[usize],
        predicted: &HashSet<usize>,
    ) -> Result<Vec<(usize, usize)>> {
        let requested_set: HashSet<_> = requested.iter().copied().collect();
        ensure!(
            requested_set.len() <= self.contents.len(),
            "expert window must hold at least one token's top-k"
        );
        let mut reserved = HashSet::new();
        let mut missing = Vec::new();
        for &expert in requested {
            if self.slot(expert).is_none() {
                let slot = self.victim(&requested_set, predicted, &reserved)?;
                reserved.insert(slot);
                missing.push((expert, slot));
            }
        }
        Ok(missing)
    }
}

/// Final LRU working set after chronological prompt requests. The staged walk
/// only imports this final set into the persistent window.
pub(super) fn tail_experts(ids: &[i32], capacity: usize) -> HashSet<usize> {
    let mut result = HashSet::new();
    for &id in ids.iter().rev() {
        result.insert(id as usize);
        if result.len() == capacity {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pending_imports_do_not_publish_or_evict_requested_weights() -> Result<()> {
        let mut w = Window::new(3);
        w.publish(0, 4);
        w.publish(1, 5);
        w.publish(2, 6);
        let plan = w.plan(&[4, 7, 8], &HashSet::new())?;
        assert_eq!(plan, [(7, 1), (8, 2)]);
        assert_eq!(w.slot(4), Some(0));
        assert_eq!(w.slot(7), None);
        assert_eq!(w.slot(5), Some(1));
        Ok(())
    }
    #[test]
    fn predictions_protect_lru_entries_with_a_fallback_when_all_are_predicted() -> Result<()> {
        let mut w = Window::new(3);
        for id in 0..3 {
            w.publish(id, id);
        }
        assert_eq!(w.plan(&[2, 3], &HashSet::from([0]))?, [(3, 1)]);
        assert_eq!(w.plan(&[2, 3], &HashSet::from([0, 1]))?, [(3, 0)]);
        w.touch(0);
        assert_eq!(w.plan(&[2, 3], &HashSet::new())?, [(3, 1)]);
        Ok(())
    }
    #[test]
    fn prompt_seed_uses_recent_unique_experts() {
        assert_eq!(
            tail_experts(&[0, 0, 0, 1, 2, 3, 2], 3),
            HashSet::from([1, 2, 3])
        );
    }
}
