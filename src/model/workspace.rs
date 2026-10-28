//! Row shapes and phase transitions shared by target decoders.
use crate::npu::Backend;
use anyhow::Result;

#[derive(Default)]
pub(crate) struct Workspaces {
    decoding: bool,
}

impl Workspaces {
    pub fn is_decoding(&self) -> bool {
        self.decoding
    }

    pub fn reset(&mut self) {
        self.decoding = false;
    }

    pub fn prepare(&mut self, rows: &mut Vec<usize>, decode_rows: &[usize]) {
        self.reset();
        rows.extend_from_slice(decode_rows);
        rows.sort_unstable();
        rows.dedup();
    }

    /// Keep only decode/verification shapes after the prompt has been consumed.
    pub fn enter_decode<'a>(
        &mut self,
        backends: impl IntoIterator<Item = &'a Backend>,
        rows: &[usize],
    ) -> Result<bool> {
        if self.is_decoding() {
            return Ok(false);
        }
        for backend in backends {
            backend.retain_workspaces(rows)?;
        }
        self.decoding = true;
        Ok(true)
    }
}

/// Convert logical prompt lengths to the physical row shapes they execute.
pub(crate) fn physical_rows(lengths: impl IntoIterator<Item = usize>, ubatch: usize) -> Vec<usize> {
    let mut rows = Vec::new();
    for length in lengths {
        if length == 0 {
            continue;
        }
        rows.push(length.min(ubatch));
        let tail = length % ubatch;
        if tail > 0 {
            rows.push(tail);
        }
    }
    rows.sort_unstable();
    rows.dedup();
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_replay_and_logical_prompt_blocks_keep_only_executed_shapes() {
        assert_eq!(physical_rows([4], 128), [4]);
        assert_eq!(physical_rows([20, 1024, 330, 4], 128), [4, 20, 74, 128]);
        assert_eq!(physical_rows([128, 256], 128), [128]);
        assert!(physical_rows([0], 128).is_empty());
    }
}
