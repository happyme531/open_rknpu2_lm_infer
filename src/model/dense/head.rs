use super::{Dense, elapsed};
use anyhow::Result;
use std::time::Instant;

impl Dense {
    pub(super) fn output_head(&mut self, rows: usize) -> Result<()> {
        if !self.work.workspaces.is_decoding() {
            let (workspaces, targets) = self.workspace_targets();
            workspaces.finish_prefill(targets)?;
        }
        let start = Instant::now();
        let batch = &mut self.work.batches[0];
        self.work.cpu.norm(
            &batch.hidden[(rows - 1) * self.weights.geometry.h..rows * self.weights.geometry.h],
            &mut batch.normalized[..self.weights.geometry.h],
            &self.weights.final_norm,
            self.weights.geometry.epsilon,
        );
        self.weights.trace.save(
            self.sequences.single().position,
            -1,
            "head_input",
            &self.work.batches[0].normalized[..self.weights.geometry.h],
        )?;
        self.weights.head.run(
            &self.work.cpu,
            &self.work.batches[0].normalized[..self.weights.geometry.h],
            &mut self.work.logits,
            1,
        )?;
        if self.weights.logits_scaling != 1.0 {
            for value in &mut self.work.logits {
                *value /= self.weights.logits_scaling;
            }
        }
        self.work.timings.head += elapsed(start);
        self.weights.trace.save(
            self.sequences.single().position,
            -1,
            "logits",
            &self.work.logits,
        )?;
        debug_assert!(
            self.work.logits.iter().all(|f| f.is_finite()),
            "nonfinite logits"
        );
        Ok(())
    }
}
