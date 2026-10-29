use super::{Dense, elapsed};
use anyhow::Result;
use std::time::Instant;

impl Dense {
    pub(super) fn output_head(&mut self, rows: usize) -> Result<()> {
        self.workspaces.enter_decode(
            self.backends
                .iter()
                .chain(self.head.backend())
                .chain(self.secondary.iter().flat_map(|lane| lane.backends())),
            &[1],
        )?;
        let start = Instant::now();
        let batch = &mut self.batches[0];
        self.cpu.norm(
            &batch.hidden[(rows - 1) * self.geometry.h..rows * self.geometry.h],
            &mut batch.normalized[..self.geometry.h],
            &self.final_norm,
            self.geometry.epsilon,
        );
        self.trace(
            -1,
            "head_input",
            &self.batches[0].normalized[..self.geometry.h],
        )?;
        self.head.run(
            &self.cpu,
            &self.batches[0].normalized[..self.geometry.h],
            &mut self.logits,
            1,
        )?;
        if self.logits_scaling != 1.0 {
            for value in &mut self.logits {
                *value /= self.logits_scaling;
            }
        }
        self.timings.head += elapsed(start);
        self.trace(-1, "logits", &self.logits)?;
        debug_assert!(
            self.logits.iter().all(|f| f.is_finite()),
            "nonfinite logits"
        );
        Ok(())
    }
}
