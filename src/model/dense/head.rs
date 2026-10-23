use super::{Dense, elapsed};
use anyhow::Result;
use std::time::Instant;

impl Dense {
    pub(super) fn output_head(&mut self, rows: usize) -> Result<()> {
        let start = Instant::now();
        self.cpu.norm(
            &self.hidden[(rows - 1) * self.h..rows * self.h],
            &mut self.normalized[..self.h],
            &self.final_norm,
            self.epsilon,
        );
        self.trace(-1, "head_input", &self.normalized[..self.h])?;
        self.head
            .run(&self.cpu, &self.normalized[..self.h], &mut self.logits, 1)?;
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
