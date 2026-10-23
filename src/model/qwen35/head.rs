//! Qwen3.5 output-weight selection and final zero-centered normalization.
use super::runtime::Qwen35;
use crate::{
    model::{
        ModelOptions,
        checkpoint::{QuantLoader, Tensor},
        cpu::Cpu,
        output_head::OutputHead,
    },
    ops::norm,
};
use anyhow::{Result, ensure};
use std::{sync::Arc, time::Instant};

pub(super) fn load(
    options: &ModelOptions,
    domain: u32,
    loader: &Arc<QuantLoader>,
    cpu: &Cpu,
    embedding: &Tensor,
    tied: bool,
) -> Result<OutputHead> {
    let (vocab, width) = (embedding.shape[0], embedding.shape[1]);
    if loader.store.contains("lm_head.weight_packed")
        || loader.store.contains("lm_head.weight_scale")
    {
        OutputHead::quantized(options, domain, loader, cpu, vocab, width)
    } else {
        let tensor = if loader.store.contains("lm_head.weight") {
            loader.store.get("lm_head.weight")?
        } else {
            ensure!(tied, "missing untied lm_head");
            embedding.clone()
        };
        OutputHead::floating(tensor, vocab, width)
    }
}

impl Qwen35 {
    pub(super) fn output_head(&mut self, rows: usize) -> Result<()> {
        let start = Instant::now();
        let h = self.config.hidden_size;
        norm::rms_norm(
            &self.work.hidden[(rows - 1) * h..rows * h],
            &self.final_norm,
            self.config.epsilon,
            true,
            &mut self.work.normalized[..h],
        )?;
        self.trace
            .save(self.position, -1, "head_input", &self.work.normalized[..h])?;
        self.logits.resize(self.info.vocab_size, 0.0);
        self.head
            .run(&self.cpu, &self.work.normalized[..h], &mut self.logits, 1)?;
        self.head_ms += start.elapsed().as_secs_f64() * 1000.0;
        debug_assert!(
            self.logits.iter().all(|v| v.is_finite()),
            "nonfinite Qwen3.5 logits"
        );
        self.trace.save(self.position, -1, "logits", &self.logits)
    }
}
