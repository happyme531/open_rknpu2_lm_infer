use super::{Config, layer::Layer};
use crate::{
    model::{cpu::Cpu, projection::Linear},
    npu::{Backend, FloatBackend},
    ops::rope::MRope,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, Ordering};

pub struct DFlash {
    pub config: Config,
    pub(super) layers: Vec<Layer>,
    pub(super) fc: Linear,
    pub(super) hidden_norm: Vec<f32>,
    pub(super) final_norm: Vec<f32>,
    pub(super) rope: MRope,
    pub(super) cpu: Cpu,
    pub(super) backends: Vec<Backend>,
    pub(super) floating: Vec<FloatBackend>,
    pub(super) context: usize,
    pub(super) chunk: usize,
    pub(super) position: usize,
    pub(super) markov: Option<super::markov::Markov>,
}
impl DFlash {
    fn context_projection(&self, features: &[f32]) -> Result<Vec<f32>> {
        let rows = features.len() / self.config.context_width();
        ensure!(
            rows > 0 && rows * self.config.context_width() == features.len(),
            "DFlash feature shape mismatch"
        );
        let mut values = vec![0.; rows * self.config.hidden_size];
        self.fc.run(features, &mut values, rows)?;
        let mut normalized = vec![0.; values.len()];
        self.cpu.norm(
            &values,
            &mut normalized,
            &self.hidden_norm,
            self.config.rms_norm_eps,
        );
        Ok(normalized)
    }
    pub fn reset(&mut self) -> Result<()> {
        if let Some(markov) = &mut self.markov {
            markov.reset();
        }
        for l in &self.layers {
            l.cache.reset()?;
        }
        self.position = 0;
        Ok(())
    }
    pub(crate) fn abort(&self) -> Result<()> {
        for layer in &self.layers {
            layer.cache.truncate(self.position)?;
        }
        Ok(())
    }
    pub(crate) fn backends(&self) -> &[Backend] {
        &self.backends
    }
    pub fn prepare_context(&mut self, features: &[f32], cancel: Option<&AtomicBool>) -> Result<()> {
        let width = self.config.context_width();
        ensure!(
            features.len().is_multiple_of(width),
            "invalid DFlash prefix features"
        );
        for raw in features.chunks(self.chunk * width) {
            ensure!(
                !cancel.is_some_and(|c| c.load(Ordering::Relaxed)),
                "generation cancelled"
            );
            let rows = raw.len() / width;
            let context = self.context_projection(raw)?;
            for layer in &self.layers {
                ensure!(
                    !cancel.is_some_and(|c| c.load(Ordering::Relaxed)),
                    "generation cancelled"
                );
                layer.append_context(
                    &self.config,
                    &self.cpu,
                    &self.rope,
                    &context,
                    rows,
                    self.position,
                )?;
            }
            self.position += rows;
        }
        Ok(())
    }
    pub fn draft(
        &mut self,
        features: &[f32],
        embedding: &[f32],
        position: usize,
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        let rows = embedding.len() / self.config.hidden_size;
        let new_rows = features.len() / self.config.context_width();
        ensure!(
            position == self.position + new_rows && position + rows <= self.context,
            "DFlash cache/target position mismatch"
        );
        ensure!(
            ((1 + self.config.proposal_offset())..=self.config.dflash_config.block_size)
                .contains(&rows),
            "invalid DFlash block length"
        );
        let context = if new_rows > 0 {
            self.context_projection(features)?
        } else {
            Vec::new()
        };
        let mut hidden = embedding.to_vec();
        for l in &self.layers {
            ensure!(!cancel.load(Ordering::Relaxed), "generation cancelled");
            l.forward(
                &self.config,
                &self.cpu,
                &self.rope,
                &mut hidden,
                &context,
                position,
            )?;
        }
        self.position = position;
        let mut normalized = vec![0.; hidden.len()];
        self.cpu.norm(
            &hidden,
            &mut normalized,
            &self.final_norm,
            self.config.rms_norm_eps,
        );
        if let Some(markov) = &mut self.markov {
            markov.remember(&normalized);
        }
        Ok(normalized)
    }
    pub fn adjust(&self, row: usize, previous: u32, logits: &mut [f32]) -> Result<Option<f64>> {
        match &self.markov {
            Some(markov) => markov.adjust(&self.cpu, row, previous, logits),
            None => Ok(None),
        }
    }
    pub fn prepare(&self, prefix_rows: &[usize], max_draft: usize) -> Result<()> {
        let c = &self.config;
        let h = c.hidden_size;
        let f = c.intermediate_size;
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        let parts = c.context_width().div_ceil(4096);
        let fc_k = c.context_width().div_ceil(parts).div_ceil(32) * 32;
        let down_parts = f.div_ceil(4096);
        let down_k = f.div_ceil(down_parts).div_ceil(32) * 32;
        for b in &self.floating {
            let first = 1 + c.proposal_offset();
            let last = max_draft + c.proposal_offset();
            for m in first..=last {
                for (k, n) in [(h, q), (q, h), (h, f), (down_k, h)] {
                    b.prepare(k, n, m)?;
                }
            }
            let mut rows = prefix_rows.to_vec();
            rows.extend(1..=last);
            rows.sort_unstable();
            rows.dedup();
            for m in rows {
                for (k, n) in [(h, kv), (fc_k, h)] {
                    b.prepare(k, n, m)?;
                }
            }
        }
        Ok(())
    }
    pub fn stats(&self) -> Value {
        json!({"position": self.position, "linear_dtype":"fp16", "context_projection":"fp16", "backends": self.backends.iter().map(Backend::stats).collect::<Vec<_>>()})
    }
}
