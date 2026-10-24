//! The model's single NextN block, with a private attention cache. Inputs are
//! final-normalized target hidden[t] and embedding(token[t+1]); output predicts
//! token[t+2]. Recursive draft steps use the MTP's own normalized hidden state.
mod adapter;
mod load;
pub(super) struct Snapshot {
    seed: Vec<f32>,
    seed_rope: [usize; 3],
    canonical: usize,
}
impl Snapshot {
    pub fn bytes(&self) -> usize {
        self.seed.len() * 4
    }
}
#[cfg(test)]
mod tests;
use super::{
    config::Config,
    decoder::{Buffers, Execution, Layer, Mixer},
    timings::Timings,
};
use crate::{
    model::{checkpoint::Tensor, cpu::Cpu, projection::Linear, trace::Trace},
    ops::norm,
};
use anyhow::{Result, ensure};

pub(super) struct Predictor {
    pub backend: crate::npu::Backend,
    layer: Layer,
    fc: Linear,
    hidden_norm: Vec<f32>,
    embedding_norm: Vec<f32>,
    final_norm: Vec<f32>,
    adapter: Option<adapter::Adapter>,
    work: Buffers,
    concat: Vec<f32>,
    input: Vec<f32>,
    normalized: Vec<f32>,
    /// Last canonical target hidden, converted to the MTP coordinate basis.
    seed: Vec<f32>,
    seed_rope: [usize; 3],
    recursive: Vec<f32>,
    position: usize,
    canonical: usize,
    pub max_draft: usize,
    pub elapsed_ms: f64,
}
impl Predictor {
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            seed: self.seed.clone(),
            seed_rope: self.seed_rope,
            canonical: self.canonical,
        }
    }
    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        self.truncate(snapshot.canonical)?;
        self.canonical = snapshot.canonical;
        self.seed.clone_from(&snapshot.seed);
        self.seed_rope = snapshot.seed_rope;
        self.recursive.clear();
        Ok(())
    }
    pub fn reset(&mut self) -> Result<()> {
        if let Mixer::Attention { cache, .. } = &self.layer.mixer {
            cache.reset()?;
        }
        self.seed.clear();
        self.recursive.clear();
        self.position = 0;
        self.canonical = 0;
        self.elapsed_ms = 0.;
        Ok(())
    }
    fn truncate(&mut self, position: usize) -> Result<()> {
        if let Mixer::Attention { cache, .. } = &self.layer.mixer {
            cache.truncate(position)?;
        }
        self.position = position;
        Ok(())
    }
    pub fn abort(&mut self) -> Result<()> {
        self.truncate(self.canonical)?;
        self.recursive.clear();
        Ok(())
    }
    fn fuse(&mut self, c: &Config, hidden: &[f32], embedding: &[f32], rows: usize) -> Result<()> {
        let h = c.hidden_size;
        ensure!(
            rows > 0 && hidden.len() == rows * h && embedding.len() == rows * h,
            "MTP shifted hidden/embedding shape mismatch"
        );
        self.input.resize(rows * h, 0.);
        self.concat.resize(rows * 2 * h, 0.);
        norm::rms_norm(hidden, &self.hidden_norm, c.epsilon, true, &mut self.input)?;
        norm::rms_norm(
            embedding,
            &self.embedding_norm,
            c.epsilon,
            true,
            &mut self.work.normalized[..rows * h],
        )?;
        for r in 0..rows {
            self.concat[r * 2 * h..r * 2 * h + h]
                .copy_from_slice(&self.work.normalized[r * h..(r + 1) * h]);
            self.concat[r * 2 * h + h..(r + 1) * 2 * h]
                .copy_from_slice(&self.input[r * h..(r + 1) * h]);
        }
        self.fc
            .run(&self.concat, &mut self.work.hidden[..rows * h], rows)?;
        Ok(())
    }
    /// A teacher-forced one-layer MTP prefix only needs its keys and values.
    /// No draft hidden output is consumed: the next seed comes from the target.
    fn prefill_kv(
        &mut self,
        c: &Config,
        hidden: &[f32],
        embedding: &[f32],
        positions: &[[usize; 3]],
    ) -> Result<()> {
        let start = std::time::Instant::now();
        let rows = positions.len();
        let h = c.hidden_size;
        self.fuse(c, hidden, embedding, rows)?;
        norm::rms_norm(
            &self.work.hidden[..rows * h],
            &self.layer.input_norm,
            c.epsilon,
            true,
            &mut self.work.normalized[..rows * h],
        )?;
        let kv = c.num_key_value_heads * c.head_dim;
        self.layer.inputs.run_from(
            1,
            &self.work.normalized[..rows * h],
            &mut self.work.mixed[..rows * 2 * kv],
            rows,
            &mut self.work.scatter,
        )?;
        for (r, row) in self.work.mixed[..rows * 2 * kv]
            .chunks_exact(2 * kv)
            .enumerate()
        {
            self.work.key[r * kv..(r + 1) * kv].copy_from_slice(&row[..kv]);
            self.work.value[r * kv..(r + 1) * kv].copy_from_slice(&row[kv..]);
        }
        if let Mixer::Attention { cache, k_norm, .. } = &self.layer.mixer {
            norm::rms_norm(
                &self.work.key[..rows * kv],
                k_norm,
                c.epsilon,
                true,
                &mut self.work.norm_temp[..rows * kv],
            )?;
            self.work.key[..rows * kv].copy_from_slice(&self.work.norm_temp[..rows * kv]);
            c.rope.apply(
                &mut self.work.key[..rows * kv],
                c.num_key_value_heads,
                positions,
            )?;
            cache.append(
                &self.work.key[..rows * kv],
                &self.work.value[..rows * kv],
                rows,
                self.position,
            )?;
        } else {
            anyhow::bail!("MTP requires a full-attention layer");
        }
        self.position += rows;
        self.elapsed_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
    fn run(
        &mut self,
        c: &Config,
        cpu: &Cpu,
        hidden: &[f32],
        embedding: &[f32],
        positions: &[[usize; 3]],
        trace: &Trace,
    ) -> Result<()> {
        let start = std::time::Instant::now();
        let h = c.hidden_size;
        let rows = positions.len();
        self.fuse(c, hidden, embedding, rows)?;
        self.layer.forward(
            &mut self.work,
            &Execution {
                config: c,
                cpu,
                rows,
                position: self.position,
                index: c.layers.len() as i32,
                trace,
                rope_positions: positions,
                verify: false,
            },
            &mut Timings::default(),
        )?;
        self.normalized.resize(rows * h, 0.);
        norm::rms_norm(
            &self.work.hidden[..rows * h],
            &self.final_norm,
            c.epsilon,
            true,
            &mut self.normalized,
        )?;
        self.position += rows;
        self.elapsed_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
    /// Catch up canonical draft KV using verified target rows. The first target
    /// token has no preceding hidden state and is deliberately omitted.
    pub fn catch_up(
        &mut self,
        c: &Config,
        _cpu: &Cpu,
        embeddings: &[f32],
        hidden: &[f32],
        positions: &[[usize; 3]],
        _trace: &Trace,
    ) -> Result<()> {
        self.abort()?;
        let h = c.hidden_size;
        let rows = positions.len();
        let mut target = hidden.to_vec();
        let mut embeddings = embeddings.to_vec();
        if let Some(a) = &self.adapter {
            a.target_hidden(&mut target);
            a.embedding(&mut embeddings);
        }
        let skip = usize::from(self.seed.is_empty());
        let mut shifted = self.seed.clone();
        shifted.extend_from_slice(&target[..(rows - 1) * h]);
        let mut rope = Vec::with_capacity(rows);
        if skip == 0 {
            rope.push(self.seed_rope);
        }
        rope.extend_from_slice(&positions[..rows - 1]);
        if rows > skip {
            self.prefill_kv(c, &shifted, &embeddings[skip * h..], &rope)?;
        }
        self.seed.clear();
        self.seed.extend_from_slice(&target[(rows - 1) * h..]);
        self.seed_rope = positions[rows - 1];
        self.canonical = self.position;
        Ok(())
    }
    pub fn draft(
        &mut self,
        c: &Config,
        cpu: &Cpu,
        embedding: &Tensor,
        token: u32,
        trace: &Trace,
    ) -> Result<Vec<f32>> {
        ensure!(!self.seed.is_empty(), "MTP prompt was not prefetched");
        ensure!(
            self.position - self.canonical < self.max_draft,
            "MTP proposal window exceeded"
        );
        let h = c.hidden_size;
        let mut e = vec![0.; h];
        embedding.row(token as usize, &mut e);
        if let Some(a) = &self.adapter {
            a.embedding(&mut e);
        }
        let seed = if self.recursive.is_empty() {
            self.seed.clone()
        } else {
            self.recursive.clone()
        };
        let offset = self.position - self.canonical;
        let rope = [self.seed_rope.map(|p| p + offset)];
        self.run(c, cpu, &seed, &e, &rope, trace)?;
        self.recursive.clone_from(&self.normalized);
        let mut output = self.normalized.clone();
        if let Some(a) = &self.adapter {
            a.head_input(&mut output);
        }
        Ok(output)
    }
}
