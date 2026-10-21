//! DSpark's vanilla Markov correction and learned acceptance head.
//! Equations follow DeepSeek's DeepSpec deepspec/modeling/dspark/{markov_head,common}.py;
//! see DeepSpec-LICENSE for its MIT license.
use super::Config;
use crate::{
    model::{
        checkpoint::{Tensor, TensorStore},
        cpu::Cpu,
        projection::Linear,
    },
    npu::FloatBackend,
};
use anyhow::{Result, ensure};

pub(super) struct Markov {
    previous: Tensor,
    projection: Linear,
    confidence: Option<(Vec<f32>, f32, bool)>,
    rank: usize,
    hidden: usize,
    draft_hidden: Vec<f32>,
}

impl Markov {
    pub fn load(c: &Config, store: &TensorStore, backend: &FloatBackend) -> Result<Self> {
        let m = &c.dflash_config;
        ensure!(
            m.markov_rank > 0 && m.markov_head_type == "vanilla",
            "DSpark requires vanilla Markov head"
        );
        let previous = store.get("markov_head.markov_w1.weight")?;
        previous.require_shape(
            &[c.vocab_size, m.markov_rank],
            "DSpark previous-token embeddings",
        )?;
        let confidence = if m.enable_confidence_head {
            let n = c.hidden_size
                + if m.confidence_head_with_markov {
                    m.markov_rank
                } else {
                    0
                };
            let weight = store.get("confidence_head.proj.weight")?;
            weight.require_shape(&[1, n], "DSpark confidence weight")?;
            let bias = store.get("confidence_head.proj.bias")?;
            bias.require_shape(&[1], "DSpark confidence bias")?;
            Some((
                weight.floats()?,
                bias.floats()?[0],
                m.confidence_head_with_markov,
            ))
        } else {
            None
        };
        Ok(Self {
            previous,
            projection: Linear::floating(
                backend,
                store,
                "markov_head.markov_w2",
                c.vocab_size,
                m.markov_rank,
            )?,
            confidence,
            rank: m.markov_rank,
            hidden: c.hidden_size,
            draft_hidden: Vec::new(),
        })
    }
    pub fn remember(&mut self, hidden: &[f32]) {
        self.draft_hidden.clear();
        self.draft_hidden.extend_from_slice(hidden);
    }
    pub fn reset(&mut self) {
        self.draft_hidden.clear();
    }
    pub fn adjust(
        &self,
        cpu: &Cpu,
        row: usize,
        previous: u32,
        logits: &mut [f32],
    ) -> Result<Option<f64>> {
        ensure!(
            (previous as usize) < self.previous.shape[0]
                && (row + 1) * self.hidden <= self.draft_hidden.len(),
            "invalid DSpark proposal row/token"
        );
        let mut latent = vec![0.; self.rank];
        self.previous.row(previous as usize, &mut latent);
        let mut bias = vec![0.; logits.len()];
        self.projection.run(&latent, &mut bias, 1)?;
        cpu.add(logits, &bias);
        Ok(self.confidence.as_ref().map(|(weight, bias, with_markov)| {
            let hidden = &self.draft_hidden[row * self.hidden..(row + 1) * self.hidden];
            let mut score = *bias;
            for (&x, &w) in hidden.iter().zip(weight) {
                score = x.mul_add(w, score);
            }
            if *with_markov {
                for (&x, &w) in latent.iter().zip(&weight[self.hidden..]) {
                    score = x.mul_add(w, score);
                }
            }
            1. / (1. + (-(score as f64)).exp())
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::ModelOptions, npu::Backend, validation::BoardLock};
    use std::path::PathBuf;
    #[test]
    #[ignore = "requires MINICPM_TEST_DSPARK and exclusive RK3588 access"]
    fn minicpm_dspark_markov_matches_floating_reference() -> Result<()> {
        let _lock = BoardLock::acquire()?;
        let mut options = ModelOptions {
            path: std::env::var("MINICPM_TEST_DSPARK")?.into(),
            context: 512,
            chunk: 128,
            threads: 4,
            npu_mask: 7,
            cpu_mask: 0xf0,
            tune: false,
            tune_search: false,
            tune_cache: PathBuf::new(),
            attention_tile: 512,
            vision_dtype: Default::default(),
            speculation: Default::default(),
            experts: Default::default(),
        };
        crate::npu::configure_cpu(&mut options)?;
        let c = Config::load(&options.path, options.context)?;
        let store = TensorStore::open(&options.path)?;
        let backend = FloatBackend::for_decoder(Backend::new(&options, 2)?);
        let mut markov = Markov::load(&c, &store, &backend)?;
        let cpu = Cpu::new(4)?;
        let hidden: Vec<_> = (0..c.hidden_size)
            .map(|i| ((i % 37) as f32 * 0.17).sin())
            .collect();
        markov.remember(&hidden);
        let w2 = store.get("markov_head.markov_w2.weight")?;
        for previous in [17, 695] {
            let mut actual = vec![0.; c.vocab_size];
            let confidence = markov.adjust(&cpu, 0, previous, &mut actual)?.unwrap();
            let mut latent = vec![0.; markov.rank];
            markov.previous.row(previous as usize, &mut latent);
            let mut expected = vec![0.; c.vocab_size];
            cpu.dense_head(&w2, &latent, &mut expected);
            let delta: f64 = actual
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                .sum();
            let energy: f64 = expected.iter().map(|&v| (v as f64).powi(2)).sum();
            let nrmse = (delta / energy).sqrt();
            println!("DSpark Markov previous={previous} nrmse={nrmse} confidence={confidence}");
            ensure!(
                nrmse < 0.002,
                "DSpark Markov projection differs from floating reference"
            );
            let (w, bias, with_markov) = markov.confidence.as_ref().unwrap();
            let mut score = *bias as f64;
            for (&x, &w) in hidden
                .iter()
                .chain(
                    latent
                        .iter()
                        .take(if *with_markov { latent.len() } else { 0 }),
                )
                .zip(w)
            {
                score += x as f64 * w as f64;
            }
            let expected = 1. / (1. + (-score).exp());
            ensure!(
                (expected - confidence).abs() < 1e-5,
                "DSpark confidence head mismatch"
            );
        }
        Ok(())
    }
}
