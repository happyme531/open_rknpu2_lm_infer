//! Shared output projection. Models choose tied/explicit weights, normalization,
//! output scaling and when to prepare row counts; this owns only the projection.
use super::{
    ModelOptions,
    checkpoint::{QuantLoader, Tensor},
    cpu::Cpu,
};
use crate::npu::{Backend, Linear};
use anyhow::{Result, ensure};
use std::sync::Arc;

enum Weights {
    Quantized {
        backend: Backend,
        linear: Linear,
        bits: u32,
    },
    Floating(Tensor),
}

pub(crate) struct OutputHead {
    weights: Weights,
    width: usize,
    vocab: usize,
}

impl OutputHead {
    pub fn quantized(
        options: &ModelOptions,
        domain: u32,
        loader: &Arc<QuantLoader>,
        cpu: &Cpu,
        vocab: usize,
        width: usize,
    ) -> Result<Self> {
        let backend = Backend::new(options, domain)?;
        let matrix = cpu.pool.install(|| loader.load("lm_head", vocab, width))?;
        let bits = matrix.bits;
        let reload = loader.clone();
        let pool = cpu.pool.clone();
        let linear = backend.load(
            matrix,
            Some(Box::new(move || {
                pool.install(|| reload.load("lm_head", vocab, width))
            })),
        )?;
        Ok(Self {
            weights: Weights::Quantized {
                backend,
                linear,
                bits,
            },
            width,
            vocab,
        })
    }

    pub fn floating(tensor: Tensor, vocab: usize, width: usize) -> Result<Self> {
        tensor.require_shape(&[vocab, width], "lm_head")?;
        ensure!(tensor.floating(), "lm_head must be BF16/F16/F32");
        Ok(Self {
            weights: Weights::Floating(tensor),
            width,
            vocab,
        })
    }

    pub fn backend(&self) -> Option<&Backend> {
        match &self.weights {
            Weights::Quantized { backend, .. } => Some(backend),
            Weights::Floating(_) => None,
        }
    }

    pub fn bits(&self) -> Option<u32> {
        match &self.weights {
            Weights::Quantized { bits, .. } => Some(*bits),
            Weights::Floating(_) => None,
        }
    }

    pub fn prepare(&self, rows: usize) -> Result<()> {
        if let Some(backend) = self.backend() {
            backend.prepare(rows)?;
        }
        Ok(())
    }

    pub fn run(&self, cpu: &Cpu, input: &[f32], output: &mut [f32], rows: usize) -> Result<()> {
        ensure!(
            input.len() == rows * self.width && output.len() == rows * self.vocab,
            "lm_head shape mismatch"
        );
        match &self.weights {
            Weights::Quantized {
                backend, linear, ..
            } => {
                linear.run(input, output, rows)?;
                if linear.compression_enabled() {
                    crate::npu::release_compression_scratch(std::iter::once(backend));
                }
                Ok(())
            }
            Weights::Floating(tensor) => {
                for (x, y) in input.chunks(self.width).zip(output.chunks_mut(self.vocab)) {
                    cpu.dense_head(tensor, x, y);
                }
                Ok(())
            }
        }
    }
}
