use super::{
    checkpoint::{QuantLoader, concatenate},
    cpu::Cpu,
};
use crate::npu::{Backend, FloatBackend, FloatLinear};
use anyhow::{Result, ensure};
use std::sync::Arc;

/// Decoder projections share orchestration across integer target weights and
/// floating auxiliary networks; neither path changes checkpoint quantization.
pub(crate) enum Linear {
    Quantized(crate::npu::Linear),
    Floating {
        linear: FloatLinear,
        n: usize,
    },
    SplitK {
        parts: Vec<(Linear, usize, usize)>,
        k: usize,
        n: usize,
    },
}
impl From<crate::npu::Linear> for Linear {
    fn from(value: crate::npu::Linear) -> Self {
        Self::Quantized(value)
    }
}
impl Linear {
    fn n(&self) -> usize {
        match self {
            Self::Quantized(l) => l.n(),
            Self::Floating { n, .. } | Self::SplitK { n, .. } => *n,
        }
    }
    pub(crate) fn floating(
        backend: &FloatBackend,
        store: &super::checkpoint::TensorStore,
        name: &str,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        let w = store.get(&format!("{name}.weight"))?;
        w.require_shape(&[n, k], name)?;
        let weight = w.floats()?;
        if k > 4096 {
            // Bound native FP16 K without modifying or requantizing weights.
            // Each public Linear returns FP16-rounded partials, summed in FP32.
            let tile = k.div_ceil(k.div_ceil(4096)).div_ceil(32) * 32;
            let mut parts = Vec::new();
            for start in (0..k).step_by(tile) {
                let width = tile.min(k - start);
                let values: Vec<_> = weight
                    .chunks_exact(k)
                    .flat_map(|row| row[start..start + width].iter().copied())
                    .collect();
                parts.push((
                    Self::Floating {
                        linear: backend.load(&values, &vec![0.; n], width, n)?,
                        n,
                    },
                    start,
                    width,
                ));
            }
            return Ok(Self::SplitK { parts, k, n });
        }
        Ok(Self::Floating {
            linear: backend.load(&weight, &vec![0.; n], k, n)?,
            n,
        })
    }
    pub(crate) fn run(&self, input: &[f32], output: &mut [f32], rows: usize) -> Result<()> {
        match self {
            Self::Quantized(l) => l.run(input, output, rows),
            Self::SplitK { parts, k, n } => {
                ensure!(
                    input.len() == rows * k && output.len() == rows * n,
                    "invalid K-partitioned projection shape"
                );
                output.fill(0.);
                let mut partial = vec![0.; rows * n];
                for (part, start, width) in parts {
                    let packed: Vec<_> = input
                        .chunks_exact(*k)
                        .flat_map(|row| row[*start..start + width].iter().copied())
                        .collect();
                    part.run(&packed, &mut partial, rows)?;
                    for (out, value) in output.iter_mut().zip(&partial) {
                        *out += value;
                    }
                }
                Ok(())
            }
            Self::Floating { linear, .. } => linear.run(input, output, rows),
        }
    }
    fn run_swiglu(&self, input: &[f32], output: &mut [f32], rows: usize) -> Result<bool> {
        match self {
            Self::Quantized(l) => l.run_swiglu(input, output, rows),
            Self::Floating { .. } | Self::SplitK { .. } => Ok(false),
        }
    }
}

pub(crate) struct Projection {
    parts: Vec<(Linear, usize)>,
    width: usize,
}

impl Projection {
    pub(crate) fn floating(
        backend: &FloatBackend,
        store: &super::checkpoint::TensorStore,
        modules: Vec<(String, usize)>,
        k: usize,
    ) -> Result<Self> {
        let mut parts = Vec::new();
        let mut width = 0;
        for (name, n) in modules {
            parts.push((Linear::floating(backend, store, &name, n, k)?, width));
            width += n;
        }
        Ok(Self { parts, width })
    }
    pub(crate) fn scratch_width(&self) -> usize {
        if self.parts.len() <= 1 {
            0
        } else {
            self.parts
                .iter()
                .map(|(linear, _)| linear.n())
                .max()
                .unwrap_or(0)
        }
    }
    pub(crate) fn load(
        backend: &Backend,
        loader: &Arc<QuantLoader>,
        cpu: &Cpu,
        modules: Vec<(String, usize)>,
        k: usize,
    ) -> Result<Self> {
        let mut result = Self {
            parts: Vec::new(),
            width: 0,
        };
        let mut first = 0;
        while first < modules.len() {
            let bits = loader.bits(&modules[first].0)?;
            let mut last = first + 1;
            while last < modules.len() && loader.bits(&modules[last].0)? == bits {
                last += 1;
            }
            let group = modules[first..last].to_vec();
            let reload_loader = loader.clone();
            let reload_group = group.clone();
            let reload_pool = cpu.pool.clone();
            let reload = move || {
                reload_pool.install(|| {
                    concatenate(
                        reload_group
                            .iter()
                            .map(|(name, n)| reload_loader.load(name, *n, k))
                            .collect::<Result<Vec<_>>>()?,
                    )
                })
            };
            let matrix = cpu.pool.install(|| {
                concatenate(
                    group
                        .iter()
                        .map(|(name, n)| loader.load(name, *n, k))
                        .collect::<Result<Vec<_>>>()?,
                )
            })?;
            let n = matrix.n;
            let linear = backend.load(matrix, Some(Box::new(reload)))?;
            result.parts.push((linear.into(), result.width));
            result.width += n;
            first = last;
        }
        Ok(result)
    }
    pub(crate) fn run(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        scratch: &mut [f32],
    ) -> Result<()> {
        self.run_from(0, input, output, rows, scratch)
    }
    /// Run a suffix of independently stored projection modules (e.g. KV only).
    pub(crate) fn run_from(
        &self,
        first: usize,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        scratch: &mut [f32],
    ) -> Result<()> {
        ensure!(first < self.parts.len(), "invalid projection suffix");
        let base = self.parts[first].1;
        let width = self.width - base;
        ensure!(
            output.len() == rows * width,
            "projection suffix shape mismatch"
        );
        if self.parts.len() - first == 1 {
            return self.parts[first].0.run(input, output, rows);
        }
        for (linear, offset) in &self.parts[first..] {
            let offset = offset - base;
            let n = linear.n();
            if rows == 1 {
                linear.run(input, &mut output[offset..offset + n], 1)?;
            } else {
                linear.run(input, &mut scratch[..rows * n], rows)?;
                for r in 0..rows {
                    output[r * width + offset..r * width + offset + n]
                        .copy_from_slice(&scratch[r * n..(r + 1) * n]);
                }
            }
        }
        Ok(())
    }
    pub(crate) fn run_swiglu(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
    ) -> Result<bool> {
        if self.parts.len() != 1 {
            return Ok(false);
        }
        self.parts[0].0.run_swiglu(input, output, rows)
    }
}
