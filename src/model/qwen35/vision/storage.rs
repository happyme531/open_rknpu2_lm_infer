//! Tensor operations shared by the two ViT activation storage modes.
use crate::{
    layers::layer_norm::LayerNorm,
    model::trace::Trace,
    npu::{FloatLinear, FullAttention},
    ops::{gelu, vision_half},
    vision::math,
};
use anyhow::{Result, ensure};
use rayon::ThreadPool;
use std::borrow::Cow;

pub(super) trait Element: Copy + Default {
    type Rotary;
    fn input(values: &[f32]) -> Cow<'_, [Self]>;
    fn output(values: Vec<Self>) -> Vec<f32>;
    fn linear(layer: &FloatLinear, input: &[Self], output: &mut [Self], rows: usize) -> Result<()>;
    fn norm(
        layer: &LayerNorm,
        pool: &ThreadPool,
        input: &[Self],
        output: &mut [Self],
    ) -> Result<()>;
    fn attention(
        layer: &mut FullAttention,
        q: &[Self],
        k: &[Self],
        v: &[Self],
        output: &mut [Self],
        rows: usize,
    ) -> Result<()>;
    fn position(
        hidden: &mut [Self],
        table: &[f32],
        width: usize,
        side: usize,
        h: usize,
        w: usize,
    ) -> Result<()>;
    fn prepare_rotary(heads: usize, dim: usize, positions: &[[usize; 2]]) -> Result<Self::Rotary>;
    fn rotary(rotary: &Self::Rotary, pool: &ThreadPool, values: &mut [Self]) -> Result<()>;
    fn add(pool: &ThreadPool, hidden: &mut [Self], projected: &[Self]) -> Result<()>;
    fn linear_gelu(
        layer: &FloatLinear,
        pool: &ThreadPool,
        input: &[Self],
        output: &mut [Self],
        rows: usize,
    ) -> Result<()>;
    fn merger_gelu(pool: &ThreadPool, values: &mut [Self]);
    fn trace(trace: &Trace, layer: i32, name: &str, values: &[Self]) -> Result<()>;
}

impl Element for u16 {
    type Rotary = vision_half::Rotary;
    fn input(values: &[f32]) -> Cow<'_, [Self]> {
        Cow::Owned(vision_half::narrow(values))
    }
    fn output(values: Vec<Self>) -> Vec<f32> {
        vision_half::widen(&values)
    }
    fn linear(layer: &FloatLinear, input: &[Self], output: &mut [Self], rows: usize) -> Result<()> {
        layer.run_half(input, output, rows)
    }
    fn norm(
        layer: &LayerNorm,
        pool: &ThreadPool,
        input: &[Self],
        output: &mut [Self],
    ) -> Result<()> {
        layer.run_half(pool, input, output)
    }
    fn attention(
        layer: &mut FullAttention,
        q: &[Self],
        k: &[Self],
        v: &[Self],
        output: &mut [Self],
        rows: usize,
    ) -> Result<()> {
        layer.run_full_half(q, k, v, output, rows)
    }
    fn position(
        hidden: &mut [Self],
        table: &[f32],
        width: usize,
        side: usize,
        h: usize,
        w: usize,
    ) -> Result<()> {
        math::add_interpolated_position_half(hidden, table, width, side, h, w)
    }
    fn prepare_rotary(heads: usize, dim: usize, positions: &[[usize; 2]]) -> Result<Self::Rotary> {
        vision_half::Rotary::new(heads, dim, positions)
    }
    fn rotary(rotary: &Self::Rotary, pool: &ThreadPool, values: &mut [Self]) -> Result<()> {
        rotary.apply(pool, values)
    }
    fn add(pool: &ThreadPool, hidden: &mut [Self], projected: &[Self]) -> Result<()> {
        vision_half::add(pool, hidden, projected)
    }
    fn linear_gelu(
        layer: &FloatLinear,
        pool: &ThreadPool,
        input: &[Self],
        output: &mut [Self],
        rows: usize,
    ) -> Result<()> {
        layer.run_half_gelu(pool, input, output, rows)
    }
    fn merger_gelu(pool: &ThreadPool, values: &mut [Self]) {
        vision_half::gelu(pool, values);
    }
    fn trace(trace: &Trace, layer: i32, name: &str, values: &[Self]) -> Result<()> {
        trace.save_half(0, layer, name, values)
    }
}

pub(super) struct FloatRotary {
    heads: usize,
    dim: usize,
    positions: Vec<[usize; 2]>,
}
impl Element for f32 {
    type Rotary = FloatRotary;
    fn input(values: &[f32]) -> Cow<'_, [Self]> {
        Cow::Borrowed(values)
    }
    fn output(values: Vec<Self>) -> Vec<f32> {
        values
    }
    fn linear(layer: &FloatLinear, input: &[Self], output: &mut [Self], rows: usize) -> Result<()> {
        layer.run(input, output, rows)
    }
    fn norm(
        layer: &LayerNorm,
        pool: &ThreadPool,
        input: &[Self],
        output: &mut [Self],
    ) -> Result<()> {
        layer.run(pool, input, output)
    }
    fn attention(
        layer: &mut FullAttention,
        q: &[Self],
        k: &[Self],
        v: &[Self],
        output: &mut [Self],
        rows: usize,
    ) -> Result<()> {
        layer.run_full(q, k, v, output, rows)
    }
    fn position(
        hidden: &mut [Self],
        table: &[f32],
        width: usize,
        side: usize,
        h: usize,
        w: usize,
    ) -> Result<()> {
        math::add_interpolated_position(hidden, table, width, side, h, w)
    }
    fn prepare_rotary(heads: usize, dim: usize, positions: &[[usize; 2]]) -> Result<Self::Rotary> {
        Ok(FloatRotary {
            heads,
            dim,
            positions: positions.to_vec(),
        })
    }
    fn rotary(rotary: &Self::Rotary, _pool: &ThreadPool, values: &mut [Self]) -> Result<()> {
        math::rotary(values, rotary.heads, rotary.dim, &rotary.positions)
    }
    fn add(_pool: &ThreadPool, hidden: &mut [Self], projected: &[Self]) -> Result<()> {
        ensure!(
            hidden.len() == projected.len(),
            "vision residual shape mismatch"
        );
        for (x, y) in hidden.iter_mut().zip(projected) {
            *x += y;
        }
        Ok(())
    }
    fn linear_gelu(
        layer: &FloatLinear,
        pool: &ThreadPool,
        input: &[Self],
        output: &mut [Self],
        rows: usize,
    ) -> Result<()> {
        layer.run_gelu(pool, input, output, rows)
    }
    fn merger_gelu(pool: &ThreadPool, values: &mut [Self]) {
        gelu::apply(pool, values);
    }
    fn trace(trace: &Trace, layer: i32, name: &str, values: &[Self]) -> Result<()> {
        trace.save(0, layer, name, values)
    }
}
