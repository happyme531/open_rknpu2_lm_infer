use super::{config::Config, load::Loader};
use crate::model::trace;
use crate::{
    model::projection::{Linear, Projection},
    npu::{FloatLinear, Mla},
    ops::{causal_conv::ConvState, kda, norm, rope::InterleavedRope},
};
use anyhow::Result;

struct Convolution {
    weights: Vec<f32>,
    state: ConvState,
}
pub(super) struct Kda {
    inputs: Projection,
    conv: Vec<Convolution>,
    beta: Linear,
    output: Linear,
    a: Vec<f32>,
    dt: Vec<f32>,
    norm: Vec<f32>,
    state: kda::State,
}
pub(super) struct Latent {
    rope: InterleavedRope,
    q_a: FloatLinear,
    q_b: FloatLinear,
    kv_a: FloatLinear,
    gate: FloatLinear,
    output: Linear,
    q_norm: Vec<f32>,
    kv_norm: Vec<f32>,
    cache: Mla,
}
pub(super) enum Mixer {
    Kda(Box<Kda>),
    Mla(Box<Latent>),
}

impl Mixer {
    pub fn load(l: &Loader<'_>, p: &str, index: usize) -> Result<Self> {
        let c = l.config;
        if c.mla(index) {
            let weights = l.matrix(&format!("{p}.kv_b_proj"), 4096, 512)?;
            Ok(Self::Mla(Box::new(Latent {
                rope: InterleavedRope::new(64, c.rope_theta)?,
                q_a: l.fp16(&format!("{p}.q_a_proj"), 256, c.hidden)?,
                q_b: l.fp16(&format!("{p}.q_b_proj"), 3072, 256)?,
                kv_a: l.fp16(&format!("{p}.kv_a_proj_with_mqa"), 576, c.hidden)?,
                gate: l.fp16(&format!("{p}.g_proj"), 16, c.hidden)?,
                output: l.npu_linear(&format!("{p}.dense"), c.hidden, 2048)?,
                q_norm: l.vector(&format!("{p}.q_a_layernorm.weight"), 256)?,
                kv_norm: l.vector(&format!("{p}.kv_a_layernorm.weight"), 512)?,
                cache: l.backend.mla(&weights)?,
            })))
        } else {
            let mut conv = Vec::new();
            for name in ["q", "k", "v"] {
                let w = l.weights.store.get(&format!("{p}.{name}_conv1d.weight"))?;
                w.require_shape(&[2048, 1, c.conv], "KDA convolution")?;
                conv.push(Convolution {
                    weights: w.floats()?,
                    state: ConvState::new(2048, c.conv)?,
                });
            }
            Ok(Self::Kda(Box::new(Kda {
                inputs: Projection::load(
                    l.backend,
                    l.weights,
                    l.cpu,
                    ["q", "k", "v", "f", "g"]
                        .into_iter()
                        .map(|name| (format!("{p}.{name}_proj"), 2048))
                        .collect(),
                    c.hidden,
                )?,
                conv,
                beta: l.linear(&format!("{p}.b_proj"), 16, c.hidden)?,
                output: l.linear(&format!("{p}.o_proj"), c.hidden, 2048)?,
                a: l.vector(&format!("{p}.A_log"), 16)?
                    .into_iter()
                    .map(f32::exp)
                    .collect(),
                dt: l.vector(&format!("{p}.dt_bias"), 2048)?,
                norm: l.vector(&format!("{p}.o_norm.weight"), 128)?,
                state: kda::State::new(16, 128)?,
            })))
        }
    }
    pub fn reset(&mut self) -> Result<()> {
        match self {
            Self::Kda(k) => {
                k.state.reset();
                for conv in &mut k.conv {
                    conv.state.reset();
                }
                Ok(())
            }
            Self::Mla(m) => m.cache.reset(),
        }
    }
    pub fn reserve(&self, capacity: usize) -> Result<()> {
        if let Self::Mla(m) = self {
            m.cache.reserve(capacity)?;
        }
        Ok(())
    }
    pub fn prepare(&self, rows: usize) -> Result<()> {
        if let Self::Mla(m) = self {
            for projection in [&m.q_a, &m.q_b, &m.kv_a, &m.gate] {
                projection.prepare(rows)?;
            }
        }
        Ok(())
    }
    pub fn stats(&self) -> Result<serde_json::Value> {
        Ok(match self {
            Self::Mla(m) => m.cache.info()?,
            Self::Kda(k) => {
                serde_json::json!({"type":"kda","backend":"cpu","state_bytes":k.state.allocated_bytes(),
                "conv_bytes":k.conv.iter().map(|c| c.state.allocated_bytes()).sum::<usize>()})
            }
        })
    }
    pub fn forward(
        &mut self,
        input: &[f32],
        rows: usize,
        position: usize,
        c: &Config,
        cpu: &crate::model::cpu::Cpu,
    ) -> Result<Vec<f32>> {
        let mut attended = vec![0.; rows * 2048];
        let (output, output_phase) = match self {
            Self::Kda(k) => {
                let stride = 5 * 2048;
                let mut projected = vec![0.; rows * stride];
                let mut scratch = vec![0.; rows * k.inputs.scratch_width()];
                trace::measure(c"ling.kda_input_projection", || {
                    k.inputs.run(input, &mut projected, rows, &mut scratch)
                })?;
                let mut qkv = Vec::new();
                for (index, conv) in k.conv.iter_mut().enumerate() {
                    let mut values = vec![0.; rows * 2048];
                    let first = index * 2048;
                    trace::measure(c"ling.kda_convolution", || {
                        conv.state.run_parallel_strided(
                            &cpu.pool,
                            &projected[first..(rows - 1) * stride + first + 2048],
                            stride,
                            &conv.weights,
                            &mut values,
                        )
                    })?;
                    qkv.push(values);
                }
                trace::measure(c"ling.kda_qk_norm_gates", || -> Result<()> {
                    for values in &mut qkv[..2] {
                        if rows == 1 {
                            norm::l2_normalize(values, 128)?;
                        } else {
                            norm::l2_normalize_parallel(&cpu.pool, values, 128)?;
                        }
                    }
                    Ok(())
                })?;
                let mut decay = vec![0.; rows * 2048];
                let mut beta = vec![0.; rows * 16];
                trace::measure(c"ling.kda_beta_projection", || {
                    k.beta.run(input, &mut beta, rows)
                })?;
                trace::measure(c"ling.kda_qk_norm_gates", || {
                    kda::activate_gates(
                        &cpu.pool,
                        &projected[3 * 2048..(rows - 1) * stride + 4 * 2048],
                        stride,
                        &k.a,
                        &k.dt,
                        c.lower_bound,
                        &mut beta,
                        &mut decay,
                    )
                })?;
                trace::measure(c"ling.kda_recurrence", || {
                    k.state.run(
                        &qkv[0],
                        &qkv[1],
                        &qkv[2],
                        &decay,
                        &beta,
                        &mut attended,
                        cpu.pool.current_num_threads(),
                    )
                })?;
                trace::measure(c"ling.kda_output_norm_gate", || {
                    norm::sigmoid_gated_norm_in_place(
                        &cpu.pool,
                        &mut attended,
                        &projected[4 * 2048..],
                        stride,
                        2048,
                        &k.norm,
                        c.epsilon,
                    )
                })?;
                (&k.output, c"ling.kda_output_projection")
            }
            Self::Mla(m) => {
                let mut qa = vec![0.; rows * 256];
                let mut qn = qa.clone();
                trace::measure(c"ling.mla_q_a_projection", || {
                    m.q_a.run(input, &mut qa, rows)
                })?;
                trace::measure(c"ling.mla_norm_rope", || {
                    norm::rms_norm(&qa, &m.q_norm, c.epsilon, false, &mut qn)
                })?;
                let mut q = vec![0.; rows * 3072];
                trace::measure(c"ling.mla_q_b_projection", || m.q_b.run(&qn, &mut q, rows))?;
                let mut kv = vec![0.; rows * 576];
                let mut latent = vec![0.; rows * 512];
                let mut rope = vec![0.; rows * 64];
                let mut gates = vec![0.; rows * 16];
                trace::measure(c"ling.mla_kv_a_projection", || {
                    m.kv_a.run(input, &mut kv, rows)
                })?;
                trace::measure(c"ling.mla_gate_projection", || {
                    m.gate.run(input, &mut gates, rows)
                })?;
                trace::measure(c"ling.mla_norm_rope", || -> Result<()> {
                    for r in 0..rows {
                        norm::rms_norm(
                            &kv[r * 576..r * 576 + 512],
                            &m.kv_norm,
                            c.epsilon,
                            false,
                            &mut latent[r * 512..(r + 1) * 512],
                        )?;
                        rope[r * 64..(r + 1) * 64]
                            .copy_from_slice(&kv[r * 576 + 512..(r + 1) * 576]);
                        m.rope.apply(
                            &mut q[r * 3072..(r + 1) * 3072],
                            &mut rope[r * 64..(r + 1) * 64],
                            192,
                            position + r,
                        )?;
                    }
                    Ok(())
                })?;
                trace::measure(c"ling.mla_core", || {
                    m.cache
                        .run(&q, &latent, &rope, &gates, &mut attended, rows, position)
                })?;
                (&m.output, c"ling.mla_output_projection")
            }
        };
        let mut result = vec![0.; rows * c.hidden];
        trace::measure(output_phase, || output.run(&attended, &mut result, rows))?;
        Ok(result)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interleaved_rope_rotates_adjacent_pairs_at_nonzero_position() {
        let mut x: Vec<_> = (0..64).map(|i| (i as f32 * 0.13).sin()).collect();
        let before = x.clone();
        InterleavedRope::new(64, 6_000_000.)
            .unwrap()
            .apply(&mut [], &mut x, 192, 37)
            .unwrap();
        for i in 0..32 {
            let angle = 37_f64 * 6_000_000_f64.powf(-(i as f64) / 32.);
            let (s, c) = angle.sin_cos();
            assert!(
                (x[2 * i] as f64 - (before[2 * i] as f64 * c - before[2 * i + 1] as f64 * s)).abs()
                    < 3e-6
            );
            assert!(
                (x[2 * i + 1] as f64 - (before[2 * i + 1] as f64 * c + before[2 * i] as f64 * s))
                    .abs()
                    < 3e-6
            );
        }
    }
}
