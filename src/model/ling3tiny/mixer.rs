use super::{config::Config, load::Loader};
use crate::model::trace;
use crate::{
    model::projection::Linear,
    npu::{FloatLinear, Mla},
    ops::{activation::sigmoid, causal_conv::ConvState, kda, norm},
};
use anyhow::Result;

struct Convolution {
    projection: Linear,
    weights: Vec<f32>,
    state: ConvState,
}
pub(super) struct Kda {
    conv: Vec<Convolution>,
    forget: Linear,
    beta: Linear,
    gate: Linear,
    output: Linear,
    a: Vec<f32>,
    dt: Vec<f32>,
    norm: Vec<f32>,
    state: kda::State,
}
pub(super) struct Latent {
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

pub(super) fn rotate_interleaved(values: &mut [f32], position: usize, theta: f32) {
    for (i, pair) in values.chunks_exact_mut(2).enumerate() {
        let angle = position as f32 * theta.powf(-((2 * i) as f32) / 64.);
        let (s, c) = angle.sin_cos();
        let (a, b) = (pair[0], pair[1]);
        pair[0] = a * c - b * s;
        pair[1] = b * c + a * s;
    }
}

impl Mixer {
    pub fn load(l: &Loader<'_>, p: &str, index: usize) -> Result<Self> {
        let c = l.config;
        if c.mla(index) {
            let weights = l.matrix(&format!("{p}.kv_b_proj"), 4096, 512)?;
            Ok(Self::Mla(Box::new(Latent {
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
                    projection: l.linear(&format!("{p}.{name}_proj"), 2048, c.hidden)?,
                    weights: w.floats()?,
                    state: ConvState::new(2048, c.conv)?,
                });
            }
            Ok(Self::Kda(Box::new(Kda {
                conv,
                forget: l.linear(&format!("{p}.f_proj"), 2048, c.hidden)?,
                beta: l.linear(&format!("{p}.b_proj"), 16, c.hidden)?,
                gate: l.linear(&format!("{p}.g_proj"), 2048, c.hidden)?,
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
                let mut qkv = Vec::new();
                for conv in &mut k.conv {
                    let mut projected = vec![0.; rows * 2048];
                    trace::measure(c"ling.kda_qkv_projection", || {
                        conv.projection.run(input, &mut projected, rows)
                    })?;
                    let mut values = vec![0.; rows * 2048];
                    trace::measure(c"ling.kda_convolution", || {
                        conv.state
                            .run_parallel(&cpu.pool, &projected, &conv.weights, &mut values)
                    })?;
                    qkv.push(values);
                }
                trace::measure(c"ling.kda_qk_norm_gates", || -> Result<()> {
                    norm::l2_normalize(&mut qkv[0], 128)?;
                    norm::l2_normalize(&mut qkv[1], 128)
                })?;
                let mut decay = vec![0.; rows * 2048];
                let mut beta = vec![0.; rows * 16];
                trace::measure(c"ling.kda_f_g_projection", || {
                    k.forget.run(input, &mut decay, rows)
                })?;
                trace::measure(c"ling.kda_beta_projection", || {
                    k.beta.run(input, &mut beta, rows)
                })?;
                trace::measure(c"ling.kda_qk_norm_gates", || {
                    for (i, x) in decay.iter_mut().enumerate() {
                        *x = c.lower_bound * sigmoid((*x + k.dt[i % 2048]) * k.a[(i / 128) % 16]);
                    }
                    for b in &mut beta {
                        *b = sigmoid(*b);
                    }
                });
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
                let mut gated = vec![0.; rows * 2048];
                trace::measure(c"ling.kda_f_g_projection", || {
                    k.gate.run(input, &mut gated, rows)
                })?;
                let mut normalized = vec![0.; rows * 2048];
                trace::measure(c"ling.kda_output_norm_gate", || -> Result<()> {
                    norm::rms_norm(&attended, &k.norm, c.epsilon, false, &mut normalized)?;
                    for ((out, x), gate) in attended.iter_mut().zip(normalized).zip(gated) {
                        *out = x * sigmoid(gate);
                    }
                    Ok(())
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
                        rotate_interleaved(
                            &mut rope[r * 64..(r + 1) * 64],
                            position + r,
                            c.rope_theta,
                        );
                        for head in q[r * 3072..(r + 1) * 3072].chunks_mut(192) {
                            rotate_interleaved(&mut head[128..], position + r, c.rope_theta);
                        }
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
        rotate_interleaved(&mut x, 37, 6_000_000.);
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
