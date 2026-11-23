use super::super::attention;
use super::super::timings::{Phase, Timings};
use super::{Buffers, Execution, LayerSeqState, MixerSpan, MixerWeights};
use crate::model::projection::Projection;
use crate::{
    layers::gated_delta::Projected,
    ops::{activation, norm},
};
use anyhow::Result;
use rayon::prelude::*;
use std::{ops::DerefMut, time::Instant};

impl MixerWeights {
    pub fn forward<'a, S: DerefMut<Target = LayerSeqState>>(
        &self,
        inputs: &Projection,
        w: &mut Buffers,
        e: &Execution<'_>,
        timings: &mut Timings,
        spans: impl Iterator<Item = MixerSpan<'a, S>>,
    ) -> Result<usize> {
        let Execution {
            config: c,
            cpu,
            rows,
            index,
            trace,
            verify,
            ..
        } = *e;
        let h = c.hidden_size;
        let attended_width = match self {
            MixerWeights::Recurrent(weights) => {
                let (cw, vw, hv) = (c.gdn.conv_width(), c.gdn.value_width(), c.gdn.value_heads());
                let width = cw + vw + 2 * hv;
                timings.stage(Phase::Linear, "target GDN input projection", || {
                    inputs.run(
                        &w.normalized[..rows * h],
                        &mut w.mixed[..rows * width],
                        rows,
                        &mut w.scatter,
                    )
                })?;
                for mut span in spans {
                    let LayerSeqState::Recurrent(state) = &mut *span.state else {
                        unreachable!("GDN layer must have recurrent sequence state");
                    };
                    let first = span.rows.start;
                    let end = span.rows.end;
                    let mixed = &w.mixed[first * width..end * width];
                    let (split_ms, copy_ms) = timings.stage(
                        Phase::Gdn,
                        "target GDN state",
                        || -> Result<(f64, f64)> {
                            let split_ms = if trace.enabled() {
                                let start = Instant::now();
                                let mut qkv = Vec::with_capacity(span.rows.len() * cw);
                                for row in mixed.chunks_exact(width) {
                                    qkv.extend_from_slice(&row[..cw]);
                                }
                                let split_ms = start.elapsed().as_secs_f64() * 1000.;
                                trace.save(span.position, index, "gdn_qkv", &qkv)?;
                                split_ms
                            } else {
                                0.
                            };
                            let input = Projected::from_merged(c.gdn, span.rows.len(), mixed)?;
                            let output = &mut w.attended[first * vw..end * vw];
                            if verify {
                                output.copy_from_slice(state.verify(
                                    &cpu.pool,
                                    weights,
                                    input,
                                    &mut w.recurrent,
                                )?);
                            } else {
                                state.forward_into(
                                    &cpu.pool,
                                    weights,
                                    input,
                                    &mut w.recurrent,
                                    output,
                                )?;
                            }
                            Ok((split_ms, 0.))
                        },
                    )?;
                    timings.record_gdn_io(split_ms, copy_ms);
                }
                vw
            }
            MixerWeights::Attention { q_norm, k_norm } => {
                let (q, kv, d) = (
                    c.num_attention_heads * c.head_dim,
                    c.num_key_value_heads * c.head_dim,
                    c.head_dim,
                );
                let width = 2 * q + 2 * kv;
                timings.stage(Phase::Linear, "target attention input projection", || {
                    inputs.run(
                        &w.normalized[..rows * h],
                        &mut w.mixed[..rows * width],
                        rows,
                        &mut w.scatter,
                    )
                })?;
                timings.stage(Phase::Cpu, "target QKV norm rope", || -> Result<()> {
                    for (t, row) in w.mixed[..rows * width].chunks(width).enumerate() {
                        attention::split_query_gate(
                            &row[..2 * q],
                            d,
                            &mut w.query[t * q..(t + 1) * q],
                            &mut w.gate[t * q..(t + 1) * q],
                        )?;
                        w.key[t * kv..(t + 1) * kv].copy_from_slice(&row[2 * q..2 * q + kv]);
                        w.value[t * kv..(t + 1) * kv].copy_from_slice(&row[2 * q + kv..]);
                    }
                    norm::rms_norm(
                        &w.query[..rows * q],
                        q_norm,
                        c.epsilon,
                        true,
                        &mut w.norm_temp[..rows * q],
                    )?;
                    w.query[..rows * q].copy_from_slice(&w.norm_temp[..rows * q]);
                    norm::rms_norm(
                        &w.key[..rows * kv],
                        k_norm,
                        c.epsilon,
                        true,
                        &mut w.norm_temp[..rows * kv],
                    )?;
                    w.key[..rows * kv].copy_from_slice(&w.norm_temp[..rows * kv]);
                    Ok(())
                })?;
                for span in spans {
                    let LayerSeqState::Attention(cache) = &*span.state else {
                        unreachable!("GQA layer must have attention sequence state");
                    };
                    let query = span.rows.start * q..span.rows.end * q;
                    let kv = span.rows.start * kv..span.rows.end * kv;
                    timings.stage(Phase::Cpu, "target QKV rope", || -> Result<()> {
                        c.rope.apply(
                            &mut w.query[query.clone()],
                            c.num_attention_heads,
                            span.rope,
                        )?;
                        c.rope
                            .apply(&mut w.key[kv.clone()], c.num_key_value_heads, span.rope)?;
                        trace.save(span.position, index, "query", &w.query[query.clone()])?;
                        trace.save(span.position, index, "key", &w.key[kv.clone()])?;
                        trace.save(span.position, index, "value", &w.value[kv.clone()])?;
                        Ok(())
                    })?;
                    timings.stage(Phase::Attention, "target attention", || {
                        cache.run(
                            &w.query[query.clone()],
                            &w.key[kv.clone()],
                            &w.value[kv],
                            &mut w.attended[query],
                            span.rows.len(),
                            span.position,
                        )
                    })?;
                }
                if verify && rows > 1 {
                    timings.record_attention_verification(rows);
                }
                timings.stage(Phase::Cpu, "target attention gate", || {
                    let output = &mut w.attended[..rows * q];
                    let gate = &w.gate[..rows * q];
                    if rows <= 1 || cpu.pool.current_num_threads() == 1 {
                        return activation::sigmoid_gate(output, gate);
                    }
                    cpu.pool.install(|| {
                        output
                            .par_chunks_mut(q)
                            .zip(gate.par_chunks(q))
                            .try_for_each(|(out, g)| activation::sigmoid_gate(out, g))
                    })
                })?;
                q
            }
        };
        Ok(attended_width)
    }
}
