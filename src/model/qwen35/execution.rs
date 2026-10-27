//! Per-batch buffers and model bookkeeping around the shared layer pipeline.
use super::{
    decoder::{Buffers, Execution},
    multimodal::EmbeddedChunk,
    runtime::Qwen35,
    timings::{Phase, Timings},
};
use crate::{
    model::{checkpoint::Tensor, pipeline},
    ops::norm,
};
use anyhow::{Result, ensure};
use std::{
    borrow::Cow,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

fn embed<'a>(
    embedding: &Tensor,
    work: &mut Buffers,
    tokens: &[u32],
    position: usize,
    rope_start: usize,
    prepared: Option<EmbeddedChunk<'a>>,
) -> Result<Cow<'a, [[usize; 3]]>> {
    let h = embedding.shape[1];
    for (row, token) in tokens.iter().enumerate() {
        embedding.row(*token as usize, &mut work.hidden[row * h..(row + 1) * h]);
    }
    if let Some(EmbeddedChunk {
        positions,
        features,
    }) = prepared
    {
        ensure!(
            positions.len() == tokens.len(),
            "prepared position shape mismatch"
        );
        for (start, values) in features {
            let first = position.max(*start);
            let last = (position + tokens.len()).min(start + values.len() / h);
            if first < last {
                work.hidden[(first - position) * h..(last - position) * h]
                    .copy_from_slice(&values[(first - start) * h..(last - start) * h]);
            }
        }
        Ok(Cow::Borrowed(positions))
    } else {
        Ok(Cow::Owned(
            (rope_start..rope_start + tokens.len())
                .map(|p| [p; 3])
                .collect(),
        ))
    }
}

impl Qwen35 {
    pub(super) fn execute(
        &mut self,
        inputs: &[(&[u32], Option<EmbeddedChunk<'_>>)],
        cancelled: Option<&AtomicBool>,
        verify: bool,
    ) -> Result<()> {
        ensure!(
            (1..=2).contains(&inputs.len()) && (inputs.len() == 1 || !verify),
            "invalid decoder group"
        );
        let start = Instant::now();
        let (tokens, prepared) = inputs[0];
        let rows = tokens.len();
        let position = self.position;
        let rope = self.phases.measure(Phase::Cpu, || {
            embed(
                &self.embedding,
                &mut self.work,
                tokens,
                position,
                self.next_rope_position,
                prepared,
            )
        })?;
        let embeddings = self
            .mtp
            .as_ref()
            .filter(|_| !verify)
            .map(|_| self.work.hidden[..rows * self.config.hidden_size].to_vec());
        self.features.begin(rows);
        let mut other_timings = Timings::default();
        let mut other_rope = Cow::Borrowed(&[][..]);
        let mut other_embeddings = None;
        let execution = Execution {
            config: &self.config,
            cpu: &self.cpu,
            rows,
            position,
            index: 0,
            trace: &self.trace,
            rope_positions: &rope,
            verify,
        };
        if inputs.len() == 2 {
            let (tokens, prepared) = inputs[1];
            let work = self.secondary_work.as_mut().expect("prepared second lane");
            other_rope = self.phases.measure(Phase::Cpu, || {
                embed(
                    &self.embedding,
                    work,
                    tokens,
                    position + rows,
                    self.next_rope_position + rows,
                    prepared,
                )
            })?;
            other_embeddings = self
                .mtp
                .as_ref()
                .map(|_| work.hidden[..tokens.len() * self.config.hidden_size].to_vec());
            let other_features = self
                .secondary_features
                .as_mut()
                .expect("prepared second feature buffer");
            other_features.begin(tokens.len());
            let next = Execution {
                rows: tokens.len(),
                position: position + rows,
                rope_positions: &other_rope,
                ..execution
            };
            let lane = self.secondary.as_ref().expect("prepared second lane");
            let first = &mut self.work;
            let features = &mut self.features;
            let timings = &mut self.phases;
            pipeline::run(
                &self.layers,
                cancelled,
                |i, layer| {
                    layer.forward(
                        first,
                        &Execution {
                            index: i as i32,
                            ..execution
                        },
                        timings,
                    )?;
                    features.record(i, &first.hidden[..rows * self.config.hidden_size]);
                    Ok(())
                },
                |i, layer| {
                    lane.run(|| {
                        layer.forward(
                            work,
                            &Execution {
                                index: i as i32,
                                ..next
                            },
                            &mut other_timings,
                        )?;
                        other_features
                            .record(i, &work.hidden[..tokens.len() * self.config.hidden_size]);
                        Ok(())
                    })
                },
            )?;
            self.phases.merge(other_timings);
        } else {
            for (i, layer) in self.layers.iter_mut().enumerate() {
                ensure!(
                    !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
                    "generation cancelled"
                );
                layer.forward(
                    &mut self.work,
                    &Execution {
                        index: i as i32,
                        ..execution
                    },
                    &mut self.phases,
                )?;
                self.features
                    .record(i, &self.work.hidden[..rows * self.config.hidden_size]);
            }
        }
        if !verify {
            self.finish_chunk(rows, &rope, embeddings.as_deref(), false, cancelled)?;
            if inputs.len() == 2 {
                // Only the result crosses lanes. Keep GDN scratch and its
                // prepared row set attached to their original execution lane.
                std::mem::swap(
                    &mut self.work.hidden,
                    &mut self.secondary_work.as_mut().unwrap().hidden,
                );
                self.finish_chunk(
                    inputs[1].0.len(),
                    &other_rope,
                    other_embeddings.as_deref(),
                    true,
                    cancelled,
                )?;
                self.dual_groups += 1;
            }
        }
        if self.options.compress_w8a8 {
            crate::npu::release_compression_scratch(&self.backends);
        }
        self.forward_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }

    fn finish_chunk(
        &mut self,
        rows: usize,
        rope: &[[usize; 3]],
        embeddings: Option<&[f32]>,
        secondary: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        if let (Some(mtp), Some(embeddings)) = (&mut self.mtp, embeddings) {
            let h = self.config.hidden_size;
            norm::rms_norm(
                &self.work.hidden[..rows * h],
                &self.final_norm,
                self.config.epsilon,
                true,
                &mut self.work.normalized[..rows * h],
            )?;
            mtp.catch_up(
                &self.config,
                &self.cpu,
                embeddings,
                &self.work.normalized[..rows * h],
                rope,
                &self.trace,
            )?;
        }
        if secondary {
            self.features
                .commit_pending(self.secondary_features.as_mut().unwrap(), rows);
        } else {
            self.features.commit(rows);
        }
        self.flush_features(cancelled)?;
        self.position += rows;
        self.next_rope_position = rope
            .iter()
            .flatten()
            .copied()
            .max()
            .unwrap_or(self.next_rope_position)
            + 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
