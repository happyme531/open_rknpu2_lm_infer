//! Layer-wise logical prompt chunks, adapted from Strata's staged expert walk
//! (`src/prefill/prefill.cpp`, cbfd5b926b794b356195d3cb600191b0c3567540).
//! Mixer kernels retain their physical ubatch; routed experts see the complete
//! logical chunk, so an expert is loaded once for all its assigned rows.
//! See ../experts/Strata-LICENSE for the upstream MIT notice.
use super::{
    decoder::{Execution, FeedForward},
    multimodal::EmbeddedChunk,
    runtime::Qwen35,
    timings::Phase,
};
use anyhow::{Result, ensure};
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

impl Qwen35 {
    pub(super) fn prompt_chunk_rows(&self) -> usize {
        if self.paging.is_some() && self.mtp.is_none() {
            self.options.chunk.max(1024)
        } else {
            self.options.chunk
        }
    }

    pub(super) fn prompt_chunk(
        &mut self,
        tokens: &[u32],
        cancel: Option<&AtomicBool>,
        prepared: Option<EmbeddedChunk<'_>>,
    ) -> Result<()> {
        if self.paging.is_none() || self.mtp.is_some() || tokens.len() <= self.options.chunk {
            return self.chunk(tokens, cancel, prepared);
        }
        let start = Instant::now();
        let h = self.config.hidden_size;
        let rows = tokens.len();
        let position = self.position;
        let mut hidden = vec![0.; rows * h];
        self.phases.measure(Phase::Cpu, || {
            for (&token, dest) in tokens.iter().zip(hidden.chunks_exact_mut(h)) {
                self.embedding.row(token as usize, dest);
            }
        });
        let positions;
        let rope_positions = if let Some(EmbeddedChunk {
            positions,
            features,
        }) = prepared
        {
            ensure!(
                positions.len() == rows,
                "prepared prompt position shape mismatch"
            );
            for (first, embedding) in features {
                let begin = position.max(*first);
                let end = (position + rows).min(first + embedding.len() / h);
                if begin < end {
                    hidden[(begin - position) * h..(end - position) * h]
                        .copy_from_slice(&embedding[(begin - first) * h..(end - first) * h]);
                }
            }
            positions
        } else {
            positions = (self.next_rope_position..self.next_rope_position + rows)
                .map(|p| [p; 3])
                .collect::<Vec<_>>();
            &positions
        };
        let mut normalized = vec![0.; rows * h];
        let mut routed = vec![0.; rows * h];
        self.features.begin(rows);
        for (index, layer) in self.layers.iter_mut().enumerate() {
            for offset in (0..rows).step_by(self.options.chunk) {
                ensure!(
                    !cancel.is_some_and(|c| c.load(Ordering::Relaxed)),
                    "generation cancelled"
                );
                let end = (offset + self.options.chunk).min(rows);
                let count = end - offset;
                self.work.hidden[..count * h].copy_from_slice(&hidden[offset * h..end * h]);
                layer.mix(
                    &mut self.work,
                    &Execution {
                        config: &self.config,
                        cpu: &self.cpu,
                        rows: count,
                        position: position + offset,
                        index: index as i32,
                        trace: &self.trace,
                        rope_positions: &rope_positions[offset..end],
                        verify: false,
                    },
                    &mut self.phases,
                )?;
                hidden[offset * h..end * h].copy_from_slice(&self.work.hidden[..count * h]);
                normalized[offset * h..end * h].copy_from_slice(&self.work.normalized[..count * h]);
            }
            let FeedForward::Routed(moe) = &mut layer.ffn else {
                unreachable!("streamed expert model has routed layers")
            };
            ensure!(
                !cancel.is_some_and(|c| c.load(Ordering::Relaxed)),
                "generation cancelled"
            );
            moe.routed(
                &normalized,
                &mut routed,
                &Execution {
                    config: &self.config,
                    cpu: &self.cpu,
                    rows,
                    position,
                    index: index as i32,
                    trace: &self.trace,
                    rope_positions,
                    verify: false,
                },
                &mut self.phases,
            )?;
            for offset in (0..rows).step_by(self.options.chunk) {
                ensure!(
                    !cancel.is_some_and(|c| c.load(Ordering::Relaxed)),
                    "generation cancelled"
                );
                let end = (offset + self.options.chunk).min(rows);
                let count = end - offset;
                self.work.normalized[..count * h].copy_from_slice(&normalized[offset * h..end * h]);
                self.work.projected[..count * h].copy_from_slice(&routed[offset * h..end * h]);
                moe.shared(
                    &mut self.work,
                    &Execution {
                        config: &self.config,
                        cpu: &self.cpu,
                        rows: count,
                        position: position + offset,
                        index: index as i32,
                        trace: &self.trace,
                        rope_positions: &rope_positions[offset..end],
                        verify: false,
                    },
                    &mut self.phases,
                )?;
                self.trace.save(
                    position + offset,
                    index as i32,
                    "ffn",
                    &self.work.projected[..count * h],
                )?;
                self.phases.measure(Phase::Cpu, || {
                    self.cpu.add(
                        &mut hidden[offset * h..end * h],
                        &self.work.projected[..count * h],
                    );
                });
                self.work.hidden[..count * h].copy_from_slice(&hidden[offset * h..end * h]);
                self.trace.save(
                    position + offset,
                    index as i32,
                    "output",
                    &self.work.hidden[..count * h],
                )?;
            }
            self.features.record(index, &hidden);
        }
        self.features.commit(rows);
        self.flush_features(cancel)?;
        self.position += rows;
        self.next_rope_position = rope_positions
            .iter()
            .flatten()
            .copied()
            .max()
            .unwrap_or(self.next_rope_position)
            + 1;
        self.forward_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
}
