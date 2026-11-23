use super::{
    decoder::{Execution, LayerSeqState, MixerSpan},
    runtime::Qwen35,
    timings::Phase,
};
use crate::{
    model::{
        ModelInfo,
        batch::{
            self, Admission, BatchCapabilities, BatchModel, SeqId, Span, StepLayout, StepOutput,
            StepResult,
        },
    },
    ops::norm,
};
use anyhow::Result;
use serde_json::Value;
use std::time::Instant;

impl BatchModel for Qwen35 {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> BatchCapabilities {
        self.sequences.capabilities(self.options.chunk)
    }

    fn try_admit(&mut self, total_tokens: usize) -> Admission {
        self.sequences.try_admit(total_tokens)
    }

    fn release(&mut self, seq: SeqId) -> Result<()> {
        self.sequences
            .release(seq, |state| state.reset(|cache| cache.truncate(0)))
    }

    fn execute_step(&mut self, spans: &[Span<'_>]) -> StepResult<StepOutput<'_>> {
        let layout = self.sequences.layout(
            spans,
            self.options.chunk,
            self.info.vocab_size,
            |s| s.position,
            |s| s.next_rope_position,
        )?;
        let execution = self.step(&layout);
        self.sequences.complete_step(
            layout,
            execution,
            &self.logits,
            self.info.vocab_size,
            |state, span| state.advance(span.positions),
        )
    }

    fn stats(&self) -> Value {
        let mut stats = batch::memory_stats(
            crate::model::metrics::backend_stats(&self.backends, self.weights.head.backend()),
            self.sequences.states().map(|seq| {
                let mut caches = Vec::new();
                let mut bytes = 0;
                for layer in seq.mixers.iter() {
                    match &*layer {
                        LayerSeqState::Attention(cache) => caches.push(cache.batch_memory()),
                        LayerSeqState::Recurrent(state) => {
                            let mem = state.memory_usage();
                            bytes += mem.canonical_state_bytes
                                + mem.convolution_history_bytes
                                + mem.verification_state_bytes
                                + mem.update_log_bytes;
                        }
                    }
                }
                (caches, bytes)
            }),
            self.sequences.reset_stats(),
        );
        stats["gdn_workspace"] = self.work.recurrent.backend_stats();
        stats["cpu_scratch_bytes"] =
            serde_json::json!(self.work.allocated_bytes() + self.logits.capacity() * 4);
        stats
    }
}

impl Qwen35 {
    fn step(&mut self, layout: &StepLayout<'_, '_>) -> Result<()> {
        let weights = &mut self.weights;
        let work = &mut self.work;
        let h = weights.config.hidden_size;
        self.phases.measure(Phase::Cpu, || {
            let tokens = layout.spans.iter().flat_map(|placed| placed.span.tokens);
            for (&token, row) in tokens.zip(work.hidden[..layout.rows * h].chunks_exact_mut(h)) {
                weights.embedding.row(token as usize, row);
            }
        });
        for (index, layer) in weights.layers.iter_mut().enumerate() {
            let e = Execution {
                config: &weights.config,
                cpu: &weights.cpu,
                rows: layout.rows,
                position: layout.spans[0].span.start,
                index: index as i32,
                trace: &weights.trace,
                rope_positions: &[],
                verify: false,
            };
            let inputs = layout.spans.iter().map(|placed| MixerSpan {
                state: self.sequences.state(placed).mixers.lock(index),
                rows: placed.rows.clone(),
                position: placed.span.start,
                rope: placed.span.positions,
            });
            layer.forward_spans(work, &e, &mut self.phases, weights.paging.as_ref(), inputs)?;
        }
        if self.options.compress_w8a8 {
            crate::npu::release_compression_scratch(&self.backends);
        }
        let start = Instant::now();
        for (index, placed) in layout.head_rows().enumerate() {
            let row = placed.rows.end - 1;
            let out = index * h;
            norm::rms_norm(
                &work.hidden[row * h..(row + 1) * h],
                &weights.final_norm,
                weights.config.epsilon,
                true,
                &mut work.normalized[out..out + h],
            )?;
        }
        let head_rows = layout.head_rows().count();
        self.logits.resize(head_rows * self.info.vocab_size, 0.);
        if head_rows != 0 {
            weights.head.run(
                &weights.cpu,
                &work.normalized[..head_rows * h],
                &mut self.logits,
                head_rows,
            )?;
        }
        self.head_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
}
