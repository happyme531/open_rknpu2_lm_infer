use super::{Dense, LayerPass, elapsed};
use crate::model::{
    ModelInfo,
    batch::{
        self, Admission, BatchCapabilities, BatchModel, SeqId, Span, StepLayout, StepOutput,
        StepResult,
    },
};
use anyhow::Result;
use serde_json::Value;
use std::time::Instant;

impl BatchModel for Dense {
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
            |s| s.position,
        )?;
        let execution = self.step(&layout);
        self.sequences.complete_step(
            layout,
            execution,
            &self.work.batch_logits,
            self.info.vocab_size,
            |state, span| state.position += span.tokens.len(),
        )
    }

    fn stats(&self) -> Value {
        let mut stats = batch::memory_stats(
            crate::model::metrics::backend_stats(&self.backends, self.weights.head.backend()),
            self.sequences.states().map(|s| {
                (
                    s.caches.iter().map(|cache| cache.batch_memory()).collect(),
                    0,
                )
            }),
            self.sequences.reset_stats(),
        );
        stats["cpu_scratch_bytes"] = serde_json::json!(
            self.work
                .batches
                .iter()
                .map(super::Batch::bytes)
                .sum::<usize>()
                + 4 * self.work.batch_logits.capacity()
        );
        stats
    }
}

impl Dense {
    fn step(&mut self, layout: &StepLayout<'_, '_>) -> Result<()> {
        let weights = &self.weights;
        let work = &mut self.work;
        let b = &mut work.batches[0];
        let g = weights.geometry;
        let start = Instant::now();
        b.rows = layout.rows;
        b.position = layout.spans[0].span.start;
        for placed in &layout.spans {
            b.embed_at(
                placed.rows.start,
                placed.span.tokens,
                &weights.embedding,
                &work.cpu,
                g,
            );
        }
        work.timings.cpu += elapsed(start);
        let pass = LayerPass {
            cpu: &work.cpu,
            trace: &weights.trace,
            cosine: &weights.cosine,
            sine: &weights.sine,
            geometry: g,
        };
        for (i, layer) in weights.layers.iter().enumerate() {
            let inputs = layout.spans.iter().map(|placed| {
                let cache = &self.sequences.state(placed).caches[i];
                (placed.rows.clone(), placed.span.start, cache)
            });
            pass.run(&layer, i as i32, b, &mut work.timings, inputs)?;
        }
        if self.options.compress_w8a8 {
            crate::npu::release_compression_scratch(&self.backends);
        }
        let start = Instant::now();
        for (index, placed) in layout.head_rows().enumerate() {
            let row = placed.rows.end - 1;
            let out = index * g.h;
            work.cpu.norm(
                &b.hidden[row * g.h..(row + 1) * g.h],
                &mut b.normalized[out..out + g.h],
                &weights.final_norm,
                g.epsilon,
            );
        }
        let head_rows = layout.head_rows().count();
        work.batch_logits
            .resize(head_rows * self.info.vocab_size, 0.0);
        if head_rows != 0 {
            weights.head.run(
                &work.cpu,
                &b.normalized[..head_rows * g.h],
                &mut work.batch_logits,
                head_rows,
            )?;
            if weights.logits_scaling != 1.0 {
                for value in &mut work.batch_logits {
                    *value /= weights.logits_scaling;
                }
            }
        }
        work.timings.head += elapsed(start);
        Ok(())
    }
}
