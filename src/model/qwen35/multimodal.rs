use super::runtime::Qwen35;
use super::vision::Encoder;
use crate::model::{LanguageModel, PreparedPrefill};
use anyhow::Context;
use anyhow::{Result, ensure};
use std::sync::atomic::{AtomicBool, Ordering};

pub(super) struct EmbeddedChunk<'a> {
    pub positions: &'a [[usize; 3]],
    pub features: &'a [(usize, std::sync::Arc<[f32]>)],
}

impl Qwen35 {
    pub(super) fn consume_prepared(
        &mut self,
        prepared: PreparedPrefill<'_>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let PreparedPrefill {
            input,
            start,
            steps,
            checkpoint,
        } = prepared;
        input.validate()?;
        ensure!(
            self.round.is_none(),
            "Qwen3.5 verification transaction is active"
        );
        ensure!(
            start == self.position
                && start < input.tokens.len()
                && input.positions.len() == input.tokens.len(),
            "prepared input position mismatch"
        );
        self.check_input(&input.tokens[start..])?;
        ensure!(
            checkpoint == self.prefix_cache.last_position()
                && steps.first().is_some_and(|s| s.range.start == start)
                && steps
                    .last()
                    .is_some_and(|s| s.range.end == input.tokens.len()),
            "prepared prefill no longer matches the prefix cache"
        );
        self.reserve(input.tokens.len())?;
        let result = (|| {
            let mut features = Vec::new();
            for span in input.images.iter().filter(|s| s.end() > start) {
                ensure!(!cancel.load(Ordering::Relaxed), "generation cancelled");
                if self.vision.is_none() {
                    let domain = 14u32
                        .checked_sub(self.backends.len() as u32)
                        .context("no free IOMMU domain for vision")?;
                    self.vision = Some(Encoder::load_cancellable(
                        &self.options,
                        &self.loader.store,
                        domain,
                        Some(cancel),
                    )?);
                }
                let feature =
                    self.vision
                        .as_mut()
                        .unwrap()
                        .encode(&span.image, &self.cpu.pool, cancel)?;
                ensure!(
                    feature.len() == span.image.tokens() * self.config.hidden_size,
                    "vision/text hidden size mismatch"
                );
                features.push((span.start, feature));
            }
            let mut rows = 0;
            self.prefix_cache.begin_request();
            for step in steps {
                let (offset, end) = (step.range.start, step.range.end);
                if step.checkpoint {
                    self.save_prefix_checkpoint()?;
                }
                self.prompt_chunk(
                    &input.tokens[offset..end],
                    Some(cancel),
                    Some(EmbeddedChunk {
                        positions: &input.positions[offset..end],
                        features: &features,
                    }),
                )?;
                rows = (end - offset - 1) % self.options.chunk + 1;
            }
            ensure!(!cancel.load(Ordering::Relaxed), "generation cancelled");
            self.next_rope_position = input.next_position;
            self.output_head(rows)
        })();
        if result.is_err() {
            self.reset()?;
        }
        result
    }
}
