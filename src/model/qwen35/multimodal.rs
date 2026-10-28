use super::{runtime::Qwen35, vision::Encoder};
use crate::model::{LanguageModel, PreparedPrefill};
use anyhow::{Context, Result, ensure};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy)]
pub(super) struct EmbeddedChunk<'a> {
    pub positions: &'a [[usize; 3]],
    pub features: &'a [(usize, std::sync::Arc<[f32]>)],
}

impl Qwen35 {
    pub(super) fn consume_prepared(
        &mut self,
        prepared: PreparedPrefill<'_>,
        cancel: &AtomicBool,
        mut progress: Option<&mut dyn FnMut(crate::progress::ModelProgress)>,
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
            let images = input.images.iter().filter(|s| s.end() > start).count();
            for (index, span) in input.images.iter().filter(|s| s.end() > start).enumerate() {
                if let Some(progress) = progress.as_mut() {
                    progress(crate::progress::ModelProgress::Vision {
                        completed: index,
                        total: images,
                    });
                }
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
            if let Some(progress) = progress.as_mut() {
                progress(crate::progress::ModelProgress::Prefill {
                    completed: 0,
                    total: input.tokens.len() - start,
                });
            }
            self.prefix_cache.begin_request();
            for group in crate::model::prefill::groups(
                &steps,
                self.options.chunk,
                self.options.prefill_ubatches,
            ) {
                if group[0].checkpoint {
                    self.save_prefix_checkpoint()?;
                }
                let mut inputs = [(&[][..], None); 2];
                for (input_batch, step) in inputs.iter_mut().zip(group) {
                    *input_batch = (
                        &input.tokens[step.range.clone()],
                        Some(EmbeddedChunk {
                            positions: &input.positions[step.range.clone()],
                            features: &features,
                        }),
                    );
                }
                crate::progress::prefill_shape(group[0].range.len() != self.options.chunk, || {
                    self.execute(&inputs[..group.len()], Some(cancel), false)
                })?;
                rows = group.last().unwrap().range.len();
                let completed = group.last().unwrap().range.end - start;
                let total = input.tokens.len() - start;
                if completed < total {
                    if let Some(progress) = progress.as_mut() {
                        progress(crate::progress::ModelProgress::Prefill { completed, total });
                    }
                }
            }
            ensure!(!cancel.load(Ordering::Relaxed), "generation cancelled");
            self.next_rope_position = input.next_position;
            self.output_head(rows)?;
            if let Some(progress) = progress.as_mut() {
                let total = input.tokens.len() - start;
                progress(crate::progress::ModelProgress::Prefill {
                    completed: total,
                    total,
                });
            }
            Ok(())
        })();
        if result.is_err() {
            self.reset()?;
        }
        result
    }
}
