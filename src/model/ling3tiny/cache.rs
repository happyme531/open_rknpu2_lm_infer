//! KDA and convolution checkpoints; absorbed MLA KV remains resident.
use super::*;
use crate::{input::PreparedInput, model::PreparedPrefill};

pub(super) struct Checkpoint {
    recurrent: Vec<mixer::Snapshot>,
}
pub(super) type PrefixCache = crate::model::prefix_cache::PrefixCache<Checkpoint>;

impl Ling3Tiny {
    fn save_prefix_checkpoint(&mut self) -> Result<()> {
        if self.position == 0 || self.prefix_cache.settings.capacity == 0 {
            return Ok(());
        }
        if self.prefix_cache.reuse(self.position) {
            return Ok(());
        }
        let start = Instant::now();
        self.prefix_cache.make_room();
        let recurrent: Vec<_> = self
            .layers
            .iter()
            .filter_map(|l| l.mixer.snapshot())
            .collect();
        let bytes = recurrent.iter().map(mixer::Snapshot::bytes).sum();
        self.prefix_cache.insert(
            self.position,
            Checkpoint { recurrent },
            bytes,
            start.elapsed().as_secs_f64() * 1000.,
        );
        Ok(())
    }

    pub(super) fn restore_cached_prefix(&mut self, maximum: usize) -> Result<usize> {
        ensure!(maximum <= self.position, "cannot extend Ling prefix");
        if maximum == self.position {
            return Ok(maximum);
        }
        let Some(checkpoint) = self
            .prefix_cache
            .entries
            .iter()
            .rev()
            .find(|c| c.position <= maximum)
        else {
            return Ok(0);
        };
        let position = checkpoint.position;
        let result = (|| -> Result<()> {
            let mut snapshots = checkpoint.state.recurrent.iter();
            for layer in &mut self.layers {
                let snapshot = match &layer.mixer {
                    mixer::Mixer::Kda(_) => snapshots.next(),
                    mixer::Mixer::Mla(_) => None,
                };
                layer.mixer.restore(snapshot, position)?;
            }
            self.position = position;
            self.last_hidden.clear();
            self.logits.clear();
            Ok(())
        })();
        if let Err(error) = result {
            self.reset()?;
            return Err(error);
        }
        // A new request may overwrite every discarded KV row.
        self.prefix_cache.entries.retain(|c| c.position <= position);
        self.prefix_cache.hits += 1;
        Ok(position)
    }

    pub(super) fn prepare_prompt<'a>(
        &mut self,
        input: &'a PreparedInput,
        start: usize,
    ) -> Result<PreparedPrefill<'a>> {
        input.validate()?;
        ensure!(input.images.is_empty(), "Ling does not support images");
        ensure!(
            start == self.position
                && start < input.tokens.len()
                && input.tokens.len() <= self.options.context,
            "invalid Ling prepared prompt"
        );
        let prepared = PreparedPrefill {
            input,
            start,
            steps: self.prefix_cache.plan(input, start, self.options.chunk),
            checkpoint: self.prefix_cache.last_position(),
        };
        self.prepare_rows(prepared.steps.iter().map(|s| s.range.len()).collect())?;
        Ok(prepared)
    }

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
        ensure!(
            start == self.position
                && start < input.tokens.len()
                && checkpoint == self.prefix_cache.last_position()
                && steps.first().is_some_and(|s| s.range.start == start)
                && steps
                    .last()
                    .is_some_and(|s| s.range.end == input.tokens.len()),
            "Ling prepared prefill no longer matches the prefix cache"
        );
        self.check_input(&input.tokens[start..])?;
        self.reserve(input.tokens.len())?;
        let result = (|| -> Result<()> {
            self.prefix_cache.begin_request();
            for step in steps {
                ensure!(!cancel.load(Ordering::Relaxed), "generation cancelled");
                if step.checkpoint {
                    self.save_prefix_checkpoint()?;
                }
                self.chunk(&input.tokens[step.range], Some(cancel))?;
            }
            ensure!(!cancel.load(Ordering::Relaxed), "generation cancelled");
            self.output()?;
            Ok(())
        })();
        if result.is_err() {
            self.reset()?;
        }
        result
    }
}

#[cfg(test)]
mod tests;
