//! Checkpoint scheduling from llama.cpp server-context.cpp at 7894e8b92481.
use crate::{
    input::PreparedInput,
    model::{CheckpointOptions, PrefillStep},
};

pub(super) fn plan(
    input: &PreparedInput,
    start: usize,
    ubatch: usize,
    settings: CheckpointOptions,
    mut last_checkpoint: Option<usize>,
) -> Vec<PrefillStep> {
    let length = input.tokens.len();
    let enabled = settings.capacity > 0;
    let last_user = input.user_starts.last().copied();
    let mut result = Vec::new();
    let mut position = start;
    while position < length {
        // This serial engine submits one logical prompt at a time. Split its
        // logical batches at the same points as llama-server, then microbatch.
        let mut end = length;
        if enabled {
            for &user in &input.user_starts {
                if user > position
                    && (Some(user) == last_user
                        || last_checkpoint.is_none()
                        || user > last_checkpoint.unwrap() + settings.min_step)
                {
                    end = end.min(user);
                }
            }
            for offset in [4 + ubatch, 4] {
                if let Some(boundary) = length.checked_sub(offset)
                    && boundary > position
                {
                    end = end.min(boundary);
                }
            }
        }
        let near_end = length < end + ubatch;
        let is_user = input.user_starts.contains(&position);
        let is_last_user = Some(position) == last_user;
        let has_image = input
            .images
            .iter()
            .any(|s| s.start < end && s.end() > position);
        let checkpoint = enabled
            && position > 0
            && !has_image
            && (end == length || is_user || near_end)
            && (last_checkpoint.is_none()
                || is_last_user
                || near_end
                || position > last_checkpoint.unwrap() + settings.min_step);
        if checkpoint {
            last_checkpoint = Some(position);
        }
        for offset in (position..end).step_by(ubatch) {
            result.push(PrefillStep {
                range: offset..(offset + ubatch).min(end),
                checkpoint: checkpoint && offset == position,
            });
        }
        position = end;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn checkpoints(input: &PreparedInput, settings: CheckpointOptions) -> Vec<usize> {
        plan(input, 0, 128, settings, None)
            .into_iter()
            .filter(|s| s.checkpoint)
            .map(|s| s.range.start)
            .collect()
    }
    #[test]
    fn saves_before_last_user_and_tail_batches_without_per_ubatch_snapshots() {
        let mut input = PreparedInput::text(vec![1; 600]);
        let settings = CheckpointOptions::default();
        assert_eq!(checkpoints(&input, settings), [468, 596]);
        input.user_starts = vec![20, 200, 350];
        assert_eq!(checkpoints(&input, settings), [20, 350, 468, 596]);
        input = PreparedInput::text(vec![1; 80]);
        input.user_starts = vec![20];
        assert_eq!(checkpoints(&input, settings), [20, 76]);
        assert!(
            checkpoints(
                &input,
                CheckpointOptions {
                    capacity: 0,
                    ..settings
                }
            )
            .is_empty()
        );
    }
    #[test]
    fn multiple_user_and_tail_boundaries_prepare_every_physical_row_shape() {
        let mut input = PreparedInput::text(vec![1; 600]);
        input.user_starts = vec![20, 350];
        let steps = plan(&input, 0, 128, CheckpointOptions::default(), None);
        assert_eq!(
            steps.iter().map(|s| s.range.len()).collect::<Vec<_>>(),
            [20, 128, 128, 74, 118, 128, 4]
        );
    }
    #[test]
    fn intermediate_user_boundaries_obey_minimum_spacing() {
        let mut input = PreparedInput::text(vec![1; 20_000]);
        input.user_starts = vec![20, 200, 9000, 15_000];
        assert_eq!(
            checkpoints(&input, CheckpointOptions::default()),
            [20, 9000, 15_000, 19_868, 19_996]
        );
    }
}
