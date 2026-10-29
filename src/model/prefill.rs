use crate::input::PreparedInput;
use std::ops::Range;

pub(crate) struct PrefillStep {
    pub range: Range<usize>,
    pub checkpoint: bool,
}

#[cfg(test)]
impl PrefillStep {
    pub fn chunks(tokens: usize, chunk: usize) -> Vec<Self> {
        (0..tokens)
            .step_by(chunk)
            .map(|start| Self {
                range: start..(start + chunk).min(tokens),
                checkpoint: false,
            })
            .collect()
    }
}

/// Raw input has no checkpoint plan; keep batch references on the stack.
pub(crate) fn batches(
    tokens: &[u32],
    chunk: usize,
    lanes: usize,
) -> impl Iterator<Item = (&[u32], Option<&[u32]>)> {
    let mut chunks = tokens.chunks(chunk).peekable();
    std::iter::from_fn(move || {
        let first = chunks.next()?;
        let pair = lanes == 2
            && first.len() == chunk
            && chunks.peek().is_some_and(|next| next.len() == chunk);
        Some((first, if pair { chunks.next() } else { None }))
    })
}

/// Pair complete adjacent batches, draining before every checkpoint boundary.
pub(crate) fn groups(
    steps: &[PrefillStep],
    chunk: usize,
    lanes: usize,
) -> impl Iterator<Item = &[PrefillStep]> {
    let mut offset = 0;
    std::iter::from_fn(move || {
        let first = steps.get(offset)?;
        let pair = lanes == 2
            && first.range.len() == chunk
            && steps.get(offset + 1).is_some_and(|next| {
                !next.checkpoint && next.range.len() == chunk && next.range.start == first.range.end
            });
        let end = offset + if pair { 2 } else { 1 };
        let result = &steps[offset..end];
        offset = end;
        Some(result)
    })
}

impl PrefillStep {
    /// Join adjacent physical batches within a logical block. A checkpoint
    /// always starts a new block so its save position cannot move.
    pub fn coalesce(steps: Vec<Self>, max_rows: usize) -> Vec<Self> {
        let mut result: Vec<Self> = Vec::new();
        for step in steps {
            if let Some(previous) = result.last_mut()
                && !step.checkpoint
                && previous.range.end == step.range.start
                && step.range.end - previous.range.start <= max_rows
            {
                previous.range.end = step.range.end;
            } else {
                result.push(step);
            }
        }
        result
    }
}

/// One prepared forward. The input stays borrowed until its plan is consumed.
pub struct PreparedPrefill<'a> {
    pub(crate) input: &'a PreparedInput,
    pub(crate) start: usize,
    pub(crate) steps: Vec<PrefillStep>,
    pub(crate) checkpoint: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_batches_pair_only_full_chunks() {
        let tokens: Vec<_> = (0..641).collect();
        for count in [0, 1, 127, 128, 129, 255, 256, 257, 384, 513, 641] {
            for lanes in [1, 2] {
                let mut actual = Vec::new();
                for (first, second) in batches(&tokens[..count], 128, lanes) {
                    actual.extend_from_slice(first);
                    if let Some(second) = second {
                        assert_eq!(lanes, 2);
                        assert_eq!((first.len(), second.len()), (128, 128));
                        actual.extend_from_slice(second);
                    }
                }
                assert_eq!(actual, tokens[..count]);
            }
        }
        assert_eq!(
            batches(&tokens, 128, 2)
                .map(|(a, b)| (a.len(), b.map(<[_]>::len)))
                .collect::<Vec<_>>(),
            [(128, Some(128)), (128, Some(128)), (128, None), (1, None)]
        );
    }
    #[test]
    fn pairing_preserves_tail_and_checkpoint_boundaries() {
        let mut steps = PrefillStep::chunks(641, 128);
        steps[1].checkpoint = true;
        let lengths: Vec<_> = groups(&steps, 128, 2)
            .map(|group| {
                group
                    .iter()
                    .map(|step| step.range.len())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(
            lengths,
            [vec![128], vec![128, 128], vec![128, 128], vec![1]]
        );
        assert_eq!(groups(&steps, 128, 1).count(), steps.len());
    }

    #[test]
    fn logical_streaming_blocks_preserve_checkpoints_and_physical_tail() {
        let mut steps = PrefillStep::chunks(1409, 128);
        steps[1].checkpoint = true;
        steps[6].checkpoint = true;
        let steps = PrefillStep::coalesce(steps, 1024);
        let plan: Vec<_> = groups(&steps, 128, 1)
            .map(|group| {
                assert_eq!(group.len(), 1);
                (group[0].range.clone(), group[0].checkpoint)
            })
            .collect();
        assert_eq!(plan, [(0..128, false), (128..768, true), (768..1409, true)]);
        assert_eq!((steps.last().unwrap().range.len() - 1) % 128 + 1, 1);
    }
}
