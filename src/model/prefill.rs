use crate::input::PreparedInput;
use std::ops::Range;

pub(crate) struct PrefillStep {
    pub range: Range<usize>,
    pub checkpoint: bool,
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
