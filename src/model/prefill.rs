use crate::input::PreparedInput;
use std::ops::Range;

pub(crate) struct PrefillStep {
    pub range: Range<usize>,
    pub checkpoint: bool,
}

/// One prepared forward. The input stays borrowed until its plan is consumed.
pub struct PreparedPrefill<'a> {
    pub(crate) input: &'a PreparedInput,
    pub(crate) start: usize,
    pub(crate) steps: Vec<PrefillStep>,
    pub(crate) checkpoint: Option<usize>,
}
