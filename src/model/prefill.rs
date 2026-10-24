use crate::input::PreparedInput;
#[cfg(any(feature = "arch-qwen35", feature = "model-ling3tiny"))]
use std::ops::Range;

#[cfg(any(feature = "arch-qwen35", feature = "model-ling3tiny"))]
pub(crate) struct PrefillStep {
    pub range: Range<usize>,
    pub checkpoint: bool,
}

/// One prepared forward. The input stays borrowed until its plan is consumed.
pub struct PreparedPrefill<'a> {
    pub(crate) input: &'a PreparedInput,
    pub(crate) start: usize,
    #[cfg(any(feature = "arch-qwen35", feature = "model-ling3tiny"))]
    pub(crate) steps: Vec<PrefillStep>,
    #[cfg(any(feature = "arch-qwen35", feature = "model-ling3tiny"))]
    pub(crate) checkpoint: Option<usize>,
}
