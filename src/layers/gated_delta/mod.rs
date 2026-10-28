//! Gated delta layer between input/output projections: causal convolution,
//! normalized recurrence, output gating, and per-layer transactional state.
//! Weights and projected activations are supplied by the model adapter.
mod mixer;
mod state;

pub use mixer::{MixerWeights, Projected};
pub use state::{LayerState, MemoryUsage, Scratch, Snapshot};
