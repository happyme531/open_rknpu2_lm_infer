//! Reusable layers composed from numerical operators, without model loading.
#[cfg(feature = "arch-qwen35")]
pub mod gated_delta;

#[cfg(feature = "vision")]
pub mod layer_norm;
