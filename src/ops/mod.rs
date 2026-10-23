//! Model-independent CPU numerical operators. Callers supply dimensions and
//! already-resolved parameters; checkpoint names and config defaults stay in models.
pub mod activation;
pub mod causal_conv;
#[cfg(feature = "arch-qwen35")]
pub mod gated_delta;
pub mod norm;
pub mod rope;

#[cfg(feature = "vision")]
pub mod gelu;
#[cfg(feature = "model-ling3tiny")]
pub mod kda;
#[cfg(feature = "vision")]
pub mod vision_half;
