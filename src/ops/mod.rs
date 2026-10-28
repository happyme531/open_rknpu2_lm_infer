//! Model-independent CPU numerical operators. Callers supply dimensions and
//! already-resolved parameters; checkpoint names and config defaults stay in models.
pub mod activation;
pub mod causal_conv;
pub mod gated_delta;
pub mod norm;
pub mod rope;

pub mod gelu;
pub mod kda;
pub mod vision_half;
