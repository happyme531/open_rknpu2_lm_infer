//! Block-parallel DFlash draft inference, using the existing public NPU ABI.
//! Adapted from Z Lab's DFlash reference; see dflash/LICENSE.
mod config;
mod layer;
mod load;
mod markov;
mod runtime;

pub use config::Config;
pub use runtime::DFlash;
