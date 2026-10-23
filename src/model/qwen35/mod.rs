//! Shared Qwen3.5 decoder architecture, also used by MiniCPM-V 4.7 text.
//! Model-family availability is controlled by the registry, independently of reuse.
mod attention;
pub mod config;
mod decoder;
mod features;
mod head;
mod load;
mod moe;
mod mtp;
mod multimodal;
mod prefill;
mod rope;
mod runtime;
mod stats;
mod timings;
mod transaction;
#[cfg(feature = "vision")]
pub mod vision;
pub mod weights;

pub use runtime::Qwen35;

#[cfg(all(test, feature = "model-minicpmv47", feature = "draft-dflash"))]
mod minicpm_spec_tests;
#[cfg(all(test, feature = "model-minicpmv47"))]
mod minicpm_tests;
#[cfg(all(test, feature = "model-minicpmv47"))]
mod paging_tests;
