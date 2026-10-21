//! Qwen3.5 hybrid text model: config, checkpoint mapping and decoder orchestration.
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
pub mod vision;
pub mod weights;

pub use runtime::Qwen35;

#[cfg(test)]
mod minicpm_spec_tests;
#[cfg(test)]
mod minicpm_tests;
#[cfg(test)]
mod paging_tests;
