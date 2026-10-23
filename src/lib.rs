#[cfg(all(
    feature = "draft-dflash",
    not(any(feature = "model-qwen35", feature = "model-minicpmv47"))
))]
compile_error!("draft-dflash requires model-qwen35 or model-minicpmv47");

pub const PROJECT_NAME: &str = env!("RKNPU_PROJECT_NAME");

pub mod api;
pub mod bench;
pub mod cli;
pub mod engine;
pub mod input;
pub mod layers;
pub mod model;
pub mod npu;
pub mod ops;
pub mod ppl;
pub mod sampling;
pub mod speculative;
pub mod text;
pub mod tui;
pub mod validation;
pub mod vision;
