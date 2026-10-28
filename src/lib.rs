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
