//! Target-specific adapter for the shared DFlash benchmark.
#[cfg(any(feature = "model-qwen35", feature = "model-minicpmv47"))]
#[path = "dflash_bench/target.rs"]
mod target;

fn main() -> anyhow::Result<()> {
    #[cfg(any(feature = "model-qwen35", feature = "model-minicpmv47"))]
    {
        target::run()
    }
    #[cfg(not(any(feature = "model-qwen35", feature = "model-minicpmv47")))]
    anyhow::bail!("this benchmark requires model-qwen35 or model-minicpmv47")
}
