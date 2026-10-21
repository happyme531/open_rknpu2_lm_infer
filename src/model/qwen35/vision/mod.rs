//! C7 BF16 ViT, lazy-loaded onto a dedicated FP16 device backend.
mod encoder;
mod load;
mod storage;
use crate::{
    layers::layer_norm::LayerNorm,
    npu::{Backend, FloatLinear},
};
pub use encoder::Encoder;

pub struct LinearShape {
    pub name: &'static str,
    pub rows: usize,
    pub k: usize,
    pub n: usize,
}

/// Static projection shapes from the model configuration, without loading weights.
pub fn tuning_shapes(path: &std::path::Path, rows: &[usize]) -> anyhow::Result<Vec<LinearShape>> {
    let c = load::config(path)?;
    let mut shapes = Vec::new();
    for &m in rows {
        anyhow::ensure!(
            m > 0 && m.is_multiple_of(4),
            "vision patch rows must be a positive multiple of four"
        );
        let h = c.hidden;
        for (name, rows, k, n) in [
            ("patch", m, 1536, h),
            ("qkv", m, h, 3 * h),
            ("attention_output", m, h, h),
            ("mlp_up", m, h, c.intermediate),
            ("mlp_down", m, c.intermediate, h),
            ("merger_fc1", m / 4, 4 * h, 4 * h),
            ("merger_fc2", m / 4, 4 * h, c.out),
        ] {
            shapes.push(LinearShape { name, rows, k, n });
        }
    }
    Ok(shapes)
}

struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: FloatLinear,
    proj: FloatLinear,
    fc1: FloatLinear,
    fc2: FloatLinear,
}
struct Merger {
    norm: LayerNorm,
    fc1: FloatLinear,
    fc2: FloatLinear,
}
struct Config {
    hidden: usize,
    intermediate: usize,
    heads: usize,
    depth: usize,
    side: usize,
    out: usize,
}
pub(super) struct Weights {
    config: Config,
    patch: FloatLinear,
    position: Vec<f32>,
    blocks: Vec<Block>,
    merger: Merger,
    backend: Backend,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tuning_shapes_use_config_and_merged_image_rows() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let mut config: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../llm_generated_docs/evidence/qwen35_mtp/official_config.json"
        ))?;
        // Make the test independent of the hard-coded C7 benchmark geometries.
        config["vision_config"]["hidden_size"] = 512.into();
        config["vision_config"]["intermediate_size"] = 2048.into();
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&config)?)?;
        let shapes = tuning_shapes(dir.path(), &[384])?;
        assert_eq!(shapes.len(), 7);
        assert_eq!((shapes[0].rows, shapes[0].k, shapes[0].n), (384, 1536, 512));
        assert_eq!((shapes[4].k, shapes[4].n), (2048, 512));
        assert_eq!((shapes[5].rows, shapes[5].k, shapes[5].n), (96, 2048, 2048));
        assert!(tuning_shapes(dir.path(), &[3]).is_err());
        let large = tuning_shapes(dir.path(), &[4096])?;
        assert_eq!(large[0].rows, 4096);
        assert_eq!(large[5].rows, 1024);
        Ok(())
    }
}
