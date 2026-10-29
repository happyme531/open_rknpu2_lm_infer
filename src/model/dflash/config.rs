use anyhow::{Result, ensure};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Deserialize)]
pub struct DraftConfig {
    #[serde(default)]
    pub block_size: usize,
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
    #[serde(default)]
    pub markov_rank: usize,
    #[serde(default)]
    pub markov_head_type: String,
    #[serde(default)]
    pub enable_confidence_head: bool,
    #[serde(default)]
    pub confidence_head_with_markov: bool,
}

#[derive(Clone, Deserialize)]
pub struct RopeConfig {
    pub rope_theta: f32,
}
#[derive(Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub block_size: usize,
    pub architectures: Vec<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f32,
    pub sliding_window: Option<usize>,
    pub layer_types: Vec<String>,
    pub rope_parameters: RopeConfig,
    pub dflash_config: DraftConfig,
}
impl Config {
    pub fn load(path: &Path, context: usize) -> Result<Self> {
        let mut c: Self = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        if c.dspark() {
            c.dflash_config.block_size = c.block_size;
        }
        ensure!(
            c.dflash_config.block_size > c.proposal_offset(),
            "draft block must contain at least one proposal row"
        );
        ensure!(
            c.architectures
                .iter()
                .any(|s| s == "DFlashDraftModel" || s == "Qwen3DSparkModel"),
            "expected a DFlashDraftModel or Qwen3DSparkModel checkpoint"
        );
        ensure!(
            c.layer_types.len() == c.num_hidden_layers,
            "invalid DFlash layer list"
        );
        ensure!(
            c.sliding_window.is_none_or(|window| context <= window),
            "DFlash ctx-size exceeds the sliding window supported by this backend"
        );
        ensure!(
            c.layer_types
                .iter()
                .all(|s| matches!(s.as_str(), "sliding_attention" | "full_attention")),
            "unsupported DFlash attention type"
        );
        Ok(c)
    }
    pub fn dspark(&self) -> bool {
        self.architectures.iter().any(|s| s == "Qwen3DSparkModel")
    }
    /// DFlash skips the anchor's output; DSpark's first row is a proposal.
    pub fn proposal_offset(&self) -> usize {
        usize::from(!self.dspark())
    }
    pub fn max_proposals(&self) -> usize {
        self.dflash_config.block_size - self.proposal_offset()
    }
    pub fn context_width(&self) -> usize {
        self.hidden_size * self.dflash_config.target_layer_ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn minicpm_dspark_uses_five_target_features_and_full_attention() -> Result<()> {
        let c = Config::load(Path::new("tests/fixtures/minicpmv47/dspark"), 4096)?;
        assert!(c.dspark());
        assert_eq!(c.dflash_config.target_layer_ids, [3, 11, 19, 27, 35]);
        assert_eq!(c.context_width(), 10240);
        assert_eq!(c.dflash_config.block_size, 15);
        assert_eq!(c.dflash_config.markov_rank, 256);
        assert!(c.sliding_window.is_none());
        Ok(())
    }
}
