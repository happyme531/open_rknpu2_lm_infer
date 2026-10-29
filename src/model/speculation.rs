use crate::sampling::DraftSampling;
use anyhow::{Result, ensure};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum SpeculativeType {
    #[default]
    None,
    DraftMtp,
    DraftDflash,
    DraftDspark,
}
impl SpeculativeType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::DraftMtp => "draft-mtp",
            Self::DraftDflash => "draft-dflash",
            Self::DraftDspark => "draft-dspark",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    pub kind: SpeculativeType,
    pub model: Option<PathBuf>,
    /// Number of proposals, excluding the pending anchor and verifier bonus.
    pub max_draft: usize,
    pub p_min: f64,
    pub sampling: DraftSampling,
}
impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            kind: SpeculativeType::None,
            model: None,
            max_draft: 3,
            p_min: 0.,
            sampling: DraftSampling::Greedy,
        }
    }
}
impl SpeculativeConfig {
    pub fn window(&self) -> usize {
        if self.kind == SpeculativeType::None {
            0
        } else {
            self.max_draft
        }
    }
    pub(crate) fn validate(&self, chunk: usize) -> Result<()> {
        let max = if self.kind == SpeculativeType::DraftDspark {
            15
        } else {
            7
        };
        ensure!(
            self.max_draft <= max,
            "spec-draft-n-max must be in 0..{max}"
        );
        ensure!(
            self.window() == 0 || self.max_draft < chunk,
            "spec-draft-n-max must be smaller than ubatch-size"
        );
        ensure!(
            self.p_min.is_finite() && (0. ..=1.).contains(&self.p_min),
            "spec-draft-p-min must be finite and in [0,1]"
        );
        ensure!(
            self.p_min == 0. || self.window() > 0,
            "spec-draft-p-min requires enabled speculative decoding"
        );
        ensure!(
            matches!(
                self.kind,
                SpeculativeType::DraftDflash | SpeculativeType::DraftDspark
            ) || self.model.is_none(),
            "spec-draft-model requires --spec-type draft-dflash or draft-dspark"
        );
        ensure!(
            !matches!(
                self.kind,
                SpeculativeType::DraftDflash | SpeculativeType::DraftDspark
            ) || self.window() == 0
                || self.model.is_some(),
            "draft-dflash/draft-dspark requires --spec-draft-model"
        );
        Ok(())
    }
}
