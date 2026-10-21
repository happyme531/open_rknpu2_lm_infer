//! Model-independent file storage and scheduling for packed expert weights.
mod io;
mod paged;
pub(crate) mod prepare;
pub mod store;
mod timeline;
mod window;
pub(crate) use paged::{Paged, Paging};

use anyhow::{Result, ensure};
use std::path::PathBuf;

pub const DEFAULT_CACHE_PERCENT: f64 = 8.;

fn validate_percent(percent: f64) -> Result<()> {
    ensure!(
        percent.is_finite() && percent > 0. && percent <= 100.,
        "expert cache percentage must be greater than 0 and at most 100"
    );
    Ok(())
}

pub fn parse_cache_percent(value: &str) -> std::result::Result<f64, String> {
    let percent = value.parse::<f64>().map_err(|e| e.to_string())?;
    validate_percent(percent).map_err(|e| e.to_string())?;
    Ok(percent)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Storage {
    #[default]
    Resident,
    Stream,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub storage: Storage,
    pub cache: Option<PathBuf>,
    pub cache_percent: f64,
    pub prefetch: bool,
    pub direct: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            storage: Storage::Resident,
            cache: None,
            cache_percent: DEFAULT_CACHE_PERCENT,
            prefetch: true,
            direct: true,
        }
    }
}

impl Options {
    /// Per-layer routed-expert count, rounded down with one token's top-k as minimum.
    pub fn slots(&self, experts: usize, top_k: usize) -> Result<usize> {
        validate_percent(self.cache_percent)?;
        ensure!(
            experts > 0 && top_k > 0 && top_k <= experts,
            "invalid expert geometry"
        );
        Ok(((experts as f64 * self.cache_percent / 100.).floor() as usize).clamp(top_k, experts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_fraction_scales_with_the_model_and_preserves_top_k() -> Result<()> {
        let mut options = Options::default();
        assert_eq!(options.slots(256, 8)?, 20);
        options.cache_percent = 25.;
        assert_eq!(options.slots(256, 8)?, 64);
        assert_eq!(options.slots(64, 8)?, 16);
        assert_eq!(options.slots(16, 2)?, 4);
        options.cache_percent = 15.625;
        assert_eq!(options.slots(256, 8)?, 40);
        options.cache_percent = 1.;
        assert_eq!(options.slots(256, 8)?, 8);
        options.cache_percent = 100.;
        assert_eq!(options.slots(256, 8)?, 256);
        Ok(())
    }

    #[test]
    fn invalid_cache_percentages_are_rejected_before_allocation() {
        for value in ["0", "-1", "100.1", "NaN", "inf", "text"] {
            assert!(parse_cache_percent(value).is_err(), "{value}");
        }
        assert_eq!(parse_cache_percent("15.625"), Ok(15.625));
    }
}
