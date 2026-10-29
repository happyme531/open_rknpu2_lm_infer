//! Approximate route pruning, inspired by Strata's STRATA_DS4_DISK_SKIP_W
//! (src/dsv41/ds4_model.cu, revision 4b9e05a2a806a3fe24f7b2560d98626c37dade35).
//! Unlike its cold-only decode policy, apply to hot/cold routes in both phases.
//! Preserve the strongest route and never renormalize the remaining weights.
//! See Strata-LICENSE for the upstream MIT notice.
use anyhow::{Result, ensure};
use serde::Serialize;

pub(crate) fn validate(value: f32) -> Result<()> {
    ensure!(
        value.is_finite() && (0. ..=1.).contains(&value),
        "expert minimum weight must be finite and between 0 and 1"
    );
    Ok(())
}

pub fn parse(value: &str) -> std::result::Result<f32, String> {
    let value = value.parse::<f32>().map_err(|e| e.to_string())?;
    validate(value).map_err(|e| e.to_string())?;
    Ok(value)
}

#[derive(Default, Serialize)]
pub(crate) struct Stats {
    pub rows: u64,
    pub assignments: u64,
    pub skipped_assignments: u64,
    pub dropped_weight: f64,
}

/// Weights are sorted descending and already normalized over selected top-k.
pub(crate) fn apply(weights: &mut [f32], minimum: f32) {
    if minimum == 0. {
        return;
    }
    for weight in weights.iter_mut().skip(1) {
        if *weight < minimum {
            *weight = 0.;
        }
    }
}

impl Stats {
    pub fn record_unpruned(&mut self, assignments: usize, rows: usize) {
        self.rows += rows as u64;
        self.assignments += assignments as u64;
    }

    pub fn record(&mut self, original: &[f32], retained: &[f32], rows: usize) {
        self.record_unpruned(original.len(), rows);
        for (&before, &after) in original.iter().zip(retained) {
            if before > 0. && after == 0. {
                self.skipped_assignments += 1;
                self.dropped_weight += f64::from(before);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn threshold_keeps_boundary_and_best_without_renormalizing() {
        let original = [0.5, 0.25, 0.125, 0.125];
        let mut weights = original;
        apply(&mut weights, 0.25);
        assert_eq!(weights, [0.5, 0.25, 0., 0.]);
        let mut stats = Stats::default();
        stats.record(&original, &weights, 1);
        assert_eq!(stats.skipped_assignments, 2);
        assert_eq!(stats.dropped_weight, 0.25);
        stats.record_unpruned(8, 2);
        assert_eq!(stats.rows, 3);
        assert_eq!(stats.assignments, 12);
        assert_eq!(stats.skipped_assignments, 2);
        assert_eq!(stats.dropped_weight, 0.25);
        apply(&mut weights, 1.);
        assert_eq!(weights, [0.5, 0., 0., 0.]);
        let mut weights = original;
        apply(&mut weights, 0.);
        assert_eq!(weights, original);
    }
    #[test]
    fn invalid_thresholds_are_rejected() {
        for value in ["NaN", "inf", "-0.1", "1.01", "text"] {
            assert!(parse(value).is_err(), "{value}");
        }
        assert_eq!(parse("0"), Ok(0.));
        assert_eq!(parse("1"), Ok(1.));
    }
}
