use anyhow::{Result, ensure};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub const DEFAULT_TEMPERATURE: f32 = 1.0;
pub const DEFAULT_MAX_TOKENS: usize = 2048;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum DraftSampling {
    #[default]
    Greedy,
    Probabilistic,
}
impl DraftSampling {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::Probabilistic => "probabilistic",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub seed: Option<u64>,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: None,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl SamplingParams {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature.is_finite() && (0.0..=2.0).contains(&self.temperature),
            "temperature must be in [0, 2]"
        );
        ensure!(
            self.top_p.is_finite() && (0.0..=1.0).contains(&self.top_p),
            "top_p must be in [0, 1]"
        );
        ensure!(
            self.min_p.is_finite() && (0.0..=1.0).contains(&self.min_p),
            "min_p must be in [0, 1]"
        );
        ensure!(
            self.repeat_penalty.is_finite() && self.repeat_penalty > 0.0,
            "repeat_penalty must be positive"
        );
        ensure!(
            self.presence_penalty.is_finite() && (-2.0..=2.0).contains(&self.presence_penalty),
            "presence_penalty must be in [-2, 2]"
        );
        ensure!(
            self.frequency_penalty.is_finite() && (-2.0..=2.0).contains(&self.frequency_penalty),
            "frequency_penalty must be in [-2, 2]"
        );
        Ok(())
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: ChaCha8Rng,
    counts: Vec<u32>,
    candidates: Vec<(u32, f32)>,
}
impl Clone for Sampler {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            rng: self.rng.clone(),
            counts: self.counts.clone(),
            candidates: Vec::with_capacity(self.counts.len()),
        }
    }
}

/// Normalized post-processor distribution. Sparse storage preserves top-k/p
/// truncation exactly and avoids a vocabulary-sized buffer per proposal.
#[derive(Clone, Debug)]
pub struct Distribution {
    values: Vec<(u32, f64)>,
}
impl Distribution {
    pub fn probability(&self, token: u32) -> f64 {
        self.values
            .iter()
            .find(|(t, _)| *t == token)
            .map_or(0., |(_, p)| *p)
    }
    pub fn residual(&self, proposal: &Self) -> Result<Self> {
        let mut q = std::collections::HashMap::with_capacity(proposal.values.len());
        q.extend(proposal.values.iter().copied());
        let mut values: Vec<_> = self
            .values
            .iter()
            .map(|(t, p)| (*t, (p - q.get(t).unwrap_or(&0.)).max(0.)))
            .filter(|(_, p)| *p > 0.)
            .collect();
        let sum: f64 = values.iter().map(|(_, p)| p).sum();
        ensure!(sum > 0. && sum.is_finite(), "empty rejection residual");
        for (_, p) in &mut values {
            *p /= sum;
        }
        Ok(Self { values })
    }
}

impl Sampler {
    fn processed_logit(&self, token: usize, original: f32) -> f32 {
        let count = self.counts[token];
        if count == 0 {
            return original;
        }
        let value = if original >= 0. {
            original / self.params.repeat_penalty
        } else {
            original * self.params.repeat_penalty
        };
        value - (self.params.presence_penalty + self.params.frequency_penalty * count as f32)
    }

    /// Maximum softmax probability among the selected candidates. Greedy draft
    /// confidence uses temperature one; probabilistic draft uses its temperature.
    /// A zero top_k selects the full vocabulary for benchmark comparisons.
    /// It consumes no RNG and does not change history or sampling processors.
    pub fn draft_confidence(&self, logits: &[f32], top_k: usize) -> Result<f64> {
        ensure!(
            !logits.is_empty() && logits.len() == self.counts.len(),
            "invalid confidence vocabulary"
        );
        let mut values = Vec::with_capacity(logits.len());
        for (token, &original) in logits.iter().enumerate() {
            let value = self.processed_logit(token, original);
            debug_assert!(value.is_finite(), "nonfinite draft confidence logit");
            values.push(value);
        }
        if top_k > 0 && top_k < values.len() {
            values.select_nth_unstable_by(top_k - 1, |a, b| b.total_cmp(a));
            values.truncate(top_k);
        }
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let temperature = if self.greedy() {
            1.
        } else {
            self.params.temperature
        };
        let sum: f64 = values
            .iter()
            .map(|&value| ((value as f64 - maximum as f64) / temperature as f64).exp())
            .sum();
        Ok(sum.recip())
    }

    pub fn new(params: SamplingParams, vocab_size: usize, history: &[u32]) -> Result<Self> {
        params.validate()?;
        let rng = match params.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };
        let mut sampler = Self {
            params,
            rng,
            counts: vec![0; vocab_size],
            candidates: Vec::with_capacity(vocab_size),
        };
        for &token in history {
            sampler.observe(token);
        }
        Ok(sampler)
    }

    pub fn observe(&mut self, token: u32) {
        if let Some(count) = self.counts.get_mut(token as usize) {
            *count += 1;
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        if self.greedy() {
            return self.greedy_token(logits);
        }
        let distribution = self.distribution(logits)?;
        Ok(self.draw(&distribution))
    }
    /// Use another sampler's processors and history with this sampler's RNG.
    pub fn sample_via(&mut self, history: &mut Self, logits: &[f32]) -> Result<u32> {
        if history.greedy() {
            return history.greedy_token(logits);
        }
        let distribution = history.distribution(logits)?;
        Ok(self.draw(&distribution))
    }

    fn greedy_token(&self, logits: &[f32]) -> Result<u32> {
        ensure!(
            !logits.is_empty() && logits.len() == self.counts.len(),
            "invalid logits vocabulary"
        );
        let mut best = (0, f32::NEG_INFINITY);
        for (token, &original) in logits.iter().enumerate() {
            debug_assert!(original.is_finite(), "nonfinite logit at token {token}");
            let value = self.processed_logit(token, original);
            // Retain the first token when processed logits tie.
            if value > best.1 {
                best = (token as u32, value);
            }
        }
        Ok(best.0)
    }
    /// Proposal history and RNG are independent of committed target history.
    pub fn fork(&self) -> Self {
        let mut result = self.clone();
        let seed = result.rng.r#gen::<u64>() ^ 0x85eb_ca6b_d192_ed03;
        result.rng = ChaCha8Rng::seed_from_u64(seed);
        result
    }
    /// Match llama.cpp's MTP draft chain: raw top-10 logits, no target history
    /// penalties or top-p/min-p; only temperature carries over in probabilistic mode.
    pub fn draft(&self, mode: DraftSampling) -> Self {
        let mut result = self.fork();
        result.params = SamplingParams {
            temperature: if mode == DraftSampling::Probabilistic {
                self.params.temperature
            } else {
                0.
            },
            top_k: 10,
            ..Default::default()
        };
        result
    }
    pub fn greedy(&self) -> bool {
        self.params.temperature == 0.
    }
    pub fn accept(
        &mut self,
        target: &Distribution,
        proposal: &Distribution,
        token: u32,
    ) -> Result<bool> {
        let q = proposal.probability(token);
        ensure!(q > 0., "sampled proposal has zero probability");
        let ratio = (target.probability(token) / q).min(1.);
        Ok(ratio >= 1. || self.rng.r#gen::<f64>() < ratio)
    }
    pub fn draw(&mut self, distribution: &Distribution) -> u32 {
        if distribution.values.len() == 1 {
            return distribution.values[0].0;
        }
        let target = self.rng.r#gen::<f64>();
        let mut accum = 0.;
        for &(token, p) in &distribution.values {
            accum += p;
            if target < accum {
                return token;
            }
        }
        distribution.values.last().unwrap().0
    }
    pub fn distribution(&mut self, logits: &[f32]) -> Result<Distribution> {
        if self.greedy() {
            return Ok(Distribution {
                values: vec![(self.greedy_token(logits)?, 1.)],
            });
        }
        ensure!(
            !logits.is_empty() && logits.len() == self.counts.len(),
            "invalid logits vocabulary"
        );
        self.candidates.clear();
        for (token, &original) in logits.iter().enumerate() {
            debug_assert!(original.is_finite(), "nonfinite logit at token {token}");
            let value = self.processed_logit(token, original);
            self.candidates.push((token as u32, value));
        }
        let order = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        let k = self.params.top_k;
        if k > 0 && k < self.candidates.len() {
            self.candidates.select_nth_unstable_by(k - 1, order);
            self.candidates.truncate(k);
        }
        let truncated = k == 0
            && self.candidates.len() > 64
            && (self.params.top_p < 1. || self.params.min_p > 0.)
            && self.truncate_candidates();
        if !truncated {
            self.candidates.sort_unstable_by(order);
        }
        let maximum = self.candidates[0].1;
        let mut sum = 0.0f64;
        for candidate in &mut self.candidates {
            candidate.1 = ((candidate.1 - maximum) / self.params.temperature).exp();
            sum += candidate.1 as f64;
        }
        if !truncated {
            let minimum = self.candidates[0].1 * self.params.min_p;
            let mut cumulative = 0.0f64;
            let mut keep = 0;
            for candidate in &self.candidates {
                if candidate.1 < minimum {
                    break;
                }
                cumulative += candidate.1 as f64;
                keep += 1;
                if cumulative >= sum * self.params.top_p as f64 {
                    break;
                }
            }
            self.candidates.truncate(keep.max(1));
        }
        let total: f64 = self.candidates.iter().map(|c| c.1 as f64).sum();
        Ok(Distribution {
            values: self
                .candidates
                .iter()
                .map(|&(t, p)| (t, p as f64 / total))
                .collect(),
        })
    }

    /// Select and sort only the logit prefix needed by top-p/min-p. Keep logits
    /// intact: exponentiation can round distinct logits to the same weight.
    fn truncate_candidates(&mut self) -> bool {
        let order = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        let (minimum, maximum) = self.candidates.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), c| (minimum.min(c.1), maximum.max(c.1)),
        );
        let weight = |value: f32| ((value - maximum) / self.params.temperature).exp();
        let minimum_weight = weight(minimum);
        // If min-p retains every token and top-p needs at least a quarter of
        // the vocabulary, a full sort avoids repeated prefix partitions.
        if minimum_weight >= self.params.min_p && minimum_weight * self.params.top_p >= 0.25 {
            return false;
        }
        let sum: f64 = self.candidates.iter().map(|c| weight(c.1) as f64).sum();
        let cutoff = sum * self.params.top_p as f64;
        // Both positive f64 sums differ from the exact sum by at most roughly
        // n*epsilon. Resolve a cutoff within that interval with the full sort,
        // preserving the original descending accumulation and kept token set.
        let error = 2. * self.candidates.len() as f64 * f64::EPSILON * sum;
        let mut head = self.candidates.len().min(64);
        while head < self.candidates.len() {
            self.candidates.select_nth_unstable_by(head - 1, order);
            self.candidates[..head].sort_unstable_by(order);
            let mut cumulative = 0.;
            for (index, candidate) in self.candidates[..head].iter().enumerate() {
                let value = weight(candidate.1);
                if value < self.params.min_p {
                    self.candidates.truncate(index.max(1));
                    return true;
                }
                cumulative += value as f64;
                if cumulative >= cutoff + error {
                    self.candidates.truncate(index + 1);
                    return true;
                }
                if cumulative >= cutoff - error {
                    return false;
                }
            }
            head = (head * 4).min(self.candidates.len());
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_sort_distribution(sampler: &Sampler, logits: &[f32]) -> Distribution {
        let params = &sampler.params;
        let mut sorted: Vec<_> = logits
            .iter()
            .enumerate()
            .map(|(i, &value)| (i as u32, sampler.processed_logit(i, value)))
            .collect();
        sorted.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        if params.top_k > 0 {
            sorted.truncate(params.top_k);
        }
        let maximum = sorted[0].1;
        for (_, value) in &mut sorted {
            *value = ((*value - maximum) / params.temperature).exp();
        }
        let sum: f64 = sorted.iter().map(|(_, value)| *value as f64).sum();
        let minimum = sorted[0].1 * params.min_p;
        let mut cumulative = 0.;
        let mut kept = Vec::new();
        for candidate in sorted {
            if candidate.1 < minimum {
                break;
            }
            cumulative += candidate.1 as f64;
            kept.push(candidate);
            if cumulative >= sum * params.top_p as f64 {
                break;
            }
        }
        let total: f64 = kept.iter().map(|(_, value)| *value as f64).sum();
        Distribution {
            values: kept
                .into_iter()
                .map(|(i, value)| (i, value as f64 / total))
                .collect(),
        }
    }

    #[test]
    fn untruncated_sampling_preserves_the_seeded_token_sequence() -> Result<()> {
        let mut sampler = Sampler::new(
            SamplingParams {
                seed: Some(42),
                ..Default::default()
            },
            4,
            &[],
        )?;
        let actual: Vec<_> = (0..16)
            .map(|_| sampler.sample(&[-1., 0.5, 1.5, -2.]).unwrap())
            .collect();
        assert_eq!(actual, [1, 0, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 0, 2, 1, 2]);
        Ok(())
    }

    #[test]
    fn nucleus_cutoff_preserves_logit_order_when_exp_rounds_equal() -> Result<()> {
        let mut sampler = Sampler::new(
            SamplingParams {
                top_p: 0.5,
                ..Default::default()
            },
            2,
            &[],
        )?;
        let distribution = sampler.distribution(&[0., 1e-8])?;
        assert_eq!(distribution.probability(0), 0.);
        assert_eq!(distribution.probability(1), 1.);
        Ok(())
    }

    #[test]
    fn partial_nucleus_matches_full_sort_probabilities_and_seeded_draws() -> Result<()> {
        for vocab in [257, 4096] {
            for case in 0..6 {
                let mut logits: Vec<_> = (0..vocab)
                    .map(|i| match case {
                        0 => 0.,
                        1 => ((i * 17) % 113) as f32 * 0.25 - 14.,
                        2 => -(((i * 31) % vocab) as f32) * 0.1,
                        3 => -100.,
                        4 => -(((i * 7) % 11) as f32) * 1e-8,
                        _ => {
                            if i % 2 == 0 {
                                -0.
                            } else {
                                0.
                            }
                        }
                    })
                    .collect();
                if case == 3 {
                    logits[1] = 0.;
                }
                for (top_p, min_p) in [
                    (0., 0.),
                    (0.5, 0.),
                    (0.95, 0.),
                    (1., 0.),
                    (1., 0.05),
                    (0.95, 0.05),
                    (1., 1.),
                ] {
                    for temperature in [0.6, 1.] {
                        let params = SamplingParams {
                            temperature,
                            top_p,
                            min_p,
                            seed: Some(42),
                            repeat_penalty: 1.1,
                            presence_penalty: 0.2,
                            frequency_penalty: 0.1,
                            ..Default::default()
                        };
                        let mut actual = Sampler::new(params, vocab, &[0, 0])?;
                        let mut reference = actual.clone();
                        for _ in 0..4 {
                            let expected = full_sort_distribution(&reference, &logits);
                            assert_eq!(
                                actual.distribution(&logits)?.values,
                                expected.values,
                                "vocab={vocab},case={case},p={top_p},min_p={min_p},t={temperature}"
                            );
                            let token = actual.sample(&logits)?;
                            assert_eq!(token, reference.draw(&expected));
                            actual.observe(token);
                            reference.observe(token);
                        }
                    }
                }
            }
        }
        let mut logits = vec![-100.; 257];
        logits[0] = 0.;
        logits[1] = 1e-8;
        let mut sampler = Sampler::new(
            SamplingParams {
                top_p: 0.5,
                ..Default::default()
            },
            logits.len(),
            &[],
        )?;
        assert_eq!(sampler.distribution(&logits)?.values, vec![(1, 1.)]);
        Ok(())
    }

    #[test]
    fn sample_via_uses_history_processors_and_caller_rng() -> Result<()> {
        for temperature in [0., 0.7, 1.] {
            for top_p in [0.5, 1.] {
                let mut caller = Sampler::new(
                    SamplingParams {
                        seed: Some(17),
                        ..Default::default()
                    },
                    257,
                    &[2],
                )?;
                let mut history = Sampler::new(
                    SamplingParams {
                        temperature,
                        top_p,
                        seed: Some(42),
                        frequency_penalty: 0.5,
                        ..Default::default()
                    },
                    257,
                    &[0, 0, 1],
                )?;
                let mut reference_caller = caller.clone();
                let mut reference_history = history.clone();
                let logits: Vec<_> = (0..257).map(|i| (i % 7) as f32).collect();
                for _ in 0..16 {
                    let distribution = reference_history.distribution(&logits)?;
                    let expected = reference_caller.draw(&distribution);
                    assert_eq!(caller.sample_via(&mut history, &logits)?, expected);
                    history.observe(expected);
                    reference_history.observe(expected);
                }
                assert_eq!(caller.counts, [vec![0, 0, 1], vec![0; 254]].concat());
                assert_eq!(history.counts, reference_history.counts);
                assert_eq!(caller.sample(&logits)?, reference_caller.sample(&logits)?);
                assert!(caller.sample_via(&mut history, &[]).is_err());
                assert!(history.sample(&[]).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn single_candidate_step_does_not_advance_the_rng() -> Result<()> {
        let mut sampler = Sampler::new(
            SamplingParams {
                seed: Some(42),
                ..Default::default()
            },
            2,
            &[],
        )?;
        assert_eq!(sampler.sample(&[0., -100.])?, 0);
        let actual: Vec<_> = (0..8).map(|_| sampler.sample(&[0., 0.]).unwrap()).collect();
        assert_eq!(actual, [1, 1, 0, 1, 0, 0, 0, 1]);
        Ok(())
    }

    #[test]
    fn partial_top_k_matches_full_sort_probabilities_and_seeded_draws() -> Result<()> {
        for vocab in [1, 37, 4096] {
            for k in [0, 1, 20, vocab, vocab + 1] {
                for step in 0..4 {
                    let params = SamplingParams {
                        temperature: if step % 2 == 0 { 1. } else { 0.6 },
                        top_k: k,
                        top_p: [0., 0.95, 1., 0.8][step],
                        min_p: 0.05,
                        repeat_penalty: 1.1,
                        presence_penalty: 0.2,
                        frequency_penalty: 0.1,
                        seed: Some(42),
                    };
                    let mut actual = Sampler::new(params.clone(), vocab, &[0, 0])?;
                    let mut reference = actual.clone();
                    let logits: Vec<_> = (0..vocab)
                        .map(|i| {
                            if step == 0 {
                                0.
                            } else {
                                ((i * 17 + step * 31) % 61) as f32 * 0.25 - 7.5
                            }
                        })
                        .collect();
                    let expected = full_sort_distribution(&reference, &logits);
                    assert_eq!(
                        actual.distribution(&logits)?.values,
                        expected.values,
                        "vocab={vocab},k={k},step={step}"
                    );
                    assert_eq!(actual.sample(&logits)?, reference.draw(&expected));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn greedy_preserves_first_token_ties() -> Result<()> {
        let mut sampler = Sampler::new(
            SamplingParams {
                temperature: 0.,
                ..Default::default()
            },
            3,
            &[],
        )?;
        assert_eq!(sampler.sample(&[-0., 0., -1.])?, 0);
        assert_eq!(sampler.sample(&[-1., 2., 2.])?, 1);
        Ok(())
    }

    #[test]
    fn mtp_draft_uses_raw_top_ten_and_independent_sampling_settings() -> Result<()> {
        let parent = Sampler::new(
            SamplingParams {
                temperature: 0.7,
                top_k: 1,
                top_p: 0.1,
                min_p: 0.9,
                repeat_penalty: 2.,
                presence_penalty: 1.,
                frequency_penalty: 0.5,
                ..Default::default()
            },
            20,
            &[0, 0],
        )?;
        let mut logits = vec![0.; 20];
        logits[0] = 2.;
        let mut greedy = parent.draft(DraftSampling::Greedy);
        assert_eq!(greedy.distribution(&logits)?.probability(0), 1.);
        let top_ten = 1. / (1. + 9. * (-2f64).exp());
        assert!((greedy.draft_confidence(&logits, 10)? - top_ten).abs() < 1e-12);
        let full = 1. / (1. + 19. * (-2f64).exp());
        assert!((greedy.draft_confidence(&logits, 0)? - full).abs() < 1e-12);
        let mut probabilistic = parent.draft(DraftSampling::Probabilistic);
        let distribution = probabilistic.distribution(&logits)?;
        assert_eq!(distribution.values.len(), 10);
        assert!(distribution.probability(0) > top_ten);
        assert!(probabilistic.draft_confidence(&logits, 10)? > top_ten);
        Ok(())
    }

    #[test]
    fn draft_confidence_precedes_greedy_and_sampling_truncation() -> Result<()> {
        let mut sampler = Sampler::new(
            SamplingParams {
                temperature: 0.,
                repeat_penalty: 2.,
                presence_penalty: 1.,
                frequency_penalty: 0.5,
                ..Default::default()
            },
            3,
            &[0, 0],
        )?;
        // Processed logits are [-1,0,0]; greedy q is still a point mass.
        let logits = [2., 0., 0.];
        let expected = 1. / (2. + (-1f64).exp());
        assert!((sampler.draft_confidence(&logits, 10)? - expected).abs() < 1e-12);
        assert_eq!(sampler.distribution(&logits)?.probability(1), 1.);
        let sampler = Sampler::new(SamplingParams::default(), 3, &[])?;
        assert!((sampler.draft_confidence(&[10000.; 3], 10)? - 1. / 3.).abs() < 1e-12);
        assert!(sampler.draft_confidence(&[0.], 10).is_err());
        Ok(())
    }
    #[test]
    fn confidence_reads_do_not_advance_rng_or_history() -> Result<()> {
        let mut a = Sampler::new(
            SamplingParams {
                temperature: 0.8,
                seed: Some(311),
                top_k: 2,
                ..Default::default()
            },
            3,
            &[2],
        )?;
        let mut b = a.clone();
        for _ in 0..100 {
            a.draft_confidence(&[1., 0.5, 0.], 10)?;
            let x = a.sample(&[1., 0.5, 0.])?;
            let y = b.sample(&[1., 0.5, 0.])?;
            assert_eq!(x, y);
            a.observe(x);
            b.observe(y);
        }
        Ok(())
    }
    #[test]
    fn rejection_distribution_is_exact_with_disjoint_and_truncated_supports() {
        for (p, q) in [
            (vec![0.1, 0.3, 0.6], vec![0.6, 0.3, 0.1]),
            (vec![0., 0.4, 0.6], vec![1., 0., 0.]),
            (vec![0.5, 0., 0.5], vec![0.1, 0.9, 0.]),
        ] {
            let make = |v: &Vec<f64>| Distribution {
                values: v
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, p)| *p > 0.)
                    .map(|(i, p)| (i as u32, p))
                    .collect(),
            };
            let (target, proposal) = (make(&p), make(&q));
            let residual = target.residual(&proposal).unwrap();
            let reject: f64 = p.iter().zip(&q).map(|(p, q)| (q - p).max(0.)).sum();
            for i in 0..p.len() {
                let mass = p[i].min(q[i]) + reject * residual.probability(i as u32);
                assert!((mass - p[i]).abs() < 1e-14);
            }
        }
    }
    #[test]
    fn stochastic_accept_and_residual_reproduce_target_probabilities() {
        let p = Distribution {
            values: vec![(0, 0.1), (1, 0.3), (2, 0.6)],
        };
        let q = Distribution {
            values: vec![(0, 0.6), (1, 0.3), (2, 0.1)],
        };
        let residual = p.residual(&q).unwrap();
        let mut target = Sampler::new(
            SamplingParams {
                temperature: 0.8,
                seed: Some(17),
                ..Default::default()
            },
            3,
            &[],
        )
        .unwrap();
        let mut draft = target.fork();
        let mut counts = [0; 3];
        for _ in 0..100_000 {
            let proposed = draft.draw(&q);
            let token = if target.accept(&p, &q, proposed).unwrap() {
                proposed
            } else {
                target.draw(&residual)
            };
            counts[token as usize] += 1;
        }
        for (i, count) in counts.into_iter().enumerate() {
            assert!((count as f64 / 100_000. - p.probability(i as u32)).abs() < 0.005);
        }
    }
    #[test]
    fn proposal_history_does_not_change_committed_processors() {
        let mut target = Sampler::new(
            SamplingParams {
                temperature: 0.,
                frequency_penalty: 2.,
                ..Default::default()
            },
            3,
            &[1],
        )
        .unwrap();
        let mut draft = target.fork();
        draft.observe(0);
        draft.observe(0);
        assert_eq!(target.sample(&[3., 2., 1.]).unwrap(), 0);
        assert_eq!(draft.sample(&[3., 2., 1.]).unwrap(), 2);
        assert_eq!(target.counts, [0, 1, 0]);
    }
    #[test]
    fn zero_top_p_keeps_the_most_likely_token() {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(params, 3, &[]).unwrap();
        assert_eq!(sampler.sample(&[1.0, 3.0, 2.0]).unwrap(), 1);
    }
    #[test]
    fn penalties_change_greedy_choice() {
        let params = SamplingParams {
            temperature: 0.,
            frequency_penalty: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(params, 3, &[0, 0]).unwrap();
        assert_eq!(sampler.sample(&[3.0, 2.0, 1.0]).unwrap(), 1);
    }
    #[test]
    fn seeded_sampling_reproduces_and_top_k_limits() {
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 2,
            seed: Some(17),
            ..Default::default()
        };
        let mut a = Sampler::new(params.clone(), 3, &[]).unwrap();
        let mut b = Sampler::new(params, 3, &[]).unwrap();
        let x: Vec<_> = (0..32)
            .map(|_| a.sample(&[2.0, 1.9, -10.0]).unwrap())
            .collect();
        let y: Vec<_> = (0..32)
            .map(|_| b.sample(&[2.0, 1.9, -10.0]).unwrap())
            .collect();
        assert_eq!(x, y);
        assert!(x.iter().all(|&token| token < 2));
        assert!(x.contains(&0) && x.contains(&1));
    }
}
