use super::RoundPlan;
use crate::{
    model::LanguageModel,
    sampling::{DraftSampling, Sampler},
};
use anyhow::{Result, ensure};
use std::{sync::atomic::AtomicBool, time::Instant};

#[derive(Clone, Copy, Debug)]
pub struct DraftOptions {
    pub p_min: f64,
    /// Keep the already computed proposal, then stop continuation below p_min.
    /// Otherwise omit the low-confidence proposal, as llama.cpp does.
    pub keep_low_confidence: bool,
    pub collect_confidence: bool,
    pub sampling: DraftSampling,
    pub confidence_top_k: usize,
}
impl Default for DraftOptions {
    fn default() -> Self {
        Self {
            p_min: 0.,
            keep_low_confidence: false,
            collect_confidence: false,
            sampling: DraftSampling::Greedy,
            confidence_top_k: 10,
        }
    }
}

pub struct VerifiedRound {
    pub inputs: Vec<u32>,
    pub output: Vec<u32>,
    pub logits: Vec<f32>,
    pub accepted: usize,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub draft_attempts: usize,
    pub draft_confidence: Vec<f64>,
    pub confidence_stopped: bool,
}

/// Leave a verified transaction open so the caller can apply EOS/stop/length
/// before committing the exact prefix and publishing any streaming events.
pub fn run_round(
    model: &mut dyn LanguageModel,
    sampler: &mut Sampler,
    anchor: u32,
    remaining: usize,
    cancel: &AtomicBool,
) -> Result<VerifiedRound> {
    let options = DraftOptions {
        p_min: model.speculation_p_min(),
        sampling: model.speculation_sampling(),
        ..Default::default()
    };
    run_round_with_options(model, sampler, anchor, remaining, cancel, options)
}

pub fn run_round_with_options(
    model: &mut dyn LanguageModel,
    sampler: &mut Sampler,
    anchor: u32,
    remaining: usize,
    cancel: &AtomicBool,
    options: DraftOptions,
) -> Result<VerifiedRound> {
    ensure!(
        options.p_min.is_finite() && (0. ..=1.).contains(&options.p_min),
        "draft p_min must be in [0,1]"
    );
    let plan = RoundPlan::new(
        model.position(),
        model.info().context,
        remaining,
        model.speculation_window(),
    )?
    .ok_or_else(|| anyhow::anyhow!("no speculative context/output budget"))?;
    let result = (|| -> Result<VerifiedRound> {
        let mut proposal_sampler = sampler.draft(options.sampling);
        let mut target_history = sampler.clone();
        let mut distributions = Vec::with_capacity(plan.draft_tokens());
        let mut inputs = vec![anchor];
        let mut draft_confidence = Vec::new();
        let mut draft_attempts = 0;
        let mut confidence_stopped = false;
        let start = Instant::now();
        let mut block = if plan.draft_tokens() > 0 {
            model.draft_block_logits(anchor, plan.draft_tokens(), cancel)?
        } else {
            None
        };
        let vocab = model.info().vocab_size;
        ensure!(
            block
                .as_ref()
                .is_none_or(|values| values.len() == plan.draft_tokens() * vocab),
            "incorrect parallel draft logits shape"
        );
        for row in 0..plan.draft_tokens() {
            let previous = *inputs.last().unwrap();
            let (logits, learned) = if let Some(values) = &mut block {
                let logits = &mut values[row * vocab..(row + 1) * vocab];
                let learned = model.adjust_draft_logits(row, previous, logits)?;
                (&*logits, learned)
            } else {
                (model.draft_logits(previous, cancel)?, None)
            };
            draft_attempts += 1;
            // The stopping decision is made before sampling, independently of
            // the proposed token. q remains the true processed distribution.
            let low = if options.p_min > 0. || options.collect_confidence {
                let confidence = if let Some(confidence) = learned {
                    confidence
                } else if options.sampling == DraftSampling::Probabilistic
                    && proposal_sampler.greedy()
                {
                    1.
                } else {
                    proposal_sampler.draft_confidence(logits, options.confidence_top_k)?
                };
                draft_confidence.push(confidence);
                confidence < options.p_min
            } else {
                false
            };
            if low && !options.keep_low_confidence {
                confidence_stopped = true;
                break;
            }
            let q = proposal_sampler.distribution(logits)?;
            let token = proposal_sampler.draw(&q);
            proposal_sampler.observe(token);
            inputs.push(token);
            distributions.push(q);
            if low {
                confidence_stopped = true;
                break;
            }
        }
        let draft_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let logits = model.verify_tokens(&inputs, cancel)?.to_vec();
        let verify_ms = start.elapsed().as_secs_f64() * 1000.;
        let vocab = model.info().vocab_size;
        ensure!(
            logits.len() == inputs.len() * vocab,
            "incorrect verification logits shape"
        );
        let mut output = Vec::new();
        let mut accepted = 0;
        for (r, q) in distributions.iter().enumerate() {
            let p = target_history.distribution(&logits[r * vocab..(r + 1) * vocab])?;
            let proposal = inputs[r + 1];
            let sampled = if options.sampling == DraftSampling::Greedy {
                Some(sampler.draw(&p))
            } else {
                None
            };
            let matches = if let Some(token) = sampled {
                token == proposal
            } else {
                sampler.accept(&p, q, proposal)?
            };
            if matches {
                output.push(proposal);
                target_history.observe(proposal);
                accepted += 1;
            } else {
                output.push(if let Some(token) = sampled {
                    token
                } else {
                    sampler.draw(&p.residual(q)?)
                });
                break;
            }
        }
        if accepted == distributions.len() {
            let r = accepted;
            let p = target_history.distribution(&logits[r * vocab..(r + 1) * vocab])?;
            output.push(sampler.draw(&p));
        }
        Ok(VerifiedRound {
            inputs,
            output,
            logits,
            accepted,
            draft_ms,
            verify_ms,
            draft_attempts,
            draft_confidence,
            confidence_stopped,
        })
    })();
    if result.is_err() {
        model.abort_verification()?;
    }
    result
}
