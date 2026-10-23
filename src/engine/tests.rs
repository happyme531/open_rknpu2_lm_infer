use super::*;
use crate::sampling::SamplingParams;

/// A history-length oracle with controllable proposal rejection. It implements
/// real pending transactions, so emitting/committing too far fails the test.
struct Model {
    info: ModelInfo,
    tokens: Vec<u32>,
    logits: Vec<f32>,
    pending: Option<Vec<u32>>,
    window: usize,
    draft_step: usize,
    reject_at: usize,
    eos_after: usize,
    fail_commit: bool,
    draft_gaps: Vec<f32>,
    p_min: f64,
    sampling: crate::sampling::DraftSampling,
    target_gap: f32,
    parallel: bool,
}
impl Model {
    fn new(window: usize, reject_at: usize, eos_after: usize) -> Self {
        Self {
            info: ModelInfo {
                id: "oracle".into(),
                architecture: "oracle".into(),
                context: 64,
                vocab_size: 5,
                eos_tokens: vec![0],
            },
            tokens: Vec::new(),
            logits: Vec::new(),
            pending: None,
            window,
            draft_step: 0,
            reject_at,
            eos_after,
            fail_commit: false,
            draft_gaps: Vec::new(),
            p_min: 0.,
            sampling: Default::default(),
            target_gap: 20.,
            parallel: false,
        }
    }
    fn row(&self, position: usize) -> Vec<f32> {
        let next = if position >= self.eos_after {
            0
        } else {
            position % 4 + 1
        };
        let mut row = vec![-self.target_gap / 2.; 5];
        row[next] = self.target_gap / 2.;
        row
    }
}
impl LanguageModel for Model {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn position(&self) -> usize {
        self.tokens.len()
    }
    fn reset(&mut self) -> Result<()> {
        self.tokens.clear();
        self.pending = None;
        self.draft_step = 0;
        Ok(())
    }
    fn prepare(&mut self, _: usize) -> Result<()> {
        Ok(())
    }
    fn forward(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        ensure!(self.pending.is_none(), "uncommitted transaction");
        self.tokens.extend_from_slice(tokens);
        self.logits = self.row(self.tokens.len());
        Ok(&self.logits)
    }
    fn stats(&self) -> Value {
        json!({"position":self.position()})
    }
    fn speculation_window(&self) -> usize {
        self.window
    }
    fn speculation_p_min(&self) -> f64 {
        self.p_min
    }
    fn speculation_sampling(&self) -> crate::sampling::DraftSampling {
        self.sampling
    }
    fn draft_logits(&mut self, _: u32, _: &AtomicBool) -> Result<&[f32]> {
        self.logits = self.row(self.tokens.len() + self.draft_step + 1);
        if self.draft_step == self.reject_at {
            self.logits.rotate_left(1);
        }
        if let Some(&gap) = self.draft_gaps.get(self.draft_step) {
            let best = self.logits.iter().position(|x| *x == 10.).unwrap();
            self.logits.fill(0.);
            self.logits[best] = gap;
        }
        self.draft_step += 1;
        Ok(&self.logits)
    }
    fn draft_block_logits(
        &mut self,
        anchor: u32,
        drafts: usize,
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<f32>>> {
        if !self.parallel {
            return Ok(None);
        }
        let mut logits = Vec::new();
        for _ in 0..drafts {
            logits.extend_from_slice(self.draft_logits(anchor, cancel)?);
        }
        Ok(Some(logits))
    }
    fn verify_tokens(&mut self, tokens: &[u32], _: &AtomicBool) -> Result<&[f32]> {
        ensure!(self.pending.is_none(), "duplicate transaction");
        self.logits = (1..=tokens.len())
            .flat_map(|i| self.row(self.tokens.len() + i))
            .collect();
        self.pending = Some(tokens.to_vec());
        Ok(&self.logits)
    }
    fn finish_verification(&mut self, consumed: usize) -> Result<()> {
        ensure!(!self.fail_commit, "injected device failure");
        let tokens = self.pending.take().context("no transaction")?;
        self.tokens.extend_from_slice(&tokens[..consumed]);
        self.draft_step = 0;
        Ok(())
    }
    fn abort_verification(&mut self) -> Result<()> {
        self.pending = None;
        self.draft_step = 0;
        Ok(())
    }
}

#[test]
fn confidence_truncation_preserves_greedy_commit_and_budget_boundaries() -> Result<()> {
    use crate::speculative::{DraftOptions, run_round_with_options};
    for low_at in 0..=3 {
        for keep in [false, true] {
            for remaining in 1usize..=4 {
                for reject_at in 0..=3 {
                    let mut model = Model::new(3, reject_at, 40);
                    model.draft_gaps = (0..3)
                        .map(|i| if i == low_at { 0.2 } else { 12. })
                        .collect();
                    let anchor =
                        model.forward(&[4])?.iter().position(|x| *x == 10.).unwrap() as u32;
                    let mut sampler = Sampler::new(SamplingParams::default(), 5, &[4, anchor])?;
                    let round = run_round_with_options(
                        &mut model,
                        &mut sampler,
                        anchor,
                        remaining,
                        &AtomicBool::new(false),
                        DraftOptions {
                            p_min: 0.8,
                            keep_low_confidence: keep,
                            collect_confidence: true,
                            ..Default::default()
                        },
                    )?;
                    let planned = remaining - 1;
                    let proposed = if low_at < planned {
                        low_at + usize::from(keep)
                    } else {
                        planned
                    };
                    assert_eq!(round.inputs.len(), proposed + 1);
                    assert_eq!(round.draft_attempts, planned.min(low_at + 1));
                    assert_eq!(round.draft_confidence.len(), round.draft_attempts);
                    assert_eq!(round.confidence_stopped, low_at < planned);
                    for (i, &token) in round.output.iter().enumerate() {
                        assert_eq!(
                            token as usize,
                            model
                                .row(model.position() + i + 1)
                                .iter()
                                .position(|x| *x == 10.)
                                .unwrap()
                        );
                    }
                    model.finish_verification(round.output.len())?;
                    assert_eq!(model.position(), 1 + round.output.len());
                    assert!(model.pending.is_none());
                }
            }
        }
    }
    Ok(())
}

fn codec() -> Result<(tempfile::TempDir, TextCodec)> {
    let dir = tempfile::tempdir()?;
    let vocab = [("[EOS]", 0), ("A", 1), ("B", 2), ("C", 3), ("D", 4)]
        .into_iter()
        .map(|(s, i)| (s.to_owned(), i))
        .collect();
    let tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("A".into())
            .build()
            .map_err(|e| anyhow!("{e}"))?,
    );
    tokenizer
        .save(dir.path().join("tokenizer.json"), false)
        .map_err(|e| anyhow!("{e}"))?;
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"chat_template":"unused"}"#,
    )?;
    std::fs::write(dir.path().join("config.json"), r#"{"model_type":"oracle"}"#)?;
    let codec = TextCodec::load(dir.path())?;
    Ok((dir, codec))
}
fn run(
    model: &mut Model,
    codec: &TextCodec,
    max_tokens: usize,
    stop: Vec<String>,
) -> Result<(GenerationResult, String)> {
    run_with_sampling(model, codec, max_tokens, stop, SamplingParams::default())
}
fn run_with_sampling(
    model: &mut Model,
    codec: &TextCodec,
    max_tokens: usize,
    stop: Vec<String>,
    sampling: SamplingParams,
) -> Result<(GenerationResult, String)> {
    let (events, mut receiver) = mpsc::channel(64);
    let job = Job {
        preserve_tool_markers: false,
        input: crate::input::PreparedInput::text(vec![4]),
        request: GenerationRequest {
            prompt: Prompt::Tokens(vec![4]),
            sampling,
            max_tokens,
            stop,
        },
        events,
        cancel: Arc::new(AtomicBool::new(false)),
        queued: Instant::now(),
    };
    let mut cache = PrefixCache::default();
    run_generation(model, codec, &job, &mut cache)?;
    assert_eq!(cache.rows_len(), model.position());
    let mut stream = String::new();
    while let Ok(event) = receiver.try_recv() {
        match event {
            GenerationEvent::Token(s) => stream.push_str(&s),
            GenerationEvent::Finished(result) => return Ok((result, stream)),
            GenerationEvent::Error(e) => panic!("{e}"),
        }
    }
    anyhow::bail!("missing completion")
}

#[test]
fn configured_confidence_truncation_reaches_generation_and_statistics() -> Result<()> {
    let (_dir, codec) = codec()?;
    let (baseline, _) = run(&mut Model::new(0, 0, 40), &codec, 17, vec![])?;
    let mut model = Model::new(3, 3, 40);
    model.p_min = 0.8;
    model.draft_gaps = vec![12., 0.2, 12.];
    let (actual, stream) = run(&mut model, &codec, 17, vec![])?;
    assert_eq!(actual.text, baseline.text);
    assert_eq!(stream, baseline.text);
    assert_eq!(actual.completion_tokens, baseline.completion_tokens);
    assert!(actual.timings["mtp_confidence_stops"].as_u64().unwrap() > 0);
    assert!(
        actual.timings["mtp_draft_attempts"].as_u64() > actual.timings["mtp_proposed"].as_u64()
    );
    assert!(
        actual.timings["mtp_accepted"].as_u64() < actual.timings["mtp_draft_attempts"].as_u64()
    );
    Ok(())
}

#[test]
fn parallel_drafter_preserves_stream_stop_budget_and_rejection_boundaries() -> Result<()> {
    let (_dir, codec) = codec()?;
    for window in [1, 3, 7] {
        for reject_at in 0..=window {
            for maximum in [1, 2, 8, 17] {
                for stop in [vec![], vec!["BC".into()]] {
                    let (baseline, baseline_stream) =
                        run(&mut Model::new(0, 0, 12), &codec, maximum, stop.clone())?;
                    let mut model = Model::new(window, reject_at, 12);
                    model.parallel = true;
                    let (actual, stream) = run(&mut model, &codec, maximum, stop)?;
                    assert_eq!(actual.text, baseline.text);
                    assert_eq!(stream, baseline_stream);
                    assert_eq!(actual.completion_tokens, baseline.completion_tokens);
                    assert_eq!(actual.finish_reason, baseline.finish_reason);
                    assert!(model.pending.is_none());
                }
            }
        }
    }
    Ok(())
}

#[test]
fn greedy_draft_sample_and_match_preserves_seeded_target_sampling() -> Result<()> {
    let (_dir, codec) = codec()?;
    for seed in [1, 42, 1337] {
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 4,
            top_p: 0.9,
            min_p: 0.05,
            repeat_penalty: 1.1,
            presence_penalty: 0.2,
            frequency_penalty: 0.1,
            seed: Some(seed),
        };
        let make_model = |window, p_min| {
            let mut model = Model::new(window, 2, 40);
            model.target_gap = 0.4;
            model.info.eos_tokens.clear();
            model.p_min = p_min;
            model
        };
        let (baseline, _) =
            run_with_sampling(&mut make_model(0, 0.), &codec, 17, vec![], params.clone())?;
        for window in [1, 3, 7] {
            for p_min in [0., 0.25, 0.75] {
                let (actual, stream) = run_with_sampling(
                    &mut make_model(window, p_min),
                    &codec,
                    17,
                    vec![],
                    params.clone(),
                )?;
                assert_eq!(
                    actual.text, baseline.text,
                    "seed={seed},K={window},p_min={p_min}"
                );
                assert_eq!(stream, baseline.text);
                assert_eq!(actual.completion_tokens, baseline.completion_tokens);
            }
        }
    }
    Ok(())
}

#[test]
fn speculative_streaming_matches_ar_at_all_stop_and_accept_boundaries() -> Result<()> {
    let (_dir, codec) = codec()?;
    for (maximum, stop, eos) in [
        (0, vec![], 40),
        (1, vec![], 40),
        (2, vec![], 40),
        (17, vec![], 40),
        (24, vec![], 6),
        (24, vec!["D".into()], 40),
        (24, vec!["D A".into()], 40),
    ] {
        let (baseline, stream) = run(&mut Model::new(0, 0, eos), &codec, maximum, stop.clone())?;
        assert_eq!(baseline.text, stream);
        for window in [1, 2, 3, 4, 7] {
            for reject in 0..=window {
                let mut model = Model::new(window, reject, eos);
                let (actual, stream) = run(&mut model, &codec, maximum, stop.clone())?;
                assert_eq!(
                    actual.text, baseline.text,
                    "window={window},reject={reject}"
                );
                assert_eq!(actual.text, stream);
                assert_eq!(actual.finish_reason, baseline.finish_reason);
                assert_eq!(actual.completion_tokens, baseline.completion_tokens);
                assert_eq!(
                    model.position(),
                    1 + actual.completion_tokens.saturating_sub(1)
                );
                assert!(model.pending.is_none());
            }
        }
    }
    Ok(())
}
#[test]
fn failed_commit_does_not_publish_the_speculative_batch() -> Result<()> {
    let (_dir, codec) = codec()?;
    let (events, mut receiver) = mpsc::channel(64);
    let job = Job {
        preserve_tool_markers: false,
        input: crate::input::PreparedInput::text(vec![4]),
        request: GenerationRequest {
            prompt: Prompt::Tokens(vec![4]),
            sampling: SamplingParams::default(),
            max_tokens: 8,
            stop: vec![],
        },
        events,
        cancel: Arc::new(AtomicBool::new(false)),
        queued: Instant::now(),
    };
    let mut model = Model::new(4, 4, 40);
    model.fail_commit = true;
    assert!(run_generation(&mut model, &codec, &job, &mut PrefixCache::default()).is_err());
    let mut published = 0;
    while let Ok(event) = receiver.try_recv() {
        assert!(matches!(event, GenerationEvent::Token(_)));
        published += 1;
    }
    assert_eq!(
        published, 1,
        "only the initial anchor may have been published"
    );
    Ok(())
}
