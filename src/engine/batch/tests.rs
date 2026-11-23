use super::*;
use crate::{
    engine::{GenerationEvent, GenerationHandle, GenerationRequest, GenerationResult, Prompt},
    input::PreparedInput,
    model::{
        ModelInfo,
        batch::{Admission, BatchCapabilities, BatchModel, SeqId, Span, StepError, StepOutput},
    },
    sampling::SamplingParams,
};
use anyhow::anyhow;
use scheduler::Scheduler;
use serde_json::{Value, json};
use std::time::{Duration, Instant};

struct Model {
    info: ModelInfo,
    positions: [Option<usize>; 2],
    generations: [u32; 2],
    logits: Vec<f32>,
    fail: Arc<AtomicBool>,
    resets: usize,
    mixed: usize,
}

impl Model {
    fn new(fail: Arc<AtomicBool>) -> Self {
        Self {
            info: ModelInfo {
                id: "test".into(),
                architecture: "test".into(),
                context: 128,
                vocab_size: 5,
                eos_tokens: vec![0],
            },
            positions: [None; 2],
            generations: [0; 2],
            logits: Vec::new(),
            fail,
            resets: 0,
            mixed: 0,
        }
    }
}

impl BatchModel for Model {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> BatchCapabilities {
        BatchCapabilities {
            max_slots: 2,
            max_rows: 4,
            max_span_rows: 4,
            vision: false,
        }
    }

    fn try_admit(&mut self, total_tokens: usize) -> Admission {
        if total_tokens > self.info.context {
            return Admission::CannotFit;
        }
        let Some(slot) = self.positions.iter().position(Option::is_none) else {
            return Admission::Wait;
        };
        self.positions[slot] = Some(0);
        self.generations[slot] += 1;
        Admission::Ready(SeqId {
            slot: slot as u16,
            generation: self.generations[slot],
        })
    }

    fn release(&mut self, seq: SeqId) -> Result<()> {
        let slot = usize::from(seq.slot);
        assert_eq!(seq.generation, self.generations[slot]);
        assert!(self.positions[slot].take().is_some());
        self.resets += 1;
        Ok(())
    }

    fn execute_step(
        &mut self,
        spans: &[Span<'_>],
    ) -> std::result::Result<StepOutput<'_>, StepError> {
        if self.fail.load(Ordering::Acquire) {
            return Err(StepError::Device(anyhow!("test submit failure")));
        }
        let rows = spans.iter().map(|span| span.tokens.len()).sum::<usize>();
        assert!(rows <= 4);
        self.mixed +=
            usize::from(spans.len() > 1 && spans.iter().any(|span| span.tokens.len() > 1));
        let mut sequences = Vec::new();
        self.logits.clear();
        for span in spans {
            let slot = usize::from(span.seq.slot);
            assert_eq!(self.positions[slot], Some(span.start));
            assert_eq!(self.generations[slot], span.seq.generation);
            self.positions[slot] = Some(span.start + span.tokens.len());
            if span.logits {
                sequences.push(span.seq);
                self.logits.extend([-99.0, 1.0, 0.85, 0.75, -99.0]);
            }
        }
        StepOutput::new(rows, sequences, &self.logits, self.info.vocab_size)
            .map_err(StepError::Device)
    }

    fn stats(&self) -> Value {
        json!({"resets": self.resets, "mixed": self.mixed})
    }
}

fn submit(
    sender: &Sender,
    prompt: usize,
    max_tokens: usize,
    seed: u64,
    stops: &[&str],
) -> Result<GenerationHandle> {
    let (events_tx, events) = mpsc::channel(1);
    let cancel = Arc::new(AtomicBool::new(false));
    let tokens = vec![4; prompt];
    sender.send(Job {
        preserve_tool_markers: false,
        input: PreparedInput::text(tokens.clone()),
        request: GenerationRequest {
            prompt: Prompt::Tokens(tokens),
            max_tokens,
            sampling: SamplingParams {
                seed: Some(seed),
                temperature: 1.,
                ..SamplingParams::default()
            },
            stop: stops.iter().map(|s| (*s).to_owned()).collect(),
            cache_prompt: None,
        },
        events: events_tx,
        cancel: cancel.clone(),
        queued: Instant::now(),
    })?;
    Ok(GenerationHandle { events, cancel })
}

async fn finish(mut handle: GenerationHandle) -> Result<GenerationResult> {
    let mut text = String::new();
    while let Some(event) = handle.events.recv().await {
        match event {
            GenerationEvent::Token(piece) => text.push_str(&piece),
            GenerationEvent::Finished(result) => {
                assert_eq!(result.text, text);
                return Ok(result);
            }
            GenerationEvent::Error(error) => return Err(anyhow!("{error}")),
        }
    }
    Err(anyhow!("stream closed before finishing"))
}

async fn until(condition: impl Fn() -> bool) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("scheduler state deadline");
}

async fn exercise(
    test: impl AsyncFnOnce(&Sender, &Arc<RwLock<EngineStats>>, Arc<AtomicBool>) -> Result<()>,
) -> Result<()> {
    let (_dir, codec) = crate::engine::tests::codec()?;
    let (tx, rx) = mpsc::unbounded_channel();
    let sender = Sender {
        tx,
        waiting: Arc::default(),
        wake: Arc::default(),
        next_id: AtomicU64::new(1),
    };
    let stats = Arc::new(RwLock::new(EngineStats::default()));
    let shutdown = Arc::new(AtomicBool::new(false));
    let fail = Arc::new(AtomicBool::new(false));
    let scheduler = Scheduler::new(Box::new(Model::new(fail.clone())), &codec, stats.clone());
    let client = async {
        let outcome =
            tokio::time::timeout(Duration::from_secs(5), test(&sender, &stats, fail)).await;
        shutdown.store(true, Ordering::Release);
        sender.wake.notify_one();
        outcome.context("test client timed out")?
    };
    let (_, outcome) = tokio::join!(
        scheduler.run(rx, sender.wake.clone(), shutdown.clone()),
        client
    );
    outcome
}

#[tokio::test]
async fn mixed_requests_keep_seed_and_stops_independent() -> Result<()> {
    exercise(async |sender, stats, _| {
        let baseline = finish(submit(sender, 1, 12, 7, &[])?).await?;
        let a = submit(sender, 1, 12, 7, &[])?;
        let b = submit(sender, 11, 12, 8, &[])?;
        let (a, b) = tokio::try_join!(finish(a), finish(b))?;
        assert_eq!(a.text, baseline.text);
        assert_ne!(a.text, b.text);
        assert_eq!(a.cached_tokens, 0);
        assert_eq!(b.completion_tokens, 12);
        {
            let stats = stats.read().unwrap();
            assert!(stats.diagnostics["mixed"].as_u64().unwrap() > 0);
            assert_eq!(stats.serving.tokens.prefill, 13);
            assert_eq!(stats.serving.tokens.decode, 33);
        }
        let a = submit(sender, 1, 12, 7, &["A", "B", "C"])?;
        let b = submit(sender, 11, 12, 8, &[])?;
        let (a, b) = tokio::try_join!(finish(a), finish(b))?;
        assert_eq!(a.finish_reason, "stop");
        assert_eq!(a.text, "");
        assert_eq!(b.finish_reason, "length");
        assert_eq!(b.completion_tokens, 12);
        assert_eq!(
            finish(submit(sender, 1, 0, 7, &[])?)
                .await?
                .completion_tokens,
            0
        );
        Ok(())
    })
    .await
}

#[tokio::test]
async fn backpressure_and_cancellation_free_only_their_slots() -> Result<()> {
    exercise(async |sender, stats, _| {
        let slow = submit(sender, 1, 100, 42, &[])?;
        let fast = submit(sender, 1, 100, 42, &[])?;
        until(|| stats.read().unwrap().serving.blocked == 2).await;
        let queued = submit(sender, 4, 8, 42, &[])?;
        let cancelled = (0..8)
            .map(|seed| submit(sender, 4, 8, seed, &[]))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(sender.waiting(), 9);
        until(|| stats.read().unwrap().serving.waiting == 9).await;
        drop(cancelled);
        until(|| sender.waiting() == 1).await;
        drop(fast);
        let result = finish(queued).await?;
        assert_eq!(result.completion_tokens, 8);
        until(|| stats.read().unwrap().serving.blocked == 1).await;
        drop(slow);
        until(|| stats.read().unwrap().serving.active == 0).await;
        assert_eq!(stats.read().unwrap().diagnostics["resets"], 3);
        Ok(())
    })
    .await
}

#[tokio::test]
async fn device_failure_closes_admission_and_ends_active_and_waiting_streams() -> Result<()> {
    exercise(async |sender, stats, fail| {
        let slow = submit(sender, 1, 100, 7, &[])?;
        let fast = submit(sender, 1, 100, 8, &[])?;
        until(|| stats.read().unwrap().serving.blocked == 2).await;
        let queued = submit(sender, 4, 8, 42, &[])?;
        fail.store(true, Ordering::Release);
        let inspect = async |mut handle: GenerationHandle| {
            while let Some(event) = handle.events.recv().await {
                if let GenerationEvent::Error(error) = event {
                    assert!(error.contains("test submit failure"));
                    return Ok(());
                }
            }
            Err(anyhow!("poisoned stream lacked a terminal error"))
        };
        tokio::try_join!(inspect(fast), inspect(slow), inspect(queued))?;
        assert!(!sender.available());
        assert!(matches!(submit(sender, 4, 8, 42, &[]), Err(err) if err.is::<EngineUnavailable>()));
        assert_eq!(sender.waiting(), 0);
        Ok(())
    })
    .await
}
