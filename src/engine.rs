use crate::{
    model::{self, LanguageModel, ModelInfo, ModelOptions},
    sampling::Sampler,
    text::{OutputDecoder, StopFilter, TextCodec},
};
use anyhow::{Context, Result, anyhow, ensure};
use serde_json::{Value, json};
use std::{
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
mod context;
#[cfg(test)]
mod tests;
use context::{PrefixCache, validate_request_tokens};

mod types;
pub use types::*;

struct Job {
    preserve_tool_markers: bool,
    input: crate::input::PreparedInput,
    request: GenerationRequest,
    events: mpsc::Sender<GenerationEvent>,
    cancel: Arc<AtomicBool>,
    queued: Instant,
}

struct EngineInner {
    sender: Option<mpsc::Sender<Job>>,
    info: ModelInfo,
    codec: Arc<TextCodec>,
    stats: Arc<RwLock<Value>>,
    active: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    shutdown: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Ok(active) = self.active.lock()
            && let Some(cancel) = active.as_ref()
        {
            cancel.store(true, Ordering::Relaxed);
        }
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

impl Engine {
    pub fn start(options: ModelOptions) -> Result<Self> {
        let codec = Arc::new(TextCodec::load(&options.path)?);
        let worker_codec = codec.clone();
        let (sender, mut receiver) = mpsc::channel::<Job>(4);
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let stats = Arc::new(RwLock::new(Value::Null));
        let worker_stats = stats.clone();
        let active = Arc::new(Mutex::new(None));
        let worker_active = active.clone();
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker_shutdown = shutdown.clone();
        let worker = thread::Builder::new()
            .name("rknpu-inference".into())
            .spawn(move || {
                let loaded = (|| -> Result<_> {
                    let chunk = options.chunk;
                    let mut options = options;
                    options.tune_search = false;
                    let mut model = model::load(options)?;
                    model.prepare(chunk)?;
                    Ok(model)
                })();
                let mut model = match loaded {
                    Ok(loaded) => loaded,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                if let Ok(mut latest) = worker_stats.write() {
                    *latest = model.stats();
                }
                if ready_tx.send(Ok(model.info().clone())).is_err() {
                    return;
                }
                let mut cache = PrefixCache::default();
                while let Some(job) = receiver.blocking_recv() {
                    if worker_shutdown.load(Ordering::Relaxed) || job.cancel.load(Ordering::Relaxed)
                    {
                        continue;
                    }
                    if let Ok(mut running) = worker_active.lock() {
                        *running = Some(job.cancel.clone());
                    }
                    let outcome = run_generation(model.as_mut(), &worker_codec, &job, &mut cache);
                    if let Err(error) = outcome {
                        let _ = model.reset();
                        cache.clear();
                        if !job.cancel.load(Ordering::Relaxed) {
                            let _ = send_event(&job, GenerationEvent::Error(format!("{error:#}")));
                        }
                    }
                    if let Ok(mut latest) = worker_stats.write() {
                        *latest = model.stats();
                    }
                    if let Ok(mut running) = worker_active.lock() {
                        *running = None;
                    }
                }
            })?;
        let info = ready_rx
            .recv()
            .context("inference worker stopped during loading")??;
        Ok(Self {
            inner: Arc::new(EngineInner {
                sender: Some(sender),
                info,
                codec,
                stats,
                active,
                shutdown,
                worker: Some(worker),
            }),
        })
    }

    pub fn info(&self) -> &ModelInfo {
        &self.inner.info
    }
    pub fn stats(&self) -> Value {
        self.inner
            .stats
            .read()
            .map(|v| v.clone())
            .unwrap_or(Value::Null)
    }
    pub fn tokenize(&self, prompt: &Prompt) -> Result<Vec<u32>> {
        Ok(self.inner.codec.prepare(prompt)?.tokens)
    }
    pub fn prompt_opens_thinking(&self, prompt: &Prompt) -> Result<bool> {
        Ok(matches!(prompt, Prompt::Chat { .. })
            && self
                .inner
                .codec
                .format(prompt)?
                .trim_end()
                .ends_with("<think>"))
    }
    pub fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::Relaxed);
        if let Ok(active) = self.inner.active.lock()
            && let Some(cancel) = active.as_ref()
        {
            cancel.store(true, Ordering::Relaxed);
        }
    }

    pub fn generate(&self, request: GenerationRequest) -> Result<GenerationHandle> {
        self.generate_with_tool_markers(request, false)
    }

    pub(crate) fn generate_with_tool_markers(
        &self,
        request: GenerationRequest,
        preserve_tool_markers: bool,
    ) -> Result<GenerationHandle> {
        ensure!(
            !self.inner.shutdown.load(Ordering::Relaxed),
            "inference worker is shutting down"
        );
        request.sampling.validate()?;
        ensure!(
            request.stop.len() <= 16,
            "at most 16 stop sequences are supported"
        );
        ensure!(
            request.stop.iter().all(|stop| !stop.is_empty()),
            "stop sequences must not be empty"
        );
        let input = self.inner.codec.prepare(&request.prompt)?;
        validate_request_tokens(&input.tokens, request.max_tokens, self.info())?;
        let (events_tx, events) = mpsc::channel(32);
        let cancel = Arc::new(AtomicBool::new(false));
        let job = Job {
            preserve_tool_markers,
            input,
            request,
            events: events_tx,
            cancel: cancel.clone(),
            queued: Instant::now(),
        };
        match self.inner.sender.as_ref().unwrap().try_send(job) {
            Ok(()) => Ok(GenerationHandle { events, cancel }),
            Err(mpsc::error::TrySendError::Full(_)) => Err(EngineBusy.into()),
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(anyhow!("inference worker is unavailable"))
            }
        }
    }
}

fn send_event(job: &Job, mut event: GenerationEvent) -> Result<()> {
    loop {
        ensure!(!job.cancel.load(Ordering::Relaxed), "generation cancelled");
        match job.events.try_send(event) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Closed(_)) => anyhow::bail!("generation cancelled"),
            Err(mpsc::error::TrySendError::Full(returned)) => {
                event = returned;
                thread::sleep(Duration::from_millis(5));
            }
        }
    }
}

fn run_generation(
    model: &mut dyn LanguageModel,
    codec: &TextCodec,
    job: &Job,
    cache: &mut PrefixCache,
) -> Result<()> {
    let start = Instant::now();
    let queue_ms = job.queued.elapsed().as_secs_f64() * 1000.0;
    let mut cached_tokens = cache.reusable(&job.input, model.position());
    if cached_tokens > 0 && cached_tokens < model.position() {
        if model.truncate(cached_tokens)? {
            cache.truncate(cached_tokens);
        } else {
            cached_tokens = 0;
        }
    }
    if cached_tokens == 0 {
        model.reset()?;
        cache.clear();
    }
    let remaining = &job.input.tokens[cached_tokens..];
    let prepare_start = Instant::now();
    model.reserve(job.input.tokens.len() + job.request.max_tokens)?;
    if !remaining.is_empty() {
        model.prepare(remaining.len())?;
    }
    let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_start = Instant::now();
    if !remaining.is_empty() {
        cache.logits.clear();
        cache.logits.extend_from_slice(model.forward_prepared(
            &job.input,
            cached_tokens,
            &job.cancel,
        )?);
        cache.extend_input(&job.input, cached_tokens);
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let mut sampler = Sampler::new(
        job.request.sampling.clone(),
        model.info().vocab_size,
        &job.input.tokens,
    )?;
    let mut filter = StopFilter::new(job.request.stop.clone())?;
    let mut decoder = OutputDecoder::new(&codec.tokenizer, job.preserve_tool_markers);
    let mut text = String::new();
    let mut generated = 0;
    let mut reason = "length";
    let mut ttft_ms = None;
    let mut decode_calls = 0;
    let mut decode_model_ms = 0.0;
    let decode_start = Instant::now();
    let mut pending = None;
    let mut draft_ms = 0.;
    let mut verify_ms = 0.;
    let mut repair_ms = 0.;
    let mut proposed = 0;
    let mut draft_attempts = 0;
    let mut confidence_stops = 0;
    let mut accepted = 0;
    let mut consumed_tokens = 0;
    let mut accepted_histogram = vec![0usize; model.speculation_window() + 1];
    let mut spec_rounds = 0;
    let mut verify_input_rows = 0;
    let mut verify_physical_rows = 0;
    let mut replay_rows = 0;
    let mut repair_rows = 0;
    let mut events = Vec::new();
    while generated < job.request.max_tokens {
        ensure!(!job.cancel.load(Ordering::Relaxed), "generation cancelled");
        let call = Instant::now();
        let round = if let Some(anchor) = pending.filter(|_| model.speculation_window() > 0) {
            Some(crate::speculative::run_round(
                model,
                &mut sampler,
                anchor,
                job.request.max_tokens - generated,
                &job.cancel,
            )?)
        } else {
            None
        };
        let sampled;
        let batch: &[u32] = if let Some(round) = &round {
            draft_ms += round.draft_ms;
            verify_ms += round.verify_ms;
            proposed += round.inputs.len() - 1;
            draft_attempts += round.draft_attempts;
            confidence_stops += usize::from(round.confidence_stopped);
            &round.output
        } else {
            if let Some(anchor) = pending {
                let logits = model.forward_cancellable(&[anchor], &job.cancel)?;
                cache.logits.clear();
                cache.logits.extend_from_slice(logits);
                cache.append_token(anchor);
                decode_calls += 1;
                consumed_tokens += 1;
            }
            sampled = [sampler.sample(&cache.logits)?];
            &sampled
        };
        let mut emitted = 0;
        let mut stopped = false;
        for &token in batch {
            ensure!(!job.cancel.load(Ordering::Relaxed), "generation cancelled");
            generated += 1;
            emitted += 1;
            pending = Some(token);
            if model.info().eos_tokens.contains(&token) {
                stopped = true;
                reason = "stop";
                break;
            }
            sampler.observe(token);
            if let Some(piece) = decoder
                .step(token)
                .map_err(|e| anyhow!("stream decode: {e}"))?
            {
                let (output, stop) = filter.push(&piece);
                if !output.is_empty() {
                    events.push(output);
                }
                if stop {
                    stopped = true;
                    reason = "stop";
                    break;
                }
            }
        }
        if let Some(round) = round {
            let repair_start = Instant::now();
            model.finish_verification(emitted)?;
            repair_ms += repair_start.elapsed().as_secs_f64() * 1000.;
            let vocab = model.info().vocab_size;
            cache.logits.clear();
            cache
                .logits
                .extend_from_slice(&round.logits[(emitted - 1) * vocab..emitted * vocab]);
            for &input in &round.inputs[..emitted] {
                cache.append_token(input);
            }
            accepted += round.accepted.min(emitted);
            accepted_histogram[round.accepted.min(emitted)] += 1;
            spec_rounds += 1;
            verify_input_rows += round.inputs.len();
            verify_physical_rows += round.inputs.len().next_power_of_two();
            if emitted < round.inputs.len() {
                replay_rows += emitted;
            }
            repair_rows += emitted;
            decode_calls += 1;
            consumed_tokens += emitted;
        }
        if generated > emitted {
            decode_model_ms += call.elapsed().as_secs_f64() * 1000.;
        }
        for output in events.drain(..) {
            text.push_str(&output);
            if ttft_ms.is_none() {
                ttft_ms = Some(start.elapsed().as_secs_f64() * 1000.);
            }
            send_event(job, GenerationEvent::Token(output))?;
        }
        if stopped {
            break;
        }
    }
    let tail = filter.finish();
    if !tail.is_empty() {
        text.push_str(&tail);
        send_event(job, GenerationEvent::Token(tail))?;
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let result = GenerationResult {
        text,
        finish_reason: reason.into(),
        prompt_tokens: job.input.tokens.len(),
        completion_tokens: generated,
        cached_tokens,
        timings: json!({"queue_ms":queue_ms,"prepare_ms":prepare_ms,"prefill_ms":prefill_ms,
            "ttft_ms":ttft_ms,"decode_ms":decode_ms,"decode_model_ms":decode_model_ms,"decode_calls":decode_calls,
            "decode_tokens_per_second": if decode_model_ms > 0.0 { Some(consumed_tokens as f64 * 1000.0 / decode_model_ms) } else { None },
            "mtp_proposed":proposed,"mtp_accepted":accepted,
            "spec_proposed":proposed,"spec_accepted":accepted,
            "spec_draft_attempts":draft_attempts,"spec_confidence_stops":confidence_stops,
            "spec_acceptance":if proposed>0 {Some(accepted as f64/proposed as f64)} else {None},
            "spec_accept_length":if spec_rounds>0 {Some(consumed_tokens as f64/spec_rounds as f64)} else {None},
            "spec_draft_ms":draft_ms,"spec_verify_ms":verify_ms,"spec_repair_ms":repair_ms,
            "mtp_draft_attempts":draft_attempts,"mtp_confidence_stops":confidence_stops,
            "mtp_accepted_per_attempt":if draft_attempts > 0 { Some(accepted as f64 / draft_attempts as f64) } else { None },
            "accepted_prefix_histogram":accepted_histogram,"spec_rounds":spec_rounds,
            "verify_input_rows":verify_input_rows,"verify_physical_rows":verify_physical_rows,
            "target_consumed_tokens":model.position(),"decode_consumed_tokens":consumed_tokens,"emitted_tokens":generated,
            "replay_rows":replay_rows,"draft_repair_rows":repair_rows,"fallback_reason":null,
            "mtp_draft_ms":draft_ms,"mtp_verify_ms":verify_ms,"mtp_repair_ms":repair_ms,
            "generation_ms":start.elapsed().as_secs_f64()*1000.0}),
        stats: model.stats(),
    };
    tracing::info!(prompt_tokens = result.prompt_tokens, cached_tokens = result.cached_tokens,
        completion_tokens = result.completion_tokens, timings = %result.timings,
        tune_cache_hits = %result.stats["tune_cache_hits"],
        tune_cache_misses = %result.stats["tune_cache_misses"],
        tune_calls = %result.stats["tune_calls"],
        weight_reload_calls = %result.stats["weight_reload_calls"],
        weight_profile_creations = %result.stats["weight_profile_creations"],
        strategy_fallback_calls = %result.stats["strategy_fallback_calls"],
        strategy_fallbacks = %result.stats["strategy_fallbacks"], "generation finished");
    send_event(job, GenerationEvent::Finished(result))
}
