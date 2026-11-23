use super::{Queued, WaitingCount, output::Outbox};
use crate::{
    engine::{GeneratedText, GenerationEvent, GenerationRequest, GenerationResult, Job},
    input::PreparedInput,
    model::{
        ModelInfo,
        batch::{SeqId, Span},
    },
    sampling::Sampler,
    text::TextCodec,
};
use anyhow::Result;
use serde_json::json;
use std::{sync::Arc, time::Instant};
use tokio::sync::Notify;

pub(super) struct Waiting {
    id: u64,
    input: PreparedInput,
    request: GenerationRequest,
    preserve_tool_markers: bool,
    queued: Instant,
    pub outbox: Outbox,
    _waiting: WaitingCount,
}

impl Waiting {
    pub fn new(queued: Queued, wake: Arc<Notify>) -> Self {
        let Queued { id, job, waiting } = queued;
        let Job {
            preserve_tool_markers,
            input,
            request,
            events,
            cancel,
            queued,
        } = job;
        Self {
            id,
            input,
            request,
            preserve_tool_markers,
            queued,
            outbox: Outbox::new(events, cancel, wake),
            _waiting: waiting,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.input.tokens.len() + self.request.max_tokens
    }

    pub fn admit<'a>(
        self,
        seq: SeqId,
        codec: &'a TextCodec,
        info: &ModelInfo,
    ) -> std::result::Result<Request<'a>, Outbox> {
        let sampler = Sampler::new(self.request.sampling, info.vocab_size, &self.input.tokens);
        let output = GeneratedText::new(
            codec,
            self.preserve_tool_markers,
            self.request.max_tokens,
            self.request.stop,
        );
        let (sampler, output) = match sampler.and_then(|sampler| Ok((sampler, output?))) {
            Ok(pair) => pair,
            Err(error) => {
                let mut outbox = self.outbox;
                outbox.error(format!("{error:#}"));
                return Err(outbox);
            }
        };
        Ok(Request {
            id: self.id,
            seq,
            input: self.input,
            queued: self.queued,
            admitted: Instant::now(),
            outbox: self.outbox,
            sampler,
            output,
            position: 0,
            pending: None,
            ttft_ms: None,
            last_emit: None,
            itl_ms: Vec::new(),
            prefill_ms: 0.,
            decode_ms: 0.,
            first_decode: None,
        })
    }
}

struct Pending {
    token: [u32; 1],
    position: [[usize; 3]; 1],
}

pub(super) struct Request<'a> {
    pub id: u64,
    pub seq: SeqId,
    input: PreparedInput,
    queued: Instant,
    admitted: Instant,
    pub outbox: Outbox,
    sampler: Sampler,
    output: GeneratedText<'a>,
    position: usize,
    pending: Option<Pending>,
    ttft_ms: Option<f64>,
    last_emit: Option<Instant>,
    itl_ms: Vec<f64>,
    prefill_ms: f64,
    decode_ms: f64,
    first_decode: Option<Instant>,
}

impl Request<'_> {
    pub fn is_prefill(&self) -> bool {
        self.position < self.input.tokens.len()
    }

    pub fn prefill_remaining(&self) -> usize {
        self.input.tokens.len() - self.position
    }

    pub fn is_complete(&self) -> bool {
        !self.is_prefill() && self.output.reason().is_some()
    }

    pub fn span(&self, rows: usize) -> Span<'_> {
        let (tokens, positions, logits) = if let Some(pending) = &self.pending {
            (&pending.token[..], &pending.position[..], true)
        } else {
            let end = self.position + rows;
            (
                &self.input.tokens[self.position..end],
                &self.input.positions[self.position..end],
                end == self.input.tokens.len() && self.output.reason().is_none(),
            )
        };
        Span {
            seq: self.seq,
            start: self.position,
            tokens,
            positions,
            embeddings: &[],
            logits,
        }
    }

    pub fn advance(&mut self, rows: usize, step_ms: f64, now: Instant) {
        if self.is_prefill() {
            self.prefill_ms += step_ms;
        } else {
            self.decode_ms += step_ms;
        }
        self.position += rows;
        if !self.is_prefill() {
            self.first_decode.get_or_insert(now);
        }
    }

    fn emit(&mut self, output: String, now: Instant) -> Result<()> {
        if output.is_empty() {
            return Ok(());
        }
        self.ttft_ms
            .get_or_insert_with(|| (now - self.queued).as_secs_f64() * 1000.);
        if let Some(last) = self.last_emit.replace(now)
            && last != now
        {
            self.itl_ms.push((now - last).as_secs_f64() * 1000.);
        }
        self.outbox.push(GenerationEvent::Token(output))
    }

    pub fn sample(&mut self, logits: &[f32], eos: &[u32], now: Instant) -> Result<()> {
        let token = self.sampler.sample(logits)?;
        if let Some(piece) = self.output.push(token, eos, &mut self.sampler)? {
            self.emit(piece, now)?;
        }
        if self.output.reason().is_none() {
            self.pending = Some(Pending {
                token: [token],
                position: [[self.position; 3]],
            });
        }
        Ok(())
    }

    pub fn finish(mut self, now: Instant) -> Outbox {
        let tail = self.output.flush();
        if let Err(error) = self.emit(tail, now) {
            self.outbox.error(format!("{error:#}"));
            return self.outbox;
        }
        self.itl_ms.sort_by(f64::total_cmp);
        let percentile = |p: f64| {
            let index = (self.itl_ms.len() as f64 * p).ceil() as usize;
            self.itl_ms.get(index.saturating_sub(1)).copied()
        };
        let completion_tokens = self.output.generated();
        let finish_reason = self.output.reason().unwrap().into();
        let result = GenerationResult {
            text: self.output.into_text(),
            finish_reason,
            prompt_tokens: self.input.tokens.len(),
            completion_tokens,
            cached_tokens: 0,
            timings: json!({
                "queue_ms": (self.admitted - self.queued).as_secs_f64() * 1000.,
                "prefill_ms": self.prefill_ms,
                "ttft_ms": self.ttft_ms,
                "decode_ms": self.first_decode.map(|first| (now - first).as_secs_f64() * 1000.),
                "decode_model_ms": self.decode_ms,
                "itl_p50_ms": percentile(0.50), "itl_p95_ms": percentile(0.95),
                "itl_max_ms": self.itl_ms.last(), "itl_samples": self.itl_ms.len(),
                "generation_ms": (now - self.queued).as_secs_f64() * 1000.,
            }),
            stats: json!({"request_id": self.id, "slot": self.seq.slot, "target_consumed_tokens": self.position}),
        };
        tracing::info!(request_id = self.id, slot = self.seq.slot, prompt_tokens = result.prompt_tokens,
            completion_tokens = result.completion_tokens, timings = %result.timings, "batch generation finished");
        if let Err(error) = self.outbox.push(GenerationEvent::Finished(result)) {
            self.outbox.error(format!("{error:#}"));
        }
        self.outbox
    }
}
