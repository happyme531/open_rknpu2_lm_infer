use super::{
    Queued,
    output::Outbox,
    request::{Request, Waiting},
};
use crate::{
    engine::{EngineStats, ServingSnapshot, ServingTokens},
    model::{
        ModelInfo,
        batch::{Admission, BatchCapabilities, BatchModel, StepError},
    },
    text::TextCodec,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
use tokio::sync::{Notify, mpsc};

struct Planned {
    slot: usize,
    rows: usize,
    prefill: bool,
}

#[derive(Default)]
struct Metrics {
    steps: u64,
    tokens: ServingTokens,
    prefill_ms: f64,
    decode_ms: f64,
    mixed_ms: f64,
    last: Value,
    refills: u64,
    refill_ms: f64,
    refill_max_ms: f64,
}

impl Metrics {
    fn record(&mut self, plan: &[Planned], rows: usize, bucket: usize, ms: f64) {
        let decode = plan.iter().filter(|work| !work.prefill).count();
        let prefill = rows - decode;
        self.steps += 1;
        self.tokens.prefill += prefill as u64;
        self.tokens.decode += decode as u64;
        match (prefill, decode) {
            (0, _) => self.decode_ms += ms,
            (_, 0) => self.prefill_ms += ms,
            _ => self.mixed_ms += ms,
        }
        self.last = json!({"rows": rows, "m_bucket": bucket,
            "prefill_rows": prefill, "decode_rows": decode, "elapsed_ms": ms});
    }
}

pub(super) struct Scheduler<'a> {
    model: Box<dyn BatchModel>,
    codec: &'a TextCodec,
    caps: BatchCapabilities,
    info: ModelInfo,
    active: Vec<Option<Request<'a>>>,
    waiting: VecDeque<Waiting>,
    terminal: Vec<Outbox>,
    prefill_cursor: usize,
    metrics: Metrics,
    stats: Arc<RwLock<EngineStats>>,
    refill_started: Vec<Option<Instant>>,
}

impl<'a> Scheduler<'a> {
    pub fn new(
        model: Box<dyn BatchModel>,
        codec: &'a TextCodec,
        stats: Arc<RwLock<EngineStats>>,
    ) -> Self {
        let caps = model.capabilities();
        let info = model.info().clone();
        let active = (0..caps.max_slots).map(|_| None).collect();
        let scheduler = Self {
            model,
            codec,
            caps,
            info,
            active,
            waiting: VecDeque::new(),
            terminal: Vec::new(),
            prefill_cursor: 0,
            metrics: Metrics::default(),
            stats,
            refill_started: vec![None; caps.max_slots],
        };
        scheduler.publish(true);
        scheduler
    }

    fn publish(&self, refresh_model: bool) {
        let serving = ServingSnapshot {
            tokens: self.metrics.tokens,
            active: self.active.iter().flatten().count() as u64,
            blocked: self
                .active
                .iter()
                .flatten()
                .filter(|request| request.outbox.blocked())
                .count() as u64,
            parallel: self.caps.max_slots as u64,
            waiting: self.waiting.len() as u64,
            speculation: "none",
        };
        let model_stats = refresh_model.then(|| self.model.stats());
        let mut stats = self.stats.write().unwrap();
        if let Some(model_stats) = model_stats {
            stats.diagnostics = model_stats;
        }
        stats.serving = serving;
        stats.diagnostics["batch"] = json!({
            "parallel": serving.parallel,
            "active": serving.active,
            "blocked": serving.blocked,
            "waiting": serving.waiting,
            "terminal": self.terminal.len(),
            "steps": self.metrics.steps, "prefill_rows": serving.tokens.prefill, "decode_rows": serving.tokens.decode,
            "prefill_step_ms": self.metrics.prefill_ms, "decode_step_ms": self.metrics.decode_ms,
            "mixed_step_ms": self.metrics.mixed_ms, "vision_ms": 0.0, "last_step": self.metrics.last,
            "slot_refills": self.metrics.refills, "slot_refill_ms": self.metrics.refill_ms,
            "slot_refill_max_ms": self.metrics.refill_max_ms,
        });
    }

    fn reject(&mut self, mut waiting: Waiting, message: String) {
        waiting.outbox.error(message);
        self.terminal.push(waiting.outbox);
    }

    fn admit(&mut self) -> Result<()> {
        while let Some(waiting) = self.waiting.front() {
            let admission = self.model.try_admit(waiting.total_tokens());
            if admission == Admission::Wait {
                break;
            }
            let waiting = self.waiting.pop_front().unwrap();
            match admission {
                Admission::Ready(seq) => match waiting.admit(seq, self.codec, &self.info) {
                    Ok(request) => {
                        let slot = usize::from(seq.slot);
                        self.active[slot] = Some(request);
                        if let Some(start) = self.refill_started[slot].take() {
                            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                            self.metrics.refills += 1;
                            self.metrics.refill_ms += elapsed;
                            self.metrics.refill_max_ms = self.metrics.refill_max_ms.max(elapsed);
                        }
                    }
                    Err(outbox) => {
                        self.terminal.push(outbox);
                        self.model.release(seq)?;
                    }
                },
                Admission::CannotFit => {
                    self.reject(waiting, "request exceeds the slot context".into())
                }
                Admission::Wait => unreachable!(),
            }
        }
        Ok(())
    }

    fn retire(&mut self, slot: usize, error: Option<String>, now: Instant) -> Result<()> {
        let request = self.active[slot].take().unwrap();
        self.refill_started[slot] = (!self.waiting.is_empty()).then(Instant::now);
        let released = self.model.release(request.seq);
        let error = released
            .as_ref()
            .err()
            .map(|err| format!("batch device failure: {err:#}"))
            .or(error);
        if let Some(message) = error {
            let mut outbox = request.outbox;
            outbox.error(message);
            self.terminal.push(outbox);
        } else if request.is_complete() && !request.outbox.cancelled() {
            self.terminal.push(request.finish(now));
        }
        released
    }

    fn reap(&mut self, now: Instant) -> Result<bool> {
        self.waiting.retain(|request| !request.outbox.cancelled());
        let mut released = false;
        for slot in 0..self.active.len() {
            if let Some(request) = &mut self.active[slot]
                && (!request.outbox.flush(now) || request.is_complete())
            {
                self.retire(slot, None, now)?;
                released = true;
            }
        }
        Ok(released)
    }

    fn plan(&mut self) -> Vec<Planned> {
        let mut plan = Vec::with_capacity(self.caps.max_slots);
        for (slot, request) in self.active.iter().enumerate() {
            if request
                .as_ref()
                .is_some_and(|r| !r.outbox.blocked() && !r.is_prefill())
            {
                plan.push(Planned {
                    slot,
                    rows: 1,
                    prefill: false,
                });
            }
        }
        let remaining = self.caps.max_rows - plan.len();
        if remaining != 0 {
            for offset in 0..self.active.len() {
                let slot = (self.prefill_cursor + offset) % self.active.len();
                if let Some(r) = &self.active[slot]
                    && !r.outbox.blocked()
                    && r.is_prefill()
                {
                    let rows = r
                        .prefill_remaining()
                        .min(remaining)
                        .min(self.caps.max_span_rows);
                    plan.push(Planned {
                        slot,
                        rows,
                        prefill: true,
                    });
                    self.prefill_cursor = (slot + 1) % self.active.len();
                    break;
                }
            }
        }
        plan
    }

    fn execute(&mut self, plan: &[Planned]) -> Result<()> {
        let start = Instant::now();
        let spans: Vec<_> = plan
            .iter()
            .map(|work| self.active[work.slot].as_ref().unwrap().span(work.rows))
            .collect();
        let result = self.model.execute_step(&spans);
        let now = Instant::now();
        let ms = (now - start).as_secs_f64() * 1000.;
        let out = match result {
            Ok(out) => out,
            Err(StepError::Rejected(error)) => {
                let message = format!("batch step rejected: {error:#}");
                for work in plan {
                    self.retire(work.slot, Some(message.clone()), now)?;
                }
                return Ok(());
            }
            Err(StepError::Device(error)) => return Err(error),
        };
        ensure!(
            out.rows == spans.iter().map(|span| span.tokens.len()).sum::<usize>()
                && out
                    .logits_rows()
                    .map(|(seq, _)| seq)
                    .eq(spans.iter().filter(|span| span.logits).map(|span| span.seq)),
            "model step returned logits for the wrong batch"
        );
        self.metrics.record(plan, out.rows, out.bucket, ms);
        for work in plan {
            self.active[work.slot]
                .as_mut()
                .unwrap()
                .advance(work.rows, ms, now);
        }
        let mut errors = Vec::new();
        for (id, logits) in out.logits_rows() {
            let slot = usize::from(id.slot);
            let request = self.active[slot].as_mut().unwrap();
            debug_assert_eq!(request.seq, id);
            if !request.outbox.cancelled()
                && let Err(error) = request.sample(logits, &self.info.eos_tokens, now)
            {
                errors.push((slot, format!("{error:#}")));
            }
        }
        for (slot, message) in errors {
            self.retire(slot, Some(message), now)?;
        }
        Ok(())
    }

    fn poison(
        &mut self,
        error: anyhow::Error,
        rx: &mut mpsc::UnboundedReceiver<Queued>,
        wake: &Arc<Notify>,
    ) {
        rx.close();
        let message = format!("batch device failure: {error:#}");
        tracing::error!("{message}");
        for request in self.active.iter_mut().filter_map(Option::take) {
            let mut output = request.outbox;
            output.error(message.clone());
            self.terminal.push(output);
        }
        while let Ok(queued) = rx.try_recv() {
            self.waiting.push_back(Waiting::new(queued, wake.clone()));
        }
        while let Some(waiting) = self.waiting.pop_front() {
            self.reject(waiting, message.clone());
        }
        self.publish(false);
    }

    pub async fn run(
        mut self,
        mut rx: mpsc::UnboundedReceiver<Queued>,
        wake: Arc<Notify>,
        shutdown: Arc<AtomicBool>,
    ) {
        while !shutdown.load(Ordering::Acquire) {
            // The relay tasks share this runtime and run between device steps.
            tokio::task::yield_now().await;
            // A continuous stream of arrivals must leave time for active requests.
            for _ in 0..rx.len() {
                let Ok(queued) = rx.try_recv() else { break };
                self.waiting.push_back(Waiting::new(queued, wake.clone()));
            }
            let now = Instant::now();
            let released = match self.reap(now).and_then(|released| {
                self.admit()?;
                Ok(released)
            }) {
                Ok(released) => released,
                Err(error) => {
                    self.poison(error, &mut rx, &wake);
                    continue;
                }
            };
            self.terminal
                .retain_mut(|output| output.flush(now) && output.blocked());
            if self.waiting.is_empty() {
                self.refill_started.fill(None);
            }
            let plan = self.plan();
            self.publish(released);
            if !plan.is_empty() {
                if let Err(error) = self.execute(&plan) {
                    self.poison(error, &mut rx, &wake);
                }
                continue;
            }
            let deadline = self
                .active
                .iter()
                .flatten()
                .map(|r| &r.outbox)
                .chain(&self.terminal)
                .filter_map(Outbox::deadline)
                .min();
            if let Some(deadline) = deadline {
                tokio::select! {
                    _ = wake.notified() => (),
                    _ = tokio::time::sleep_until(deadline.into()) => (),
                }
            } else {
                wake.notified().await;
            }
        }
    }
}
