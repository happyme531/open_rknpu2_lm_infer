use super::super::GenerationEvent;
use anyhow::{Result, ensure};
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{Notify, mpsc};

pub(super) const TIMEOUT: Duration = Duration::from_secs(30);
const MAX_BYTES: usize = 1024 * 1024;

pub(super) struct Outbox {
    tx: mpsc::Sender<GenerationEvent>,
    cancel: Arc<AtomicBool>,
    events: VecDeque<GenerationEvent>,
    bytes: usize,
    blocked_since: Option<Instant>,
}

fn bytes(event: &GenerationEvent) -> usize {
    match event {
        GenerationEvent::Token(text) | GenerationEvent::Error(text) => text.len(),
        GenerationEvent::Finished(result) => result.text.len(),
    }
}

impl Outbox {
    pub fn new(
        downstream: mpsc::Sender<GenerationEvent>,
        cancel: Arc<AtomicBool>,
        wake: Arc<Notify>,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel(1);
        let relay_cancel = cancel.clone();
        tokio::spawn(async move {
            loop {
                let ready = tokio::select! {
                    _ = downstream.closed() => break,
                    event = rx.recv() => event,
                };
                wake.notify_one();
                let Some(event) = ready else { return };
                match tokio::time::timeout(TIMEOUT, downstream.send(event)).await {
                    Ok(Ok(())) => (),
                    _ => break,
                }
            }
            relay_cancel.store(true, Ordering::Release);
            wake.notify_one();
        });
        Self {
            tx,
            cancel,
            events: VecDeque::new(),
            bytes: 0,
            blocked_since: None,
        }
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire) || self.tx.is_closed()
    }

    pub fn deadline(&self) -> Option<Instant> {
        self.blocked_since.map(|start| start + TIMEOUT)
    }

    pub fn blocked(&self) -> bool {
        !self.events.is_empty()
    }

    pub fn push(&mut self, event: GenerationEvent) -> Result<()> {
        let size = bytes(&event);
        ensure!(
            size <= MAX_BYTES - self.bytes,
            "generation output exceeded the outbox limit"
        );
        self.bytes += size;
        self.events.push_back(event);
        Ok(())
    }

    pub fn error(&mut self, message: String) {
        self.events.clear();
        self.bytes = 0;
        let _ = self.push(GenerationEvent::Error(message));
    }

    /// False indicates cancellation or an expired backpressure deadline.
    pub fn flush(&mut self, now: Instant) -> bool {
        if self.cancelled() || self.deadline().is_some_and(|deadline| now >= deadline) {
            return false;
        }
        while let Some(event) = self.events.pop_front() {
            let size = bytes(&event);
            match self.tx.try_send(event) {
                Ok(()) => {
                    self.bytes -= size;
                    self.blocked_since = None;
                }
                Err(mpsc::error::TrySendError::Full(event)) => {
                    self.events.push_front(event);
                    self.blocked_since.get_or_insert(now);
                    return true;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => return false,
            }
        }
        self.blocked_since = None;
        true
    }
}
