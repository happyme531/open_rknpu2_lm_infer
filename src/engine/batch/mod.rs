//! Continuous batching on one model worker.
mod output;
mod request;
mod scheduler;
#[cfg(test)]
mod tests;

use super::{Engine, EngineInner, EngineStats, EngineUnavailable, Job, JobSender};
use crate::{
    model::{
        ModelOptions,
        batch::{self, BatchConfig},
    },
    progress::{GenerationProgress, Phase},
    text::TextCodec,
};
use anyhow::{Context, Result};
use std::{
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread,
};
use tokio::sync::{Notify, mpsc, watch};

pub(super) struct Sender {
    tx: mpsc::UnboundedSender<Queued>,
    waiting: Arc<AtomicUsize>,
    next_id: AtomicU64,
    pub wake: Arc<Notify>,
}

struct Queued {
    id: u64,
    job: Job,
    waiting: WaitingCount,
}

// Track a request across the ingress channel and scheduler queue.
struct WaitingCount(Arc<AtomicUsize>);

impl Drop for WaitingCount {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Sender {
    pub fn available(&self) -> bool {
        !self.tx.is_closed()
    }

    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Relaxed)
    }

    pub fn send(&self, job: Job) -> Result<()> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.waiting.fetch_add(1, Ordering::Relaxed);
        self.tx
            .send(Queued {
                id,
                job,
                waiting: WaitingCount(self.waiting.clone()),
            })
            .map_err(|_| EngineUnavailable)?;
        self.wake.notify_one();
        Ok(())
    }
}

impl Engine {
    pub fn start_batch(mut options: ModelOptions, config: BatchConfig) -> Result<Self> {
        config.validate(&options)?;
        options.tune_search = false;
        crate::model::ensure_enabled(&options.path)?;
        let sampling_defaults = crate::model::metadata::sampling_defaults(&options.path)?;
        let codec = Arc::new(TextCodec::load(&options.path)?);
        let worker_codec = codec.clone();
        let (tx, rx) = mpsc::unbounded_channel();
        let wake = Arc::new(Notify::new());
        let worker_wake = wake.clone();
        let stats = Arc::new(RwLock::new(EngineStats::default()));
        let worker_stats = stats.clone();
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker_shutdown = shutdown.clone();
        let (progress, _) = watch::channel(GenerationProgress::default());
        let worker_progress = progress.clone();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("rknpu-batch".into())
            .spawn(move || {
                let loaded = (|| -> Result<_> {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_time()
                        .build()?;
                    let model = batch::load(options, config)?;
                    Ok((runtime, model))
                })();
                let (runtime, model) = match loaded {
                    Ok(loaded) => loaded,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                let info = model.info().clone();
                let scheduler = scheduler::Scheduler::new(model, &worker_codec, worker_stats);
                let mut reporter = crate::progress::Reporter::new(worker_progress, "none");
                reporter.phase(Phase::Ready);
                if ready_tx.send(Ok(info)).is_ok() {
                    runtime.block_on(scheduler.run(rx, worker_wake, worker_shutdown.clone()));
                }
                if !worker_shutdown.swap(true, Ordering::AcqRel) {
                    reporter.phase(Phase::Error);
                }
            })?;
        let info = ready_rx
            .recv()
            .context("batch worker stopped during loading")??;
        Ok(Self {
            inner: Arc::new(EngineInner {
                sender: Some(JobSender::Batch(Sender {
                    tx,
                    waiting: Arc::default(),
                    next_id: AtomicU64::new(1),
                    wake,
                })),
                info,
                sampling_defaults,
                codec,
                stats,
                active: Default::default(),
                shutdown,
                worker: Some(worker),
                user_boundaries: false,
                progress,
            }),
        })
    }
}
