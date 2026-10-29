//! Latest-value inference status, independent of the generated text stream.
use crate::model::ModelOptions;
use std::{
    cell::Cell,
    collections::VecDeque,
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::watch;

mod terminal;
pub(crate) use terminal::TerminalProgress;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Phase {
    #[default]
    Ready,
    Preparing,
    Vision,
    Prefill,
    Decode,
    Finished,
    Cancelled,
    Error,
}

/// Model callbacks count committed tokens, including both lanes of a group.
#[derive(Debug, Clone, Copy)]
pub enum ModelProgress {
    Vision { completed: usize, total: usize },
    Prefill { completed: usize, total: usize },
}

#[derive(Debug, Clone, Default)]
pub struct DecodeProgress {
    pub generated: usize,
    pub consumed: usize,
    pub model_ms: f64,
    pub tokens_per_second: Option<f64>,
    pub average_tokens_per_second: Option<f64>,
    pub rounds: usize,
    pub proposed: usize,
    pub accepted: usize,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub confidence_stops: usize,
}

#[derive(Debug, Clone, Default)]
pub struct GenerationProgress {
    pub request: u64,
    pub phase: Phase,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completed: usize,
    pub total: usize,
    pub elapsed_ms: f64,
    pub speculation: &'static str,
    pub decode: DecodeProgress,
}

fn rate(tokens: usize, ms: f64) -> Option<f64> {
    (ms > 0.).then(|| tokens as f64 * 1000. / ms)
}
fn speed(value: Option<f64>) -> String {
    value.map_or_else(|| "--".into(), |v| format!("{v:.1}"))
}
pub(crate) fn bar(done: usize, total: usize) -> String {
    let fill = (done.min(total) * 20).checked_div(total).unwrap_or(0);
    format!("[{}{}]", "#".repeat(fill), ".".repeat(20 - fill))
}
pub(crate) fn eta(done: usize, total: usize, seconds: f64) -> String {
    if done == 0 {
        "--".into()
    } else {
        format!(
            "{:.1}s",
            seconds * total.saturating_sub(done) as f64 / done as f64
        )
    }
}

impl GenerationProgress {
    pub fn lines(&self) -> Vec<String> {
        let mut lines = vec![match self.phase {
            Phase::Ready => "Ready".into(),
            Phase::Preparing => format!(
                "Preparing · {} prompt · {} cached",
                self.prompt_tokens, self.cached_tokens
            ),
            Phase::Vision => format!(
                "Vision encoding · image {}/{} · {:.1}s",
                (self.completed + 1).min(self.total),
                self.total,
                self.elapsed_ms / 1000.
            ),
            Phase::Prefill => format!(
                "Prefill {} {}/{} ({:.0}%) · {} cached · {} tok/s · ETA {}",
                bar(self.completed, self.total),
                self.completed,
                self.total,
                if self.total > 0 {
                    self.completed as f64 * 100. / self.total as f64
                } else {
                    0.
                },
                self.cached_tokens,
                speed(rate(self.completed, self.elapsed_ms)),
                eta(self.completed, self.total, self.elapsed_ms / 1000.)
            ),
            phase => format!(
                "{} · {} output · {} tok/s · avg {} tok/s",
                match phase {
                    Phase::Decode => "Decode",
                    Phase::Finished => "Done",
                    Phase::Cancelled => "Cancelled",
                    _ => "Error",
                },
                self.decode.generated,
                speed(self.decode.tokens_per_second),
                speed(self.decode.average_tokens_per_second)
            ),
        }];
        if self.speculation != "none"
            && !self.speculation.is_empty()
            && matches!(
                self.phase,
                Phase::Decode | Phase::Finished | Phase::Cancelled | Phase::Error
            )
        {
            let d = &self.decode;
            let acceptance = (d.proposed > 0).then(|| d.accepted as f64 * 100. / d.proposed as f64);
            let length = (d.rounds > 0).then(|| d.consumed as f64 / d.rounds as f64);
            lines.push(format!(
                "{} · accepted {}/{} ({}%) · {} tok/round · {} rounds",
                if self.speculation == "draft-mtp" {
                    "MTP"
                } else {
                    "DFlash"
                },
                d.accepted,
                d.proposed,
                speed(acceptance),
                speed(length),
                d.rounds
            ));
            lines.push(format!(
                "draft/verify {}/{} ms/round · confidence stops {}",
                speed((d.rounds > 0).then(|| d.draft_ms / d.rounds as f64)),
                speed((d.rounds > 0).then(|| d.verify_ms / d.rounds as f64)),
                d.confidence_stops
            ));
        }
        lines
    }
}

pub(crate) struct Reporter {
    sender: watch::Sender<GenerationProgress>,
    pub state: GenerationProgress,
    phase_start: Instant,
    published: Instant,
    recent: VecDeque<(Instant, usize, f64)>,
}
impl Default for Reporter {
    fn default() -> Self {
        Self::new(watch::channel(GenerationProgress::default()).0, "none")
    }
}
impl Reporter {
    pub fn new(sender: watch::Sender<GenerationProgress>, speculation: &'static str) -> Self {
        Self {
            state: GenerationProgress {
                speculation,
                ..Default::default()
            },
            sender,
            phase_start: Instant::now(),
            published: Instant::now(),
            recent: VecDeque::new(),
        }
    }
    pub fn phase(&mut self, phase: Phase) {
        self.state.phase = phase;
        self.state.elapsed_ms = 0.;
        self.phase_start = Instant::now();
        self.publish(true);
    }
    pub fn begin(&mut self, prompt_tokens: usize) {
        self.state = GenerationProgress {
            request: self.state.request + 1,
            prompt_tokens,
            speculation: self.state.speculation,
            ..Default::default()
        };
        self.recent.clear();
        self.phase(Phase::Preparing);
    }
    pub fn model(&mut self, update: ModelProgress) {
        let (phase, completed, total) = match update {
            ModelProgress::Vision { completed, total } => (Phase::Vision, completed, total),
            ModelProgress::Prefill { completed, total } => (Phase::Prefill, completed, total),
        };
        let changed = self.state.phase != phase;
        if changed {
            self.phase_start = Instant::now();
        }
        self.state.phase = phase;
        self.state.completed = completed;
        self.state.total = total;
        self.state.elapsed_ms = self.phase_start.elapsed().as_secs_f64() * 1000.;
        self.publish(changed || completed == total);
    }
    pub fn decoding(&mut self) {
        self.recent.push_back((Instant::now(), 0, 0.));
        self.phase(Phase::Decode);
    }
    pub fn decode(&mut self, mut value: DecodeProgress) {
        let now = Instant::now();
        while self.recent.len() > 1
            && now.duration_since(self.recent[1].0) >= Duration::from_secs(1)
        {
            self.recent.pop_front();
        }
        if let Some(&(_, tokens, ms)) = self.recent.front() {
            value.tokens_per_second = rate(value.consumed - tokens, value.model_ms - ms);
        }
        value.average_tokens_per_second = rate(value.consumed, value.model_ms);
        self.recent.push_back((now, value.consumed, value.model_ms));
        self.state.decode = value;
        self.state.elapsed_ms = self.phase_start.elapsed().as_secs_f64() * 1000.;
        self.publish(false);
    }
    fn publish(&mut self, force: bool) {
        if force || self.published.elapsed() >= Duration::from_millis(250) {
            self.sender.send_replace(self.state.clone());
            self.published = Instant::now();
        }
    }
}

thread_local! { static TAIL: Cell<bool> = const { Cell::new(false) }; }

/// Partial prefill groups execute serially; complete pipeline groups keep the
/// normal diagnostic context on both worker threads.
pub(crate) fn prefill_shape<R>(tail: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            TAIL.set(self.0);
        }
    }
    let _restore = Restore(TAIL.replace(tail || TAIL.get()));
    f()
}

#[derive(Default)]
struct UntunedNotice {
    message: OnceLock<String>,
    shown: AtomicBool,
}
impl UntunedNotice {
    fn record(&self, enabled: bool, searching: bool, tail: bool, message: impl FnOnce() -> String) {
        if enabled && !searching && !tail {
            self.message.get_or_init(message);
        }
    }
    fn take(&self) -> Option<String> {
        let message = self.message.get()?;
        (!self.shown.swap(true, Ordering::Relaxed)).then(|| message.clone())
    }
}
static UNTUNED: UntunedNotice = UntunedNotice {
    message: OnceLock::new(),
    shown: AtomicBool::new(false),
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TuningSource {
    Target,
    Vision,
    Mtp,
    Dflash,
}

fn untuned_message(
    options: &ModelOptions,
    kind: &str,
    m: usize,
    k: usize,
    n: usize,
    source: TuningSource,
) -> String {
    let origin = match source {
        TuningSource::Target => "",
        TuningSource::Vision => "vision ",
        TuningSource::Mtp => "MTP ",
        TuningSource::Dflash => "DFlash ",
    };
    let detail =
        format!("Untuned shape: {origin}{kind} M={m} K={k} N={n}; using the default strategy.");
    if source == TuningSource::Mtp {
        return format!(
            "{detail} MTP floating projections are not supported by the current tune command."
        );
    }
    let common = format!(
        "tune --model {:?} --tune-cache {:?} --threads {} --cpu-mask 0x{:x} --npu-mask {}",
        options.path, options.tune_cache, options.threads, options.cpu_mask, options.npu_mask
    );
    let command = if source == TuningSource::Vision {
        format!(
            "{common} --vision --vision-dtype {} --image PATH",
            options.vision_dtype.as_str()
        )
    } else {
        let mut command = format!(
            "{common} --ubatch-size {} --batches {}",
            options.chunk,
            m.min(options.chunk)
        );
        if options.speculation.window() > 0
            && options.speculation.kind == crate::model::SpeculativeType::DraftDflash
        {
            command.push_str(&format!(
                " --spec-type {} --spec-draft-n-max {}",
                options.speculation.kind.as_str(),
                options.speculation.max_draft
            ));
            if let Some(path) = &options.speculation.model {
                command.push_str(&format!(" --spec-draft-model {path:?}"));
            }
        }
        command
    };
    format!("{detail} Run: open_rknpu2_lm_infer {command}")
}

pub(crate) fn untuned(
    options: &ModelOptions,
    kind: &str,
    m: usize,
    k: usize,
    n: usize,
    source: TuningSource,
) {
    if notice_recorded() {
        return;
    }
    UNTUNED.record(options.tune, options.tune_search, TAIL.get(), || {
        untuned_message(options, kind, m, k, n, source)
    });
}
pub(crate) fn take_notice() -> Option<String> {
    UNTUNED.take()
}
pub(crate) fn notice_recorded() -> bool {
    UNTUNED.message.get().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options() -> ModelOptions {
        use clap::Parser;
        crate::cli::Cli::try_parse_from([
            "infer",
            "-m",
            "/tmp/progress-model",
            "--ctx-size",
            "1024",
            "--tune-cache",
            "/tmp/progress-cache",
            "run",
        ])
        .unwrap()
        .model
        .options()
        .unwrap()
    }
    #[test]
    fn mtp_hint_distinguishes_auxiliary_projections_from_the_target() {
        let mut options = options();
        // The source is explicit even when MTP was enabled through the model API.
        let hint = untuned_message(&options, "FP16", 1, 2560, 5120, TuningSource::Mtp);
        assert!(hint.contains("MTP FP16"));
        assert!(hint.contains("not supported by the current tune command"));
        assert!(!hint.contains("Run:"));
        options.speculation.kind = crate::model::SpeculativeType::DraftMtp;
        let target = untuned_message(&options, "W4", 2, 2560, 5120, TuningSource::Target);
        assert!(target.contains("Run: open_rknpu2_lm_infer tune"));
        assert!(target.contains("--tune-cache \"/tmp/progress-cache\""));
        assert!(target.contains("--batches 2"));
        assert!(!target.contains("--spec-type draft-mtp"));
    }
    #[test]
    fn dflash_and_vision_hints_keep_their_supported_tuning_commands() {
        let mut options = options();
        options.speculation.kind = crate::model::SpeculativeType::DraftDflash;
        options.speculation.model = Some("/tmp/draft-model".into());
        let draft = untuned_message(&options, "FP16", 4, 2560, 5120, TuningSource::Dflash);
        assert!(draft.contains("DFlash FP16"));
        assert!(draft.contains("--spec-type draft-dflash"));
        assert!(draft.contains("--spec-draft-model \"/tmp/draft-model\""));
        let vision = untuned_message(&options, "FP16", 256, 1536, 1024, TuningSource::Vision);
        assert!(vision.contains("--vision --vision-dtype fp16 --image PATH"));
        assert!(!vision.contains("--spec-type"));
    }
    #[test]
    #[ignore = "requires exclusive RK3588 NPU access and a fresh test process"]
    fn mtp_floating_cache_miss_has_no_target_tuning_suggestion() -> anyhow::Result<()> {
        let _lock = crate::validation::BoardLock::acquire()?;
        let cache = tempfile::tempdir()?;
        let mut options = options();
        options.tune_cache = cache.path().into();
        crate::npu::configure_cpu(&mut options)?;
        let device = crate::npu::Backend::new(&options, 0)?;
        let floating = crate::npu::FloatBackend::for_mtp(device);
        floating.prepare(128, 128, 1)?;
        let notice = take_notice().expect("MTP floating cache miss");
        assert!(notice.contains("MTP FP16/BF16 M=1 K=128 N=128"));
        assert!(notice.contains("not supported by the current tune command"));
        assert!(!notice.contains("Run:"));
        eprintln!("{notice}");
        floating.prepare(128, 128, 1)?;
        assert!(take_notice().is_none());
        Ok(())
    }
    #[test]
    fn tail_and_disabled_tuning_do_not_consume_the_notice() {
        let notice = UntunedNotice::default();
        notice.record(true, false, true, || "tail".into());
        notice.record(false, false, false, || "disabled".into());
        notice.record(true, true, false, || "search".into());
        assert!(notice.take().is_none());
        notice.record(true, false, false, || "decode".into());
        notice.record(true, false, false, || "vision".into());
        assert_eq!(notice.take().as_deref(), Some("decode"));
        assert!(notice.take().is_none());
    }
    #[test]
    fn progress_does_not_need_a_consumer_and_final_rates_match_complete_calls() {
        let (tx, rx) = watch::channel(GenerationProgress::default());
        let mut reporter = Reporter::new(tx, "draft-mtp");
        reporter.begin(100);
        reporter.state.cached_tokens = 40;
        reporter.model(ModelProgress::Prefill {
            completed: 0,
            total: 60,
        });
        reporter.model(ModelProgress::Prefill {
            completed: 60,
            total: 60,
        });
        assert_eq!((rx.borrow().completed, rx.borrow().total), (60, 60));
        reporter.decoding();
        reporter.decode(DecodeProgress {
            generated: 1,
            ..Default::default()
        });
        assert!(reporter.state.decode.average_tokens_per_second.is_none());
        reporter.decode(DecodeProgress {
            generated: 5,
            consumed: 4,
            model_ms: 400.,
            rounds: 2,
            proposed: 6,
            accepted: 3,
            ..Default::default()
        });
        reporter.phase(Phase::Finished);
        assert_eq!(rx.borrow().decode.average_tokens_per_second, Some(10.));
        assert!(rx.borrow().lines()[1].contains("50.0%"));
        reporter.begin(20);
        assert_eq!(
            (
                reporter.state.decode.generated,
                reporter.state.cached_tokens
            ),
            (0, 0)
        );
    }
    #[test]
    fn tail_context_restores_after_nested_calls_and_errors() {
        assert!(!TAIL.get());
        prefill_shape(true, || {
            assert!(TAIL.get());
            prefill_shape(false, || assert!(TAIL.get()));
        });
        assert!(!TAIL.get());
    }
}
