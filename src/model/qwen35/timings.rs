//! Synchronous stage totals; fused Linear finish remains part of its call.
use crate::model::experts::timeline::Timeline;
use serde::Serialize;
use std::time::Instant;

#[derive(Default, Serialize)]
pub(super) struct Timings {
    linear_ms: f64,
    gdn_ms: f64,
    attention_ms: f64,
    cpu_ms: f64,
    gdn_split_ms: f64,
    gdn_output_copy_ms: f64,
    attention_verify_calls: usize,
    attention_verify_rows: usize,
    #[serde(skip)]
    timeline: Timeline,
    #[serde(skip)]
    context: (usize, usize, i32, bool),
}

pub(super) enum Phase {
    Linear,
    Gdn,
    Attention,
    Cpu,
}

impl Timings {
    pub fn with_timeline(timeline: Timeline) -> Self {
        Self {
            timeline,
            ..Self::default()
        }
    }
    pub fn reset(&mut self) {
        *self = Self::with_timeline(self.timeline.clone());
    }
    pub fn context(&mut self, position: usize, rows: usize, layer: i32, verify: bool) {
        self.context = (position, rows, layer, verify);
    }
    pub fn stage<T>(
        &mut self,
        phase: Phase,
        name: &'static str,
        operation: impl FnOnce() -> T,
    ) -> T {
        let start = Instant::now();
        let result = self.measure(phase, operation);
        self.span(name, start);
        result
    }
    pub fn trace<T>(&self, name: &'static str, operation: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = operation();
        self.span(name, start);
        result
    }
    pub fn span(&self, name: &'static str, start: Instant) {
        let (position, rows, layer, verify) = self.context;
        self.timeline.span(name, 0, start, || {
            serde_json::json!({
                "position":position,"rows":rows,"layer":layer,"verify":verify
            })
        });
    }
    pub fn record_attention_verification(&mut self, rows: usize) {
        self.attention_verify_calls += 1;
        self.attention_verify_rows += rows;
    }
    pub fn record_gdn_io(&mut self, split_ms: f64, output_copy_ms: f64) {
        self.gdn_split_ms += split_ms;
        self.gdn_output_copy_ms += output_copy_ms;
    }
    pub fn measure<T>(&mut self, phase: Phase, operation: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = operation();
        let milliseconds = start.elapsed().as_secs_f64() * 1000.0;
        *match phase {
            Phase::Linear => &mut self.linear_ms,
            Phase::Gdn => &mut self.gdn_ms,
            Phase::Attention => &mut self.attention_ms,
            Phase::Cpu => &mut self.cpu_ms,
        } += milliseconds;
        result
    }
}
