use crate::sampling::SamplingParams;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::mpsc;

#[derive(Clone, Copy, Default)]
pub(crate) struct ServingTokens {
    pub prefill: u64,
    pub cached: u64,
    pub decode: u64,
    pub spec_rounds: u64,
    pub proposed: u64,
    pub accepted: u64,
}

impl ServingTokens {
    pub fn since(self, previous: Self) -> Self {
        Self {
            prefill: self.prefill - previous.prefill,
            cached: self.cached - previous.cached,
            decode: self.decode - previous.decode,
            spec_rounds: self.spec_rounds - previous.spec_rounds,
            proposed: self.proposed - previous.proposed,
            accepted: self.accepted - previous.accepted,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct ServingSnapshot {
    pub tokens: ServingTokens,
    pub speculation: &'static str,
    pub active: u64,
    pub parallel: u64,
    pub waiting: u64,
    pub blocked: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum QuietWindow {
    Idle,
    Waiting,
    OutputBlocked,
    NoCompletedWork,
}

pub(crate) struct ServingWindow {
    pub tokens: ServingTokens,
    pub prefill: bool,
    pub decode: bool,
    pub busy: bool,
    pub quiet: Option<QuietWindow>,
}

impl ServingSnapshot {
    pub fn window(self, previous: Self, was_busy: bool) -> ServingWindow {
        let tokens = self.tokens.since(previous.tokens);
        let prefill = tokens.prefill > 0 || tokens.cached > 0;
        let decode = tokens.decode > 0;
        let completed = prefill || decode || tokens.spec_rounds > 0;
        let busy = completed || self.active > 0 || self.waiting > 0;
        let quiet = if completed {
            None
        } else if self.active > 0 {
            Some(if self.blocked == self.active {
                QuietWindow::OutputBlocked
            } else {
                QuietWindow::NoCompletedWork
            })
        } else if self.waiting > 0 {
            Some(QuietWindow::Waiting)
        } else {
            was_busy.then_some(QuietWindow::Idle)
        };
        ServingWindow {
            tokens,
            prefill,
            decode,
            busy,
            quiet,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
    Chat {
        messages: Vec<Value>,
        tools: Option<Value>,
        enable_thinking: bool,
    },
}

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub prompt: Prompt,
    /// None inherits the server's --cache-prompt setting.
    pub cache_prompt: Option<bool>,
    pub sampling: SamplingParams,
    pub max_tokens: usize,
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub text: String,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub timings: Value,
    pub stats: Value,
}

#[derive(Debug, Clone)]
pub enum GenerationEvent {
    Token(String),
    Finished(GenerationResult),
    Error(String),
}

pub struct GenerationHandle {
    pub events: mpsc::Receiver<GenerationEvent>,
    pub cancel: Arc<AtomicBool>,
}
impl Drop for GenerationHandle {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub struct EngineBusy;
impl std::fmt::Display for EngineBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("inference request queue is full")
    }
}
impl std::error::Error for EngineBusy {}

#[derive(Debug)]
pub struct EngineUnavailable;
impl std::fmt::Display for EngineUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("inference worker is unavailable")
    }
}
impl std::error::Error for EngineUnavailable {}
