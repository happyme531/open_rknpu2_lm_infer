use crate::sampling::SamplingParams;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::mpsc;

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
