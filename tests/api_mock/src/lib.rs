//! Hardware-free tests compile the production API, DTOs, tokenizer and parser.
#![allow(dead_code)]
#[path = "../../../src/api.rs"]
mod api;
#[path = "../../../src/engine/types.rs"]
mod engine_types;
#[path = "../../../src/sampling.rs"]
mod sampling;
#[allow(unused_imports)]
#[path = "../../../src/text.rs"]
mod text;
mod engine {
    pub use crate::engine_types::*;
    #[derive(Clone)]
    pub struct Engine;
    pub struct Info {
        pub id: String,
        pub context: usize,
    }
    impl Engine {
        pub fn info(&self) -> Info {
            Info {
                id: "mock".into(),
                context: 4096,
            }
        }
        pub fn shutdown(&self) {}
        pub fn prompt_opens_thinking(&self, _: &Prompt) -> anyhow::Result<bool> {
            Ok(false)
        }
        pub fn generate_with_tool_markers(
            &self,
            request: GenerationRequest,
            preserve: bool,
        ) -> anyhow::Result<GenerationHandle> {
            let input = match &request.prompt {
                Prompt::Chat { messages, .. } => {
                    messages.last().unwrap()["content"].as_str().unwrap()
                }
                Prompt::Text(text) => text.as_str(),
                _ => anyhow::bail!("mock expects text or chat"),
            };
            let (expected, output) = input.split_once('\n').unwrap();
            assert_eq!(
                preserve,
                expected == "preserve",
                "wrong request decoding policy"
            );
            let result = GenerationResult {
                text: output.into(),
                finish_reason: "stop".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                cached_tokens: 0,
                timings: serde_json::Value::Null,
                stats: serde_json::Value::Null,
            };
            let (sender, events) = tokio::sync::mpsc::channel(2);
            let output = output.to_owned();
            tokio::spawn(async move {
                for c in output.chars() {
                    if sender
                        .send(GenerationEvent::Token(c.to_string()))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                let _ = sender.send(GenerationEvent::Finished(result)).await;
            });
            Ok(GenerationHandle {
                events,
                cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            })
        }
    }
}
include!("local_support.rs");
#[cfg(test)]
mod http_tests;
