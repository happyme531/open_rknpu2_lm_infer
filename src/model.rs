use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod checkpoint;
pub(crate) mod cpu;
pub mod dense;
pub mod dflash;
pub(crate) mod metadata;
pub(crate) mod metrics;
pub(crate) mod output_head;
pub(crate) mod pipeline;
mod prefill;
pub(crate) mod projection;
pub(crate) use prefill::PrefillStep;
pub use prefill::PreparedPrefill;
mod speculation;
pub use speculation::{SpeculativeConfig, SpeculativeType};
pub mod qwen35;
pub(crate) mod trace;

/// ViT activation storage and CPU arithmetic. NPU GEMMs remain FP16 in both modes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum VisionDtype {
    #[default]
    Fp16,
    Fp32,
}
impl VisionDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelOptions {
    pub path: PathBuf,
    pub context: usize,
    pub chunk: usize,
    /// Concurrent prefill batches. One keeps serial execution.
    pub prefill_ubatches: usize,
    pub threads: usize,
    pub npu_mask: u32,
    pub cpu_mask: u64,
    pub tune: bool,
    /// Permit device search only in the explicit tuning command.
    pub tune_search: bool,
    /// Lossless W8 storage; native tiles stay raw unless compression shrinks them.
    pub compress_w8a8: bool,
    pub tune_cache: PathBuf,
    pub attention_tile: usize,
    pub vision_dtype: VisionDtype,
    pub checkpoints: CheckpointOptions,
    pub speculation: SpeculativeConfig,
}

#[derive(Debug, Clone, Copy)]
pub struct CheckpointOptions {
    pub capacity: usize,
    pub min_step: usize,
}
impl Default for CheckpointOptions {
    fn default() -> Self {
        Self {
            capacity: 32,
            min_step: 8192,
        }
    }
}
impl ModelOptions {
    pub(crate) fn validate_speculation(&self) -> Result<()> {
        self.speculation.validate(self.chunk)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub architecture: String,
    pub context: usize,
    pub vocab_size: usize,
    pub eos_tokens: Vec<u32>,
}

/// An instance owns one reusable KV cache and is used by a single worker.
/// New architectures register through `load` and implement this contract.
pub trait LanguageModel {
    fn info(&self) -> &ModelInfo;
    fn uses_user_boundaries(&self) -> bool {
        false
    }
    fn reset(&mut self) -> Result<()>;
    /// Reserve KV for a complete request before tuning and timed forwards.
    fn reserve(&mut self, _tokens: usize) -> Result<()> {
        Ok(())
    }
    fn prepare(&mut self, prompt_tokens: usize) -> Result<()>;
    fn prepare_prepared<'a>(
        &mut self,
        input: &'a crate::input::PreparedInput,
        start: usize,
    ) -> Result<PreparedPrefill<'a>> {
        self.prepare(input.tokens.len() - start)?;
        Ok(PreparedPrefill {
            input,
            start,
            steps: Vec::new(),
            checkpoint: None,
        })
    }
    /// Consume prompt/history tokens without evaluating final output norm or
    /// lm_head. Generation uses `forward` when next-token logits are required.
    fn prefill(&mut self, _tokens: &[u32]) -> Result<()> {
        anyhow::bail!("this model has no head-free prefill implementation")
    }
    fn forward(&mut self, tokens: &[u32]) -> Result<&[f32]>;
    /// Consume a suffix of a prepared prompt. The worker owns all model/device
    /// state; image decoding and token expansion have already completed.
    fn forward_prepared(
        &mut self,
        prepared: PreparedPrefill<'_>,
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<&[f32]> {
        let PreparedPrefill { input, start, .. } = prepared;
        anyhow::ensure!(
            input.images.is_empty(),
            "this model does not support images"
        );
        anyhow::ensure!(start < input.tokens.len(), "empty prepared suffix");
        self.forward_cancellable(&input.tokens[start..], cancelled)
    }
    /// Observe completed prefill groups without changing their scheduling.
    fn forward_prepared_with_progress(
        &mut self,
        prepared: PreparedPrefill<'_>,
        cancelled: &std::sync::atomic::AtomicBool,
        progress: &mut dyn FnMut(crate::progress::ModelProgress),
    ) -> Result<&[f32]> {
        let total = prepared.input.tokens.len() - prepared.start;
        progress(crate::progress::ModelProgress::Prefill {
            completed: 0,
            total,
        });
        let result = self.forward_prepared(prepared, cancelled)?;
        progress(crate::progress::ModelProgress::Prefill {
            completed: total,
            total,
        });
        Ok(result)
    }
    fn forward_cancellable(
        &mut self,
        tokens: &[u32],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<&[f32]> {
        anyhow::ensure!(
            !cancelled.load(std::sync::atomic::Ordering::Relaxed),
            "generation cancelled"
        );
        self.forward(tokens)
    }
    fn position(&self) -> usize;
    /// Optional NextN capability. Enabling before prefill builds the complete
    /// shifted draft prompt history alongside target execution.
    /// Configure proposals before prefill; Qwen3.5 accepts zero to disable MTP.
    fn enable_mtp(&mut self, _max_draft: usize) -> Result<()> {
        anyhow::bail!("this model does not support MTP")
    }
    fn speculation_window(&self) -> usize {
        0
    }
    /// Minimum top-candidate probability required to admit the current draft.
    /// Zero disables confidence truncation.
    fn speculation_p_min(&self) -> f64 {
        0.
    }
    fn speculation_sampling(&self) -> crate::sampling::DraftSampling {
        crate::sampling::DraftSampling::Greedy
    }
    fn draft_logits(
        &mut self,
        _token: u32,
        _cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<&[f32]> {
        anyhow::bail!("this model does not support MTP")
    }
    /// Parallel drafter logits in proposal order. Sequential drafters return
    /// None and are driven through draft_logits with each sampled predecessor.
    fn draft_block_logits(
        &mut self,
        _anchor: u32,
        _drafts: usize,
        _cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }
    fn verify_tokens(
        &mut self,
        _tokens: &[u32],
        _cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<&[f32]> {
        anyhow::bail!("this model does not support target verification")
    }
    fn finish_verification(&mut self, _consumed: usize) -> Result<()> {
        anyhow::bail!("this model does not support target verification")
    }
    fn abort_verification(&mut self) -> Result<()> {
        Ok(())
    }
    /// Score the next-token targets for every input row, retaining the KV prefix.
    fn token_nll(&mut self, tokens: &[u32], targets: &[u32]) -> Result<Vec<f64>> {
        anyhow::ensure!(
            tokens.len() == targets.len(),
            "scoring input/target length mismatch"
        );
        tokens
            .iter()
            .zip(targets)
            .map(|(&token, &target)| {
                crate::ppl::negative_log_likelihood(self.forward(&[token])?, target)
            })
            .collect()
    }
    /// Keep an already computed prefix. Return false when the backend cannot
    /// truncate its cache and the caller should reset and recompute instead.
    fn truncate(&mut self, _position: usize) -> Result<bool> {
        Ok(false)
    }
    /// Restore a usable prefix no longer than `maximum`. Zero asks the caller
    /// to reset. Recurrent models may restore an earlier checkpoint.
    fn restore_prefix(&mut self, maximum: usize) -> Result<usize> {
        if maximum == self.position() || self.truncate(maximum)? {
            Ok(maximum)
        } else {
            Ok(0)
        }
    }
    fn stats(&self) -> serde_json::Value;
}

pub fn load(mut options: ModelOptions) -> Result<Box<dyn LanguageModel>> {
    anyhow::ensure!(
        (1..=2).contains(&options.prefill_ubatches),
        "prefill-ubatches must be 1 or 2"
    );
    options.validate_speculation()?;
    let spec = options.speculation.clone();
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(options.path.join("config.json"))?)?;
    anyhow::ensure!(
        spec.window() == 0
            || matches!(
                config["model_type"].as_str(),
                Some("qwen3_5" | "qwen3_5_text")
            ),
        "speculative decoding is supported only for Qwen3.5 checkpoints"
    );
    crate::npu::configure_cpu(&mut options)?;
    let mut model: Box<dyn LanguageModel> = match config["model_type"].as_str() {
        Some("qwen3" | "llama" | "granite") => Box::new(dense::Dense::load(options)?),
        Some("qwen3_5" | "qwen3_5_text") => Box::new(qwen35::Qwen35::load(options)?),
        other => anyhow::bail!(
            "unsupported model architecture: {other:?}; supported: qwen3, llama, granite, qwen3_5, qwen3_5_text"
        ),
    };
    if spec.kind == SpeculativeType::DraftMtp && spec.window() > 0 {
        model.enable_mtp(spec.window())?;
    }
    Ok(model)
}
