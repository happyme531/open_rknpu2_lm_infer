use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod checkpoint;
#[allow(
    dead_code,
    reason = "Shared CPU adapters remain available in every model build."
)]
pub(crate) mod cpu;
#[cfg(any(
    feature = "model-qwen3",
    feature = "model-llama",
    feature = "model-granite"
))]
pub mod dense;
pub mod dflash;
#[allow(
    dead_code,
    reason = "Shared expert scheduling remains available in every model build."
)]
pub mod experts;
#[cfg(feature = "model-ling3tiny")]
pub mod ling3tiny;
pub(crate) mod metadata;
pub(crate) mod metrics;
pub(crate) mod output_head;
#[allow(
    dead_code,
    reason = "Prepared input has a stable shape across model builds."
)]
mod prefill;
#[allow(
    dead_code,
    reason = "Shared prefix caching is compiled independently of its model consumers."
)]
mod prefix_cache;
pub(crate) mod projection;
pub(crate) use prefill::PrefillStep;
pub use prefill::PreparedPrefill;
mod speculation;
pub use speculation::{SpeculativeConfig, SpeculativeType};
#[cfg(any(feature = "model-qwen35", feature = "model-minicpmv47"))]
pub mod qwen35;
mod registry;
pub use registry::{build_info, ensure_enabled};
#[allow(
    dead_code,
    reason = "Shared tracing and test helpers are independent of model selection."
)]
pub(crate) mod trace;
pub(crate) mod workspace;

pub struct LinearShape {
    pub name: &'static str,
    pub rows: usize,
    pub k: usize,
    pub n: usize,
}

pub fn vision_tuning_shapes(path: &std::path::Path, rows: &[usize]) -> Result<Vec<LinearShape>> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    let family = registry::require(&config)?;
    let shapes = family
        .vision_shapes
        .ok_or_else(|| anyhow::anyhow!("this model family has no supported vision encoder"))?;
    shapes(path, rows)
}

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
    pub threads: usize,
    pub npu_mask: u32,
    pub cpu_mask: u64,
    pub tune: bool,
    /// Permit device search only in the explicit tuning command.
    pub tune_search: bool,
    pub tune_cache: PathBuf,
    pub attention_tile: usize,
    pub vision_dtype: VisionDtype,
    pub experts: experts::Options,
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
    /// Optional sequential correction to a parallel draft row. Returns a
    /// learned acceptance probability when the drafter supplies one. Below
    /// cutoff, correction may be omitted and the caller must discard that row.
    fn adjust_draft_logits(
        &self,
        _row: usize,
        _previous: u32,
        _logits: &mut [f32],
        _cutoff: f64,
    ) -> Result<Option<f64>> {
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
    options.validate_speculation()?;
    let spec = options.speculation.clone();
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(options.path.join("config.json"))?)?;
    let family = registry::require(&config)?;
    anyhow::ensure!(
        options.experts.storage != experts::Storage::Stream
            || matches!(
                config["model_type"].as_str(),
                Some("qwen3_5_moe" | "qwen3_5_moe_text" | "minicpmv4_7" | "bailing_hybrid")
            ),
        "expert streaming requires a MoE checkpoint"
    );
    anyhow::ensure!(
        spec.window() == 0
            || matches!(
                config["model_type"].as_str(),
                Some(
                    "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "minicpmv4_7"
                )
            ),
        "speculative decoding requires a Qwen3.5 or MiniCPM-V 4.7 checkpoint"
    );
    crate::npu::configure_cpu(&mut options)?;
    let mut model = (family.load)(options)?;
    if spec.kind == SpeculativeType::DraftMtp && spec.window() > 0 {
        model.enable_mtp(spec.window())?;
    }
    Ok(model)
}
