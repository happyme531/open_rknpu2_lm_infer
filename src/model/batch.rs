//! Fixed sequence slots and the model-facing step contract.
use super::{ModelInfo, ModelOptions, SpeculativeType};
use crate::npu::{self, Attention, Backend};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeMap, ops::Range, time::Instant};

#[derive(Clone, Copy, Debug)]
pub struct BatchConfig {
    pub slots: usize,
    pub max_rows: usize,
}

impl BatchConfig {
    pub fn validate(self, options: &ModelOptions) -> Result<()> {
        ensure!(
            (2..=self.max_rows).contains(&self.slots) && self.max_rows <= 256,
            "batch mode requires 2 <= parallel <= ubatch-size <= 256"
        );
        ensure!(
            options.prefill_ubatches == 1,
            "batch mode requires prefill-ubatches=1"
        );
        ensure!(
            options.speculation.kind == SpeculativeType::None,
            "batch mode requires spec-type=none"
        );
        Ok(())
    }

    pub fn decoder_buckets(self) -> Vec<usize> {
        // Any prompt tail can share a step with decode, so keep each reachable bucket.
        buckets(self.max_rows)
    }

    pub fn head_buckets(self) -> Vec<usize> {
        buckets(self.slots)
    }
}

fn buckets(max_rows: usize) -> Vec<usize> {
    let mut shapes: Vec<_> = (1..=max_rows)
        .map(|rows| npu::bucket(rows).unwrap())
        .collect();
    shapes.dedup();
    shapes
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SeqId {
    pub slot: u16,
    pub generation: u32,
}

pub struct EmbeddedRow<'a> {
    pub row: usize,
    pub values: &'a [f32],
}

pub struct Span<'a> {
    pub seq: SeqId,
    pub start: usize,
    pub tokens: &'a [u32],
    pub positions: &'a [[usize; 3]],
    pub embeddings: &'a [EmbeddedRow<'a>],
    pub logits: bool,
}

#[derive(Debug, Eq, PartialEq)]
pub enum Admission {
    Ready(SeqId),
    Wait,
    CannotFit,
}

#[derive(Clone, Copy, Debug)]
pub struct BatchCapabilities {
    pub max_slots: usize,
    pub max_rows: usize,
    pub max_span_rows: usize,
    pub vision: bool,
}

#[derive(Debug)]
pub enum StepError {
    Rejected(anyhow::Error),
    Device(anyhow::Error),
}

pub struct StepOutput<'a> {
    pub rows: usize,
    pub bucket: usize,
    sequences: Vec<SeqId>,
    logits: &'a [f32],
    vocab_size: usize,
}

impl<'a> StepOutput<'a> {
    pub fn new(
        rows: usize,
        sequences: Vec<SeqId>,
        logits: &'a [f32],
        vocab_size: usize,
    ) -> Result<Self> {
        let bucket = npu::bucket(rows).context("invalid model step size")?;
        ensure!(
            vocab_size > 0 && sequences.len().checked_mul(vocab_size) == Some(logits.len()),
            "batched logits do not match the selected sequences"
        );
        ensure!(
            logits.iter().all(|value| value.is_finite()),
            "nonfinite batched logits"
        );
        Ok(Self {
            rows,
            bucket,
            sequences,
            logits,
            vocab_size,
        })
    }

    pub fn logits_rows(&self) -> impl Iterator<Item = (SeqId, &[f32])> + '_ {
        self.sequences
            .iter()
            .copied()
            .zip(self.logits.chunks_exact(self.vocab_size))
    }
}

pub trait BatchModel {
    fn info(&self) -> &ModelInfo;
    fn capabilities(&self) -> BatchCapabilities;
    fn try_admit(&mut self, total_tokens: usize) -> Admission;
    fn release(&mut self, seq: SeqId) -> Result<()>;
    /// Execute one step and return one logits row per selected span, in input order.
    fn execute_step(
        &mut self,
        spans: &[Span<'_>],
    ) -> std::result::Result<StepOutput<'_>, StepError>;
    fn stats(&self) -> Value;
}

pub(super) type StepResult<T> = std::result::Result<T, StepError>;

pub(super) struct PlacedSpan<'a, 'input> {
    pub span: &'a Span<'input>,
    pub rows: Range<usize>,
    slot: usize,
}

pub(super) struct StepLayout<'a, 'input> {
    pub spans: Vec<PlacedSpan<'a, 'input>>,
    pub rows: usize,
}

impl<'a, 'input> StepLayout<'a, 'input> {
    pub fn head_rows(&self) -> impl Iterator<Item = &PlacedSpan<'a, 'input>> {
        self.spans.iter().filter(|placed| placed.span.logits)
    }
}

pub(super) struct SeqPool<S> {
    default_seq: Option<S>,
    slots: Vec<S>,
    ledger: Vec<Slot>,
    context: usize,
    config: Option<BatchConfig>,
    poisoned: bool,
    resets: u64,
    reset_ms: f64,
    max_reset_ms: f64,
}

#[derive(Default)]
struct Slot {
    generation: u32,
    limit: Option<usize>,
}

impl<S> SeqPool<S> {
    pub fn new(mut states: Vec<S>, context: usize, config: Option<BatchConfig>) -> Self {
        let default_seq = if config.is_none() { states.pop() } else { None };
        let ledger = (0..states.len()).map(|_| Slot::default()).collect();
        Self {
            default_seq,
            slots: states,
            ledger,
            context,
            config,
            poisoned: false,
            resets: 0,
            reset_ms: 0.0,
            max_reset_ms: 0.0,
        }
    }

    pub fn single(&self) -> &S {
        self.default_seq.as_ref().expect("single-request model")
    }

    pub fn single_mut(&mut self) -> &mut S {
        self.default_seq.as_mut().expect("single-request model")
    }

    pub fn states(&self) -> impl Iterator<Item = &S> {
        self.default_seq.iter().chain(&self.slots)
    }

    pub fn capabilities(&self, max_span_rows: usize) -> BatchCapabilities {
        let config = self.config.expect("batch model");
        BatchCapabilities {
            max_slots: config.slots,
            max_rows: config.max_rows,
            max_span_rows,
            vision: false,
        }
    }

    pub fn try_admit(&mut self, total_tokens: usize) -> Admission {
        if self.poisoned || total_tokens == 0 || total_tokens > self.context {
            return Admission::CannotFit;
        }
        let Some((slot, entry)) = self
            .ledger
            .iter_mut()
            .enumerate()
            .find(|(_, e)| e.limit.is_none())
        else {
            return Admission::Wait;
        };
        entry.generation = entry.generation.wrapping_add(1);
        entry.limit = Some(total_tokens);
        Admission::Ready(SeqId {
            slot: slot as u16,
            generation: entry.generation,
        })
    }

    fn slot(&self, id: SeqId) -> Result<(usize, usize)> {
        let slot = usize::from(id.slot);
        let entry = self.ledger.get(slot).context("invalid sequence slot")?;
        ensure!(
            entry.generation == id.generation,
            "stale sequence generation"
        );
        Ok((slot, entry.limit.context("sequence slot is free")?))
    }

    pub fn release(&mut self, seq: SeqId, reset: impl FnOnce(&mut S) -> Result<()>) -> Result<()> {
        // Only healthy slots are recycled. A partially executed step stops the worker.
        ensure!(!self.poisoned, "batch model is poisoned");
        let (slot, _) = self.slot(seq)?;
        let start = Instant::now();
        let result = reset(&mut self.slots[slot]);
        self.poison(result)?;
        self.ledger[slot].limit = None;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        self.resets += 1;
        self.reset_ms += elapsed;
        self.max_reset_ms = self.max_reset_ms.max(elapsed);
        Ok(())
    }

    pub fn layout<'a, 'input>(
        &self,
        spans: &'a [Span<'input>],
        chunk: usize,
        vocab: usize,
        position: impl Fn(&S) -> usize,
        rope_position: impl Fn(&S) -> usize,
    ) -> StepResult<StepLayout<'a, 'input>> {
        if self.poisoned {
            return Err(StepError::Device(anyhow::anyhow!(
                "batch model is poisoned"
            )));
        }
        let validate = || -> Result<StepLayout<'a, 'input>> {
            let config = self.config.expect("batch model");
            ensure!(
                !spans.is_empty() && spans.len() <= config.slots,
                "invalid span count"
            );
            let mut total = 0;
            let mut used = [false; 256];
            let mut placed = Vec::with_capacity(spans.len());
            for span in spans {
                let (slot, limit) = self.slot(span.seq)?;
                ensure!(
                    !std::mem::replace(&mut used[slot], true),
                    "duplicate sequence in step"
                );
                let seq = &self.slots[slot];
                let len = span.tokens.len();
                ensure!(
                    span.start == position(seq),
                    "span starts at the wrong sequence position"
                );
                ensure!(
                    (1..=chunk).contains(&len) && len <= limit - span.start,
                    "span exceeds sequence or ubatch capacity"
                );
                ensure!(
                    span.tokens.iter().all(|&t| (t as usize) < vocab),
                    "span token outside vocabulary"
                );
                ensure!(span.embeddings.is_empty(), "batch mode accepts text spans");
                let rope = rope_position(seq);
                ensure!(
                    span.positions.len() == len
                        && span
                            .positions
                            .iter()
                            .enumerate()
                            .all(|(i, p)| *p == [rope + i; 3]),
                    "span positions do not match the text sequence"
                );
                let end = total + len;
                placed.push(PlacedSpan {
                    span,
                    rows: total..end,
                    slot,
                });
                total = end;
            }
            ensure!(
                total <= config.max_rows,
                "step exceeds the ubatch-size token limit"
            );
            Ok(StepLayout {
                spans: placed,
                rows: total,
            })
        };
        validate().map_err(StepError::Rejected)
    }

    pub fn state(&self, placed: &PlacedSpan<'_, '_>) -> &S {
        &self.slots[placed.slot]
    }

    pub fn complete_step<'a>(
        &mut self,
        layout: StepLayout<'_, '_>,
        execution: Result<()>,
        logits: &'a [f32],
        vocab_size: usize,
        advance: impl Fn(&mut S, &Span<'_>),
    ) -> StepResult<StepOutput<'a>> {
        let result = execution.and_then(|()| {
            let sequences = layout.head_rows().map(|placed| placed.span.seq).collect();
            StepOutput::new(layout.rows, sequences, logits, vocab_size)
        });
        let output = self.poison(result).map_err(StepError::Device)?;
        for placed in layout.spans {
            advance(&mut self.slots[placed.slot], placed.span);
        }
        Ok(output)
    }

    fn poison<T>(&mut self, result: Result<T>) -> Result<T> {
        self.poisoned |= result.is_err();
        result
    }

    pub fn reset_stats(&self) -> Value {
        json!({"slot_resets": self.resets, "slot_reset_ms": self.reset_ms,
            "slot_reset_max_ms": self.max_reset_ms})
    }
}

pub(super) fn prepare(
    config: BatchConfig,
    backends: &[Backend],
    head: Option<&Backend>,
) -> Result<()> {
    for backend in backends {
        for rows in config.decoder_buckets() {
            backend.prepare(rows).with_context(|| {
                format!(
                    "prepare batch M={rows} in IOMMU domain {}",
                    backend.domain_id()
                )
            })?;
        }
        backend.prepare_batch_padding(config.max_rows)?;
    }
    if let Some(backend) = head {
        for rows in config.head_buckets() {
            backend.prepare(rows).with_context(|| {
                format!(
                    "prepare batch head M={rows} in IOMMU domain {}",
                    backend.domain_id()
                )
            })?;
        }
        backend.prepare_batch_padding(config.slots)?;
    }
    Ok(())
}

pub(super) fn kv_tail_budget(options: &ModelOptions, kh: usize, dimension: usize) -> usize {
    let rows = (options.context.div_ceil(32) * 32).min(options.attention_tile - 32);
    rows * kh * (dimension + 32) * 2
}

pub(super) fn reserve_cache(
    backend: &Backend,
    slot: usize,
    layer: usize,
    qh: usize,
    kh: usize,
    dimension: usize,
    scale: f32,
) -> Result<Attention> {
    backend
        .reserved_attention(qh, kh, dimension, scale)
        .with_context(|| {
            format!(
                "reserve KV for slot {slot}, layer {layer}, IOMMU domain {}",
                backend.domain_id()
            )
        })
}

pub(super) fn memory_stats(
    mut stats: Value,
    slots: impl IntoIterator<Item = (Vec<(u32, u64, u64)>, usize)>,
    reset_stats: Value,
) -> Value {
    let mut domain_kv = BTreeMap::<u32, u64>::new();
    let mut sequences = Vec::new();
    for (caches, state_bytes) in slots {
        let mut kv_bytes = 0;
        let mut kv_cpu_bytes = 0;
        for (domain, bytes, cpu_bytes) in caches {
            kv_bytes += bytes;
            kv_cpu_bytes += cpu_bytes;
            *domain_kv.entry(domain).or_default() += bytes;
        }
        sequences.push(json!({"kv_payload_bytes": kv_bytes,
            "kv_cpu_metadata_bytes_estimate": kv_cpu_bytes, "recurrent_state_bytes": state_bytes}));
    }
    if let Some(domains) = stats["domains"].as_array_mut() {
        for domain in domains {
            let id = domain["iommu_domain_id"].as_u64().unwrap() as u32;
            let kv = domain_kv.get(&id).copied().unwrap_or_default();
            let known = kv
                + domain["weight_payload_bytes"].as_u64().unwrap_or_default()
                + domain["linear_device_workspace_bytes"]
                    .as_u64()
                    .unwrap_or_default()
                + domain["attention_device_workspace_bytes_estimate"]
                    .as_u64()
                    .unwrap_or_default();
            let limit = 2u64 << 30;
            domain["kv_payload_bytes"] = json!(kv);
            domain["known_device_payload_bytes"] = json!(known);
            domain["iova_budget_bytes"] = json!(limit);
            domain["iova_headroom_estimate_bytes"] = json!(limit.saturating_sub(known));
        }
    }
    for key in [
        "kv_payload_bytes",
        "kv_cpu_metadata_bytes_estimate",
        "recurrent_state_bytes",
    ] {
        stats[key] = json!(
            sequences
                .iter()
                .map(|s| s[key].as_u64().unwrap())
                .sum::<u64>()
        );
    }
    stats["slots"] = json!(sequences);
    stats["slot_reset"] = reset_stats;
    stats
}

pub fn load(mut options: ModelOptions, config: BatchConfig) -> Result<Box<dyn BatchModel>> {
    config.validate(&options)?;
    let raw: Value = serde_json::from_slice(&std::fs::read(options.path.join("config.json"))?)?;
    super::registry::require(&raw)?;
    ensure!(
        options.experts.storage != super::experts::Storage::Stream
            || matches!(
                raw["model_type"].as_str(),
                Some("qwen3_5_moe" | "qwen3_5_moe_text")
            ),
        "expert streaming requires a supported MoE checkpoint"
    );
    npu::configure_cpu(&mut options)?;
    match raw["model_type"].as_str() {
        #[cfg(any(
            feature = "model-qwen3",
            feature = "model-llama",
            feature = "model-granite"
        ))]
        Some("qwen3" | "llama" | "granite") => {
            Ok(Box::new(super::dense::Dense::load_batch(options, config)?))
        }
        #[cfg(feature = "model-qwen35")]
        Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text") => Ok(Box::new(
            super::qwen35::Qwen35::load_batch(options, config)?,
        )),
        _ => anyhow::bail!(
            "batch mode is unavailable for model type {}",
            raw["model_type"]
        ),
    }
}

pub(super) fn report(model: &dyn BatchModel) -> Result<()> {
    let stats = model.stats();
    eprintln!(
        "Batch startup: slots={}, total KV={} B, KV CPU metadata estimate={} B, recurrent state={} B",
        model.capabilities().max_slots,
        stats["kv_payload_bytes"],
        stats["kv_cpu_metadata_bytes_estimate"],
        stats["recurrent_state_bytes"],
    );
    for (slot, bytes) in stats["slots"].as_array().unwrap().iter().enumerate() {
        eprintln!("  Slot {slot}: {bytes}");
    }
    for domain in stats["domains"].as_array().unwrap() {
        eprintln!(
            "  IOMMU domain {}: KV={} B, weights={} B, Linear work={} B, Attention work estimate={} B, IOVA budget={} B, estimated headroom={} B",
            domain["iommu_domain_id"],
            domain["kv_payload_bytes"],
            domain["weight_payload_bytes"],
            domain["linear_device_workspace_bytes"],
            domain["attention_device_workspace_bytes_estimate"],
            domain["iova_budget_bytes"],
            domain["iova_headroom_estimate_bytes"],
        );
        ensure!(
            domain["iova_headroom_estimate_bytes"].as_u64().unwrap() > 0,
            "batch payload exceeds IOVA budget in domain {}",
            domain["iommu_domain_id"],
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests;
