//! FP32 gated delta recurrence, adapted from llama.cpp's head/window loop.
//! State is [value_head, value_channel, key_channel] (transposed from HF).
//! Upstream MIT attribution: llm_generated_docs/THIRD_PARTY_NOTICES.md.

use anyhow::{Result, ensure};
use rayon::ThreadPool;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnDimensions {
    pub(crate) key_heads: usize,
    pub(crate) value_heads: usize,
    pub(crate) key_dim: usize,
    pub(crate) value_dim: usize,
}

impl GdnDimensions {
    pub fn new(
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Result<Self> {
        ensure!(
            key_heads > 0 && value_heads > 0 && key_dim > 0 && value_dim > 0,
            "GDN dimensions must be nonzero"
        );
        ensure!(
            value_heads.is_multiple_of(key_heads),
            "GDN value heads must be divisible by key heads"
        );
        let key_width = key_heads.checked_mul(key_dim);
        let value_width = value_heads.checked_mul(value_dim);
        ensure!(
            key_width
                .and_then(|v| v.checked_mul(2))
                .zip(value_width)
                .and_then(|(k, v)| k.checked_add(v))
                .is_some(),
            "GDN width overflow"
        );
        ensure!(
            value_width
                .and_then(|v| v.checked_mul(key_dim))
                .and_then(|v| v.checked_mul(4))
                .is_some(),
            "GDN state size overflow"
        );
        Ok(Self {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        })
    }
    pub fn key_heads(self) -> usize {
        self.key_heads
    }
    pub fn value_heads(self) -> usize {
        self.value_heads
    }
    pub fn key_dim(self) -> usize {
        self.key_dim
    }
    pub fn value_dim(self) -> usize {
        self.value_dim
    }
    pub fn key_width(self) -> usize {
        self.key_heads * self.key_dim
    }
    pub fn value_width(self) -> usize {
        self.value_heads * self.value_dim
    }
    pub fn conv_width(self) -> usize {
        2 * self.key_width() + self.value_width()
    }
    pub fn state_elements(self) -> usize {
        self.value_width() * self.key_dim
    }

    fn validate_updates(self, state: &[f32], updates: Updates<'_>) -> Result<()> {
        ensure!(
            state.len() == self.state_elements(),
            "GDN state shape mismatch"
        );
        let rows = updates.rows;
        ensure!(
            rows.checked_mul(self.conv_width()).is_some(),
            "GDN batch size overflow"
        );
        ensure!(
            updates.key.len() == rows * self.key_width()
                && updates.value.len() == rows * self.value_width()
                && updates.log_decay.len() == rows * self.value_heads
                && updates.beta.len() == rows * self.value_heads,
            "GDN update shape mismatch"
        );
        ensure!(
            updates.log_decay.iter().all(|v| *v <= 0.0)
                && updates.beta.iter().all(|v| (0.0..=1.0).contains(v)),
            "invalid GDN update values"
        );
        Ok(())
    }
}

/// Keys have already undergone L2 normalization; values are post-convolution.
/// Rows are logical tokens only. No Linear padding may enter the recurrence.
#[derive(Clone, Copy)]
pub struct Updates<'a> {
    pub rows: usize,
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub log_decay: &'a [f32],
    pub beta: &'a [f32],
}

impl Updates<'_> {
    pub fn prefix(self, dimensions: GdnDimensions, rows: usize) -> Result<Self> {
        ensure!(
            rows <= self.rows,
            "GDN replay prefix exceeds verification rows"
        );
        ensure!(
            self.rows.checked_mul(dimensions.conv_width()).is_some()
                && self.key.len() == self.rows * dimensions.key_width()
                && self.value.len() == self.rows * dimensions.value_width()
                && self.log_decay.len() == self.rows * dimensions.value_heads
                && self.beta.len() == self.rows * dimensions.value_heads,
            "GDN replay log shape mismatch"
        );
        Ok(Self {
            rows,
            key: &self.key[..rows * dimensions.key_width()],
            value: &self.value[..rows * dimensions.value_width()],
            log_decay: &self.log_decay[..rows * dimensions.value_heads],
            beta: &self.beta[..rows * dimensions.value_heads],
        })
    }
}

mod native;
mod scalar;
pub use scalar::run_scalar;

#[derive(Default)]
pub struct Workspace {
    unfused_reference: bool,
}

impl Workspace {
    /// Literal ggml four-pass loop for paired component benchmarks.
    pub fn ggml_loop_reference() -> Self {
        Self {
            unfused_reference: true,
        }
    }

    pub fn allocated_bytes(&self) -> usize {
        0
    }

    pub fn run(
        &mut self,
        pool: &ThreadPool,
        dim: GdnDimensions,
        state: &mut [f32],
        query: &[f32],
        updates: Updates<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        self.validate(dim, state, query, updates, output)?;
        // The C++ kernel skips memcpy when source and destination are equal.
        native::run(
            pool,
            dim,
            None,
            state,
            Some(query),
            updates,
            Some(output),
            self.unfused_reference,
        );
        Ok(())
    }

    /// Copy each initial head on its owning worker, matching ggml's state copy.
    /// This also eliminates the head-major output scratch and serial transpose.
    #[allow(clippy::too_many_arguments)]
    pub fn run_from(
        &mut self,
        pool: &ThreadPool,
        dim: GdnDimensions,
        initial: &[f32],
        state: &mut [f32],
        query: &[f32],
        updates: Updates<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        ensure!(
            initial.len() == dim.state_elements(),
            "GDN initial state shape mismatch"
        );
        self.validate(dim, state, query, updates, output)?;
        native::run(
            pool,
            dim,
            Some(initial),
            state,
            Some(query),
            updates,
            Some(output),
            self.unfused_reference,
        );
        Ok(())
    }

    fn validate(
        &self,
        dim: GdnDimensions,
        state: &[f32],
        query: &[f32],
        updates: Updates<'_>,
        output: &[f32],
    ) -> Result<()> {
        dim.validate_updates(state, updates)?;
        ensure!(
            query.len() == updates.rows * dim.key_width()
                && output.len() == updates.rows * dim.value_width(),
            "GDN query/output shape mismatch"
        );
        Ok(())
    }
}

/// Restore only state transitions; queries and NPU projections are unnecessary.
pub fn replay(
    pool: &ThreadPool,
    dim: GdnDimensions,
    state: &mut [f32],
    updates: Updates<'_>,
) -> Result<()> {
    dim.validate_updates(state, updates)?;
    native::run(pool, dim, None, state, None, updates, None, false);
    Ok(())
}
