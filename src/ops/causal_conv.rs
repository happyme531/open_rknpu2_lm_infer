//! Stateful depthwise causal convolution followed by SiLU.
use super::activation::silu;
use anyhow::{Context, Result, ensure};
use rayon::{ThreadPool, prelude::*};

mod kernel;

/// Channel-major, oldest-to-newest K-1 samples. K=1 has empty history.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvState {
    channels: usize,
    kernel: usize,
    history: Vec<f32>,
}

impl ConvState {
    pub fn new(channels: usize, kernel: usize) -> Result<Self> {
        ensure!(channels > 0 && kernel > 0, "invalid convolution dimensions");
        let elements = channels
            .checked_mul(kernel - 1)
            .context("convolution size overflow")?;
        Ok(Self {
            channels,
            kernel,
            history: vec![0.0; elements],
        })
    }
    pub fn history(&self) -> &[f32] {
        &self.history
    }
    #[cfg(feature = "arch-qwen35")]
    pub(crate) fn history_mut(&mut self) -> &mut [f32] {
        &mut self.history
    }
    pub fn reset(&mut self) {
        self.history.fill(0.0);
    }
    /// Restore compatible history without replacing its allocation.
    pub fn restore(&mut self, snapshot: &Self) -> Result<()> {
        ensure!(
            self.channels == snapshot.channels && self.kernel == snapshot.kernel,
            "convolution checkpoint dimensions mismatch"
        );
        self.history.copy_from_slice(&snapshot.history);
        Ok(())
    }
    pub fn allocated_bytes(&self) -> usize {
        self.history.capacity() * 4
    }

    pub fn run(&mut self, input: &[f32], weights: &[f32], output: &mut [f32]) -> Result<()> {
        ensure!(
            input.len().is_multiple_of(self.channels)
                && input.len() == output.len()
                && self.channels.checked_mul(self.kernel) == Some(weights.len()),
            "convolution shape mismatch"
        );
        for (input, output) in input
            .chunks(self.channels)
            .zip(output.chunks_mut(self.channels))
        {
            for c in 0..self.channels {
                let base = c * (self.kernel - 1);
                let history = &self.history[base..base + self.kernel - 1];
                let weight = &weights[c * self.kernel..(c + 1) * self.kernel];
                let mut value = history.iter().zip(weight).map(|(x, w)| x * w).sum::<f32>();
                value += input[c] * weight[self.kernel - 1];
                output[c] = silu(value);
            }
            self.append_row(input);
        }
        Ok(())
    }

    /// Tokens read their causal window directly from input plus the old history.
    /// No per-token history moves or scalar SiLU calls in the batch path.
    pub fn run_parallel(
        &mut self,
        pool: &ThreadPool,
        input: &[f32],
        weights: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        self.run_parallel_strided(pool, input, self.channels, weights, output)
    }

    pub(crate) fn run_parallel_strided(
        &mut self,
        pool: &ThreadPool,
        input: &[f32],
        stride: usize,
        weights: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        let rows = output.len() / self.channels;
        ensure!(
            output.len().is_multiple_of(self.channels)
                && stride >= self.channels
                && (rows == 0 && input.is_empty()
                    || rows
                        .checked_sub(1)
                        .and_then(|n| n.checked_mul(stride))
                        .and_then(|n| n.checked_add(self.channels))
                        == Some(input.len()))
                && self.channels.checked_mul(self.kernel) == Some(weights.len()),
            "convolution shape mismatch"
        );
        if rows == 1 {
            kernel::row(self, input, stride, weights, 0, output);
        } else {
            pool.install(|| {
                output
                    .par_chunks_mut(self.channels)
                    .enumerate()
                    .for_each(|(t, out)| kernel::row(self, input, stride, weights, t, out));
            });
        }
        self.append_strided(input, stride, rows);
        Ok(())
    }

    fn append_row(&mut self, row: &[f32]) {
        if self.kernel == 1 {
            return;
        }
        for (c, history) in self.history.chunks_mut(self.kernel - 1).enumerate() {
            history.copy_within(1.., 0);
            history[self.kernel - 2] = row[c];
        }
    }

    pub fn append(&mut self, input: &[f32]) -> Result<()> {
        ensure!(
            input.len().is_multiple_of(self.channels),
            "convolution replay shape mismatch"
        );
        let rows = input.len() / self.channels;
        self.append_strided(input, self.channels, rows);
        Ok(())
    }

    fn append_strided(&mut self, input: &[f32], stride: usize, rows: usize) {
        if self.kernel == 1 || rows == 0 {
            return;
        }
        let keep = rows.min(self.kernel - 1);
        for (c, history) in self.history.chunks_mut(self.kernel - 1).enumerate() {
            history.copy_within(keep.., 0);
            for (j, value) in history[self.kernel - 1 - keep..].iter_mut().enumerate() {
                *value = input[(rows - keep + j) * stride + c];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn restore_reuses_history_and_rejects_incompatible_geometry() -> Result<()> {
        for channels in [3, 2048] {
            for kernel in [1, 4] {
                let mut snapshot = ConvState::new(channels, kernel)?;
                let input: Vec<_> = (0..channels * 5)
                    .map(|i| if i % 7 == 0 { -0.0 } else { i as f32 * 0.03 })
                    .collect();
                snapshot.append(&input)?;
                let mut state = ConvState::new(channels, kernel)?;
                let pointer = state.history.as_ptr();
                let capacity = state.history.capacity();
                state.restore(&snapshot)?;
                assert_eq!(state.history.as_ptr(), pointer);
                assert_eq!(state.history.capacity(), capacity);
                assert!(
                    state
                        .history
                        .iter()
                        .zip(&snapshot.history)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );

                let mut expected = snapshot.clone();
                let weights = vec![0.25; channels * kernel];
                let mut actual = vec![0.; input.len()];
                let mut reference = actual.clone();
                state.run(&input, &weights, &mut actual)?;
                expected.run(&input, &weights, &mut reference)?;
                assert_eq!(actual, reference);
                assert_eq!(state, expected);
            }
        }
        let snapshot = ConvState::new(3, 3)?;
        let mut state = ConvState::new(6, 2)?;
        state.append(&[2.; 6])?;
        let before = state.clone();
        let pointer = state.history.as_ptr();
        // Equal buffer lengths still require equal convolution geometry.
        assert!(state.restore(&snapshot).is_err());
        assert_eq!(state, before);
        assert_eq!(state.history.as_ptr(), pointer);
        Ok(())
    }

    #[test]
    fn strided_decode_matches_batch_bits_and_history() -> Result<()> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        for channels in [5, 2048] {
            let (rows, stride) = (9, channels * 5);
            let weights: Vec<_> = (0..channels * 4)
                .map(|i| (i as f32 * 0.017).sin())
                .collect();
            let input: Vec<_> = (0..(rows - 1) * stride + channels)
                .map(|i| (i as f32 * 0.013).cos())
                .collect();
            let mut full = ConvState::new(channels, 4)?;
            full.append(&vec![0.3; channels * 3])?;
            let mut split = full.clone();
            let mut expected = vec![0.; rows * channels];
            let mut actual = expected.clone();
            full.run_parallel_strided(&pool, &input, stride, &weights, &mut expected)?;
            for t in 0..rows {
                split.run_parallel_strided(
                    &pool,
                    &input[t * stride..t * stride + channels],
                    stride,
                    &weights,
                    &mut actual[t * channels..(t + 1) * channels],
                )?;
            }
            assert_eq!(actual, expected);
            assert_eq!(split, full);
        }
        Ok(())
    }
}
