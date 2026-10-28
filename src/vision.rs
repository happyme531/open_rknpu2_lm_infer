//! Shared image input contract and decoding, independent of model selection.
use anyhow::{Result, ensure};
use std::sync::Arc;
pub mod math;
mod processor;
mod resize;
pub use processor::data_url;

pub const MAX_IMAGES: usize = 8;
pub const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_REQUEST_IMAGE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct PreparedImage {
    pub identity: [u8; 32],
    pub grid: [usize; 3],
    pub patches: Arc<[f32]>,
}
impl PreparedImage {
    pub fn validate(&self) -> Result<()> {
        let [t, h, w] = self.grid;
        ensure!(
            t == 1
                && h >= 2
                && w >= 2
                && h.is_multiple_of(2)
                && w.is_multiple_of(2)
                && h.checked_mul(w)
                    .is_some_and(|n| n.checked_mul(1536) == Some(self.patches.len())),
            "invalid prepared image geometry"
        );
        ensure!(
            self.patches.iter().all(|x| x.is_finite()),
            "nonfinite image patches"
        );
        Ok(())
    }
    pub fn tokens(&self) -> usize {
        self.grid[1] * self.grid[2] / 4
    }
}

#[derive(Debug, Clone)]
pub struct ImageProcessor {
    pub image_token: u32,
    pub vision_start: u32,
    pub vision_end: u32,
    pub video_token: u32,
    min_pixels: usize,
    max_pixels: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

pub(crate) fn processor_limit() -> usize {
    MAX_REQUEST_IMAGE_BYTES
}
