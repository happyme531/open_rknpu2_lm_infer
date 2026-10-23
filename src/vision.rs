//! Shared image input contract; decoding is included only with vision support.
use anyhow::{Result, ensure};
use std::sync::Arc;
#[cfg(feature = "vision")]
pub mod math;
#[cfg(feature = "vision")]
mod processor;
#[cfg(feature = "vision")]
mod resize;
#[cfg(feature = "vision")]
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
    #[cfg(feature = "vision")]
    min_pixels: usize,
    #[cfg(feature = "vision")]
    max_pixels: usize,
    #[cfg(feature = "vision")]
    mean: [f32; 3],
    #[cfg(feature = "vision")]
    std: [f32; 3],
}

pub(crate) fn processor_limit() -> usize {
    MAX_REQUEST_IMAGE_BYTES
}

#[cfg(not(feature = "vision"))]
impl ImageProcessor {
    pub fn load(_path: &std::path::Path) -> Result<Option<Self>> {
        Ok(None)
    }
    pub fn prepare_url(&self, _url: &str) -> Result<(PreparedImage, usize)> {
        anyhow::bail!("image support is not included in this build")
    }
    pub fn prepare(&self, _bytes: &[u8]) -> Result<PreparedImage> {
        anyhow::bail!("image support is not included in this build")
    }
}
#[cfg(not(feature = "vision"))]
pub fn data_url(_path: &std::path::Path) -> Result<String> {
    anyhow::bail!("image support is not included in this build")
}
