//! Prepared multimodal input. KV rows, RoPE coordinates and image identities
//! are distinct; NPU batch padding is never represented here.
use crate::vision::{ImageProcessor, MAX_IMAGES, PreparedImage};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ImageSpan {
    pub start: usize,
    pub image: Arc<PreparedImage>,
}
impl ImageSpan {
    pub fn end(&self) -> usize {
        self.start + self.image.tokens()
    }
}

#[derive(Debug, Clone)]
pub struct PreparedInput {
    pub tokens: Vec<u32>,
    pub positions: Vec<[usize; 3]>,
    pub images: Vec<ImageSpan>,
    pub next_position: usize,
    /// User message delimiters, in expanded KV rows.
    pub user_starts: Vec<usize>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RowIdentity {
    pub token: u32,
    pub position: [usize; 3],
    pub image: Option<([u8; 32], usize)>,
}
impl PreparedInput {
    pub fn text(tokens: Vec<u32>) -> Self {
        let next_position = tokens.len();
        let positions = (0..next_position).map(|p| [p; 3]).collect();
        Self {
            tokens,
            positions,
            images: vec![],
            next_position,
            user_starts: Vec::new(),
        }
    }
    pub fn expand(
        tokens: Vec<u32>,
        images: Vec<Arc<PreparedImage>>,
        processor: &ImageProcessor,
    ) -> Result<Self> {
        let mut result = Self::text(Vec::new());
        let mut images = images.into_iter();
        let mut pos = 0;
        for (i, &token) in tokens.iter().enumerate() {
            ensure!(
                token != processor.video_token,
                "video inputs are unsupported"
            );
            if token == processor.image_token {
                ensure!(
                    i > 0
                        && tokens[i - 1] == processor.vision_start
                        && tokens.get(i + 1) == Some(&processor.vision_end),
                    "invalid image placeholder; use image content parts"
                );
                let image = images
                    .next()
                    .context("image placeholder has no image content")?;
                image.validate()?;
                let (h, w) = (image.grid[1] / 2, image.grid[2] / 2);
                let start = result.tokens.len();
                for y in 0..h {
                    for x in 0..w {
                        result.tokens.push(token);
                        result.positions.push([pos, pos + y, pos + x]);
                    }
                }
                result.images.push(ImageSpan { start, image });
                pos += h.max(w);
            } else {
                result.tokens.push(token);
                result.positions.push([pos; 3]);
                pos += 1;
            }
        }
        ensure!(
            images.next().is_none(),
            "chat template did not render every image"
        );
        result.next_position = pos;
        Ok(result)
    }
    pub fn set_user_starts(&mut self, raw_positions: &[usize]) {
        self.user_starts = raw_positions
            .iter()
            .map(|&position| {
                let mut extra = 0;
                for span in &self.images {
                    if span.start - extra >= position {
                        break;
                    }
                    extra += span.image.tokens() - 1;
                }
                position + extra
            })
            .collect();
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.tokens.len() == self.positions.len(),
            "prepared token/position shape mismatch"
        );
        ensure!(self.images.len() <= MAX_IMAGES, "too many prepared images");
        ensure!(
            self.positions
                .iter()
                .flatten()
                .all(|p| *p < self.next_position),
            "invalid prepared RoPE coordinates"
        );
        let mut end = 0;
        for span in &self.images {
            span.image.validate()?;
            ensure!(
                span.start >= end
                    && span.start <= self.tokens.len()
                    && span.image.tokens() <= self.tokens.len() - span.start,
                "overlapping or out-of-range image span"
            );
            end = span.end();
        }
        Ok(())
    }
    pub(crate) fn row(&self, index: usize) -> RowIdentity {
        let image = self
            .images
            .iter()
            .find(|s| index >= s.start && index < s.end())
            .map(|s| (s.image.identity, index - s.start));
        RowIdentity {
            token: self.tokens[index],
            position: self.positions[index],
            image,
        }
    }
}

/// Validate content and prepare images on the caller thread, without NPU work.
pub fn message_images(
    messages: &[Value],
    processor: Option<&ImageProcessor>,
) -> Result<Vec<Arc<PreparedImage>>> {
    let mut images = vec![];
    let mut bytes = 0;
    for message in messages {
        if let Some(parts) = message["content"].as_array() {
            for part in parts {
                match part["type"].as_str() {
                    Some("text") => {
                        ensure!(part["text"].is_string(), "text parts need text");
                    }
                    Some("image_url") => {
                        ensure!(
                            message["role"] == "user",
                            "images are supported in user messages only"
                        );
                        let processor = processor.context("this model does not support images")?;
                        ensure!(
                            images.len() < MAX_IMAGES,
                            "at most 8 images are supported per conversation"
                        );
                        let url = part["image_url"]["url"]
                            .as_str()
                            .context("image_url needs a URL string")?;
                        let (image, size) = processor.prepare_url(url)?;
                        bytes += size;
                        ensure!(
                            bytes <= crate::vision::processor_limit(),
                            "image content exceeds 16 MiB request limit"
                        );
                        images.push(Arc::new(image));
                    }
                    _ => anyhow::bail!("supported content part types are text and image_url"),
                }
            }
        }
    }
    Ok(images)
}
