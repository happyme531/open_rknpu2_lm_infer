use anyhow::{Context, Result, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{io::Cursor, path::Path};

use super::{ImageProcessor, MAX_IMAGE_BYTES, PreparedImage};

impl ImageProcessor {
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let root: Value = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        if root["model_type"] != "qwen3_5" || root["vision_config"].is_null() {
            return Ok(None);
        }
        let p: Value =
            serde_json::from_slice(&std::fs::read(path.join("preprocessor_config.json"))?)?;
        ensure!(
            p["patch_size"] == 16 && p["merge_size"] == 2 && p["temporal_patch_size"] == 2,
            "unsupported vision processor patch geometry"
        );
        let id = |key: &str| {
            root[key]
                .as_u64()
                .and_then(|x| u32::try_from(x).ok())
                .with_context(|| format!("missing {key}"))
        };
        let mean: [f32; 3] = serde_json::from_value(p["image_mean"].clone())?;
        let std: [f32; 3] = serde_json::from_value(p["image_std"].clone())?;
        ensure!(
            mean.iter().all(|x| x.is_finite()) && std.iter().all(|x| x.is_finite() && *x > 0.0),
            "invalid image normalization"
        );
        let max_pixels = p["size"]["longest_edge"]
            .as_u64()
            .context("missing image max pixels")? as usize;
        let min_pixels = p["size"]["shortest_edge"]
            .as_u64()
            .context("missing image min pixels")? as usize;
        ensure!(
            min_pixels > 0 && min_pixels <= max_pixels,
            "invalid image pixel budget"
        );
        Ok(Some(Self {
            image_token: id("image_token_id")?,
            vision_start: id("vision_start_token_id")?,
            vision_end: id("vision_end_token_id")?,
            video_token: id("video_token_id")?,
            min_pixels,
            max_pixels,
            mean,
            std,
        }))
    }
    pub fn prepare_url(&self, url: &str) -> Result<(PreparedImage, usize)> {
        let bytes = decode_data_url(url)?;
        let len = bytes.len();
        Ok((self.prepare(&bytes)?, len))
    }
    pub fn prepare(&self, bytes: &[u8]) -> Result<PreparedImage> {
        ensure!(
            bytes.len() <= MAX_IMAGE_BYTES,
            "image exceeds 8 MiB encoded limit"
        );
        let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
        ensure!(
            matches!(reader.format(), Some(ImageFormat::Png | ImageFormat::Jpeg)),
            "images must be PNG or JPEG"
        );
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(32768);
        limits.max_image_height = Some(32768);
        limits.max_alloc = Some(128 * 1024 * 1024);
        reader.limits(limits);
        let mut decoder = reader.into_decoder().context("decode image header")?;
        let orientation = decoder.orientation().context("read image orientation")?;
        let mut decoded = DynamicImage::from_decoder(decoder).context("decode image")?;
        decoded.apply_orientation(orientation);
        let decoded = decoded.to_rgb8();
        let (sw, sh) = decoded.dimensions();
        let (height, width) =
            smart_resize(sh as usize, sw as usize, self.min_pixels, self.max_pixels)?;
        let rgb = super::resize::bicubic(decoded, width as u32, height as u32);
        let (gh, gw) = (height / 16, width / 16);
        let mut patches = Vec::with_capacity(gh * gw * 1536);
        for bh in 0..gh / 2 {
            for bw in 0..gw / 2 {
                for ih in 0..2 {
                    for iw in 0..2 {
                        for c in 0..3 {
                            for _ in 0..2 {
                                for y in 0..16 {
                                    for x in 0..16 {
                                        let pixel = rgb.get_pixel(
                                            (bw * 32 + iw * 16 + x) as u32,
                                            (bh * 32 + ih * 16 + y) as u32,
                                        )[c];
                                        patches.push(
                                            (pixel as f32 * (1.0 / 255.0) - self.mean[c])
                                                / self.std[c],
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut hash = Sha256::new();
        hash.update(b"qwen35-pil-bicubic-v1");
        hash.update((height as u64).to_le_bytes());
        hash.update((width as u64).to_le_bytes());
        for v in self.mean.iter().chain(&self.std) {
            hash.update(v.to_le_bytes());
        }
        hash.update(rgb.as_raw());
        Ok(PreparedImage {
            identity: hash.finalize().into(),
            grid: [1, gh, gw],
            patches: patches.into(),
        })
    }
}

fn smart_resize(h: usize, w: usize, min: usize, max: usize) -> Result<(usize, usize)> {
    ensure!(
        h > 0 && w > 0 && h.max(w) as f64 / h.min(w) as f64 <= 200.0,
        "invalid image aspect ratio (maximum 200)"
    );
    let f = 32.0;
    let (mut rh, mut rw) = (
        (h as f64 / f).round_ties_even() as usize * 32,
        (w as f64 / f).round_ties_even() as usize * 32,
    );
    if rh * rw > max {
        let beta = ((h * w) as f64 / max as f64).sqrt();
        rh = ((h as f64 / beta / f).floor() as usize * 32).max(32);
        rw = ((w as f64 / beta / f).floor() as usize * 32).max(32);
    } else if rh * rw < min {
        let beta = (min as f64 / (h * w) as f64).sqrt();
        rh = (h as f64 * beta / f).ceil() as usize * 32;
        rw = (w as f64 * beta / f).ceil() as usize * 32;
    }
    Ok((rh, rw))
}
pub fn decode_data_url(url: &str) -> Result<Vec<u8>> {
    let (header, encoded) = url
        .split_once(',')
        .context("image_url must be an inline PNG/JPEG base64 data URL")?;
    ensure!(
        matches!(header, "data:image/png;base64" | "data:image/jpeg;base64"),
        "image_url must be an inline PNG/JPEG base64 data URL; remote URLs are unsupported"
    );
    ensure!(
        encoded.len() <= MAX_IMAGE_BYTES.div_ceil(3) * 4,
        "image exceeds 8 MiB encoded limit"
    );
    let bytes = STANDARD.decode(encoded).context("invalid image base64")?;
    ensure!(
        bytes.len() <= MAX_IMAGE_BYTES,
        "image exceeds 8 MiB encoded limit"
    );
    Ok(bytes)
}
pub fn data_url(path: &Path) -> Result<String> {
    let len = std::fs::metadata(path)?.len();
    ensure!(
        len <= MAX_IMAGE_BYTES as u64,
        "image exceeds 8 MiB encoded limit"
    );
    let bytes = std::fs::read(path)?;
    let mime = match image::guess_format(&bytes)? {
        ImageFormat::Png => "image/png",
        ImageFormat::Jpeg => "image/jpeg",
        _ => anyhow::bail!("images must be PNG or JPEG"),
    };
    Ok(format!("data:{mime};base64,{}", STANDARD.encode(bytes)))
}
