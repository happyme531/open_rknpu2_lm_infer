use super::storage::Element;
use super::*;
use crate::{
    model::{ModelOptions, VisionDtype, checkpoint::TensorStore},
    vision::{PreparedImage, math},
};
use anyhow::{Result, ensure};
use rayon::ThreadPool;
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

const CACHE_BYTES: usize = 32 * 1024 * 1024;
type Features = Arc<[f32]>;
struct StageTimer {
    last: Option<Instant>,
    totals: BTreeMap<&'static str, f64>,
}
impl StageTimer {
    fn new(enabled: bool) -> Self {
        Self {
            last: enabled.then(Instant::now),
            totals: BTreeMap::new(),
        }
    }
    fn mark(&mut self, name: &'static str) {
        if let Some(previous) = self.last {
            let now = Instant::now();
            *self.totals.entry(name).or_default() +=
                now.duration_since(previous).as_secs_f64() * 1000.0;
            self.last = Some(now);
        }
    }
    fn finish(self) -> Option<BTreeMap<&'static str, f64>> {
        self.last.map(|_| self.totals)
    }
}
pub struct Encoder {
    weights: Weights,
    dtype: VisionDtype,
    cache: VecDeque<([u8; 32], Features)>,
    bytes: usize,
    trace: crate::model::trace::Trace,
    profile_ms: Option<BTreeMap<&'static str, f64>>,
    pub calls: usize,
    pub hits: usize,
    pub elapsed_ms: f64,
}
impl Encoder {
    pub fn load(options: &ModelOptions, store: &TensorStore, domain: u32) -> Result<Self> {
        Self::load_cancellable(options, store, domain, None)
    }
    pub(crate) fn load_cancellable(
        options: &ModelOptions,
        store: &TensorStore,
        domain: u32,
        cancel: Option<&AtomicBool>,
    ) -> Result<Self> {
        Ok(Self {
            weights: super::load::load(options, store, domain, cancel)?,
            dtype: options.vision_dtype,
            cache: VecDeque::new(),
            bytes: 0,
            trace: crate::model::trace::Trace::from_environment()?,
            profile_ms: None,
            calls: 0,
            hits: 0,
            elapsed_ms: 0.0,
        })
    }
    /// Opt-in complete-encoder stage timings for diagnostic probes.
    pub fn enable_profiling(&mut self) {
        self.profile_ms = Some(BTreeMap::new());
    }
    pub fn profile_ms(&self) -> Option<&BTreeMap<&'static str, f64>> {
        self.profile_ms.as_ref()
    }
    /// Drop image features while retaining weights and prepared workspaces.
    pub fn clear_feature_cache(&mut self) {
        self.cache.clear();
        self.bytes = 0;
    }
    pub fn encode(
        &mut self,
        image: &PreparedImage,
        pool: &ThreadPool,
        cancel: &AtomicBool,
    ) -> Result<Features> {
        match self.dtype {
            VisionDtype::Fp16 => self.encode_with::<u16>(image, pool, cancel),
            VisionDtype::Fp32 => self.encode_with::<f32>(image, pool, cancel),
        }
    }
    fn encode_with<T: Element>(
        &mut self,
        image: &PreparedImage,
        pool: &ThreadPool,
        cancel: &AtomicBool,
    ) -> Result<Features> {
        ensure!(!cancel.load(Ordering::Relaxed), "generation cancelled");
        image.validate()?;
        if let Some(i) = self.cache.iter().position(|(id, _)| id == &image.identity) {
            let entry = self.cache.remove(i).unwrap();
            let features = entry.1.clone();
            self.cache.push_back(entry);
            self.hits += 1;
            return Ok(features);
        }
        let start = Instant::now();
        let mut stages = StageTimer::new(self.profile_ms.is_some());
        let w = &self.weights;
        let c = &w.config;
        let rows = image.grid[1] * image.grid[2];
        let h = c.hidden;
        let mut attention = w.backend.full_attention(c.heads, h / c.heads, rows)?;
        let mut hidden = vec![T::default(); rows * h];
        let mut normalized = hidden.clone();
        let mut qkv = vec![T::default(); rows * 3 * h];
        let mut q = hidden.clone();
        let mut k = hidden.clone();
        let mut v = hidden.clone();
        let mut attended = hidden.clone();
        let mut projected = hidden.clone();
        let mut ff = vec![T::default(); rows * c.intermediate];
        let positions = math::patch_positions(image.grid[1], image.grid[2]);
        stages.mark("allocation");
        let rotary = T::prepare_rotary(c.heads, h / c.heads, &positions)?;
        stages.mark("rope_prepare");
        T::linear(&w.patch, &T::input(&image.patches), &mut hidden, rows)?;
        stages.mark("patch_linear");
        T::position(
            &mut hidden,
            &w.position,
            h,
            c.side,
            image.grid[1],
            image.grid[2],
        )?;
        T::trace(&self.trace, -1, "vision_patch_position", &hidden)?;
        stages.mark("position");
        for (index, block) in w.blocks.iter().enumerate() {
            ensure!(!cancel.load(Ordering::Relaxed), "generation cancelled");
            T::norm(&block.norm1, pool, &hidden, &mut normalized)?;
            stages.mark("norm1");
            T::linear(&block.qkv, &normalized, &mut qkv, rows)?;
            stages.mark("qkv_linear");
            for r in 0..rows {
                q[r * h..(r + 1) * h].copy_from_slice(&qkv[r * 3 * h..r * 3 * h + h]);
                k[r * h..(r + 1) * h].copy_from_slice(&qkv[r * 3 * h + h..r * 3 * h + 2 * h]);
                v[r * h..(r + 1) * h].copy_from_slice(&qkv[r * 3 * h + 2 * h..(r + 1) * 3 * h]);
            }
            T::rotary(&rotary, pool, &mut q)?;
            T::rotary(&rotary, pool, &mut k)?;
            stages.mark("qkv_split_rope");
            T::attention(&mut attention, &q, &k, &v, &mut attended, rows)?;
            stages.mark("attention");
            T::linear(&block.proj, &attended, &mut projected, rows)?;
            stages.mark("attention_output_linear");
            T::add(pool, &mut hidden, &projected)?;
            stages.mark("residual1");
            T::norm(&block.norm2, pool, &hidden, &mut normalized)?;
            stages.mark("norm2");
            T::linear_gelu(&block.fc1, pool, &normalized, &mut ff, rows)?;
            stages.mark("mlp_up_gelu");
            T::linear(&block.fc2, &ff, &mut projected, rows)?;
            stages.mark("mlp_down_linear");
            T::add(pool, &mut hidden, &projected)?;
            T::trace(&self.trace, index as i32, "vision_hidden", &hidden)?;
            stages.mark("residual2_trace");
        }
        drop(attention);
        ensure!(!cancel.load(Ordering::Relaxed), "generation cancelled");
        T::norm(&w.merger.norm, pool, &hidden, &mut normalized)?;
        stages.mark("merger_norm");
        let merged = rows / 4;
        let mut temp = vec![T::default(); rows * h];
        stages.mark("merger_allocation");
        T::linear(&w.merger.fc1, &normalized, &mut temp, merged)?;
        stages.mark("merger_fc1");
        T::merger_gelu(pool, &mut temp);
        stages.mark("merger_gelu");
        let mut features = vec![T::default(); merged * c.out];
        T::linear(&w.merger.fc2, &temp, &mut features, merged)?;
        stages.mark("merger_fc2");
        // The language-model embedding interface consumes FP32 features.
        let features = T::output(features);
        stages.mark("features_widen");
        self.trace.save(0, -1, "vision_features", &features)?;
        ensure!(!cancel.load(Ordering::Relaxed), "generation cancelled");
        let features: Features = features.into();
        let size = features.len() * 4;
        while self.bytes + size > CACHE_BYTES {
            if let Some((_, old)) = self.cache.pop_front() {
                self.bytes -= old.len() * 4;
            } else {
                break;
            }
        }
        if size <= CACHE_BYTES {
            self.cache.push_back((image.identity, features.clone()));
            self.bytes += size;
        }
        self.calls += 1;
        stages.mark("finish");
        self.profile_ms = stages.finish();
        self.elapsed_ms += start.elapsed().as_secs_f64() * 1000.0;
        Ok(features)
    }
    pub fn cache_bytes(&self) -> usize {
        self.bytes
    }
}
