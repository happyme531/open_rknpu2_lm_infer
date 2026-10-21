use crate::model::ModelOptions;
use anyhow::{Context, Result, bail, ensure};
use half::f16;
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    ffi::{CStr, CString},
    ptr,
    rc::Rc,
};

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/npu_bindings.rs"));
}
use ffi::*;
mod attention;
pub use attention::{Attention, FullAttention};
mod gated_delta;
pub use gated_delta::GatedDelta;
mod floating;
pub use floating::{FloatBackend, FloatLinear};
mod moe;
pub use moe::Moe;
mod policy;
#[cfg(test)]
mod vision_tests;

#[cfg(test)]
#[path = "npu/narrow_projection.rs"]
mod narrow_projection;

#[cfg(test)]
#[path = "npu/ubatch_probe.rs"]
mod ubatch_probe;

unsafe extern "C" {
    fn omp_set_dynamic(value: libc::c_int);
    fn omp_set_num_threads(value: libc::c_int);
}

pub fn configure_cpu(options: &mut ModelOptions) -> Result<()> {
    ensure!(
        (1..=4).contains(&options.threads),
        "threads must be within 1..4"
    );
    ensure!(
        (1..=256).contains(&options.chunk),
        "ubatch-size must be within 1..256"
    );
    ensure!(
        (1..=32768).contains(&options.context),
        "ctx-size must be within 1..32768"
    );
    ensure!(
        [256, 512, 1024, 2048, 4096].contains(&options.attention_tile),
        "attention tile must be 256/512/1024/2048/4096"
    );
    ensure!(
        [1, 2, 4, 3, 7].contains(&options.npu_mask),
        "NPU mask must be 1/2/4/3/7"
    );
    unsafe {
        let mut available: libc::cpu_set_t = std::mem::zeroed();
        ensure!(
            libc::sched_getaffinity(0, std::mem::size_of_val(&available), &mut available) == 0,
            "cannot read CPU affinity: {}",
            std::io::Error::last_os_error()
        );
        let mut mask = options.cpu_mask;
        if mask == 0 {
            for cpu in (4..8).chain(0..64) {
                if mask.count_ones() as usize == options.threads {
                    break;
                }
                if libc::CPU_ISSET(cpu, &available) {
                    mask |= 1u64 << cpu;
                }
            }
        }
        ensure!(
            mask.count_ones() as usize == options.threads,
            "CPU mask must contain exactly --threads available CPUs"
        );
        let mut selected: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut selected);
        for cpu in 0..64 {
            if mask & (1u64 << cpu) != 0 {
                ensure!(
                    libc::CPU_ISSET(cpu, &available),
                    "requested CPU {cpu} unavailable"
                );
                libc::CPU_SET(cpu, &mut selected);
            }
        }
        ensure!(
            libc::sched_setaffinity(0, std::mem::size_of_val(&selected), &selected) == 0,
            "cannot pin CPU workers: {}",
            std::io::Error::last_os_error()
        );
        options.cpu_mask = mask;
        omp_set_dynamic(0);
        omp_set_num_threads(options.threads as i32);
    }
    Ok(())
}

pub struct QuantMatrix {
    pub bits: u32,
    pub k: usize,
    pub n: usize,
    pub codes: Vec<i8>,
    pub scales: Vec<f32>,
}
impl QuantMatrix {
    fn validate(&self) -> Result<()> {
        ensure!(
            [4, 8].contains(&self.bits),
            "linear weights must use 4 or 8 bits"
        );
        ensure!(
            self.k > 0 && self.n > 0 && self.k <= i32::MAX as usize && self.n <= i32::MAX as usize,
            "invalid linear dimensions"
        );
        ensure!(
            self.k.checked_mul(self.n) == Some(self.codes.len()) && self.scales.len() == self.n,
            "linear weight shape mismatch"
        );
        Ok(())
    }
}
pub type Reload = Box<dyn Fn() -> Result<QuantMatrix>>;
type Key = (u32, usize, usize, usize);
type WorkKey = (Key, usize);

struct Handle<T> {
    raw: *mut T,
    free: unsafe extern "C" fn(*mut T),
}
impl<T> Handle<T> {
    fn new(raw: *mut T, free: unsafe extern "C" fn(*mut T), name: &str) -> Result<Self> {
        ensure!(!raw.is_null(), "cannot allocate {name}");
        Ok(Self { raw, free })
    }
}
impl<T> Drop for Handle<T> {
    fn drop(&mut self) {
        unsafe { (self.free)(self.raw) };
    }
}
fn check(rc: i32, name: &str) -> Result<()> {
    if rc != 0 {
        let detail = unsafe { CStr::from_ptr(rk_npu_strerror(rc)) }.to_string_lossy();
        bail!("{name}: {detail} ({rc})");
    }
    Ok(())
}
fn check_cache_miss(rc: i32, name: &str) -> Result<()> {
    if rc == RK_NPU_ERR_CACHE_MISS {
        Ok(())
    } else {
        check(rc, name)
    }
}
fn bucket(rows: usize) -> Result<usize> {
    ensure!(
        (1..=256).contains(&rows),
        "linear batch must be within 1..256"
    );
    Ok(if rows <= 128 {
        rows.next_power_of_two()
    } else {
        rows.div_ceil(64) * 64
    })
}

struct Work {
    w4: Option<Handle<rk_npu_w4a8_workspace>>,
    w8: Option<Handle<rk_npu_matmul_workspace>>,
    input: Vec<f32>,
    output: Vec<f32>,
    bytes: u64,
    w4_reduce: rk_npu_w4a8_reduce_backend,
}
struct AttentionWork {
    workspace: Handle<rk_npu_attention_f16_workspace>,
    config: rk_npu_attention_f16_config,
    active_core_mask: u32,
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    output: Vec<f32>,
    output_half: Vec<u16>,
}
impl AttentionWork {
    fn new(
        domain: *mut rk_npu_iommu_domain,
        config: rk_npu_attention_f16_config,
        kv_rows: usize,
    ) -> Result<Self> {
        let mut sizes = rk_npu_attention_f16_sizes::default();
        check(
            unsafe {
                rk_npu_attention_f16_query(
                    &config,
                    config.initial_capacity,
                    config.max_query_rows,
                    config.initial_capacity,
                    &mut sizes,
                )
            },
            "query native Attention sizes",
        )?;
        let mut raw = ptr::null_mut();
        check(
            unsafe { rk_npu_attention_f16_prepare(domain, &config, &mut raw) },
            "prepare native Attention",
        )?;
        let query_elements =
            config.query_heads as usize * config.max_query_rows as usize * config.head_dim as usize;
        let kv_elements = config.kv_heads as usize * kv_rows * config.head_dim as usize;
        Ok(Self {
            workspace: Handle::new(
                raw,
                rk_npu_attention_f16_workspace_free,
                "attention workspace",
            )?,
            config,
            active_core_mask: sizes.active_core_mask,
            q: vec![0; query_elements],
            k: vec![0; kv_elements],
            v: vec![0; kv_elements],
            output: vec![0.0; query_elements],
            output_half: Vec::new(),
        })
    }
}
struct Material {
    matrix: QuantMatrix,
    reload: Option<Reload>,
    profile: Option<Handle<rk_npu_w4a8_tuning_profile>>,
}
struct BackendInner {
    work: BTreeMap<WorkKey, Work>,
    moe_work: BTreeMap<moe::Key, moe::Work>,
    attention: Option<AttentionWork>,
    material: BTreeMap<Key, Material>,
    w8_partitions: BTreeMap<(usize, usize), usize>,
    w8_strategies: BTreeMap<WorkKey, rk_npu_matmul_strategy>,
    w4_strategies: BTreeMap<WorkKey, rk_npu_w4a8_strategy_ex>,
    options: ModelOptions,
    domain_id: u32,
    cache_hits: usize,
    cache_misses: usize,
    tune_calls: usize,
    fallbacks: BTreeSet<WorkKey>,
    fallback_calls: usize,
    weight_reload_calls: usize,
    loaded_w4: usize,
    loaded_w8: usize,
    weight_bytes: u64,
    domain: Handle<rk_npu_iommu_domain>,
    context: Handle<rk_npu_ctx>,
}

/// A domain owns shared workspaces; layer weights keep it alive through `Rc`.
#[derive(Clone)]
pub struct Backend(Rc<RefCell<BackendInner>>);
pub struct Linear {
    w4: Option<Handle<rk_npu_w4a8_weights>>,
    w8: Option<Handle<rk_npu_f32i8f32_weights>>,
    key: Key,
    owner: Backend,
}

fn attention_config(
    options: &ModelOptions,
    query_heads: usize,
    kv_heads: usize,
    dimension: usize,
    softmax_scale: f32,
) -> rk_npu_attention_f16_config {
    let mut config = rk_npu_attention_f16_config::default();
    unsafe { rk_npu_attention_f16_config_init(&mut config) };
    config.query_heads = query_heads as i32;
    config.kv_heads = kv_heads as i32;
    config.head_dim = dimension as i32;
    config.softmax_scale = softmax_scale;
    // Keep the public Attention call size independent of the Linear batch.
    config.max_query_rows = options.chunk.min(128) as i32;
    config.max_capacity = options.context.div_ceil(32) as i32 * 32;
    config.initial_capacity = config.max_capacity.min(128);
    config.core_mask = options.npu_mask;
    config.kv_tile = options.attention_tile as i32;
    config.flags = RK_NPU_ATTENTION_F16_STABLE_SOFTMAX;
    config
}

impl Backend {
    pub fn attention_cache_bytes(
        options: &ModelOptions,
        query_heads: usize,
        kv_heads: usize,
        dimension: usize,
        softmax_scale: f32,
    ) -> Result<u64> {
        let config = attention_config(options, query_heads, kv_heads, dimension, softmax_scale);
        let mut sizes = rk_npu_attention_f16_sizes::default();
        check(
            unsafe {
                rk_npu_attention_f16_query(
                    &config,
                    config.max_capacity,
                    1,
                    config.max_capacity,
                    &mut sizes,
                )
            },
            "query native KV memory",
        )?;
        Ok(sizes.key_bytes + sizes.value_bytes + sizes.tail_bytes)
    }

    pub fn new(options: &ModelOptions, domain_id: u32) -> Result<Self> {
        if let Ok(value) = std::env::var("RK_NPU_W4A8_K_TILE_MULTIPLIER") {
            ensure!(
                value.parse::<f64>().ok() == Some(1.0),
                "certified W4 tiles require RK_NPU_W4A8_K_TILE_MULTIPLIER=1"
            );
        }
        if let Ok(value) = std::env::var("RK_NPU_W8A8_NPU_DEQUANT") {
            ensure!(
                value == "0",
                "RK_NPU_W8A8_NPU_DEQUANT must be 0 to preserve FP32 scales"
            );
        }
        ensure!(
            domain_id < RK_NPU_IOMMU_DOMAIN_COUNT,
            "IOMMU domain must be within 0..15"
        );
        let context = Handle::new(
            unsafe { rk_npu_open(ptr::null()) },
            rk_npu_close,
            "RK3588 NPU DRM context",
        )?;
        let domain = Handle::new(
            unsafe { rk_npu_iommu_domain_create(context.raw, domain_id) },
            rk_npu_iommu_domain_free,
            "NPU IOMMU domain",
        )
        .with_context(|| {
            format!(
                "cannot create NPU IOMMU domain {domain_id}; nonzero domains require RKNPU driver >= 0.9.0 with IOMMU enabled"
            )
        })?;
        if options.tune_search && !options.tune_cache.as_os_str().is_empty() {
            std::fs::create_dir_all(&options.tune_cache)?;
        }
        Ok(Self(Rc::new(RefCell::new(BackendInner {
            work: BTreeMap::new(),
            moe_work: BTreeMap::new(),
            attention: None,
            material: BTreeMap::new(),
            w8_partitions: BTreeMap::new(),
            w8_strategies: BTreeMap::new(),
            w4_strategies: BTreeMap::new(),
            options: options.clone(),
            domain_id,
            cache_hits: 0,
            cache_misses: 0,
            tune_calls: 0,
            fallbacks: BTreeSet::new(),
            fallback_calls: 0,
            weight_reload_calls: 0,
            loaded_w4: 0,
            loaded_w8: 0,
            weight_bytes: 0,
            domain,
            context,
        }))))
    }

    pub fn load(&self, mut matrix: QuantMatrix, reload: Option<Reload>) -> Result<Linear> {
        matrix.validate()?;
        let mut b = self.0.borrow_mut();
        let mut w4 = None;
        let mut w8 = None;
        let mut profile = None;
        let kt;
        if matrix.bits == 4 {
            let cfg = rk_npu_i4_weight_config {
                K: matrix.k as i32,
                N: matrix.n as i32,
                k_tile: 0,
            };
            let handle = Handle::new(
                unsafe {
                    rk_npu_w4a8_weights_create(
                        b.domain.raw,
                        &cfg,
                        matrix.codes.as_ptr(),
                        matrix.scales.as_ptr(),
                    )
                },
                rk_npu_w4a8_weights_free,
                "certified W4 weights",
            )?;
            let mut info = rk_npu_w4a8_weight_info::default();
            check(
                unsafe { rk_npu_w4a8_weights_query(handle.raw, &mut info) },
                "W4 weight query",
            )?;
            ensure!(
                info.bound_relaxed == 0 && info.multiplier == 1.0,
                "uncertified W4 weights"
            );
            kt = info.config.k_tile as usize;
            let mut bytes = 0;
            check(
                unsafe { rk_npu_i4_weights_memory_query(&info.config, &mut bytes) },
                "W4 weight memory",
            )?;
            b.weight_bytes += bytes + matrix.n as u64 * 8;
            b.loaded_w4 += 1;
            profile = Some(Handle::new(
                unsafe { rk_npu_w4a8_tuning_profile_from_weights(handle.raw) },
                rk_npu_w4a8_tuning_profile_free,
                "W4 tuning profile",
            )?);
            w4 = Some(handle);
        } else {
            kt = b.w8_partition(matrix.k, matrix.n)?;
            let cfg = rk_npu_matmul_weight_config {
                K: matrix.k as i32,
                N: matrix.n as i32,
                k_tile: kt as i32,
            };
            let handle = Handle::new(
                unsafe {
                    rk_npu_f32i8f32_weights_create(
                        b.domain.raw,
                        &cfg,
                        matrix.codes.as_ptr(),
                        matrix.scales.as_ptr(),
                    )
                },
                rk_npu_f32i8f32_weights_free,
                "W8 weights",
            )?;
            let mut info = rk_npu_matmul_weight_requirements::default();
            check(
                unsafe { rk_npu_f32i8f32_weights_memory_query(&cfg, &mut info) },
                "W8 weight memory",
            )?;
            b.weight_bytes += info.bytes;
            b.loaded_w8 += 1;
            w8 = Some(handle);
        }
        let key = (matrix.bits, matrix.k, matrix.n, kt);
        if !b.material.contains_key(&key) {
            if matrix.bits == 8 || !b.options.tune_search {
                matrix.codes = Vec::new();
            }
            let reload = if b.options.tune_search { reload } else { None };
            b.material.insert(
                key,
                Material {
                    matrix,
                    reload,
                    profile,
                },
            );
        }
        b.get_work(key, 1)?;
        let material = b.material.get_mut(&key).unwrap();
        if material.reload.is_some() {
            material.matrix.codes = Vec::new();
        }
        Ok(Linear {
            w4,
            w8,
            key,
            owner: self.clone(),
        })
    }

    pub fn attention(
        &self,
        query_heads: usize,
        kv_heads: usize,
        dimension: usize,
        softmax_scale: f32,
    ) -> Result<Attention> {
        self.attention_masked(query_heads, kv_heads, dimension, softmax_scale, false)
    }
    /// Temporary K/V and query scratch sized to one complete input sequence.
    pub fn full_attention(
        &self,
        heads: usize,
        dimension: usize,
        rows: usize,
    ) -> Result<FullAttention> {
        ensure!(
            rows > 0 && rows <= (i32::MAX as usize / 32) * 32,
            "image Attention rows exceed the native integer range: {rows}"
        );
        let b = self.0.borrow();
        let mut config = attention_config(
            &b.options,
            heads,
            heads,
            dimension,
            (dimension as f32).sqrt().recip(),
        );
        config.mask_mode = RK_NPU_ATTENTION_F16_NO_MASK;
        config.max_capacity = rows.div_ceil(32) as i32 * 32;
        config.initial_capacity = config.max_capacity;
        let work = AttentionWork::new(b.domain.raw, config, 0)?;
        let mut raw = ptr::null_mut();
        check(
            unsafe { rk_npu_attention_f16_cache_create(b.domain.raw, &config, &mut raw) },
            "allocate image K/V buffers",
        )?;
        Ok(FullAttention {
            kv: Handle::new(raw, rk_npu_attention_f16_cache_free, "image K/V buffers")?,
            work,
            owner: self.clone(),
        })
    }
    /// Bidirectional grouped-query attention for block diffusion drafting.
    pub fn noncausal_gqa(
        &self,
        query_heads: usize,
        kv_heads: usize,
        dimension: usize,
    ) -> Result<Attention> {
        self.attention_masked(
            query_heads,
            kv_heads,
            dimension,
            (dimension as f32).sqrt().recip(),
            true,
        )
    }
    fn attention_masked(
        &self,
        query_heads: usize,
        kv_heads: usize,
        dimension: usize,
        softmax_scale: f32,
        noncausal: bool,
    ) -> Result<Attention> {
        let mut b = self.0.borrow_mut();
        ensure!(
            kv_heads > 0
                && kv_heads <= 32
                && query_heads >= kv_heads
                && query_heads.is_multiple_of(kv_heads)
                && dimension > 0
                && dimension.is_multiple_of(32)
                && softmax_scale.is_finite()
                && softmax_scale > 0.0,
            "native GQA requires Hq divisible by Hkv, Hkv within 1..32, D divisible by 32 and a positive softmax scale"
        );
        if b.attention.is_none() {
            let mut config =
                attention_config(&b.options, query_heads, kv_heads, dimension, softmax_scale);
            if noncausal {
                config.mask_mode = RK_NPU_ATTENTION_F16_NO_MASK;
            }
            b.attention = Some(AttentionWork::new(b.domain.raw, config, b.options.chunk)?);
        }
        let cfg = &b.attention.as_ref().unwrap().config;
        ensure!(
            cfg.query_heads as usize == query_heads
                && cfg.kv_heads as usize == kv_heads
                && cfg.head_dim as usize == dimension
                && cfg.softmax_scale == softmax_scale
                && (cfg.mask_mode == RK_NPU_ATTENTION_F16_NO_MASK) == noncausal,
            "attention workspace geometry mismatch"
        );
        let mut raw = ptr::null_mut();
        check(
            unsafe { rk_npu_attention_f16_cache_create(b.domain.raw, cfg, &mut raw) },
            "create native KV cache",
        )?;
        Ok(Attention {
            cache: Handle::new(raw, rk_npu_attention_f16_cache_free, "KV cache")?,
            query_heads,
            kv_heads,
            dimension,
            owner: self.clone(),
        })
    }

    pub fn prepare(&self, rows: usize) -> Result<()> {
        let m = bucket(rows)?;
        let mut b = self.0.borrow_mut();
        let keys: Vec<_> = b.material.keys().copied().collect();
        for (index, key) in keys.iter().copied().enumerate() {
            if b.options.tune_search && !b.work.contains_key(&(key, m)) {
                eprintln!(
                    "Tuning shapes {}/{}: domain={} W{} M={m} K={} N={} Ktile={}",
                    index + 1,
                    keys.len(),
                    b.domain_id,
                    key.0,
                    key.1,
                    key.2,
                    key.3
                );
            }
            b.get_work(key, m)?;
        }
        Ok(())
    }
    pub fn retain_workspaces(&self, rows: &[usize]) -> Result<()> {
        let keep: Vec<_> = rows.iter().map(|&r| bucket(r)).collect::<Result<_>>()?;
        self.0
            .borrow_mut()
            .work
            .retain(|(_, m), _| keep.contains(m));
        Ok(())
    }
    pub fn weight_bytes(&self) -> u64 {
        self.0.borrow().weight_bytes
    }
    pub fn tune_operations(&self) -> usize {
        let backend = self.0.borrow();
        backend.cache_hits + backend.cache_misses
    }
    pub fn stats(&self) -> Value {
        let b = self.0.borrow();
        let workspace: u64 = b
            .work
            .values()
            .map(|w| w.bytes + (w.input.capacity() + w.output.capacity()) as u64 * 4)
            .sum::<u64>()
            + b.moe_work.values().map(|w| w.bytes).sum::<u64>();
        let material: usize = b
            .material
            .values()
            .map(|m| m.matrix.codes.capacity() + m.matrix.scales.capacity() * 4)
            .sum();
        let staging = b.attention.as_ref().map_or(0, |w| {
            (w.q.capacity() + w.k.capacity() + w.v.capacity()) * 2 + w.output.capacity() * 4
        });
        json!({"backend": "open-rknpu2", "loaded_w4": b.loaded_w4, "loaded_w8": b.loaded_w8,
            "weight_payload_bytes": b.weight_bytes, "workspace_payload_bytes": workspace,
            "tuning_material_bytes": material, "workspace_count": b.work.len(),
            "attention_cpu_staging_bytes": staging, "tune_cache_hits": b.cache_hits,
            "tune_cache_misses": b.cache_misses, "npu_core_mask": b.options.npu_mask,
            "tune_calls": b.tune_calls, "strategy_fallbacks": b.fallbacks.len(),
            "strategy_fallback_calls": b.fallback_calls, "weight_reload_calls": b.weight_reload_calls,
            "weight_profile_creations": b.loaded_w4,
            "cpu_core_mask": b.options.cpu_mask, "certified_w4_tiles": true,
            "attention_geometry": b.attention.as_ref().map(|a| json!({
                "query_heads": a.config.query_heads, "kv_heads": a.config.kv_heads,
                "head_dim": a.config.head_dim, "softmax_scale": a.config.softmax_scale,
                "core_mask": a.active_core_mask,
            })),
            "weight_geometries": b.material.len(), "iommu_domain_id": b.domain_id})
    }
}

impl BackendInner {
    fn cache_path(&self, key: Key, m: usize) -> Result<Option<CString>> {
        if self.options.tune_cache.as_os_str().is_empty() {
            return Ok(None);
        }
        let (bits, k, n, kt) = key;
        let prefix = if bits == 4 {
            format!("d{}-", self.domain_id)
        } else {
            String::new()
        };
        let path = self
            .options
            .tune_cache
            .join(format!("{prefix}w{bits}-m{m}-k{k}-n{n}-kt{kt}.tune"));
        Ok(Some(CString::new(path.as_os_str().as_encoded_bytes())?))
    }
    fn w8_partition(&mut self, k: usize, n: usize) -> Result<usize> {
        if let Some(&tile) = self.w8_partitions.get(&(k, n)) {
            return Ok(tile);
        }
        let waves = k.div_ceil(4096);
        let balanced = |count: usize| k.div_ceil(count).div_ceil(32).saturating_mul(32).min(k);
        let prefill_rows = bucket(self.options.chunk)? as i32;
        let permitted = |tile: usize| unsafe {
            rk_npu_matmul_k_partition_query(prefill_rows, k as i32, tile as i32) == 0
        };
        let mut kt = k.min(4096);
        if !permitted(kt) {
            kt = k.min(2048);
        }
        if self.options.tune {
            let mut candidates = vec![k.min(4096), balanced(waves), k.min(2048)];
            if waves > 1 {
                candidates.push(balanced(waves + 1));
            }
            candidates.sort_unstable();
            candidates.dedup();
            let mut best = f64::INFINITY;
            for candidate in candidates {
                if !permitted(candidate) {
                    continue;
                }
                let key = (8, k, n, candidate);
                let decode = self.w8_strategy(key, 1)?;
                let prefill = self.w8_strategy(key, bucket(self.options.chunk)?)?;
                if decode.total_us <= 0.0 || prefill.total_us <= 0.0 {
                    continue;
                }
                let cost = 64.0 * decode.total_us
                    + 128usize.div_ceil(self.options.chunk) as f64 * prefill.total_us;
                if cost < best {
                    best = cost;
                    kt = candidate;
                }
            }
        }
        ensure!(
            permitted(kt),
            "no legal W8 K partition for K={k}, M={prefill_rows}"
        );
        self.w8_partitions.insert((k, n), kt);
        Ok(kt)
    }
    fn w8_strategy(&mut self, key: Key, m: usize) -> Result<rk_npu_matmul_strategy> {
        if let Some(&strategy) = self.w8_strategies.get(&(key, m)) {
            return Ok(strategy);
        }
        let (_, k, n, kt) = key;
        let mut fastest = rk_npu_matmul_strategy::default();
        let mut found = false;
        if self.options.tune {
            let mut cfg = rk_npu_matmul_autotune_config::default();
            unsafe { rk_npu_matmul_autotune_config_init(&mut cfg, m as i32, n as i32, k as i32) };
            cfg.required_k_tile = kt as i32;
            cfg.allowed_npu_core_mask = self.options.npu_mask;
            cfg.allowed_cpu_core_mask = self.options.cpu_mask;
            cfg.warmup = 1;
            cfg.loops = 3;
            cfg.repeats = 3;
            let mut stable = rk_npu_matmul_strategy::default();
            let path = self.cache_path(key, m)?;
            let rc = if let Some(path) = &path {
                unsafe {
                    rk_npu_matmul_autotune_cache_load(
                        self.context.raw,
                        path.as_ptr(),
                        RK_NPU_MATMUL_F32I8F32_DYNAMIC,
                        &cfg,
                        &mut fastest,
                        &mut stable,
                    )
                }
            } else {
                RK_NPU_ERR_CACHE_MISS
            };
            if rc == 0 {
                self.cache_hits += 1;
                found = true;
            } else {
                check_cache_miss(rc, "W8 cache load")?;
                self.cache_misses += 1;
            }
            if !found && self.options.tune_search {
                cfg.verbose = 1;
                let mut hit = 0;
                let start = std::time::Instant::now();
                eprintln!("Tuning W8: M={m} K={k} N={n} Ktile={kt}, cache miss");
                self.tune_calls += 1;
                check(
                    unsafe {
                        rk_npu_matmul_autotune_cached(
                            self.context.raw,
                            &cfg,
                            RK_NPU_MATMUL_F32I8F32_DYNAMIC,
                            path.as_ref().map_or(ptr::null(), |p| p.as_ptr()),
                            0,
                            &mut fastest,
                            &mut stable,
                            &mut hit,
                        )
                    },
                    "W8 tune",
                )?;
                eprintln!(
                    "Tuning W8 done: {:.2}s, best={:.1}us",
                    start.elapsed().as_secs_f64(),
                    fastest.total_us
                );
                found = true;
            } else if self.options.tune_search {
                eprintln!(
                    "Tuning W8: M={m} K={k} N={n} Ktile={kt}, cache hit ({:.1}us)",
                    fastest.total_us
                );
            }
        }
        if !found {
            self.fallbacks.insert((key, m));
            fastest.op_kind = RK_NPU_MATMUL_F32I8F32_DYNAMIC;
            fastest.M = m as i32;
            fastest.N = n as i32;
            fastest.K = k as i32;
            fastest.k_tile = kt as i32;
            let active_mask = policy::w8_active_mask(self.options.npu_mask, n);
            let cores = active_mask.count_ones() as usize;
            fastest.n_tile = (n.div_ceil(32) / cores * 32).clamp(32, 1024) as i32;
            fastest.a_layout = RK_NPU_MATMUL_A_LAYOUT_NORMAL;
            fastest.c_layout = RK_NPU_MATMUL_C_LAYOUT_NATIVE;
            fastest.wave_count = k.div_ceil(kt) as i32;
            fastest.n_groups = (n.div_ceil(32) * 32).div_ceil(fastest.n_tile as usize) as i32;
            fastest.npu_core_mask = active_mask;
            fastest.cpu_core_mask = self.options.cpu_mask;
            fastest.cpu_threads = self.options.threads as i32;
        }
        self.w8_strategies.insert((key, m), fastest);
        Ok(fastest)
    }
    fn w4_strategy(&mut self, key: Key, m: usize) -> Result<rk_npu_w4a8_strategy_ex> {
        if let Some(&strategy) = self.w4_strategies.get(&(key, m)) {
            return Ok(strategy);
        }
        let (_, k, n, kt) = key;
        let mut fastest = rk_npu_w4a8_strategy_ex::default();
        let mut found = false;
        if self.options.tune {
            let material = self
                .material
                .get(&key)
                .context("missing W4 tuning material")?;
            let mut cfg = rk_npu_w4a8_autotune_config::default();
            unsafe { rk_npu_w4a8_autotune_config_init(&mut cfg, m as i32, n as i32, k as i32) };
            cfg.required_k_tile = kt as i32;
            cfg.k_tile_multiplier = 1.0;
            cfg.allowed_npu_core_mask = self.options.npu_mask;
            cfg.allowed_cpu_core_mask = self.options.cpu_mask;
            cfg.warmup = 1;
            cfg.loops = 3;
            cfg.repeats = 3;
            let mut stable = rk_npu_w4a8_strategy_ex::default();
            let path = self.cache_path(key, m)?;
            let rc = if let Some(path) = &path {
                unsafe {
                    rk_npu_w4a8_autotune_cache_load_profile_ex(
                        self.context.raw,
                        &cfg,
                        material.profile.as_ref().context("missing W4 profile")?.raw,
                        path.as_ptr(),
                        &mut fastest,
                        &mut stable,
                    )
                }
            } else {
                RK_NPU_ERR_CACHE_MISS
            };
            if rc == 0 {
                self.cache_hits += 1;
                found = true;
            } else {
                check_cache_miss(rc, "W4 cache load")?;
                self.cache_misses += 1;
            }
            if !found && self.options.tune_search {
                let reloaded;
                let source = if material.matrix.codes.is_empty() {
                    self.weight_reload_calls += 1;
                    reloaded = material
                        .reload
                        .as_ref()
                        .context("missing W4 tuning weight reload")?(
                    )?;
                    ensure!(
                        reloaded.bits == 4
                            && reloaded.k == k
                            && reloaded.n == n
                            && reloaded.codes.len() == k * n
                            && reloaded.scales.len() == n,
                        "tuning weight reload geometry mismatch"
                    );
                    &reloaded
                } else {
                    &material.matrix
                };
                cfg.verbose = 2;
                let mut hit = 0;
                let start = std::time::Instant::now();
                eprintln!(
                    "Tuning W4: domain={} M={m} K={k} N={n} Ktile={kt}, cache miss",
                    self.domain_id
                );
                self.tune_calls += 1;
                check(
                    unsafe {
                        rk_npu_w4a8_autotune_cached_ex(
                            self.context.raw,
                            &cfg,
                            source.codes.as_ptr(),
                            source.scales.as_ptr(),
                            path.as_ref().map_or(ptr::null(), |p| p.as_ptr()),
                            0,
                            &mut fastest,
                            &mut stable,
                            &mut hit,
                        )
                    },
                    "W4 tune",
                )?;
                eprintln!(
                    "Tuning W4 done: {:.2}s, best={:.1}us",
                    start.elapsed().as_secs_f64(),
                    fastest.base.total_us
                );
                found = true;
            } else if self.options.tune_search {
                eprintln!(
                    "Tuning W4: domain={} M={m} K={k} N={n} Ktile={kt}, cache hit ({:.1}us)",
                    self.domain_id, fastest.base.total_us
                );
            }
        }
        if !found {
            self.fallbacks.insert((key, m));
            unsafe {
                rk_npu_w4a8_config_init(&mut fastest.base.config, m as i32, n as i32, k as i32);
                rk_npu_w4a8_options_init(&mut fastest.execution);
            }
            fastest.base.config.k_tile = kt as i32;
            fastest.base.config.m_tile = m.min(64) as i32;
            fastest.base.config.n_tile = (n.div_ceil(64) * 64).min(1024) as i32;
            fastest.base.config.npu_core_mask = self.options.npu_mask;
            fastest.base.config.cpu_threads = self.options.threads as i32;
            fastest.base.config.pipeline = RK_NPU_W4A8_PIPELINED as i32;
            fastest.base.cpu_core_mask = self.options.cpu_mask;
        }
        self.w4_strategies.insert((key, m), fastest);
        Ok(fastest)
    }
    fn get_work(&mut self, key: Key, m: usize) -> Result<&mut Work> {
        let wk = (key, m);
        if !self.work.contains_key(&wk) {
            let (bits, k, n, _) = key;
            let mut work = Work {
                w4: None,
                w8: None,
                input: vec![0.0; m * k],
                output: vec![0.0; m * n],
                bytes: 0,
                w4_reduce: RK_NPU_W4A8_REDUCE_CPU,
            };
            if bits == 4 {
                let strategy = self.w4_strategy(key, m)?;
                let mut info = rk_npu_w4a8_memory_info_ex::default();
                check(
                    unsafe {
                        rk_npu_w4a8_memory_query_ex(
                            &strategy.base.config,
                            &strategy.execution,
                            &mut info,
                        )
                    },
                    "W4 memory query",
                )?;
                work.bytes = info.workspace_bytes;
                work.w4_reduce = info.effective_reduce_backend;
                work.w4 = Some(Handle::new(
                    unsafe {
                        rk_npu_w4a8_workspace_create_ex(
                            self.domain.raw,
                            &strategy.base.config,
                            &strategy.execution,
                        )
                    },
                    rk_npu_w4a8_workspace_free,
                    "W4 workspace",
                )?);
            } else {
                let strategy = self.w8_strategy(key, m)?;
                let mut info = rk_npu_matmul_workspace_requirements::default();
                check(
                    unsafe { rk_npu_matmul_workspace_memory_query(&strategy, &mut info) },
                    "W8 memory query",
                )?;
                work.bytes = info.bytes;
                work.w8 = Some(Handle::new(
                    unsafe { rk_npu_matmul_workspace_create(self.domain.raw, &strategy) },
                    rk_npu_matmul_workspace_free,
                    "W8 workspace",
                )?);
            }
            self.work.insert(wk, work);
        }
        Ok(self.work.get_mut(&wk).unwrap())
    }
}

impl Linear {
    pub fn k(&self) -> usize {
        self.key.1
    }
    pub fn n(&self) -> usize {
        self.key.2
    }
    pub fn run(&self, input: &[f32], output: &mut [f32], rows: usize) -> Result<()> {
        self.run_npu(input, output, rows, false)
    }
    pub fn run_swiglu(&self, input: &[f32], output: &mut [f32], rows: usize) -> Result<bool> {
        let mut b = self.owner.0.borrow_mut();
        if self.w4.is_none() || rows < 32 || b.options.threads < 2 || !self.n().is_multiple_of(64) {
            return Ok(false);
        }
        let m = bucket(rows)?;
        let reduce = b.get_work(self.key, m)?.w4_reduce;
        if reduce != RK_NPU_W4A8_REDUCE_CPU
            || b.w4_strategies[&(self.key, m)].base.config.cpu_threads < 2
        {
            return Ok(false);
        }
        drop(b);
        self.run_npu(input, output, rows, true)?;
        Ok(true)
    }
    fn run_npu(&self, input: &[f32], output: &mut [f32], rows: usize, swiglu: bool) -> Result<()> {
        let m = bucket(rows)?;
        let width = if swiglu { self.n() / 2 } else { self.n() };
        ensure!(
            input.len() >= rows * self.k() && output.len() >= rows * width,
            "linear input/output shape mismatch"
        );
        let mut b = self.owner.0.borrow_mut();
        let work = b.get_work(self.key, m)?;
        let (a, c) = if m != rows {
            work.input[..rows * self.k()].copy_from_slice(&input[..rows * self.k()]);
            work.input[rows * self.k()..].fill(0.0);
            (work.input.as_ptr(), work.output.as_mut_ptr())
        } else {
            (input.as_ptr(), output.as_mut_ptr())
        };
        if let Some(weights) = &self.w4 {
            let raw = work.w4.as_ref().unwrap().raw;
            let rc = unsafe {
                if swiglu {
                    rk_npu_w4a8_run_swiglu_f32(raw, weights.raw, a, c, ptr::null_mut())
                } else {
                    rk_npu_w4a8_run_f32(raw, weights.raw, a, c, ptr::null_mut())
                }
            };
            check(rc, if swiglu { "W4 SwiGLU run" } else { "W4 run" })?;
        } else {
            check(
                unsafe {
                    rk_npu_f32i8f32_run_dynamic(
                        work.w8.as_ref().unwrap().raw,
                        self.w8.as_ref().unwrap().raw,
                        a,
                        c,
                    )
                },
                "W8 run",
            )?;
        }
        if m != rows {
            output[..rows * width].copy_from_slice(&work.output[..rows * width]);
        }
        if b.fallbacks.contains(&(self.key, m)) {
            b.fallback_calls += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_weight_buffers_must_match_declared_dimensions() {
        for bits in [4, 8] {
            let mut matrix = QuantMatrix {
                bits,
                k: 64,
                n: 128,
                codes: vec![0; 64 * 128],
                scales: vec![1.; 128],
            };
            assert!(matrix.validate().is_ok());
            matrix.codes.pop();
            assert!(matrix.validate().is_err());
            matrix.codes.push(0);
            matrix.scales.clear();
            assert!(matrix.validate().is_err());
            matrix.scales.resize(129, 1.);
            assert!(matrix.validate().is_err());
            matrix.k = usize::MAX;
            assert!(matrix.validate().is_err());
        }
    }

    fn options(cache: &std::path::Path) -> Result<ModelOptions> {
        let mut options = ModelOptions {
            path: Default::default(),
            context: 128,
            chunk: 64,
            threads: 2,
            npu_mask: 1,
            cpu_mask: 0,
            tune: false,
            tune_search: false,
            tune_cache: cache.into(),
            attention_tile: 256,
            vision_dtype: Default::default(),
            experts: Default::default(),
            speculation: Default::default(),
        };
        configure_cpu(&mut options)?;
        Ok(options)
    }

    fn oracle(matrix: &QuantMatrix, input: &[f32], rows: usize) -> Vec<f32> {
        let mut output = vec![0.0; rows * matrix.n];
        for row in 0..rows {
            let a = &input[row * matrix.k..(row + 1) * matrix.k];
            let maximum = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let scale = if maximum == 0.0 {
                1.0
            } else {
                (maximum / 127.0).max(f32::MIN_POSITIVE)
            };
            let quantized: Vec<_> = a
                .iter()
                .map(|x| (x * (1.0 / scale)).round().clamp(-127.0, 127.0) as i32)
                .collect();
            for n in 0..matrix.n {
                let sum: i32 = quantized
                    .iter()
                    .enumerate()
                    .map(|(k, a)| a * matrix.codes[k * matrix.n + n] as i32)
                    .sum();
                output[row * matrix.n + n] = (sum as f32 * scale) * matrix.scales[n];
            }
        }
        output
    }

    fn compare(actual: &[f32], expected: &[f32], tolerance: f32) {
        for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && (a - e).abs() <= tolerance * e.abs().max(1.0),
                "element {index}: actual={a} expected={e}"
            );
        }
    }

    #[test]
    fn w8_partition_policy_is_owned_by_the_operator() {
        for (rows, k, tile) in [
            (1, 8192, 2752),
            (4, 8192, 2752),
            (64, 8192, 2048),
            (64, 8192, 3072),
            (64, 8192, 4096),
            (64, 6144, 3072),
            (64, 2560, 2560),
            (64, 5120, 4096),
        ] {
            assert_eq!(unsafe { rk_npu_matmul_k_partition_query(rows, k, tile) }, 0);
        }
        for (rows, k, tile) in [
            (64, 8192, 2752),
            (5, 8192, 2752),
            (64, 3840, 3840),
            (64, 3840, 4096),
            (0, 8192, 4096),
            (64, 8192, 0),
        ] {
            assert_eq!(
                unsafe { rk_npu_matmul_k_partition_query(rows, k, tile) },
                RK_NPU_ERR_PARAM
            );
        }
    }

    #[test]
    #[ignore = "requires exclusive RK3588 NPU access"]
    fn quantized_linear_padding_and_swiglu() -> Result<()> {
        let _lock = crate::validation::BoardLock::acquire()?;
        let cache = tempfile::tempdir()?;
        let mut options = options(cache.path())?;
        options.tune = true;
        options.chunk = 256;
        options.context = 256;
        let backend = Backend::new(&options, 15)?;
        for bits in [4, 8] {
            let matrix = QuantMatrix {
                bits,
                k: 256,
                n: 256,
                codes: (0..256 * 256)
                    .map(|i| {
                        ((i * 13 + i / 256 * 7) % if bits == 4 { 16 } else { 255 }) as i16
                            - if bits == 4 { 8 } else { 127 }
                    })
                    .map(|x| x as i8)
                    .collect(),
                scales: (0..256).map(|n| 0.002 + (n % 11) as f32 * 0.0001).collect(),
            };
            let input: Vec<_> = (0..256 * 256)
                .map(|i| ((i * 17 + i / 256 * 3) % 127) as f32 / 63.0 - 1.0)
                .collect();
            let reference = oracle(&matrix, &input, 256);
            let linear = backend.load(
                matrix,
                Some(Box::new(|| bail!("cache-only path reloaded weights"))),
            )?;
            for rows in [
                1, 2, 3, 5, 9, 17, 32, 33, 65, 127, 128, 129, 191, 192, 193, 255, 256,
            ] {
                let mut output = vec![0.0; rows * 256];
                linear.run(&input, &mut output, rows)?;
                compare(&output, &reference[..rows * 256], 2e-5);
                if bits == 4 && rows >= 32 {
                    let mut middle = vec![0.0; rows * 128];
                    assert!(linear.run_swiglu(&input, &mut middle, rows)?);
                    let expected: Vec<_> = (0..rows)
                        .flat_map(|r| (0..128).map(move |n| (r, n)))
                        .map(|(r, n)| {
                            let gate = reference[r * 256 + n];
                            gate / (1.0 + (-gate).exp()) * reference[r * 256 + 128 + n]
                        })
                        .collect();
                    compare(&middle, &expected, 3e-5);
                }
            }
        }
        backend.retain_workspaces(&[1, 64])?;
        assert_eq!(backend.stats()["workspace_count"], 4);
        let lookups = backend.tune_operations();
        backend.prepare(3)?;
        backend.retain_workspaces(&[1, 64])?;
        backend.prepare(3)?;
        assert_eq!(backend.tune_operations(), lookups);
        assert_eq!(backend.stats()["tune_calls"], 0);
        assert_eq!(backend.stats()["weight_reload_calls"], 0);
        assert_eq!(std::fs::read_dir(cache.path())?.count(), 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive RK3588 NPU access"]
    fn native_gqa_append_and_layout() -> Result<()> {
        let _lock = crate::validation::BoardLock::acquire()?;
        let cache = tempfile::tempdir()?;
        let mut options = options(cache.path())?;
        options.context = 512;
        let backend = Backend::new(&options, 15)?;
        let attention = backend.attention(4, 1, 128, 1.0 / 128.0f32.sqrt())?;
        let q: Vec<_> = (0..4 * 4 * 128)
            .map(|i| ((i * 19) % 101) as f32 * 0.006 - 0.3)
            .collect();
        let k: Vec<_> = (0..4 * 128)
            .map(|i| ((i * 11) % 83) as f32 * 0.007 - 0.28)
            .collect();
        let v: Vec<_> = (0..4 * 128)
            .map(|i| ((i * 7) % 73) as f32 * 0.009 - 0.31)
            .collect();
        let half = |x: f32| f16::from_f32(x).to_f32();
        let mut expected = vec![0.0; q.len()];
        for r in 0..4 {
            for h in 0..4 {
                let mut scores: Vec<f32> = (0..=r)
                    .map(|j| {
                        (0..128)
                            .map(|d| {
                                half(q[(r * 4 + h) * 128 + d]) / 128.0f32.sqrt()
                                    * half(k[j * 128 + d])
                            })
                            .sum()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for score in &mut scores {
                    *score = (*score - max).exp();
                }
                let denominator: f32 = scores.iter().sum();
                for d in 0..128 {
                    expected[(r * 4 + h) * 128 + d] = scores
                        .iter()
                        .enumerate()
                        .map(|(j, s)| s / denominator * half(v[j * 128 + d]))
                        .sum();
                }
            }
        }
        let mut output = vec![0.0; q.len()];
        attention.run(
            &q[..3 * 4 * 128],
            &k[..3 * 128],
            &v[..3 * 128],
            &mut output[..3 * 4 * 128],
            3,
            0,
        )?;
        attention.run(
            &q[3 * 4 * 128..],
            &k[3 * 128..],
            &v[3 * 128..],
            &mut output[3 * 4 * 128..],
            1,
            3,
        )?;
        compare(&output, &expected, 0.02);
        ensure!(attention.bytes()? > 0, "empty KV allocation");
        let initial_bytes = attention.bytes()?;
        attention.reserve(512)?;
        assert!(attention.bytes()? > initial_bytes);

        let mut before = rk_npu_attention_f16_cache_info::default();
        check(
            unsafe { rk_npu_attention_f16_cache_get_info(attention.cache.raw, &mut before) },
            "KV info",
        )?;
        attention.truncate(2)?;
        attention.truncate(2)?;
        assert!(attention.truncate(3).is_err());
        assert_eq!(
            unsafe { rk_npu_attention_f16_cache_truncate(attention.cache.raw, -1) },
            RK_NPU_ERR_PARAM
        );
        let mut after = rk_npu_attention_f16_cache_info::default();
        check(
            unsafe { rk_npu_attention_f16_cache_get_info(attention.cache.raw, &mut after) },
            "KV info",
        )?;
        assert_eq!(after.length, 2);
        assert_eq!(after.capacity, before.capacity);
        assert_eq!(after.generation, before.generation);
        assert_eq!(
            (after.key_bytes, after.value_bytes, after.tail_bytes),
            (before.key_bytes, before.value_bytes, before.tail_bytes)
        );
        {
            let mut b = backend.0.borrow_mut();
            let context = b.context.raw;
            let workspace = b.attention.as_mut().unwrap().workspace.raw;
            let query: Vec<_> = q[4 * 128..2 * 4 * 128]
                .iter()
                .map(|&x| f16::from_f32(x).to_bits())
                .collect();
            check(
                unsafe {
                    rk_npu_attention_f16_run(
                        context,
                        workspace,
                        attention.cache.raw,
                        query.as_ptr(),
                        1,
                        1,
                        output.as_mut_ptr(),
                        ptr::null_mut(),
                        ptr::null(),
                    )
                },
                "query retained KV",
            )?;
        }
        compare(&output[..4 * 128], &expected[4 * 128..2 * 4 * 128], 0.02);
        let mut replacement_k = k.clone();
        let mut replacement_v = v.clone();
        for x in &mut replacement_k[2 * 128..] {
            *x = -*x + 0.17;
        }
        for x in &mut replacement_v[2 * 128..] {
            *x = -*x - 0.21;
        }
        let mut reused = vec![0.0; 2 * 4 * 128];
        attention.run(
            &q[2 * 4 * 128..],
            &replacement_k[2 * 128..],
            &replacement_v[2 * 128..],
            &mut reused,
            2,
            2,
        )?;
        let fresh = backend.attention(4, 1, 128, 1.0 / 128.0f32.sqrt())?;
        let mut recomputed = vec![0.0; q.len()];
        fresh.run(
            &q[..2 * 4 * 128],
            &replacement_k[..2 * 128],
            &replacement_v[..2 * 128],
            &mut recomputed[..2 * 4 * 128],
            2,
            0,
        )?;
        fresh.run(
            &q[2 * 4 * 128..],
            &replacement_k[2 * 128..],
            &replacement_v[2 * 128..],
            &mut recomputed[2 * 4 * 128..],
            2,
            2,
        )?;
        compare(&reused, &recomputed[2 * 4 * 128..], 1e-6);
        attention.truncate(0)?;
        attention.run(
            &q[..4 * 128],
            &k[..128],
            &v[..128],
            &mut output[..4 * 128],
            1,
            0,
        )?;
        compare(&output[..4 * 128], &expected[..4 * 128], 0.02);
        attention.reset()?;
        attention.run(
            &q[..4 * 128],
            &k[..128],
            &v[..128],
            &mut output[..4 * 128],
            1,
            0,
        )?;
        compare(&output[..4 * 128], &expected[..4 * 128], 0.02);
        Ok(())
    }

    #[test]
    #[ignore = "requires RK3588 NPU"]
    fn generic_attention_geometries_match_cpu() -> Result<()> {
        let _lock = crate::validation::BoardLock::acquire()?;
        for (qh, kh, d, scale) in [
            (40, 8, 64, 1.0 / 64.0),
            (16, 2, 128, 1.0 / 128.0f32.sqrt()),
            (4, 4, 32, 1.0 / 32.0f32.sqrt()),
            (12, 4, 96, 1.0 / 96.0f32.sqrt()),
            (20, 4, 160, 1.0 / 160.0f32.sqrt()),
            (16, 1, 192, 1.0 / 192.0f32.sqrt()),
            (16, 4, 256, 1.0 / 256.0f32.sqrt()),
        ] {
            let cache = tempfile::tempdir()?;
            let mut opts = options(cache.path())?;
            opts.npu_mask = 7;
            opts.context = 512;
            let backend = Backend::new(&opts, 0)?;
            let attention = backend.attention(qh, kh, d, scale)?;
            let count = 66;
            let q: Vec<_> = (0..count * qh * d)
                .map(|i| ((i * 17 % 257) as f32 - 128.0) * 0.013)
                .collect();
            let k: Vec<_> = (0..count * kh * d)
                .map(|i| ((i * 13 % 251) as f32 - 125.0) * 0.017)
                .collect();
            let v: Vec<_> = (0..count * kh * d)
                .map(|i| ((i * 7 % 241) as f32 - 120.0) * 0.011)
                .collect();
            let half = |x: f32| f16::from_f32(x).to_f32();
            let oracle = |row: usize, keys: &[f32], values: &[f32]| {
                let mut expected = vec![0.0; qh * d];
                for h in 0..qh {
                    let kv = h / (qh / kh);
                    let mut scores: Vec<f32> = (0..=row)
                        .map(|pos| {
                            (0..d)
                                .map(|j| {
                                    half(q[(row * qh + h) * d + j])
                                        * half(keys[(pos * kh + kv) * d + j])
                                        * scale
                                })
                                .sum()
                        })
                        .collect();
                    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    for s in &mut scores {
                        *s = (*s - maximum).exp();
                    }
                    let denominator: f32 = scores.iter().sum();
                    for j in 0..d {
                        expected[h * d + j] = scores
                            .iter()
                            .enumerate()
                            .map(|(pos, s)| s / denominator * half(values[(pos * kh + kv) * d + j]))
                            .sum();
                    }
                }
                expected
            };
            let mut output = vec![0.0; count * qh * d];
            for (position, rows) in [(0, 33), (33, 32), (65, 1)] {
                attention.run(
                    &q[position * qh * d..(position + rows) * qh * d],
                    &k[position * kh * d..(position + rows) * kh * d],
                    &v[position * kh * d..(position + rows) * kh * d],
                    &mut output[position * qh * d..(position + rows) * qh * d],
                    rows,
                    position,
                )?;
            }
            for row in [0, 1, 16, 31, 32, 33, 64, 65] {
                compare(
                    &output[row * qh * d..(row + 1) * qh * d],
                    &oracle(row, &k, &v),
                    0.01,
                );
            }
            attention.reserve(512)?;
            attention.truncate(17)?;
            let mut new_v = v.clone();
            for x in &mut new_v[17 * kh * d..] {
                *x = -*x + 0.4;
            }
            attention.run(
                &q[17 * qh * d..19 * qh * d],
                &k[17 * kh * d..19 * kh * d],
                &new_v[17 * kh * d..19 * kh * d],
                &mut output[..2 * qh * d],
                2,
                17,
            )?;
            compare(&output[qh * d..2 * qh * d], &oracle(18, &k, &new_v), 0.01);
            let mask = backend
                .0
                .borrow()
                .attention
                .as_ref()
                .unwrap()
                .active_core_mask;
            assert_eq!(
                mask,
                match kh {
                    1 => 1,
                    2 => 3,
                    _ => 7,
                }
            );
        }
        Ok(())
    }
}
