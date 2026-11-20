//! Static floating Linear through the public FP16/BF16 MM ABI. Ordinary calls
//! load shape profiles; explicit preparation can search complete host calls.
use super::*;
use crate::progress::TuningSource;
#[path = "floating_gelu.rs"]
mod gelu;
#[cfg(test)]
#[path = "floating_tests.rs"]
mod tests;

pub(super) struct FloatWork {
    untuned: bool,
    gelu: Option<Handle<rk_npu_mm_gelu_workspace>>,
    workspace: Handle<rk_npu_mm_tuned_workspace>,
    strategy: rk_npu_mm_strategy,
}
type WorkKey = (usize, usize, usize);
pub(super) type Workspaces = BTreeMap<WorkKey, FloatWork>;
#[derive(Clone)]
pub struct FloatBackend {
    // Each geometry keeps one changing row shape and, for participating
    // decoders, a separate full prefill shape.
    work: Arc<Mutex<Workspaces>>,
    device: Backend,
    dtype: rk_npu_mm_dtype,
    host_type: rk_npu_mm_host_type,
    source: TuningSource,
    prefill_chunk: Option<usize>,
}
pub struct FloatLinear {
    weight: Handle<rk_npu_mm_tuned_weights>,
    bias: Vec<f32>,
    k: usize,
    n: usize,
    owner: FloatBackend,
}
fn cache_directory(options: &ModelOptions) -> Result<Option<CString>> {
    if options.tune_cache.as_os_str().is_empty() {
        Ok(None)
    } else {
        Ok(Some(CString::new(
            options
                .tune_cache
                .join("floating")
                .to_string_lossy()
                .as_bytes(),
        )?))
    }
}
fn config(
    b: &BackendInner,
    k: usize,
    n: usize,
    rows: usize,
    dtype: rk_npu_mm_dtype,
    host_type: rk_npu_mm_host_type,
) -> Result<rk_npu_mm_autotune_config> {
    ensure!(
        [k, n, rows].iter().all(|&v| v > 0 && v <= 65504),
        "floating Linear dimensions must be within 1..65504"
    );
    let mut c = rk_npu_mm_autotune_config::default();
    unsafe { rk_npu_mm_autotune_config_init(&mut c, rows as i32, n as i32, k as i32) };
    c.dtype = dtype;
    c.host_type = host_type;
    c.with_bias = 1;
    c.allowed_npu_core_mask = b.options.npu_mask;
    c.allowed_cpu_core_mask = b.options.cpu_mask;
    Ok(c)
}
impl FloatBackend {
    pub fn new(device: Backend) -> Self {
        Self::with_dtype(device, RK_NPU_MM_F16)
    }
    /// FP16 auxiliary decoder matrices. Cache misses use short M tiles and
    /// distribute independent output channels across the permitted NPU cores.
    pub fn for_decoder(device: Backend) -> Self {
        let mut backend = Self::new(device);
        backend.source = TuningSource::Dflash;
        backend
    }
    #[cfg(feature = "model-ling3tiny")]
    pub(crate) fn with_prefill_workspace(mut self, rows: usize) -> Self {
        self.prefill_chunk = Some(rows);
        self
    }
    fn work_key(&self, k: usize, n: usize, rows: usize) -> WorkKey {
        (
            k,
            n,
            if self.prefill_chunk == Some(rows) {
                rows
            } else {
                0
            },
        )
    }
    fn prepared<'a>(&self, works: &'a Workspaces, k: usize, n: usize) -> Option<&'a FloatWork> {
        works
            .get(&(k, n, 0))
            .or_else(|| works.get(&(k, n, self.prefill_chunk?)))
    }
    pub(crate) fn select_prefill_workspaces(&self, rows: &[usize]) {
        let mut work = self.work.lock().unwrap();
        work.retain(|_, w| rows.contains(&(w.strategy.M as usize)));
        let b = self.device.0.lock().unwrap();
        let mut idle = b.domain.idle.lock().unwrap();
        work.extend(
            idle[b.lane]
                .floating
                .extract_if(.., |_, w| rows.contains(&(w.strategy.M as usize))),
        );
    }
    pub(crate) fn finish_prefill(&self, rows: &[usize], prefill: Option<usize>) {
        let mut work = self.work.lock().unwrap();
        if let Some(prefill) = prefill.filter(|m| !rows.contains(m)) {
            let b = self.device.0.lock().unwrap();
            b.domain.idle.lock().unwrap()[b.lane]
                .floating
                .extend(work.extract_if(.., |&(_, _, m), _| m == prefill));
        }
        work.retain(|_, w| rows.contains(&(w.strategy.M as usize)));
    }
    #[allow(
        dead_code,
        reason = "The shared floating backend is available without a Qwen decoder."
    )]
    pub(crate) fn for_mtp(device: Backend) -> Self {
        let mut backend = Self::for_decoder(device);
        backend.source = TuningSource::Mtp;
        backend
    }
    fn is_decoder(&self) -> bool {
        matches!(self.source, TuningSource::Mtp | TuningSource::Dflash)
    }
    /// Native BF16 arithmetic and resident BF16 weights, with FP32 host IO.
    pub fn bf16(device: Backend) -> Self {
        Self::with_dtype(device, RK_NPU_MM_BF16)
    }
    /// FP16 activation storage; no full-tensor FP32 conversion at Linear boundaries.
    pub fn fp16_storage(device: Backend) -> Self {
        let mut backend = Self::with_dtype(device, RK_NPU_MM_F16);
        backend.host_type = RK_NPU_MM_HOST_STORAGE;
        backend
    }
    fn with_dtype(device: Backend, dtype: rk_npu_mm_dtype) -> Self {
        Self {
            work: Arc::new(Mutex::new(BTreeMap::new())),
            device,
            dtype,
            host_type: RK_NPU_MM_HOST_F32,
            source: TuningSource::Vision,
            prefill_chunk: None,
        }
    }
    /// Search is controlled by the explicit tune_search option. Ordinary
    /// inference loads profiles and uses the baseline on a cache miss.
    pub fn prepare(&self, k: usize, n: usize, rows: usize) -> Result<()> {
        self.prepare_for_weights(k, n, rows, ptr::null())
    }
    fn prepare_for_weights(
        &self,
        k: usize,
        n: usize,
        rows: usize,
        weight: *const rk_npu_mm_tuned_weights,
    ) -> Result<()> {
        let mut works = self.work.lock().unwrap();
        let key = self.work_key(k, n, rows);
        if works.get(&key).is_some_and(|w| {
            w.strategy.M as usize == rows
                && (weight.is_null()
                    || unsafe { rk_npu_mm_tuned_weights_match(weight, &w.strategy) } == 0)
        }) {
            if works[&key].untuned && !crate::progress::notice_recorded() {
                self.report_untuned(rows, k, n);
            }
            return Ok(());
        }
        let mut b = self.device.0.lock().unwrap();
        let cfg = config(&b, k, n, rows, self.dtype, self.host_type)?;
        let mut strategy = rk_npu_mm_strategy::default();
        let mut found = false;
        if b.options.tune {
            let path = cache_directory(&b.options)?;
            let rc = if let Some(path) = &path {
                unsafe {
                    rk_npu_mm_autotune_cache_load(
                        b.context.raw,
                        &cfg,
                        path.as_ptr(),
                        &mut strategy,
                        ptr::null_mut(),
                    )
                }
            } else {
                RK_NPU_ERR_CACHE_MISS
            };
            if rc == 0 {
                b.cache_hits += 1;
                found = true;
            } else {
                check_cache_miss(rc, "floating cache load")?;
                b.cache_misses += 1;
            }
            if !found && b.options.tune_search {
                let mut cfg = cfg;
                cfg.verbose = i32::from(tracing::enabled!(tracing::Level::DEBUG));
                eprintln!("Tuning floating: dtype={} M={rows} K={k} N={n}", self.dtype);
                b.tune_calls += 1;
                check(
                    unsafe {
                        if let Some(path) = &path {
                            rk_npu_mm_autotune_cached(
                                b.domain.raw,
                                &cfg,
                                path.as_ptr(),
                                0,
                                &mut strategy,
                                ptr::null_mut(),
                                ptr::null_mut(),
                            )
                        } else {
                            rk_npu_mm_autotune(b.domain.raw, &cfg, &mut strategy, ptr::null_mut())
                        }
                    },
                    "floating tune",
                )?;
                found = true;
            }
        }
        if found
            && self.is_decoder()
            && (strategy.options.n_tile > 4096 || (strategy.options.n_tile == 0 && n > 4096))
        {
            // Keep the validated mixed FP16/INT8 recipe even when a
            // float-only profile selected a wider output-channel task.
            found = false;
            b.fallback_calls += 1;
        }
        if found && !weight.is_null() {
            let rc = unsafe { rk_npu_mm_tuned_weights_match(weight, &strategy) };
            if rc != 0 {
                check_cache_miss(rc, "floating resident strategy")?;
                found = false;
                b.fallback_calls += 1;
            }
        }
        if !found {
            check(
                unsafe { rk_npu_mm_strategy_default(&cfg, &mut strategy) },
                "default floating strategy",
            )?;
            // The library baseline reserves 128 physical rows even for one
            // logical token. Wide auxiliary projections need only their real
            // rows; keeping the tile small also bounds the input CBUF footprint.
            if self.is_decoder() {
                strategy.m_tile = rows.min(32) as i32;
                let cores = b.options.npu_mask.count_ones() as usize;
                strategy.options.n_tile =
                    n.div_ceil(cores).div_ceil(32).saturating_mul(32).min(4096) as i32;
            }
        }
        let mut raw = ptr::null_mut();
        check(
            unsafe { rk_npu_mm_tuned_workspace_create(b.domain.raw, &strategy, &mut raw) },
            "floating workspace",
        )?;
        works.insert(
            key,
            FloatWork {
                untuned: !found,
                gelu: None,
                workspace: Handle::new(raw, rk_npu_mm_tuned_workspace_free, "floating workspace")?,
                strategy,
            },
        );
        if !found {
            crate::progress::untuned(&b.options, "FP16/BF16", rows, k, n, self.source);
        }
        Ok(())
    }
    fn report_untuned(&self, rows: usize, k: usize, n: usize) {
        let b = self.device.0.lock().unwrap();
        crate::progress::untuned(&b.options, "FP16/BF16", rows, k, n, self.source);
    }
    /// The currently prepared strategy, including measured complete-call cost.
    pub fn profile(&self, k: usize, n: usize) -> Result<Value> {
        let works = self.work.lock().unwrap();
        let s = &self
            .prepared(&works, k, n)
            .context("floating shape is not prepared")?
            .strategy;
        Ok(json!({
            "m": s.M, "k": s.K, "n": s.N, "dtype": s.dtype, "host_type": s.host_type,
            "m_tile": s.m_tile, "n_tile": s.options.n_tile,
            "task_m_tile": s.task_m_tile, "fused_io": s.fused_io,
            "k_tile": s.k_tile, "npu_reduce": i32::from(s.k_tile > 0),
            "a_layout": s.options.a_layout, "c_layout": s.options.c_layout,
            "split_k": s.options.split_k, "npu_mask": s.options.allowed_npu_core_mask,
            "cpu_mask": s.cpu_core_mask, "cpu_threads": s.cpu_threads,
            "revision": s.tuning_revision, "total_us": s.total_us,
            "robust_us": s.robust_us, "jitter_pct": s.jitter_pct,
            "workspace_us": s.workspace_us, "weight_us": s.weight_us,
            "max_abs_error": s.max_abs_error,
        }))
    }
    /// Weight is row-major [N,K], as in safetensors/torch Linear.
    pub fn load(&self, weight: &[f32], bias: &[f32], k: usize, n: usize) -> Result<FloatLinear> {
        ensure!(
            k.checked_mul(n) == Some(weight.len()) && bias.len() == n,
            "floating weight shape mismatch"
        );
        ensure!(
            bias.iter().all(|x| x.is_finite()),
            "nonfinite floating bias"
        );
        let b = self.device.0.lock().unwrap();
        let cfg = config(&b, k, n, 1, self.dtype, self.host_type)?;
        let mut half = vec![0; weight.len()];
        for i in 0..k {
            for j in 0..n {
                let value = weight[j * k + i];
                half[i * n + j] = if self.dtype == RK_NPU_MM_BF16 {
                    let x = half::bf16::from_f32(value);
                    ensure!(x.is_finite(), "nonfinite/overflowed BF16 weight");
                    x.to_bits()
                } else {
                    let x = f16::from_f32(value);
                    ensure!(x.is_finite(), "nonfinite/overflowed FP16 weight");
                    x.to_bits()
                };
            }
        }
        let path = if b.options.tune {
            cache_directory(&b.options)?
        } else {
            None
        };
        let mut raw = ptr::null_mut();
        check(
            unsafe {
                rk_npu_mm_tuned_weights_create(
                    b.domain.raw,
                    &cfg,
                    half.as_ptr(),
                    (half.len() * 2) as u64,
                    path.as_ref().map_or(ptr::null(), |p| p.as_ptr()),
                    &mut raw,
                )
            },
            "create floating resident weights",
        )?;
        Ok(FloatLinear {
            weight: Handle::new(
                raw,
                rk_npu_mm_tuned_weights_free,
                "floating resident weights",
            )?,
            bias: bias.to_vec(),
            k,
            n,
            owner: self.clone(),
        })
    }
}
impl FloatLinear {
    #[cfg(test)]
    #[allow(dead_code, reason = "Shared workspace inspection for model tests.")]
    pub(crate) fn prepared_workspace(&self) -> Option<(usize, usize)> {
        let work = self.owner.work.lock().unwrap();
        self.owner
            .prepared(&work, self.k, self.n)
            .map(|w| (w.strategy.M as usize, w.workspace.raw as usize))
    }
    pub fn prepare(&self, rows: usize) -> Result<()> {
        self.owner
            .prepare_for_weights(self.k, self.n, rows, self.weight.raw)
    }
    pub fn run(&self, input: &[f32], output: &mut [f32], rows: usize) -> Result<()> {
        ensure!(
            self.owner.host_type == RK_NPU_MM_HOST_F32,
            "Linear requires FP32 host IO"
        );
        self.run_typed(input, output, rows)
    }
    pub fn run_half(&self, input: &[u16], output: &mut [u16], rows: usize) -> Result<()> {
        ensure!(
            self.owner.host_type == RK_NPU_MM_HOST_STORAGE && self.owner.dtype == RK_NPU_MM_F16,
            "Linear requires FP16 storage IO"
        );
        self.run_typed(input, output, rows)
    }
    /// Fused NPU GEMM, channel bias and tanh GELU; direct FP16 collection.
    pub fn run_half_gelu(
        &self,
        pool: &rayon::ThreadPool,
        input: &[u16],
        output: &mut [u16],
        rows: usize,
    ) -> Result<()> {
        self.with_gelu_view(input, rows, |view| gelu::collect_half(pool, view, output))
    }
    /// FP32 activation storage keeps the same fused NPU FFN operation.
    pub fn run_gelu(
        &self,
        pool: &rayon::ThreadPool,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
    ) -> Result<()> {
        let input = crate::ops::vision_half::narrow(input);
        self.with_gelu_view(&input, rows, |view| gelu::collect_float(pool, view, output))
    }
    fn with_gelu_view(
        &self,
        input: &[u16],
        rows: usize,
        collect: impl FnOnce(&rk_npu_mm_gelu_view) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            self.owner.dtype == RK_NPU_MM_F16,
            "fused GELU requires FP16 weights"
        );
        ensure!(
            rows > 0 && rows.checked_mul(self.k) == Some(input.len()),
            "fused GELU input shape mismatch"
        );
        self.prepare(rows)?;
        let mut works = self.owner.work.lock().unwrap();
        let w = works
            .get_mut(&self.owner.work_key(self.k, self.n, rows))
            .unwrap();
        if w.gelu.is_none() {
            let device = self.owner.device.0.lock().unwrap();
            let mut cfg = rk_npu_mm_gelu_config::default();
            unsafe {
                rk_npu_mm_gelu_config_init(&mut cfg, rows as i32, self.n as i32, self.k as i32)
            };
            cfg.core_mask = device.options.npu_mask;
            let mut raw = ptr::null_mut();
            check(
                unsafe { rk_npu_mm_gelu_prepare(device.domain.raw, &cfg, &mut raw) },
                "fused GELU workspace",
            )?;
            w.gelu = Some(Handle::new(
                raw,
                rk_npu_mm_gelu_free,
                "fused GELU workspace",
            )?);
        }
        let mut view = rk_npu_mm_gelu_view::default();
        check(
            unsafe {
                rk_npu_mm_gelu_run_weights(
                    w.gelu.as_ref().unwrap().raw,
                    input.as_ptr(),
                    std::mem::size_of_val(input) as u64,
                    self.weight.raw,
                    self.bias.as_ptr(),
                    &mut view,
                )
            },
            "fused GELU run",
        )?;
        // Keep the shared workspace borrowed until collection finishes.
        collect(&view)
    }
    // Only the two typed entrypoints above choose host storage. The ABI accepts
    // either representation with byte lengths; preparation and execution agree.
    fn run_typed<T>(&self, input: &[T], output: &mut [T], rows: usize) -> Result<()> {
        ensure!(
            rows > 0
                && rows.checked_mul(self.k) == Some(input.len())
                && rows.checked_mul(self.n) == Some(output.len()),
            "floating Linear input/output shape mismatch"
        );
        self.prepare(rows)?;
        let works = self.owner.work.lock().unwrap();
        let w = works
            .get(&self.owner.work_key(self.k, self.n, rows))
            .unwrap();
        check(
            unsafe {
                rk_npu_mm_tuned_run_weights(
                    w.workspace.raw,
                    input.as_ptr().cast(),
                    std::mem::size_of_val(input) as u64,
                    self.weight.raw,
                    self.bias.as_ptr(),
                    output.as_mut_ptr().cast(),
                    std::mem::size_of_val(output) as u64,
                )
            },
            "floating Linear run",
        )
        .with_context(|| format!("floating Linear M={rows} K={} N={}", self.k, self.n))
    }
}
