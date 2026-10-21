//! Shared projected-input NPU GDN prefill; canonical state stays caller-owned.
use super::*;
use crate::ops::gated_delta::GdnDimensions;

pub struct GatedDelta {
    // Drop device work before releasing the context. Keep the configured full
    // window and at most one tail, shared across all recurrent layers.
    work: BTreeMap<usize, Handle<rk_npu_gdn_workspace>>,
    device: Backend,
    main_rows: usize,
    calls: usize,
    tokens: usize,
    elapsed_us: f64,
    hardware_us: f64,
    cpu_work_us: [f64; 4],
    compensated: bool,
}
impl GatedDelta {
    pub const CPU_MAX_ROWS: usize = 12;
    pub fn supports(dim: GdnDimensions, kernel: usize) -> bool {
        dim.key_heads() == 16
            && dim.value_heads() == 32
            && dim.key_dim() == 128
            && dim.value_dim() == 128
            && kernel == 4
    }
    pub fn new(device: Backend) -> Self {
        let main_rows = device.0.borrow().options.chunk;
        Self {
            work: BTreeMap::new(),
            device,
            main_rows,
            calls: 0,
            tokens: 0,
            elapsed_us: 0.,
            hardware_us: 0.,
            cpu_work_us: [0.; 4],
            compensated: false,
        }
    }
    pub fn compensated(mut self) -> Self {
        self.work.clear();
        self.compensated = true;
        self
    }
    pub fn is_compensated(&self) -> bool {
        self.compensated
    }
    pub fn prepare(&mut self, rows: usize) -> Result<()> {
        if rows <= Self::CPU_MAX_ROWS || self.work.contains_key(&rows) {
            return Ok(());
        }
        ensure!(rows <= 2048, "NPU GDN window exceeds 2048 tokens");
        if rows != self.main_rows {
            self.work.retain(|&m, _| m == self.main_rows);
        }
        let device = self.device.0.borrow();
        let mut cfg = rk_npu_gdn_config::default();
        unsafe { rk_npu_gdn_config_init(&mut cfg, rows as i32) };
        cfg.chunk_size = rows.next_power_of_two().clamp(16, 64) as i32;
        cfg.precision = if self.compensated {
            RK_NPU_GDN_COMPENSATED_TF32
        } else {
            RK_NPU_GDN_TF32
        };
        cfg.cpu_threads = device.options.threads as i32;
        cfg.core_mask = device.options.npu_mask;
        // The checkpoint repeats each key head over adjacent value heads.
        cfg.head_mapping = RK_NPU_GDN_HEAD_GROUPED;
        let mut raw = ptr::null_mut();
        check(
            unsafe { rk_npu_gdn_prepare(device.domain.raw, &cfg, &mut raw) },
            "prepare NPU GDN",
        )?;
        self.work.insert(
            rows,
            Handle::new(raw, rk_npu_gdn_workspace_free, "NPU GDN workspace")?,
        );
        Ok(())
    }
    pub fn retain_workspaces(&mut self, rows: &[usize]) {
        self.work.retain(|m, _| rows.contains(m));
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_projected(
        &mut self,
        rows: usize,
        projected: &[f32],
        projected_stride: usize,
        weights: &[f32],
        history: &mut [f32],
        log_decay: &[f32],
        beta: &[f32],
        state: &mut [f32],
        output: &mut [f32],
    ) -> Result<()> {
        ensure!(
            rows > Self::CPU_MAX_ROWS
                && projected_stride >= 8192
                && projected.len() == (rows - 1) * projected_stride + 8192
                && weights.len() == 8192 * 4
                && history.len() == 8192 * 3
                && log_decay.len() == rows * 32
                && beta.len() == rows * 32
                && state.len() == 32 * 128 * 128
                && output.len() == rows * 4096,
            "NPU GDN projected/state shape mismatch"
        );
        self.prepare(rows)?;
        let work = self.work.get(&rows).unwrap();
        let mut timings = rk_npu_gdn_timings::default();
        // Both canonical state and scratch history may alias their respective
        // inputs under this ABI. Failure leaves them unchanged.
        let rc = unsafe {
            rk_npu_gdn_run_projected_strided(
                work.raw,
                projected.as_ptr(),
                projected_stride as u64,
                weights.as_ptr(),
                history.as_ptr(),
                log_decay.as_ptr(),
                beta.as_ptr(),
                state.as_ptr(),
                output.as_mut_ptr(),
                state.as_mut_ptr(),
                history.as_mut_ptr(),
                &mut timings,
            )
        };
        if rc != 0 {
            // A failed device workspace cannot be reused. Propagate the error;
            // the model's normal failure path resets partially processed layers.
            self.work.remove(&rows);
            return check(rc, "NPU GDN projected prefill");
        }
        self.calls += 1;
        self.tokens += rows;
        self.elapsed_us += timings.total_us;
        self.hardware_us += timings.hardware_us;
        for (total, elapsed) in self.cpu_work_us.iter_mut().zip([
            timings.prepare_us,
            timings.coefficients_us,
            timings.solve_us,
            timings.collect_us,
        ]) {
            *total += elapsed;
        }
        Ok(())
    }
    pub fn reset_statistics(&mut self) {
        self.calls = 0;
        self.tokens = 0;
        self.elapsed_us = 0.;
        self.hardware_us = 0.;
        self.cpu_work_us.fill(0.);
    }
    pub fn stats(&self) -> Value {
        json!({"gdn_npu_precision":"tf32","gdn_npu_calls":self.calls,"gdn_npu_tokens":self.tokens,
            "gdn_npu_ms":self.elapsed_us/1000.,"gdn_npu_hardware_ms":self.hardware_us/1000.,
            "gdn_npu_prepare_ms":self.cpu_work_us[0]/1000.,
            "gdn_npu_coefficients_ms":self.cpu_work_us[1]/1000.,
            "gdn_npu_solve_ms":self.cpu_work_us[2]/1000.,
            "gdn_npu_collect_ms":self.cpu_work_us[3]/1000.,
            "gdn_npu_chunks":self.work.keys().map(|&m| json!({"rows":m,
                "chunk":m.next_power_of_two().clamp(16,64)})).collect::<Vec<_>>(),
            "gdn_npu_workspace_rows":self.work.keys().copied().collect::<Vec<_>>()})
    }
}
