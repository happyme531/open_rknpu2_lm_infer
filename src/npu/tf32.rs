//! Resident TF32 Linear through the public raw MM ABI, with FP32 host IO.
use super::*;

struct Work {
    workspace: Handle<rk_npu_mm_workspace>,
    _plan: Handle<rk_npu_mm_plan>,
}
type Workspaces = BTreeMap<(usize, usize, usize), Work>;
#[derive(Clone)]
pub struct Tf32Backend {
    work: Rc<RefCell<Workspaces>>,
    device: Backend,
}
pub struct Tf32Linear {
    weight: Handle<rk_npu_mm_packed_b>,
    k: usize,
    n: usize,
    owner: Tf32Backend,
}
impl Tf32Backend {
    pub fn new(device: Backend) -> Self {
        Self {
            work: Rc::new(RefCell::new(BTreeMap::new())),
            device,
        }
    }
    fn plan(&self, k: usize, n: usize, rows: usize) -> Result<Handle<rk_npu_mm_plan>> {
        ensure!(
            k > 0 && k <= 4096 && n > 0 && n <= 65504 && rows > 0 && rows <= 65504,
            "TF32 Linear requires K within 1..4096 and M/N within 1..65504"
        );
        let b = self.device.0.borrow();
        let mut desc = rk_npu_mm_desc::default();
        let mut options = rk_npu_mm_options::default();
        unsafe {
            rk_npu_mm_desc_init(&mut desc, rows as i32, n as i32, k as i32, 1);
            rk_npu_mm_options_init(&mut options);
        }
        desc.a_type = RK_NPU_MM_TF32;
        desc.b_type = RK_NPU_MM_TF32;
        desc.c_type = RK_NPU_MM_F32;
        desc.trans_b = 1; // Checkpoint [N,K], packed once without host narrowing.
        options.allowed_npu_core_mask = b.options.npu_mask;
        options.split_k = 0;
        options.a_layout = RK_NPU_MM_LAYOUT_NATIVE;
        options.c_layout = RK_NPU_MM_LAYOUT_NATIVE;
        options.n_tile = n
            .div_ceil(b.options.npu_mask.count_ones() as usize)
            .div_ceil(32) as i32
            * 32;
        let mut raw = ptr::null_mut();
        check(
            unsafe { rk_npu_mm_plan_create(&desc, &options, &mut raw) },
            "TF32 Linear plan",
        )?;
        Handle::new(raw, rk_npu_mm_plan_free, "TF32 Linear plan")
    }
    pub fn prepare(&self, k: usize, n: usize, rows: usize) -> Result<()> {
        let mut work = self.work.borrow_mut();
        if work.contains_key(&(k, n, rows)) {
            return Ok(());
        }
        let plan = self.plan(k, n, rows)?;
        let b = self.device.0.borrow();
        let mut raw = ptr::null_mut();
        check(
            unsafe {
                rk_npu_mm_workspace_create(
                    b.domain.raw,
                    plan.raw,
                    RK_NPU_MM_HOST_PACKED_B,
                    &mut raw,
                )
            },
            "TF32 Linear workspace",
        )?;
        work.insert(
            (k, n, rows),
            Work {
                workspace: Handle::new(raw, rk_npu_mm_workspace_free, "TF32 Linear workspace")?,
                _plan: plan,
            },
        );
        Ok(())
    }
    pub fn retain_workspaces(&self, rows: &[usize]) {
        self.work
            .borrow_mut()
            .retain(|&(_, _, m), _| rows.contains(&m));
    }
    /// Compact row-major checkpoint weight [N,K]; bias is handled by the caller.
    pub fn load(&self, weight: &[f32], k: usize, n: usize) -> Result<Tf32Linear> {
        ensure!(
            k.checked_mul(n) == Some(weight.len()) && weight.iter().all(|x| x.is_finite()),
            "invalid TF32 Linear weight"
        );
        let plan = self.plan(k, n, 1)?;
        let b = self.device.0.borrow();
        let mut raw = ptr::null_mut();
        check(
            unsafe {
                rk_npu_mm_packed_b_create(
                    b.domain.raw,
                    plan.raw,
                    weight.as_ptr().cast(),
                    std::mem::size_of_val(weight) as u64,
                    &mut raw,
                )
            },
            "TF32 resident weights",
        )?;
        Ok(Tf32Linear {
            weight: Handle::new(raw, rk_npu_mm_packed_b_free, "TF32 weights")?,
            k,
            n,
            owner: self.clone(),
        })
    }
}
impl Tf32Linear {
    pub fn run(&self, input: &[f32], output: &mut [f32], rows: usize) -> Result<()> {
        ensure!(
            rows.checked_mul(self.k) == Some(input.len())
                && rows.checked_mul(self.n) == Some(output.len()),
            "TF32 Linear shape mismatch"
        );
        self.owner.prepare(self.k, self.n, rows)?;
        let work = self.owner.work.borrow();
        let w = work.get(&(self.k, self.n, rows)).unwrap();
        check(
            unsafe {
                rk_npu_mm_run_packed_b(
                    w.workspace.raw,
                    input.as_ptr().cast(),
                    std::mem::size_of_val(input) as u64,
                    self.weight.raw,
                    output.as_mut_ptr().cast(),
                    std::mem::size_of_val(output) as u64,
                )
            },
            "TF32 Linear run",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires exclusive RK3588 NPU access"]
    fn resident_router_matches_truncating_oracle_across_batches() -> Result<()> {
        let _lock = crate::validation::BoardLock::acquire()?;
        let mut options = ModelOptions {
            path: Default::default(),
            context: 256,
            chunk: 128,
            threads: 4,
            npu_mask: 7,
            cpu_mask: 0xf0,
            tune: false,
            tune_search: false,
            tune_cache: Default::default(),
            attention_tile: 512,
            vision_dtype: Default::default(),
            experts: Default::default(),
            speculation: Default::default(),
        };
        configure_cpu(&mut options)?;
        let (k, n) = (1536, 128);
        let weights: Vec<_> = (0..k * n)
            .map(|i| half::bf16::from_f32(((i * 13 % 71) as f32 - 35.) * 0.001).to_f32())
            .collect();
        let truncate = |x: f32| f32::from_bits(x.to_bits() & !0x1fff);
        for mask in [1, 3, 7] {
            options.npu_mask = mask;
            let backend = Tf32Backend::new(Backend::new(&options, 12)?);
            let linear = backend.load(&weights, k, n)?;
            let zero = backend.load(&vec![0.; k * n], k, n)?;
            let mut first = Vec::new();
            for rows in [1, 17, 128, 3, 1] {
                let input: Vec<_> = (0..rows * k)
                    .map(|i| ((i * 7 % 83) as f32 - 41.) * 0.01317)
                    .collect();
                let mut output = vec![0.; rows * n];
                zero.run(&input, &mut output, rows)?;
                assert!(output.iter().all(|&x| x == 0.));
                linear.run(&input, &mut output, rows)?;
                for row in 0..rows {
                    for col in 0..n {
                        let expected = (0..k)
                            .map(|i| {
                                f64::from(truncate(input[row * k + i]))
                                    * f64::from(truncate(weights[col * k + i]))
                            })
                            .sum::<f64>();
                        assert!(
                            (f64::from(output[row * n + col]) - expected).abs() < 5e-5,
                            "mask={mask} rows={rows} row={row} col={col}"
                        );
                    }
                }
                assert!(
                    output.iter().any(|x| x.to_bits() & 0x1fff != 0),
                    "FP32 output was narrowed"
                );
                if rows == 1 {
                    if first.is_empty() {
                        first = output;
                    } else {
                        assert_eq!(first, output);
                    }
                }
            }
            backend.retain_workspaces(&[1, 128]);
            assert_eq!(backend.work.borrow().len(), 2);
        }
        Ok(())
    }
}
