//! Grouped routed FFN. Immutable expert banks share scratch within a domain.
use super::*;

pub(super) type Key = (usize, usize, usize, usize);
pub(super) struct Work {
    workspace: Handle<rk_npu_moe_w8_workspace>,
    pub bytes: u64,
}

pub struct Moe {
    weights: Handle<rk_npu_moe_w8_weights>,
    owner: Backend,
    key: Key,
}

impl Backend {
    pub fn load_moe(&self, gate: &[QuantMatrix], down: &[QuantMatrix], top: usize) -> Result<Moe> {
        ensure!(
            !gate.is_empty() && gate.len() == down.len(),
            "invalid expert bank"
        );
        gate.iter()
            .chain(down)
            .try_for_each(QuantMatrix::validate)?;
        ensure!(
            gate.iter().chain(down).all(|m| m.bits == 4),
            "grouped MoE requires W4 gate/up and down weights"
        );
        let (h, f, e) = (gate[0].k, gate[0].n / 2, gate.len());
        ensure!(
            gate.iter().all(|g| g.k == h && g.n == 2 * f)
                && down.iter().all(|d| d.k == f && d.n == h),
            "incompatible expert matrices"
        );
        let experts: Vec<_> = gate
            .iter()
            .zip(down)
            .map(|(g, d)| rk_npu_moe_w8_expert {
                gate_up: g.codes.as_ptr(),
                gate_up_scales: g.scales.as_ptr(),
                down: d.codes.as_ptr(),
                down_scales: d.scales.as_ptr(),
            })
            .collect();
        self.moe_bank((h, f, e, top), |domain, config| unsafe {
            rk_npu_moe_quant_weights_create(domain, config, experts.as_ptr())
        })
    }

    pub fn moe_slots(&self, h: usize, f: usize, slots: usize) -> Result<Moe> {
        self.moe_packed(h, f, slots, 1)
    }
    pub fn moe_packed(&self, h: usize, f: usize, slots: usize, top: usize) -> Result<Moe> {
        self.moe_bank((h, f, slots, top), |domain, config| unsafe {
            rk_npu_moe_quant_slots_create(domain, config)
        })
    }

    fn moe_bank(
        &self,
        key: Key,
        create: impl FnOnce(
            *mut rk_npu_iommu_domain,
            &rk_npu_moe_w8_weight_config,
        ) -> *mut rk_npu_moe_w8_weights,
    ) -> Result<Moe> {
        let (h, f, e, top) = key;
        let mut b = self.0.borrow_mut();
        ensure!(
            [1, 3, 7].contains(&b.options.npu_mask),
            "grouped MoE requires NPU mask 1/3/7"
        );
        let mut config = rk_npu_moe_w8_config::default();
        unsafe {
            rk_npu_moe_w8_config_init(
                &mut config,
                h as i32,
                f as i32,
                e as i32,
                0,
                top as i32,
                b.options.chunk as i32,
            )
        };
        config.npu_cores = b.options.npu_mask.count_ones() as i32;
        config.cpu_threads = b.options.threads as i32;
        let weights = Handle::new(
            create(b.domain.raw, &config.weights),
            rk_npu_moe_w8_weights_free,
            "packed MoE expert bank",
        )?;
        if !b.moe_work.contains_key(&key) {
            let workspace = Handle::new(
                unsafe { rk_npu_moe_quant_workspace_create(b.domain.raw, &config) },
                rk_npu_moe_w8_workspace_free,
                "grouped MoE workspace",
            )?;
            let bytes = unsafe { rk_npu_moe_w8_workspace_device_bytes(workspace.raw) };
            b.moe_work.insert(key, Work { workspace, bytes });
        }
        b.weight_bytes += unsafe { rk_npu_moe_w8_weights_device_bytes(weights.raw) };
        Ok(Moe {
            weights,
            owner: self.clone(),
            key,
        })
    }
}

impl Moe {
    pub fn weight_bytes(&self) -> u64 {
        unsafe { rk_npu_moe_w8_weights_device_bytes(self.weights.raw) }
    }
    pub fn blob_version() -> u32 {
        RK_NPU_MOE_QUANT_BLOB_VERSION
    }
    pub fn blob_bytes(&self) -> usize {
        (unsafe { rk_npu_moe_quant_expert_blob_bytes(self.weights.raw) }) as usize
    }
    pub fn export(&self, expert: usize) -> Result<Vec<u8>> {
        ensure!(expert < self.key.2, "expert index out of range");
        let bytes = unsafe { rk_npu_moe_quant_expert_blob_bytes(self.weights.raw) } as usize;
        ensure!(bytes > 0, "expert bank has no packed cache representation");
        let mut data = vec![0; bytes];
        check(
            unsafe {
                rk_npu_moe_quant_export_expert(
                    self.weights.raw,
                    expert as i32,
                    data.as_mut_ptr().cast(),
                    bytes as u64,
                )
            },
            "export packed expert",
        )?;
        Ok(data)
    }
    pub fn import(&mut self, slot: usize, data: &[u8]) -> Result<()> {
        ensure!(slot < self.key.2, "expert slot out of range");
        check(
            unsafe {
                rk_npu_moe_quant_import_expert(
                    self.weights.raw,
                    slot as i32,
                    data.as_ptr().cast(),
                    data.len() as u64,
                )
            },
            "import packed expert",
        )
    }
    pub fn max_rows(&self) -> usize {
        self.owner.0.borrow().options.chunk
    }
    pub fn run(
        &self,
        input: &[f32],
        ids: &[i32],
        routes: &[f32],
        output: &mut [f32],
        rows: usize,
    ) -> Result<()> {
        let (h, _, _, top) = self.key;
        ensure!(
            input.len() == rows * h
                && output.len() == rows * h
                && ids.len() == rows * top
                && routes.len() == ids.len(),
            "grouped MoE input shape mismatch"
        );
        let b = self.owner.0.borrow();
        let work = b.moe_work.get(&self.key).context("missing MoE workspace")?;
        check(
            unsafe {
                rk_npu_moe_w8_run_f32(
                    work.workspace.raw,
                    self.weights.raw,
                    rows as i32,
                    input.as_ptr(),
                    ids.as_ptr(),
                    routes.as_ptr(),
                    output.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "grouped routed FFN",
        )
    }
}
