//! Absorbed Ling MLA: shared latent512 + rotary64 cache, no expanded head KV.
use super::*;

pub struct Mla {
    cache: Handle<rk_npu_mla_f16_cache>,
    workspace: Handle<rk_npu_mla_f16_workspace>,
    _weights: Handle<rk_npu_mla_f16_weights>,
    owner: Backend,
    io: RefCell<HostIo>,
}

#[derive(Default)]
struct HostIo {
    query: Vec<u16>,
    latent: Vec<u16>,
    rotary: Vec<u16>,
}

impl Backend {
    /// kv_b is compact HF [16,256,512], content K128 then V128 per head.
    pub fn mla(&self, kv_b: &[f32]) -> Result<Mla> {
        ensure!(kv_b.len() == 16 * 256 * 512, "Ling MLA kv_b shape mismatch");
        let b = self.0.borrow();
        let mut config = rk_npu_mla_f16_config::default();
        unsafe { rk_npu_mla_f16_config_init(&mut config) };
        config.mode = RK_NPU_MLA_F16_ABSORBED;
        config.max_query_rows = 32;
        config.kv_tile = b.options.attention_tile as i32;
        config.initial_capacity = 32;
        config.max_capacity = b.options.context.div_ceil(32) as i32 * 32;
        config.core_mask = b.options.npu_mask;
        config.flags = RK_NPU_MLA_F16_STABLE_SOFTMAX;
        let encoded: Vec<_> = kv_b.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
        ensure!(
            encoded.iter().all(|&v| v & 0x7c00 != 0x7c00),
            "nonfinite MLA weights"
        );
        let mut weights = ptr::null_mut();
        check(
            unsafe { rk_npu_mla_f16_weights_create(b.domain.raw, encoded.as_ptr(), &mut weights) },
            "MLA weights",
        )?;
        let weights = Handle::new(weights, rk_npu_mla_f16_weights_free, "MLA weights")?;
        let mut cache = ptr::null_mut();
        check(
            unsafe { rk_npu_mla_f16_cache_create(b.domain.raw, &config, weights.raw, &mut cache) },
            "absorbed MLA cache",
        )?;
        let cache = Handle::new(cache, rk_npu_mla_f16_cache_free, "MLA cache")?;
        let mut workspace = ptr::null_mut();
        check(
            unsafe { rk_npu_mla_f16_prepare(b.domain.raw, &config, weights.raw, &mut workspace) },
            "MLA workspace",
        )?;
        let workspace = Handle::new(workspace, rk_npu_mla_f16_workspace_free, "MLA workspace")?;
        Ok(Mla {
            cache,
            workspace,
            _weights: weights,
            owner: self.clone(),
            io: RefCell::default(),
        })
    }
}
impl Mla {
    pub fn info(&self) -> Result<Value> {
        let mut info = rk_npu_mla_f16_cache_info::default();
        check(
            unsafe { rk_npu_mla_f16_cache_get_info(self.cache.raw, &mut info) },
            "MLA cache info",
        )?;
        let io = self.io.borrow();
        let host_bytes = (io.query.capacity() + io.latent.capacity() + io.rotary.capacity()) * 2;
        Ok(
            json!({"mode":"absorbed", "length":info.length,"capacity":info.capacity,
            "max_capacity":info.max_capacity,"key_bytes":info.key_bytes,"value_bytes":info.value_bytes,
            "tail_bytes":info.tail_bytes,"latent_dim":512,"rotary_dim":64,"kv_heads":1,
            "host_scratch_bytes":host_bytes}),
        )
    }
    pub fn reserve(&self, capacity: usize) -> Result<()> {
        let b = self.owner.0.borrow();
        ensure!(capacity <= b.options.context, "MLA reserve exceeds context");
        check(
            unsafe { rk_npu_mla_f16_cache_reserve(b.context.raw, self.cache.raw, capacity as i32) },
            "MLA reserve",
        )
    }
    pub fn reset(&self) -> Result<()> {
        let b = self.owner.0.borrow();
        check(
            unsafe {
                rk_npu_mla_f16_cache_load(
                    b.context.raw,
                    self.cache.raw,
                    ptr::null(),
                    ptr::null(),
                    0,
                )
            },
            "MLA reset",
        )
    }
    pub fn truncate(&self, position: usize) -> Result<()> {
        let position = i32::try_from(position).context("MLA prefix position overflow")?;
        check(
            unsafe { rk_npu_mla_f16_cache_truncate(self.cache.raw, position) },
            "MLA prefix truncate",
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        query: &[f32],
        latent: &[f32],
        rotary: &[f32],
        gates: &[f32],
        output: &mut [f32],
        rows: usize,
        position: usize,
    ) -> Result<()> {
        ensure!(
            rows > 0
                && query.len() == rows * 16 * 192
                && latent.len() == rows * 512
                && rotary.len() == rows * 64
                && gates.len() == rows * 16
                && output.len() == rows * 16 * 128,
            "MLA input shape mismatch"
        );
        let b = self.owner.0.borrow();
        let mut info = rk_npu_mla_f16_cache_info::default();
        check(
            unsafe { rk_npu_mla_f16_cache_get_info(self.cache.raw, &mut info) },
            "MLA position",
        )?;
        ensure!(
            info.length as usize == position && position + rows <= b.options.context,
            "MLA cache position/context mismatch"
        );
        let mut io = self.io.borrow_mut();
        io.query.resize(query.len(), 0);
        io.latent.resize(latent.len(), 0);
        io.rotary.resize(rotary.len(), 0);
        super::attention::pack_half(query, &mut io.query)?;
        super::attention::pack_half(latent, &mut io.latent)?;
        super::attention::pack_half(rotary, &mut io.rotary)?;
        check(
            unsafe {
                rk_npu_mla_f16_cache_append(
                    b.context.raw,
                    self.cache.raw,
                    io.latent.as_ptr(),
                    io.rotary.as_ptr(),
                    rows as i32,
                )
            },
            "MLA latent append",
        )?;
        for first in (0..rows).step_by(32) {
            let count = 32.min(rows - first);
            check(
                unsafe {
                    rk_npu_mla_f16_run(
                        b.context.raw,
                        self.workspace.raw,
                        self.cache.raw,
                        io.query.as_ptr().add(first * 16 * 192),
                        gates.as_ptr().add(first * 16),
                        count as i32,
                        (position + first) as i32,
                        output.as_mut_ptr().add(first * 16 * 128),
                        ptr::null_mut(),
                        ptr::null(),
                    )
                },
                "absorbed MLA run",
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires exclusive RK3588 NPU access"]
    fn absorbed_cache_matches_expanded_oracle_across_prefill_decode_and_reset() -> Result<()> {
        let _lock = crate::validation::BoardLock::acquire()?;
        let mut options = ModelOptions {
            path: Default::default(),
            context: 96,
            chunk: 64,
            threads: 4,
            npu_mask: 7,
            cpu_mask: 0,
            tune: false,
            tune_search: false,
            tune_cache: Default::default(),
            attention_tile: 256,
            vision_dtype: Default::default(),
            checkpoints: Default::default(),

            experts: Default::default(),
            speculation: Default::default(),
        };
        configure_cpu(&mut options)?;
        let backend = Backend::new(&options, 15)?;
        let mut weights = vec![0.; 16 * 256 * 512];
        for h in 0..16 {
            for j in 0..128 {
                weights[(h * 256 + j) * 512 + j] = if h % 2 == 0 { 0.5 } else { 1. };
                weights[(h * 256 + 128 + j) * 512 + j] = if h % 2 == 0 { 0.25 } else { -0.5 };
            }
        }
        let mla = backend.mla(&weights)?;
        let values = |n, salt| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 7 + salt) % 31) as f32 / 64. - 15. / 64.)
                .collect()
        };
        let q = values(34 * 16 * 192, 3);
        let latent = values(34 * 512, 11);
        let rotary = values(34 * 64, 17);
        let gates = values(34 * 16, 29);
        let mut output = vec![0.; 34 * 16 * 128];
        mla.run(
            &q[..33 * 16 * 192],
            &latent[..33 * 512],
            &rotary[..33 * 64],
            &gates[..33 * 16],
            &mut output[..33 * 16 * 128],
            33,
            0,
        )?;
        let host_buffers = |mla: &Mla| {
            let io = mla.io.borrow();
            [
                (io.query.as_ptr() as usize, io.query.capacity()),
                (io.latent.as_ptr() as usize, io.latent.capacity()),
                (io.rotary.as_ptr() as usize, io.rotary.capacity()),
            ]
        };
        let prepared_host_buffers = host_buffers(&mla);
        mla.reserve(96)?;
        assert_eq!(mla.info()?["length"], 33);
        mla.run(
            &q[33 * 16 * 192..],
            &latent[33 * 512..],
            &rotary[33 * 64..],
            &gates[33 * 16..],
            &mut output[33 * 16 * 128..],
            1,
            33,
        )?;
        assert_eq!(host_buffers(&mla), prepared_host_buffers);
        for row in [0, 1, 16, 31, 32, 33] {
            for h in 0..16 {
                let k_scale = if h % 2 == 0 { 0.5 } else { 1. };
                let v_scale = if h % 2 == 0 { 0.25 } else { -0.5 };
                let mut scores = Vec::new();
                for t in 0..=row {
                    let content = (0..128)
                        .map(|d| {
                            q[(row * 16 + h) * 192 + d] as f64
                                * k_scale
                                * latent[t * 512 + d] as f64
                        })
                        .sum::<f64>();
                    let rope = (0..64)
                        .map(|d| {
                            q[(row * 16 + h) * 192 + 128 + d] as f64 * rotary[t * 64 + d] as f64
                        })
                        .sum::<f64>();
                    scores.push(((content + rope) / 192_f64.sqrt()).exp());
                }
                let sum = scores.iter().sum::<f64>();
                let gate = 1. / (1. + (-(gates[row * 16 + h] as f64)).exp());
                for d in 0..128 {
                    let expected = scores
                        .iter()
                        .enumerate()
                        .map(|(t, p)| p / sum * latent[t * 512 + d] as f64 * v_scale)
                        .sum::<f64>()
                        * gate;
                    let actual = output[(row * 16 + h) * 128 + d] as f64;
                    assert!(
                        (actual - expected).abs() < 3e-4,
                        "row={row} head={h} d={d}: {actual} vs {expected}"
                    );
                }
            }
        }
        let info = mla.info()?;
        assert_eq!(info["length"], 34);
        let bytes = info["key_bytes"].as_u64().unwrap()
            + info["value_bytes"].as_u64().unwrap()
            + info["tail_bytes"].as_u64().unwrap();
        assert!(
            bytes < 96 * 16 * (192 + 128) * 2,
            "expanded head KV allocated: {info}"
        );
        mla.reset()?;
        assert_eq!(mla.info()?["length"], 0);
        let mut fresh = vec![0.; 16 * 128];
        mla.run(
            &q[..16 * 192],
            &latent[..512],
            &rotary[..64],
            &gates[..16],
            &mut fresh,
            1,
            0,
        )?;
        for (a, b) in fresh.iter().zip(&output[..16 * 128]) {
            assert!((a - b).abs() < 3e-4);
        }
        Ok(())
    }
}
