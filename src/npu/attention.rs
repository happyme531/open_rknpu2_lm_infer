//! Native FP16 Attention cache, staging and execution.
use super::*;

pub struct Attention {
    pub(super) cache: Handle<rk_npu_attention_f16_cache>,
    pub(super) query_heads: usize,
    pub(super) kv_heads: usize,
    pub(super) dimension: usize,
    pub(super) owner: Backend,
}

/// Noncausal Attention with sequence-local scratch, freed on drop.
pub struct FullAttention {
    pub(super) kv: Handle<rk_npu_attention_f16_cache>,
    pub(super) work: AttentionWork,
    pub(super) owner: Backend,
}

fn pack_half(input: &[f32], output: &mut [u16]) -> Result<()> {
    let mut first = 0;
    #[cfg(target_arch = "aarch64")]
    while first + 8 <= input.len() {
        unsafe {
            core::arch::asm!(
                "ld1 {{v0.4s, v1.4s}}, [{source}]",
                "fcvtn v2.4h, v0.4s",
                "fcvtn2 v2.8h, v1.4s",
                "st1 {{v2.8h}}, [{target}]",
                source = in(reg) input.as_ptr().add(first),
                target = in(reg) output.as_mut_ptr().add(first),
                out("v0") _, out("v1") _, out("v2") _,
                options(nostack),
            );
        }
        first += 8;
    }
    for (&value, encoded) in input[first..].iter().zip(output[first..].iter_mut()) {
        *encoded = f16::from_f32(value).to_bits();
    }
    ensure!(
        output[..input.len()]
            .iter()
            .all(|&value| value & 0x7c00 != 0x7c00),
        "nonfinite/overflowed FP16 GQA input"
    );
    Ok(())
}
impl FullAttention {
    /// Stateless image attention with FP16 storage at both boundaries.
    pub fn run_full_half(
        &mut self,
        q: &[u16],
        k: &[u16],
        v: &[u16],
        output: &mut [u16],
        rows: usize,
    ) -> Result<()> {
        let b = self.owner.0.borrow();
        let w = &mut self.work;
        let (heads, d) = (w.config.query_heads as usize, w.config.head_dim as usize);
        ensure!(
            rows > 0 && rows <= w.config.max_capacity as usize,
            "image FP16 Attention rows={rows} exceed prepared buffer capacity={}",
            w.config.max_capacity
        );
        ensure!(
            [q.len(), k.len(), v.len(), output.len()]
                .iter()
                .all(|n| *n == rows * heads * d),
            "full FP16 attention tensor shape mismatch"
        );
        let context = b.context.raw;
        w.k.resize(rows * heads * d, 0);
        w.v.resize(rows * heads * d, 0);
        for h in 0..heads {
            for r in 0..rows {
                let s = (r * heads + h) * d;
                let t = (h * rows + r) * d;
                w.k[t..t + d].copy_from_slice(&k[s..s + d]);
                w.v[t..t + d].copy_from_slice(&v[s..s + d]);
            }
        }
        check(
            unsafe {
                rk_npu_attention_f16_cache_load(
                    context,
                    self.kv.raw,
                    w.k.as_ptr(),
                    w.v.as_ptr(),
                    rows as i32,
                )
            },
            "load full FP16 image KV",
        )?;
        let step = w.config.max_query_rows as usize;
        w.output_half.resize(step * heads * d, 0);
        for offset in (0..rows).step_by(step) {
            let m = step.min(rows - offset);
            for h in 0..heads {
                for r in 0..m {
                    let s = ((offset + r) * heads + h) * d;
                    let t = (h * m + r) * d;
                    w.q[t..t + d].copy_from_slice(&q[s..s + d]);
                }
            }
            check(
                unsafe {
                    rk_npu_attention_f16_run_half(
                        context,
                        w.workspace.raw,
                        self.kv.raw,
                        w.q.as_ptr(),
                        m as i32,
                        offset as i32,
                        w.output_half.as_mut_ptr(),
                        ptr::null_mut(),
                        ptr::null(),
                    )
                },
                "noncausal FP16 image attention",
            )?;
            for h in 0..heads {
                for r in 0..m {
                    let t = ((offset + r) * heads + h) * d;
                    let s = (h * m + r) * d;
                    output[t..t + d].copy_from_slice(&w.output_half[s..s + d]);
                }
            }
        }
        Ok(())
    }
    /// Stateless noncausal attention over one complete image. All K/V rows are
    /// loaded before tiled queries; no padding rows enter the cache.
    pub fn run_full(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        rows: usize,
    ) -> Result<()> {
        let b = self.owner.0.borrow();
        let w = &mut self.work;
        let (heads, d) = (w.config.query_heads as usize, w.config.head_dim as usize);
        ensure!(
            rows > 0 && rows <= w.config.max_capacity as usize,
            "image Attention rows={rows} exceed prepared buffer capacity={}",
            w.config.max_capacity
        );
        ensure!(
            [q.len(), k.len(), v.len(), output.len()]
                .iter()
                .all(|n| *n == rows * heads * d),
            "full attention tensor shape mismatch"
        );
        let context = b.context.raw;
        w.k.resize(rows * heads * d, 0);
        w.v.resize(rows * heads * d, 0);
        for h in 0..heads {
            for r in 0..rows {
                let s = (r * heads + h) * d;
                let t = (h * rows + r) * d;
                pack_half(&k[s..s + d], &mut w.k[t..t + d])?;
                pack_half(&v[s..s + d], &mut w.v[t..t + d])?;
            }
        }
        check(
            unsafe {
                rk_npu_attention_f16_cache_load(
                    context,
                    self.kv.raw,
                    w.k.as_ptr(),
                    w.v.as_ptr(),
                    rows as i32,
                )
            },
            "load full image KV",
        )?;
        let step = w.config.max_query_rows as usize;
        for offset in (0..rows).step_by(step) {
            let m = step.min(rows - offset);
            for h in 0..heads {
                for r in 0..m {
                    let s = ((offset + r) * heads + h) * d;
                    let t = (h * m + r) * d;
                    pack_half(&q[s..s + d], &mut w.q[t..t + d])?;
                }
            }
            check(
                unsafe {
                    rk_npu_attention_f16_run(
                        context,
                        w.workspace.raw,
                        self.kv.raw,
                        w.q.as_ptr(),
                        m as i32,
                        offset as i32,
                        w.output.as_mut_ptr(),
                        ptr::null_mut(),
                        ptr::null(),
                    )
                },
                "noncausal image attention",
            )?;
            for h in 0..heads {
                for r in 0..m {
                    let t = ((offset + r) * heads + h) * d;
                    let s = (h * m + r) * d;
                    output[t..t + d].copy_from_slice(&w.output[s..s + d]);
                }
            }
        }
        Ok(())
    }
}
impl Attention {
    pub fn truncate(&self, length: usize) -> Result<()> {
        ensure!(
            length <= i32::MAX as usize,
            "invalid native KV prefix length"
        );
        check(
            unsafe { rk_npu_attention_f16_cache_truncate(self.cache.raw, length as i32) },
            "truncate native KV cache",
        )
    }
    pub fn reset(&self) -> Result<()> {
        let b = self.owner.0.borrow();
        check(
            unsafe {
                rk_npu_attention_f16_cache_load(
                    b.context.raw,
                    self.cache.raw,
                    ptr::null(),
                    ptr::null(),
                    0,
                )
            },
            "reset native KV cache",
        )
    }
    pub fn bytes(&self) -> Result<u64> {
        let mut info = rk_npu_attention_f16_cache_info::default();
        check(
            unsafe { rk_npu_attention_f16_cache_get_info(self.cache.raw, &mut info) },
            "native KV memory",
        )?;
        Ok(info.key_bytes + info.value_bytes + info.tail_bytes)
    }
    pub fn reserve(&self, capacity: usize) -> Result<()> {
        let b = self.owner.0.borrow();
        ensure!(
            capacity <= b.options.context,
            "native GQA context capacity exceeded"
        );
        check(
            unsafe {
                rk_npu_attention_f16_cache_reserve(
                    b.context.raw,
                    self.cache.raw,
                    (capacity.div_ceil(32) * 32) as i32,
                )
            },
            "reserve native KV cache",
        )
    }
    /// Append already projected/normalized/rotated KV without computing QK/PV.
    pub fn append(&self, k: &[f32], v: &[f32], rows: usize, position: usize) -> Result<()> {
        let mut b = self.owner.0.borrow_mut();
        self.append_impl(&mut b, k, v, rows, position)
    }
    fn append_impl(
        &self,
        b: &mut BackendInner,
        k: &[f32],
        v: &[f32],
        rows: usize,
        position: usize,
    ) -> Result<()> {
        let (kh, d) = (self.kv_heads, self.dimension);
        ensure!(
            rows > 0 && rows <= b.options.chunk && position + rows <= b.options.context,
            "native KV append capacity exceeded"
        );
        ensure!(
            k.len() == rows * kh * d && v.len() == rows * kh * d,
            "native KV append shape mismatch"
        );
        let mut info = rk_npu_attention_f16_cache_info::default();
        check(
            unsafe { rk_npu_attention_f16_cache_get_info(self.cache.raw, &mut info) },
            "native KV position",
        )?;
        ensure!(
            info.length as usize == position,
            "native KV cache position mismatch"
        );
        let work = b.attention.as_mut().unwrap();
        for h in 0..kh {
            for r in 0..rows {
                let source = (r * kh + h) * d;
                let target = (h * rows + r) * d;
                pack_half(&k[source..source + d], &mut work.k[target..target + d])?;
                pack_half(&v[source..source + d], &mut work.v[target..target + d])?;
            }
        }
        check(
            unsafe {
                rk_npu_attention_f16_cache_append(
                    b.context.raw,
                    self.cache.raw,
                    work.k.as_ptr(),
                    work.v.as_ptr(),
                    rows as i32,
                )
            },
            "append native KV",
        )?;
        Ok(())
    }
    pub fn run(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        rows: usize,
        position: usize,
    ) -> Result<()> {
        let mut b = self.owner.0.borrow_mut();
        let (qh, kh, d) = (self.query_heads, self.kv_heads, self.dimension);
        ensure!(
            rows > 0 && rows <= b.options.chunk && position + rows <= b.options.context,
            "native GQA context capacity exceeded"
        );
        ensure!(
            q.len() >= rows * qh * d
                && output.len() >= rows * qh * d
                && k.len() >= rows * kh * d
                && v.len() >= rows * kh * d,
            "attention input/output shape mismatch"
        );
        self.append_impl(
            &mut b,
            &k[..rows * kh * d],
            &v[..rows * kh * d],
            rows,
            position,
        )?;
        let context = b.context.raw;
        let work = b.attention.as_mut().unwrap();
        let step = work.config.max_query_rows as usize;
        for offset in (0..rows).step_by(step) {
            let m = step.min(rows - offset);
            for h in 0..qh {
                for r in 0..m {
                    let source = ((offset + r) * qh + h) * d;
                    let target = (h * m + r) * d;
                    pack_half(&q[source..source + d], &mut work.q[target..target + d])?;
                }
            }
            check(
                unsafe {
                    rk_npu_attention_f16_run(
                        context,
                        work.workspace.raw,
                        self.cache.raw,
                        work.q.as_ptr(),
                        m as i32,
                        (position + offset) as i32,
                        work.output.as_mut_ptr(),
                        ptr::null_mut(),
                        ptr::null(),
                    )
                },
                "native GQA run",
            )?;
            for h in 0..qh {
                for r in 0..m {
                    let target = ((offset + r) * qh + h) * d;
                    let source = (h * m + r) * d;
                    output[target..target + d].copy_from_slice(&work.output[source..source + d]);
                }
            }
        }
        Ok(())
    }
}
