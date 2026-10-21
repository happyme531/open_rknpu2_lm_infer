use crate::model::checkpoint::Tensor;
use anyhow::Result;
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use std::sync::Arc;

pub struct Cpu {
    pub pool: Arc<ThreadPool>,
}

#[inline]
fn square_sum(x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut a = vdupq_n_f32(0.0);
        let mut b = a;
        let mut c = a;
        let mut d = a;
        let mut i = 0;
        while i + 16 <= x.len() {
            let p = x.as_ptr().add(i);
            let x0 = vld1q_f32(p);
            let x1 = vld1q_f32(p.add(4));
            let x2 = vld1q_f32(p.add(8));
            let x3 = vld1q_f32(p.add(12));
            a = vfmaq_f32(a, x0, x0);
            b = vfmaq_f32(b, x1, x1);
            c = vfmaq_f32(c, x2, x2);
            d = vfmaq_f32(d, x3, x3);
            i += 16;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
        for v in &x[i..] {
            sum += v * v;
        }
        sum
    }
    #[cfg(not(target_arch = "aarch64"))]
    x.iter().map(|v| v * v).sum()
}

#[inline]
fn rotate(a: f32, b: f32, cosine: f32, sine: f32) -> (f32, f32) {
    ((-b).mul_add(sine, a * cosine), b.mul_add(cosine, a * sine))
}

fn norm_row(x: &[f32], y: &mut [f32], w: &[f32], epsilon: f32) {
    let scale = (square_sum(x) / x.len() as f32 + epsilon).sqrt().recip();
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while i + 4 <= x.len() {
            let value = vmulq_f32(
                vmulq_n_f32(vld1q_f32(x.as_ptr().add(i)), scale),
                vld1q_f32(w.as_ptr().add(i)),
            );
            vst1q_f32(y.as_mut_ptr().add(i), value);
            i += 4;
        }
    }
    while i < x.len() {
        y[i] = (x[i] * scale) * w[i];
        i += 1;
    }
}

fn add_chunk(y: &mut [f32], x: &[f32]) {
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while i + 4 <= y.len() {
            vst1q_f32(
                y.as_mut_ptr().add(i),
                vaddq_f32(vld1q_f32(y.as_ptr().add(i)), vld1q_f32(x.as_ptr().add(i))),
            );
            i += 4;
        }
    }
    while i < y.len() {
        y[i] += x[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn exp2(x: std::arch::aarch64::float64x2_t) -> std::arch::aarch64::float64x2_t {
    use std::arch::aarch64::*;
    unsafe {
        let n = vrndmq_f64(vfmaq_n_f64(vdupq_n_f64(0.5), x, std::f64::consts::LOG2_E));
        let r = vfmsq_n_f64(x, n, std::f64::consts::LN_2);
        let r = vfmsq_n_f64(r, n, 2.319_046_813_846_299_6e-17);
        let mut p = vdupq_n_f64(1.0 / 479001600.0);
        for coefficient in [
            1.0 / 39916800.0,
            1.0 / 3628800.0,
            1.0 / 362880.0,
            1.0 / 40320.0,
            1.0 / 5040.0,
            1.0 / 720.0,
            1.0 / 120.0,
            1.0 / 24.0,
            1.0 / 6.0,
            0.5,
            1.0,
            1.0,
        ] {
            p = vfmaq_f64(vdupq_n_f64(coefficient), p, r);
        }
        let power = vreinterpretq_f64_s64(vshlq_n_s64::<52>(vaddq_s64(
            vcvtq_s64_f64(n),
            vdupq_n_s64(1023),
        )));
        vmulq_f64(p, power)
    }
}

fn swiglu_block(gates: &[f32], ups: &[f32], y: &mut [f32]) {
    let width = y.len();
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while i + 4 <= width {
            let gate = vld1q_f32(gates.as_ptr().add(i));
            let up = vld1q_f32(ups.as_ptr().add(i));
            if vminvq_u32(vcleq_f32(vabsq_f32(gate), vdupq_n_f32(80.0))) != 0 {
                let x = vnegq_f32(gate);
                let e = vcombine_f32(
                    vcvt_f32_f64(exp2(vcvt_f64_f32(vget_low_f32(x)))),
                    vcvt_f32_f64(exp2(vcvt_f64_f32(vget_high_f32(x)))),
                );
                vst1q_f32(
                    y.as_mut_ptr().add(i),
                    vmulq_f32(vdivq_f32(gate, vaddq_f32(vdupq_n_f32(1.0), e)), up),
                );
            } else {
                for j in i..i + 4 {
                    let g = gates[j];
                    y[j] = (g / (1.0 + (-g).exp())) * ups[j];
                }
            }
            i += 4;
        }
    }
    while i < width {
        let g = gates[i];
        y[i] = (g / (1.0 + (-g).exp())) * ups[i];
        i += 1;
    }
}

pub(crate) fn dense_dot(weight: &Tensor, row: usize, input: &[f32]) -> f32 {
    let base = row * input.len();
    let mut sum = 0.0;
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        if weight.dtype == safetensors::Dtype::BF16 {
            let pointer = weight.data().as_ptr().cast::<u16>().add(base);
            let mut a = vdupq_n_f32(0.0);
            let mut b = a;
            while i + 8 <= input.len() {
                let w = vld1q_u16(pointer.add(i));
                let lo = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(w)));
                let hi = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(w)));
                a = vfmaq_f32(a, lo, vld1q_f32(input.as_ptr().add(i)));
                b = vfmaq_f32(b, hi, vld1q_f32(input.as_ptr().add(i + 4)));
                i += 8;
            }
            sum = vaddvq_f32(vaddq_f32(a, b));
        } else if weight.dtype == safetensors::Dtype::F32 {
            let pointer = weight.data().as_ptr().cast::<f32>().add(base);
            let mut a = vdupq_n_f32(0.0);
            let mut b = a;
            let mut c = a;
            let mut d = a;
            while i + 16 <= input.len() {
                a = vfmaq_f32(
                    a,
                    vld1q_f32(pointer.add(i)),
                    vld1q_f32(input.as_ptr().add(i)),
                );
                b = vfmaq_f32(
                    b,
                    vld1q_f32(pointer.add(i + 4)),
                    vld1q_f32(input.as_ptr().add(i + 4)),
                );
                c = vfmaq_f32(
                    c,
                    vld1q_f32(pointer.add(i + 8)),
                    vld1q_f32(input.as_ptr().add(i + 8)),
                );
                d = vfmaq_f32(
                    d,
                    vld1q_f32(pointer.add(i + 12)),
                    vld1q_f32(input.as_ptr().add(i + 12)),
                );
                i += 16;
            }
            sum = vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
        }
    }
    while i < input.len() {
        sum += weight.number(base + i) * input[i];
        i += 1;
    }
    sum
}

impl Cpu {
    pub fn new(threads: usize) -> Result<Self> {
        Ok(Self {
            pool: Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|i| format!("model-cpu-{i}"))
                    .build()?,
            ),
        })
    }
    pub fn norm(&self, x: &[f32], y: &mut [f32], weight: &[f32], epsilon: f32) {
        let width = weight.len();
        if x.len() == width {
            norm_row(x, y, weight, epsilon);
        } else {
            self.pool.install(|| {
                y.par_chunks_mut(width)
                    .zip(x.par_chunks(width))
                    .for_each(|(y, x)| norm_row(x, y, weight, epsilon))
            });
        }
    }
    pub fn add(&self, y: &mut [f32], x: &[f32]) {
        if y.len() <= 16384 {
            add_chunk(y, x);
        } else {
            self.pool.install(|| {
                y.par_chunks_mut(4096)
                    .zip(x.par_chunks(4096))
                    .for_each(|(y, x)| add_chunk(y, x))
            });
        }
    }
    pub fn scale(&self, values: &mut [f32], scale: f32) {
        if scale != 1.0 {
            for value in values {
                *value *= scale;
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &mut [f32],
        heads: usize,
        d: usize,
        weight: &[f32],
        epsilon: f32,
        cosine: &[f32],
        sine: &[f32],
        position: usize,
    ) {
        let half = d / 2;
        let run = |head: usize, p: &mut [f32]| {
            let scale = if weight.is_empty() {
                1.0
            } else {
                (square_sum(p) / d as f32 + epsilon).sqrt().recip()
            };
            let offset = (position + head / heads) * half;
            let cosine = &cosine[offset..offset + half];
            let sine = &sine[offset..offset + half];
            let mut i = 0;
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let values = p.as_mut_ptr();
                while i + 4 <= half {
                    let mut a = vld1q_f32(values.add(i));
                    let mut b = vld1q_f32(values.add(half + i));
                    if !weight.is_empty() {
                        a = vmulq_f32(vmulq_n_f32(a, scale), vld1q_f32(weight.as_ptr().add(i)));
                        b = vmulq_f32(
                            vmulq_n_f32(b, scale),
                            vld1q_f32(weight.as_ptr().add(half + i)),
                        );
                    }
                    let c = vld1q_f32(cosine.as_ptr().add(i));
                    let s = vld1q_f32(sine.as_ptr().add(i));
                    let first = vfmsq_f32(vmulq_f32(a, c), b, s);
                    let second = vfmaq_f32(vmulq_f32(a, s), b, c);
                    vst1q_f32(values.add(i), first);
                    vst1q_f32(values.add(half + i), second);
                    i += 4;
                }
            }
            while i < half {
                let a = if weight.is_empty() {
                    p[i]
                } else {
                    (p[i] * scale) * weight[i]
                };
                let b = if weight.is_empty() {
                    p[i + half]
                } else {
                    (p[i + half] * scale) * weight[i + half]
                };
                (p[i], p[i + half]) = rotate(a, b, cosine[i], sine[i]);
                i += 1;
            }
        };
        if x.len() <= 4096 {
            for (head, p) in x.chunks_mut(d).enumerate() {
                run(head, p);
            }
        } else {
            self.pool.install(|| {
                x.par_chunks_mut(d)
                    .enumerate()
                    .for_each(|(head, p)| run(head, p))
            });
        }
    }
    pub fn swiglu(&self, gu: &[f32], output: &mut [f32], width: usize) {
        let block = 256.min(1usize << width.trailing_zeros());
        let run = |index: usize, y: &mut [f32]| {
            let start = index * block;
            let row = start / width;
            let column = start % width;
            let base = row * width * 2 + column;
            swiglu_block(
                &gu[base..base + y.len()],
                &gu[base + width..base + width + y.len()],
                y,
            );
        };
        if output.len() <= 1024 {
            for (index, y) in output.chunks_mut(block).enumerate() {
                run(index, y);
            }
        } else {
            self.pool.install(|| {
                output
                    .par_chunks_mut(block)
                    .enumerate()
                    .for_each(|(index, y)| run(index, y))
            });
        }
    }
    pub fn dense_head(&self, weight: &Tensor, input: &[f32], output: &mut [f32]) {
        self.pool.install(|| {
            output.par_iter_mut().enumerate().for_each(|(n, out)| {
                *out = dense_dot(weight, n, input);
            })
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= tolerance * (1.0 + b.abs()),
                "element {i}: {a} != {b}"
            );
        }
    }
    #[test]
    fn rope_without_qk_norm_preserves_projection_magnitude() {
        let cpu = Cpu::new(4).unwrap();
        for (d, heads) in [(64, 40), (128, 16)] {
            let position = 7;
            let mut input: Vec<f32> = (0..3 * heads * d)
                .map(|i| (i as f32 * 0.07).sin() * 4.0)
                .collect();
            let mut expected = input.clone();
            let cosine: Vec<f32> = (0..10 * d / 2).map(|i| (i as f32 * 0.11).cos()).collect();
            let sine: Vec<f32> = (0..10 * d / 2).map(|i| (i as f32 * 0.11).sin()).collect();
            for (i, head) in expected.chunks_mut(d).enumerate() {
                for j in 0..d / 2 {
                    let (a, b) = (head[j], head[j + d / 2]);
                    let p = (position + i / heads) * d / 2 + j;
                    head[j] = a * cosine[p] - b * sine[p];
                    head[j + d / 2] = a * sine[p] + b * cosine[p];
                }
            }
            cpu.rope(&mut input, heads, d, &[], 1e-6, &cosine, &sine, position);
            close(&input, &expected, 1e-6);
        }
    }
    #[test]
    fn rms_and_residual_match_scalar() {
        let cpu = Cpu::new(4).unwrap();
        for (rows, width) in [(1, 128), (1, 2560), (32, 2560)] {
            let x: Vec<f32> = (0..rows * width)
                .map(|i| ((i as f32) * 0.173).sin() * 2.0)
                .collect();
            let w: Vec<f32> = (0..width).map(|i| 0.7 + 0.001 * (i % 201) as f32).collect();
            let mut y = vec![0.0; x.len()];
            let mut expected = y.clone();
            cpu.norm(&x, &mut y, &w, 1e-6);
            for r in 0..rows {
                let row = &x[r * width..(r + 1) * width];
                let scale = (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + 1e-6)
                    .sqrt()
                    .recip();
                for i in 0..width {
                    expected[r * width + i] = (row[i] * scale) * w[i];
                }
            }
            close(&y, &expected, 3e-6);
            cpu.add(&mut y, &x);
            for (e, x) in expected.iter_mut().zip(&x) {
                *e += x;
            }
            close(&y, &expected, 3e-6);
        }
    }
    #[test]
    fn rope_matches_scalar_at_nonzero_position() {
        let cpu = Cpu::new(4).unwrap();
        let (rows, heads, d, position) = (3, 8, 128, 7);
        let mut x: Vec<f32> = (0..rows * heads * d)
            .map(|i| (i as f32 * 0.073).cos())
            .collect();
        let mut expected = x.clone();
        let weight: Vec<f32> = (0..d).map(|i| 0.8 + i as f32 * 0.002).collect();
        let cosine: Vec<f32> = (0..(position + rows) * d / 2)
            .map(|i| (i as f32 * 0.0017).cos())
            .collect();
        let sine: Vec<f32> = (0..cosine.len())
            .map(|i| (i as f32 * 0.0017).sin())
            .collect();
        cpu.rope(&mut x, heads, d, &weight, 1e-6, &cosine, &sine, position);
        for (head, p) in expected.chunks_mut(d).enumerate() {
            let scale = (p.iter().map(|v| v * v).sum::<f32>() / d as f32 + 1e-6)
                .sqrt()
                .recip();
            let offset = (position + head / heads) * d / 2;
            for i in 0..d / 2 {
                let a = p[i] * scale * weight[i];
                let b = p[i + d / 2] * scale * weight[i + d / 2];
                p[i] = ((a * cosine[offset + i]) as f64 - (b as f64) * (sine[offset + i] as f64))
                    as f32;
                p[i + d / 2] = ((b as f64) * (cosine[offset + i] as f64)
                    + (a * sine[offset + i]) as f64) as f32;
            }
        }
        close(&x, &expected, 2e-6);
    }
    #[test]
    fn rope_rotation_preserves_single_rounding() {
        let upper = f32::from_bits(0x3f800001);
        let lower = f32::from_bits(0x3f7ffffe);
        for (a, b, c, s, component, bits) in [
            (1.0, upper, 1.0, lower, 0, 0x28800000),
            (-1.0, upper, lower, 1.0, 1, 0xa8800000),
        ] {
            let actual = rotate(a, b, c, s);
            let expected = [
                ((a * c) as f64 - (b as f64) * (s as f64)) as f32,
                ((b as f64) * (c as f64) + (a * s) as f64) as f32,
            ];
            assert_eq!(actual.0.to_bits(), expected[0].to_bits());
            assert_eq!(actual.1.to_bits(), expected[1].to_bits());
            assert_eq!(expected[component].to_bits(), bits);
            let unfused = [a * c - b * s, b * c + a * s];
            assert_ne!(unfused[component].to_bits(), bits);
        }
    }
    #[test]
    fn swiglu_matches_scalar_for_decode_and_prefill() {
        let cpu = Cpu::new(4).unwrap();
        for (rows, width) in [(1, 128), (1, 9728), (3, 513), (4, 9728)] {
            let gu: Vec<f32> = (0..rows * 2 * width)
                .map(|i| (i as f32 * 0.019).sin() * 90.0)
                .collect();
            let mut y = vec![0.0; rows * width];
            let mut expected = y.clone();
            cpu.swiglu(&gu, &mut y, width);
            for r in 0..rows {
                for i in 0..width {
                    let g = gu[r * 2 * width + i];
                    let up = gu[r * 2 * width + width + i];
                    expected[r * width + i] = (g / (1.0 + (-g).exp())) * up;
                }
            }
            close(&y, &expected, 3e-7);
        }
    }
}
