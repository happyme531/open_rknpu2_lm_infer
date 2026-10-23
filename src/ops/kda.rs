//! FP32 CPU Kimi Delta Attention, with per-key-channel decay.
use anyhow::{Result, ensure};
use rayon::{ThreadPool, prelude::*};

/// Activate a strided forget projection and per-head beta in one pass.
#[allow(clippy::too_many_arguments)]
pub fn activate_gates(
    pool: &ThreadPool,
    forget: &[f32],
    stride: usize,
    a: &[f32],
    dt: &[f32],
    lower_bound: f32,
    beta: &mut [f32],
    decay: &mut [f32],
) -> Result<()> {
    let heads = a.len();
    let width = dt.len();
    ensure!(
        heads > 0
            && width > 0
            && width.is_multiple_of(heads)
            && beta.len().is_multiple_of(heads)
            && stride >= width,
        "KDA gate dimensions mismatch"
    );
    let rows = beta.len() / heads;
    ensure!(
        decay.len() == rows * width
            && forget.len() == rows.saturating_sub(1) * stride + usize::from(rows > 0) * width,
        "KDA gate buffers mismatch"
    );
    let dimension = width / heads;
    let apply = |(index, (out, beta)): (usize, (&mut [f32], &mut f32))| {
        let head = index % heads;
        let at = (index / heads) * stride + head * dimension;
        for (j, value) in out.iter_mut().enumerate() {
            *value = lower_bound
                * super::activation::sigmoid((forget[at + j] + dt[head * dimension + j]) * a[head]);
        }
        *beta = super::activation::sigmoid(*beta);
    };
    if rows <= 1 {
        decay
            .chunks_mut(dimension)
            .zip(beta)
            .enumerate()
            .for_each(apply);
    } else {
        pool.install(|| {
            decay
                .par_chunks_mut(dimension)
                .zip(beta.par_iter_mut())
                .enumerate()
                .for_each(apply)
        });
    }
    Ok(())
}

#[repr(C)]
struct Params {
    rows: usize,
    heads: usize,
    dimension: usize,
    threads: usize,
    state: *mut f32,
    query: *const f32,
    key: *const f32,
    value: *const f32,
    log_decay: *const f32,
    beta: *const f32,
    output: *mut f32,
}

unsafe extern "C" {
    fn rknpu2_kda_run(params: *const Params);
}

pub struct State {
    heads: usize,
    dimension: usize,
    values: Vec<f32>,
}

fn valid_inputs(query: &[f32], key: &[f32], value: &[f32], decay: &[f32], beta: &[f32]) -> bool {
    // Full boolean reductions vectorize without changing which values pass.
    let finite = |values: &[f32]| values.iter().fold(true, |ok, x| ok & x.is_finite());
    finite(query)
        && finite(key)
        && finite(value)
        && decay
            .iter()
            .fold(true, |ok, x| ok & x.is_finite() & (*x <= 0.))
        && beta.iter().fold(true, |ok, x| ok & (*x >= 0.) & (*x <= 1.))
}

impl State {
    pub fn new(heads: usize, dimension: usize) -> Result<Self> {
        let size = heads
            .checked_mul(dimension)
            .and_then(|n| n.checked_mul(dimension));
        ensure!(
            heads > 0 && dimension > 0 && size.is_some(),
            "invalid KDA dimensions"
        );
        Ok(Self {
            heads,
            dimension,
            values: vec![0.; size.unwrap()],
        })
    }
    pub fn reset(&mut self) {
        self.values.fill(0.);
    }
    pub fn allocated_bytes(&self) -> usize {
        self.values.len() * 4
    }

    /// Q/K are normalized, log_decay is nonpositive and beta is in [0, 1].
    /// Value diagnostics are enabled in debug builds; shapes protect the C ABI.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        log_decay: &[f32],
        beta: &[f32],
        output: &mut [f32],
        threads: usize,
    ) -> Result<()> {
        let width = self.heads * self.dimension;
        ensure!(
            query.len().is_multiple_of(width) && threads > 0 && threads <= 4,
            "invalid KDA query shape/threads"
        );
        let rows = query.len() / width;
        ensure!(
            [key.len(), value.len(), log_decay.len(), output.len()]
                .iter()
                .all(|&n| n == query.len())
                && beta.len() == rows * self.heads,
            "KDA buffer shape mismatch"
        );
        debug_assert!(
            valid_inputs(query, key, value, log_decay, beta),
            "invalid KDA inputs"
        );
        let params = Params {
            rows,
            heads: self.heads,
            dimension: self.dimension,
            threads,
            state: self.values.as_mut_ptr(),
            query: query.as_ptr(),
            key: key.as_ptr(),
            value: value.as_ptr(),
            log_decay: log_decay.as_ptr(),
            beta: beta.as_ptr(),
            output: output.as_mut_ptr(),
        };
        unsafe {
            rknpu2_kda_run(&params);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vectorized_input_checks_preserve_finite_and_gate_boundaries() {
        for length in [1, 7, 17, 2048] {
            let values = vec![0.25; length];
            let decay = vec![-0.25; length];
            let beta = vec![0.5; length];
            for at in [0, length / 2, length - 1] {
                for x in [
                    f32::NAN,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    -0.01,
                    -0.,
                    0.,
                    1.,
                    1.01,
                ] {
                    let mut changed = values.clone();
                    changed[at] = x;
                    assert_eq!(
                        valid_inputs(&changed, &values, &values, &decay, &beta),
                        x.is_finite()
                    );
                    assert_eq!(
                        valid_inputs(&values, &changed, &values, &decay, &beta),
                        x.is_finite()
                    );
                    assert_eq!(
                        valid_inputs(&values, &values, &changed, &decay, &beta),
                        x.is_finite()
                    );
                    let mut gate = decay.clone();
                    gate[at] = x;
                    assert_eq!(
                        valid_inputs(&values, &values, &values, &gate, &beta),
                        x.is_finite() && x <= 0.
                    );
                    let mut b = beta.clone();
                    b[at] = x;
                    assert_eq!(
                        valid_inputs(&values, &values, &values, &decay, &b),
                        x.is_finite() && (0. ..=1.).contains(&x)
                    );
                }
            }
        }
    }
    #[test]
    fn strided_gates_match_contiguous_scalar_for_decode_and_prefill() -> Result<()> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        let (heads, d, stride) = (16, 128, 10240);
        let width = heads * d;
        let a: Vec<_> = (0..heads).map(|i| 0.1 + i as f32).collect();
        let dt: Vec<_> = (0..width).map(|i| (i as f32 * 0.13).sin()).collect();
        for rows in [1, 3, 128] {
            let f: Vec<_> = (0..rows * width)
                .map(|i| (i as f32 * 0.17).cos() * 20.)
                .collect();
            let b: Vec<_> = (0..rows * heads)
                .map(|i| (i as f32 * 0.3).sin() * 100.)
                .collect();
            let mut strided = vec![f32::NAN; (rows - 1) * stride + width];
            for t in 0..rows {
                strided[t * stride..t * stride + width]
                    .copy_from_slice(&f[t * width..(t + 1) * width]);
            }
            let expected: Vec<_> = f
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    -5. * crate::ops::activation::sigmoid((x + dt[i % width]) * a[(i / d) % heads])
                })
                .collect();
            let mut actual = vec![0.; f.len()];
            let mut beta = b.clone();
            activate_gates(
                &pool,
                &strided,
                stride,
                &a,
                &dt,
                -5.,
                &mut beta,
                &mut actual,
            )?;
            assert_eq!(actual, expected);
            assert_eq!(
                beta,
                b.into_iter()
                    .map(crate::ops::activation::sigmoid)
                    .collect::<Vec<_>>()
            );
        }
        Ok(())
    }
    #[test]
    fn vector_decay_matches_independent_f64_recurrence_and_chunking() -> Result<()> {
        for (heads, d, rows) in [(3, 7, 13), (3, 17, 9), (16, 128, 17), (16, 128, 128)] {
            let width = heads * d;
            let values = |salt: usize| -> Vec<f32> {
                (0..rows * width)
                    .map(|i| (((i * 17 + salt) % 113) as f32 - 56.) / 173.)
                    .collect()
            };
            let (mut q, mut k, v) = (values(7), values(29), values(43));
            crate::ops::norm::l2_normalize(&mut q, d)?;
            crate::ops::norm::l2_normalize(&mut k, d)?;
            let g: Vec<_> = (0..rows * width)
                .map(|i| -0.0001 - ((i * 13) % 17) as f32 * 0.07)
                .collect();
            let beta: Vec<_> = (0..rows * heads)
                .map(|i| 0.1 + (i % 9) as f32 * 0.09)
                .collect();
            let mut state = State::new(heads, d)?;
            for (i, x) in state.values.iter_mut().enumerate() {
                *x = ((i * 7 % 31) as f32 - 15.) * 0.003;
            }
            let mut split = State::new(heads, d)?;
            split.values.clone_from(&state.values);
            let mut reference: Vec<f64> = state.values.iter().map(|&x| x as f64).collect();
            let mut expected = vec![0.; rows * width];
            for t in 0..rows {
                for h in 0..heads {
                    for j in 0..d {
                        let mut prediction = 0.;
                        for i in 0..d {
                            let at = (h * d + j) * d + i;
                            reference[at] *= (g[t * width + h * d + i] as f64).exp();
                            prediction += reference[at] * k[t * width + h * d + i] as f64;
                        }
                        let delta = (v[t * width + h * d + j] as f64 - prediction)
                            * beta[t * heads + h] as f64;
                        let mut output = 0.;
                        for i in 0..d {
                            let at = (h * d + j) * d + i;
                            reference[at] += delta * k[t * width + h * d + i] as f64;
                            output += reference[at] * q[t * width + h * d + i] as f64;
                        }
                        expected[t * width + h * d + j] = (output / (d as f64).sqrt()) as f32;
                    }
                }
            }
            let mut actual = vec![0.; rows * width];
            state.run(&q, &k, &v, &g, &beta, &mut actual, 4)?;
            for (a, e) in actual.iter().zip(&expected) {
                assert!((a - e).abs() < 2e-6);
            }
            for (&a, &e) in state.values.iter().zip(&reference) {
                assert!((a as f64 - e).abs() < 2e-6);
            }
            let mut pieces = vec![0.; rows * width];
            for (first, count) in [(0, 1), (1, rows - 2), (rows - 1, 1)] {
                let range = first * width..(first + count) * width;
                split.run(
                    &q[range.clone()],
                    &k[range.clone()],
                    &v[range.clone()],
                    &g[range.clone()],
                    &beta[first * heads..(first + count) * heads],
                    &mut pieces[range],
                    1,
                )?;
            }
            assert_eq!(pieces, actual);
            assert_eq!(state.values, split.values);
            state.reset();
            assert!(state.values.iter().all(|x| *x == 0.));
        }
        Ok(())
    }
    #[test]
    fn rejects_invalid_shapes_before_native_call() -> Result<()> {
        let mut state = State::new(2, 4)?;
        let x = vec![0.; 8];
        let mut out = x.clone();
        assert!(state.run(&x, &x, &x, &x, &[0.5], &mut out, 1).is_err());
        assert!(
            state
                .run(&x, &x, &x, &x[..7], &[0.5; 2], &mut out, 1)
                .is_err()
        );
        assert!(state.run(&x, &x, &x, &x, &[0.5; 2], &mut out, 0).is_err());
        Ok(())
    }
}
