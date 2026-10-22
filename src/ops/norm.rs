//! FP32 normalization with a fixed scalar reduction order.
use super::activation::{sigmoid, silu_norm};
use anyhow::{Result, ensure};
use rayon::{ThreadPool, prelude::*};

pub fn l2_normalize(values: &mut [f32], width: usize) -> Result<()> {
    ensure!(
        width > 0 && values.len().is_multiple_of(width),
        "L2 norm shape mismatch"
    );
    values.chunks_mut(width).for_each(l2_row);
    Ok(())
}

fn l2_row(row: &mut [f32]) {
    let scale = (row.iter().map(|v| v * v).sum::<f32>() + 1e-6)
        .sqrt()
        .recip();
    for v in row {
        *v *= scale;
    }
}

/// Parallelize independent heads while retaining each head's reduction order.
pub fn l2_normalize_parallel(pool: &ThreadPool, values: &mut [f32], width: usize) -> Result<()> {
    ensure!(
        width > 0 && values.len().is_multiple_of(width),
        "L2 norm shape mismatch"
    );
    pool.install(|| values.par_chunks_mut(width).for_each(l2_row));
    Ok(())
}

/// In-place weighted RMSNorm followed by sigmoid gating. The gate may be a
/// strided view of a fused projection; a single token runs without dispatch.
#[allow(clippy::too_many_arguments)]
pub fn sigmoid_gated_norm_in_place(
    pool: &ThreadPool,
    values: &mut [f32],
    gate: &[f32],
    gate_stride: usize,
    token_width: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<()> {
    let width = weight.len();
    ensure!(
        width > 0
            && token_width > 0
            && token_width.is_multiple_of(width)
            && values.len().is_multiple_of(token_width)
            && gate_stride >= token_width
            && epsilon.is_finite()
            && epsilon > 0.,
        "sigmoid gated norm shape/epsilon mismatch"
    );
    let rows = values.len() / token_width;
    ensure!(
        gate.len() == rows.saturating_sub(1) * gate_stride + usize::from(rows > 0) * token_width,
        "sigmoid gated norm gate shape mismatch"
    );
    let heads = token_width / width;
    let apply = |(index, row): (usize, &mut [f32])| {
        let start = (index / heads) * gate_stride + (index % heads) * width;
        let scale = (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + epsilon)
            .sqrt()
            .recip();
        for ((x, w), g) in row.iter_mut().zip(weight).zip(&gate[start..start + width]) {
            *x = ((*x * scale) * w) * sigmoid(*g);
        }
    };
    if rows <= 1 {
        values.chunks_mut(width).enumerate().for_each(apply);
    } else {
        pool.install(|| values.par_chunks_mut(width).enumerate().for_each(apply));
    }
    Ok(())
}

/// `zero_centered` selects effective weights `1 + weight` instead of `weight`.
pub fn rms_norm(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    zero_centered: bool,
    output: &mut [f32],
) -> Result<()> {
    let width = weight.len();
    ensure!(
        width > 0 && input.len().is_multiple_of(width) && output.len() == input.len(),
        "RMSNorm shape mismatch"
    );
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "invalid RMSNorm epsilon"
    );
    for (x, y) in input.chunks(width).zip(output.chunks_mut(width)) {
        let scale = (x.iter().map(|v| v * v).sum::<f32>() / width as f32 + epsilon)
            .sqrt()
            .recip();
        for ((x, y), w) in x.iter().zip(y).zip(weight) {
            *y = (x * scale) * (if zero_centered { 1.0 + w } else { *w });
        }
    }
    Ok(())
}

/// Each token contains token_width values; z can be a view of a wider projection.
#[allow(clippy::too_many_arguments)]
pub fn gated_norm_parallel_strided(
    pool: &ThreadPool,
    input: &[f32],
    z: &[f32],
    z_stride: usize,
    token_width: usize,
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<()> {
    ensure!(
        !weight.is_empty()
            && token_width > 0
            && token_width.is_multiple_of(weight.len())
            && input.len().is_multiple_of(token_width)
            && z_stride >= token_width
            && input.len().checked_div(token_width).and_then(|rows| {
                if rows == 0 {
                    Some(0)
                } else {
                    (rows - 1)
                        .checked_mul(z_stride)
                        .and_then(|n| n.checked_add(token_width))
                }
            }) == Some(z.len())
            && output.len() == input.len()
            && epsilon.is_finite()
            && epsilon > 0.0,
        "GDN gated norm shape/epsilon mismatch"
    );
    pool.install(|| {
        let heads = token_width / weight.len();
        input
            .par_chunks(weight.len())
            .zip(output.par_chunks_mut(weight.len()))
            .enumerate()
            .for_each(|(row, (x, out))| {
                let start = (row / heads) * z_stride + (row % heads) * weight.len();
                let z = &z[start..start + weight.len()];
                // Preserve the scalar norm's FP32 reduction order. The expensive
                // gate uses ggml's vector SiLU on AArch64.
                let scale = (x.iter().map(|v| v * v).sum::<f32>() / weight.len() as f32 + epsilon)
                    .sqrt()
                    .recip();
                silu_norm(x, z, weight, scale, out);
            });
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn parallel_l2_and_strided_sigmoid_norm_preserve_scalar_bits() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for rows in [1, 3, 128] {
            let (width, token_width, stride) = (128, 2048, 5 * 2048);
            let input: Vec<f32> = (0..rows * token_width)
                .map(|i| (i as f32 * 0.017).sin())
                .collect();
            let weight: Vec<f32> = (0..width).map(|i| 0.5 + (i as f32 * 0.13).cos()).collect();
            let mut expected = input.clone();
            let mut actual = input.clone();
            super::l2_normalize(&mut expected, width).unwrap();
            super::l2_normalize_parallel(&pool, &mut actual, width).unwrap();
            assert_eq!(actual, expected);
            let gates: Vec<f32> = (0..input.len())
                .map(|i| [-1000., -80., -1., -0., 0., 0.5, 80., 1000.][i % 8])
                .collect();
            let mut strided = vec![f32::NAN; (rows - 1) * stride + token_width];
            for t in 0..rows {
                strided[t * stride..t * stride + token_width]
                    .copy_from_slice(&gates[t * token_width..(t + 1) * token_width]);
            }
            super::rms_norm(&input, &weight, 1e-6, false, &mut expected).unwrap();
            for (x, &g) in expected.iter_mut().zip(&gates) {
                *x *= super::sigmoid(g);
            }
            actual.copy_from_slice(&input);
            super::sigmoid_gated_norm_in_place(
                &pool,
                &mut actual,
                &strided,
                stride,
                token_width,
                &weight,
                1e-6,
            )
            .unwrap();
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }

    #[test]
    fn fused_strided_gates_preserve_normalization_bits() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for width in [1, 3, 7, 128] {
            let token_width = width * 3;
            let stride = token_width + 19;
            let rows = 9;
            let weight: Vec<f32> = (0..width).map(|i| 0.5 + (i as f32 * 0.17).sin()).collect();
            let input: Vec<f32> = (0..rows * token_width)
                .map(|i| (i as f32 * 0.13).cos())
                .collect();
            let z: Vec<f32> = (0..input.len())
                .map(|i| [-1000., 1000., -81., -80., -0., 0., 0.3, 1., 79., 81.][i % 10])
                .collect();
            let mut strided = vec![f32::NAN; (rows - 1) * stride + token_width];
            for t in 0..rows {
                strided[t * stride..t * stride + token_width]
                    .copy_from_slice(&z[t * token_width..(t + 1) * token_width]);
            }
            let mut expected = z.clone();
            for (x, out) in input.chunks(width).zip(expected.chunks_mut(width)) {
                let scale = (x.iter().map(|v| v * v).sum::<f32>() / width as f32 + 1e-6)
                    .sqrt()
                    .recip();
                crate::ops::activation::silu_in_place(out);
                for ((out, x), w) in out.iter_mut().zip(x).zip(&weight) {
                    *out *= (x * scale) * w;
                }
            }
            let mut actual = vec![0.; input.len()];
            super::gated_norm_parallel_strided(
                &pool,
                &input,
                &strided,
                stride,
                token_width,
                &weight,
                1e-6,
                &mut actual,
            )
            .unwrap();
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "width={width}"
            );
        }
    }
}
