use anyhow::{Result, ensure};

/// Adjacent-pair RoPE shared by query heads and a rotary-only key.
pub struct InterleavedRope {
    inv_freq: Vec<f32>,
}

impl InterleavedRope {
    pub fn new(rotary_dim: usize, theta: f32) -> Result<Self> {
        ensure!(
            rotary_dim > 0 && rotary_dim.is_multiple_of(2) && theta.is_finite() && theta > 0.,
            "invalid interleaved RoPE geometry"
        );
        Ok(Self {
            inv_freq: (0..rotary_dim / 2)
                .map(|i| theta.powf(-((2 * i) as f32) / rotary_dim as f32))
                .collect(),
        })
    }

    /// Rotate the last rotary_dim channels of each Q head and the shared K.
    pub fn apply(
        &self,
        query: &mut [f32],
        key: &mut [f32],
        head_dim: usize,
        position: usize,
    ) -> Result<()> {
        let rotary_dim = self.inv_freq.len() * 2;
        ensure!(
            head_dim >= rotary_dim
                && query.len().is_multiple_of(head_dim)
                && key.len() == rotary_dim,
            "interleaved RoPE input shape mismatch"
        );
        let rotate = |pair: &mut [f32], s, c| {
            let (a, b) = (pair[0], pair[1]);
            pair[0] = a * c - b * s;
            pair[1] = b * c + a * s;
        };
        for (i, &freq) in self.inv_freq.iter().enumerate() {
            let (s, c) = (position as f32 * freq).sin_cos();
            rotate(&mut key[2 * i..2 * i + 2], s, c);
            let at = head_dim - rotary_dim + 2 * i;
            for head in query.chunks_mut(head_dim) {
                rotate(&mut head[at..at + 2], s, c);
            }
        }
        Ok(())
    }
}

/// Positions are [temporal, height, width], independent of token/KV indices.
#[derive(Debug, Clone)]
pub struct MRope {
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: Vec<f32>,
    sections: [usize; 3],
    interleaved: bool,
}

impl MRope {
    pub fn new(
        head_dim: usize,
        rotary_dim: usize,
        theta: f32,
        sections: [usize; 3],
        interleaved: bool,
    ) -> Result<Self> {
        ensure!(
            rotary_dim > 0 && rotary_dim.is_multiple_of(2) && rotary_dim <= head_dim,
            "invalid partial rotary dimension"
        );
        ensure!(theta.is_finite() && theta > 0.0, "invalid RoPE theta");
        ensure!(
            sections.iter().try_fold(0usize, |a, b| a.checked_add(*b)) == Some(rotary_dim / 2),
            "M-RoPE sections must cover half the rotary dimension"
        );
        let inv_freq = (0..rotary_dim / 2)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / rotary_dim as f32))
            .collect();
        Ok(Self {
            head_dim,
            rotary_dim,
            inv_freq,
            sections,
            interleaved,
        })
    }

    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    fn axis(&self, i: usize) -> usize {
        if self.interleaved {
            match i % 3 {
                1 if i / 3 < self.sections[1] => 1,
                2 if i / 3 < self.sections[2] => 2,
                _ => 0,
            }
        } else if i < self.sections[0] {
            0
        } else if i < self.sections[0] + self.sections[1] {
            1
        } else {
            2
        }
    }

    pub fn apply(&self, values: &mut [f32], heads: usize, positions: &[[usize; 3]]) -> Result<()> {
        ensure!(
            heads > 0
                && positions
                    .len()
                    .checked_mul(heads)
                    .and_then(|v| v.checked_mul(self.head_dim))
                    == Some(values.len()),
            "M-RoPE input/position shape mismatch"
        );
        let half = self.rotary_dim / 2;
        for (position, row) in positions
            .iter()
            .zip(values.chunks_mut(heads * self.head_dim))
        {
            for i in 0..half {
                let angle = position[self.axis(i)] as f32 * self.inv_freq[i];
                let (sine, cosine) = angle.sin_cos();
                for head in row.chunks_mut(self.head_dim) {
                    let (a, b) = (head[i], head[i + half]);
                    head[i] = a * cosine - b * sine;
                    head[i + half] = b * cosine + a * sine;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_interleaved_angles_match_each_head_and_position() -> Result<()> {
        let rope = InterleavedRope::new(64, 6_000_000.)?;
        for pos in [0, 37, 511, 32767] {
            let mut query: Vec<_> = (0..16 * 192).map(|i| (i as f32 * 0.13).sin()).collect();
            let mut key: Vec<_> = (0..64).map(|i| (i as f32 * 0.17).cos()).collect();
            let mut q_ref = query.clone();
            let mut k_ref = key.clone();
            for values in q_ref
                .chunks_mut(192)
                .map(|h| &mut h[128..])
                .chain(std::iter::once(k_ref.as_mut_slice()))
            {
                for (i, pair) in values.chunks_mut(2).enumerate() {
                    let angle = pos as f32 * 6_000_000_f32.powf(-((2 * i) as f32) / 64.);
                    let (s, c) = angle.sin_cos();
                    let (a, b) = (pair[0], pair[1]);
                    pair[0] = a * c - b * s;
                    pair[1] = b * c + a * s;
                }
            }
            rope.apply(&mut query, &mut key, 192, pos)?;
            assert_eq!(query, q_ref);
            assert_eq!(key, k_ref);
        }
        Ok(())
    }

    #[test]
    fn text_positions_degenerate_and_preserve_unrotated_tail() {
        let rope = MRope::new(24, 12, 10000.0, [2, 2, 2], true).unwrap();
        let plain = MRope::new(24, 12, 10000.0, [2, 2, 2], false).unwrap();
        let input: Vec<f32> = (0..2 * 3 * 24).map(|i| (i as f32 * 0.19).sin()).collect();
        let mut a = input.clone();
        let mut b = input.clone();
        rope.apply(&mut a, 3, &[[71; 3], [72; 3]]).unwrap();
        plain.apply(&mut b, 3, &[[71; 3], [72; 3]]).unwrap();
        assert_eq!(a, b);
        for (rotated, original) in a.chunks(24).zip(input.chunks(24)) {
            assert_eq!(&rotated[12..], &original[12..]);
        }
        assert!(rope.apply(&mut a, 3, &[[71; 3]]).is_err());
    }
}
