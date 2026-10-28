//! Replay the production KDA input scans separately from the native kernel.
use std::{fs, hint::black_box, path::Path, time::Instant};
fn read(path: &Path) -> Vec<f32> {
    fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn valid(q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32]) -> bool {
    q.iter().chain(k).chain(v).all(|x| x.is_finite())
        && g.iter().all(|x| x.is_finite() && *x <= 0.)
        && beta.iter().all(|x| x.is_finite() && (0. ..=1.).contains(x))
}

fn vectorizable(q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32]) -> bool {
    let finite = |x: &[f32]| x.iter().fold(true, |ok, v| ok & v.is_finite());
    finite(q)
        && finite(k)
        && finite(v)
        && g.iter().fold(true, |ok, v| ok & v.is_finite() & (*v <= 0.))
        && beta.iter().fold(true, |ok, v| ok & (*v >= 0.) & (*v <= 1.))
}
fn main() {
    let root = std::env::args().nth(1).expect("CAPTURE_DIRECTORY");
    for name in ["m128_zero", "m128_nonzero", "m1_nonzero"] {
        let path = Path::new(&root).join(name);
        let q = read(&path.join("q.f32"));
        let k = read(&path.join("k.f32"));
        let v = read(&path.join("v.f32"));
        let g = read(&path.join("decay.f32"));
        let beta = read(&path.join("beta.f32"));
        assert_eq!(
            valid(&q, &k, &v, &g, &beta),
            vectorizable(&q, &k, &v, &g, &beta)
        );
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad_q = q.clone();
            let last = bad_q.len() - 1;
            bad_q[last] = bad;
            assert!(!vectorizable(&bad_q, &k, &v, &g, &beta));
            let mut bad_g = g.clone();
            bad_g[0] = bad;
            assert!(!vectorizable(&q, &k, &v, &bad_g, &beta));
            let mut bad_b = beta.clone();
            bad_b[0] = bad;
            assert!(!vectorizable(&q, &k, &v, &g, &bad_b));
        }
        let loops = if q.len() == 2048 { 10000 } else { 100 };
        let mut original = Vec::new();
        let mut folded = Vec::new();
        for round in 0..8 {
            for j in 0..2 {
                let which = if round % 2 == 0 { j } else { 1 - j };
                let check = if which == 0 { valid } else { vectorizable };
                let start = Instant::now();
                for _ in 0..loops {
                    assert!(black_box(check(
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        black_box(&g),
                        black_box(&beta)
                    )));
                }
                let us = start.elapsed().as_secs_f64() * 1e6 / loops as f64;
                if which == 0 {
                    original.push(us);
                } else {
                    folded.push(us);
                }
            }
        }
        println!(
            "{{\"fixture\":\"{name}\",\"original_us\":{original:?},\"folded_us\":{folded:?}}}"
        );
    }
}
