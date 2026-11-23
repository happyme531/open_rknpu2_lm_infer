use super::*;
use crate::{cli::Cli, validation::BoardLock};
use clap::Parser;

#[test]
fn slot_lifecycle_and_whole_step_validation() -> Result<()> {
    let config = BatchConfig {
        slots: 2,
        max_rows: 4,
    };
    let mut pool = SeqPool::new(vec![0, 0], 8, Some(config));
    assert!(pool.default_seq.is_none());
    assert_eq!(pool.try_admit(9), Admission::CannotFit);
    let Admission::Ready(a) = pool.try_admit(8) else {
        panic!("first slot")
    };
    let Admission::Ready(b) = pool.try_admit(8) else {
        panic!("second slot")
    };
    assert_eq!(pool.try_admit(8), Admission::Wait);
    let span = |seq| Span {
        seq,
        start: 0,
        tokens: &[7],
        positions: &[[0; 3]],
        embeddings: &[],
        logits: true,
    };
    let check = |pool: &SeqPool<usize>, spans: &[Span<'_>]| {
        pool.layout(spans, 3, 10, |&pos| pos, |&pos| pos)
            .map(|layout| layout.rows)
    };
    assert_eq!(check(&pool, &[span(a), span(b)]).unwrap(), 2);
    assert!(matches!(
        check(&pool, &[span(a), span(a)]),
        Err(StepError::Rejected(_))
    ));
    assert!(matches!(
        check(
            &pool,
            &[
                span(a),
                Span {
                    start: 1,
                    ..span(b)
                }
            ]
        ),
        Err(StepError::Rejected(_))
    ));
    assert!(matches!(
        check(
            &pool,
            &[
                span(a),
                Span {
                    tokens: &[10],
                    ..span(b)
                }
            ]
        ),
        Err(StepError::Rejected(_))
    ));
    pool.release(a, |pos| {
        *pos = 0;
        Ok(())
    })?;
    let Admission::Ready(c) = pool.try_admit(8) else {
        panic!("returned slot")
    };
    assert_eq!(c.slot, a.slot);
    assert_ne!(c.generation, a.generation);
    assert!(matches!(
        check(&pool, &[span(a)]),
        Err(StepError::Rejected(_))
    ));
    assert_eq!(check(&pool, &[span(c)]).unwrap(), 1);
    assert!(
        pool.release(b, |_| anyhow::bail!("simulated device error"))
            .is_err()
    );
    assert!(matches!(
        check(&pool, &[span(c)]),
        Err(StepError::Device(_))
    ));
    Ok(())
}

#[test]
fn step_commits_only_valid_logits_in_span_order() -> Result<()> {
    let config = BatchConfig {
        slots: 2,
        max_rows: 4,
    };
    let mut pool = SeqPool::new(vec![0, 0], 8, Some(config));
    let Admission::Ready(a) = pool.try_admit(8) else {
        panic!("first slot")
    };
    let Admission::Ready(b) = pool.try_admit(8) else {
        panic!("second slot")
    };
    let span = |seq, logits| Span {
        seq,
        start: 0,
        tokens: &[0, 1],
        positions: &[[0; 3], [1; 3]],
        embeddings: &[],
        logits,
    };
    let spans = [span(b, false), span(a, true)];
    let layout = pool.layout(&spans, 4, 2, |&pos| pos, |&pos| pos).unwrap();
    assert_eq!(layout.spans[0].rows, 0..2);
    assert_eq!(layout.spans[1].rows, 2..4);
    let advance = |pos: &mut usize, span: &Span<'_>| *pos += span.tokens.len();
    let out = pool
        .complete_step(layout, Ok(()), &[1., 2.], 2, advance)
        .unwrap();
    assert_eq!(out.logits_rows().collect::<Vec<_>>(), [(a, &[1., 2.][..])]);
    assert_eq!(pool.slots, [2, 2]);

    let spans = [Span {
        start: 2,
        positions: &[[2; 3], [3; 3]],
        ..span(b, true)
    }];
    let layout = pool.layout(&spans, 4, 2, |&pos| pos, |&pos| pos).unwrap();
    assert!(matches!(
        pool.complete_step(layout, Ok(()), &[], 2, advance),
        Err(StepError::Device(_))
    ));
    assert_eq!(pool.slots, [2, 2]);
    assert!(matches!(
        pool.layout(&spans, 4, 2, |&pos| pos, |&pos| pos),
        Err(StepError::Device(_))
    ));
    Ok(())
}

fn ready(model: &mut dyn BatchModel) -> SeqId {
    let Admission::Ready(seq) = model.try_admit(512) else {
        panic!("admission")
    };
    seq
}

fn positions(start: usize, len: usize) -> Vec<[usize; 3]> {
    (start..start + len).map(|pos| [pos; 3]).collect()
}

fn step(
    model: &mut dyn BatchModel,
    spans: &[(SeqId, usize, &[u32], bool)],
) -> Result<Vec<(SeqId, Vec<f32>)>> {
    let rope: Vec<_> = spans
        .iter()
        .map(|(_, start, tokens, _)| positions(*start, tokens.len()))
        .collect();
    let spans: Vec<_> = spans
        .iter()
        .zip(&rope)
        .map(|(&(seq, start, tokens, logits), positions)| Span {
            seq,
            start,
            tokens,
            positions,
            logits,
            embeddings: &[],
        })
        .collect();
    let out = model
        .execute_step(&spans)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(out
        .logits_rows()
        .map(|(seq, values)| (seq, values.to_vec()))
        .collect())
}

fn exact(a: &[f32], b: &[f32], label: &str) -> Result<()> {
    ensure!(a.len() == b.len(), "{label}: different logits shapes");
    if let Some((i, (a, b))) = a
        .iter()
        .zip(b)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits())
    {
        anyhow::bail!("{label}: logit {i}: {a} != {b}");
    }
    Ok(())
}

fn compare(reference: &[f32], actual: &[f32]) -> (f32, f64, bool) {
    let max_index = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let reference_top = max_index(reference);
    let actual_top = max_index(actual);
    let log_z = |values: &[f32], peak: usize| {
        let max = f64::from(values[peak]);
        max + values
            .iter()
            .map(|&v| (f64::from(v) - max).exp())
            .sum::<f64>()
            .ln()
    };
    let (a, b) = (log_z(reference, reference_top), log_z(actual, actual_top));
    let mut kl = 0.;
    let mut max_abs: f32 = 0.;
    for (&x, &y) in reference.iter().zip(actual) {
        max_abs = max_abs.max((x - y).abs());
        let log_p = f64::from(x) - a;
        kl += log_p.exp() * (log_p - (f64::from(y) - b));
    }
    (max_abs, kl.max(0.), reference_top == actual_top)
}

#[test]
#[ignore = "requires RK3588 and RKNPU_BATCH_MODEL"]
fn slot_batch_logits_against_independent_sequences() -> Result<()> {
    let _lock = BoardLock::acquire()?;
    let model_dir = std::env::var("RKNPU_BATCH_MODEL")?;
    let cli = Cli::try_parse_from([
        "infer",
        "-m",
        &model_dir,
        "--ctx-size",
        "512",
        "--ubatch-size",
        "128",
        "--no-tune",
        "run",
    ])?;
    let options = cli.model.options()?;
    let history = [31, 63, 95, 127];
    let prompts: Vec<Vec<u32>> = history
        .iter()
        .enumerate()
        .map(|(slot, &len)| (1000 + slot as u32 * 521..).take(len).collect())
        .collect();
    let teacher: Vec<[u32; 3]> = (0..4)
        .map(|slot| [32 + slot, 70 + slot, 117 + slot])
        .collect();
    let mixed_tail: Vec<u32> = (3100..3123).collect();
    let short_prompt = [4099, 4100, 4101];
    let mut expected_decode = Vec::new();
    let mut expected_mixed = Vec::new();
    let mut expected_reuse = Vec::new();
    let expected_short = {
        let mut serial_options = options.clone();
        if std::env::var("RKNPU_BATCH_SERIAL_UBATCHES").as_deref() == Ok("2") {
            // The fixture fits in one ubatch. Two-lane loading uses smaller IOVA domains.
            serial_options.prefill_ubatches = 2;
        }
        let mut serial = crate::model::load(serial_options)?;
        for (prompt, decode) in prompts.iter().zip(&teacher) {
            serial.reset()?;
            serial.reserve(512)?;
            serial.prepare(prompt.len())?;
            serial.prefill(prompt)?;
            let mut logits = Vec::new();
            for token in &decode[..2] {
                logits.push(serial.forward(&[*token])?.to_vec());
            }
            expected_decode.push(logits);
            expected_mixed.push(serial.forward(&decode[2..])?.to_vec());
            expected_reuse.push(serial.forward(&mixed_tail)?.to_vec());
        }
        serial.reset()?;
        serial.reserve(512)?;
        serial.prepare(short_prompt.len())?;
        serial.forward(&short_prompt)?.to_vec()
    };
    for width in [2, 4] {
        let mut model = super::load(
            options.clone(),
            BatchConfig {
                slots: width,
                max_rows: 128,
            },
        )?;
        let initialized = model.stats();
        let reserved = &initialized["kv_payload_bytes"];
        let workspaces = &initialized["workspace_payload_bytes"];
        let ids: Vec<_> = (0..width).map(|_| ready(model.as_mut())).collect();
        for (&seq, prompt) in ids.iter().zip(&prompts) {
            assert!(step(model.as_mut(), &[(seq, 0, prompt, false)])?.is_empty());
        }
        for tick in 0..2 {
            let inputs: Vec<_> = (0..width)
                .rev()
                .map(|slot| {
                    (
                        ids[slot],
                        history[slot] + tick,
                        &teacher[slot][tick..tick + 1],
                        true,
                    )
                })
                .collect();
            let logits = step(model.as_mut(), &inputs)?;
            assert_eq!(logits.len(), width);
            for (id, values) in &logits {
                let slot = usize::from(id.slot);
                exact(
                    &expected_decode[slot][tick],
                    values,
                    &format!("B={width} slot={slot} tick={tick}"),
                )?;
            }
        }
        eprintln!("B={width}: both decode steps bit-exact across all logit rows");
        let last = width - 1;
        let mut expected_prefill = None;
        for cycle in 0..3 {
            let dead = if cycle == 0 {
                ids[last]
            } else {
                expected_prefill.unwrap()
            };
            model.release(dead)?;
            let new = ready(model.as_mut());
            assert_eq!(new.slot, dead.slot);
            assert_ne!(new.generation, dead.generation);
            assert!(step(model.as_mut(), &[(new, 0, &prompts[last], false)])?.is_empty());
            for tick in 0..3 {
                let values = step(
                    model.as_mut(),
                    &[(
                        new,
                        history[last] + tick,
                        &teacher[last][tick..tick + 1],
                        true,
                    )],
                )?;
                let expected = if tick < 2 {
                    &expected_decode[last][tick]
                } else {
                    &expected_mixed[last]
                };
                exact(expected, &values[0].1, "slot reused after reset")?;
            }
            expected_prefill = Some(new);
        }
        let last_seq = expected_prefill.unwrap();
        let mut inputs: Vec<_> = (0..last)
            .map(|slot| (ids[slot], history[slot] + 2, &teacher[slot][2..], true))
            .collect();
        inputs.push((last_seq, history[last] + 3, mixed_tail.as_slice(), true));
        let actual = step(model.as_mut(), &inputs)?;
        assert_eq!(actual.len(), width);
        let mut max_abs: f32 = 0.;
        let mut max_kl: f64 = 0.;
        let mut top1 = 0;
        for (id, values) in &actual {
            let slot = usize::from(id.slot);
            let expected = if *id == last_seq {
                &expected_reuse[slot]
            } else {
                &expected_mixed[slot]
            };
            let (abs, kl, same) = compare(expected, values);
            max_abs = max_abs.max(abs);
            max_kl = max_kl.max(kl);
            top1 += usize::from(same);
        }
        eprintln!(
            "B={width}: mixed M={} max_abs={max_abs:.8} KL_max={max_kl:.8} top1={top1}/{width}",
            last + mixed_tail.len()
        );
        ensure!(
            max_abs == 0. && max_kl == 0. && top1 == width,
            "mixed logits differ from independent sequences (max_abs={max_abs}, KL={max_kl}, top1={top1}/{width})"
        );
        model.release(last_seq)?;
        let reused = ready(model.as_mut());
        assert_eq!(last_seq.slot, reused.slot);
        assert_ne!(last_seq.generation, reused.generation);
        let short = step(model.as_mut(), &[(reused, 0, &short_prompt, true)])?;
        assert_eq!(short.len(), 1);
        exact(
            &expected_short,
            &short[0].1,
            "short prompt in recycled slot",
        )?;
        model.release(reused)?;
        for &seq in &ids[..last] {
            model.release(seq)?;
        }
        let final_stats = model.stats();
        assert_eq!(&final_stats["kv_payload_bytes"], reserved);
        assert_eq!(&final_stats["workspace_payload_bytes"], workspaces);
    }
    Ok(())
}
