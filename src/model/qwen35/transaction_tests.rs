use super::*;
use crate::{model::ModelOptions, validation::BoardLock};
use std::path::PathBuf;

fn snapshot(m: &Qwen35) -> Vec<f32> {
    m.layers
        .iter()
        .flat_map(|l| match &l.mixer {
            Mixer::Recurrent { state, .. } => state
                .canonical()
                .iter()
                .chain(state.convolution())
                .copied()
                .collect(),
            _ => Vec::new(),
        })
        .collect()
}
fn near(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let error: f64 = a
        .iter()
        .zip(b)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum();
    let energy: f64 = a.iter().map(|a| (*a as f64).powi(2)).sum();
    let relative = (error / energy.max(1e-30)).sqrt();
    eprintln!(
        "target comparison nrmse={relative} max_abs={}",
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max)
    );
    assert!(
        relative < 0.001,
        "target logits differ after commit/abort: {relative}"
    );
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(argmax(a), argmax(b));
}
#[test]
#[ignore = "requires RK3588 and RKNPU_MTP_MODEL"]
fn complete_target_commit_abort_and_next_output() -> Result<()> {
    let _lock = BoardLock::acquire()?;
    let mut options = ModelOptions {
        path: PathBuf::from(std::env::var("RKNPU_MTP_MODEL")?),
        context: 128,
        chunk: 128,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: false,
        tune_search: false,
        tune_cache: PathBuf::new(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        experts: Default::default(),
        speculation: Default::default(),
    };
    if let Some(path) = std::env::var_os("RKNPU_MTP_TUNE_CACHE") {
        options.tune = true;
        options.tune_cache = path.into();
    }
    crate::npu::configure_cpu(&mut options)?;
    let mut m = Qwen35::load(options)?;
    let cancel = AtomicBool::new(false);
    let prompt = [1, 42, 128, 5, 17];
    let tokens = [21, 3, 7, 100, 2, 25, 20, 6];
    if let Some(root) = std::env::var_os("RKNPU_MTP_TRACE") {
        let root = PathBuf::from(root);
        m.trace = crate::model::trace::Trace::in_directory(root.join("serial"))?;
        m.forward(&prompt)?;
        let expected = m.forward(&tokens[..1])?.to_vec();
        m.reset()?;
        m.trace = crate::model::trace::Trace::in_directory(root.join("verify"))?;
        m.forward(&prompt)?;
        let actual = m.verify_tokens(&tokens, &cancel)?.to_vec();
        near(&expected, &actual[..m.info.vocab_size]);
        return Ok(());
    }
    let mut expected = Vec::new();
    let mut next = Vec::new();
    for consumed in 1..=8 {
        eprintln!("baseline consumed={consumed}");
        m.reset()?;
        m.forward(&prompt)?;
        let mut row = Vec::new();
        for t in &tokens[..consumed] {
            row = m.forward(&[*t])?.to_vec();
        }
        expected.push(row);
        next.push(m.forward(&[19])?.to_vec());
    }
    for consumed in 1..=8 {
        eprintln!("verify consumed={consumed}");
        m.reset()?;
        m.forward(&prompt)?;
        let before = snapshot(&m);
        let verified = m.verify_tokens(&tokens, &cancel)?.to_vec();
        assert_eq!(m.position(), prompt.len());
        assert_eq!(
            snapshot(&m),
            before,
            "verification modified canonical GDN state"
        );
        near(
            &expected[consumed - 1],
            &verified[(consumed - 1) * m.info.vocab_size..consumed * m.info.vocab_size],
        );
        m.finish_verification(consumed)?;
        assert_eq!(m.position(), prompt.len() + consumed);
        near(&next[consumed - 1], m.forward(&[19])?);
    }
    m.reset()?;
    m.forward(&prompt)?;
    let before = snapshot(&m);
    for _ in 0..3 {
        m.verify_tokens(&tokens, &cancel)?;
        m.abort_verification()?;
        assert_eq!(snapshot(&m), before);
        assert_eq!(m.position(), prompt.len());
    }
    assert!(m.verify_tokens(&tokens, &AtomicBool::new(true)).is_err());
    assert_eq!(snapshot(&m), before);
    m.forward(&tokens[..1])?;
    near(&next[0], m.forward(&[19])?);
    Ok(())
}
