use super::*;
use crate::{
    model::{ModelOptions, checkpoint::TensorStore},
    validation::BoardLock,
};
use std::path::PathBuf;

fn read(path: &std::path::Path, name: &str) -> Result<Vec<f32>> {
    Ok(std::fs::read(path.join(format!("{name}.f32")))?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn difference(a: &[f32], b: &[f32]) -> (f64, f32) {
    assert_eq!(a.len(), b.len());
    let error: f64 = a
        .iter()
        .zip(b)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum();
    let energy: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
    (
        (error / energy.max(1e-30)).sqrt(),
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max),
    )
}

#[test]
#[ignore = "requires RK3588 and RKNPU_MTP_MODEL / RKNPU_MTP_REFERENCE"]
fn official_fp32_mtp_and_coordinate_adapter() -> Result<()> {
    let _lock = BoardLock::acquire()?;
    let path = PathBuf::from(std::env::var("RKNPU_MTP_MODEL")?);
    let fixture = PathBuf::from(std::env::var("RKNPU_MTP_REFERENCE")?);
    let raw = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    let c = Config::parse(&raw)?;
    let mut options = ModelOptions {
        path,
        context: 128,
        chunk: 32,
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
    crate::npu::configure_cpu(&mut options)?;
    let store = TensorStore::open(&options.path)?;
    let cpu = Cpu::new(4)?;
    let mut m = Predictor::load(&options, &c, &store, 15, 7)?;
    let h = c.hidden_size;
    let positions: Vec<_> = (0..8).map(|p| [p; 3]).collect();
    let mut hidden = read(&fixture, "target_hidden")?;
    let mut embedding = read(&fixture, "embedding")?;
    let adapter = m.adapter.as_ref().unwrap();
    adapter.target_hidden(&mut hidden);
    adapter.embedding(&mut embedding);
    let hidden_diff = difference(&read(&fixture, "original_hidden")?, &hidden);
    let embedding_diff = difference(&read(&fixture, "original_embedding")?, &embedding);
    ensure!(
        hidden_diff.0 < 1e-6 && embedding_diff.0 < 1e-6,
        "C7 input basis differs from dense PyTorch oracle"
    );
    let trace = Trace::from_environment()?;
    m.run(&c, &cpu, &hidden, &embedding, &positions, &trace)?;
    let full = m.normalized.clone();
    let original_diff = difference(&read(&fixture, "output")?, &full);
    let mut projected = full.clone();
    m.adapter.as_ref().unwrap().head_input(&mut projected);
    let head_diff = difference(&read(&fixture, "head")?, &projected);
    eprintln!("full MTP output={original_diff:?} head={head_diff:?}");
    ensure!(
        original_diff.0 < 0.005 && head_diff.0 < 0.005,
        "MTP full-batch reference gate failed"
    );
    m.reset()?;
    let mut cached = Vec::new();
    for (r, p) in positions.iter().enumerate() {
        m.run(
            &c,
            &cpu,
            &hidden[r * h..(r + 1) * h],
            &embedding[r * h..(r + 1) * h],
            &[*p],
            &trace,
        )?;
        cached.extend_from_slice(&m.normalized);
    }
    let cached_diff = difference(&full, &cached);
    eprintln!(
        "{}",
        serde_json::json!({"adapter_hidden":hidden_diff,"adapter_embedding":embedding_diff,
        "official_fp32_output":original_diff,"shared_head_input":head_diff,"cached_vs_full":cached_diff})
    );
    ensure!(
        original_diff.0 < 0.005 && head_diff.0 < 0.005 && cached_diff.0 < 0.001,
        "MTP FP16 network or shifted cache failed numerical gate"
    );
    // Canonical prefix repair may skip attention/output/FFN because a single
    // MTP layer's cache only depends on FC -> input norm -> K/V projections.
    let target = read(&fixture, "target_hidden")?;
    let embeddings = read(&fixture, "embedding")?;
    let mut original_h = target.clone();
    let mut original_e = embeddings.clone();
    m.adapter.as_ref().unwrap().target_hidden(&mut original_h);
    m.adapter.as_ref().unwrap().embedding(&mut original_e);
    m.reset()?;
    m.run(
        &c,
        &cpu,
        &original_h[..7 * h],
        &original_e[h..],
        &positions[..7],
        &trace,
    )?;
    m.seed = original_h[7 * h..].to_vec();
    m.seed_rope = [7; 3];
    m.canonical = 7;
    let table = store.get(&format!("{}.embed_tokens.weight", c.text_prefix))?;
    let expected_next = m.draft(&c, &cpu, &table, 42, &trace)?;
    m.reset()?;
    m.catch_up(&c, &cpu, &embeddings, &target, &positions, &trace)?;
    let actual_next = m.draft(&c, &cpu, &table, 42, &trace)?;
    ensure!(
        expected_next == actual_next,
        "KV-only prompt repair changed the next MTP output"
    );
    Ok(())
}
