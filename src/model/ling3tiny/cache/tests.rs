use super::*;
use crate::{model::CheckpointOptions, validation::BoardLock};
use std::path::PathBuf;

fn state(model: &Ling3Tiny) -> Vec<mixer::Snapshot> {
    model
        .layers
        .iter()
        .filter_map(|l| l.mixer.snapshot())
        .collect()
}

fn prepared_workspaces(model: &Ling3Tiny, rows: usize) -> Vec<[(usize, usize); 4]> {
    let workspaces: Vec<_> = model
        .layers
        .iter()
        .filter_map(|l| l.mixer.prepared_workspaces())
        .collect();
    assert!(!workspaces.is_empty());
    assert!(workspaces.iter().flatten().all(|&(m, _)| m == rows));
    workspaces
}

fn exact(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let differing = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    assert_eq!(differing, 0, "max_abs={max_abs}");
}

#[test]
#[ignore = "requires RK3588 and LING_SSM_MODEL"]
fn ling_prefix_restore_matches_replay_and_preserves_kda_convolution_and_mla() -> Result<()> {
    let _lock = BoardLock::acquire()?;
    let mut options = ModelOptions {
        path: PathBuf::from(std::env::var("LING_SSM_MODEL")?),
        context: 640,
        chunk: 128,
        prefill_ubatches: 1,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: false,
        tune_search: false,
        compress_w8a8: false,
        tune_cache: PathBuf::new(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        checkpoints: CheckpointOptions {
            capacity: 4,
            min_step: 0,
        },
        experts: crate::model::experts::Options {
            storage: if std::env::var_os("LING_SSM_STREAM").is_some() {
                crate::model::experts::Storage::Stream
            } else {
                crate::model::experts::Storage::Resident
            },
            cache: std::env::var_os("LING_SSM_EXPERT_STORE").map(PathBuf::from),
            prefetch: false,
            direct: false,
            ..Default::default()
        },
        speculation: Default::default(),
    };
    crate::npu::configure_cpu(&mut options)?;
    let codec = crate::text::TextCodec::load(&options.path)?;
    let chat = codec.prepare(
        &crate::engine::Prompt::Chat {
            messages: vec![
                serde_json::json!({"role":"system","content":"请保留记录。"}),
                serde_json::json!({"role":"user","content":"我的数字是17。"}),
                serde_json::json!({"role":"assistant","content":"记住了。"}),
                serde_json::json!({"role":"tool","content":"工具记录。"}),
                serde_json::json!({"role":"user","content":"现在是多少？"}),
            ],
            tools: None,
            enable_thinking: false,
        },
        true,
    )?;
    assert_eq!(
        chat.user_starts.len(),
        2,
        "Bailing user boundaries must exclude tools"
    );
    let mut model = Ling3Tiny::load(options)?;
    let cancel = AtomicBool::new(false);
    let mut input = PreparedInput::text((0..600).map(|i| ((i * 17 + 31) % 128) as u32).collect());
    input.user_starts = vec![20, 350];
    assert!(model.uses_user_boundaries());
    let prepared = model.prepare_prepared(&input, 0)?;
    assert_eq!(prepared.steps[0].range.len(), 20);
    let first = prepared.steps[0].range.clone();
    let first_workspaces = prepared_workspaces(&model, 20);
    model.chunk(&input.tokens[first], Some(&cancel))?;
    assert_eq!(
        prepared_workspaces(&model, 20),
        first_workspaces,
        "first planned FP16 batch should already be prepared"
    );
    model.reset()?;
    let prepared = model.prepare_prepared(&input, 0)?;
    let expected = model.forward_prepared(prepared, &cancel)?.to_vec();
    let expected_state = state(&model);
    assert_eq!(model.prefix_cache.positions(), [20, 350, 468, 596]);
    let stats = model.stats();
    assert!(model.prefix_cache.bytes() > 0);
    let mut expected_decode = Vec::new();
    for token in [42, 43, 44] {
        expected_decode.extend_from_slice(model.forward(&[token])?);
    }
    assert_eq!(model.restore_prefix(599)?, 596);
    assert_eq!(
        model.stats()["attention_cache_payload_bytes"],
        stats["attention_cache_payload_bytes"]
    );
    let prepared = model.prepare_prepared(&input, 596)?;
    assert_eq!(prepared.steps.len(), 1);
    assert_eq!(prepared.steps[0].range.len(), 4);
    let replay_workspaces = prepared_workspaces(&model, 4);
    exact(model.forward_prepared(prepared, &cancel)?, &expected);
    assert_eq!(
        prepared_workspaces(&model, 4),
        replay_workspaces,
        "four-row replay should reuse its prepared workspaces"
    );
    assert!(
        state(&model) == expected_state,
        "KDA or convolution changed on replay"
    );
    let mut actual_decode = Vec::new();
    for token in [42, 43, 44] {
        actual_decode.extend_from_slice(model.forward(&[token])?);
    }
    exact(&actual_decode, &expected_decode);
    assert!(!model.truncate(599)?);
    assert!(model.truncate(596)?);
    assert!(model.truncate(597).is_err());
    assert!(model.restore_prefix(597).is_err());
    assert_eq!(model.position(), 596);

    let mut branch = input.clone();
    branch.tokens[500..].fill(2);
    assert_eq!(model.restore_prefix(500)?, 468);
    assert_eq!(model.prefix_cache.positions(), [20, 350, 468]);
    let prepared = model.prepare_prepared(&branch, 468)?;
    let branched = model.forward_prepared(prepared, &cancel)?.to_vec();
    let branched_state = state(&model);
    model.reset()?;
    let prepared = model.prepare_prepared(&branch, 0)?;
    exact(model.forward_prepared(prepared, &cancel)?, &branched);
    assert!(
        state(&model) == branched_state,
        "branch replay changed recurrent state"
    );

    // Reusing a snapshot at the current checkpoint must not evict the earlier
    // one or take another copy, even when the cache is full.
    model.reset()?;
    model.prefix_cache.settings.capacity = 2;
    let plain = PreparedInput::text(input.tokens.clone());
    let prepared = model.prepare_prepared(&plain, 0)?;
    let plain_expected = model.forward_prepared(prepared, &cancel)?.to_vec();
    assert_eq!(model.prefix_cache.positions(), [468, 596]);
    let saves = model.prefix_cache.saves;
    let snapshot = model.prefix_cache.entries[1].state.recurrent.as_ptr();
    assert_eq!(model.restore_prefix(599)?, 596);
    let prepared = model.prepare_prepared(&plain, 596)?;
    exact(model.forward_prepared(prepared, &cancel)?, &plain_expected);
    assert_eq!(model.prefix_cache.positions(), [468, 596]);
    assert_eq!(model.prefix_cache.saves, saves);
    assert_eq!(
        model.prefix_cache.entries[1].state.recurrent.as_ptr(),
        snapshot
    );

    assert_eq!(model.restore_prefix(10)?, 0);
    assert_eq!(model.position(), 600); // A miss lets the engine decide to reset.
    assert_eq!(model.restore_prefix(500)?, 468);
    let prepared = model.prepare_prepared(&plain, 468)?;
    cancel.store(true, Ordering::Relaxed);
    assert!(model.forward_prepared(prepared, &cancel).is_err());
    assert_eq!(model.position(), 0);
    assert_eq!(model.prefix_cache.count(), 0);
    cancel.store(false, Ordering::Relaxed);
    let prepared = model.prepare_prepared(&plain, 0)?;
    model.forward_prepared(prepared, &cancel)?;
    assert_eq!(model.restore_prefix(500)?, 468);
    let prepared = model.prepare_prepared(&plain, 468)?;
    model.reset()?;
    assert!(model.forward_prepared(prepared, &cancel).is_err());
    assert_eq!(model.position(), 0);

    model.prefix_cache.settings.capacity = 0;
    assert!(!model.uses_user_boundaries());
    let prepared = model.prepare_prepared(&plain, 0)?;
    assert!(prepared.steps.iter().all(|s| !s.checkpoint));
    model.forward_prepared(prepared, &cancel)?;
    assert_eq!(model.prefix_cache.count(), 0);
    assert_eq!(model.restore_prefix(599)?, 0);
    assert!(model.truncate(0)?);
    eprintln!(
        "Ling SSM prefix PASS: exact prompt/decode/branch logits, KDA+conv, resident MLA allocation, capacity/reuse/miss/reset/cancellation/disabled"
    );
    Ok(())
}
