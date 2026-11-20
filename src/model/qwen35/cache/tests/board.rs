use super::*;
use crate::{
    engine::Prompt,
    model::{ModelOptions, SpeculativeType},
    text::TextCodec,
    validation::BoardLock,
};
use std::{path::PathBuf, sync::atomic::AtomicBool};

fn state(model: &Qwen35) -> Vec<f32> {
    model
        .layers
        .iter()
        .flat_map(|l| match &l.mixer {
            Mixer::Recurrent { state, .. } => state
                .canonical()
                .iter()
                .chain(state.convolution())
                .copied()
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}

fn exact(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let differing = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    assert_eq!(differing, 0, "max_abs={maximum}");
}

fn board_options(kind: SpeculativeType) -> Result<ModelOptions> {
    let mut options = ModelOptions {
        path: PathBuf::from(std::env::var("RKNPU_MTP_MODEL")?),
        context: 1024,
        chunk: 128,
        prefill_ubatches: std::env::var("RKNPU_PREFILL_UBATCHES")
            .unwrap_or_else(|_| "1".into())
            .parse()?,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: false,
        tune_search: false,
        compress_w8a8: false,
        tune_cache: PathBuf::new(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        checkpoints: Default::default(),
        experts: Default::default(),

        speculation: crate::model::SpeculativeConfig {
            kind,
            max_draft: 3,
            model: std::env::var_os("RKNPU_PREFIX_DRAFT").map(PathBuf::from),
            ..Default::default()
        },
    };
    if let Some(cache) = std::env::var_os("RKNPU_MTP_TUNE_CACHE") {
        options.tune = true;
        options.tune_cache = cache.into();
    }
    if let Some(store) = std::env::var_os("RKNPU_PREFIX_EXPERT_STORE") {
        options.experts.cache = Some(store.into());
    }
    if std::env::var_os("RKNPU_PREFIX_STREAM").is_some() {
        options.experts.storage = crate::model::experts::Storage::Stream;
    }
    crate::npu::configure_cpu(&mut options)?;
    Ok(options)
}

#[test]
#[ignore = "requires RK3588 and RKNPU_MTP_MODEL"]
fn prefix_restore_matches_all_logits_and_draft_state() -> Result<()> {
    let _lock = BoardLock::acquire()?;
    let kind = match std::env::var("RKNPU_PREFIX_SPEC").as_deref() {
        Ok("mtp") => SpeculativeType::DraftMtp,
        Ok("dflash") => SpeculativeType::DraftDflash,
        _ => SpeculativeType::None,
    };
    let options = board_options(kind)?;
    let codec = TextCodec::load(&options.path)?;
    let mut model = Qwen35::load(options)?;
    if kind == SpeculativeType::DraftMtp {
        model.enable_mtp(3)?;
    }
    let input = codec.prepare(&Prompt::Chat {
        messages: vec![serde_json::json!({"role":"system","content":"请简洁回答，保留上下文中的事实。"}),
            serde_json::json!({"role":"user","content":format!("{}\n问题：最后一个数字是什么？", "记录：北京，上海，广州，数字 17。\n".repeat(22))})],
        tools: None, enable_thinking: false,
    }, model.uses_user_boundaries())?;
    assert!(
        !input.user_starts.is_empty(),
        "system prefix boundary missing"
    );
    let cancel = AtomicBool::new(false);
    let prepared = model.prepare_prepared(&input, 0)?;
    if model.paging.is_some() && model.mtp.is_none() {
        assert!(
            prepared
                .steps
                .iter()
                .any(|step| step.range.len() > model.options.chunk),
            "streaming prepared input did not retain a logical prompt block"
        );
    }
    let expected = model.forward_prepared(prepared, &cancel)?.to_vec();
    let expected_positions = vec![
        input.user_starts[0],
        input.tokens.len() - 132,
        input.tokens.len() - 4,
    ];
    assert_eq!(model.prefix_cache.positions(), expected_positions);
    eprintln!(
        "initial checkpoint positions={expected_positions:?} prompt_tokens={}",
        input.tokens.len()
    );
    let expected_state = state(&model);
    let rope = model.next_rope_position;
    let draft = |model: &mut Qwen35| -> Result<Vec<f32>> {
        match kind {
            SpeculativeType::DraftDflash => Ok(model.draft_block_logits(42, 3, &cancel)?.unwrap()),
            SpeculativeType::DraftMtp => Ok(model.draft_logits(42, &cancel)?.to_vec()),
            _ => Ok(Vec::new()),
        }
    };
    let expected_draft = draft(&mut model)?;
    model.abort_verification()?;
    model.forward(&[42, 43, 44])?;
    let restored = model.restore_prefix(input.tokens.len() - 1)?;
    assert_eq!(restored, input.tokens.len() - 4);
    let prepared = model.prepare_prepared(&input, restored)?;
    exact(model.forward_prepared(prepared, &cancel)?, &expected);
    exact(&state(&model), &expected_state);
    assert_eq!(model.next_rope_position, rope);
    exact(&draft(&mut model)?, &expected_draft);
    model.abort_verification()?;

    let mut branch = input.clone();
    let shared = branch.tokens.len() - 100;
    for token in &mut branch.tokens[shared..] {
        *token = 100;
    }
    let restored = model.restore_prefix(shared)?;
    assert_eq!(restored, branch.tokens.len() - 132);
    assert!(
        model
            .prefix_cache
            .entries
            .iter()
            .all(|c| c.position <= restored)
    );
    let prepared = model.prepare_prepared(&branch, restored)?;
    let branched = model.forward_prepared(prepared, &cancel)?.to_vec();
    let branched_state = state(&model);
    model.reset()?;
    let prepared = model.prepare_prepared(&branch, 0)?;
    exact(model.forward_prepared(prepared, &cancel)?, &branched);
    exact(&state(&model), &branched_state);
    assert!(model.prefix_cache.count() <= model.options.checkpoints.capacity);
    eprintln!(
        "prefix restore PASS mode={} logits={} branch_restore={} checkpoint_bytes={}",
        kind.as_str(),
        expected.len(),
        restored,
        model.prefix_cache.bytes()
    );
    let npu_gdn = model.stats()["gdn_npu_workspace_rows"].is_array();
    if npu_gdn {
        planned_tail_workspaces_survive_prefill(&mut model, &cancel)?;
    }
    for cpu in [false, true].into_iter().filter(|cpu| npu_gdn || *cpu) {
        if cpu {
            model.work.recurrent = crate::layers::gated_delta::Scratch::default();
            if let Some(work) = &mut model.secondary_work {
                work.recurrent = crate::layers::gated_delta::Scratch::default();
            }
        }
        restored_tail_keeps_earlier_checkpoint(&mut model, &cancel)?;
        eprintln!(
            "checkpoint reuse PASS mode={} gdn={} positions=[468,596] branch_restore=468",
            kind.as_str(),
            if cpu { "cpu-fp32" } else { "npu-tf32" }
        );
    }
    Ok(())
}

fn planned_tail_workspaces_survive_prefill(model: &mut Qwen35, cancel: &AtomicBool) -> Result<()> {
    for (users, expected) in [
        (vec![20, 350], vec![20, 74, 118, 128]),
        (vec![20, 200], vec![20, 52, 128]),
        (vec![], vec![84, 128]),
    ] {
        model.reset()?;
        let mut input = crate::input::PreparedInput::text(vec![1; 600]);
        input.user_starts = users;
        let prepared = model.prepare_prepared(&input, 0)?;
        let before = model.stats();
        assert_eq!(
            before["gdn_npu_workspace_rows"],
            serde_json::json!(expected)
        );
        let creations = before["gdn_npu_workspace_creations"].as_u64().unwrap();
        model.forward_prepared(prepared, cancel)?;
        let after = model.stats();
        assert!(after["gdn_npu_calls"].as_u64().unwrap() > 0);
        assert_eq!(after["gdn_npu_workspace_creations"], creations);
        eprintln!("planned GDN workspaces PASS rows={expected:?} prefill_creations=0");
    }
    Ok(())
}

#[test]
#[ignore = "requires RK3588 and RKNPU_MTP_MODEL pointing to Qwen3.5-4B C7"]
fn prefill_cache_retention_and_secondary_pairing() -> Result<()> {
    let _lock = BoardLock::acquire()?;
    let mut options = board_options(SpeculativeType::None)?;
    options.prefill_ubatches = 2;
    let expect_retention = match std::env::var("RK_NPU_PREFILL_CACHE_DOMAIN_MIB").as_deref() {
        Ok("0") => false,
        Ok("1792") | Err(std::env::VarError::NotPresent) => true,
        _ => anyhow::bail!("test requires cache budget 0 or 1792 MiB"),
    };
    let mut model = Qwen35::load(options)?;
    let cancel = AtomicBool::new(false);
    let secondary_rows = |model: &Qwen35| {
        model
            .secondary_work
            .as_ref()
            .unwrap()
            .recurrent
            .backend_stats()["gdn_npu_workspace_rows"]
            .clone()
    };
    let check_secondary_empty = |model: &Qwen35| {
        assert_eq!(secondary_rows(model), serde_json::json!([]));
        for backend in model.secondary.as_ref().unwrap().backends() {
            assert_eq!(backend.stats()["workspace_count"], 0);
        }
    };

    model.reserve(384)?;
    model.prepare(129)?;
    check_secondary_empty(&model);
    model.forward(&[1; 129])?;
    assert_eq!(model.dual_groups, 0);

    model.reset()?;
    let mut input = crate::input::PreparedInput::text(vec![1; 260]);
    input.user_starts = vec![128];
    let prepared = model.prepare_prepared(&input, 0)?;
    assert_eq!(prepared.steps[1].range, 128..256);
    assert!(prepared.steps[1].checkpoint);
    check_secondary_empty(&model);
    model.forward_prepared(prepared, &cancel)?;
    assert_eq!(model.dual_groups, 0);

    let creates = |model: &Qwen35| -> u64 {
        model
            .backends
            .iter()
            .map(|backend| {
                backend.stats()["domain_memory"]["creates"]
                    .as_u64()
                    .unwrap()
            })
            .sum()
    };
    let expected_rows = if expect_retention {
        serde_json::json!([128])
    } else {
        serde_json::json!([])
    };
    let mut expected_logits = Vec::new();
    for request in 0..2 {
        model.reset()?;
        let before = creates(&model);
        model.prepare(256)?;
        let allocations = creates(&model) - before;
        let expected_gdn_creates = match (expect_retention, request) {
            (true, 0) => 1,
            (true, _) => 0,
            (false, _) => 2,
        };
        assert_eq!(
            model.stats()["gdn_npu_workspace_creations"],
            expected_gdn_creates
        );
        assert_eq!(allocations == 0, expect_retention && request == 1);
        let logits = model.forward(&[1; 256])?;
        if request == 0 {
            expected_logits = logits.to_vec();
        } else {
            exact(logits, &expected_logits);
        }
        assert_eq!(model.dual_groups, 1);
        assert_eq!(model.stats()["gdn_npu_workspace_rows"], expected_rows);
        assert_eq!(secondary_rows(&model), expected_rows);
        eprintln!(
            "prefill retention={expect_retention} request={request} prepare_creates={allocations} gdn_creates={expected_gdn_creates}"
        );
    }
    let appended = model.forward(&[1; 128])?.to_vec();
    assert_eq!(model.position(), 384);
    assert_eq!(
        model.stats()["prefill_workspace_cached_domains"],
        serde_json::json!([])
    );
    assert_eq!(
        model.stats()["gdn_npu_workspace_rows"],
        serde_json::json!([])
    );
    assert_eq!(secondary_rows(&model), serde_json::json!([]));
    model.reset()?;
    model.prepare(384)?;
    exact(model.forward(&[1; 384])?, &appended);

    model.reset()?;
    model.prepare(256)?;
    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(model.forward_cancellable(&[1; 256], &cancel).is_err());
    assert_eq!(model.position(), 0);
    assert_eq!(
        model.stats()["prefill_workspace_cached_domains"],
        serde_json::json!([])
    );
    assert_eq!(
        model.stats()["gdn_npu_workspace_rows"],
        serde_json::json!([])
    );
    eprintln!("unprepared append and cancelled prefill PASS");
    cancel.store(false, std::sync::atomic::Ordering::Relaxed);
    let paired = crate::input::PreparedInput::text(vec![1; 388]);
    let mut checkpointed = paired.clone();
    checkpointed.user_starts = vec![128];
    let obsolete = model.prepare_prepared(&paired, 0)?;
    let current = model.prepare_prepared(&checkpointed, 0)?;
    let groups = |plan: &crate::model::PreparedPrefill<'_>| {
        crate::model::prefill::groups(&plan.steps, 128, 2).any(|group| group.len() == 2)
    };
    assert!(groups(&obsolete) && !groups(&current));
    let before = creates(&model);
    let error = model.forward_prepared(obsolete, &cancel).unwrap_err();
    assert!(error.to_string().contains("replaced or invalidated"));
    assert_eq!(model.position(), 0);
    assert_eq!(creates(&model), before);
    model.forward_prepared(current, &cancel)?;
    assert_eq!(model.position(), 388);
    assert_eq!(model.dual_groups, 0);
    eprintln!("same-span distinct prepared plans PASS");
    Ok(())
}

fn restored_tail_keeps_earlier_checkpoint(model: &mut Qwen35, cancel: &AtomicBool) -> Result<()> {
    model.reset()?;
    model.prefix_cache.settings = crate::model::CheckpointOptions {
        capacity: 2,
        min_step: 0,
    };
    assert!(model.uses_user_boundaries());
    let input = crate::input::PreparedInput::text(vec![1; 600]);
    let prepared = model.prepare_prepared(&input, 0)?;
    let expected = model.forward_prepared(prepared, cancel)?.to_vec();
    let expected_state = state(model);
    assert_eq!(model.prefix_cache.positions(), [468, 596]);
    let saves = model.prefix_cache.saves;
    let save_ms = model.prefix_cache.save_ms;
    let snapshot = model.prefix_cache.entries[1].state.recurrent.as_ptr();
    assert_eq!(model.restore_prefix(599)?, 596);
    let prepared = model.prepare_prepared(&input, 596)?;
    exact(model.forward_prepared(prepared, cancel)?, &expected);
    exact(&state(model), &expected_state);
    assert_eq!(model.prefix_cache.positions(), [468, 596]);
    assert_eq!(
        model.prefix_cache.entries[1].state.recurrent.as_ptr(),
        snapshot
    );
    assert_eq!(model.prefix_cache.saves, saves);
    assert_eq!(model.prefix_cache.save_ms, save_ms);
    assert_eq!(model.prefix_cache.entries[1].task, model.prefix_cache.task);

    let mut branch = input.clone();
    branch.tokens[500..].fill(2);
    assert_eq!(model.restore_prefix(500)?, 468);
    let prepared = model.prepare_prepared(&branch, 468)?;
    let branched = model.forward_prepared(prepared, cancel)?.to_vec();
    let branched_state = state(model);
    model.reset()?;
    let prepared = model.prepare_prepared(&branch, 0)?;
    exact(model.forward_prepared(prepared, cancel)?, &branched);
    exact(&state(model), &branched_state);

    assert_eq!(model.restore_prefix(599)?, 596);
    let prepared = model.prepare_prepared(&branch, 596)?;
    model.reset()?;
    assert!(model.forward_prepared(prepared, cancel).is_err());
    assert_eq!(model.position(), 0);
    assert_eq!(model.prefix_cache.count(), 0);
    model.prefix_cache.settings.capacity = 0;
    assert!(!model.uses_user_boundaries());
    Ok(())
}
