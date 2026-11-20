use anyhow::{Result, ensure};
use open_rknpu2_lm_infer::{
    model::{self, LanguageModel, ModelOptions, SpeculativeConfig, SpeculativeType},
    validation::BoardLock,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf, sync::atomic::AtomicBool};

use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum Case {
    Dense,
    Prepare,
    Verify,
}

#[derive(Parser)]
struct Args {
    case: Case,
    model: PathBuf,
    tune_cache: PathBuf,
}

type Arm = unsafe extern "C" fn(i32, i32, i32);
type Disarm = unsafe extern "C" fn() -> i32;
struct Fault {
    arm: Arm,
    disarm: Disarm,
}
impl Fault {
    fn load() -> Result<Self> {
        let arm = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"workspace_fault_arm".as_ptr()) };
        let disarm = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"workspace_fault_disarm".as_ptr()) };
        ensure!(
            !arm.is_null() && !disarm.is_null(),
            "run with tests/board/workspace_reclaim.sh"
        );
        // Both symbols are provided by the board test's ioctl interposer.
        Ok(Self {
            arm: unsafe { std::mem::transmute::<*mut libc::c_void, Arm>(arm) },
            disarm: unsafe { std::mem::transmute::<*mut libc::c_void, Disarm>(disarm) },
        })
    }
    fn run<T>(&self, phase: i32, count: i32, op: impl FnOnce() -> Result<T>) -> Result<Result<T>> {
        unsafe { (self.arm)(phase, count, 14) };
        let result = op();
        let injected = unsafe { (self.disarm)() };
        ensure!(
            injected == count,
            "injected={injected}, expected={count}, op={:?}",
            result.as_ref().err()
        );
        Ok(result)
    }
}

fn digest(logits: &[f32]) -> String {
    let mut hash = Sha256::new();
    for &v in logits {
        hash.update(v.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn counters(stats: &Value, key: &str) -> BTreeMap<u32, u64> {
    stats["domains"]
        .as_array()
        .unwrap()
        .iter()
        .map(|domain| {
            (
                domain["iommu_domain_id"].as_u64().unwrap() as u32,
                domain["domain_memory"][key].as_u64().unwrap(),
            )
        })
        .filter(|(id, _)| *id == 14 || *id == 15)
        .collect()
}

fn diff(before: &Value, after: &Value, key: &str) -> BTreeMap<u32, u64> {
    let before = counters(before, key);
    counters(after, key)
        .into_iter()
        .map(|(k, v)| (k, v - before[&k]))
        .collect()
}

fn prefill(m: &mut dyn LanguageModel, prompt: &[u32]) -> Result<String> {
    m.reset()?;
    m.prepare(prompt.len())?;
    Ok(digest(m.forward(prompt)?))
}

struct Outcome {
    initial: String,
    logits: String,
    creates: BTreeMap<u32, u64>,
    destroys: BTreeMap<u32, u64>,
}

fn run_case(args: &Args, fault: &Fault, count: i32) -> Result<Outcome> {
    let tokens = [42; 512];
    let mtp = args.case == Case::Verify;
    let dense = args.case == Case::Dense;
    let mut model = model::load(ModelOptions {
        path: args.model.clone(),
        context: 1024,
        chunk: 128,
        prefill_ubatches: if dense { 1 } else { 2 },
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: true,
        tune_search: false,
        compress_w8a8: false,
        tune_cache: args.tune_cache.clone(),
        attention_tile: 512,
        vision_dtype: Default::default(),
        checkpoints: Default::default(),
        experts: Default::default(),
        speculation: if mtp {
            SpeculativeConfig {
                kind: SpeculativeType::DraftMtp,
                max_draft: 3,
                ..Default::default()
            }
        } else {
            Default::default()
        },
    })?;
    model.reserve(768)?;
    let initial = prefill(model.as_mut(), &tokens)?;
    let initial_stats = model.stats();
    let original_cache: Vec<u32> =
        serde_json::from_value(initial_stats["prefill_workspace_cached_domains"].clone())?;
    ensure!(
        original_cache.contains(&14) && (dense || original_cache == [14, 15]),
        "expected warm cache: {original_cache:?}",
    );
    let expected_cache = json!(
        original_cache
            .iter()
            .copied()
            .filter(|&id| count == 0 || id != 14)
            .collect::<Vec<_>>()
    );
    let cancel = AtomicBool::new(false);
    let (before, after, output) = if args.case == Case::Prepare {
        model.reset()?;
        let before = model.stats();
        let prompt = &tokens[..145];
        fault.run(1, count, || model.prepare(prompt.len()))??;
        let after = model.stats();
        ensure!(
            after["gdn_npu_workspace_rows"] == json!([17, 128]),
            "required GDN rows lost"
        );
        ensure!(
            after["gdn_npu_workspace_creations"] == 1,
            "full GDN recreated"
        );
        let output = digest(model.forward(prompt)?);
        ensure!(
            after["prefill_workspace_cached_domains"] == expected_cache,
            "prepare cache mismatch"
        );
        (before, after, output)
    } else {
        let before = initial_stats;
        let mut execute = || -> Result<String> {
            if mtp {
                Ok(digest(model.verify_tokens(&[43, 44, 45, 46], &cancel)?))
            } else {
                Ok(digest(model.forward(&[43])?))
            }
        };
        let result = fault.run(2, count, &mut execute)?;
        let after = model.stats();
        let output = if count == 2 {
            ensure!(result.is_err(), "a failed retry unexpectedly succeeded");
            eprintln!("persistent allocation error={:#}", result.unwrap_err());
            ensure!(
                prefill(model.as_mut(), &tokens)? == initial,
                "next request prefill changed"
            );
            if mtp {
                digest(model.verify_tokens(&[43, 44, 45, 46], &cancel)?)
            } else {
                digest(model.forward(&[43])?)
            }
        } else {
            result?
        };
        ensure!(
            after["prefill_workspace_cached_domains"] == expected_cache,
            "graph cache mismatch"
        );
        if mtp {
            model.finish_verification(2)?;
            ensure!(model.position() == 514, "verify commit lost KV position");
        } else {
            ensure!(model.position() == 513, "decode lost KV position");
        }
        (before, after, output)
    };
    Ok(Outcome {
        initial,
        logits: output,
        creates: diff(&before, &after, "creates"),
        destroys: diff(&before, &after, "destroys"),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _lock = BoardLock::acquire()?;
    let fault = Fault::load()?;
    let control = run_case(&args, &fault, 0)?;
    let retry = run_case(&args, &fault, 1)?;
    ensure!(
        control.initial == retry.initial && control.logits == retry.logits,
        "logits changed after reclaim"
    );
    ensure!(
        control.creates == retry.creates,
        "unexpected GEM creation after reclaim"
    );
    ensure!(
        control.destroys[&14] < retry.destroys[&14] && control.destroys[&15] == retry.destroys[&15],
        "unexpected domain reclaim counts"
    );
    if args.case == Case::Verify {
        let recovered = run_case(&args, &fault, 2)?;
        ensure!(
            recovered.logits == control.logits,
            "verify logits changed after failed retry"
        );
    }
    println!("{:?} passed", args.case);
    Ok(())
}
