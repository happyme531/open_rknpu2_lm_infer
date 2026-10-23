use anyhow::Result;
use clap::Parser;
use serde_json::{Value, json};
use std::process::Command;

fn binary() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_open_rknpu2_lm_infer"));
    command.env_remove("RKNPU_MODEL");
    command
}

#[test]
fn compiled_model_list_needs_no_checkpoint_or_device() -> Result<()> {
    let output = binary().arg("models").output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(actual, open_rknpu2_lm_infer::model::build_info());
    assert!(!actual["model_families"].as_array().unwrap().is_empty());
    Ok(())
}

#[test]
fn cli_identity_matches_the_compiled_project_name() -> Result<()> {
    let name = open_rknpu2_lm_infer::PROJECT_NAME;
    let version = binary().arg("--version").output()?;
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout)?,
        format!("{name} {}\n", env!("CARGO_PKG_VERSION"))
    );
    let help = binary().arg("--help").output()?;
    assert!(help.status.success());
    assert!(String::from_utf8(help.stdout)?.contains(&format!("Usage: {name} ")));
    let cli = open_rknpu2_lm_infer::cli::Cli::try_parse_from([name, "--model", "unused", "run"])?;
    assert_eq!(
        cli.model
            .options()?
            .tune_cache
            .file_name()
            .unwrap()
            .to_str(),
        Some(name)
    );
    Ok(())
}

#[test]
fn excluded_families_fail_before_loading_tokenizer_or_weights() -> Result<()> {
    let enabled = open_rknpu2_lm_infer::model::build_info();
    let enabled: Vec<_> = enabled["model_families"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|f| f["model_types"].as_array().unwrap())
        .map(|v| v.as_str().unwrap())
        .collect();
    let directory = tempfile::tempdir()?;
    for kind in [
        "qwen3",
        "llama",
        "granite",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "minicpmv4_7",
        "bailing_hybrid",
        "unknown",
    ] {
        if enabled.contains(&kind) {
            continue;
        }
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&json!({"model_type":kind}))?,
        )?;
        let output = binary()
            .arg("--model")
            .arg(directory.path())
            .args([
                "run",
                "--prompt",
                "test",
                "--ctx-size",
                "128",
                "--ubatch-size",
                "128",
                "-n",
                "1",
            ])
            .output()?;
        assert!(!output.status.success(), "accepted excluded {kind}");
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("not enabled in this build") && error.contains(kind),
            "{kind}: {error}"
        );
        assert!(!error.contains("tokenizer.json"), "{kind}: {error}");
    }
    Ok(())
}
