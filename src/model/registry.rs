//! Compiled model-family entry points. Families may share a decoder architecture.
use super::{LanguageModel, ModelOptions};
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::Path;

pub(super) struct Family {
    name: &'static str,
    model_types: &'static [&'static str],
    pub load: fn(ModelOptions) -> Result<Box<dyn LanguageModel>>,
}

#[cfg(any(
    feature = "model-qwen3",
    feature = "model-llama",
    feature = "model-granite"
))]
fn dense(options: ModelOptions) -> Result<Box<dyn LanguageModel>> {
    Ok(Box::new(super::dense::Dense::load(options)?))
}

#[cfg(any(feature = "model-qwen35", feature = "model-minicpmv47"))]
fn qwen35(options: ModelOptions) -> Result<Box<dyn LanguageModel>> {
    Ok(Box::new(super::qwen35::Qwen35::load(options)?))
}

#[cfg(feature = "model-ling3tiny")]
fn ling3tiny(options: ModelOptions) -> Result<Box<dyn LanguageModel>> {
    Ok(Box::new(super::ling3tiny::Ling3Tiny::load(options)?))
}

const FAMILIES: &[Family] = &[
    #[cfg(feature = "model-qwen3")]
    Family {
        name: "qwen3",
        model_types: &["qwen3"],
        load: dense,
    },
    #[cfg(feature = "model-llama")]
    Family {
        name: "llama",
        model_types: &["llama"],
        load: dense,
    },
    #[cfg(feature = "model-granite")]
    Family {
        name: "granite",
        model_types: &["granite"],
        load: dense,
    },
    #[cfg(feature = "model-qwen35")]
    Family {
        name: "qwen35",
        model_types: &["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"],
        load: qwen35,
    },
    #[cfg(feature = "model-minicpmv47")]
    Family {
        name: "minicpmv47",
        model_types: &["minicpmv4_7"],
        load: qwen35,
    },
    #[cfg(feature = "model-ling3tiny")]
    Family {
        name: "ling3tiny",
        model_types: &["bailing_hybrid"],
        load: ling3tiny,
    },
];

pub fn build_info() -> Value {
    json!({
        "project_name": crate::PROJECT_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "model_families": FAMILIES.iter().map(|f| json!({"name":f.name,"model_types":f.model_types})).collect::<Vec<_>>(),
        "capabilities": {"vision":cfg!(feature = "vision"),"draft_dflash":cfg!(feature = "draft-dflash")}
    })
}

pub(super) fn require(config: &Value) -> Result<&'static Family> {
    let kind = config["model_type"]
        .as_str()
        .context("model config requires model_type")?;
    FAMILIES.iter().find(|f| f.model_types.contains(&kind)).with_context(|| {
        let enabled = FAMILIES.iter().map(|f| f.name).collect::<Vec<_>>().join(", ");
        format!("model architecture {kind:?} is not enabled in this build; compiled families: {enabled}")
    })
}

/// Reject excluded families before tokenizers, weights or device initialization.
pub fn ensure_enabled(path: &Path) -> Result<()> {
    let config = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    require(&config)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_aliases_follow_build_features() {
        for (types, enabled) in [
            (&["qwen3"][..], cfg!(feature = "model-qwen3")),
            (&["llama"][..], cfg!(feature = "model-llama")),
            (&["granite"][..], cfg!(feature = "model-granite")),
            (
                &["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"][..],
                cfg!(feature = "model-qwen35"),
            ),
            (&["minicpmv4_7"][..], cfg!(feature = "model-minicpmv47")),
            (&["bailing_hybrid"][..], cfg!(feature = "model-ling3tiny")),
        ] {
            for kind in types {
                assert_eq!(
                    require(&json!({"model_type":kind})).is_ok(),
                    enabled,
                    "{kind}"
                );
            }
        }
        assert!(require(&json!({"model_type":"unknown"})).is_err());
        assert!(require(&json!({})).is_err());
    }

    #[test]
    fn excluded_checkpoint_fails_before_tokenizer_or_device() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("config.json"),
            br#"{"model_type":"not_in_this_build"}"#,
        )?;
        let error = ensure_enabled(dir.path()).unwrap_err().to_string();
        assert!(error.contains("not enabled in this build"));
        assert!(!error.contains("tokenizer"));
        Ok(())
    }
}
