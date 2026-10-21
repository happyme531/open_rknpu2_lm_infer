use super::*;
use crate::{
    model::{ModelOptions, VisionDtype, checkpoint::TensorStore},
    npu::{Backend, FloatBackend},
};
use anyhow::{Context, Result, ensure};
use serde_json::Value;

fn dimension(v: &Value, key: &str) -> Result<usize> {
    v[key]
        .as_u64()
        .and_then(|x| usize::try_from(x).ok())
        .filter(|x| *x > 0)
        .with_context(|| format!("invalid vision {key}"))
}
fn norm(store: &TensorStore, name: &str, width: usize) -> Result<LayerNorm> {
    let w = store.get(&format!("{name}.weight"))?;
    let b = store.get(&format!("{name}.bias"))?;
    w.require_shape(&[width], name)?;
    b.require_shape(&[width], name)?;
    Ok(LayerNorm {
        weight: w.floats()?,
        bias: b.floats()?,
        epsilon: 1e-6,
    })
}
fn linear(
    store: &TensorStore,
    b: &FloatBackend,
    name: &str,
    k: usize,
    n: usize,
    patch: bool,
) -> Result<FloatLinear> {
    let w = store.get(&format!("{name}.weight"))?;
    let bias = store.get(&format!("{name}.bias"))?;
    if patch {
        w.require_shape(&[n, 3, 2, 16, 16], name)?;
    } else {
        w.require_shape(&[n, k], name)?;
    }
    bias.require_shape(&[n], name)?;
    b.load(&w.floats()?, &bias.floats()?, k, n)
}
pub(super) fn config(path: &std::path::Path) -> Result<Config> {
    let root: Value = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    let v = &root["vision_config"];
    ensure!(
        root["model_type"] == "qwen3_5"
            && v["patch_size"] == 16
            && v["spatial_merge_size"] == 2
            && v["temporal_patch_size"] == 2
            && v["in_channels"] == 3
            && v["hidden_act"] == "gelu_pytorch_tanh"
            && v["deepstack_visual_indexes"]
                .as_array()
                .is_none_or(Vec::is_empty),
        "unsupported Qwen3.5 vision configuration"
    );
    let positions = dimension(v, "num_position_embeddings")?;
    let config = Config {
        hidden: dimension(v, "hidden_size")?,
        intermediate: dimension(v, "intermediate_size")?,
        heads: dimension(v, "num_heads")?,
        depth: dimension(v, "depth")?,
        side: positions.isqrt(),
        out: dimension(v, "out_hidden_size")?,
    };
    ensure!(
        config.side * config.side == positions
            && config.hidden.is_multiple_of(config.heads)
            && (config.hidden / config.heads).is_multiple_of(32),
        "unsupported vision position/head geometry"
    );
    Ok(config)
}
pub(super) fn load(
    options: &ModelOptions,
    store: &TensorStore,
    domain: u32,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Weights> {
    let config = config(&options.path)?;
    let positions = config.side * config.side;
    ensure!(domain < 16, "no free IOMMU domain for vision weights");
    let mut options = options.clone();
    options.chunk = 128;
    let device = Backend::new(&options, domain)?;
    let backend = match options.vision_dtype {
        VisionDtype::Fp16 => FloatBackend::fp16_storage(device.clone()),
        VisionDtype::Fp32 => FloatBackend::new(device.clone()),
    };
    let h = config.hidden;
    let f = config.intermediate;
    let patch = linear(
        store,
        &backend,
        "model.visual.patch_embed.proj",
        1536,
        h,
        true,
    )?;
    let pos = store.get("model.visual.pos_embed.weight")?;
    pos.require_shape(&[positions, h], "vision pos_embed")?;
    let position = pos.floats()?;
    let mut blocks = Vec::with_capacity(config.depth);
    for i in 0..config.depth {
        ensure!(
            !cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)),
            "generation cancelled"
        );
        let p = format!("model.visual.blocks.{i}");
        blocks.push(Block {
            norm1: norm(store, &format!("{p}.norm1"), h)?,
            norm2: norm(store, &format!("{p}.norm2"), h)?,
            qkv: linear(store, &backend, &format!("{p}.attn.qkv"), h, 3 * h, false)?,
            proj: linear(store, &backend, &format!("{p}.attn.proj"), h, h, false)?,
            fc1: linear(store, &backend, &format!("{p}.mlp.linear_fc1"), h, f, false)?,
            fc2: linear(store, &backend, &format!("{p}.mlp.linear_fc2"), f, h, false)?,
        });
    }
    let merger = Merger {
        norm: norm(store, "model.visual.merger.norm", h)?,
        fc1: linear(
            store,
            &backend,
            "model.visual.merger.linear_fc1",
            4 * h,
            4 * h,
            false,
        )?,
        fc2: linear(
            store,
            &backend,
            "model.visual.merger.linear_fc2",
            4 * h,
            config.out,
            false,
        )?,
    };
    Ok(Weights {
        config,
        patch,
        position,
        blocks,
        merger,
        backend: device,
    })
}
