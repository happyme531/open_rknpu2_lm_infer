use super::{Layer, Ling3Tiny, config::Config, mixer::Mixer, moe::FeedForward};
use crate::{
    model::{
        ModelInfo, ModelOptions,
        checkpoint::QuantLoader,
        cpu::Cpu,
        experts::{self, Paging, Storage},
        output_head::OutputHead,
        projection::Linear,
    },
    npu::{Backend, FloatBackend, FloatLinear, Tf32Backend},
};
use anyhow::{Result, ensure};
use std::{sync::Arc, time::Instant};

pub(super) struct Loader<'a> {
    pub backend: &'a Backend,
    pub floating: &'a FloatBackend,
    pub router: &'a Tf32Backend,
    pub weights: &'a Arc<QuantLoader>,
    pub cpu: &'a Cpu,
    pub config: &'a Config,
}
impl Loader<'_> {
    pub fn vector(&self, name: &str, width: usize) -> Result<Vec<f32>> {
        let t = self.weights.store.get(name)?;
        t.require_shape(&[width], name)?;
        t.floats()
    }
    pub fn linear(&self, name: &str, n: usize, k: usize) -> Result<Linear> {
        Linear::load(self.backend, self.weights, self.cpu, name, n, k)
    }
    pub fn fp16(&self, name: &str, n: usize, k: usize) -> Result<FloatLinear> {
        self.floating
            .load(&self.matrix(name, n, k)?, &vec![0.; n], k, n)
    }
    /// Keep quantized NPU weights; send protected floating weights through FP16.
    pub fn npu_linear(&self, name: &str, n: usize, k: usize) -> Result<Linear> {
        if self.weights.quantized(name) {
            self.linear(name, n, k)
        } else {
            Ok(Linear::Floating {
                linear: self.fp16(name, n, k)?,
                n,
            })
        }
    }
    /// Reconstruct a checkpoint matrix once when loading FP16 NPU weights.
    pub fn matrix(&self, name: &str, n: usize, k: usize) -> Result<Vec<f32>> {
        if !self.weights.quantized(name) {
            let t = self.weights.store.get(&format!("{name}.weight"))?;
            t.require_shape(&[n, k], name)?;
            return t.floats();
        }
        let matrix = self.cpu.pool.install(|| self.weights.load(name, n, k))?;
        Ok((0..n * k)
            .map(|i| matrix.codes[(i % k) * n + i / k] as f32 * matrix.scales[i / k])
            .collect())
    }
}

pub(super) fn load(options: ModelOptions) -> Result<Ling3Tiny> {
    let start = Instant::now();
    let raw = serde_json::from_slice(&std::fs::read(options.path.join("config.json"))?)?;
    let config = Config::parse(&raw)?;
    ensure!(
        options.context <= config.max_context && options.context > 0,
        "Ling context exceeds model/runtime limit"
    );
    ensure!(
        options.chunk > 0 && options.chunk <= options.context,
        "invalid Ling chunk size"
    );
    ensure!(
        [1, 3, 7].contains(&options.npu_mask),
        "Ling MoE requires NPU mask 1/3/7"
    );
    ensure!(
        options.speculation.window() == 0,
        "Ling speculative decoding is unsupported"
    );
    let cpu = Cpu::new(options.threads)?;
    let weights = Arc::new(QuantLoader::open(&options.path, &raw)?);
    let store = options
        .experts
        .cache
        .as_ref()
        .map(|directory| {
            experts::prepare::open(
                directory,
                experts::store::Geometry {
                    native_version: crate::npu::Moe::blob_version(),
                    layers: config.layers - config.dense_layers,
                    experts: config.experts,
                    hidden: config.hidden,
                    intermediate: config.intermediate,
                    down_bits: 4,
                },
                &options,
                || weights.source_stamps(),
                |layer, expert| {
                    let result = cpu.pool.install(|| {
                        weights.expert(
                            &format!("model.layers.{}.mlp.experts", layer + config.dense_layers),
                            expert,
                            config.hidden,
                            config.intermediate,
                        )
                    })?;
                    if expert + 1 == config.experts {
                        weights.store.release_pages()?;
                    }
                    Ok(result)
                },
            )
        })
        .transpose()?;
    let paging = if options.experts.storage == Storage::Stream {
        let slots = options.experts.slots(config.experts, config.top)?;
        eprintln!(
            "Ling expert cache: {slots}/{} routed experts per layer",
            config.experts
        );
        Some(Paging::new(
            &options,
            store
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("streaming requires --expert-store"))?
                .clone(),
            slots,
        )?)
    } else {
        None
    };
    let embedding = weights.store.get("model.word_embeddings.weight")?;
    embedding.require_shape(&[config.vocab, config.hidden], "Ling embedding")?;
    ensure!(embedding.floating(), "Ling embedding must be floating");
    let mut backends = Vec::new();
    let mut router_backends = Vec::new();
    let mut floating = None;
    let mut layers = Vec::new();
    for index in 0..config.layers {
        // Keep four layers' weights and maximum compressed KV below the
        // native 2 GiB per-domain address budget on Ling-3.0-tiny.
        if index % 4 == 0 {
            ensure!(backends.len() < 12, "Ling exceeds available weight domains");
            backends.push(Backend::new(&options, 15 - backends.len() as u32)?);
            floating = Some(FloatBackend::for_decoder(backends.last().unwrap().clone()));
            router_backends.push(Tf32Backend::new(backends.last().unwrap().clone()));
        }
        let l = Loader {
            backend: backends.last().unwrap(),
            floating: floating.as_ref().unwrap(),
            router: router_backends.last().unwrap(),
            weights: &weights,
            cpu: &cpu,
            config: &config,
        };
        let p = format!("model.layers.{index}");
        layers.push(Layer {
            input_norm: l.vector(&format!("{p}.input_layernorm.weight"), config.hidden)?,
            post_norm: l.vector(
                &format!("{p}.post_attention_layernorm.weight"),
                config.hidden,
            )?,
            mixer: Mixer::load(&l, &format!("{p}.attention"), index)?,
            ffn: FeedForward::load(
                &l,
                &format!("{p}.mlp"),
                index,
                store.as_ref(),
                paging.as_ref(),
            )?,
        });
        weights.store.release_pages()?;
        eprintln!(
            "loaded Ling layer {}/{} ({})",
            index + 1,
            config.layers,
            if config.mla(index) { "MLA" } else { "KDA" }
        );
    }
    let norm = weights.store.get("model.norm.weight")?;
    norm.require_shape(&[config.hidden], "Ling final norm")?;
    let final_norm = norm.floats()?;
    let head = if weights.quantized("lm_head") {
        OutputHead::quantized(
            &options,
            15 - backends.len() as u32,
            &weights,
            &cpu,
            config.vocab,
            config.hidden,
        )?
    } else {
        OutputHead::floating(
            weights.store.get("lm_head.weight")?,
            config.vocab,
            config.hidden,
        )?
    };
    let info = ModelInfo {
        id: options
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        architecture: "bailing_hybrid".into(),
        context: options.context,
        vocab_size: config.vocab,
        eos_tokens: super::super::metadata::eos_tokens(&options.path, &config.eos, config.vocab)?,
    };
    let trace = crate::model::trace::Trace::from_environment()?;
    Ok(Ling3Tiny {
        options,
        info,
        config,
        weights,
        cpu,
        backends,
        router_backends,
        layers,
        paging,
        embedding,
        final_norm,
        head,
        position: 0,
        logits: Vec::new(),
        last_hidden: Vec::new(),
        trace,
        load_ms: start.elapsed().as_secs_f64() * 1000.,
        forward_ms: 0.,
        head_ms: 0.,
    })
}
