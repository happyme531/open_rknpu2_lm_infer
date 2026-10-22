use crate::{
    model::{self, ModelOptions},
    validation::BoardLock,
};
use anyhow::{Result, ensure};
use serde_json::json;
use std::path::PathBuf;
mod phases;
use phases::measure_phases;

fn summary(samples: &[f64], tokens: usize) -> serde_json::Value {
    let speeds: Vec<_> = samples
        .iter()
        .map(|ms| tokens as f64 * 1000.0 / ms)
        .collect();
    let mean = speeds.iter().sum::<f64>() / speeds.len() as f64;
    let stddev = if speeds.len() > 1 {
        (speeds.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (speeds.len() - 1) as f64).sqrt()
    } else {
        0.0
    };
    json!({"tokens":tokens,"samples_ms":samples,"samples_tokens_per_second":speeds,"mean_tokens_per_second":mean,"stddev_tokens_per_second":stddev})
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Case {
    pub prompt: usize,
    pub generate: usize,
    pub depth: usize,
}
impl Case {
    pub fn new(prompt: usize, generate: usize, depth: usize) -> Self {
        Self {
            prompt,
            generate,
            depth,
        }
    }
    pub fn capacity(&self) -> Result<usize> {
        self.depth
            .checked_add(self.prompt)
            .and_then(|tokens| tokens.checked_add(self.generate))
            .ok_or_else(|| anyhow::anyhow!("benchmark token counts exceed context capacity"))
    }
}
pub fn suite_cases(name: &str) -> Result<Vec<Case>> {
    let cases: &[(usize, usize, usize)] = match name {
        "representative" => &[(512, 0, 0), (4096, 0, 0), (0, 128, 0), (0, 128, 2048)],
        "contexts" => &[
            (512, 0, 0),
            (1024, 0, 0),
            (4096, 0, 0),
            (0, 128, 512),
            (0, 128, 1024),
            (0, 128, 4096),
        ],
        other => anyhow::bail!("unknown benchmark suite: {other}"),
    };
    Ok(cases.iter().map(|&(p, n, d)| Case::new(p, n, d)).collect())
}
pub fn run(
    options: ModelOptions,
    cases: &[Case],
    repetitions: usize,
    report: Option<PathBuf>,
) -> Result<()> {
    ensure!(
        repetitions > 0
            && !cases.is_empty()
            && cases
                .iter()
                .all(|case| case.prompt > 0 || case.generate > 0),
        "benchmark needs repetitions and at least one phase"
    );
    let capacity = cases.iter().try_fold(0, |capacity, case| {
        Ok::<_, anyhow::Error>(capacity.max(case.capacity()?))
    })?;
    ensure!(
        capacity <= options.context,
        "benchmark needs {capacity} KV tokens but --ctx-size is {}; increase --ctx-size or use -c 0",
        options.context
    );
    let _lock = BoardLock::acquire()?;
    let chunk = options.chunk;
    let mut model = model::load(options)?;
    let mut rng = Mt19937::new(42);
    let tokens: Vec<_> = (0..capacity)
        .map(|_| rng.next() % model.info().vocab_size as u32)
        .collect();
    let head_free_check = phases::verify_head_free(model.as_mut(), &tokens, chunk)?;
    let mut results = Vec::new();
    for (case_index, &case) in cases.iter().enumerate() {
        let Case {
            prompt,
            generate,
            depth,
        } = case;
        model.reserve(case.capacity()?)?;
        eprintln!("bench: pp{prompt} tg{generate} depth{depth}, warmup");
        measure_phases(model.as_mut(), &tokens, case)?;
        let mut pp = Vec::new();
        let mut tg = Vec::new();
        let mut pg = Vec::new();
        let mut stage_samples = Vec::new();
        for repetition in 0..repetitions {
            let recording = crate::model::trace::Recording::for_sample(case_index, repetition)?;
            let measurement = measure_phases(model.as_mut(), &tokens, case)?;
            if let Some(recording) = recording {
                recording.finish()?;
            }
            let (prefill, decode) = (measurement.prefill, measurement.decode);
            eprintln!(
                "bench: sample {}/{repetitions}, prefill {prefill:.3} ms, decode {decode:.3} ms",
                repetition + 1
            );
            pp.push(prefill);
            tg.push(decode);
            pg.push(prefill + decode);
            stage_samples.push(measurement.stages);
        }
        results.push(json!({"prompt":prompt,"generate":generate,"depth":depth,
            "prefill":if prompt>0 {summary(&pp,prompt)} else {serde_json::Value::Null},
            "decode":if generate>0 {summary(&tg,generate)} else {serde_json::Value::Null},
            "prompt_gen":if prompt>0 && generate>0 {summary(&pg,prompt+generate)} else {serde_json::Value::Null},
            "stage_samples":stage_samples,"stats":model.stats()}));
    }
    let mut result = json!({"model":model.info(),"warmup":1,"repetitions":repetitions,
        "timing":"head-free prefill and history; decode includes one output head per consumed token; pg preserves prompt state; excludes load, preparation, untimed depth prefill, tokenization, sampling and stats snapshots",
        "prefill_without_lm_head":true,"head_free_check":head_free_check,
        "seed":42,"token_source":"mt19937(seed) % vocab_size","synthetic_token_ids":tokens,
        "cases":results,"stats":model.stats()});
    if cases.len() == 1 {
        for key in [
            "prompt",
            "generate",
            "depth",
            "prefill",
            "decode",
            "prompt_gen",
        ] {
            result[key] = results[0][key].clone();
        }
    }
    println!("{}", serde_json::to_string_pretty(&result)?);
    if let Some(path) = report {
        std::fs::write(
            path,
            format!("{}\n", serde_json::to_string_pretty(&result)?),
        )?;
    }
    Ok(())
}

struct Mt19937 {
    state: [u32; 624],
    index: usize,
}
impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }
    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let bits = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (bits >> 1)
                    ^ if bits & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut v = self.state[self.index];
        self.index += 1;
        v ^= v >> 11;
        v ^= (v << 7) & 0x9d2c5680;
        v ^= (v << 15) & 0xefc60000;
        v ^= v >> 18;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LanguageModel;

    struct RecordingModel {
        info: crate::model::ModelInfo,
        position: usize,
        resets: usize,
        forwards: Vec<(usize, Vec<u32>)>,
        logits: [f32; 1],
        head_calls: usize,
    }
    impl LanguageModel for RecordingModel {
        fn info(&self) -> &crate::model::ModelInfo {
            &self.info
        }
        fn reset(&mut self) -> Result<()> {
            self.resets += 1;
            self.position = 0;
            Ok(())
        }
        fn prepare(&mut self, _: usize) -> Result<()> {
            Ok(())
        }
        fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
            self.forwards.push((self.position, tokens.to_vec()));
            self.position += tokens.len();
            Ok(())
        }
        fn forward(&mut self, tokens: &[u32]) -> Result<&[f32]> {
            self.head_calls += 1;
            self.prefill(tokens)?;
            Ok(&self.logits)
        }
        fn position(&self) -> usize {
            self.position
        }
        fn stats(&self) -> serde_json::Value {
            json!({})
        }
    }

    #[test]
    fn pg_preserves_prefill_kv_and_consumes_continuation_ids() -> Result<()> {
        let mut model = RecordingModel {
            info: crate::model::ModelInfo {
                id: "test".into(),
                architecture: "test".into(),
                context: 16,
                vocab_size: 100,
                eos_tokens: vec![],
            },
            position: 0,
            resets: 0,
            forwards: vec![],
            logits: [0.0],
            head_calls: 0,
        };
        measure_phases(&mut model, &[10, 11, 12, 13, 14, 15], Case::new(3, 2, 1))?;
        assert_eq!(model.resets, 1);
        assert_eq!(
            model.forwards,
            vec![
                (0, vec![10]),
                (1, vec![11, 12, 13]),
                (4, vec![14]),
                (5, vec![15])
            ]
        );
        assert_eq!(model.position, 6);
        assert_eq!(
            model.head_calls, 2,
            "prompt and history must not evaluate lm_head"
        );
        Ok(())
    }

    #[test]
    fn synthetic_ids_follow_std_mt19937() {
        let mut rng = super::Mt19937::new(5489);
        let values: Vec<_> = (0..5).map(|_| rng.next()).collect();
        assert_eq!(
            values,
            vec![3499211612, 581869302, 3890346734, 3586334585, 545404204]
        );
    }
}
