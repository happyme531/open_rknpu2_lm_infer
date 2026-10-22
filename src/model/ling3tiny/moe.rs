use super::{config::Config, load::Loader};
use crate::model::trace;
use crate::{
    model::{
        cpu::Cpu,
        experts::{Paged, Paging, store::Store},
        projection::{Linear, Projection},
    },
    npu::{Moe, Tf32Linear},
    ops::activation::sigmoid,
};
use anyhow::{Result, ensure};
use std::{rc::Rc, sync::Arc};

pub(super) struct Dense {
    gate_up: Projection,
    down: Linear,
    intermediate: usize,
}

impl Dense {
    pub fn load(l: &Loader<'_>, p: &str, f: usize) -> Result<Self> {
        Ok(Self {
            gate_up: Projection::load(
                l.backend,
                l.weights,
                l.cpu,
                vec![(format!("{p}.gate_proj"), f), (format!("{p}.up_proj"), f)],
                l.config.hidden,
            )?,
            down: l.linear(&format!("{p}.down_proj"), l.config.hidden, f)?,
            intermediate: f,
        })
    }
    pub fn run(&self, input: &[f32], output: &mut [f32], rows: usize, cpu: &Cpu) -> Result<()> {
        let f = self.intermediate;
        let mut middle = vec![0.; rows * f];
        if !self.gate_up.run_swiglu(input, &mut middle, rows)? {
            let mut gu = vec![0.; rows * 2 * f];
            let mut scratch = vec![0.; rows * self.gate_up.scratch_width()];
            self.gate_up.run(input, &mut gu, rows, &mut scratch)?;
            cpu.swiglu(&gu, &mut middle, f);
        }
        self.down.run(&middle, output, rows)
    }
}
enum Bank {
    Resident(Moe),
    Paged(Box<Paged>),
}
pub(super) enum FeedForward {
    Dense(Dense),
    Routed(Box<Routed>),
}
pub(super) struct Routed {
    router: Tf32Linear,
    bias: Vec<f32>,
    bank: Bank,
    shared: Dense,
}

/// Bias affects group/expert selection; contribution weights use unbiased sigmoid.
pub(super) fn routes(logits: &[f32], bias: &[f32], c: &Config) -> Result<(Vec<i32>, Vec<f32>)> {
    ensure!(
        logits.len() == c.experts
            && bias.len() == c.experts
            && logits.iter().all(|x| x.is_finite()),
        "Ling router shape/value mismatch"
    );
    let scores: Vec<_> = logits.iter().map(|&x| sigmoid(x)).collect();
    let selected: Vec<_> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
    let width = c.experts / c.groups;
    let mut group_scores = Vec::new();
    for group in selected.chunks_exact(width) {
        let mut indices: Vec<_> = (0..width).collect();
        indices.sort_unstable_by(|&a, &b| group[b].total_cmp(&group[a]).then(a.cmp(&b)));
        group_scores.push(group[indices[0]] + group[indices[1]]);
    }
    let mut groups: Vec<_> = (0..c.groups).collect();
    groups.sort_unstable_by(|&a, &b| group_scores[b].total_cmp(&group_scores[a]).then(a.cmp(&b)));
    groups.truncate(c.top_groups);
    let mut ids: Vec<_> = (0..c.experts)
        .filter(|id| groups.contains(&(id / width)))
        .collect();
    ids.sort_unstable_by(|&a, &b| selected[b].total_cmp(&selected[a]).then(a.cmp(&b)));
    ids.truncate(c.top);
    let sum = ids.iter().map(|&i| scores[i]).sum::<f32>() + 1e-20;
    let weights = ids
        .iter()
        .map(|&i| {
            let weight = if c.top > 1 {
                scores[i] / sum
            } else {
                scores[i]
            };
            weight * c.routing_scale
        })
        .collect();
    Ok((ids.into_iter().map(|i| i as i32).collect(), weights))
}
impl FeedForward {
    pub fn load(
        l: &Loader<'_>,
        p: &str,
        index: usize,
        store: Option<&Arc<Store>>,
        paging: Option<&Rc<Paging>>,
    ) -> Result<Self> {
        let c = l.config;
        if index < c.dense_layers {
            return Ok(Self::Dense(Dense::load(l, p, c.dense_intermediate)?));
        }
        let layer = index - c.dense_layers;
        let bank = if let Some(paging) = paging {
            Bank::Paged(Box::new(Paged::new(
                l.backend,
                store.unwrap(),
                layer,
                paging.clone(),
            )?))
        } else if let Some(store) = store {
            let mut bank = l
                .backend
                .moe_packed(c.hidden, c.intermediate, c.experts, c.top)?;
            for expert in 0..c.experts {
                bank.import(expert, store.read(layer, expert)?.as_bytes())?;
            }
            Bank::Resident(bank)
        } else {
            let mut gate = Vec::new();
            let mut down = Vec::new();
            for expert in 0..c.experts {
                let (g, d) = l.cpu.pool.install(|| {
                    l.weights
                        .expert(&format!("{p}.experts"), expert, c.hidden, c.intermediate)
                })?;
                gate.push(g);
                down.push(d);
            }
            Bank::Resident(l.backend.load_moe(&gate, &down, c.top)?)
        };
        let router = l.weights.store.get(&format!("{p}.gate.weight"))?;
        router.require_shape(&[c.experts, c.hidden], "Ling router")?;
        ensure!(
            router.floating(),
            "Ling router must retain floating weights"
        );
        Ok(Self::Routed(Box::new(Routed {
            router: l.router.load(&router.floats()?, c.hidden, c.experts)?,
            bias: l.vector(&format!("{p}.gate.expert_bias"), c.experts)?,
            bank,
            shared: Dense::load(l, &format!("{p}.shared_experts"), c.shared_intermediate)?,
        })))
    }
    pub fn run(
        &mut self,
        input: &[f32],
        rows: usize,
        position: usize,
        c: &Config,
        cpu: &Cpu,
    ) -> Result<Vec<f32>> {
        let mut out = vec![0.; rows * c.hidden];
        match self {
            Self::Dense(d) => {
                trace::measure(c"ling.dense_ffn", || d.run(input, &mut out, rows, cpu))?
            }
            Self::Routed(r) => {
                let mut logits = vec![0.; rows * c.experts];
                trace::measure(c"ling.moe_router_projection", || {
                    r.router.run(input, &mut logits, rows)
                })?;
                let (ids, weights) = trace::measure(c"ling.moe_router", || -> Result<_> {
                    let mut ids = Vec::with_capacity(rows * c.top);
                    let mut weights = Vec::with_capacity(rows * c.top);
                    for row in logits.chunks_exact(c.experts) {
                        let (i, w) = routes(row, &r.bias, c)?;
                        ids.extend(i);
                        weights.extend(w);
                    }
                    Ok((ids, weights))
                })?;
                trace::measure(c"ling.moe_routed_experts", || match &mut r.bank {
                    Bank::Resident(b) => b.run(input, &ids, &weights, &mut out, rows),
                    Bank::Paged(b) => {
                        let request = b.request(&ids, position, rows);
                        b.run(input, &ids, &weights, &mut out, request)
                    }
                })?;
                trace::measure(c"ling.shared_expert_ffn", || -> Result<()> {
                    let mut shared = vec![0.; rows * c.hidden];
                    r.shared.run(input, &mut shared, rows, cpu)?;
                    cpu.add(&mut out, &shared);
                    Ok(())
                })?;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn group_bias_selects_experts_but_weights_use_original_sigmoid() -> Result<()> {
        let mut c = Config::parse(&super::super::config::official())?;
        c.experts = 8;
        c.groups = 4;
        c.top_groups = 2;
        c.top = 3;
        let logits = [3., 2., -2., -1., 0., 0.5, 4., 3.5];
        let bias = [0., 0., 3., 3., 4., 4., 0., 0.];
        let (ids, weights) = routes(&logits, &bias, &c)?;
        assert_eq!(ids, [5, 4, 3]);
        let sum = sigmoid(0.5) + sigmoid(0.) + sigmoid(-1.);
        for (actual, value) in weights.iter().zip([0.5, 0., -1.]) {
            assert!((actual - sigmoid(value) / sum * 2.5).abs() < 1e-6);
        }
        assert!((weights.iter().sum::<f32>() - 2.5).abs() < 1e-6);
        Ok(())
    }
}
