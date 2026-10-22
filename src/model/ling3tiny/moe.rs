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
    routing: RoutingScratch,
    bias: Vec<f32>,
    bank: Bank,
    shared: Dense,
}

struct RoutingScratch {
    scores: Vec<f32>,
    selected: Vec<f32>,
    group_scores: Vec<f32>,
    groups: Vec<usize>,
    enabled: Vec<bool>,
    candidates: Vec<usize>,
}
impl RoutingScratch {
    fn new(c: &Config) -> Self {
        Self {
            scores: vec![0.; c.experts],
            selected: vec![0.; c.experts],
            group_scores: vec![0.; c.groups],
            groups: (0..c.groups).collect(),
            enabled: vec![false; c.groups],
            candidates: Vec::with_capacity(c.experts),
        }
    }
    /// Bias affects selection; contribution weights use unbiased sigmoid.
    fn routes(
        &mut self,
        logits: &[f32],
        bias: &[f32],
        c: &Config,
        ids: &mut [i32],
        weights: &mut [f32],
    ) -> Result<()> {
        ensure!(
            logits.len() == c.experts
                && bias.len() == c.experts
                && ids.len() == c.top
                && weights.len() == c.top
                && logits.iter().all(|x| x.is_finite()),
            "Ling router shape/value mismatch"
        );
        for (i, (&logit, &bias)) in logits.iter().zip(bias).enumerate() {
            self.scores[i] = sigmoid(logit);
            self.selected[i] = self.scores[i] + bias;
        }
        let width = c.experts / c.groups;
        for (group, score) in self
            .selected
            .chunks_exact(width)
            .zip(&mut self.group_scores)
        {
            let (mut first, mut second) = (group[0], group[1]);
            if second.total_cmp(&first).is_gt() {
                std::mem::swap(&mut first, &mut second);
            }
            for &value in &group[2..] {
                if value.total_cmp(&first).is_gt() {
                    second = first;
                    first = value;
                } else if value.total_cmp(&second).is_gt() {
                    second = value;
                }
            }
            *score = first + second;
        }
        if c.top_groups < c.groups {
            self.groups.select_nth_unstable_by(c.top_groups, |&a, &b| {
                self.group_scores[b]
                    .total_cmp(&self.group_scores[a])
                    .then(a.cmp(&b))
            });
        }
        self.enabled.fill(false);
        for &group in &self.groups[..c.top_groups] {
            self.enabled[group] = true;
        }
        self.candidates.clear();
        self.candidates
            .extend((0..c.experts).filter(|id| self.enabled[id / width]));
        let compare = |&a: &usize, &b: &usize| {
            self.selected[b]
                .total_cmp(&self.selected[a])
                .then(a.cmp(&b))
        };
        if c.top < self.candidates.len() {
            self.candidates.select_nth_unstable_by(c.top, compare);
        }
        let chosen = &mut self.candidates[..c.top];
        chosen.sort_unstable_by(compare);
        let sum = chosen.iter().map(|&i| self.scores[i]).sum::<f32>() + 1e-20;
        for ((id, weight), &expert) in ids.iter_mut().zip(weights).zip(chosen.iter()) {
            *id = expert as i32;
            *weight = (if c.top > 1 {
                self.scores[expert] / sum
            } else {
                self.scores[expert]
            }) * c.routing_scale;
        }
        Ok(())
    }
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
            routing: RoutingScratch::new(c),
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
                    let mut ids = vec![0; rows * c.top];
                    let mut weights = vec![0.; rows * c.top];
                    for ((row, ids), weights) in logits
                        .chunks_exact(c.experts)
                        .zip(ids.chunks_mut(c.top))
                        .zip(weights.chunks_mut(c.top))
                    {
                        r.routing.routes(row, &r.bias, c, ids, weights)?;
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
    fn selection_matches_sorted_oracle_with_ties_and_reused_scratch() -> Result<()> {
        let mut c = Config::parse(&super::super::config::official())?;
        for (experts, groups, top_groups, top) in [(128, 8, 4, 8), (8, 4, 4, 1), (8, 4, 2, 4)] {
            c.experts = experts;
            c.groups = groups;
            c.top_groups = top_groups;
            c.top = top;
            let mut scratch = RoutingScratch::new(&c);
            for step in 0..100 {
                let logits: Vec<_> = (0..experts)
                    .map(|i| {
                        if step % 10 == 0 {
                            0.
                        } else {
                            ((i * 17 + step * 31) % 29) as f32 * 0.25 - 3.5
                        }
                    })
                    .collect();
                let bias: Vec<_> = (0..experts)
                    .map(|i| {
                        if step % 10 == 0 {
                            0.
                        } else {
                            ((i * 13 + step * 7) % 11) as f32 * 0.125 - 0.625
                        }
                    })
                    .collect();
                let scores: Vec<_> = logits.iter().map(|&x| sigmoid(x)).collect();
                let selected: Vec<_> = scores.iter().zip(&bias).map(|(x, b)| x + b).collect();
                let width = experts / groups;
                let group_scores: Vec<_> = selected
                    .chunks(width)
                    .map(|g| {
                        let mut sorted = g.to_vec();
                        sorted.sort_unstable_by(|a, b| b.total_cmp(a));
                        sorted[0] + sorted[1]
                    })
                    .collect();
                let mut groups: Vec<_> = (0..c.groups).collect();
                groups.sort_unstable_by(|&a, &b| {
                    group_scores[b].total_cmp(&group_scores[a]).then(a.cmp(&b))
                });
                groups.truncate(top_groups);
                let mut reference: Vec<_> = (0..experts)
                    .filter(|i| groups.contains(&(i / width)))
                    .collect();
                reference
                    .sort_unstable_by(|&a, &b| selected[b].total_cmp(&selected[a]).then(a.cmp(&b)));
                reference.truncate(top);
                let sum = reference.iter().map(|&i| scores[i]).sum::<f32>() + 1e-20;
                let expected: Vec<_> = reference
                    .iter()
                    .map(|&i| (if top > 1 { scores[i] / sum } else { scores[i] }) * c.routing_scale)
                    .collect();
                let mut ids = vec![0; top];
                let mut weights = vec![0.; top];
                scratch.routes(&logits, &bias, &c, &mut ids, &mut weights)?;
                assert_eq!(
                    ids,
                    reference.into_iter().map(|i| i as i32).collect::<Vec<_>>()
                );
                assert_eq!(weights, expected);
            }
        }
        Ok(())
    }
    #[test]
    fn group_bias_selects_experts_but_weights_use_original_sigmoid() -> Result<()> {
        let mut c = Config::parse(&super::super::config::official())?;
        c.experts = 8;
        c.groups = 4;
        c.top_groups = 2;
        c.top = 3;
        let logits = [3., 2., -2., -1., 0., 0.5, 4., 3.5];
        let bias = [0., 0., 3., 3., 4., 4., 0., 0.];
        let mut ids = vec![0; c.top];
        let mut weights = vec![0.; c.top];
        RoutingScratch::new(&c).routes(&logits, &bias, &c, &mut ids, &mut weights)?;
        assert_eq!(ids, [5, 4, 3]);
        let sum = sigmoid(0.5) + sigmoid(0.) + sigmoid(-1.);
        for (actual, value) in weights.iter().zip([0.5, 0., -1.]) {
            assert!((actual - sigmoid(value) / sum * 2.5).abs() < 1e-6);
        }
        assert!((weights.iter().sum::<f32>() - 2.5).abs() < 1e-6);
        Ok(())
    }
}
