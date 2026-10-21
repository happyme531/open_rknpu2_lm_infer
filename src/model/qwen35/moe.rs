//! Routed FFN orchestration, independent of checkpoint storage and weight paging.
use super::{
    config::Config,
    decoder::{Buffers, Execution},
    timings::{Phase, Timings},
};
use crate::{
    model::{
        checkpoint::{QuantLoader, Tensor},
        cpu::Cpu,
        experts::{Paged, Paging, store::Store},
        projection::{Linear, Projection},
    },
    npu::{Backend, Moe},
};
use anyhow::{Result, ensure};
use std::{rc::Rc, sync::Arc};

enum Bank {
    Resident(Moe),
    Paged(Box<Paged>),
}

pub(super) struct Storage<'a> {
    pub options: &'a crate::model::ModelOptions,
    pub store: Option<&'a Arc<Store>>,
    pub paging: Option<&'a Rc<Paging>>,
}

pub(super) struct Routed {
    router: Tensor,
    shared_gate: Tensor,
    experts: Bank,
    prediction: Option<(Tensor, Rc<Paging>, usize)>,
    shared_gu: Projection,
    shared_down: Linear,
    h: usize,
    f: usize,
    count: usize,
    top: usize,
}

pub(super) fn routes(logits: &[f32], top: usize) -> Result<(Vec<i32>, Vec<f32>)> {
    ensure!(
        top > 0 && top <= logits.len() && logits.iter().all(|v| v.is_finite()),
        "invalid router logits"
    );
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    ids.truncate(top);
    let max = logits[ids[0]];
    let mut weights: Vec<_> = ids.iter().map(|&id| (logits[id] - max).exp()).collect();
    let sum: f32 = weights.iter().sum();
    for w in &mut weights {
        *w /= sum;
    }
    Ok((ids.into_iter().map(|id| id as i32).collect(), weights))
}

impl Routed {
    #[cfg(test)]
    pub(super) fn export(&self, expert: usize) -> Result<Vec<u8>> {
        match &self.experts {
            Bank::Resident(m) => m.export(expert),
            _ => anyhow::bail!("paged bank is not fully resident"),
        }
    }
    pub(super) fn statistics(&self) -> serde_json::Value {
        match &self.experts {
            Bank::Resident(_) => serde_json::Value::Null,
            Bank::Paged(p) => p.stats(),
        }
    }
    pub(super) fn load(
        backend: &Backend,
        loader: &Arc<QuantLoader>,
        cpu: &Cpu,
        c: &Config,
        p: &str,
        index: usize,
        storage: Storage<'_>,
    ) -> Result<Self> {
        let Storage {
            options,
            store,
            paging,
        } = storage;
        let m = c.moe.as_ref().expect("MoE configuration");
        let h = c.hidden_size;
        let experts = if let Some(paging) = paging {
            Bank::Paged(Box::new(Paged::new(
                backend,
                store.unwrap(),
                index,
                paging.clone(),
            )?))
        } else if let Some(store) = store {
            let mut bank = backend.moe_packed(h, c.intermediate_size, m.experts, m.top_k)?;
            for expert in 0..m.experts {
                bank.import(expert, store.read(index, expert)?.as_bytes())?;
            }
            Bank::Resident(bank)
        } else {
            let mut gate = Vec::with_capacity(m.experts);
            let mut down = Vec::with_capacity(m.experts);
            let name = format!("{p}.mlp.experts");
            for expert in 0..m.experts {
                let (g, d) = cpu
                    .pool
                    .install(|| loader.expert(&name, expert, h, c.intermediate_size))?;
                gate.push(g);
                down.push(d);
            }
            let experts = backend.load_moe(&gate, &down, m.top_k)?;
            drop((gate, down));
            loader.store.release_pages()?;
            Bank::Resident(experts)
        };
        let prediction = if let Some(paging) =
            paging.filter(|_| options.experts.prefetch && index + 1 < c.layers.len())
        {
            let router = loader.store.get(&format!(
                "{}.layers.{}.mlp.gate.weight",
                c.text_prefix,
                index + 1
            ))?;
            router.require_shape(&[m.experts, h], "LES next-layer router")?;
            Some((router, paging.clone(), index + 1))
        } else {
            None
        };
        let router = loader.store.get(&format!("{p}.mlp.gate.weight"))?;
        router.require_shape(&[m.experts, h], "MoE router")?;
        let shared_gate = loader
            .store
            .get(&format!("{p}.mlp.shared_expert_gate.weight"))?;
        shared_gate.require_shape(&[1, h], "shared expert gate")?;
        ensure!(
            router.floating() && shared_gate.floating(),
            "router and shared gate must be floating"
        );
        let f = m.shared_intermediate;
        Ok(Self {
            router,
            shared_gate,
            experts,
            prediction,
            shared_gu: Projection::load(
                backend,
                loader,
                cpu,
                vec![
                    (format!("{p}.mlp.shared_expert.gate_proj"), f),
                    (format!("{p}.mlp.shared_expert.up_proj"), f),
                ],
                h,
            )?,
            shared_down: Linear::load(
                backend,
                loader,
                cpu,
                &format!("{p}.mlp.shared_expert.down_proj"),
                h,
                f,
            )?,
            h,
            f,
            count: m.experts,
            top: m.top_k,
        })
    }

    pub(super) fn forward(
        &mut self,
        w: &mut Buffers,
        e: &Execution<'_>,
        timings: &mut Timings,
    ) -> Result<()> {
        self.routed(
            &w.normalized[..e.rows * self.h],
            &mut w.projected[..e.rows * self.h],
            e,
            timings,
        )?;
        self.shared(w, e, timings)
    }

    pub(super) fn routed(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        e: &Execution<'_>,
        timings: &mut Timings,
    ) -> Result<()> {
        let rows = e.rows;
        let mut ids = Vec::with_capacity(rows * self.top);
        let mut weights = Vec::with_capacity(rows * self.top);
        let mut logits = vec![0.; self.count];
        timings.measure(Phase::Cpu, || -> Result<()> {
            for x in input.chunks_exact(self.h) {
                e.cpu.dense_head(&self.router, x, &mut logits);
                let (i, r) = routes(&logits, self.top)?;
                ids.extend(i);
                weights.extend(r);
            }
            Ok(())
        })?;
        let request = timings.measure(Phase::Cpu, || -> Result<_> {
            let request = match &self.experts {
                Bank::Paged(bank) => Some(bank.request(&ids, e.position, rows)),
                Bank::Resident(_) => None,
            };
            if rows == 1
                && let Some((router, paging, next)) = &self.prediction
            {
                let mut logits = vec![0.; self.count];
                e.cpu.dense_head(router, &input[..self.h], &mut logits);
                paging.predict(*next, &routes(&logits, self.top)?.0);
            }
            Ok(request)
        })?;
        if e.trace.enabled() {
            e.trace.save(
                e.position,
                e.index,
                "router_ids",
                &ids.iter().map(|&i| i as f32).collect::<Vec<_>>(),
            )?;
            e.trace
                .save(e.position, e.index, "router_weights", &weights)?;
        }
        timings.measure(Phase::Linear, || match &mut self.experts {
            Bank::Resident(bank) => bank.run(input, &ids, &weights, output, rows),
            Bank::Paged(bank) => bank.run(
                input,
                &ids,
                &weights,
                output,
                request.expect("paged expert demand"),
            ),
        })?;
        Ok(())
    }

    pub(super) fn shared(
        &self,
        w: &mut Buffers,
        e: &Execution<'_>,
        timings: &mut Timings,
    ) -> Result<()> {
        let rows = e.rows;
        timings.measure(Phase::Linear, || {
            self.shared_gu.run(
                &w.normalized[..rows * self.h],
                &mut w.gate_up[..rows * 2 * self.f],
                rows,
                &mut w.scatter,
            )
        })?;
        e.cpu.swiglu(
            &w.gate_up[..rows * 2 * self.f],
            &mut w.middle[..rows * self.f],
            self.f,
        );
        timings.measure(Phase::Linear, || {
            self.shared_down.run(
                &w.middle[..rows * self.f],
                &mut w.attended[..rows * self.h],
                rows,
            )
        })?;
        timings.measure(Phase::Cpu, || {
            let mut gate = [0.];
            for row in 0..rows {
                e.cpu.dense_head(
                    &self.shared_gate,
                    &w.normalized[row * self.h..(row + 1) * self.h],
                    &mut gate,
                );
                let scale = 1. / (1. + (-gate[0]).exp());
                for col in 0..self.h {
                    w.projected[row * self.h + col] += scale * w.attended[row * self.h + col];
                }
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn routing_is_normalized_over_selected_experts() {
        let (ids, weights) = routes(&[-100., 1., 3., 2.], 2).unwrap();
        assert_eq!(ids, [2, 3]);
        assert!((weights[0] - 1. / (1. + (-1f32).exp())).abs() < 1e-6);
        assert!((weights.iter().sum::<f32>() - 1.).abs() < 1e-6);
        assert!(routes(&[f32::NAN], 1).is_err());
    }
}
