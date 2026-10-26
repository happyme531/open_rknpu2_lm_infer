//! Hot expert banks with one shared cold bank and bounded read-ahead buffers.
use super::timeline::Timeline;
use super::{
    DECODE_BATCH_MAX_ROWS,
    io::Reader,
    store::Store,
    window::{Window, tail_experts},
};
use crate::{
    model::ModelOptions,
    npu::{Backend, ImportBank, Moe},
};
use anyhow::{Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use std::{cell::RefCell, collections::HashSet, rc::Rc, sync::Arc, time::Instant};

const STAGING_SLOTS: usize = 32;

pub(crate) struct Paging {
    // Reader joins workers before the staging banks and their contexts drop.
    reader: Reader,
    staging: Vec<RefCell<Option<Rc<RefCell<ImportBank>>>>>,
    store: Arc<Store>,
    cold: RefCell<Moe>,
    hot: Option<RefCell<Vec<HashSet<usize>>>>,
    slots: usize,
    experts: usize,
    cache_percent: f64,
    predictions: RefCell<Vec<Option<Vec<usize>>>>,
    prediction_counts: RefCell<(u64, u64)>,
    timeline: Timeline,
    // Layers execute serially; retain the large prompt workspace only once.
    workspace: RefCell<Workspace>,
}
impl Paging {
    pub fn timeline(&self) -> Timeline {
        self.timeline.clone()
    }
    pub fn reset(&self) {
        self.reader.reset();
        self.predictions.borrow_mut().fill(None);
    }
    pub fn new(options: &ModelOptions, store: Arc<Store>, slots: usize) -> Result<Rc<Self>> {
        let g = store.geometry();
        let cold = Backend::new(options, 1)?.moe_slots(g.hidden, g.intermediate, 16)?;
        let (layers, experts) = (g.layers, g.experts);
        let timeline = Timeline::from_environment();
        Ok(Rc::new(Self {
            reader: Reader::new(store.clone(), timeline.clone())?,
            staging: (0..layers).map(|_| RefCell::new(None)).collect(),
            store,
            cold: RefCell::new(cold),
            hot: options
                .experts
                .prefetch
                .then(|| RefCell::new(vec![HashSet::new(); layers])),
            slots,
            experts,
            cache_percent: options.experts.cache_percent,
            predictions: RefCell::new(vec![None; layers]),
            prediction_counts: RefCell::new((0, 0)),
            timeline,
            workspace: RefCell::new(Workspace::default()),
        }))
    }
    pub fn predict(&self, layer: usize, ids: &[i32]) {
        let Some(hot) = &self.hot else { return };
        self.timeline
            .mark("LES prediction", || json!({"layer":layer,"experts":ids}));
        let mut seen = HashSet::new();
        let ids: Vec<_> = ids
            .iter()
            .map(|&id| id as usize)
            .filter(|&id| seen.insert(id))
            .collect();
        self.predictions.borrow_mut()[layer] = Some(ids.clone());
        let hot = hot.borrow();
        let ids: Vec<_> = ids
            .iter()
            .copied()
            .filter(|id| !hot[layer].contains(id))
            .collect();
        self.reader.predict(layer, &ids);
    }
    pub fn stats(&self) -> Value {
        let (correct, total) = *self.prediction_counts.borrow();
        let mut banks = HashSet::new();
        let staging_bytes: u64 = self
            .staging
            .iter()
            .filter_map(|bank| {
                let bank = bank.borrow();
                let bank = bank.as_ref()?;
                banks
                    .insert(Rc::as_ptr(bank))
                    .then(|| bank.borrow().weight_bytes())
            })
            .sum();
        json!({"io":self.reader.stats(),"read_buffers":self.store.buffer_stats(),
            "cold_bank_weight_bytes":self.cold.borrow().weight_bytes(),"io_workers":self.reader.worker_count(),
            "cache_percent":self.cache_percent,"slots_per_layer":self.slots,
            "effective_cache_percent":100. * self.slots as f64 / self.experts as f64,
            "prefetch":self.hot.is_some(),"read_buffer_limit":32,"les_correct":correct,"les_total":total,
            "staged_imports":self.hot.is_some(),"staging_policy":"confirmed-demand",
            "import_workers":1,"staging_slots_per_bank":STAGING_SLOTS,
            "staging_banks":banks.len(),"staging_weight_bytes":staging_bytes})
    }
    #[cfg(test)]
    pub fn save_timeline(&self) -> Result<()> {
        self.timeline.save()
    }
    #[cfg(test)]
    pub fn forward_marker(&self, position: usize, rows: usize, begin: bool) {
        self.timeline.mark(
            if begin {
                "forward begin"
            } else {
                "forward end"
            },
            || json!({"position":position,"rows":rows}),
        );
    }
}

#[derive(Default, Serialize)]
struct Stats {
    calls: u64,
    assignments: u64,
    skipped_assignments: u64,
    hot_assignments: u64,
    unique_demand: u64,
    cold_experts: u64,
    imports: u64,
    import_ms: f64,
    compute_ms: f64,
    staged_promotions: u64,
}
pub(crate) struct Paged {
    hot: Moe,
    window: Window,
    experts: usize,
    layer: usize,
    shared: Rc<Paging>,
    stats: Stats,
    decode: Stats,
}
pub(crate) struct Request {
    rows: usize,
    position: usize,
    demand: Vec<usize>,
    cold: Vec<usize>,
}
impl Paged {
    pub fn new(backend: &Backend, store: &Store, layer: usize, shared: Rc<Paging>) -> Result<Self> {
        let g = store.geometry();
        ensure!(
            shared.slots > 0 && shared.slots <= g.experts,
            "invalid hot expert slot count"
        );
        let hot = backend.moe_slots(g.hidden, g.intermediate, shared.slots)?;
        if shared.hot.is_some() {
            let bank = shared
                .staging
                .iter()
                .find_map(|entry| {
                    entry
                        .borrow()
                        .as_ref()
                        .filter(|bank| bank.borrow().shares_context(&hot))
                        .cloned()
                })
                .map(Ok)
                .unwrap_or_else(|| {
                    hot.import_bank(STAGING_SLOTS)
                        .map(|bank| Rc::new(RefCell::new(bank)))
                })?;
            // Paging drops reader before staging; workers only access loaned slots.
            shared
                .reader
                .register_staging(layer, unsafe { bank.borrow().target() }, STAGING_SLOTS);
            *shared.staging[layer].borrow_mut() = Some(bank);
        }
        Ok(Self {
            hot,
            window: Window::new(shared.slots),
            experts: g.experts,
            layer,
            shared,
            stats: Stats::default(),
            decode: Stats::default(),
        })
    }
    pub fn stats(&self) -> Value {
        json!({"all":self.stats,"decode":self.decode,"slots":self.window.contents()})
    }
    /// Queue known demand before the caller starts next-layer prediction.
    pub fn request(&self, ids: &[i32], routes: &[f32], position: usize, rows: usize) -> Request {
        let mut seen = HashSet::new();
        let demand: Vec<_> = ids
            .iter()
            .zip(routes)
            .filter_map(|(&id, &weight)| (weight > 0.).then_some(id as usize))
            .filter(|&id| seen.insert(id))
            .collect();
        let mut cold: Vec<_> = demand
            .iter()
            .copied()
            .filter(|&id| self.window.slot(id).is_none())
            .collect();
        if rows > 1 {
            cold.sort_unstable();
        }
        self.shared.timeline.mark("actual routes", || {
            json!({"layer":self.layer,"position":position,
            "rows":rows,"experts":ids.iter().zip(routes).filter_map(|(&id,&weight)|(weight>0.).then_some(id)).collect::<Vec<_>>(),"route_ids":ids,"route_weights":routes,"cold":cold})
        });
        self.shared.reader.demand(self.layer, &cold);
        if rows <= DECODE_BATCH_MAX_ROWS {
            self.shared.reader.stage_demand(self.layer, &cold);
        }
        Request {
            rows,
            position,
            demand,
            cold,
        }
    }
    pub fn run(
        &mut self,
        input: &[f32],
        ids: &[i32],
        routes: &[f32],
        output: &mut [f32],
        request: Request,
    ) -> Result<()> {
        let Request {
            rows,
            position,
            demand,
            cold,
        } = request;
        let layer_start = Instant::now();
        let prepare_start = Instant::now();
        self.shared.timeline.mark(
            "routed begin",
            || json!({"layer":self.layer,"position":position,"rows":rows}),
        );
        ensure!(rows > 0, "empty paged expert input");
        let h = input.len() / rows;
        let top = ids.len() / rows;
        ensure!(
            top > 0
                && input.len() == rows * h
                && output.len() == input.len()
                && ids.len() == rows * top
                && routes.len() == ids.len(),
            "paged expert shape mismatch"
        );
        ensure!(
            ids.iter()
                .all(|&id| id >= 0 && (id as usize) < self.experts),
            "paged expert id out of range"
        );
        let prediction = if rows <= DECODE_BATCH_MAX_ROWS {
            self.shared.predictions.borrow_mut()[self.layer].take()
        } else {
            None
        };
        if let Some(predicted) = &prediction {
            let mut counts = self.shared.prediction_counts.borrow_mut();
            counts.0 += ids
                .iter()
                .zip(routes)
                .filter(|&(id, weight)| *weight > 0. && predicted.contains(&(*id as usize)))
                .count() as u64;
            counts.1 += routes.iter().filter(|&&weight| weight > 0.).count() as u64;
        }
        let predicted: HashSet<_> = prediction.into_iter().flatten().collect();
        let hot: Vec<_> = self
            .window
            .contents()
            .iter()
            .enumerate()
            .filter_map(|(slot, &id)| id.map(|id| (id, slot)))
            .collect();
        let active = routes.iter().filter(|&&weight| weight > 0.).count();
        let hits = ids
            .iter()
            .zip(routes)
            .filter(|&(id, weight)| *weight > 0. && self.window.slot(*id as usize).is_some())
            .count();
        // Every contribution keeps its original token/rank until FP32 reduction.
        let mut workspace = self.shared.workspace.borrow_mut();
        // Hot and cold passes overwrite every retained token/rank before reduction.
        // Retained entries need no clearing between requests or row counts.
        workspace.contributions.resize(ids.len() * h, 0.);
        self.shared.timeline.span(
            "paged prepare",
            0,
            prepare_start,
            || json!({"layer":self.layer,"rows":rows,"position":position}),
        );
        let timer = Instant::now();
        workspace.execute(
            &self.hot,
            &hot,
            input,
            (ids, routes),
            h,
            (&self.shared.timeline, self.layer, "hot"),
        )?;
        let mut compute_ms = timer.elapsed().as_secs_f64() * 1000.;
        let mut import_ms = 0.;
        let mut imports = 0;
        let mut staged_promotions = 0;
        if rows <= DECODE_BATCH_MAX_ROWS {
            // Reuse the persistent window for speculative verification. Its
            // routed union can exceed the window; initial hot work is already
            // complete, so successive cold groups may replace those slots.
            let capacity = self.window.contents().len();
            let plan_start = Instant::now();
            let pinned = (demand.len() <= capacity)
                .then(|| self.window.plan(&demand, &predicted))
                .transpose()?;
            self.shared.timeline.span(
                "paged cache plan",
                0,
                plan_start,
                || json!({"layer":self.layer,"rows":rows,"position":position}),
            );
            // Two substantial cold groups amortize NPU submissions while the
            // first group hides reads/imports for the second. Larger unions
            // still respect the bounded persistent window.
            let minimum_group = cold.len().min(capacity).div_ceil(2).max(3);
            let mut remaining = cold.clone();
            while !remaining.is_empty() {
                let count = remaining.len().min(capacity);
                let mut group = Vec::with_capacity(count);
                let mut plan = Vec::with_capacity(count);
                for index in 0..count {
                    // Drain extra published records without waiting for the
                    // rest or taking over its queued background imports.
                    let (id, loaded) = if index < minimum_group {
                        self.shared.reader.take_any(self.layer, &remaining)?
                    } else if let Some(loaded) =
                        self.shared.reader.try_take_any(self.layer, &remaining)?
                    {
                        loaded
                    } else {
                        break;
                    };
                    let selection_start = Instant::now();
                    remaining.retain(|&expert| expert != id);
                    group.push(id);
                    // Pin the full request when it fits. Larger unions pin
                    // this group's already imported experts until execution.
                    let slot = if let Some(pinned) = &pinned {
                        pinned.iter().find(|&&(expert, _)| expert == id).unwrap().1
                    } else {
                        self.window.plan(&group, &predicted)?[0].1
                    };
                    plan.push((id, slot));
                    self.window.invalidate(slot);
                    self.shared.timeline.span(
                        "paged slot selection",
                        0,
                        selection_start,
                        || json!({"layer":self.layer,"expert":id}),
                    );
                    if let Some(staged) = &loaded.staged {
                        let start = Instant::now();
                        self.hot.promote(
                            slot,
                            &mut self.shared.staging[self.layer]
                                .borrow()
                                .as_ref()
                                .unwrap()
                                .borrow_mut(),
                            staged.slot,
                        )?;
                        self.shared.timeline.span("import",0,start,||json!({
                            "layer":self.layer,"expert":id,"slot":slot,"bank":"hot","staged_slot":staged.slot,"staging_bank":staged.bank}));
                        import_ms += start.elapsed().as_secs_f64() * 1000.;
                        staged_promotions += 1;
                    } else {
                        import_ms += import(
                            &mut self.hot,
                            slot,
                            loaded.record.as_bytes(),
                            (&self.shared.timeline, self.layer, id, "hot"),
                        )?;
                    }
                    imports += 1;
                    self.window.publish(slot, id);
                }
                self.shared.timeline.mark("cold group", || {
                    json!({
                        "layer":self.layer,"position":position,"rows":rows,
                        "experts":group,"remaining":remaining.len()
                    })
                });
                let timer = Instant::now();
                workspace.execute(
                    &self.hot,
                    &plan,
                    input,
                    (ids, routes),
                    h,
                    (&self.shared.timeline, self.layer, "hot"),
                )?;
                compute_ms += timer.elapsed().as_secs_f64() * 1000.;
            }
        } else {
            // Strata's short-prompt staged walk: visit the routed expert union
            // in file order. Prompt-tail experts go directly to their final
            // hot slots, avoiding a second weight copy after cold execution.
            let wanted = tail_experts(ids, routes, self.window.contents().len());
            for group in cold.chunks(16) {
                let mut seeded = Vec::new();
                let records: Vec<_> = group
                    .iter()
                    .map(|&id| self.shared.reader.take(self.layer, id))
                    .collect::<Result<_>>()?;
                let mut bank = self.shared.cold.borrow_mut();
                let mut map = Vec::new();
                for (&id, record) in group.iter().zip(&records) {
                    if wanted.contains(&id) {
                        let slot = self
                            .window
                            .victim(&wanted, &HashSet::new(), &HashSet::new())?;
                        self.window.invalidate(slot);
                        import_ms += import(
                            &mut self.hot,
                            slot,
                            record.as_bytes(),
                            (&self.shared.timeline, self.layer, id, "hot"),
                        )?;
                        imports += 1;
                        self.window.publish(slot, id);
                        seeded.push((id, slot));
                        continue;
                    }
                    let slot = map.len();
                    import_ms += import(
                        &mut bank,
                        slot,
                        record.as_bytes(),
                        (&self.shared.timeline, self.layer, id, "cold"),
                    )?;
                    imports += 1;
                    map.push((id, slot));
                }
                drop(records);
                let timer = Instant::now();
                workspace.execute(
                    &bank,
                    &map,
                    input,
                    (ids, routes),
                    h,
                    (&self.shared.timeline, self.layer, "cold"),
                )?;
                compute_ms += timer.elapsed().as_secs_f64() * 1000.;
                let timer = Instant::now();
                workspace.execute(
                    &self.hot,
                    &seeded,
                    input,
                    (ids, routes),
                    h,
                    (&self.shared.timeline, self.layer, "hot"),
                )?;
                compute_ms += timer.elapsed().as_secs_f64() * 1000.;
            }
        }
        let update_start = Instant::now();
        for (&id, &weight) in ids.iter().zip(routes) {
            if weight > 0. {
                self.window.touch(id as usize);
            }
        }
        self.shared.timeline.span(
            "paged cache touch",
            0,
            update_start,
            || json!({"layer":self.layer,"rows":rows,"position":position}),
        );
        let reduce_start = Instant::now();
        for row in 0..rows {
            for col in 0..h {
                let mut sum = 0.;
                for rank in 0..top {
                    let assignment = row * top + rank;
                    // Skipped entries can retain stale data from an earlier call.
                    if routes[assignment] > 0. {
                        sum = workspace.contributions[assignment * h + col]
                            .mul_add(routes[assignment], sum);
                    }
                }
                output[row * h + col] = sum;
            }
        }
        self.shared.timeline.span(
            "paged reduction",
            0,
            reduce_start,
            || json!({"layer":self.layer,"rows":rows,"position":position}),
        );
        let publish_start = Instant::now();
        if let Some(hot) = &self.shared.hot {
            hot.borrow_mut()[self.layer] =
                self.window.contents().iter().flatten().copied().collect();
        }
        self.shared.timeline.span(
            "paged cache snapshot",
            0,
            publish_start,
            || json!({"layer":self.layer,"rows":rows,"position":position}),
        );
        for stats in [&mut self.stats, &mut self.decode]
            .into_iter()
            .take(if rows <= DECODE_BATCH_MAX_ROWS { 2 } else { 1 })
        {
            stats.calls += 1;
            stats.assignments += active as u64;
            stats.skipped_assignments += (ids.len() - active) as u64;
            stats.hot_assignments += hits as u64;
            stats.unique_demand += demand.len() as u64;
            stats.cold_experts += cold.len() as u64;
            stats.imports += imports;
            stats.import_ms += import_ms;
            stats.compute_ms += compute_ms;
            stats.staged_promotions += staged_promotions;
        }
        self.shared.timeline.span(
            "routed",
            0,
            layer_start,
            || json!({"layer":self.layer,"position":position,"rows":rows}),
        );
        Ok(())
    }
}

fn import(
    bank: &mut Moe,
    slot: usize,
    bytes: &[u8],
    timeline: (&Timeline, usize, usize, &'static str),
) -> Result<f64> {
    let start = Instant::now();
    bank.import(slot, bytes)?;
    timeline.0.span("import", 0, start, || {
        json!({
        "layer":timeline.1,"expert":timeline.2,"slot":slot,"bank":timeline.3})
    });
    Ok(start.elapsed().as_secs_f64() * 1000.)
}

#[derive(Default)]
struct Workspace {
    contributions: Vec<f32>,
    input: Vec<f32>,
    output: Vec<f32>,
    physical: Vec<i32>,
    routes: Vec<f32>,
    jobs: Vec<(usize, usize)>,
}

impl Workspace {
    fn execute(
        &mut self,
        bank: &Moe,
        map: &[(usize, usize)],
        input: &[f32],
        routing: (&[i32], &[f32]),
        h: usize,
        timeline: (&Timeline, usize, &'static str),
    ) -> Result<()> {
        let (ids, routes) = routing;
        let execute_start = Instant::now();
        let top = ids.len() / (input.len() / h);
        let jobs_start = Instant::now();
        self.jobs.clear();
        self.jobs
            .extend(ids.iter().enumerate().filter_map(|(assignment, &id)| {
                if routes[assignment] == 0. {
                    return None;
                }
                map.iter()
                    .find(|&&(e, _)| e == id as usize)
                    .map(|&(_, slot)| (assignment, slot))
            }));
        timeline.0.span(
            "paged jobs",
            0,
            jobs_start,
            || json!({"layer":timeline.1,"bank":timeline.2,"assignments":self.jobs.len()}),
        );
        for group in self.jobs.chunks(bank.max_rows()) {
            let gather_start = Instant::now();
            self.input.clear();
            self.physical.clear();
            for &(assignment, slot) in group {
                let row = assignment / top;
                self.input.extend_from_slice(&input[row * h..(row + 1) * h]);
                self.physical.push(slot as i32);
            }
            self.output.resize(self.input.len(), 0.);
            self.routes.resize(group.len(), 1.);
            timeline.0.span(
            "paged gather",
            0,
            gather_start,
            || json!({"layer":timeline.1,"bank":timeline.2,"rows":group.len(),"bytes":self.input.len()*4}),
        );
            let start = Instant::now();
            let native = if timeline.0.enabled() {
                bank.run_profiled(
                    &self.input,
                    &self.physical,
                    &self.routes,
                    &mut self.output,
                    group.len(),
                )?
            } else {
                bank.run(
                    &self.input,
                    &self.physical,
                    &self.routes,
                    &mut self.output,
                    group.len(),
                )?;
                Value::Null
            };
            timeline.0.span("NPU experts",0,start,||json!({"layer":timeline.1,"bank":timeline.2,
            "rows":group.len(),"slots":self.physical,"experts":group.iter().map(|&(assignment,_)|ids[assignment]).collect::<Vec<_>>(),"native":native}));
            let scatter_start = Instant::now();
            for (row, &(assignment, _)) in group.iter().enumerate() {
                self.contributions[assignment * h..(assignment + 1) * h]
                    .copy_from_slice(&self.output[row * h..(row + 1) * h]);
            }
            timeline.0.span("paged scatter", 0, scatter_start, || {
            json!({"layer":timeline.1,"bank":timeline.2,"rows":group.len(),"bytes":self.output.len()*4})
        });
        }
        timeline.0.span(
            "paged execute",
            0,
            execute_start,
            || json!({"layer":timeline.1,"bank":timeline.2}),
        );
        Ok(())
    }
}
