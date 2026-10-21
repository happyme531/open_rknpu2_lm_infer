//! Hot expert banks with one shared cold bank and bounded read-ahead buffers.
use super::timeline::Timeline;
use super::{
    io::Reader,
    store::Store,
    window::{Window, tail_experts},
};
use crate::{
    model::ModelOptions,
    npu::{Backend, Moe},
};
use anyhow::{Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use std::{cell::RefCell, collections::HashSet, rc::Rc, sync::Arc, time::Instant};

pub(crate) struct Paging {
    reader: Reader,
    cold: RefCell<Moe>,
    hot: Option<RefCell<Vec<HashSet<usize>>>>,
    slots: usize,
    experts: usize,
    cache_percent: f64,
    predictions: RefCell<Vec<Option<Vec<usize>>>>,
    prediction_counts: RefCell<(u64, u64)>,
    timeline: Timeline,
}
impl Paging {
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
            reader: Reader::new(store, timeline.clone())?,
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
        }))
    }
    pub fn predict(&self, layer: usize, ids: &[i32]) {
        let Some(hot) = &self.hot else { return };
        self.timeline
            .mark("LES prediction", || json!({"layer":layer,"experts":ids}));
        self.predictions.borrow_mut()[layer] = Some(ids.iter().map(|&id| id as usize).collect());
        let hot = hot.borrow();
        let ids: Vec<_> = ids
            .iter()
            .map(|&i| i as usize)
            .filter(|id| !hot[layer].contains(id))
            .collect();
        self.reader.predict(layer, &ids);
    }
    pub fn stats(&self) -> Value {
        let (correct, total) = *self.prediction_counts.borrow();
        json!({"io":self.reader.stats(),"cold_bank_weight_bytes":self.cold.borrow().weight_bytes(),
            "cache_percent":self.cache_percent,"slots_per_layer":self.slots,
            "effective_cache_percent":100. * self.slots as f64 / self.experts as f64,
            "prefetch":self.hot.is_some(),"read_buffer_limit":32,"les_correct":correct,"les_total":total})
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
    hot_assignments: u64,
    unique_demand: u64,
    cold_experts: u64,
    imports: u64,
    import_ms: f64,
    compute_ms: f64,
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
        Ok(Self {
            hot: backend.moe_slots(g.hidden, g.intermediate, shared.slots)?,
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
    pub fn request(&self, ids: &[i32], position: usize, rows: usize) -> Request {
        let mut seen = HashSet::new();
        let demand: Vec<_> = ids
            .iter()
            .map(|&id| id as usize)
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
            "rows":rows,"experts":ids,"cold":cold})
        });
        self.shared.reader.demand(self.layer, &cold);
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
        let prediction = if rows == 1 {
            self.shared.predictions.borrow_mut()[self.layer].take()
        } else {
            None
        };
        if let Some(predicted) = &prediction {
            let mut counts = self.shared.prediction_counts.borrow_mut();
            counts.0 += ids
                .iter()
                .filter(|&&id| predicted.contains(&(id as usize)))
                .count() as u64;
            counts.1 += ids.len() as u64;
        }
        let predicted: HashSet<_> = prediction.into_iter().flatten().collect();
        let hot: Vec<_> = self
            .window
            .contents()
            .iter()
            .enumerate()
            .filter_map(|(slot, &id)| id.map(|id| (id, slot)))
            .collect();
        let hits = ids
            .iter()
            .filter(|&&id| self.window.slot(id as usize).is_some())
            .count();
        // Every contribution keeps its original token/rank until FP32 reduction.
        let mut contributions = vec![0.; ids.len() * h];
        let timer = Instant::now();
        execute(
            &self.hot,
            &hot,
            input,
            ids,
            h,
            &mut contributions,
            (&self.shared.timeline, self.layer, "hot"),
        )?;
        let mut compute_ms = timer.elapsed().as_secs_f64() * 1000.;
        let mut import_ms = 0.;
        let mut imports = 0;
        if rows == 1 {
            // llama.cpp-omni: every real miss enters the persistent per-layer
            // window. Actual requests are pinned; prediction protection is tried
            // before the ordinary LRU fallback. Publish only after import.
            let plan = self.window.plan(&demand, &predicted)?;
            for &(id, slot) in &plan {
                let record = self.shared.reader.take(self.layer, id)?;
                self.window.invalidate(slot);
                import_ms += import(
                    &mut self.hot,
                    slot,
                    record.as_bytes(),
                    (&self.shared.timeline, self.layer, id, "hot"),
                )?;
                imports += 1;
                self.window.publish(slot, id);
            }
            let timer = Instant::now();
            execute(
                &self.hot,
                &plan,
                input,
                ids,
                h,
                &mut contributions,
                (&self.shared.timeline, self.layer, "hot"),
            )?;
            compute_ms += timer.elapsed().as_secs_f64() * 1000.;
        } else {
            // Strata's short-prompt staged walk: visit the routed expert union
            // in file order, then reuse each loaded group for all logical rows.
            // The prompt tail seeds omni's decode window.
            let wanted = tail_experts(ids, self.window.contents().len());
            for group in cold.chunks(16) {
                let records: Vec<_> = group
                    .iter()
                    .map(|&id| self.shared.reader.take(self.layer, id))
                    .collect::<Result<_>>()?;
                let mut bank = self.shared.cold.borrow_mut();
                for (slot, record) in records.iter().enumerate() {
                    import_ms += import(
                        &mut bank,
                        slot,
                        record.as_bytes(),
                        (&self.shared.timeline, self.layer, group[slot], "cold"),
                    )?;
                    imports += 1;
                }
                let map: Vec<_> = group
                    .iter()
                    .enumerate()
                    .map(|(slot, &id)| (id, slot))
                    .collect();
                let timer = Instant::now();
                execute(
                    &bank,
                    &map,
                    input,
                    ids,
                    h,
                    &mut contributions,
                    (&self.shared.timeline, self.layer, "cold"),
                )?;
                compute_ms += timer.elapsed().as_secs_f64() * 1000.;
                drop(bank);
                for (&id, record) in group.iter().zip(&records) {
                    if !wanted.contains(&id) {
                        continue;
                    }
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
                }
            }
        }
        for &id in ids {
            self.window.touch(id as usize);
        }
        for row in 0..rows {
            for col in 0..h {
                let mut sum = 0.;
                for rank in 0..top {
                    let assignment = row * top + rank;
                    sum = contributions[assignment * h + col].mul_add(routes[assignment], sum);
                }
                output[row * h + col] = sum;
            }
        }
        if let Some(hot) = &self.shared.hot {
            hot.borrow_mut()[self.layer] =
                self.window.contents().iter().flatten().copied().collect();
        }
        for stats in [&mut self.stats, &mut self.decode]
            .into_iter()
            .take(if rows == 1 { 2 } else { 1 })
        {
            stats.calls += 1;
            stats.assignments += ids.len() as u64;
            stats.hot_assignments += hits as u64;
            stats.unique_demand += demand.len() as u64;
            stats.cold_experts += cold.len() as u64;
            stats.imports += imports;
            stats.import_ms += import_ms;
            stats.compute_ms += compute_ms;
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

fn execute(
    bank: &Moe,
    map: &[(usize, usize)],
    input: &[f32],
    ids: &[i32],
    h: usize,
    contributions: &mut [f32],
    timeline: (&Timeline, usize, &'static str),
) -> Result<()> {
    let top = ids.len() / (input.len() / h);
    let jobs: Vec<_> = ids
        .iter()
        .enumerate()
        .filter_map(|(assignment, &id)| {
            map.iter()
                .find(|&&(e, _)| e == id as usize)
                .map(|&(_, slot)| (assignment, slot))
        })
        .collect();
    for group in jobs.chunks(bank.max_rows()) {
        let mut x = Vec::with_capacity(group.len() * h);
        let mut physical = Vec::with_capacity(group.len());
        for &(assignment, slot) in group {
            let row = assignment / top;
            x.extend_from_slice(&input[row * h..(row + 1) * h]);
            physical.push(slot as i32);
        }
        let mut output = vec![0.; x.len()];
        let start = Instant::now();
        bank.run(
            &x,
            &physical,
            &vec![1.; group.len()],
            &mut output,
            group.len(),
        )?;
        timeline.0.span("NPU experts",0,start,||json!({"layer":timeline.1,"bank":timeline.2,
            "rows":group.len(),"slots":physical,"experts":group.iter().map(|&(assignment,_)|ids[assignment]).collect::<Vec<_>>()}));
        for (row, &(assignment, _)) in group.iter().enumerate() {
            contributions[assignment * h..(assignment + 1) * h]
                .copy_from_slice(&output[row * h..(row + 1) * h]);
        }
    }
    Ok(())
}
