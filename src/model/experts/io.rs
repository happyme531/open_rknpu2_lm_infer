//! Bounded positional reads with Strata-style adjacent-job batches and generation ownership.
//! Demand reads precede speculative reads; see Strata-LICENSE.
use super::store::{Record, Store};
use super::timeline::Timeline;
use crate::npu::ImportTarget;
use anyhow::{Result, anyhow};
use serde::Serialize;
use serde_json::json;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, Condvar, Mutex},
    thread::JoinHandle,
    time::Instant,
};

type Key = (usize, usize);
const READ_BUFFER_LIMIT: usize = 32;
const IO_WORKERS: usize = 2;
#[derive(Default, Clone, Serialize)]
pub struct Stats {
    pub read_records: u64,
    pub read_batches: u64,
    pub read_bytes: u64,
    pub read_ms: f64,
    pub wait_ms: f64,
    pub prediction_requests: u64,
    pub predicted_demand: u64,
    pub predicted_ready: u64,
    pub predicted_ram_ready: u64,
    pub predicted_in_flight: u64,
    pub predicted_queued: u64,
    pub unused_predictions: u64,
    pub canceled_before_io: u64,
    pub canceled_in_flight: u64,
    pub canceled_ready: u64,
    pub discarded_read_bytes: u64,
    pub max_ready_records: usize,
    pub max_outstanding_records: usize,
    pub staged_records: u64,
    pub staged_copy_ms: f64,
    pub staged_validate_ms: f64,
    pub staged_sync_ms: f64,
    pub staged_import_ms: f64,
    pub staging_main_takeovers: u64,
}
pub(super) struct Loaded {
    pub record: Record,
    pub staged: Option<StageSlot>,
}
pub(super) struct StageSlot {
    pub slot: usize,
    pub bank: usize,
    pool: Arc<Mutex<Vec<usize>>>,
}
impl Drop for StageSlot {
    fn drop(&mut self) {
        self.pool.lock().unwrap().push(self.slot);
    }
}
#[derive(Clone)]
struct Stager {
    target: ImportTarget,
    bank: usize,
    pool: Arc<Mutex<Vec<usize>>>,
}
struct ImportJob {
    key: Key,
    generation: u64,
    record: Record,
    target: ImportTarget,
    lease: StageSlot,
}
#[derive(Default)]
struct State {
    demand: VecDeque<Key>,
    speculative: VecDeque<Key>,
    pending: HashSet<Key>,
    in_flight: HashMap<Key, u64>,
    generation: u64,
    predictions: HashSet<Key>,
    ready: HashMap<Key, Result<Loaded, String>>,
    ram_ready: HashSet<Key>,
    staging: HashMap<usize, Stager>,
    staging_requests: HashSet<Key>,
    imports: VecDeque<ImportJob>,
    stopped: bool,
    stats: Stats,
}
type Shared = Arc<(Mutex<State>, Condvar)>;
pub struct Reader {
    shared: Shared,
    workers: Vec<JoinHandle<()>>,
    timeline: Timeline,
    record_stride: u64,
}
impl Reader {
    pub fn new(store: Arc<Store>, timeline: Timeline) -> Result<Self> {
        Self::start(store, timeline, |builder, task| builder.spawn(task))
    }

    fn start(
        store: Arc<Store>,
        timeline: Timeline,
        mut spawn: impl FnMut(
            std::thread::Builder,
            Box<dyn FnOnce() + Send>,
        ) -> std::io::Result<JoinHandle<()>>,
    ) -> Result<Self> {
        let record_stride = store.record_stride() as u64;
        // Own each worker immediately so a later spawn failure stops and joins it.
        let mut reader = Self {
            shared: Shared::default(),
            workers: Vec::with_capacity(IO_WORKERS + 1),
            timeline,
            record_stride,
        };
        for thread in 1..=IO_WORKERS {
            let work = reader.shared.clone();
            let trace = reader.timeline.clone();
            let store = store.clone();
            let worker = spawn(
                std::thread::Builder::new().name(format!("expert-io-{thread}")),
                Box::new(move || {
                    let (lock, wake) = &*work;
                    loop {
                        let (keys, prefetch, generation) = {
                            let mut s = lock.lock().unwrap();
                            while !s.stopped
                                && (s.ready.len() + s.in_flight.len() >= READ_BUFFER_LIMIT
                                    || (s.demand.is_empty() && s.speculative.is_empty()))
                            {
                                s = wake.wait(s).unwrap();
                            }
                            if s.stopped {
                                break;
                            }
                            let mut keys = Vec::new();
                            let prefetch = if let Some(first) = s.demand.pop_front() {
                                keys.push(first);
                                // Strata's file-backed stager claims up to eight
                                // neighbouring jobs in one batch. Never read gaps.
                                while keys.len() < 8
                                    && s.ready.len() + s.in_flight.len() + keys.len()
                                        < READ_BUFFER_LIMIT
                                {
                                    let previous = *keys.last().unwrap();
                                    if s.demand.front() != Some(&(previous.0, previous.1 + 1)) {
                                        break;
                                    }
                                    keys.push(s.demand.pop_front().unwrap());
                                }
                                false
                            } else {
                                keys.push(s.speculative.pop_front().unwrap());
                                true
                            };
                            let generation = s.generation;
                            for &key in &keys {
                                s.in_flight.insert(key, generation);
                                trace.mark_thread("read issued", thread, || {
                                    json!({
                                "layer":key.0,"expert":key.1,"prefetch":prefetch,
                                "queued_demands":s.demand.len(),"batch_records":keys.len()})
                                });
                            }
                            s.stats.max_outstanding_records = s
                                .stats
                                .max_outstanding_records
                                .max(s.ready.len() + s.in_flight.len());
                            (keys, prefetch, generation)
                        };
                        let start = Instant::now();
                        let result = if keys.len() == 1 {
                            store.read(keys[0].0, keys[0].1).map(|record| vec![record])
                        } else {
                            store.read_batch(keys[0].0, keys[0].1, keys.len())
                        };
                        let records: Vec<_> = match result {
                            Ok(records) => records.into_iter().map(Ok).collect(),
                            Err(error) => keys.iter().map(|_| Err(format!("{error:#}"))).collect(),
                        };
                        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
                        for &key in &keys {
                            trace.span("pread", thread, start, || {
                                json!({
                            "layer":key.0,"expert":key.1,"prefetch":prefetch,
                            "bytes":store.record_stride(),"batch_records":keys.len()})
                            });
                        }
                        let mut s = lock.lock().unwrap();
                        s.stats.read_records += keys.len() as u64;
                        s.stats.read_batches += 1;
                        s.stats.read_bytes += keys.len() as u64 * store.record_stride() as u64;
                        s.stats.read_ms += elapsed_ms;
                        for (&key, record) in keys.iter().zip(records) {
                            if record.is_ok()
                                && s.generation == generation
                                && s.pending.contains(&key)
                            {
                                s.ram_ready.insert(key);
                                trace.mark_thread(
                                    "RAM ready",
                                    thread,
                                    || json!({"layer":key.0,"expert":key.1}),
                                );
                            }
                            let target = (s.generation == generation
                                && s.pending.contains(&key)
                                && s.staging_requests.contains(&key))
                            .then(|| s.staging.get(&key.0))
                            .flatten()
                            .and_then(|stager| {
                                let slot = stager.pool.lock().unwrap().pop()?;
                                Some((
                                    stager.target,
                                    StageSlot {
                                        slot,
                                        bank: stager.bank,
                                        pool: stager.pool.clone(),
                                    },
                                ))
                            });
                            match (record, target) {
                                (Ok(record), Some((target, lease))) => {
                                    s.imports.push_back(ImportJob {
                                        key,
                                        generation,
                                        record,
                                        target,
                                        lease,
                                    });
                                    trace.mark_thread("stage queued", thread, || {
                                        json!({
                                            "layer":key.0,"expert":key.1,"queue":s.imports.len()
                                        })
                                    });
                                }
                                (record, _) => publish(
                                    &mut s,
                                    key,
                                    generation,
                                    record.map(|record| Loaded {
                                        record,
                                        staged: None,
                                    }),
                                    &trace,
                                    thread,
                                    store.record_stride() as u64,
                                ),
                            }
                        }
                        wake.notify_all();
                    }
                }),
            )?;
            reader.workers.push(worker);
        }
        let work = reader.shared.clone();
        let trace = reader.timeline.clone();
        reader.workers.push(spawn(std::thread::Builder::new().name("expert-import".into()),Box::new(move || {
            let (lock,wake) = &*work;
            loop {
                let job = {
                    let mut s = lock.lock().unwrap();
                    while !s.stopped && s.imports.is_empty() { s = wake.wait(s).unwrap(); }
                    if s.stopped { break; }
                    let job = s.imports.pop_front().unwrap();
                    if s.generation != job.generation || !s.pending.contains(&job.key)
                        || !s.staging_requests.contains(&job.key) {
                        publish(&mut s,job.key,job.generation,Ok(Loaded {record:job.record,staged:None}),
                            &trace,3,record_stride);
                        wake.notify_all();
                        continue;
                    }
                    job
                };
                let start = Instant::now();
                // The independent slot is exclusively loaned until promotion or discard.
                // Reader::drop joins this worker before its owning bank/context drops.
                let imported = unsafe {job.target.import(job.lease.slot,job.record.as_bytes())};
                trace.span("stage import",3,start,||json!({
                    "layer":job.key.0,"expert":job.key.1,"slot":job.lease.slot,"staging_bank":job.lease.bank,
                    "copy_ms":imported.as_ref().ok().map(|t|t.copy_ms),
                    "validate_ms":imported.as_ref().ok().map(|t|t.validate_ms),
                    "sync_ms":imported.as_ref().ok().map(|t|t.sync_ms),
                    "total_ms":imported.as_ref().ok().map(|t|t.total_ms)}));
                let mut s = lock.lock().unwrap();
                let result = imported.map(|timing| {
                    s.stats.staged_records += 1;
                    s.stats.staged_copy_ms += timing.copy_ms;
                    s.stats.staged_validate_ms += timing.validate_ms;
                    s.stats.staged_sync_ms += timing.sync_ms;
                    s.stats.staged_import_ms += timing.total_ms;
                    Loaded {record:job.record,staged:Some(job.lease)}
                }).map_err(|e|format!("{e:#}"));
                publish(&mut s,job.key,job.generation,result,&trace,3,record_stride);
                wake.notify_all();
            }
        }))?);
        Ok(reader)
    }
    pub fn worker_count(&self) -> usize {
        IO_WORKERS
    }
    pub fn register_staging(&self, layer: usize, target: ImportTarget, slots: usize) {
        let mut s = self.shared.0.lock().unwrap();
        let stager = s
            .staging
            .values()
            .find(|stager| stager.target == target)
            .cloned()
            .unwrap_or_else(|| Stager {
                target,
                bank: s
                    .staging
                    .values()
                    .map(|stager| stager.bank + 1)
                    .max()
                    .unwrap_or(0),
                pool: Arc::new(Mutex::new((0..slots).rev().collect())),
            });
        s.staging.insert(layer, stager);
    }
    pub fn predict(&self, layer: usize, experts: &[usize]) {
        let (lock, wake) = &*self.shared;
        let mut s = lock.lock().unwrap();
        s.staging_requests.retain(|key| key.0 != layer);
        for &expert in experts {
            let key = (layer, expert);
            if s.pending.insert(key) {
                s.predictions.insert(key);
                if !s.in_flight.contains_key(&key) {
                    s.speculative.push_back(key);
                }
                s.stats.prediction_requests += 1;
                self.timeline.mark(
                    "enqueue prefetch",
                    || json!({"layer":layer,"expert":expert}),
                );
            }
        }
        wake.notify_all();
    }
    pub fn demand(&self, layer: usize, experts: &[usize]) {
        let wanted: HashSet<_> = experts.iter().map(|&e| (layer, e)).collect();
        let (lock, wake) = &*self.shared;
        let mut s = lock.lock().unwrap();
        // Staging is opt-in for this demand, not a property of the expert key.
        s.staging_requests.retain(|key| key.0 != layer);
        let previous = std::mem::take(&mut s.predictions);
        for key in previous {
            if key.0 != layer {
                s.predictions.insert(key);
            } else if wanted.contains(&key) {
                s.stats.predicted_demand += 1;
                if s.ram_ready.contains(&key) || s.ready.contains_key(&key) {
                    s.stats.predicted_ram_ready += 1;
                }
                if s.ready.contains_key(&key) {
                    s.stats.predicted_ready += 1;
                } else if s.in_flight.contains_key(&key) {
                    s.stats.predicted_in_flight += 1;
                } else {
                    s.stats.predicted_queued += 1;
                }
            } else {
                self.timeline.mark("cancel prefetch", || {
                    json!({"layer":key.0,"expert":key.1,
                    "ready":s.ready.contains_key(&key),"in_flight":s.in_flight.contains_key(&key)})
                });
                s.pending.remove(&key);
                s.ram_ready.remove(&key);
                if s.ready.remove(&key).is_some() {
                    s.stats.canceled_ready += 1;
                    s.stats.discarded_read_bytes += self.record_stride;
                } else if s.in_flight.contains_key(&key) {
                    s.stats.canceled_in_flight += 1;
                } else {
                    s.stats.canceled_before_io += 1;
                }
                s.stats.unused_predictions += 1;
            }
        }
        s.speculative.retain(|key| key.0 != layer);
        for &expert in experts {
            let key = (layer, expert);
            // A canceled read may still be in flight. Renew its demand even
            // when no new I/O needs to be queued, so completion is retained.
            s.pending.insert(key);
            if !s.ready.contains_key(&key)
                && !s.in_flight.contains_key(&key)
                && !s.demand.contains(&key)
            {
                s.demand.push_back(key);
                self.timeline
                    .mark("enqueue demand", || json!({"layer":layer,"expert":expert}));
            }
        }
        wake.notify_all();
    }
    /// Keep speculative reads in RAM; import only after the router confirms use.
    pub fn stage_demand(&self, layer: usize, experts: &[usize]) {
        let (lock, wake) = &*self.shared;
        let mut s = lock.lock().unwrap();
        let Some(stager) = s.staging.get(&layer).cloned() else {
            return;
        };
        let generation = s.generation;
        for &expert in experts {
            let key = (layer, expert);
            s.staging_requests.insert(key);
            let Some(Ok(loaded)) = s.ready.get(&key) else {
                continue;
            };
            if loaded.staged.is_some() {
                continue;
            }
            let Some(slot) = stager.pool.lock().unwrap().pop() else {
                continue;
            };
            let loaded = s.ready.remove(&key).unwrap().unwrap();
            s.in_flight.insert(key, generation);
            s.imports.push_back(ImportJob {
                key,
                generation,
                record: loaded.record,
                target: stager.target,
                lease: StageSlot {
                    slot,
                    bank: stager.bank,
                    pool: stager.pool.clone(),
                },
            });
            self.timeline.mark("stage queued", || {
                json!({
                    "layer":layer,"expert":expert,"queue":s.imports.len()
                })
            });
        }
        wake.notify_all();
    }
    pub fn take(&self, layer: usize, expert: usize) -> Result<Record> {
        Ok(self.take_loaded(layer, expert)?.record)
    }
    pub fn take_loaded(&self, layer: usize, expert: usize) -> Result<Loaded> {
        Ok(self.take_any(layer, &[expert])?.1)
    }
    /// Consume ready demand before waiting for another record in a bounded pool.
    pub fn take_any(&self, layer: usize, experts: &[usize]) -> Result<(usize, Loaded)> {
        let start = Instant::now();
        let (lock, wake) = &*self.shared;
        let mut s = lock.lock().unwrap();
        loop {
            if let Some(loaded) = self.take_ready(&mut s, layer, experts, start)? {
                return Ok(loaded);
            }
            if let Some(index) = s.imports.iter().position(|job| {
                job.key.0 == layer && experts.contains(&job.key.1) && job.generation == s.generation
            }) {
                // If the importer has not started, the main thread can use RAM
                // immediately instead of waiting behind speculative copies.
                let job = s.imports.remove(index).unwrap();
                let (key, expert) = (job.key, job.key.1);
                s.stats.staging_main_takeovers += 1;
                self.timeline
                    .mark("stage bypass", || json!({"layer":layer,"expert":expert}));
                publish(
                    &mut s,
                    key,
                    job.generation,
                    Ok(Loaded {
                        record: job.record,
                        staged: None,
                    }),
                    &self.timeline,
                    0,
                    self.record_stride,
                );
                continue;
            }
            let sleep = Instant::now();
            s = wake.wait(s).unwrap();
            self.timeline.span("record sleep", 0, sleep, || {
                json!({
                    "layer":layer,"wanted":experts,"ready":s.ready.len(),"imports":s.imports.len()
                })
            });
        }
    }
    /// Drain published records without waiting for I/O or taking over queued imports.
    pub fn try_take_any(&self, layer: usize, experts: &[usize]) -> Result<Option<(usize, Loaded)>> {
        let start = Instant::now();
        let mut s = self.shared.0.lock().unwrap();
        self.take_ready(&mut s, layer, experts, start)
    }
    fn take_ready(
        &self,
        s: &mut State,
        layer: usize,
        experts: &[usize],
        start: Instant,
    ) -> Result<Option<(usize, Loaded)>> {
        let Some(&expert) = experts
            .iter()
            .find(|&&id| s.ready.contains_key(&(layer, id)))
        else {
            return Ok(None);
        };
        let key = (layer, expert);
        let result = s.ready.remove(&key).unwrap();
        s.pending.remove(&key);
        s.ram_ready.remove(&key);
        s.staging_requests.remove(&key);
        s.stats.wait_ms += start.elapsed().as_secs_f64() * 1000.;
        self.timeline.span("wait record", 0, start, || {
            json!({"layer":layer,"expert":expert,"wanted":experts,
                "staged":result.as_ref().ok().is_some_and(|l|l.staged.is_some())})
        });
        self.shared.1.notify_all();
        result
            .map(|loaded| Some((expert, loaded)))
            .map_err(|e| anyhow!(e))
    }
    pub fn stats(&self) -> Stats {
        self.shared.0.lock().unwrap().stats.clone()
    }
    pub fn reset(&self) {
        self.timeline.mark("reset", || json!({}));
        let (lock, wake) = &*self.shared;
        let mut s = lock.lock().unwrap();
        s.generation += 1;
        s.in_flight.clear();
        s.demand.clear();
        s.speculative.clear();
        s.pending.clear();
        s.predictions.clear();
        s.ram_ready.clear();
        s.staging_requests.clear();
        s.stats.discarded_read_bytes +=
            (s.ready.len() + s.imports.len()) as u64 * self.record_stride;
        s.ready.clear();
        s.imports.clear();
        wake.notify_all();
    }
}
fn publish(
    s: &mut State,
    key: Key,
    generation: u64,
    result: Result<Loaded, String>,
    trace: &Timeline,
    thread: usize,
    stride: u64,
) {
    if s.in_flight.get(&key) == Some(&generation) {
        s.in_flight.remove(&key);
    }
    // Completion from an old generation cannot satisfy a request after reset.
    if s.generation == generation && s.pending.contains(&key) {
        let staged = result.as_ref().ok().is_some_and(|l| l.staged.is_some());
        s.ready.insert(key, result);
        s.stats.max_ready_records = s.stats.max_ready_records.max(s.ready.len());
        trace.mark_thread(
            "read ready",
            thread,
            || json!({"layer":key.0,"expert":key.1,"staged":staged}),
        );
    } else {
        if s.generation == generation {
            s.ram_ready.remove(&key);
        }
        s.stats.discarded_read_bytes += stride;
        trace.mark_thread(
            "read discarded",
            thread,
            || json!({"layer":key.0,"expert":key.1}),
        );
    }
}
impl Drop for Reader {
    fn drop(&mut self) {
        let (lock, wake) = &*self.shared;
        lock.lock().unwrap().stopped = true;
        wake.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
        if let Err(error) = self.timeline.save() {
            eprintln!("save expert timeline: {error:#}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::experts::store::{Geometry, Writer};

    #[test]
    fn worker_spawn_failure_joins_started_readers() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let geometry = Geometry {
            native_version: 1,
            layers: 1,
            experts: 1,
            hidden: 64,
            intermediate: 64,
            down_bits: 4,
        };
        let mut writer = Writer::create(dir.path(), geometry.clone(), vec![], 4096)?;
        writer.write(0, 0, &[0; 4096])?;
        writer.finish()?;
        let store = Arc::new(Store::open(dir.path(), &geometry, &[], false)?);
        for fail_at in [2, IO_WORKERS + 1] {
            let mut attempts = 0;
            let result = Reader::start(store.clone(), Timeline::default(), |builder, task| {
                attempts += 1;
                if attempts == fail_at {
                    Err(std::io::Error::other("injected worker spawn failure"))
                } else {
                    builder.spawn(task)
                }
            });
            assert_eq!(
                result.err().unwrap().to_string(),
                "injected worker spawn failure"
            );
            assert_eq!(attempts, fail_at);
            // Every started reader owns a Store clone until its thread exits.
            assert_eq!(Arc::strong_count(&store), 1);
        }
        Ok(())
    }

    #[test]
    fn consuming_a_record_clears_its_staging_request_before_reuse() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let geometry = Geometry {
            native_version: 1,
            layers: 1,
            experts: 1,
            hidden: 64,
            intermediate: 64,
            down_bits: 4,
        };
        let mut writer = Writer::create(dir.path(), geometry.clone(), vec![], 4096)?;
        writer.write(0, 0, &[7; 4096])?;
        writer.finish()?;
        let reader = Reader::new(
            Arc::new(Store::open(dir.path(), &geometry, &[], false)?),
            Timeline::default(),
        )?;
        reader.demand(0, &[0]);
        reader
            .shared
            .0
            .lock()
            .unwrap()
            .staging_requests
            .insert((0, 0));
        assert_eq!(reader.take(0, 0)?.as_bytes(), &[7; 4096]);
        assert!(reader.shared.0.lock().unwrap().staging_requests.is_empty());
        // Appended prefill requests the same evicted expert without staging.
        reader.demand(0, &[0]);
        assert!(reader.shared.0.lock().unwrap().staging_requests.is_empty());
        assert_eq!(reader.take(0, 0)?.as_bytes(), &[7; 4096]);
        Ok(())
    }

    #[test]
    fn parallel_reads_stay_bounded_across_reset_and_cancellation() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let geometry = Geometry {
            native_version: 1,
            layers: 2,
            experts: 96,
            hidden: 64,
            intermediate: 64,
            down_bits: 4,
        };
        let mut writer = Writer::create(dir.path(), geometry.clone(), vec![], 4096)?;
        for layer in 0..2 {
            for expert in 0..96 {
                writer.write(layer, expert, &vec![(layer * 96 + expert) as u8; 4096])?;
            }
        }
        writer.finish()?;
        let store = Arc::new(Store::open(dir.path(), &geometry, &[], false)?);
        let reader = Reader::new(store.clone(), Timeline::default())?;
        for pass in 0..8 {
            reader.predict(0, &[0, 1, 2, 3, 4, 5, 6, 7]);
            reader.reset();
            let layer = pass % 2;
            let demand: Vec<_> = (0..96).collect();
            reader.predict(layer, &[0, 2, 4, 6]);
            reader.demand(layer, &demand);
            for expert in demand {
                assert_eq!(
                    reader.take(layer, expert)?.as_bytes(),
                    vec![(layer * 96 + expert) as u8; 4096]
                );
            }
        }
        assert!(reader.stats().max_outstanding_records <= READ_BUFFER_LIMIT);
        assert!(store.buffer_stats().reuses > 0);
        Ok(())
    }

    #[test]
    fn batched_demand_drains_prefetched_records_before_waiting_for_unread_experts() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let geometry = Geometry {
            native_version: 1,
            layers: 1,
            experts: 96,
            hidden: 64,
            intermediate: 64,
            down_bits: 4,
        };
        let mut writer = Writer::create(dir.path(), geometry.clone(), vec![], 4096)?;
        for expert in 0..96 {
            writer.write(0, expert, &vec![expert as u8; 4096])?;
        }
        writer.finish()?;
        let store = Arc::new(Store::open(dir.path(), &geometry, &[], false)?);
        let reader = Reader::new(store, Timeline::default())?;
        let predicted: Vec<_> = (1..64).step_by(2).collect();
        reader.predict(0, &predicted);
        {
            let (lock, wake) = &*reader.shared;
            let (state, timeout) = wake
                .wait_timeout_while(
                    lock.lock().unwrap(),
                    std::time::Duration::from_secs(5),
                    |s| s.ready.len() < READ_BUFFER_LIMIT,
                )
                .unwrap();
            assert!(!timeout.timed_out());
            assert_eq!(state.ready.len(), READ_BUFFER_LIMIT);
        }
        let mut remaining: Vec<_> = (0..96).collect();
        reader.demand(0, &remaining);
        // Nonblocking drains must consume the published prediction while other
        // demanded records may still be queued or in flight.
        let (expert, loaded) = reader.try_take_any(0, &remaining)?.unwrap();
        assert!(predicted.contains(&expert));
        assert_eq!(loaded.record.as_bytes(), vec![expert as u8; 4096]);
        remaining.retain(|&id| id != expert);
        while !remaining.is_empty() {
            let (expert, loaded) = reader.take_any(0, &remaining)?;
            assert_eq!(loaded.record.as_bytes(), vec![expert as u8; 4096]);
            remaining.retain(|&id| id != expert);
        }
        assert!(reader.stats().max_outstanding_records <= READ_BUFFER_LIMIT);
        Ok(())
    }

    #[test]
    fn nonblocking_take_preserves_unready_demand_and_reports_read_errors() -> Result<()> {
        let reader = Reader {
            shared: Shared::default(),
            workers: Vec::new(),
            timeline: Timeline::default(),
            record_stride: 4096,
        };
        reader.demand(0, &[2, 3]);
        assert!(reader.try_take_any(0, &[2, 3])?.is_none());
        {
            let mut s = reader.shared.0.lock().unwrap();
            assert_eq!(s.demand, VecDeque::from([(0, 2), (0, 3)]));
            assert!(s.pending.contains(&(0, 2)));
            s.demand.pop_front();
            s.ready.insert((0, 2), Err("test read failure".into()));
            s.staging_requests.insert((0, 2));
        }
        assert_eq!(
            reader.try_take_any(0, &[2, 3]).err().unwrap().to_string(),
            "test read failure"
        );
        let s = reader.shared.0.lock().unwrap();
        assert!(!s.pending.contains(&(0, 2)));
        assert!(!s.staging_requests.contains(&(0, 2)));
        assert!(s.pending.contains(&(0, 3)));
        assert_eq!(s.demand, VecDeque::from([(0, 3)]));
        Ok(())
    }

    #[test]
    fn canceled_in_flight_prediction_can_be_requested_again() {
        for predict_again in [false, true] {
            let key = (0, 2);
            let mut state = State::default();
            state.pending.insert(key);
            state.predictions.insert(key);
            state.in_flight.insert(key, 0);
            let reader = Reader {
                shared: Arc::new((Mutex::new(state), Condvar::new())),
                workers: Vec::new(),
                timeline: Timeline::default(),
                record_stride: 4096,
            };
            reader.demand(0, &[]);
            assert!(!reader.shared.0.lock().unwrap().pending.contains(&key));
            if predict_again {
                reader.predict(0, &[2]);
            }
            reader.demand(0, &[2]);
            {
                let state = reader.shared.0.lock().unwrap();
                assert!(state.pending.contains(&key));
                assert!(state.in_flight.contains_key(&key));
                assert!(state.demand.is_empty());
                assert!(state.speculative.is_empty());
            }
            // Reset gives the same key a new generation and must queue a new
            // read rather than adopting the old generation's in-flight I/O.
            reader.reset();
            reader.demand(0, &[2]);
            let state = reader.shared.0.lock().unwrap();
            assert_eq!(state.generation, 1);
            assert!(state.pending.contains(&key));
            assert!(state.in_flight.is_empty());
            assert_eq!(state.demand, VecDeque::from([key]));
        }
    }

    #[test]
    fn prediction_states_and_completed_cancellations_are_accounted() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let geometry = Geometry {
            native_version: 1,
            layers: 1,
            experts: 6,
            hidden: 64,
            intermediate: 64,
            down_bits: 4,
        };
        let mut writer = Writer::create(dir.path(), geometry.clone(), vec![], 4096)?;
        for expert in 0..6 {
            writer.write(0, expert, &[0; 4096])?;
        }
        writer.finish()?;
        let store = Store::open(dir.path(), &geometry, &[], false)?;
        let mut state = State::default();
        state.predictions.extend((0..6).map(|e| (0, e)));
        state.pending = state.predictions.clone();
        state.ready.insert(
            (0, 0),
            Ok(Loaded {
                record: store.read(0, 0)?,
                staged: None,
            }),
        );
        state.ready.insert(
            (0, 3),
            Ok(Loaded {
                record: store.read(0, 3)?,
                staged: None,
            }),
        );
        state.in_flight.extend([((0, 1), 0), ((0, 4), 0)]);
        state.staging_requests.extend([(0, 3), (1, 0)]);
        state.speculative.extend([(0, 2), (0, 5)]);
        let reader = Reader {
            shared: Arc::new((Mutex::new(state), Condvar::new())),
            workers: Vec::new(),
            timeline: Timeline::default(),
            record_stride: store.record_stride() as u64,
        };
        reader.demand(0, &[3, 4, 5]);
        assert_eq!(
            reader.shared.0.lock().unwrap().staging_requests,
            HashSet::from([(1, 0)])
        );
        let stats = reader.stats();
        assert_eq!(stats.predicted_demand, 3);
        assert_eq!(stats.predicted_ready, 1);
        assert_eq!(stats.predicted_in_flight, 1);
        assert_eq!(stats.predicted_queued, 1);
        assert_eq!(stats.unused_predictions, 3);
        assert_eq!(stats.canceled_before_io, 1);
        assert_eq!(stats.canceled_in_flight, 1);
        assert_eq!(stats.canceled_ready, 1);
        assert_eq!(stats.discarded_read_bytes, store.record_stride() as u64);
        reader.reset();
        assert_eq!(
            reader.stats().discarded_read_bytes,
            2 * store.record_stride() as u64
        );
        Ok(())
    }
    #[test]
    fn predictions_promote_to_demand_and_future_layer_reads_survive() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let geometry = Geometry {
            native_version: 1,
            layers: 2,
            experts: 4,
            hidden: 64,
            intermediate: 64,
            down_bits: 4,
        };
        let mut writer = Writer::create(dir.path(), geometry.clone(), vec![], 4096)?;
        for layer in 0..2 {
            for expert in 0..4 {
                writer.write(layer, expert, &vec![(layer * 4 + expert) as u8; 4096])?;
            }
        }
        writer.finish()?;
        let reader = Reader::new(
            Arc::new(Store::open(dir.path(), &geometry, &[], false)?),
            Timeline::default(),
        )?;
        reader.predict(0, &[0, 1, 2]);
        reader.predict(1, &[1]);
        reader.demand(0, &[2, 3]);
        reader.demand(0, &[2, 3]);
        assert_eq!(reader.take(0, 3)?.as_bytes(), vec![3; 4096]);
        assert_eq!(reader.take(0, 2)?.as_bytes(), vec![2; 4096]);
        reader.demand(1, &[1]);
        assert_eq!(reader.take(1, 1)?.as_bytes(), vec![5; 4096]);
        assert_eq!(reader.stats().predicted_demand, 2);
        assert_eq!(reader.stats().unused_predictions, 2);
        std::fs::OpenOptions::new()
            .write(true)
            .open(dir.path().join("experts.npu"))?
            .set_len(4096)?;
        reader.demand(1, &[3]);
        assert!(reader.take(1, 3).is_err());
        Ok(())
    }
}
