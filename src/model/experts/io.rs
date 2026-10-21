//! Bounded positional reads with Strata-style adjacent-job batches and generation ownership.
//! Demand reads precede speculative reads; see Strata-LICENSE.
use super::store::{Record, Store};
use super::timeline::Timeline;
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
    pub predicted_in_flight: u64,
    pub predicted_queued: u64,
    pub unused_predictions: u64,
    pub canceled_before_io: u64,
    pub canceled_in_flight: u64,
    pub canceled_ready: u64,
    pub discarded_read_bytes: u64,
    pub max_ready_records: usize,
}
#[derive(Default)]
struct State {
    demand: VecDeque<Key>,
    speculative: VecDeque<Key>,
    pending: HashSet<Key>,
    in_flight: HashSet<Key>,
    generation: u64,
    predictions: HashSet<Key>,
    ready: HashMap<Key, Result<Record, String>>,
    stopped: bool,
    stats: Stats,
}
type Shared = Arc<(Mutex<State>, Condvar)>;
pub struct Reader {
    shared: Shared,
    worker: Option<JoinHandle<()>>,
    timeline: Timeline,
    record_stride: u64,
}
impl Reader {
    pub fn new(store: Arc<Store>, timeline: Timeline) -> Result<Self> {
        let record_stride = store.record_stride() as u64;
        let shared = Shared::default();
        let work = shared.clone();
        let trace = timeline.clone();
        let worker = std::thread::Builder::new()
            .name("expert-io".into())
            .spawn(move || {
                let (lock, wake) = &*work;
                loop {
                    let (keys, prefetch, generation) = {
                        let mut s = lock.lock().unwrap();
                        while !s.stopped
                            && (s.ready.len() >= 32
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
                            while keys.len() < 8 && s.ready.len() + keys.len() < 32 {
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
                        for &key in &keys {
                            s.in_flight.insert(key);
                            trace.mark_thread("read issued", 1, || {
                                json!({
                                "layer":key.0,"expert":key.1,"prefetch":prefetch,
                                "queued_demands":s.demand.len(),"batch_records":keys.len()})
                            });
                        }
                        (keys, prefetch, s.generation)
                    };
                    let start = Instant::now();
                    let result = if keys.len() == 1 {
                        store.read(keys[0].0, keys[0].1).map(|record| vec![record])
                    } else {
                        store.read_batch(keys[0].0, keys[0].1, keys.len())
                    };
                    let results: Vec<_> = match result {
                        Ok(records) => records.into_iter().map(Ok).collect(),
                        Err(error) => keys.iter().map(|_| Err(format!("{error:#}"))).collect(),
                    };
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
                    for &key in &keys {
                        trace.span("pread", 1, start, || {
                            json!({
                            "layer":key.0,"expert":key.1,"prefetch":prefetch,
                            "bytes":store.record_stride(),"batch_records":keys.len()})
                        });
                    }
                    let mut s = lock.lock().unwrap();
                    s.in_flight.clear();
                    s.stats.read_records += keys.len() as u64;
                    s.stats.read_batches += 1;
                    s.stats.read_bytes += keys.len() as u64 * store.record_stride() as u64;
                    s.stats.read_ms += elapsed_ms;
                    for (key, result) in keys.into_iter().zip(results) {
                        // A reset owns a new generation. An old read cannot
                        // satisfy a same-key request submitted after that reset.
                        if s.generation == generation && s.pending.contains(&key) {
                            s.ready.insert(key, result);
                            s.stats.max_ready_records =
                                s.stats.max_ready_records.max(s.ready.len());
                            trace.mark_thread(
                                "read ready",
                                1,
                                || json!({"layer":key.0,"expert":key.1}),
                            );
                        } else {
                            s.stats.discarded_read_bytes += store.record_stride() as u64;
                            trace.mark_thread(
                                "read discarded",
                                1,
                                || json!({"layer":key.0,"expert":key.1}),
                            );
                        }
                    }
                    wake.notify_all();
                }
            })?;
        Ok(Self {
            shared,
            worker: Some(worker),
            timeline,
            record_stride,
        })
    }
    pub fn predict(&self, layer: usize, experts: &[usize]) {
        let (lock, wake) = &*self.shared;
        let mut s = lock.lock().unwrap();
        for &expert in experts {
            let key = (layer, expert);
            if s.pending.insert(key) {
                s.predictions.insert(key);
                if !s.in_flight.contains(&key) {
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
        let previous = std::mem::take(&mut s.predictions);
        for key in previous {
            if key.0 != layer {
                s.predictions.insert(key);
            } else if wanted.contains(&key) {
                s.stats.predicted_demand += 1;
                if s.ready.contains_key(&key) {
                    s.stats.predicted_ready += 1;
                } else if s.in_flight.contains(&key) {
                    s.stats.predicted_in_flight += 1;
                } else {
                    s.stats.predicted_queued += 1;
                }
            } else {
                self.timeline.mark("cancel prefetch", || {
                    json!({"layer":key.0,"expert":key.1,
                    "ready":s.ready.contains_key(&key),"in_flight":s.in_flight.contains(&key)})
                });
                s.pending.remove(&key);
                if s.ready.remove(&key).is_some() {
                    s.stats.canceled_ready += 1;
                    s.stats.discarded_read_bytes += self.record_stride;
                } else if s.in_flight.contains(&key) {
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
                && !s.in_flight.contains(&key)
                && !s.demand.contains(&key)
            {
                s.demand.push_back(key);
                self.timeline
                    .mark("enqueue demand", || json!({"layer":layer,"expert":expert}));
            }
        }
        wake.notify_all();
    }
    pub fn take(&self, layer: usize, expert: usize) -> Result<Record> {
        let start = Instant::now();
        let key = (layer, expert);
        let (lock, wake) = &*self.shared;
        let mut s = lock.lock().unwrap();
        loop {
            if let Some(result) = s.ready.remove(&key) {
                s.pending.remove(&key);
                s.stats.wait_ms += start.elapsed().as_secs_f64() * 1000.;
                self.timeline.span(
                    "wait record",
                    0,
                    start,
                    || json!({"layer":layer,"expert":expert}),
                );
                wake.notify_all();
                return result.map_err(|e| anyhow!(e));
            }
            s = wake.wait(s).unwrap();
        }
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
        s.stats.discarded_read_bytes += s.ready.len() as u64 * self.record_stride;
        s.ready.clear();
        wake.notify_all();
    }
}
impl Drop for Reader {
    fn drop(&mut self) {
        let (lock, wake) = &*self.shared;
        lock.lock().unwrap().stopped = true;
        wake.notify_all();
        if let Some(worker) = self.worker.take() {
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
    fn canceled_in_flight_prediction_can_be_requested_again() {
        for predict_again in [false, true] {
            let key = (0, 2);
            let mut state = State::default();
            state.pending.insert(key);
            state.predictions.insert(key);
            state.in_flight.insert(key);
            let reader = Reader {
                shared: Arc::new((Mutex::new(state), Condvar::new())),
                worker: None,
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
                assert!(state.in_flight.contains(&key));
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
        state.ready.insert((0, 0), Ok(store.read(0, 0)?));
        state.ready.insert((0, 3), Ok(store.read(0, 3)?));
        state.in_flight.extend([(0, 1), (0, 4)]);
        state.speculative.extend([(0, 2), (0, 5)]);
        let reader = Reader {
            shared: Arc::new((Mutex::new(state), Condvar::new())),
            worker: None,
            timeline: Timeline::default(),
            record_stride: store.record_stride() as u64,
        };
        reader.demand(0, &[3, 4, 5]);
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
