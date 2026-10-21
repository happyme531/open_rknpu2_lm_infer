//! Opt-in Chrome/Perfetto timeline, shared by the submit and I/O threads.
use anyhow::Result;
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

#[derive(Clone, Default)]
pub struct Timeline(Option<Arc<Recording>>);
struct Recording {
    start: Instant,
    path: PathBuf,
    events: Mutex<Vec<Value>>,
}
impl Timeline {
    pub fn from_environment() -> Self {
        Self(std::env::var_os("RKNPU_EXPERT_TIMELINE").map(|path| {
            Arc::new(Recording {
                start: Instant::now(),
                path: PathBuf::from(path),
                events: Mutex::new(Vec::new()),
            })
        }))
    }
    pub fn mark(&self, name: &'static str, args: impl FnOnce() -> Value) {
        self.mark_thread(name, 0, args);
    }
    pub fn mark_thread(&self, name: &'static str, thread: usize, args: impl FnOnce() -> Value) {
        if let Some(r) = &self.0 {
            r.events
                .lock()
                .unwrap()
                .push(json!({"name":name,"ph":"i","s":"t","pid":0,"tid":thread,
                "ts":r.start.elapsed().as_secs_f64()*1e6,"args":args()}));
        }
    }
    pub fn span(
        &self,
        name: &'static str,
        thread: usize,
        start: Instant,
        args: impl FnOnce() -> Value,
    ) {
        if let Some(r) = &self.0 {
            r.events
                .lock()
                .unwrap()
                .push(json!({"name":name,"ph":"X","pid":0,"tid":thread,
                "ts":start.duration_since(r.start).as_secs_f64()*1e6,
                "dur":start.elapsed().as_secs_f64()*1e6,"args":args()}));
        }
    }
    pub fn save(&self) -> Result<()> {
        if let Some(r) = &self.0 {
            if let Some(parent) = r.path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let events = r.events.lock().unwrap();
            std::fs::write(
                &r.path,
                serde_json::to_vec(&json!({"traceEvents":*events,"displayTimeUnit":"ms"}))?,
            )?;
        }
        Ok(())
    }
}
