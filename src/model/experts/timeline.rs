//! Opt-in Chrome/Perfetto timeline, shared by the submit and I/O threads.
use anyhow::Result;
use serde::Serialize;
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

#[derive(Serialize)]
struct Capture<'a> {
    #[serde(rename = "traceEvents")]
    events: &'a [Value],
    #[serde(rename = "displayTimeUnit")]
    unit: &'static str,
}
impl Timeline {
    pub(crate) fn enabled(&self) -> bool {
        self.0.is_some()
    }
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
            let mut writer = std::io::BufWriter::new(std::fs::File::create(&r.path)?);
            serde_json::to_writer(
                &mut writer,
                &Capture {
                    events: &events,
                    unit: "ms",
                },
            )?;
            std::io::Write::flush(&mut writer)?;
        }
        Ok(())
    }
}
