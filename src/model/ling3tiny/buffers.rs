//! Reused execution buffers; no layer's persistent state is stored here.
use super::{config::Config, mixer, moe};

pub(super) struct Buffers {
    pub hidden: Vec<f32>,
    pub normalized: Vec<f32>,
    pub output: Vec<f32>,
    pub head: Vec<f32>,
    pub mixer: mixer::Scratch,
    pub ffn: moe::Scratch,
}

impl Buffers {
    pub fn new(config: &Config) -> Self {
        Self {
            hidden: Vec::new(),
            normalized: Vec::new(),
            output: Vec::new(),
            head: vec![0.; config.hidden],
            mixer: Default::default(),
            ffn: moe::Scratch::new(config),
        }
    }

    pub fn resize(&mut self, rows: usize, hidden: usize) {
        self.hidden.resize(rows * hidden, 0.);
        self.normalized.resize(rows * hidden, 0.);
        self.output.resize(rows * hidden, 0.);
    }

    pub fn allocated_bytes(&self) -> usize {
        [&self.hidden, &self.normalized, &self.output, &self.head]
            .iter()
            .map(|values| values.capacity() * 4)
            .sum::<usize>()
            + self.mixer.allocated_bytes()
            + self.ffn.allocated_bytes()
    }
}
