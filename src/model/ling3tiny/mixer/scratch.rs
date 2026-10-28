//! Layer-independent host buffers; recurrent state and MLA KV stay layer-owned.
#[derive(Default)]
pub(crate) struct Scratch {
    pub(super) attended: Vec<f32>,
    pub(super) kda: Kda,
    pub(super) mla: Mla,
}

#[derive(Default)]
pub(super) struct Kda {
    pub projected: Vec<f32>,
    pub projection: Vec<f32>,
    pub qkv: [Vec<f32>; 3],
    pub decay: Vec<f32>,
    pub beta: Vec<f32>,
}

impl Kda {
    pub fn resize(&mut self, rows: usize, scratch_width: usize) {
        self.projected.resize(rows * 5 * 2048, 0.);
        self.projection.resize(rows * scratch_width, 0.);
        for values in &mut self.qkv {
            values.resize(rows * 2048, 0.);
        }
        self.decay.resize(rows * 2048, 0.);
        self.beta.resize(rows * 16, 0.);
    }
}

#[derive(Default)]
pub(super) struct Mla {
    pub qa: Vec<f32>,
    pub normalized_q: Vec<f32>,
    pub query: Vec<f32>,
    pub kv: Vec<f32>,
    pub latent: Vec<f32>,
    pub rotary: Vec<f32>,
    pub gates: Vec<f32>,
}

impl Mla {
    pub fn resize(&mut self, rows: usize) {
        self.qa.resize(rows * 256, 0.);
        self.normalized_q.resize(rows * 256, 0.);
        self.query.resize(rows * 3072, 0.);
        self.kv.resize(rows * 576, 0.);
        self.latent.resize(rows * 512, 0.);
        self.rotary.resize(rows * 64, 0.);
        self.gates.resize(rows * 16, 0.);
    }
}

impl Scratch {
    pub fn allocated_bytes(&self) -> usize {
        [
            &self.attended,
            &self.kda.projected,
            &self.kda.projection,
            &self.kda.decay,
            &self.kda.beta,
            &self.mla.qa,
            &self.mla.normalized_q,
            &self.mla.query,
            &self.mla.kv,
            &self.mla.latent,
            &self.mla.rotary,
            &self.mla.gates,
        ]
        .into_iter()
        .chain(&self.kda.qkv)
        .map(|values| values.capacity() * 4)
        .sum()
    }
}
