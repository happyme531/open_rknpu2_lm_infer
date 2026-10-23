//! Target block outputs used as context by an external speculative drafter.
use super::runtime::Qwen35;
use crate::model::LanguageModel;
use anyhow::{Result, ensure};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

#[derive(Default)]
pub(super) struct Features {
    layers: Vec<usize>,
    hidden: usize,
    pending: Vec<f32>,
    committed: Vec<f32>,
}

impl Features {
    pub fn configure(&mut self, layers: &[usize], hidden: usize, depth: usize) -> Result<()> {
        ensure!(
            layers.iter().all(|&i| i < depth),
            "feature layer outside target"
        );
        ensure!(
            layers.windows(2).all(|p| p[0] < p[1]),
            "feature layers must increase"
        );
        self.layers = layers.to_vec();
        self.hidden = hidden;
        self.reset();
        Ok(())
    }
    pub fn begin(&mut self, rows: usize) {
        self.pending
            .resize(rows * self.hidden * self.layers.len(), 0.);
    }
    pub fn record(&mut self, layer: usize, values: &[f32]) {
        if let Some(column) = self.layers.iter().position(|&i| i == layer) {
            let width = self.hidden * self.layers.len();
            for (source, dest) in values
                .chunks_exact(self.hidden)
                .zip(self.pending.chunks_exact_mut(width))
            {
                dest[column * self.hidden..(column + 1) * self.hidden].copy_from_slice(source);
            }
        }
    }
    pub fn commit(&mut self, rows: usize) {
        let count = rows * self.hidden * self.layers.len();
        self.committed.extend_from_slice(&self.pending[..count]);
        self.pending.clear();
    }
    pub fn abort(&mut self) {
        self.pending.clear();
    }
    pub fn reset(&mut self) {
        self.pending.clear();
        self.committed.clear();
    }
}

impl Qwen35 {
    pub(super) fn load_dflash(&mut self) -> Result<()> {
        ensure!(
            !self
                .options
                .path
                .join("mtp_coordinate_adapter.json")
                .exists(),
            "DFlash requires an unrotated Qwen3.5 checkpoint, such as C4"
        );
        let start = Instant::now();
        let path = self
            .options
            .speculation
            .model
            .as_ref()
            .expect("validated draft path");
        let draft = crate::model::dflash::DFlash::load(&self.options, path)?;
        ensure!(
            draft.config.hidden_size == self.config.hidden_size
                && draft.config.vocab_size == self.info.vocab_size,
            "DFlash target hidden size or vocabulary mismatch"
        );
        ensure!(
            self.options.speculation.window() < draft.config.dflash_config.block_size,
            "spec-draft-n-max exceeds the draft checkpoint block size"
        );
        self.capture_features(&draft.config.dflash_config.target_layer_ids)?;
        self.dflash = Some(draft);
        self.load_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
    pub(super) fn flush_features(&mut self, cancel: Option<&AtomicBool>) -> Result<()> {
        if self.dflash.is_some() {
            let features = self.take_features();
            self.dflash
                .as_mut()
                .unwrap()
                .prepare_context(&features, cancel)?;
        }
        Ok(())
    }
    pub(super) fn dflash_logits(
        &mut self,
        anchor: u32,
        drafts: usize,
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<f32>>> {
        if self.dflash.is_none() {
            return Ok(None);
        }
        ensure!(
            drafts > 0 && drafts <= self.speculation_window() && self.round.is_none(),
            "invalid DFlash draft window"
        );
        let mask = self
            .dflash
            .as_ref()
            .unwrap()
            .config
            .dflash_config
            .mask_token_id;
        let mut tokens = vec![mask; drafts + 1];
        tokens[0] = anchor;
        let embeddings = self.token_embeddings(&tokens)?;
        let hidden =
            self.dflash
                .as_mut()
                .unwrap()
                .draft(&[], &embeddings, self.position, cancel)?;
        let h = self.config.hidden_size;
        Ok(Some(self.project_hidden(&hidden[h..], drafts)?))
    }
    pub fn capture_features(&mut self, layers: &[usize]) -> Result<()> {
        ensure!(
            self.position() == 0 && self.round.is_none(),
            "configure target features before prefill"
        );
        self.features
            .configure(layers, self.config.hidden_size, self.layers.len())
    }
    /// Token-major concatenation of the selected post-block states. Tentative
    /// verification rows become visible only after their prefix is committed.
    pub fn take_features(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.features.committed)
    }
    pub fn token_embeddings(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(
            tokens.iter().all(|&t| (t as usize) < self.info.vocab_size),
            "draft token outside target vocabulary"
        );
        let h = self.config.hidden_size;
        let mut values = vec![0.; tokens.len() * h];
        for (&token, row) in tokens.iter().zip(values.chunks_mut(h)) {
            self.embedding.row(token as usize, row);
        }
        Ok(values)
    }
    /// The drafter supplies its own final-normalized states; reuse the target's
    /// resident output weights without applying the target norm a second time.
    pub fn project_hidden(&mut self, hidden: &[f32], rows: usize) -> Result<Vec<f32>> {
        ensure!(
            self.round.is_none() && rows > 0,
            "invalid external head projection"
        );
        let start = Instant::now();
        let mut logits = vec![0.; rows * self.info.vocab_size];
        self.head.run(&self.cpu, hidden, &mut logits, rows)?;
        self.head_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_committed_feature_rows_escape_verification() -> Result<()> {
        let mut f = Features::default();
        f.configure(&[1, 3], 2, 4)?;
        f.begin(2);
        f.record(1, &[1., 2., 3., 4.]);
        f.record(3, &[5., 6., 7., 8.]);
        f.commit(1);
        assert_eq!(f.committed, [1., 2., 5., 6.]);
        f.begin(1);
        f.record(1, &[9., 10.]);
        f.abort();
        assert_eq!(f.committed, [1., 2., 5., 6.]);
        f.reset();
        assert!(f.committed.is_empty());
        Ok(())
    }
}
