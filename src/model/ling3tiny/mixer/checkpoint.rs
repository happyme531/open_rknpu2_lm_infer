use super::*;
use anyhow::ensure;

impl Snapshot {
    pub fn bytes(&self) -> usize {
        self.state.allocated_bytes()
            + self
                .convolution
                .iter()
                .map(ConvState::allocated_bytes)
                .sum::<usize>()
    }
}

impl Mixer {
    pub fn snapshot(&self) -> Option<Snapshot> {
        match self {
            Self::Kda(k) => Some(Snapshot {
                state: k.state.clone(),
                convolution: k.conv.iter().map(|c| c.state.clone()).collect(),
            }),
            Self::Mla(_) => None,
        }
    }

    pub fn restore(&mut self, snapshot: Option<&Snapshot>, position: usize) -> Result<()> {
        match self {
            Self::Kda(k) => {
                let snapshot = snapshot.ok_or_else(|| anyhow::anyhow!("missing KDA checkpoint"))?;
                ensure!(
                    k.conv.len() == snapshot.convolution.len(),
                    "KDA convolution checkpoint mismatch"
                );
                k.state.restore(&snapshot.state)?;
                for (conv, history) in k.conv.iter_mut().zip(&snapshot.convolution) {
                    conv.state.restore(history)?;
                }
                Ok(())
            }
            Self::Mla(m) => m.cache.truncate(position),
        }
    }
}
