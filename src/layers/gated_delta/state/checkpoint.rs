use super::*;

/// Committed recurrent state and causal convolution history at one boundary.
pub struct Snapshot {
    dim: GdnDimensions,
    kernel: usize,
    state: Vec<f32>,
    conv: ConvState,
    position: usize,
}

impl Snapshot {
    pub fn bytes(&self) -> usize {
        self.state.len() * 4 + self.conv.allocated_bytes()
    }
}

impl LayerState {
    pub fn snapshot(&self) -> Result<Snapshot> {
        ensure!(
            self.pending.is_none(),
            "cannot snapshot tentative GDN state"
        );
        Ok(Snapshot {
            dim: self.dim,
            kernel: self.kernel,
            state: self.state.clone(),
            conv: self.conv.clone(),
            position: self.position,
        })
    }

    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        ensure!(
            self.pending.is_none(),
            "GDN verification transaction is active"
        );
        ensure!(
            self.dim == snapshot.dim && self.kernel == snapshot.kernel,
            "GDN snapshot shape mismatch"
        );
        self.state.clone_from(&snapshot.state);
        self.conv.clone_from(&snapshot.conv);
        self.position = snapshot.position;
        Ok(())
    }
}
