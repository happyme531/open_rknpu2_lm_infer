//! Ordered layer pipeline. Layers own committed state; callers own lane scratch.
use anyhow::{Context, Result, ensure};
use std::sync::{
    Mutex, MutexGuard,
    atomic::{AtomicBool, Ordering},
    mpsc::sync_channel,
};

pub(crate) struct Layers<T>(Vec<Mutex<T>>);

impl<T> From<Vec<T>> for Layers<T> {
    fn from(layers: Vec<T>) -> Self {
        Self(layers.into_iter().map(Mutex::new).collect())
    }
}
impl<T> Layers<T> {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn lock(&self, index: usize) -> MutexGuard<'_, T> {
        self.0[index].lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn iter(&self) -> impl Iterator<Item = MutexGuard<'_, T>> {
        self.0
            .iter()
            .map(|layer| layer.lock().unwrap_or_else(|e| e.into_inner()))
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0
            .iter_mut()
            .map(|layer| layer.get_mut().unwrap_or_else(|e| e.into_inner()))
    }
}

/// A follower enters layer i only after its predecessor has completed that
/// layer. Both callbacks may concurrently execute different layers. Endpoints
/// are dropped before joining, including errors, so a blocked peer is released.
pub(crate) fn run<T: Send>(
    layers: &Layers<T>,
    cancelled: Option<&AtomicBool>,
    mut first: impl FnMut(usize, &mut T) -> Result<()> + Send,
    mut second: impl FnMut(usize, &mut T) -> Result<()> + Send,
) -> Result<()> {
    let (ready, wait) = sync_channel(0);
    let check = || -> Result<()> {
        ensure!(
            !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
            "generation cancelled"
        );
        Ok(())
    };
    std::thread::scope(|scope| {
        let follower = scope.spawn(move || -> Result<()> {
            for (index, layer) in layers.0.iter().enumerate() {
                check()?;
                wait.recv()
                    .context("first ubatch stopped before layer completion")?;
                check()?;
                second(index, &mut layer.lock().unwrap_or_else(|e| e.into_inner()))?;
            }
            Ok(())
        });
        let leader = (move || -> Result<()> {
            for (index, layer) in layers.0.iter().enumerate() {
                check()?;
                first(index, &mut layer.lock().unwrap_or_else(|e| e.into_inner()))?;
                ready
                    .send(())
                    .context("second ubatch stopped before layer completion")?;
            }
            Ok(())
        })();
        let follower = follower
            .join()
            .map_err(|_| anyhow::anyhow!("second ubatch worker panicked"))?;
        match (leader, follower) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(a), Err(b)) => anyhow::bail!("first ubatch: {a:#}; second ubatch: {b:#}"),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    })
}

#[cfg(test)]
mod tests;
