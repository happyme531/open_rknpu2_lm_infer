use super::*;
use std::cell::RefCell;

struct Binding {
    source: Backend,
    execution: Backend,
}
type Bindings = BTreeMap<usize, Binding>;
thread_local! { static CURRENT: RefCell<Option<Arc<Bindings>>> = const { RefCell::new(None) }; }

/// A scope selects independent native scratch without cloning model layers or
/// committed state. Bindings keep both backend identities alive.
pub(crate) struct ExecutionLane {
    bindings: Arc<Bindings>,
}
impl ExecutionLane {
    pub fn new(backends: &[Backend]) -> Result<Self> {
        let bindings = backends
            .iter()
            .map(|source| {
                Ok((
                    Arc::as_ptr(&source.0) as usize,
                    Binding {
                        source: source.clone(),
                        execution: source.fork_execution()?,
                    },
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            bindings: Arc::new(bindings),
        })
    }
    pub fn backends(&self) -> impl Iterator<Item = &Backend> {
        self.bindings.values().map(|binding| &binding.execution)
    }
    pub fn backend(&self, source: &Backend) -> Backend {
        self.bindings[&(Arc::as_ptr(&source.0) as usize)]
            .execution
            .clone()
    }
    pub fn prepare(&self, rows: usize) -> Result<()> {
        for binding in self.bindings.values() {
            binding.execution.copy_strategies_from(&binding.source);
            binding.execution.retain_workspaces(&[rows])?;
            binding.execution.prepare(rows)?;
        }
        Ok(())
    }
    pub fn run<R>(&self, operation: impl FnOnce() -> R) -> R {
        struct Restore(Option<Arc<Bindings>>);
        impl Drop for Restore {
            fn drop(&mut self) {
                CURRENT.with(|current| current.replace(self.0.take()));
            }
        }
        let _restore =
            Restore(CURRENT.with(|current| current.replace(Some(self.bindings.clone()))));
        operation()
    }
}

impl Backend {
    pub(super) fn execution(&self) -> Self {
        CURRENT
            .with(|current| {
                current
                    .borrow()
                    .as_ref()
                    .and_then(|bindings| bindings.get(&(Arc::as_ptr(&self.0) as usize)))
                    .map(|binding| binding.execution.clone())
            })
            .unwrap_or_else(|| self.clone())
    }
    /// An execution lane owns scratch and tuning state, while resident device
    /// weights, context and domain remain shared with the original lane.
    fn fork_execution(&self) -> Result<Self> {
        let source = self.0.lock().unwrap();
        let attention = source
            .attention
            .as_ref()
            .map(|a| AttentionWork::new(source.domain.raw, a.config, source.options.chunk))
            .transpose()?;
        let material = source
            .material
            .iter()
            .map(|(&key, m)| {
                (
                    key,
                    Material {
                        matrix: m.matrix.clone(),
                        reload: None,
                        profile: m.profile.clone(),
                    },
                )
            })
            .collect();
        let mut options = source.options.clone();
        options.tune_search = false;
        Ok(Self(Arc::new(Mutex::new(BackendInner {
            work: BTreeMap::new(),
            attention,
            material,
            w8_partitions: source.w8_partitions.clone(),
            w8_strategies: source.w8_strategies.clone(),
            w4_strategies: source.w4_strategies.clone(),
            options,
            domain_id: source.domain_id,
            cache_hits: 0,
            cache_misses: 0,
            tune_calls: 0,
            fallbacks: source.fallbacks.clone(),
            fallback_calls: 0,
            weight_reload_calls: 0,
            loaded_w4: 0,
            loaded_w8: 0,
            compression_scratch: source.compression_scratch.clone(),
            weight_bytes: 0,
            domain: source.domain.clone(),
            context: source.context.clone(),
        }))))
    }
    fn copy_strategies_from(&self, source: &Self) {
        let source = source.0.lock().unwrap();
        let mut lane = self.0.lock().unwrap();
        lane.w4_strategies = source.w4_strategies.clone();
        lane.w8_strategies = source.w8_strategies.clone();
        lane.w8_partitions = source.w8_partitions.clone();
        lane.fallbacks = source.fallbacks.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    #[ignore = "requires RK3588"]
    fn independent_execution_lanes_share_weights() -> Result<()> {
        let _lock = crate::validation::BoardLock::acquire()?;
        let cli = crate::cli::Cli::try_parse_from(["infer", "-m", "unused", "run"])?;
        let mut options = cli.model.options()?;
        options.tune = false;
        options.tune_cache.clear();
        configure_cpu(&mut options)?;
        for mask in [1, 3, 7] {
            options.npu_mask = mask;
            for (bits, compress) in [(4, false), (4, true), (8, false), (8, true)] {
                options.compress_w8a8 = compress;
                let compression = bits == 8 && compress;
                let primary = Backend::new(&options, 0)?;
                let (rows, k, n) = (128, 513, 512);
                let linear = primary.load(
                    QuantMatrix {
                        bits,
                        k,
                        n,
                        codes: (0..k * n).map(|i| ((i * 7 % 4) as i8) - 2).collect(),
                        scales: vec![0.01; n],
                    },
                    None,
                )?;
                assert_eq!(linear.compression_enabled(), compression);
                assert_eq!(release_compression_scratch([&primary]), compression);
                primary.prepare(rows)?;
                let lane = ExecutionLane::new(std::slice::from_ref(&primary))?;
                lane.prepare(rows)?;
                let other = lane.backend(&primary);
                assert_eq!(other.weight_bytes(), 0);
                let input0: Vec<f32> = (0..rows * k)
                    .map(|i| ((i % 127) as f32 - 63.) * 0.03125)
                    .collect();
                let input1: Vec<f32> = input0.iter().map(|v| v * (-0.75) + 0.01).collect();
                let mut expected0 = vec![0.; rows * n];
                let mut expected1 = expected0.clone();
                linear.run(&input0, &mut expected0, rows)?;
                lane.run(|| linear.run(&input1, &mut expected1, rows))?;
                let seed_matches = primary
                    .0
                    .lock()
                    .unwrap()
                    .w8_strategies
                    .get(&(linear.key, rows))
                    .is_some_and(|strategy| strategy.n_tile as usize == n.div_ceil(32) * 32);
                assert_eq!(
                    release_compression_scratch([&primary]),
                    compression && !seed_matches
                );
                if compression {
                    let mut output = vec![0.; 3 * n];
                    lane.run(|| linear.run(&input0, &mut output, 3))?;
                    assert_eq!(output, expected0[..3 * n]);
                    assert!(!release_compression_scratch([&primary]));
                    {
                        let mut b = other.0.lock().unwrap();
                        let mut strategy = b.w8_strategy(linear.key, 4)?;
                        strategy.n_tile = 64;
                        strategy.n_groups = n.div_ceil(64) as i32;
                        b.work.remove(&(linear.key, 4));
                        b.w8_strategies.insert((linear.key, 4), strategy);
                    }
                    lane.run(|| linear.run(&input0, &mut output, 3))?;
                    assert_eq!(output, expected0[..3 * n]);
                    assert!(release_compression_scratch([&primary]));
                    assert!(!release_compression_scratch([&primary]));
                }
                for _ in 0..3 {
                    let mut output0 = vec![0.; rows * n];
                    let mut output1 = output0.clone();
                    std::thread::scope(|scope| -> Result<()> {
                        let follower =
                            scope.spawn(|| lane.run(|| linear.run(&input1, &mut output1, rows)));
                        linear.run(&input0, &mut output0, rows)?;
                        follower.join().expect("second lane panicked")?;
                        Ok(())
                    })?;
                    assert!(!release_compression_scratch([&primary]));
                    assert!(
                        expected0
                            .iter()
                            .zip(&output0)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                    assert!(
                        expected1
                            .iter()
                            .zip(&output1)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                }
                eprintln!(
                    "PASS shared W{bits} weights, independent workspaces, mask={mask}, concurrent calls bit-exact"
                );
            }
        }
        Ok(())
    }
}
