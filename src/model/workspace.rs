//! Row shapes and phase transitions shared by target decoders.
use super::prefill::PlanId;
use crate::{
    layers::gated_delta::Scratch,
    npu::{Backend, ExecutionLane, FloatBackend, Tf32Backend},
};
use anyhow::{Context, Result, ensure};
use std::ops::Range;

#[derive(Default)]
pub(crate) struct Workspaces {
    phase: Phase,
    domain_limit: u64,
}

#[derive(Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Idle,
    Prepared {
        span: Range<usize>,
        id: Option<PlanId>,
    },
    Prefill {
        prepared: bool,
    },
    HeadFree {
        end: usize,
    },
    Decode,
}

#[allow(
    dead_code,
    reason = "Targets sharing prefill retention are feature gated."
)]
pub(crate) struct Targets<'a> {
    pub primary: &'a [Backend],
    pub secondary: Option<&'a ExecutionLane>,
    pub head: Option<&'a Backend>,
    pub recurrent: [Option<&'a mut Scratch>; 2],
    pub routers: Option<RouterTargets<'a>>,
    pub floating: &'a [FloatBackend],
    pub chunk: usize,
    pub decode_rows: &'a [usize],
}
pub(crate) struct RouterTargets<'a> {
    pub backends: &'a [Tf32Backend],
    pub hidden: usize,
    pub experts: usize,
}

#[allow(
    dead_code,
    reason = "Targets sharing prefill retention are feature gated."
)]
pub(crate) struct PrefillPlan<'a> {
    pub span: Range<usize>,
    pub id: Option<PlanId>,
    pub rows: &'a [usize],
    pub head_rows: &'a [usize],
    pub paired: bool,
}

#[cfg_attr(
    not(any(
        feature = "model-qwen3",
        feature = "model-llama",
        feature = "model-granite",
        feature = "model-qwen35",
        feature = "model-minicpmv47"
    )),
    allow(dead_code, reason = "Prefill retention is used by Dense and Qwen3.5")
)]
impl Workspaces {
    // RK3588 gives each domain a 2 GiB aperture. Leave room for alignment,
    // different tail shapes, Attention graph growth and other transient GEMs.
    const DEFAULT_DOMAIN_MIB: u64 = 1792;

    pub fn for_generation(enabled: bool) -> Result<Self> {
        let mib = if enabled {
            match std::env::var("RK_NPU_PREFILL_CACHE_DOMAIN_MIB") {
                Ok(mib) => mib
                    .parse::<u64>()
                    .context("invalid RK_NPU_PREFILL_CACHE_DOMAIN_MIB")?,
                Err(std::env::VarError::NotPresent) => Self::DEFAULT_DOMAIN_MIB,
                Err(error) => return Err(error.into()),
            }
        } else {
            0
        };
        ensure!(
            mib <= Self::DEFAULT_DOMAIN_MIB,
            "RK_NPU_PREFILL_CACHE_DOMAIN_MIB must be in 0..={}",
            Self::DEFAULT_DOMAIN_MIB
        );
        Ok(Self {
            domain_limit: mib << 20,
            ..Self::default()
        })
    }

    pub fn is_decoding(&self) -> bool {
        self.phase == Phase::Decode
    }

    pub fn reset(&mut self) {
        self.phase = Phase::Idle;
    }

    /// Consume a plan for just this input segment. One-token decode keeps its
    /// existing scratch, including any reclaimable prefill left by the prompt.
    pub fn begin_input(&mut self, start: usize, tokens: usize, backends: &[Backend]) {
        if self.is_decoding() && tokens == 1 {
            return;
        }
        let prepared = (tokens == 1 && self.phase == Phase::HeadFree { end: start })
            || self.phase
                == (Phase::Prepared {
                    span: start..start + tokens,
                    id: None,
                });
        if !prepared && self.phase != (Phase::Prefill { prepared: false }) {
            for backend in backends {
                backend.discard_idle_prefill();
            }
        }
        self.phase = Phase::Prefill { prepared };
    }

    pub fn finish_head_free(&mut self, end: usize) {
        if self.phase == (Phase::Prefill { prepared: true }) {
            self.phase = Phase::HeadFree { end };
        }
    }

    /// Only the latest successful explicit plan can execute on this model.
    /// A stale handle leaves the current plan and its active scratch intact.
    pub fn begin_prepared(&mut self, span: Range<usize>, id: Option<PlanId>) -> Result<()> {
        ensure!(
            id.is_some() && self.phase == (Phase::Prepared { span, id }),
            "prepared prefill has been replaced or invalidated"
        );
        self.phase = Phase::Prefill { prepared: true };
        Ok(())
    }

    /// Publish a plan only after every required allocation and model-specific
    /// preparation step has succeeded.
    pub fn prepare(
        &mut self,
        mut targets: Targets<'_>,
        plan: PrefillPlan<'_>,
        after: impl FnOnce(&Targets<'_>, &[usize]) -> Result<()>,
    ) -> Result<()> {
        self.reset();
        let rows = merge_rows(plan.rows, targets.decode_rows);
        let result = targets
            .prepare(&plan, &rows)
            .and_then(|()| after(&targets, &rows));
        if result.is_err() {
            targets.trim(None)?;
            return result;
        }
        self.phase = Phase::Prepared {
            span: plan.span,
            id: plan.id,
        };
        Ok(())
    }

    pub fn abort(&mut self, mut targets: Targets<'_>) -> Result<()> {
        self.reset();
        targets.trim(None)
    }

    pub fn finish_prefill(&mut self, mut targets: Targets<'_>) -> Result<()> {
        if self.is_decoding() {
            return Ok(());
        }
        let budget = (self.domain_limit != 0
            && matches!(
                self.phase,
                Phase::Prefill { prepared: true } | Phase::HeadFree { .. }
            ))
        .then_some(self.domain_limit);
        self.reset();
        targets.trim(budget)?;
        self.phase = Phase::Decode;
        Ok(())
    }
}

#[allow(
    dead_code,
    reason = "Targets sharing prefill retention are feature gated."
)]
impl Targets<'_> {
    fn prepare(&mut self, plan: &PrefillPlan<'_>, rows: &[usize]) -> Result<()> {
        let full_chunk = [self.chunk];
        let secondary_rows = if plan.paired { &full_chunk[..] } else { &[] };
        // Activate both lanes and recurrent work before allocating on any domain.
        for backend in self.primary {
            backend.select_prefill_workspaces(rows)?;
        }
        if let Some(lane) = self.secondary {
            for backend in lane.backends() {
                backend.select_prefill_workspaces(secondary_rows)?;
            }
        }
        for (scratch, rows) in self.recurrent.iter_mut().zip([plan.rows, secondary_rows]) {
            if let Some(scratch) = scratch {
                scratch.select_workspaces(rows)?;
            }
        }
        if let Some(routers) = &self.routers {
            for backend in routers.backends {
                backend.select_prefill_workspaces(rows);
            }
        }
        for backend in self.floating {
            backend.select_prefill_workspaces(rows);
        }
        for backend in self.primary {
            for &rows in rows {
                crate::progress::prefill_shape(rows > 1 && rows != self.chunk, || {
                    backend.prepare(rows)
                })?;
            }
        }
        if let Some(head) = self.head {
            head.retain_workspaces(plan.head_rows)?;
            for &rows in plan.head_rows {
                head.prepare(rows)?;
            }
        }
        if plan.paired
            && let Some(lane) = self.secondary
        {
            lane.prepare(self.chunk)?;
        }
        for (scratch, rows) in self.recurrent.iter_mut().zip([plan.rows, secondary_rows]) {
            if let Some(scratch) = scratch {
                scratch.prepare_selected(rows)?;
            }
        }
        if let Some(routers) = &self.routers {
            for backend in routers.backends {
                for &n in rows {
                    backend.prepare(routers.hidden, routers.experts, n)?;
                }
            }
        }
        Ok(())
    }

    fn trim(&mut self, budget: Option<u64>) -> Result<()> {
        if let Some(head) = self.head {
            head.retain_workspaces(self.decode_rows)?;
        }
        for scratch in self.recurrent.iter_mut().flatten() {
            scratch.finish_prefill(budget.is_some());
        }
        let prefill = budget.map(|_| self.chunk);
        if let Some(routers) = &self.routers {
            for backend in routers.backends {
                backend.finish_prefill(self.decode_rows, prefill);
            }
        }
        for backend in self.floating {
            backend.finish_prefill(self.decode_rows, prefill);
        }
        for (backend, keep) in self
            .primary
            .iter()
            .map(|backend| (backend, self.decode_rows))
            .chain(
                self.secondary
                    .into_iter()
                    .flat_map(ExecutionLane::backends)
                    .map(|backend| (backend, &[][..])),
            )
        {
            if budget.is_some() {
                backend.park_prefill_workspace(keep, self.chunk)?;
            } else {
                backend.retain_workspaces(keep)?;
            }
        }
        // GDN and both lanes share these domain budgets.
        for backend in self.primary {
            if budget.is_none_or(|limit| backend.domain_live_bytes() > limit) {
                backend.discard_idle_prefill();
            }
        }
        Ok(())
    }
}

pub(crate) fn merge_rows(rows: &[usize], decode_rows: &[usize]) -> Vec<usize> {
    let mut rows = [rows, decode_rows].concat();
    rows.sort_unstable();
    rows.dedup();
    rows
}

#[allow(
    dead_code,
    reason = "Only models retaining prefill report cache domains."
)]
pub(crate) fn cached_domains(backends: &[Backend]) -> Vec<u32> {
    let mut domains: Vec<_> = backends
        .iter()
        .filter_map(Backend::idle_prefill_domain)
        .collect();
    domains.sort_unstable();
    domains
}

/// Convert logical prompt lengths to the physical row shapes they execute.
pub(crate) fn physical_rows(lengths: impl IntoIterator<Item = usize>, ubatch: usize) -> Vec<usize> {
    let mut rows = Vec::new();
    for length in lengths {
        if length == 0 {
            continue;
        }
        rows.push(length.min(ubatch));
        let tail = length % ubatch;
        if tail > 0 {
            rows.push(tail);
        }
    }
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Explicit tuning covers scoring/verification heads at the requested batches.
/// Ordinary generation prepares only its decode and verification row counts.
pub(crate) fn head_rows(prefill: &[usize], decode: &[usize], tuning: bool) -> Vec<usize> {
    merge_rows(if tuning { prefill } else { &[] }, decode)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn targets() -> Targets<'static> {
        Targets {
            primary: &[],
            secondary: None,
            head: None,
            recurrent: [None, None],
            routers: None,
            floating: &[],
            chunk: 128,
            decode_rows: &[1],
        }
    }

    #[test]
    fn successful_plan_applies_to_one_matching_input() -> Result<()> {
        let mut workspaces = Workspaces::default();
        let plan = |span| PrefillPlan {
            span,
            id: None,
            rows: &[128],
            head_rows: &[1],
            paired: false,
        };
        workspaces.prepare(targets(), plan(0..512), |_, _| Ok(()))?;
        workspaces.begin_input(0, 512, &[]);
        assert!(workspaces.phase == Phase::Prefill { prepared: true });
        workspaces.finish_prefill(targets())?;
        workspaces.begin_input(512, 1, &[]);
        assert!(workspaces.is_decoding());
        workspaces.begin_input(513, 128, &[]);
        assert!(workspaces.phase == Phase::Prefill { prepared: false });
        workspaces.prepare(targets(), plan(513..641), |_, _| Ok(()))?;
        workspaces.begin_input(513, 127, &[]);
        assert!(workspaces.phase == Phase::Prefill { prepared: false });
        assert!(
            workspaces
                .prepare(targets(), plan(640..768), |_, _| anyhow::bail!(
                    "prepare failed"
                ))
                .is_err()
        );
        workspaces.begin_input(640, 128, &[]);
        assert!(workspaces.phase == Phase::Prefill { prepared: false });
        workspaces.abort(targets())?;
        assert!(workspaces.phase == Phase::Idle);
        Ok(())
    }

    #[test]
    fn explicit_prefill_is_replaced_even_when_span_matches() -> Result<()> {
        let mut workspaces = Workspaces::default();
        let span = 0..388;
        let old = Some(PlanId::new());
        let current = Some(PlanId::new());
        let plan = |id, paired| PrefillPlan {
            span: span.clone(),
            id,
            rows: &[128],
            head_rows: &[1],
            paired,
        };
        workspaces.prepare(targets(), plan(old, true), |_, _| Ok(()))?;
        workspaces.prepare(targets(), plan(current, false), |_, _| Ok(()))?;
        assert!(workspaces.begin_prepared(span.clone(), old).is_err());
        workspaces.begin_prepared(span.clone(), current)?;
        assert!(workspaces.begin_prepared(span.clone(), current).is_err());
        workspaces.finish_head_free(span.end);
        workspaces.begin_input(span.end, 1, &[]);
        assert!(workspaces.phase == Phase::Prefill { prepared: true });
        Ok(())
    }

    #[test]
    fn short_replay_and_logical_prompt_blocks_keep_only_executed_shapes() {
        assert_eq!(physical_rows([4], 128), [4]);
        assert_eq!(physical_rows([20, 1024, 330, 4], 128), [4, 20, 74, 128]);
        assert_eq!(physical_rows([128, 256], 128), [128]);
        assert!(physical_rows([0], 128).is_empty());
    }

    #[test]
    fn explicit_tuning_covers_requested_head_batches_without_inflating_generation() {
        let prefill = physical_rows([2, 4, 8, 128], 128);
        assert_eq!(head_rows(&prefill, &[1], true), [1, 2, 4, 8, 128]);
        assert_eq!(head_rows(&prefill, &[1], false), [1]);
        assert_eq!(head_rows(&[128], &[1, 2, 3], false), [1, 2, 3]);
    }
}
