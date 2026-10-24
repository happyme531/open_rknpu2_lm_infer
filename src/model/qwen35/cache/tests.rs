use super::*;
mod board;

fn checkpoint(position: usize, task: usize) -> Checkpoint {
    Checkpoint {
        position,
        next_rope: position,
        recurrent: Vec::new(),
        mtp: None,
        task,
        bytes: 0,
    }
}

#[test]
fn restored_snapshot_reuse_preserves_earlier_branch_and_save_counters() {
    let mut cache = PrefixCache::new(CheckpointOptions {
        capacity: 2,
        min_step: 0,
    });
    cache.entries = vec![checkpoint(468, 1), checkpoint(596, 1)];
    cache.task = 2;
    cache.saves = 2;
    cache.save_ms = 17.;
    assert!(cache.reuse(596));
    assert_eq!(cache.positions(), [468, 596]);
    assert_eq!(cache.entries[1].task, 2);
    assert_eq!(cache.saves, 2);
    assert_eq!(cache.save_ms, 17.);
    assert_eq!(
        cache
            .entries
            .iter()
            .rev()
            .find(|c| c.position <= 500)
            .unwrap()
            .position,
        468
    );
}

#[test]
fn reused_snapshot_survives_old_task_spacing_compaction() {
    let mut cache = PrefixCache::new(CheckpointOptions {
        capacity: 3,
        min_step: 8192,
    });
    cache.entries = vec![checkpoint(20, 1), checkpoint(80, 1), checkpoint(100, 1)];
    cache.task = 2;
    assert!(cache.reuse(80));
    assert!(!cache.reuse(200));
    cache.make_room();
    cache.entries.push(checkpoint(200, 2));
    assert_eq!(cache.positions(), [20, 80, 200]);
}

#[test]
fn new_snapshot_still_obeys_upstream_fifo_when_full() {
    let mut cache = PrefixCache::new(CheckpointOptions {
        capacity: 2,
        min_step: 0,
    });
    cache.entries = vec![checkpoint(468, 1), checkpoint(596, 1)];
    cache.task = 2;
    assert!(!cache.reuse(728));
    cache.make_room();
    cache.entries.push(checkpoint(728, 2));
    assert_eq!(cache.positions(), [596, 728]);
}
