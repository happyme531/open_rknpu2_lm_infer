use super::support::{Inputs, close, fixtures, series};
use open_rknpu2_lm_infer::{
    layers::gated_delta::{LayerState, MixerWeights, Projected, Scratch},
    ops::gated_delta::GdnDimensions,
};

fn weights(dim: GdnDimensions, kernel: usize, offset: usize) -> MixerWeights {
    MixerWeights::new(
        dim,
        kernel,
        series(dim.conv_width() * kernel, offset, 0.2),
        series(dim.value_heads(), offset + 7, 0.4),
        series(dim.value_heads(), offset + 19, 0.2),
        vec![1.0; dim.value_dim()],
        1e-6,
    )
    .unwrap()
}

#[test]
fn complete_cpu_mixer_matches_pinned_transformers_fixture() {
    let f = fixtures().mixer;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut state = LayerState::new(f.dim(), f.kernel, 8).unwrap();
    let mut scratch = Scratch::default();
    let actual = state
        .forward(&pool, &f.weights(), f.input(), &mut scratch)
        .unwrap();
    close(actual, &f.output);
    close(state.canonical(), &f.state);
    assert_eq!(state.convolution(), f.history);
    assert_eq!(state.position(), f.rows);
}

#[test]
fn checkpoints_restore_state_history_and_next_output_after_a_branch() {
    let dim = GdnDimensions::new(2, 4, 128, 8).unwrap();
    let weights = weights(dim, 4, 3);
    let input = Inputs::new(dim, 12, 7);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut state = LayerState::new(dim, 4, 8).unwrap();
    let mut scratch = Scratch::default();
    state
        .forward(&pool, &weights, input.view(0, 5), &mut scratch)
        .unwrap();
    let saved = state.snapshot().unwrap();
    let canonical = state.canonical().to_vec();
    let history = state.convolution().to_vec();
    let expected = state
        .forward(&pool, &weights, input.view(5, 7), &mut scratch)
        .unwrap()
        .to_vec();
    state.restore(&saved).unwrap();
    assert_eq!(state.position(), 5);
    assert_eq!(state.canonical(), canonical);
    assert_eq!(state.convolution(), history);
    assert_eq!(
        state
            .forward(&pool, &weights, input.view(5, 7), &mut scratch)
            .unwrap(),
        expected
    );
    state
        .verify(&pool, &weights, input.view(0, 2), &mut scratch)
        .unwrap();
    assert!(state.snapshot().is_err());
    assert!(state.restore(&saved).is_err());
    state.abort_round();
    state.reset();
    state.restore(&saved).unwrap();
    assert_eq!(state.canonical(), canonical);
    let mut other = LayerState::new(dim, 3, 8).unwrap();
    assert!(other.restore(&saved).is_err());
}

#[test]
fn prefill_chunks_and_single_token_execution_have_identical_states_and_outputs() {
    for (kd, vd) in [(8, 5), (128, 8)] {
        check_chunks(GdnDimensions::new(2, 4, kd, vd).unwrap());
    }
}

fn check_chunks(dim: GdnDimensions) {
    let weights = weights(dim, 4, 3);
    let input = Inputs::new(dim, 129, 7);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for chunk in [1, 2, 3, 4, 5, 7, 8, 31, 32, 33, 127, 128, 129] {
        let mut all = LayerState::new(dim, 4, 8).unwrap();
        let mut pieces = LayerState::new(dim, 4, 8).unwrap();
        let mut scratch = Scratch::default();
        let expected = all
            .forward(&pool, &weights, input.view(0, 129), &mut scratch)
            .unwrap()
            .to_vec();
        let mut output = Vec::new();
        for start in (0..129).step_by(chunk) {
            output.extend_from_slice(
                pieces
                    .forward(
                        &pool,
                        &weights,
                        input.view(start, chunk.min(129 - start)),
                        &mut scratch,
                    )
                    .unwrap(),
            );
        }
        assert_eq!(output, expected);
        assert_eq!(pieces.canonical(), all.canonical());
        assert_eq!(pieces.convolution(), all.convolution());
    }
}

#[test]
fn every_acceptance_boundary_replays_only_confirmed_prefix_and_next_call_matches_recompute() {
    for (kd, vd) in [(8, 5), (128, 8)] {
        check_acceptance(GdnDimensions::new(2, 4, kd, vd).unwrap());
    }
}

fn check_acceptance(dim: GdnDimensions) {
    let weights = weights(dim, 4, 13);
    let input = Inputs::new(dim, 64, 31);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for k in [1, 2, 3, 4, 7] {
        for accepted in 0..=k {
            let mut state = LayerState::new(dim, 4, 8).unwrap();
            let mut fresh = LayerState::new(dim, 4, 8).unwrap();
            let mut scratch = Scratch::default();
            state
                .forward(&pool, &weights, input.view(0, 11), &mut scratch)
                .unwrap();
            let before = state.canonical().to_vec();
            let history = state.convolution().to_vec();
            state
                .verify(&pool, &weights, input.view(11, k + 1), &mut scratch)
                .unwrap();
            assert_eq!(state.position(), 11);
            assert_eq!(state.canonical(), before);
            assert_eq!(state.convolution(), history);
            assert!(state.finish_round(&pool, 0).is_err());
            state.finish_round(&pool, 1 + accepted).unwrap();
            assert_eq!(state.position(), 12 + accepted);
            fresh
                .forward(&pool, &weights, input.view(0, 12 + accepted), &mut scratch)
                .unwrap();
            assert_eq!(state.canonical(), fresh.canonical());
            assert_eq!(state.convolution(), fresh.convolution());
            let actual = state
                .forward(&pool, &weights, input.view(12 + accepted, 3), &mut scratch)
                .unwrap()
                .to_vec();
            let expected = fresh
                .forward(&pool, &weights, input.view(12 + accepted, 3), &mut scratch)
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(state.canonical(), fresh.canonical());
        }
    }
}

#[test]
fn failed_forward_preserves_state_history_and_position_after_preprocessing() {
    let dim = GdnDimensions::new(2, 4, 128, 8).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let good_weights = weights(dim, 4, 3);
    let good_input = Inputs::new(dim, 2, 7);
    let mut state = LayerState::new(dim, 4, 8).unwrap();
    let mut scratch = Scratch::default();
    state
        .forward(&pool, &good_weights, good_input.view(0, 1), &mut scratch)
        .unwrap();
    let before = state.canonical().to_vec();
    let history = state.convolution().to_vec();
    let bad_weights = MixerWeights::new(
        dim,
        4,
        vec![0.1; dim.conv_width() * 4],
        vec![-1000.0; dim.value_heads()],
        vec![f32::MAX; dim.value_heads()],
        vec![1.0; dim.value_dim()],
        1e-6,
    )
    .unwrap();
    let mut bad_input = Inputs::new(dim, 1, 7);
    bad_input.a.fill(f32::MAX);
    // Finite inputs pass validation, but 0 * softplus(MAX + MAX) is NaN.
    // The recurrence rejects it after convolution, before canonical mutation.
    assert!(
        state
            .forward(&pool, &bad_weights, bad_input.view(0, 1), &mut scratch)
            .is_err()
    );
    assert_eq!(state.canonical(), before);
    assert_eq!(state.convolution(), history);
    assert_eq!(state.position(), 1);
    state
        .forward(&pool, &good_weights, good_input.view(1, 1), &mut scratch)
        .unwrap();
    let mut fresh = LayerState::new(dim, 4, 8).unwrap();
    fresh
        .forward(&pool, &good_weights, good_input.view(0, 2), &mut scratch)
        .unwrap();
    assert_eq!(state.canonical(), fresh.canonical());
    assert_eq!(state.convolution(), fresh.convolution());
}

#[test]
fn scratch_reuse_cannot_overwrite_other_layer_log_and_abort_keeps_canonical_history() {
    let dim = GdnDimensions::new(2, 4, 7, 5).unwrap();
    let w1 = weights(dim, 4, 13);
    let w2 = weights(dim, 4, 23);
    let input = Inputs::new(dim, 31, 71);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut a = LayerState::new(dim, 4, 8).unwrap();
    let mut b = LayerState::new(dim, 4, 8).unwrap();
    let mut scratch = Scratch::default();
    for round in 0..4 {
        let start = a.position();
        let before = a.canonical().to_vec();
        a.verify(&pool, &w1, input.view(start, 8), &mut scratch)
            .unwrap();
        b.verify(&pool, &w2, input.view(start, 8), &mut scratch)
            .unwrap();
        if round % 2 == 0 {
            a.abort_round();
            b.abort_round();
            assert_eq!(a.canonical(), before);
            assert_eq!(a.position(), start);
            a.verify(&pool, &w1, input.view(start, 8), &mut scratch)
                .unwrap();
            b.verify(&pool, &w2, input.view(start, 8), &mut scratch)
                .unwrap();
        }
        let count = round + 1;
        a.finish_round(&pool, count).unwrap();
        b.finish_round(&pool, count).unwrap();
        for (state, weight) in [(&a, &w1), (&b, &w2)] {
            let mut fresh = LayerState::new(dim, 4, 8).unwrap();
            fresh
                .forward(&pool, weight, input.view(0, state.position()), &mut scratch)
                .unwrap();
            assert_eq!(state.canonical(), fresh.canonical());
            assert_eq!(state.convolution(), fresh.convolution());
        }
    }
    assert_eq!(
        a.memory_usage().update_log_bytes,
        8 * (dim.key_width() + dim.value_width() + 2 * dim.value_heads() + dim.conv_width()) * 4
    );
    let before = a.canonical().to_vec();
    let bad = Projected::new(1, &[], &[], &[], &[]);
    assert!(a.verify(&pool, &w1, bad, &mut scratch).is_err());
    assert_eq!(a.canonical(), before);
    assert_eq!(a.pending_rows(), None);
    assert!(
        a.verify(&pool, &w1, input.view(0, 9), &mut scratch)
            .is_err()
    );
    a.reset();
    let mut fresh = LayerState::new(dim, 4, 8).unwrap();
    let actual = a
        .forward(&pool, &w1, input.view(0, 7), &mut scratch)
        .unwrap()
        .to_vec();
    let expected = fresh
        .forward(&pool, &w1, input.view(0, 7), &mut scratch)
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn merged_projection_and_caller_output_match_contiguous_state_and_replay() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for (kh, vh, kd, vd, kernel) in [(2, 4, 128, 8, 4), (1, 2, 7, 5, 3)] {
        let dim = GdnDimensions::new(kh, vh, kd, vd).unwrap();
        let weight = weights(dim, kernel, 11);
        let mut a = LayerState::new(dim, kernel, 8).unwrap();
        let mut b = LayerState::new(dim, kernel, 8).unwrap();
        let mut sa = Scratch::default();
        let mut sb = Scratch::default();
        for rows in [1, 2, 3, 7, 13, 32, 128] {
            let input = Inputs::new(dim, rows, rows + 19);
            let mut merged = Vec::new();
            for t in 0..rows {
                merged.extend_from_slice(
                    &input.qkv[t * dim.conv_width()..(t + 1) * dim.conv_width()],
                );
                merged.extend_from_slice(
                    &input.z[t * dim.value_width()..(t + 1) * dim.value_width()],
                );
                merged.extend_from_slice(&input.a[t * vh..(t + 1) * vh]);
                merged.extend_from_slice(&input.b[t * vh..(t + 1) * vh]);
            }
            let view = Projected::from_merged(dim, rows, &merged).unwrap();
            let expected = a
                .forward(&pool, &weight, input.view(0, rows), &mut sa)
                .unwrap()
                .to_vec();
            let mut actual = vec![0.; expected.len()];
            b.forward_into(&pool, &weight, view, &mut sb, &mut actual)
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(a.canonical(), b.canonical());
            assert_eq!(a.convolution(), b.convolution());
            assert_eq!(a.position(), b.position());
            if rows <= 7 {
                let expected = a
                    .verify(&pool, &weight, input.view(0, rows), &mut sa)
                    .unwrap()
                    .to_vec();
                let actual = b.verify(&pool, &weight, view, &mut sb).unwrap();
                assert_eq!(actual, expected);
                a.finish_round(&pool, rows).unwrap();
                b.finish_round(&pool, rows).unwrap();
                assert_eq!(a.canonical(), b.canonical());
                assert_eq!(a.convolution(), b.convolution());
            }
        }
    }
}
