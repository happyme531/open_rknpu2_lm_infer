use super::support::{close, fixtures, series};
use open_rknpu2_lm_infer::{
    ops::gated_delta::{self as gdn, GdnDimensions, Updates, Workspace},
    ops::norm,
};

#[test]
fn update_ranges_reject_nan_and_positive_infinity_but_allow_zero_decay() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let dim = GdnDimensions::new(1, 1, 2, 2).unwrap();
    let mut workspace = Workspace::default();
    for (decay, beta, valid) in [
        (f32::NEG_INFINITY, 0., true),
        (f32::NEG_INFINITY, 1., true),
        (0., 0.5, true),
        (f32::NAN, 0.5, false),
        (f32::INFINITY, 0.5, false),
        (0.1, 0.5, false),
        (-1., f32::NAN, false),
        (-1., f32::INFINITY, false),
        (-1., f32::NEG_INFINITY, false),
        (-1., -0.1, false),
        (-1., 1.1, false),
    ] {
        let mut state = vec![0.25; dim.state_elements()];
        let before = state.clone();
        let mut output = vec![0.; dim.value_width()];
        let updates = Updates {
            rows: 1,
            key: &[0.5, 0.5],
            value: &[1., 2.],
            log_decay: &[decay],
            beta: &[beta],
        };
        let result = workspace.run(&pool, dim, &mut state, &[0.5, 0.5], updates, &mut output);
        assert_eq!(result.is_ok(), valid, "decay={decay}, beta={beta}");
        if valid {
            assert!(output.iter().chain(&state).all(|v| v.is_finite()));
        } else {
            assert_eq!(state, before, "invalid updates must not mutate state");
        }
    }
}

#[test]
fn scalar_and_parallel_match_pinned_transformers_nonzero_rectangular_state() {
    let f = fixtures().gdn;
    let dim = GdnDimensions::new(f.key_heads, f.value_heads, f.key_dim, f.value_dim).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let updates = Updates {
        rows: f.rows,
        key: &f.key,
        value: &f.value,
        log_decay: &f.log_decay,
        beta: &f.beta,
    };
    let mut scalar = f.initial_state.clone();
    let mut state = scalar.clone();
    let mut output = vec![0.0; f.output.len()];
    gdn::run_scalar(dim, &mut scalar, &f.query, updates, &mut output).unwrap();
    close(&output, &f.output);
    close(&scalar, &f.state);
    Workspace::default()
        .run(&pool, dim, &mut state, &f.query, updates, &mut output)
        .unwrap();
    close(&output, &f.output);
    close(&state, &f.state);
    let mut replayed = f.initial_state;
    gdn::replay(&pool, dim, &mut replayed, updates).unwrap();
    assert_eq!(state, replayed);
}

#[test]
fn full_head_neon_and_scalar_tail_match_across_window_lengths_and_decay() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut workspace = Workspace::default();
    for (kh, vh, kd, vd) in [(2, 4, 128, 128), (3, 3, 7, 5), (1, 2, 4, 3)] {
        let dim = GdnDimensions::new(kh, vh, kd, vd).unwrap();
        for rows in [1, 2, 3, 4, 5, 6, 7, 8, 9, 31, 32, 33, 127, 128, 129] {
            let mut q = series(rows * dim.key_width(), 1, 1.0);
            let mut k = series(q.len(), 9, 1.0);
            norm::l2_normalize(&mut q, kd).unwrap();
            norm::l2_normalize(&mut k, kd).unwrap();
            let value = series(rows * dim.value_width(), 19, 0.3);
            let log_decay: Vec<f32> = (0..rows * vh)
                .map(|i| match i % 5 {
                    0 => 0.0,
                    1 => -90.0,
                    _ => -0.001,
                })
                .collect();
            let beta: Vec<f32> = (0..rows * vh)
                .map(|i| match i % 3 {
                    0 => 0.0,
                    1 => 1.0,
                    _ => 0.31,
                })
                .collect();
            let updates = Updates {
                rows,
                key: &k,
                value: &value,
                log_decay: &log_decay,
                beta: &beta,
            };
            let mut scalar = series(dim.state_elements(), 29, 0.01);
            let mut parallel = scalar.clone();
            let mut a = vec![0.0; value.len()];
            let mut b = a.clone();
            gdn::run_scalar(dim, &mut scalar, &q, updates, &mut a).unwrap();
            workspace
                .run(&pool, dim, &mut parallel, &q, updates, &mut b)
                .unwrap();
            close(&b, &a);
            close(&parallel, &scalar);
        }
    }
}

#[test]
fn short_windows_match_reference_for_copied_in_place_and_replayed_state() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let dim = GdnDimensions::new(2, 4, 128, 8).unwrap();
    for rows in 1..=9 {
        let mut q = series(rows * dim.key_width(), 1, 1.0);
        let mut k = series(q.len(), 9, 1.0);
        norm::l2_normalize(&mut q, 128).unwrap();
        norm::l2_normalize(&mut k, 128).unwrap();
        let value = series(rows * dim.value_width(), 19, 0.3);
        let log_decay: Vec<_> = (0..rows * dim.value_heads())
            .map(|i| [0.0, -1e-6, -0.07, -90.0, f32::NEG_INFINITY][i % 5])
            .collect();
        let beta: Vec<_> = (0..log_decay.len())
            .map(|i| [0.0, 1.0, 0.37][i % 3])
            .collect();
        let updates = Updates {
            rows,
            key: &k,
            value: &value,
            log_decay: &log_decay,
            beta: &beta,
        };
        let initial = series(dim.state_elements(), 29, 0.01);
        let source = initial.clone();
        let mut expected_state = initial.clone();
        let mut expected = vec![0.0; value.len()];
        Workspace::ggml_loop_reference()
            .run(&pool, dim, &mut expected_state, &q, updates, &mut expected)
            .unwrap();
        let mut state = vec![123.0; initial.len()];
        let mut output = expected.clone();
        Workspace::default()
            .run_from(&pool, dim, &initial, &mut state, &q, updates, &mut output)
            .unwrap();
        assert_eq!(initial, source);
        assert_eq!(state, expected_state);
        assert_eq!(output, expected);
        state.clone_from(&initial);
        Workspace::default()
            .run(&pool, dim, &mut state, &q, updates, &mut output)
            .unwrap();
        assert_eq!(state, expected_state);
        assert_eq!(output, expected);
        state.clone_from(&initial);
        gdn::replay(&pool, dim, &mut state, updates).unwrap();
        assert_eq!(state, expected_state);
    }
}

#[test]
fn validation_precedes_state_mutation() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let dim = GdnDimensions::new(1, 1, 2, 3).unwrap();
    let mut state = vec![0.5; dim.state_elements()];
    let original = state.clone();
    let updates = Updates {
        rows: 1,
        key: &[0.0, 1.0],
        value: &[0.0; 3],
        log_decay: &[0.1],
        beta: &[0.5],
    };
    assert!(gdn::replay(&pool, dim, &mut state, updates).is_err());
    assert_eq!(state, original);
    assert!(GdnDimensions::new(3, 4, 128, 128).is_err());
}

#[test]
fn cpp_copied_and_fused_loops_copy_initial_heads_and_preserve_the_source() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for (kh, vh, kd, vd) in [
        (16, 32, 128, 128),
        (2, 4, 128, 13),
        (1, 2, 128, 3),
        (2, 4, 17, 13),
        (1, 1, 7, 5),
    ] {
        let dim = GdnDimensions::new(kh, vh, kd, vd).unwrap();
        let rows = 7;
        let mut q = series(rows * dim.key_width(), 1, 1.0);
        let mut k = series(q.len(), 9, 1.0);
        norm::l2_normalize(&mut q, kd).unwrap();
        norm::l2_normalize(&mut k, kd).unwrap();
        let value = series(rows * dim.value_width(), 19, 0.3);
        let log_decay: Vec<_> = (0..rows * vh)
            .map(|i| if i % 3 == 0 { -90.0 } else { -0.001 })
            .collect();
        let beta: Vec<_> = (0..rows * vh).map(|i| [0.0, 1.0, 0.37][i % 3]).collect();
        let updates = Updates {
            rows,
            key: &k,
            value: &value,
            log_decay: &log_decay,
            beta: &beta,
        };
        let initial = series(dim.state_elements(), 29, 0.01);
        let source = initial.clone();
        let mut expected_state = initial.clone();
        let mut expected = vec![0.0; value.len()];
        gdn::run_scalar(dim, &mut expected_state, &q, updates, &mut expected).unwrap();
        for mut workspace in [Workspace::default(), Workspace::ggml_loop_reference()] {
            let mut state = vec![123.0; initial.len()];
            let mut output = vec![0.0; expected.len()];
            workspace
                .run_from(&pool, dim, &initial, &mut state, &q, updates, &mut output)
                .unwrap();
            close(&state, &expected_state);
            close(&output, &expected);
            assert_eq!(initial, source);
        }
    }
}
