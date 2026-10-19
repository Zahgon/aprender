//! Behavioural tests for `Avx512Backend::exp` (#5635): vector body, scalar tail, measured 5% bound.

use super::*;

fn run_exp(input: &[f32]) -> Vec<f32> {
    let mut out = vec![f32::NAN; input.len()];
    // SAFETY: test-only; avx512f verified by `avx512_test`, slices are equal length
    unsafe {
        Avx512Backend::exp(input, &mut out);
    }
    out
}

fn rel_err(got: f32, want: f32) -> f32 {
    ((got - want) / want).abs()
}

#[test]
fn exp_empty_input_is_noop() {
    avx512_test(|| assert!(run_exp(&[]).is_empty()));
}

#[test]
fn exp_tail_only_matches_std_exactly() {
    avx512_test(|| {
        let input = [0.0_f32, 1.0, -1.0, 2.5, -3.5];
        let out = run_exp(&input);
        for (g, x) in out.iter().zip(input) {
            assert_eq!(*g, x.exp(), "tail lane must use f32::exp for x={x}");
        }
    });
}

#[test]
fn exp_vector_body_within_measured_bound() {
    avx512_test(|| {
        let input: Vec<f32> = (0..16).map(|i| -8.0 + i as f32).collect();
        let out = run_exp(&input);
        for (g, x) in out.iter().zip(&input) {
            assert!(rel_err(*g, x.exp()) < 0.05, "x={x} got={g} want={}", x.exp());
        }
    });
}

#[test]
fn exp_zero_vector_lane_is_one() {
    avx512_test(|| {
        let out = run_exp(&[0.0; 16]);
        assert!(out.iter().all(|g| rel_err(*g, 1.0) < 1e-6), "{out:?}");
    });
}

#[test]
fn exp_body_plus_tail_covers_every_lane() {
    avx512_test(|| {
        let input: Vec<f32> = (0..21).map(|i| (i as f32 - 10.0) * 0.3).collect();
        let out = run_exp(&input);
        assert!(out.iter().all(|g| g.is_finite() && *g > 0.0));
        for (g, x) in out.iter().zip(&input) {
            assert!(rel_err(*g, x.exp()) < 0.05, "x={x}");
        }
    });
}

#[test]
fn exp_is_strictly_monotonic_over_body() {
    avx512_test(|| {
        let input: Vec<f32> = (0..32).map(|i| -2.0 + 0.1 * i as f32).collect();
        let out = run_exp(&input);
        assert!(out.windows(2).all(|w| w[0] < w[1]), "{out:?}");
    });
}
