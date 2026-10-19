//! `fused_q4k_multirow_matmul_f32_into` is the batched form of the single-row Q4_K matvec:
//! same numbers row for row, and it refuses mis-sized buffers (#5646).

use realizar::quantize::{fused_q4k_multirow_matmul_f32_into, fused_q4k_parallel_matvec_into};

const IN_DIM: usize = 256;
const OUT_DIM: usize = 4;
const SUPER_BLOCK_BYTES: usize = 144;

/// OUT_DIM rows of one Q4_K super-block each: finite f16 d/dmin, deterministic payload.
fn weights() -> Vec<u8> {
    let mut seed = 0x9E37_79B9_u32;
    let mut w = Vec::with_capacity(OUT_DIM * SUPER_BLOCK_BYTES);
    for _ in 0..OUT_DIM {
        w.extend_from_slice(&0x3C00_u16.to_le_bytes()); // d = 1.0
        w.extend_from_slice(&0x3800_u16.to_le_bytes()); // dmin = 0.5
        for _ in 4..SUPER_BLOCK_BYTES {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            w.push((seed >> 24) as u8);
        }
    }
    w
}

fn input(m: usize) -> Vec<f32> {
    (0..m * IN_DIM)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) / 8.0)
        .collect()
}

fn per_row(w: &[u8], x: &[f32], m: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * OUT_DIM];
    for t in 0..m {
        fused_q4k_parallel_matvec_into(
            w,
            &x[t * IN_DIM..(t + 1) * IN_DIM],
            IN_DIM,
            OUT_DIM,
            &mut out[t * OUT_DIM..(t + 1) * OUT_DIM],
        )
        .expect("matvec");
    }
    out
}

#[test]
fn batched_rows_match_the_single_row_matvec() {
    let (w, m) = (weights(), 3);
    let x = input(m);
    let mut got = vec![f32::NAN; m * OUT_DIM];
    fused_q4k_multirow_matmul_f32_into(&w, &x, m, IN_DIM, OUT_DIM, &mut got).expect("multirow");
    let want = per_row(&w, &x, m);
    assert!(
        want.iter().any(|v| *v != 0.0),
        "fixture must be non-trivial"
    );
    for (g, e) in got.iter().zip(&want) {
        assert!((g - e).abs() <= 1e-3 * e.abs().max(1.0), "{g} vs {e}");
    }
}

#[test]
fn single_row_takes_the_matvec_path() {
    let w = weights();
    let x = input(1);
    let mut got = vec![0.0; OUT_DIM];
    fused_q4k_multirow_matmul_f32_into(&w, &x, 1, IN_DIM, OUT_DIM, &mut got).expect("m=1");
    assert_eq!(got, per_row(&w, &x, 1));
}

#[test]
fn wrong_input_length_is_an_invalid_shape() {
    let w = weights();
    let mut out = vec![0.0; 2 * OUT_DIM];
    let err = fused_q4k_multirow_matmul_f32_into(&w, &input(1), 2, IN_DIM, OUT_DIM, &mut out)
        .expect_err("input too short");
    assert!(err.to_string().contains("multirow f32"), "{err}");
}

#[test]
fn short_output_is_rejected_and_left_untouched() {
    let w = weights();
    let mut out = vec![7.0; 2 * OUT_DIM - 1];
    assert!(
        fused_q4k_multirow_matmul_f32_into(&w, &input(2), 2, IN_DIM, OUT_DIM, &mut out).is_err()
    );
    assert!(out.iter().all(|v| *v == 7.0));
}

#[test]
fn zero_rows_is_a_no_op() {
    let mut out: Vec<f32> = Vec::new();
    fused_q4k_multirow_matmul_f32_into(&weights(), &[], 0, IN_DIM, OUT_DIM, &mut out).expect("m=0");
}

#[test]
fn surplus_input_is_rejected_not_silently_truncated() {
    let w = weights();
    let mut out = vec![0.0; OUT_DIM];
    assert!(
        fused_q4k_multirow_matmul_f32_into(&w, &input(2), 1, IN_DIM, OUT_DIM, &mut out).is_err()
    );
}
