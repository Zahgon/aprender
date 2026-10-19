//! Serial AVX-512 Q4_K GEMV (`matmul_q4k_f32_avx512`): `matmul_q4k_f32_dispatch` takes it
//! on AVX-512 hosts when `out_dim * in_dim < 8M`. It must match the scalar reference
//! row for row, accumulate across super-blocks, and treat the masked tail of a
//! non-multiple-of-256 `in_dim` as zero (#5634).
#![cfg(target_arch = "x86_64")]

use trueno::backends::q4k::{matmul_q4k_f32_dispatch, matmul_q4k_f32_scalar};

const SB_BYTES: usize = 144;
const SB_ELEMS: usize = 256;

/// Deterministic Q4_K rows: d=1.0, dmin=0.5 (f16), LCG scales/quants.
fn weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let sbs = in_dim.div_ceil(SB_ELEMS);
    let mut state = 0xC0FF_EE11u32;
    let mut data = Vec::with_capacity(out_dim * sbs * SB_BYTES);
    for _ in 0..out_dim * sbs {
        data.extend_from_slice(&[0x00, 0x3C, 0x00, 0x38]);
        for _ in 4..SB_BYTES {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            data.push((state >> 24) as u8);
        }
    }
    data
}

fn activations(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect()
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (r, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-3 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "{what}: row {r}: got {g}, want {w}");
    }
}

fn check_vs_scalar(out_dim: usize, in_dim: usize) {
    assert!(out_dim * in_dim < 8_000_000, "must stay on the serial path");
    let (w, x) = (weights(out_dim, in_dim), activations(in_dim));
    let got = matmul_q4k_f32_dispatch(&w, &x, out_dim, in_dim);
    let want = matmul_q4k_f32_scalar(&w, &x, out_dim, in_dim);
    assert_close(&got, &want, &format!("{out_dim}x{in_dim}"));
}

#[test]
fn single_superblock_row_matches_scalar() {
    check_vs_scalar(1, 256);
}

#[test]
fn many_rows_each_read_their_own_weights() {
    // A row-stride bug (rows reading sb 0 of every row) only shows with out_dim > 1
    // AND several super-blocks per row.
    check_vs_scalar(64, 1024);
}

#[test]
fn rows_differ_when_their_weights_differ() {
    let (out_dim, in_dim) = (16, 512);
    let (w, x) = (weights(out_dim, in_dim), activations(in_dim));
    let got = matmul_q4k_f32_dispatch(&w, &x, out_dim, in_dim);
    assert!(got.windows(2).any(|p| p[0] != p[1]), "rows collapsed: {got:?}");
}

#[test]
fn ragged_tail_is_masked_to_zero() {
    // in_dim = 300 -> 2 super-blocks, the second only 44 real activations. The oracle is
    // the same weights against the input zero-padded to 512, run through the scalar
    // kernel: masked lanes must contribute exactly nothing.
    let (out_dim, in_dim, padded) = (5, 300, 512);
    let w = weights(out_dim, in_dim);
    let x = activations(in_dim);
    let mut xp = x.clone();
    xp.resize(padded, 0.0);
    let got = matmul_q4k_f32_dispatch(&w, &x, out_dim, in_dim);
    let want = matmul_q4k_f32_scalar(&w, &xp, out_dim, padded);
    assert_close(&got, &want, "ragged tail");
}

#[test]
fn tall_matrix_just_under_the_parallel_threshold() {
    check_vs_scalar(3_906, 2_048); // 7,999,488 < 8,000,000
}
