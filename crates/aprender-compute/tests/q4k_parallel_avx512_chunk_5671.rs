//! Large Q4_K matmuls (`out_dim * in_dim >= 8M`) run on the threaded path, whose
//! per-thread kernel is `compute_chunk_q4k_avx512` on AVX-512 hosts. Result must match
//! the scalar reference for every row, including the ragged last chunk and rows spanning several
//! super-blocks (#5671).
#![cfg(target_arch = "x86_64")]

use trueno::backends::q4k::{matmul_q4k_f32_dispatch, matmul_q4k_f32_scalar};

const SB_BYTES: usize = 144;
const SB_ELEMS: usize = 256;

/// Deterministic Q4_K rows: d=1.0, dmin=0.5 (f16), LCG scales/quants.
fn weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let sbs = in_dim.div_ceil(SB_ELEMS);
    let mut state = 0x9E37_79B9u32;
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

fn activations(in_dim: usize) -> Vec<f32> {
    (0..in_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect()
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (r, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-3 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "{what}: row {r}: got {g}, want {w}");
    }
}

fn check(out_dim: usize, in_dim: usize) {
    assert!(out_dim * in_dim >= 8_000_000, "must reach the parallel path");
    let w = weights(out_dim, in_dim);
    let x = activations(in_dim);
    let got = matmul_q4k_f32_dispatch(&w, &x, out_dim, in_dim);
    let want = matmul_q4k_f32_scalar(&w, &x, out_dim, in_dim);
    assert_close(&got, &want, &format!("{out_dim}x{in_dim}"));
}

#[test]
fn aligned_in_dim_with_ragged_last_chunk() {
    check(31_253, 256);
}

#[test]
fn multi_superblock_rows_accumulate_across_superblocks() {
    check(15_629, 512);
}

#[test]
fn first_and_last_rows_are_distinct_and_nonzero() {
    let (out_dim, in_dim) = (31_253, 256);
    let w = weights(out_dim, in_dim);
    let x = activations(in_dim);
    let got = matmul_q4k_f32_dispatch(&w, &x, out_dim, in_dim);
    assert!(got[0] != 0.0 && got[out_dim - 1] != 0.0);
    assert!(got[0] != got[out_dim - 1], "rows must come from distinct weights");
}
