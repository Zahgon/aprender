//! AVX-512 `sub` at/above the non-temporal threshold (8192 elements) routes through
//! `sub_nt`; its 64-wide unrolled body, 16-wide loop and scalar tail must all equal
//! `a[i] - b[i]` (#5649). Skips on hosts without AVX-512F.
#![cfg(target_arch = "x86_64")]

use trueno::backends::avx512::Avx512Backend;
use trueno::backends::VectorBackend;

const NT: usize = 8192;

fn inputs(len: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..len).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let b = (0..len).map(|i| ((i % 97) as f32) * 0.25 - 3.0).collect();
    (a, b)
}

fn run_sub(len: usize) {
    let (a, b) = inputs(len);
    let (mut buf, off) = aligned64(len);
    let out = &mut buf[off..off + len];
    // SAFETY: AVX-512F presence checked by the caller; slices are equal length and
    // `out` is 64-byte aligned, which the non-temporal stores require.
    unsafe { Avx512Backend::sub(&a, &b, out) };
    for i in 0..len {
        assert_eq!(out[i], a[i] - b[i], "len {len} index {i}");
    }
}

/// `_mm512_stream_ps` faults on a destination that is not 64-byte aligned, so the
/// result buffer is over-allocated and offset to an aligned start.
fn aligned64(len: usize) -> (Vec<f32>, usize) {
    let buf = vec![f32::NAN; len + 16];
    let off = (buf.as_ptr().align_offset(64)).min(16);
    (buf, off)
}

fn have() -> bool {
    is_x86_feature_detected!("avx512f")
}

#[test]
fn exact_threshold_multiple_of_64() {
    if have() {
        run_sub(NT);
    }
}

#[test]
fn threshold_plus_16_lane_remainder() {
    if have() {
        run_sub(NT + 16);
        run_sub(NT + 48);
    }
}

#[test]
fn threshold_plus_scalar_tail() {
    if have() {
        run_sub(NT + 1);
        run_sub(NT + 63);
        run_sub(NT + 64 + 16 + 7);
    }
}

#[test]
fn sub_is_not_commutative_through_nt_path() {
    if have() {
        let (a, b) = inputs(NT);
        let (mut abuf, ao) = aligned64(NT);
        let (mut bbuf, bo) = aligned64(NT);
        let ab = &mut abuf[ao..ao + NT];
        let ba = &mut bbuf[bo..bo + NT];
        // SAFETY: AVX-512F checked above; equal-length slices.
        unsafe {
            Avx512Backend::sub(&a, &b, ab);
            Avx512Backend::sub(&b, &a, ba);
        }
        assert!(ab.iter().zip(ba.iter()).all(|(x, y)| *x == -*y));
        assert!(ab.iter().zip(ba.iter()).any(|(x, y)| x != y));
    }
}
