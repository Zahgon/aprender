// PMAT-298: True AVX-512 Q4K x Q8K dot product.
// Uses 512-bit _mm512_maddubs_epi16 to process 2 chunks at once.
//
// Q4K data layout per super-block (128 qs bytes, 4 chunks of 32):
//   chunk c (32 bytes): lo nibbles -> Q8 block 2c, hi nibbles -> Q8 block 2c+1
//
// Q8K data layout (256 i8 values, 8 blocks of 32):
//   block b: q8[b*32 .. b*32+31]
//
// AVX-512 processes 2 chunks (64 bytes Q4K, 128 bytes Q8K) per iteration.

/// Compute Q4xQ8 dot products for 8 blocks using AVX-512 (512-bit).
///
/// Returns 8 i32 dot products, one per Q8 block (32-value sub-block).
/// Processes 2 Q4K chunks per 512-bit operation (2x vs AVX2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn compute_q4_q8_dots_8blocks_avx512(
    qs_ptr: *const u8,
    q8_ptr: *const i8,
) -> [i32; 8] {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let nibble_mask = _mm512_set1_epi8(0x0F_i8);
    let ones_16 = _mm512_set1_epi16(1);

    let mut dots = [0i32; 8];

    // Process chunks 0+1 together (64 Q4K bytes, Q8 blocks 0-3)
    {
        // Load 64 Q4K packed bytes (chunks 0 and 1)
        let q4_01 = _mm512_loadu_si512(qs_ptr.cast::<__m512i>());

        // Extract lo nibbles (64 unsigned bytes)
        let q4_01_lo = _mm512_and_si512(q4_01, nibble_mask);
        // Extract hi nibbles
        let q4_01_hi = _mm512_and_si512(_mm512_srli_epi16(q4_01, 4), nibble_mask);

        // Load Q8K data for lo nibbles: blocks 0 and 2 (64 signed bytes)
        // chunk 0 lo -> Q8 block 0 (q8[0..31]), chunk 1 lo -> Q8 block 2 (q8[64..95])
        // These are NOT contiguous! Block 0 is at offset 0, block 2 is at offset 64.
        // Load as two 256-bit halves into one 512-bit register.
        let q8_b0 = _mm256_loadu_si256(q8_ptr.cast::<__m256i>()); // block 0
        let q8_b2 = _mm256_loadu_si256(q8_ptr.add(64).cast::<__m256i>()); // block 2
        let q8_lo = _mm512_inserti64x4(_mm512_castsi256_si512(q8_b0), q8_b2, 1);

        // Load Q8K for hi nibbles: blocks 1 and 3
        let q8_b1 = _mm256_loadu_si256(q8_ptr.add(32).cast::<__m256i>()); // block 1
        let q8_b3 = _mm256_loadu_si256(q8_ptr.add(96).cast::<__m256i>()); // block 3
        let q8_hi = _mm512_inserti64x4(_mm512_castsi256_si512(q8_b1), q8_b3, 1);

        // Q4 unsigned x Q8 signed: maddubs then madd to get i32 partial sums
        let prod_lo = _mm512_maddubs_epi16(q4_01_lo, q8_lo);
        let prod_hi = _mm512_maddubs_epi16(q4_01_hi, q8_hi);
        let sum_lo = _mm512_madd_epi16(prod_lo, ones_16);
        let sum_hi = _mm512_madd_epi16(prod_hi, ones_16);

        // sum_lo has 16 i32 values: lo[0..7] from chunk 0, lo[8..15] from chunk 1
        // Need horizontal sum of each 8-element half
        let lo_256_a = _mm512_castsi512_si256(sum_lo);
        let lo_256_b = _mm512_extracti64x4_epi64(sum_lo, 1);
        dots[0] = hsum_256(lo_256_a); // chunk 0 lo = block 0
        dots[2] = hsum_256(lo_256_b); // chunk 1 lo = block 2

        let hi_256_a = _mm512_castsi512_si256(sum_hi);
        let hi_256_b = _mm512_extracti64x4_epi64(sum_hi, 1);
        dots[1] = hsum_256(hi_256_a); // chunk 0 hi = block 1
        dots[3] = hsum_256(hi_256_b); // chunk 1 hi = block 3
    }

    // Process chunks 2+3 together (Q8 blocks 4-7)
    {
        let q4_23 = _mm512_loadu_si512(qs_ptr.add(64).cast::<__m512i>());

        let q4_23_lo = _mm512_and_si512(q4_23, nibble_mask);
        let q4_23_hi = _mm512_and_si512(_mm512_srli_epi16(q4_23, 4), nibble_mask);

        let q8_b4 = _mm256_loadu_si256(q8_ptr.add(128).cast::<__m256i>());
        let q8_b6 = _mm256_loadu_si256(q8_ptr.add(192).cast::<__m256i>());
        let q8_lo = _mm512_inserti64x4(_mm512_castsi256_si512(q8_b4), q8_b6, 1);

        let q8_b5 = _mm256_loadu_si256(q8_ptr.add(160).cast::<__m256i>());
        let q8_b7 = _mm256_loadu_si256(q8_ptr.add(224).cast::<__m256i>());
        let q8_hi = _mm512_inserti64x4(_mm512_castsi256_si512(q8_b5), q8_b7, 1);

        let prod_lo = _mm512_maddubs_epi16(q4_23_lo, q8_lo);
        let prod_hi = _mm512_maddubs_epi16(q4_23_hi, q8_hi);
        let sum_lo = _mm512_madd_epi16(prod_lo, ones_16);
        let sum_hi = _mm512_madd_epi16(prod_hi, ones_16);

        let lo_256_a = _mm512_castsi512_si256(sum_lo);
        let lo_256_b = _mm512_extracti64x4_epi64(sum_lo, 1);
        dots[4] = hsum_256(lo_256_a);
        dots[6] = hsum_256(lo_256_b);

        let hi_256_a = _mm512_castsi512_si256(sum_hi);
        let hi_256_b = _mm512_extracti64x4_epi64(sum_hi, 1);
        dots[5] = hsum_256(hi_256_a);
        dots[7] = hsum_256(hi_256_b);
    }

    dots
}

/// Compute Q8K block sums using 512-bit loads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn compute_q8_sums_avx512(q8_ptr: *const i8) -> [i32; 8] {
    use std::arch::x86_64::*;

    let ones = _mm256_set1_epi16(1);
    let mut sums = [0i32; 8];

    // Each Q8 block is 32 signed bytes. Sum via maddubs(ones_u8, q8_signed).
    // maddubs treats first arg as unsigned, second as signed.
    let ones_u8 = _mm256_set1_epi8(1);

    for block in 0..8 {
        let q8 = _mm256_loadu_si256(q8_ptr.add(block * 32).cast::<__m256i>());
        // maddubs(1, q8) = sum of pairs as i16
        let pair_sums = _mm256_maddubs_epi16(ones_u8, q8);
        // madd(pair_sums, 1) = sum of pairs of i16 as i32
        let quad_sums = _mm256_madd_epi16(pair_sums, ones);
        sums[block] = hsum_256(quad_sums);
    }

    sums
}

/// Horizontal sum of 8 i32 values in a 256-bit register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn hsum_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b10_11_00_01));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b00_00_10_10));
    _mm_cvtsi128_si32(sum32)
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx512_detection() {
        let has = crate::isa::cpu_feature!("avx512f") && crate::isa::cpu_feature!("avx512bw");
        eprintln!("AVX-512 F+BW: {has}");
    }

    #[cfg(target_arch = "x86_64")]
    fn has_avx512() -> bool {
        crate::isa::cpu_feature!("avx512f") && crate::isa::cpu_feature!("avx512bw")
    }

    /// Scalar reference: chunk c (32 bytes) lo nibbles -> block 2c, hi -> block 2c+1.
    #[cfg(target_arch = "x86_64")]
    fn scalar_dots(qs: &[u8; 128], q8: &[i8; 256]) -> [i32; 8] {
        let mut dots = [0i32; 8];
        for c in 0..4 {
            for i in 0..32 {
                let b = qs[c * 32 + i];
                dots[2 * c] += i32::from(b & 0x0F) * i32::from(q8[(2 * c) * 32 + i]);
                dots[2 * c + 1] += i32::from(b >> 4) * i32::from(q8[(2 * c + 1) * 32 + i]);
            }
        }
        dots
    }

    #[cfg(target_arch = "x86_64")]
    fn pattern() -> ([u8; 128], [i8; 256]) {
        let mut qs = [0u8; 128];
        let mut q8 = [0i8; 256];
        for (i, b) in qs.iter_mut().enumerate() {
            *b = ((i * 37 + 11) % 256) as u8;
        }
        for (i, v) in q8.iter_mut().enumerate() {
            *v = ((i as i32 * 53 + 7) % 255 - 127) as i8;
        }
        (qs, q8)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dots_8blocks_matches_scalar_reference() {
        if !has_avx512() {
            return;
        }
        let (qs, q8) = pattern();
        // SAFETY: AVX-512 F+BW checked above; buffers are 128 / 256 bytes as required.
        let got = unsafe { super::compute_q4_q8_dots_8blocks_avx512(qs.as_ptr(), q8.as_ptr()) };
        assert_eq!(got, scalar_dots(&qs, &q8));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dots_8blocks_each_block_isolated() {
        if !has_avx512() {
            return;
        }
        // Only Q8 block `b` is non-zero (all 1s) and every Q4 nibble is 15:
        // exactly dots[b] must equal 32 * 15 and every other lane must be 0.
        // Catches swapped lane/block routing (e.g. dots[1] <-> dots[2]).
        let qs = [0xFFu8; 128];
        for b in 0..8 {
            let mut q8 = [0i8; 256];
            q8[b * 32..(b + 1) * 32].fill(1);
            // SAFETY: AVX-512 F+BW checked above; buffers sized as required.
            let got =
                unsafe { super::compute_q4_q8_dots_8blocks_avx512(qs.as_ptr(), q8.as_ptr()) };
            let mut want = [0i32; 8];
            want[b] = 32 * 15;
            assert_eq!(got, want, "block {b}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dots_8blocks_negative_q8_is_signed() {
        if !has_avx512() {
            return;
        }
        let qs = [0x11u8; 128]; // every nibble is 1
        let q8 = [-3i8; 256];
        // SAFETY: AVX-512 F+BW checked above; buffers sized as required.
        let got = unsafe { super::compute_q4_q8_dots_8blocks_avx512(qs.as_ptr(), q8.as_ptr()) };
        assert_eq!(got, [-96; 8]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_q8_sums_matches_scalar_and_is_signed() {
        if !has_avx512() {
            return;
        }
        let (_, q8) = pattern();
        // SAFETY: AVX-512 F+BW checked above; q8 holds 256 bytes.
        let got = unsafe { super::compute_q8_sums_avx512(q8.as_ptr()) };
        let mut want = [0i32; 8];
        for (b, w) in want.iter_mut().enumerate() {
            *w = q8[b * 32..(b + 1) * 32].iter().map(|&v| i32::from(v)).sum();
        }
        assert_eq!(got, want);
        let neg = [-5i8; 256];
        // SAFETY: as above.
        let got = unsafe { super::compute_q8_sums_avx512(neg.as_ptr()) };
        assert_eq!(got, [-160; 8]);
    }
}
