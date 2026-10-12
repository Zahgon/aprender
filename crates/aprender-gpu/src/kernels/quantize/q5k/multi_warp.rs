//! Multi-warp Q5_K GEMV kernel for decode throughput (#5221)
//!
//! `Q5KGemvKernel` runs one warp per row and walks the super-blocks in sequence.
//! On Qwen3.5-4B (RTX 4090) it reached 223 GB/s against 776 GB/s for
//! llama.cpp's Q5_K path, the largest per-type gap in decode. This kernel applies
//! the `MultiWarpQ6KGemvKernel` shape to Q5_K.
//!
//! # Design by Contract
//!
//! ## Preconditions
//! - `k` > 0 (need not be multiple of 256 — bounds-checked per GH-215)
//! - `n` > 0
//! - `num_warps` ∈ {1, 2, 3, 4, 6, 8}
//! - `w_ptr`: N × ceil(K/256) × 176 bytes of Q5_K super-blocks (row-major)
//! - `x_ptr`: K contiguous f32 activation values
//! - `y_ptr`: N writable f32 output values
//!
//! ## Postcondition
//! For each row r ∈ [0, N):
//!   y[r] = Σ_sb Σ_{j=0}^{255} dequant_q5k(w[r][sb], j) × x[sb*256 + j]
//! where, for sub-block s = j / 32 and l = j % 32 (ggml `block_q5_K`):
//!   ql    = (qs[32 * (s / 2) + l] >> (4 * (s % 2))) & 0xF
//!   qh    = (qh[l] >> s) & 1
//!   value = d × scale[s] × (ql + 16 × qh) − dmin × min[s]
//!
//! ## Invariants
//! - Lane l of every warp owns qs word l (one coalesced u32 load: 128 bytes
//!   per warp), i.e. chunk c = l / 8 (sub-blocks 2c and 2c+1) at positions
//!   p..p+3 with p = 4 * (l % 8), plus the matching qh word at p (one u32). The
//!   fifth bits are merged into all four bytes at once with byte-lane masks.
//! - The 12 scale bytes are three u32 loads (lanes 0-2) broadcast by shuffle;
//!   each lane selects its two (scale, min) pairs once per super-block.
//! - Each warp processes super-blocks [warp_id, warp_id + num_warps, ...]
//!   (disjoint partition — no write conflicts).
//! - Cross-warp reduction via shared memory with bar.sync 0 (PARITY-114: all
//!   threads reach the barrier before any thread reads shared memory).
//!
//! ## Parity Contract
//! For any (k, n, w, x): `MultiWarpQ5KGemvKernel.y ≈ Q5KGemvKernel.y` within f32
//! summation-order tolerance (the reduction order differs).

use crate::kernels::quantize::{Kernel, Q5K_SUPER_BLOCK_BYTES, Q5K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, VirtualReg};

/// Byte offset of the 12 packed scale/min bytes in a Q5_K super-block.
const SCALES_OFFSET: u64 = 4;
/// Byte offset of `qh[32]` (the fifth bit of every value).
const QH_OFFSET: u64 = 16;
/// Byte offset of `qs[128]` (the low four bits, two values per byte).
const QS_OFFSET: u64 = 48;

/// Multi-warp Q5_K GEMV kernel (#5221)
///
/// Multiple warps per block stride across super-blocks, with cross-warp
/// reduction via shared memory. One block per output row.
pub struct MultiWarpQ5KGemvKernel {
    /// K dimension (input dimension, need not be multiple of 256 — bounds-checked)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Number of warps per block
    pub num_warps: u32,
}

impl MultiWarpQ5KGemvKernel {
    /// Create with default 3 warps (96 threads)
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, num_warps: 3 }
    }

    /// Create with specified warp count
    ///
    /// # Contract
    /// `num_warps` must be in {1, 2, 3, 4, 6, 8}.
    #[must_use]
    pub fn with_warps(k: u32, n: u32, num_warps: u32) -> Self {
        debug_assert!(
            matches!(num_warps, 1 | 2 | 3 | 4 | 6 | 8),
            "num_warps should be in {{1,2,3,4,6,8}}, got {num_warps}"
        );
        Self { k, n, num_warps }
    }
}

impl Kernel for MultiWarpQ5KGemvKernel {
    fn name(&self) -> &str {
        "mwv_q5k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let smem_size = (num_warps * 4) as usize; // f32 per warp for cross-warp reduction

        PtxKernel::new("mwv_q5k_gemv")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .shared_memory(smem_size)
            .build(move |ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);

                // Precondition: block_id < n_dim
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "mwv_q5k_exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                // Ceiling division: (k + 255) / 256
                let k_rounded = ctx.add_u32(k_dim, Q5K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q5K_SUPER_BLOCK_SIZE);

                let sb_bytes_c = ctx.mov_u32_imm(Q5K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes_c);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Per-lane constants, fixed for the whole row. Lane l owns the
                // u32 qs word 4l, i.e. chunk c = l / 8 (sub-blocks 2c, 2c+1) and
                // positions p..p+3 with p = 4 * (l % 8); its qh word is the one at p.
                let four_u32 = ctx.mov_u32_imm(4);
                let three_u32 = ctx.mov_u32_imm(3);
                let seven_u32 = ctx.mov_u32_imm(7);
                let chunk = ctx.shr_u32(lane_id, three_u32);
                let lane_in_chunk = ctx.and_u32(lane_id, seven_u32);
                let two_u32 = ctx.mov_u32_imm(2);
                let pos = ctx.shl_u32(lane_in_chunk, two_u32);
                let pos_64 = ctx.cvt_u64_u32(pos);
                let lane_x4 = ctx.mul_u32_reg(lane_id, four_u32);
                let lane_x4_64 = ctx.cvt_u64_u32(lane_x4);
                let chunk_x2 = ctx.add_u32_reg(chunk, chunk);
                let chunk_x64 = ctx.mul_u32(chunk, 64);
                let x_lane_off = ctx.add_u32_reg(chunk_x64, pos);
                let three_c = ctx.mov_u32_imm(3);
                let is_scale_lane = ctx.setp_lt_u32(lane_id, three_c);
                let one_c = ctx.mov_u32_imm(1);
                let two_c = ctx.mov_u32_imm(2);
                let is_c1 = ctx.setp_eq_u32(chunk, one_c);
                let is_c2 = ctx.setp_eq_u32(chunk, two_c);
                let is_c3 = ctx.setp_eq_u32(chunk, three_c);

                // Invariant: each warp starts at warp_id, strides by num_warps
                let sb_idx_z = ctx.mov_u32_imm(0);
                let sb_idx = ctx.add_u32_reg(sb_idx_z, warp_id);
                let nw_reg = ctx.mov_u32_imm(num_warps);

                ctx.label("mwv_q5k_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "mwv_q5k_sb_end");

                let sb_off = ctx.mul_wide_u32(sb_idx, Q5K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // Issue the per-lane quant loads first (coalesced u32: 128 B of qs
                // and 32 B of qh per warp), so they overlap the scale decode.
                let qs_off = ctx.mov_u64_imm(QS_OFFSET);
                let qs_base = ctx.add_u64(sb_addr, qs_off);
                let qs_addr = ctx.add_u64(qs_base, lane_x4_64);
                let qs_word = ctx.ld_global_u32(qs_addr);
                let qh_off = ctx.mov_u64_imm(QH_OFFSET);
                let qh_base = ctx.add_u64(sb_addr, qh_off);
                let qh_addr = ctx.add_u64(qh_base, pos_64);
                let qh_word = ctx.ld_global_u32(qh_addr);

                // d and dmin (f16 at offsets 0 and 2)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two_64 = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two_64);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // ============================================================
                // Scales: lanes 0-2 each load one u32 of the 12 packed bytes,
                // broadcast to all lanes via warp shuffle.
                // ============================================================
                let scales_off = ctx.mov_u64_imm(SCALES_OFFSET);
                let scales_base = ctx.add_u64(sb_addr, scales_off);
                let scale_addr = ctx.add_u64(scales_base, lane_x4_64);
                let my_scale_word = ctx.mov_u32_imm(0);
                ctx.branch_if_not(is_scale_lane, "mwv_q5k_skip_scale_load");
                ctx.ld_global_u32_into(my_scale_word, scale_addr);
                ctx.label("mwv_q5k_skip_scale_load");

                let mut words = Vec::with_capacity(3);
                for i in 0..3u32 {
                    words.push(ctx.shfl_idx_u32(my_scale_word, i, 0xFFFF_FFFF));
                }
                let mask_8bit = ctx.mov_u32_imm(0xFF);
                let mut s = Vec::with_capacity(12);
                for i in 0..12usize {
                    let w = words[i / 4];
                    let byte = if i % 4 == 0 {
                        ctx.and_u32(w, mask_8bit)
                    } else {
                        let sh = ctx.mov_u32_imm(8 * (i as u32 % 4));
                        let t = ctx.shr_u32(w, sh);
                        ctx.and_u32(t, mask_8bit)
                    };
                    s.push(byte);
                }

                // llama.cpp get_scale_min_k4, with every index a compile-time constant:
                //   j < 4:  scale = s[j] & 63,  min = s[j+4] & 63
                //   j >= 4: scale = (s[j+4] & 0xF) | ((s[j-4] >> 6) << 4)
                //           min   = (s[j+4] >> 4)  | ((s[j]   >> 6) << 4)
                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let six_u32 = ctx.mov_u32_imm(6);
                let mut d_scale = Vec::with_capacity(8);
                let mut dmin_min = Vec::with_capacity(8);
                for j in 0..8usize {
                    let (sc, mn) = if j < 4 {
                        (
                            ctx.and_u32(s[j], mask_6bit),
                            ctx.and_u32(s[j + 4], mask_6bit),
                        )
                    } else {
                        let sc_lo = ctx.and_u32(s[j + 4], mask_4bit);
                        let sc_hi = ctx.shr_u32(s[j - 4], six_u32);
                        let sc_hi = ctx.shl_u32(sc_hi, four_u32);
                        let mn_lo = ctx.shr_u32(s[j + 4], four_u32);
                        let mn_hi = ctx.shr_u32(s[j], six_u32);
                        let mn_hi = ctx.shl_u32(mn_hi, four_u32);
                        (ctx.or_u32(sc_lo, sc_hi), ctx.or_u32(mn_lo, mn_hi))
                    };
                    let sc_f32 = ctx.cvt_f32_u32(sc);
                    let mn_f32 = ctx.cvt_f32_u32(mn);
                    d_scale.push(ctx.mul_f32(d, sc_f32));
                    dmin_min.push(ctx.mul_f32(dmin, mn_f32));
                }

                // This lane's sub-blocks are 2c (low nibbles) and 2c+1 (high).
                let mut pick = |v: &[VirtualReg], odd: usize| {
                    let r = ctx.selp_f32(is_c1, v[2 + odd], v[odd]);
                    let r = ctx.selp_f32(is_c2, v[4 + odd], r);
                    ctx.selp_f32(is_c3, v[6 + odd], r)
                };
                let ds_lo = pick(&d_scale, 0);
                let dm_lo = pick(&dmin_min, 0);
                let ds_hi = pick(&d_scale, 1);
                let dm_hi = pick(&dmin_min, 1);

                // ============================================================
                // Quant bytes: per byte b, low = (qs nibble) | (qh bit 2c) << 4,
                // high = (qs high nibble) | (qh bit 2c+1) << 4, all 4 bytes at once.
                // ============================================================
                let nib_mask = ctx.mov_u32_imm(0x0F0F_0F0F);
                let bit_mask = ctx.mov_u32_imm(0x0101_0101);
                let qh_s = ctx.shr_u32(qh_word, chunk_x2);
                let lo4 = ctx.and_u32(qs_word, nib_mask);
                let hi4 = ctx.shr_u32(qs_word, four_u32);
                let hi4 = ctx.and_u32(hi4, nib_mask);
                let hb_lo = ctx.and_u32(qh_s, bit_mask);
                let hb_lo = ctx.shl_u32(hb_lo, four_u32);
                let qh_s1 = ctx.shr_u32(qh_s, one_c);
                let hb_hi = ctx.and_u32(qh_s1, bit_mask);
                let hb_hi = ctx.shl_u32(hb_hi, four_u32);
                let q_lo = ctx.or_u32(lo4, hb_lo);
                let q_hi = ctx.or_u32(hi4, hb_hi);

                let sb_k_base = ctx.mul_u32(sb_idx, Q5K_SUPER_BLOCK_SIZE);
                let x_lo = ctx.add_u32_reg(sb_k_base, x_lane_off);

                let thread_partial = ctx.mov_f32_imm(0.0);
                for (half, (q, ds, dm)) in [(q_lo, ds_lo, dm_lo), (q_hi, ds_hi, dm_hi)]
                    .into_iter()
                    .enumerate()
                {
                    for b in 0..4u32 {
                        let byte = if b == 0 {
                            ctx.and_u32(q, mask_8bit)
                        } else {
                            let sh = ctx.mov_u32_imm(8 * b);
                            let t = ctx.shr_u32(q, sh);
                            ctx.and_u32(t, mask_8bit)
                        };
                        let quant_f32 = ctx.cvt_f32_u32(byte);
                        // value = d*scale*quant - dmin*min
                        let scaled = ctx.mul_f32(ds, quant_f32);
                        let dequant = ctx.sub_f32(scaled, dm);

                        // x[sb*256 + 64c + 32*half + p + b], bounds-checked (GH-215)
                        let off = ctx.mov_u32_imm(32 * half as u32 + b);
                        let x_idx = ctx.add_u32_reg(x_lo, off);
                        let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                        let x_bytes = ctx.mul_u64(x_idx_64, 4);
                        let x_addr = ctx.add_u64(x_ptr, x_bytes);
                        let in_bounds = ctx.setp_lt_u32(x_idx, k_dim);
                        let x_val = ctx.ld_global_f32_predicated(x_addr, in_bounds, 0.0);

                        ctx.fma_f32_inplace(thread_partial, x_val, dequant);
                    }
                }

                ctx.add_f32_inplace(acc, thread_partial);

                // Invariant: stride by num_warps (disjoint partition)
                ctx.add_u32_reg_inplace(sb_idx, nw_reg);
                ctx.branch("mwv_q5k_sb_loop");

                ctx.label("mwv_q5k_sb_end");

                // Phase 1: intra-warp reduction via shuffle
                let t16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t16);
                let t8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t8);
                let t4 = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t4);
                let t2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t2);
                let t1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t1);

                // Phase 2: cross-warp reduction via shared memory (PARITY-114)
                let z = ctx.mov_u32_imm(0);
                let is_l0 = ctx.setp_eq_u32(lane_id, z);
                ctx.branch_if_not(is_l0, "mwv_q5k_skip_sm");
                let f4 = ctx.mov_u32_imm(4);
                let wo = ctx.mul_u32_reg(warp_id, f4);
                let sa = ctx.cvt_u64_u32(wo);
                ctx.st_shared_f32(sa, acc);
                ctx.label("mwv_q5k_skip_sm");

                // PARITY-114: ALL threads reach this barrier
                ctx.bar_sync(0);

                let is_t0 = ctx.setp_eq_u32(thread_id, z);
                ctx.branch_if_not(is_t0, "mwv_q5k_exit");

                let fs = ctx.mov_f32_imm(0.0);
                for w in 0..num_warps {
                    let wo = ctx.mov_u64_imm(u64::from(w * 4));
                    let pv = ctx.ld_shared_f32(wo);
                    ctx.add_f32_inplace(fs, pv);
                }

                let yo = ctx.mul_wide_u32(block_id, 4);
                let ya = ctx.add_u64(y_ptr, yo);
                ctx.st_global_f32(ya, fs);

                ctx.label("mwv_q5k_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Contract: kernel builds valid PTX for Qwen3.5-4B attn_qkv dimensions
    #[test]
    fn test_mwv_q5k_builds_qwen35_4b() {
        let kernel = MultiWarpQ5KGemvKernel::with_warps(2560, 8192, 4);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".visible .entry mwv_q5k_gemv"));
        assert!(
            ptx.contains(".shared"),
            "Must use shared memory for cross-warp reduction"
        );
        assert!(
            ptx.contains("bar.sync"),
            "Must have barrier for cross-warp safety"
        );
    }

    /// Contract: kernel has the same parameter signature as Q5KGemvKernel
    #[test]
    fn test_mwv_q5k_parameters() {
        let ptx = MultiWarpQ5KGemvKernel::new(256, 64).emit_ptx();
        for p in ["y_ptr", "w_ptr", "x_ptr", "k_dim", "n_dim"] {
            assert!(ptx.contains(p), "Must have parameter {p}");
        }
    }

    /// Contract: shared memory size = num_warps * 4 bytes
    #[test]
    fn test_mwv_q5k_shared_memory_size() {
        for warps in [1, 2, 3, 4, 6, 8] {
            let k = MultiWarpQ5KGemvKernel::with_warps(256, 64, warps).build_ptx();
            assert_eq!(k.shared_memory_bytes(), (warps * 4) as usize);
        }
    }

    /// PARITY-114: barrier safety contract, every warp count
    #[test]
    fn test_mwv_q5k_barrier_safety() {
        for warps in [1, 2, 3, 4, 6, 8] {
            let result =
                MultiWarpQ5KGemvKernel::with_warps(2560, 2560, warps).analyze_barrier_safety();
            assert!(result.is_safe, "{warps} warps: {:?}", result.violations);
        }
    }

    /// Contract: name is dimension-independent and distinct from the single-warp kernel
    #[test]
    fn test_mwv_q5k_name() {
        let a = MultiWarpQ5KGemvKernel::new(1536, 1536);
        let b = MultiWarpQ5KGemvKernel::new(4096, 4096);
        assert_eq!(a.name(), b.name());
        assert_eq!(a.name(), "mwv_q5k_gemv");
    }

    /// Contract: the quant layout offsets match ggml block_q5_K
    /// (d, dmin, scales[12], qh[32], qs[128] = 176 bytes)
    #[test]
    fn test_mwv_q5k_layout_offsets() {
        assert_eq!(SCALES_OFFSET, 4);
        assert_eq!(QH_OFFSET, SCALES_OFFSET + 12);
        assert_eq!(QS_OFFSET, QH_OFFSET + 32);
        assert_eq!(QS_OFFSET + 128, u64::from(Q5K_SUPER_BLOCK_BYTES));
    }
}
