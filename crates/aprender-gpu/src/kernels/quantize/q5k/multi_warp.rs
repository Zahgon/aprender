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
//! - Lane l of every warp handles value l of each of the 8 sub-blocks, so the
//!   sub-block of each of a lane's 8 values is a compile-time constant and the
//!   scale/min decode happens once per super-block, not once per value.
//! - The 32 lanes of a warp read 32 consecutive bytes of `qh` and of each `qs`
//!   chunk (coalesced), where `Q5KGemvKernel` recomputes every address per value.
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
use crate::ptx::{PtxKernel, PtxReg, PtxType};

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

                // Per-lane byte offsets, fixed for the whole row
                let lane_64 = ctx.cvt_u64_u32(lane_id);
                let lane_mod_12 = ctx.rem_u32(lane_id, 12);
                let lane_mod_12_64 = ctx.cvt_u64_u32(lane_mod_12);
                let twelve = ctx.mov_u32_imm(12);
                let is_scale_lane = ctx.setp_lt_u32(lane_id, twelve);

                // Invariant: each warp starts at warp_id, strides by num_warps
                let sb_idx_z = ctx.mov_u32_imm(0);
                let sb_idx = ctx.add_u32_reg(sb_idx_z, warp_id);
                let nw_reg = ctx.mov_u32_imm(num_warps);

                ctx.label("mwv_q5k_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "mwv_q5k_sb_end");

                let sb_off = ctx.mul_wide_u32(sb_idx, Q5K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // d and dmin (f16 at offsets 0 and 2)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two_64 = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two_64);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // ============================================================
                // Scales: lanes 0-11 each load one of the 12 packed bytes,
                // broadcast to all lanes via warp shuffle (PAR-066 pattern).
                // ============================================================
                let scales_off = ctx.mov_u64_imm(SCALES_OFFSET);
                let scales_base = ctx.add_u64(sb_addr, scales_off);
                let scale_addr = ctx.add_u64(scales_base, lane_mod_12_64);
                let my_scale_byte = ctx.mov_u32_imm(0);
                ctx.branch_if_not(is_scale_lane, "mwv_q5k_skip_scale_load");
                let scale_u8 = ctx.ld_global_u8(scale_addr);
                let scale_u32 = ctx.cvt_u32_u8(scale_u8);
                ctx.mov_u32_reg(my_scale_byte, scale_u32);
                ctx.label("mwv_q5k_skip_scale_load");

                let mut s = Vec::with_capacity(12);
                for i in 0..12u32 {
                    s.push(ctx.shfl_idx_u32(my_scale_byte, i, 0xFFFF_FFFF));
                }

                // llama.cpp get_scale_min_k4, with every index a compile-time constant:
                //   j < 4:  scale = s[j] & 63,  min = s[j+4] & 63
                //   j >= 4: scale = (s[j+4] & 0xF) | ((s[j-4] >> 6) << 4)
                //           min   = (s[j+4] >> 4)  | ((s[j]   >> 6) << 4)
                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let four_u32 = ctx.mov_u32_imm(4);
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

                // ============================================================
                // Quant bits: one qh byte per lane (bit j = sub-block j), and
                // one qs byte per lane per 64-value chunk (low nibble = even
                // sub-block, high nibble = odd sub-block).
                // ============================================================
                let qh_off = ctx.mov_u64_imm(QH_OFFSET);
                let qh_base = ctx.add_u64(sb_addr, qh_off);
                let qh_addr = ctx.add_u64(qh_base, lane_64);
                let qh_u8 = ctx.ld_global_u8(qh_addr);
                let qh_byte = ctx.cvt_u32_u8(qh_u8);

                let qs_off = ctx.mov_u64_imm(QS_OFFSET);
                let qs_base = ctx.add_u64(sb_addr, qs_off);
                let qs_lane = ctx.add_u64(qs_base, lane_64);

                let sb_k_base = ctx.mul_u32(sb_idx, Q5K_SUPER_BLOCK_SIZE);
                let x_lane_idx = ctx.add_u32_reg(sb_k_base, lane_id);

                let thread_partial = ctx.mov_f32_imm(0.0);
                let one_u32 = ctx.mov_u32_imm(1);
                let sixteen_u32 = ctx.mov_u32_imm(16);

                for chunk in 0..4u32 {
                    let chunk_off = ctx.mov_u64_imm(u64::from(chunk * 32));
                    let qs_addr = ctx.add_u64(qs_lane, chunk_off);
                    let qs_u8 = ctx.ld_global_u8(qs_addr);
                    let qs_byte = ctx.cvt_u32_u8(qs_u8);

                    for half in 0..2u32 {
                        let j = chunk * 2 + half;
                        let ql = if half == 0 {
                            ctx.and_u32(qs_byte, mask_4bit)
                        } else {
                            ctx.shr_u32(qs_byte, four_u32)
                        };
                        let j_reg = ctx.mov_u32_imm(j);
                        let qh_shifted = ctx.shr_u32(qh_byte, j_reg);
                        let qh_bit = ctx.and_u32(qh_shifted, one_u32);
                        let qh_scaled = ctx.mul_u32_reg(qh_bit, sixteen_u32);
                        let quant = ctx.add_u32_reg(ql, qh_scaled);
                        let quant_f32 = ctx.cvt_f32_u32(quant);

                        // value = d*scale[j]*quant - dmin*min[j]
                        let scaled = ctx.mul_f32(d_scale[j as usize], quant_f32);
                        let dequant = ctx.sub_f32(scaled, dmin_min[j as usize]);

                        // x[sb*256 + 32*j + lane], bounds-checked (GH-215)
                        let j_off = ctx.mov_u32_imm(j * 32);
                        let x_idx = ctx.add_u32_reg(x_lane_idx, j_off);
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
