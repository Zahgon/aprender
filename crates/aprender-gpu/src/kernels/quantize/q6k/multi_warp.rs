//! GH-118: Multi-warp Q6_K GEMV kernel for decode throughput on Orin
//!
//! # Design by Contract (PAR-118)
//!
//! ## Preconditions
//! - `k` > 0 (need not be multiple of 256 — bounds-checked per GH-215)
//! - `n` > 0
//! - `num_warps` ∈ {1, 2, 3, 4, 6, 8}
//! - `w_ptr`: N × ceil(K/256) × 210 bytes of Q6_K super-blocks (row-major)
//! - `x_ptr`: K contiguous f32 activation values
//! - `y_ptr`: N writable f32 output values
//!
//! ## Postcondition
//! For each row r ∈ [0, N):
//!   y[r] = Σ_{sb=0}^{ceil(K/256)-1} Σ_{j=0}^{255} dequant_q6k(w[r][sb], j) × x[sb*256 + j]
//! where dequant_q6k follows the GGML Q6_K formula:
//!   quant = (ql_nibble | (qh_2bits << 4)) - 32
//!   value = d × scale[sub_block] × quant
//!
//! ## Invariants
//! - Each warp processes super-blocks [warp_id, warp_id+num_warps, warp_id+2*num_warps, ...]
//!   (disjoint partition — no write conflicts)
//! - Intra-warp reduction via shfl.down.f32 (warp-synchronous — no barrier needed)
//! - Cross-warp reduction via shared memory with bar.sync 0 (PARITY-114 safe:
//!   all threads reach barrier unconditionally before any thread reads shared memory)
//! - Bounds check: x[idx] = 0.0 when idx >= k_dim (GH-215)
//!
//! ## Parity Contract
//! For any (k, n, w, x): MultiWarpQ6KGemvKernel.y == Q6KGemvKernel.y (bitwise for FMA)
//! Validated via `KernelParity` trait (structural) and GPU numerical tests (runtime).

use crate::kernels::quantize::{Kernel, Q6K_SUPER_BLOCK_BYTES, Q6K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Multi-warp Q6_K GEMV kernel (GH-118)
///
/// Applies the same multi-warp parallelism pattern as `MultiWarpVectorizedQ4KGemvKernel`
/// to Q6_K weights. Multiple warps per block stride across super-blocks,
/// with cross-warp reduction via shared memory.
///
/// # Performance Model
///
/// Q6_K: 210 bytes per 256 values = 0.82 bytes/value
/// At 102 GB/s (Orin): theoretical max = 102e9 / (210 * N_rows) tok/s
/// Multi-warp hides memory latency via warp-level parallelism (occupancy > 50%)
pub struct MultiWarpQ6KGemvKernel {
    /// K dimension (input dimension, need not be multiple of 256 — bounds-checked)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Number of warps per block (default: 3 for Orin, 4 for 4090)
    pub num_warps: u32,
}

impl MultiWarpQ6KGemvKernel {
    /// Create with default 3 warps (96 threads)
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, num_warps: 3 }
    }

    /// Create with specified warp count
    ///
    /// # Contract
    /// `num_warps` must be in {1, 2, 3, 4, 6, 8}.
    /// Values outside this set will produce correct but suboptimal results.
    #[must_use]
    pub fn with_warps(k: u32, n: u32, num_warps: u32) -> Self {
        debug_assert!(
            matches!(num_warps, 1 | 2 | 3 | 4 | 6 | 8),
            "num_warps should be in {{1,2,3,4,6,8}}, got {num_warps}"
        );
        Self { k, n, num_warps }
    }
}

impl Kernel for MultiWarpQ6KGemvKernel {
    fn name(&self) -> &str {
        "mwv_q6k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let smem_size = (num_warps * 4) as usize; // f32 per warp for cross-warp reduction

        PtxKernel::new("mwv_q6k_gemv")
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

                // Precondition: block_id < n_dim (bounds check)
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "mwv_q6k_exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                // Ceiling division: (k + 255) / 256
                let k_rounded = ctx.add_u32(k_dim, Q6K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q6K_SUPER_BLOCK_SIZE);

                // Row base address for this output element
                let sb_bytes_c = ctx.mov_u32_imm(Q6K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes_c);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Invariant: each warp starts at warp_id, strides by num_warps
                // This partitions super-blocks disjointly across warps.
                let sb_idx_z = ctx.mov_u32_imm(0);
                let sb_idx = ctx.add_u32_reg(sb_idx_z, warp_id);
                let nw_reg = ctx.mov_u32_imm(num_warps);

                ctx.label("mwv_q6k_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "mwv_q6k_sb_end");

                let sb_off = ctx.mul_wide_u32(sb_idx, Q6K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // Load d (f16 at offset 208)
                let d_offset = ctx.mov_u64_imm(208);
                let d_addr = ctx.add_u64(sb_addr, d_offset);
                let d_f16 = ctx.ld_global_f16(d_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // ================================================================
                // Scale loading: lanes 0-15 each load one scale byte,
                // broadcast to all lanes via warp shuffle (PAR-066 pattern).
                // Q6K scales are i8 at bytes 192-207 of the super-block.
                // ================================================================
                let scales_base_offset = ctx.mov_u64_imm(192);
                let scales_base = ctx.add_u64(sb_addr, scales_base_offset);

                let lane_mod_16 = ctx.rem_u32(lane_id, 16);
                let lane_offset = ctx.cvt_u64_u32(lane_mod_16);
                let scale_addr = ctx.add_u64(scales_base, lane_offset);

                let my_scale_byte = ctx.mov_u32_imm(0);
                let sixteen_const = ctx.mov_u32_imm(16);
                let is_low_lane = ctx.setp_lt_u32(lane_id, sixteen_const);
                ctx.branch_if_not(is_low_lane, "mwv_skip_scale_load");
                let scale_u8 = ctx.ld_global_u8(scale_addr);
                let scale_u32 = ctx.cvt_u32_u8(scale_u8);
                ctx.mov_u32_reg(my_scale_byte, scale_u32);
                ctx.label("mwv_skip_scale_load");

                // Broadcast all 16 scales via warp shuffle
                let mut scale_regs = Vec::with_capacity(16);
                for i in 0..16u32 {
                    scale_regs.push(ctx.shfl_idx_u32(my_scale_byte, i, 0xFFFF_FFFF));
                }

                // Convert i8 scales to signed f32: if >= 128, subtract 256
                let seven = ctx.mov_u32_imm(7);
                let twofiftysix_f32 = ctx.mov_f32_imm(256.0);

                let mut scale_f32s = Vec::with_capacity(16);
                for &sr in &scale_regs {
                    let sign_bit = ctx.shr_u32(sr, seven);
                    let raw_f32 = ctx.cvt_f32_u32(sr);
                    let sign_f32 = ctx.cvt_f32_u32(sign_bit);
                    let correction = ctx.mul_f32(sign_f32, twofiftysix_f32);
                    let signed_f32 = ctx.sub_f32(raw_f32, correction);
                    scale_f32s.push(signed_f32);
                }

                // Precompute d * scale for all 16 sub-block scales
                let mut ds = Vec::with_capacity(16);
                for &sf in &scale_f32s {
                    ds.push(ctx.mul_f32(d, sf));
                }

                // ================================================================
                // Dequantize 256 values: 8 offsets × 32 lanes = 256 values per sb
                //
                // Contract: identical dequant formula to Q6KGemvKernel
                //   quant = (ql_nibble | (qh_2bits << 4)) - 32
                //   scale_idx = 8 * n_idx + is + 2 * group
                //   value = d * scale[scale_idx] * quant
                // ================================================================
                let thread_partial = ctx.mov_f32_imm(0.0);
                let thirty_two_f32 = ctx.mov_f32_imm(32.0);

                // offset_params: (offset, n_idx, group, ds_even_idx, ds_odd_idx)
                // scale_idx = 8 * n_idx + is + 2 * group
                // is = 0 for lanes 0-15, 1 for lanes 16-31
                // ds_even = ds[8*n + 2*g], ds_odd = ds[8*n + 2*g + 1]
                let offset_params: [(u32, u32, u32, usize, usize); 8] = [
                    (0, 0, 0, 0, 1),     // n=0, g=0: scale_idx = 0 or 1
                    (32, 0, 1, 2, 3),    // n=0, g=1: scale_idx = 2 or 3
                    (64, 0, 2, 4, 5),    // n=0, g=2: scale_idx = 4 or 5
                    (96, 0, 3, 6, 7),    // n=0, g=3: scale_idx = 6 or 7
                    (128, 1, 0, 8, 9),   // n=1, g=0: scale_idx = 8 or 9
                    (160, 1, 1, 10, 11), // n=1, g=1: scale_idx = 10 or 11
                    (192, 1, 2, 12, 13), // n=1, g=2: scale_idx = 12 or 13
                    (224, 1, 3, 14, 15), // n=1, g=3: scale_idx = 14 or 15
                ];

                // Precompute lane_is: 0 for lanes 0-15, 1 for lanes 16-31
                let lane_is = ctx.div_u32(lane_id, 16);
                let lane_is_f32 = ctx.cvt_f32_u32(lane_is);

                // #5221: each lane needs only 6 distinct bytes per super-block,
                // ql[64n + l], ql[64n + 32 + l] and qh[128 + 32n + l] for n in {0, 1}.
                // Load each once (all coalesced across the warp) instead of once per
                // offset (16 byte loads). The FMA order below is unchanged.
                let lane_64 = ctx.cvt_u64_u32(lane_id);
                let lane_addr = ctx.add_u64(sb_addr, lane_64);
                let mut ql_bytes = Vec::with_capacity(4);
                for off in [0u64, 32, 64, 96] {
                    let o = ctx.mov_u64_imm(off);
                    let a = ctx.add_u64(lane_addr, o);
                    let byte = ctx.ld_global_u8(a);
                    ql_bytes.push(ctx.cvt_u32_u8(byte));
                }
                let mut qh_bytes = Vec::with_capacity(2);
                for off in [128u64, 160] {
                    let o = ctx.mov_u64_imm(off);
                    let a = ctx.add_u64(lane_addr, o);
                    let byte = ctx.ld_global_u8(a);
                    qh_bytes.push(ctx.cvt_u32_u8(byte));
                }

                let mask_0xf = ctx.mov_u32_imm(0xF);
                let mask_0x3 = ctx.mov_u32_imm(0x3);
                let four = ctx.mov_u32_imm(4);
                let sb_k_base = ctx.mul_u32(sb_idx, Q6K_SUPER_BLOCK_SIZE);
                let x_lane_idx = ctx.add_u32_reg(sb_k_base, lane_id);

                for &(offset, n_idx_val, group_val, ds_even, ds_odd) in &offset_params {
                    // Select ds based on lane position (is=0 → ds_even, is=1 → ds_odd)
                    let ds_diff = ctx.sub_f32(ds[ds_odd], ds[ds_even]);
                    let ds_selected = ctx.fma_f32(lane_is_f32, ds_diff, ds[ds_even]);

                    // ql byte = ql[64 * n + 32 * (group & 1) + l]; low nibble for
                    // group < 2, high nibble for group >= 2
                    let ql_byte_32 = ql_bytes[(2 * n_idx_val + (group_val & 1)) as usize];
                    let ql_nibble = if group_val < 2 {
                        ctx.and_u32(ql_byte_32, mask_0xf)
                    } else {
                        ctx.shr_u32(ql_byte_32, four)
                    };

                    // qh bits = (qh[32 * n + l] >> (2 * group)) & 3
                    let qh_byte_32 = qh_bytes[n_idx_val as usize];
                    let qh_2bits = if group_val == 0 {
                        ctx.and_u32(qh_byte_32, mask_0x3)
                    } else {
                        let qh_shift = ctx.mov_u32_imm(2 * group_val);
                        let qh_shifted = ctx.shr_u32(qh_byte_32, qh_shift);
                        ctx.and_u32(qh_shifted, mask_0x3)
                    };

                    // Combine: quant = ql_nibble | (qh_2bits << 4) - 32
                    let qh_shifted_up = ctx.shl_u32(qh_2bits, four);
                    let combined = ctx.or_u32(ql_nibble, qh_shifted_up);
                    let combined_f32 = ctx.cvt_f32_u32(combined);
                    let quant_signed = ctx.sub_f32(combined_f32, thirty_two_f32);

                    // Dequantize: val = ds_selected * quant
                    let dequant = ctx.mul_f32(ds_selected, quant_signed);

                    // Load activation x[sb_idx * 256 + offset + l]
                    // GH-215 FIX: Bounds-check for non-256-aligned K dimensions.
                    let offset_reg = ctx.mov_u32_imm(offset);
                    let x_idx = ctx.add_u32_reg(x_lane_idx, offset_reg);
                    let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                    let x_bytes = ctx.mul_u64(x_idx_64, 4);
                    let x_addr = ctx.add_u64(x_ptr, x_bytes);
                    let in_bounds = ctx.setp_lt_u32(x_idx, k_dim);
                    let x_val = ctx.ld_global_f32_predicated(x_addr, in_bounds, 0.0);

                    ctx.fma_f32_inplace(thread_partial, x_val, dequant);
                }

                ctx.add_f32_inplace(acc, thread_partial);

                // Invariant: stride by num_warps (disjoint partition)
                ctx.add_u32_reg_inplace(sb_idx, nw_reg);
                ctx.branch("mwv_q6k_sb_loop");

                ctx.label("mwv_q6k_sb_end");

                // ================================================================
                // Phase 1: Intra-warp reduction via shuffle (warp-synchronous)
                // Postcondition: lane 0 of each warp holds that warp's partial sum
                // ================================================================
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

                // ================================================================
                // Phase 2: Cross-warp reduction via shared memory
                // Invariant: bar.sync ensures all warps have written before any reads.
                // PARITY-114: all threads reach barrier unconditionally.
                // ================================================================
                let z = ctx.mov_u32_imm(0);
                let is_l0 = ctx.setp_eq_u32(lane_id, z);
                ctx.branch_if_not(is_l0, "mwv_q6k_skip_sm");

                let f4 = ctx.mov_u32_imm(4);
                let wo = ctx.mul_u32_reg(warp_id, f4);
                let sa = ctx.cvt_u64_u32(wo);
                ctx.st_shared_f32(sa, acc);

                ctx.label("mwv_q6k_skip_sm");

                // PARITY-114: ALL threads reach this barrier (not just lane 0)
                ctx.bar_sync(0);

                // Only thread 0 reads shared memory and writes output
                let is_t0 = ctx.setp_eq_u32(thread_id, z);
                ctx.branch_if_not(is_t0, "mwv_q6k_exit");

                // Postcondition: sum all warp partial sums
                let fs = ctx.mov_f32_imm(0.0);
                for w in 0..num_warps {
                    let wo = ctx.mov_u64_imm(u64::from(w * 4));
                    let pv = ctx.ld_shared_f32(wo);
                    ctx.add_f32_inplace(fs, pv);
                }

                // Postcondition: y[block_id] = final reduced sum
                let yo = ctx.mul_wide_u32(block_id, 4);
                let ya = ctx.add_u64(y_ptr, yo);
                ctx.st_global_f32(ya, fs);

                ctx.label("mwv_q6k_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Contract: kernel builds valid PTX for Qwen2.5-1.5B dimensions
    #[test]
    fn test_mwv_q6k_builds_qwen25_1p5b() {
        // Qwen2.5-1.5B: hidden_dim=1536, intermediate_dim=8960
        let kernel = MultiWarpQ6KGemvKernel::new(1536, 1536);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".visible .entry mwv_q6k_gemv"));
        assert!(
            ptx.contains(".shared"),
            "Must use shared memory for cross-warp reduction"
        );
        assert!(
            ptx.contains("bar.sync"),
            "Must have barrier for cross-warp safety"
        );
    }

    /// Contract: kernel builds valid PTX for 7B model dimensions
    #[test]
    fn test_mwv_q6k_builds_7b() {
        let kernel = MultiWarpQ6KGemvKernel::new(4096, 4096);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".visible .entry mwv_q6k_gemv"));
    }

    /// Contract: kernel has correct parameter signature
    #[test]
    fn test_mwv_q6k_parameters() {
        let kernel = MultiWarpQ6KGemvKernel::new(256, 64);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("y_ptr"), "Must have output pointer");
        assert!(ptx.contains("w_ptr"), "Must have weight pointer");
        assert!(ptx.contains("x_ptr"), "Must have activation pointer");
        assert!(ptx.contains("k_dim"), "Must have K dimension");
        assert!(ptx.contains("n_dim"), "Must have N dimension");
    }

    /// Contract: shared memory size = num_warps * 4 bytes
    #[test]
    fn test_mwv_q6k_shared_memory_size() {
        for warps in [1, 2, 3, 4, 6, 8] {
            let kernel = MultiWarpQ6KGemvKernel::with_warps(256, 64, warps);
            let ptx_kernel = kernel.build_ptx();
            assert_eq!(
                ptx_kernel.shared_memory_bytes(),
                (warps * 4) as usize,
                "Shared memory must be {warps} warps × 4 bytes"
            );
        }
    }

    /// PARITY-114: barrier safety contract
    #[test]
    fn test_mwv_q6k_barrier_safety() {
        let kernel = MultiWarpQ6KGemvKernel::new(1536, 1536);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "MWV Q6K must be barrier-safe: {:?}",
            result.violations
        );
    }

    /// Contract: kernel name is deterministic
    #[test]
    fn test_mwv_q6k_name_deterministic() {
        let k1 = MultiWarpQ6KGemvKernel::new(1536, 1536);
        let k2 = MultiWarpQ6KGemvKernel::new(4096, 4096);
        assert_eq!(
            k1.name(),
            k2.name(),
            "Kernel name must be dimension-independent"
        );
        assert_eq!(k1.name(), "mwv_q6k_gemv");
    }

    /// Contract: different warp counts produce structurally valid PTX
    #[test]
    fn test_mwv_q6k_warp_variants() {
        for warps in [1, 2, 3, 4, 6, 8] {
            let kernel = MultiWarpQ6KGemvKernel::with_warps(1536, 1536, warps);
            let ptx = kernel.emit_ptx();
            assert!(
                ptx.contains(".visible .entry"),
                "Must produce valid PTX for {warps} warps"
            );
            assert!(
                ptx.contains("bar.sync"),
                "Must have barrier even for 1 warp (PARITY-114)"
            );
        }
    }
}
