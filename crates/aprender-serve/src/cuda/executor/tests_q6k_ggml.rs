//! #5221: the CUDA multi-warp Q6_K GEMV (`mwv_q6k_gemv`, the decode path's Q6_K kernel) decodes
//! ggml's `block_q6_K`, judged against gguf-py's values of a llama.cpp-quantized super-block,
//! never against another aprender reader.

use super::CudaExecutor;
use serial_test::serial;

include!("../../quantize/tests/q6k_ggml_fixture.rs");

/// Run `mwv_q6k_gemv_into` with `num_warps` warps on `rows` rows of `sbs` copies of the golden
/// block each, so K = 256 × `sbs`.
fn mwv_golden(
    executor: &mut CudaExecutor,
    num_warps: u32,
    rows: u32,
    sbs: u32,
    input: &[f32],
) -> Vec<f32> {
    use trueno_gpu::driver::GpuBuffer;
    executor.gpu_profile.mwv_warps = num_warps;
    let weights = GGML_Q6K_BLOCK.repeat((rows * sbs) as usize);
    let w = GpuBuffer::from_host(&executor.context, &weights).expect("weights");
    let x = GpuBuffer::from_host(&executor.context, input).expect("input");
    let y = GpuBuffer::<f32>::new(&executor.context, rows as usize).expect("output");
    executor
        .mwv_q6k_gemv_into(w.as_ptr(), &x, &y, rows, 256 * sbs)
        .expect("mwv_q6k_gemv_into on the golden block");
    executor.stream.synchronize().expect("sync");
    let mut out = vec![0.0f32; rows as usize];
    y.copy_to_host(&mut out).expect("copy back");
    out
}

/// Every gguf-py value comes back one-hot, at super-blocks spread over a K wider than any warp
/// count, so the warp stride and every (ql, qh, scale) position are exercised.
#[test]
#[serial]
fn test_mwv_q6k_one_hot_returns_each_gguf_py_value() {
    let mut executor = crate::cuda_executor_or_skip!(0);
    let sbs = 11u32; // > any num_warps in {1,2,3,4,6,8}, and not a multiple of them
    for (i, &want) in GGML_Q6K_EXPECTED.iter().enumerate() {
        for sb in [0u32, 5, 10] {
            let mut one_hot = vec![0.0f32; (256 * sbs) as usize];
            one_hot[(sb * 256) as usize + i] = 1.0;
            for got in mwv_golden(&mut executor, 3, 3, sbs, &one_hot) {
                assert!(
                    (got - want).abs() <= 1e-6 + 1e-5 * want.abs(),
                    "value {i} of super-block {sb}: cuda {got}, gguf-py {want}"
                );
            }
        }
    }
}

/// The dot product over many super-blocks matches gguf-py for every supported warp count.
#[test]
#[serial]
fn test_mwv_q6k_ramp_matches_gguf_py_dot_every_warp_count() {
    let mut executor = crate::cuda_executor_or_skip!(0);
    let sbs = 10u32; // Qwen3.5-4B hidden = 2560 = 10 super-blocks
    let k = (256 * sbs) as usize;
    let input: Vec<f32> = (0..k).map(|i| ((i % 509) as f32 - 254.0) / 128.0).collect();
    let want: f64 = (0..k)
        .map(|i| f64::from(GGML_Q6K_EXPECTED[i % 256]) * f64::from(input[i]))
        .sum();
    for warps in [1u32, 2, 3, 4, 6, 8] {
        for got in mwv_golden(&mut executor, warps, 5, sbs, &input) {
            assert!(
                (f64::from(got) - want).abs() <= 1e-4 * want.abs().max(1.0),
                "{warps} warps: cuda {got}, gguf-py {want}"
            );
        }
    }
}
