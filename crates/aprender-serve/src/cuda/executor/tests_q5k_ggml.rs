//! FALSIFY-QDOT-008 (#3111): the CUDA Q5_K GEMV decodes ggml's `block_q5_K`, judged against
//! gguf-py's values of a llama.cpp-quantized super-block (the FALSIFY-QDOT-007 fixture), never
//! against another aprender reader.

use super::CudaExecutor;
use serial_test::serial;

include!("../../quantize/tests/q5k_ggml_fixture.rs");

/// Run the Q5_K GEMV on `rows` copies of the golden block against `input`.
fn gemv_golden(executor: &mut CudaExecutor, rows: u32, input: &[f32]) -> Vec<f32> {
    let weights = GGML_Q5K_BLOCK.repeat(rows as usize);
    let mut out = vec![0.0f32; rows as usize];
    executor
        .q5k_gemv(&weights, input, &mut out, rows, 256)
        .expect("q5k_gemv on the golden block");
    out
}

#[test]
#[serial]
fn test_q5k_gemv_one_hot_returns_each_gguf_py_value() {
    let mut executor = crate::cuda_executor_or_skip!(0);
    for (i, &want) in GGML_Q5K_EXPECTED.iter().enumerate() {
        let mut one_hot = [0.0f32; 256];
        one_hot[i] = 1.0;
        for got in gemv_golden(&mut executor, 2, &one_hot) {
            assert!(
                (got - want).abs() <= 1e-6 + 1e-5 * want.abs(),
                "value {i}: cuda {got}, gguf-py {want}"
            );
        }
    }
}

#[test]
#[serial]
fn test_q5k_gemv_ramp_matches_gguf_py_dot() {
    let mut executor = crate::cuda_executor_or_skip!(0);
    let input: Vec<f32> = (0..256u16).map(|i| (f32::from(i) - 127.5) / 64.0).collect();
    let want: f64 = GGML_Q5K_EXPECTED
        .iter()
        .zip(&input)
        .map(|(w, a)| f64::from(*w) * f64::from(*a))
        .sum();
    for got in gemv_golden(&mut executor, 4, &input) {
        assert!(
            (f64::from(got) - want).abs() <= 1e-4 * want.abs().max(1.0),
            "cuda {got}, gguf-py {want}"
        );
    }
}

/// Run `q5k_gemv_into` (the decode path, multi-warp for 256-aligned K since #5221) on `rows`
/// rows of `sbs` copies of the golden block each, so K = 256 × `sbs`.
fn gemv_into_golden(executor: &mut CudaExecutor, rows: u32, sbs: u32, input: &[f32]) -> Vec<f32> {
    use trueno_gpu::driver::GpuBuffer;
    let weights = GGML_Q5K_BLOCK.repeat((rows * sbs) as usize);
    let w = GpuBuffer::from_host(&executor.context, &weights).expect("weights");
    let x = GpuBuffer::from_host(&executor.context, input).expect("input");
    let y = GpuBuffer::<f32>::new(&executor.context, rows as usize).expect("output");
    executor
        .q5k_gemv_into(w.as_ptr(), &x, &y, rows, 256 * sbs)
        .expect("q5k_gemv_into on the golden block");
    executor.stream.synchronize().expect("sync");
    let mut out = vec![0.0f32; rows as usize];
    y.copy_to_host(&mut out).expect("copy back");
    out
}

/// #5221: the multi-warp decode kernel returns every gguf-py value, one-hot per super-block
/// position, with K spanning more super-blocks than warps so the warp stride is exercised.
#[test]
#[serial]
fn test_q5k_gemv_into_mwv_one_hot_returns_each_gguf_py_value() {
    let mut executor = crate::cuda_executor_or_skip!(0);
    let sbs = 11u32; // > any num_warps in {1,2,3,4,6,8}, and not a multiple of them
    for (i, &want) in GGML_Q5K_EXPECTED.iter().enumerate() {
        for sb in [0u32, 5, 10] {
            let mut one_hot = vec![0.0f32; (256 * sbs) as usize];
            one_hot[(sb * 256) as usize + i] = 1.0;
            for got in gemv_into_golden(&mut executor, 3, sbs, &one_hot) {
                assert!(
                    (got - want).abs() <= 1e-6 + 1e-5 * want.abs(),
                    "value {i} of super-block {sb}: cuda {got}, gguf-py {want}"
                );
            }
        }
    }
}

/// #5221: the multi-warp decode kernel's dot product over many super-blocks matches gguf-py.
#[test]
#[serial]
fn test_q5k_gemv_into_mwv_ramp_matches_gguf_py_dot() {
    let mut executor = crate::cuda_executor_or_skip!(0);
    let sbs = 10u32; // Qwen3.5-4B hidden = 2560 = 10 super-blocks
    let k = (256 * sbs) as usize;
    let input: Vec<f32> = (0..k).map(|i| ((i % 509) as f32 - 254.0) / 128.0).collect();
    let want: f64 = (0..k)
        .map(|i| f64::from(GGML_Q5K_EXPECTED[i % 256]) * f64::from(input[i]))
        .sum();
    for got in gemv_into_golden(&mut executor, 5, sbs, &input) {
        assert!(
            (f64::from(got) - want).abs() <= 1e-4 * want.abs().max(1.0),
            "cuda {got}, gguf-py {want}"
        );
    }
}
