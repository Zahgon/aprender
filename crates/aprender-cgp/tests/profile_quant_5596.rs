//! #5596: behavioural tests for `profile_quant` via `aprender-cgp profile quant`.
//! Each run uses a scratch cwd (no Cargo.toml, so `cargo metadata` fails) holding an
//! optional fake `./target/release/examples/benchmark_matrix_suite`.

#![allow(clippy::disallowed_methods)]

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Command, Output};

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cgp-5596-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn install_fake_bench(dir: &PathBuf, line: &str) {
    let bin_dir = dir.join("target/release/examples");
    std::fs::create_dir_all(&bin_dir).expect("create examples dir");
    let bin = bin_dir.join("benchmark_matrix_suite");
    std::fs::write(&bin, format!("#!/bin/sh\necho '{line}'\n")).expect("write fake bench");
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).expect("chmod");
}

fn run(dir: &PathBuf, kernel: &str, size: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_aprender-cgp"))
        .current_dir(dir)
        .args(["profile", "quant", "--kernel", kernel, "--size", size])
        .output()
        .expect("spawn aprender-cgp")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

#[test]
fn unknown_kernel_is_rejected() {
    let dir = scratch("badkernel");
    let out = run(&dir, "q9_gemv", "4096x1x4096");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("Unknown quant kernel: q9_gemv"));
}

#[test]
fn bad_dimensions_are_rejected() {
    let dir = scratch("baddims");
    let out = run(&dir, "q4k", "4096x4096");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("Dimensions must be MxNxK format"));
}

#[test]
fn without_bench_binary_prints_static_metrics_and_no_timing_hint() {
    let dir = scratch("nobench");
    let out = run(&dir, "q4k_gemv", "4096x1x4096");
    assert!(out.status.success());
    let s = stdout(&out);
    assert!(
        s.contains("=== CGP Quant Profile: q4k_gemv (4096x1x4096) ==="),
        "{s}"
    );
    assert!(s.contains("Dimensions: M=4096, N=1, K=4096"), "{s}");
    assert!(s.contains("Super-block: 256 elements, 144 bytes"), "{s}");
    assert!(s.contains("Total weights: 16777216"), "{s}");
    assert!(s.contains("Super-blocks: 65536"), "{s}");
    assert!(s.contains("Compressed size: 9.44 MB"), "{s}");
    assert!(s.contains("FP32 equivalent: 67.11 MB"), "{s}");
    assert!(s.contains("Compression ratio: 7.1x"), "{s}");
    assert!(s.contains("No timing data"), "{s}");
    assert!(!s.contains("Measured (from benchmark_matrix_suite)"), "{s}");
}

#[test]
fn nf4_uses_its_own_superblock_geometry() {
    let dir = scratch("nf4");
    let s = stdout(&run(&dir, "nf4", "64x1x64"));
    assert!(s.contains("Super-block: 64 elements, 32 bytes"), "{s}");
    assert!(s.contains("Super-blocks: 64"), "{s}");
}

#[test]
fn measured_line_with_low_bandwidth_share_is_memory_bound() {
    let dir = scratch("membound");
    install_fake_bench(
        &dir,
        "Q4K GEMV 4096x4096 ... 100.0 us  (50.0 GFLOPS, 10.0 GB/s)",
    );
    let s = stdout(&run(&dir, "q4k_gemv", "4096x1x4096"));
    assert!(s.contains("Measured (from benchmark_matrix_suite)"), "{s}");
    assert!(s.contains("Time: 100.0 us"), "{s}");
    assert!(s.contains("GFLOPS: 50.0"), "{s}");
    assert!(s.contains("Compute util: 33% of AVX-512 peak"), "{s}");
    assert!(s.contains("Bandwidth util: 25% of practical DRAM"), "{s}");
    assert!(s.contains("Bottleneck: MEMORY-BOUND"), "{s}");
    assert!(s.contains("Est. 192 GEMVs/token: 19.2 ms"), "{s}");
    assert!(s.contains("Est. tokens/sec: 52.1"), "{s}");
}

#[test]
fn measured_line_with_high_bandwidth_share_is_compute_bound() {
    let dir = scratch("compbound");
    install_fake_bench(
        &dir,
        "Q4K GEMV 4096x4096 ... 100.0 us  (15.0 GFLOPS, 20.0 GB/s)",
    );
    let s = stdout(&run(&dir, "q4k_gemv", "4096x1x4096"));
    assert!(s.contains("Compute util: 10% of AVX-512 peak"), "{s}");
    assert!(s.contains("Bandwidth util: 50% of practical DRAM"), "{s}");
    assert!(s.contains("Bottleneck: COMPUTE-BOUND"), "{s}");
}

#[test]
fn bench_line_for_other_dimensions_gives_no_timing() {
    let dir = scratch("otherdims");
    install_fake_bench(
        &dir,
        "Q4K GEMV 1536x8960 ... 100.0 us  (50.0 GFLOPS, 10.0 GB/s)",
    );
    let s = stdout(&run(&dir, "q4k_gemv", "4096x1x4096"));
    assert!(s.contains("No timing data"), "{s}");
}
