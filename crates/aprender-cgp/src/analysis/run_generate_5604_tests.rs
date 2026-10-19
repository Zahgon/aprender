//! Coverage for `run_generate` (#5604).
//!
//! `run_generate` writes `contracts/cgp/<kernel>-<size>-v1.yaml` relative to the
//! working directory and reads `/tmp/cgp-<kernel>-<size>.json`. Each test uses a
//! kernel name unique to it and removes what it wrote.

use super::*;
use crate::metrics::catalog::{FullProfile, ThroughputMetrics, TimingMetrics};

struct Cleanup(Vec<std::path::PathBuf>);

impl Drop for Cleanup {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn out_path(kernel: &str, size: u32) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("contracts/cgp/{kernel}-{size}-v1.yaml"))
}

#[test]
fn estimate_path_scales_from_the_512_baseline() {
    let kernel = "t5604-est";
    let out = out_path(kernel, 512);
    let _c = Cleanup(vec![out.clone()]);
    run_generate(kernel, 512, 10.0).expect("generate");
    let y = std::fs::read_to_string(&out).expect("written");
    assert!(y.contains("kind: PerformanceContract"));
    assert!(y.contains("name: t5604-est-512"));
    assert!(y.contains("size: [512, 512, 512]"));
    // est_time = 23.2us; max = 23.2 * 1.10; tflops = 2*512^3/23.2us ~ 11.57 -> min = 0.9x
    assert!(y.contains("max_time_us: 25.5"), "{y}");
    assert!(y.contains("min_tflops: 10.4"), "{y}");
    assert!(y.contains("max_regression_pct: 10"), "{y}");
    assert!(
        y.contains("FALSIFY-T5604_EST-001"),
        "kernel name is upper-cased with _ for -: {y}"
    );
}

#[test]
fn saved_profile_overrides_the_estimate() {
    let kernel = "t5604-prof";
    let size = 16;
    let profile_path = std::path::PathBuf::from(format!("/tmp/cgp-{kernel}-{size}.json"));
    let out = out_path(kernel, size);
    let _c = Cleanup(vec![profile_path.clone(), out.clone()]);
    let p = FullProfile {
        timing: TimingMetrics {
            wall_clock_time_us: 100.0,
            ..Default::default()
        },
        throughput: ThroughputMetrics {
            tflops: 4.0,
            ..Default::default()
        },
        ..Default::default()
    };
    std::fs::write(&profile_path, serde_json::to_string(&p).expect("json")).expect("write profile");
    run_generate(kernel, size, 50.0).expect("generate");
    let y = std::fs::read_to_string(&out).expect("written");
    assert!(y.contains("max_time_us: 150.0"), "{y}");
    assert!(y.contains("min_tflops: 2.0"), "{y}");
    assert!(y.contains("# Tolerance: 50%"), "{y}");
}

#[test]
fn corrupt_saved_profile_is_an_error_and_writes_nothing() {
    let kernel = "t5604-bad";
    let size = 8;
    let profile_path = std::path::PathBuf::from(format!("/tmp/cgp-{kernel}-{size}.json"));
    let out = out_path(kernel, size);
    let _c = Cleanup(vec![profile_path.clone(), out.clone()]);
    std::fs::write(&profile_path, "{ not json").expect("write");
    assert!(run_generate(kernel, size, 5.0).is_err());
    assert!(
        !out.exists(),
        "no contract may be written when the profile cannot be read"
    );
}

#[test]
fn zero_tolerance_pins_bounds_to_the_baseline() {
    let kernel = "t5604-zero";
    let out = out_path(kernel, 512);
    let _c = Cleanup(vec![out.clone()]);
    run_generate(kernel, 512, 0.0).expect("generate");
    let y = std::fs::read_to_string(&out).expect("written");
    assert!(y.contains("max_time_us: 23.2"), "{y}");
}
