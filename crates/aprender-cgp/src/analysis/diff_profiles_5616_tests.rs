//! Coverage for `diff_profiles` (#5616): GPU sections, skipped metrics, verdict edges.

use super::*;
use crate::metrics::catalog::{
    FullProfile, GpuComputeMetrics, GpuMemoryMetrics, ThroughputMetrics, TimingMetrics,
};

fn profile(time_us: f64, tflops: f64, bw: f64) -> FullProfile {
    FullProfile {
        timing: TimingMetrics {
            wall_clock_time_us: time_us,
            ..Default::default()
        },
        throughput: ThroughputMetrics {
            tflops,
            bandwidth_gbps: bw,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn names(d: &[MetricDiff]) -> Vec<&str> {
    d.iter().map(|m| m.name.as_str()).collect()
}

fn get<'a>(d: &'a [MetricDiff], n: &str) -> &'a MetricDiff {
    d.iter()
        .find(|m| m.name == n)
        .unwrap_or_else(|| panic!("no metric {n}"))
}

#[test]
fn all_zero_profiles_produce_no_diffs() {
    assert!(diff_profiles(&profile(0.0, 0.0, 0.0), &profile(0.0, 0.0, 0.0)).is_empty());
}

#[test]
fn without_gpu_sections_only_cpu_metrics_are_diffed() {
    let d = diff_profiles(&profile(10.0, 1.0, 2.0), &profile(10.0, 1.0, 2.0));
    assert_eq!(
        names(&d),
        ["wall_clock_time_us", "tflops", "bandwidth_gbps"]
    );
    assert!(d.iter().all(|m| m.verdict == "=" && m.change_pct == 0.0));
}

#[test]
fn gpu_sections_need_both_sides() {
    let mut base = profile(10.0, 1.0, 2.0);
    base.gpu_compute = Some(GpuComputeMetrics {
        sm_utilization_pct: 50.0,
        ..Default::default()
    });
    base.gpu_memory = Some(GpuMemoryMetrics {
        l2_hit_rate_pct: 50.0,
        ..Default::default()
    });
    let d = diff_profiles(&base, &profile(10.0, 1.0, 2.0));
    assert_eq!(
        d.len(),
        3,
        "one-sided GPU sections must be skipped: {:?}",
        names(&d)
    );
}

#[test]
fn gpu_compute_and_memory_metrics_are_compared_with_higher_is_better() {
    let mut base = profile(10.0, 1.0, 2.0);
    let mut cur = profile(10.0, 1.0, 2.0);
    base.gpu_compute = Some(GpuComputeMetrics {
        sm_utilization_pct: 50.0,
        achieved_occupancy_pct: 80.0,
        warp_execution_efficiency_pct: 90.0,
        ..Default::default()
    });
    cur.gpu_compute = Some(GpuComputeMetrics {
        sm_utilization_pct: 75.0,
        achieved_occupancy_pct: 40.0,
        warp_execution_efficiency_pct: 90.5,
        ..Default::default()
    });
    base.gpu_memory = Some(GpuMemoryMetrics {
        l2_hit_rate_pct: 60.0,
        global_load_efficiency_pct: 100.0,
        ..Default::default()
    });
    cur.gpu_memory = Some(GpuMemoryMetrics {
        l2_hit_rate_pct: 30.0,
        global_load_efficiency_pct: 150.0,
        ..Default::default()
    });
    let d = diff_profiles(&base, &cur);
    assert_eq!(d.len(), 8);
    assert_eq!(get(&d, "sm_utilization_pct").verdict, "IMPROVED");
    assert_eq!(get(&d, "sm_utilization_pct").change_pct, 50.0);
    assert_eq!(get(&d, "achieved_occupancy_pct").verdict, "REGRESSED");
    assert_eq!(get(&d, "warp_exec_efficiency_pct").verdict, "=");
    assert_eq!(get(&d, "l2_hit_rate_pct").verdict, "REGRESSED");
    assert_eq!(get(&d, "l2_hit_rate_pct").change_pct, -50.0);
    assert_eq!(get(&d, "global_load_efficiency_pct").verdict, "IMPROVED");
}

#[test]
fn bandwidth_and_tflops_regress_when_they_fall() {
    let d = diff_profiles(&profile(10.0, 10.0, 100.0), &profile(10.0, 5.0, 50.0));
    assert_eq!(get(&d, "tflops").verdict, "REGRESSED");
    assert_eq!(get(&d, "bandwidth_gbps").verdict, "REGRESSED");
    assert_eq!(get(&d, "bandwidth_gbps").change_pct, -50.0);
}

#[test]
fn zero_baseline_reports_zero_change_and_equal_verdict() {
    let d = diff_profiles(&profile(0.0, 0.0, 0.0), &profile(5.0, 0.0, 0.0));
    let t = get(&d, "wall_clock_time_us");
    assert_eq!(
        (t.baseline, t.current, t.change_pct, t.verdict),
        (0.0, 5.0, 0.0, "=")
    );
    assert_eq!(d.len(), 1, "metrics zero on both sides are dropped");
}

#[test]
fn two_percent_boundary_is_a_real_change() {
    // |change| < 2.0 is "="; exactly 2.0 is not.
    let d = diff_profiles(&profile(100.0, 1.0, 1.0), &profile(102.0, 1.0, 1.0));
    assert_eq!(get(&d, "wall_clock_time_us").verdict, "REGRESSED");
    let d = diff_profiles(&profile(100.0, 1.0, 1.0), &profile(101.9, 1.0, 1.0));
    assert_eq!(get(&d, "wall_clock_time_us").verdict, "=");
}
