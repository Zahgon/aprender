//! #5664: `MudaDetector::detect_waiting` branch coverage with exact outputs.

#![allow(clippy::disallowed_methods)]

use cgp::analysis::muda::{GpuMuda, MudaDetector, MudaThresholds};

const MEM_HINT: &str = "Increase warps per SM for latency hiding, or improve data locality";
const BAR_HINT: &str = "Reduce barrier synchronization or overlap compute with data movement";

#[test]
fn clean_profile_is_not_waiting() {
    assert!(MudaDetector::new().detect_waiting(0, 0, 5, 20.0).is_none());
}

#[test]
fn scheduler_idle_just_above_threshold_triggers_with_that_impact() {
    let d = MudaDetector::new()
        .detect_waiting(0, 0, 7, 20.5)
        .expect("20.5% > 20% threshold");
    assert_eq!(d.impact_pct, 20.5);
    assert_eq!(
        d.recommendation, BAR_HINT,
        "0 mem == 0 barrier is not mem-dominant"
    );
    assert_eq!(
        d.description,
        "Hardware idle: scheduler 20.5% idle, 0 memory stall cycles, 0 barrier stall cycles"
    );
    match d.muda {
        GpuMuda::Waiting {
            barrier_stall_cycles,
            memory_stall_cycles,
            pipeline_bubbles,
            warp_scheduler_idle_pct,
        } => assert_eq!(
            (
                barrier_stall_cycles,
                memory_stall_cycles,
                pipeline_bubbles,
                warp_scheduler_idle_pct
            ),
            (0, 0, 7, 20.5)
        ),
        other => panic!("expected Waiting, got {other:?}"),
    }
}

#[test]
fn memory_stalls_alone_trigger_with_ten_percent_floor() {
    let d = MudaDetector::new()
        .detect_waiting(0, 100, 0, 3.0)
        .expect("memory stalls trigger");
    assert_eq!(d.impact_pct, 10.0, "floor of 10 beats a 3% idle");
    assert_eq!(d.recommendation, MEM_HINT);
}

#[test]
fn idle_above_floor_wins_the_impact_max() {
    let d = MudaDetector::new()
        .detect_waiting(0, 100, 0, 35.0)
        .expect("triggers");
    assert_eq!(d.impact_pct, 35.0);
}

#[test]
fn barrier_stalls_alone_trigger_without_the_memory_floor() {
    let d = MudaDetector::new()
        .detect_waiting(50, 0, 0, 4.0)
        .expect("barrier stalls trigger");
    assert_eq!(d.impact_pct, 4.0, "no memory stalls => no 10% floor");
    assert_eq!(d.recommendation, BAR_HINT);
    assert!(d.description.contains("50 barrier stall cycles"));
}

#[test]
fn recommendation_tracks_which_stall_dominates() {
    let det = MudaDetector::new();
    assert_eq!(
        det.detect_waiting(10, 11, 0, 0.0).unwrap().recommendation,
        MEM_HINT
    );
    assert_eq!(
        det.detect_waiting(11, 10, 0, 0.0).unwrap().recommendation,
        BAR_HINT
    );
    assert_eq!(
        det.detect_waiting(10, 10, 0, 0.0).unwrap().recommendation,
        BAR_HINT
    );
}

#[test]
fn custom_idle_threshold_is_honoured() {
    let strict = MudaDetector::with_thresholds(MudaThresholds {
        max_scheduler_idle_pct: 5.0,
        ..MudaThresholds::default()
    });
    assert!(strict.detect_waiting(0, 0, 0, 6.0).is_some());
    assert!(MudaDetector::new().detect_waiting(0, 0, 0, 6.0).is_none());
}
