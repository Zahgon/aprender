//! TRACE-001 TR-09: the one `apr-trace-v1` renderer.
//!
//! `apr serve` answers `X-Trace-Level` from here, and `apr run` writes the same
//! record into its trace file from the same facts, so one request traced both
//! ways yields one record. It lived in `api`, behind the `server` feature, where
//! the `apr run` path could not reach it. The record type is
//! [`renacer_core::apr_trace`], which training can depend on too (TR-09b).

pub use renacer_core::apr_trace::{check, TraceData, TraceOperation, TraceProvenance, Violation};

/// Build trace data based on X-Trace-Level header
///
/// Returns (brick_trace, step_trace, layer_trace) tuple based on requested level.
/// Only reports wall-clock totals — per-operation breakdown requires `apr profile`
/// with BrickProfiler instrumentation. We refuse to fabricate per-op estimates.
#[must_use]
pub fn build_trace_data(
    trace_level: Option<&str>,
    latency_us: u64,
    prompt_tokens: usize,
    completion_tokens: usize,
    num_layers: usize,
) -> (Option<TraceData>, Option<TraceData>, Option<TraceData>) {
    match trace_level {
        Some("brick") => (
            Some(TraceData {
                level: "brick".to_string(),
                operations: completion_tokens,
                total_time_us: latency_us,
                breakdown: vec![
                    TraceOperation {
                        name: "total_inference".to_string(),
                        time_us: latency_us,
                        details: Some(format!(
                            "{} prompt + {} completion tokens, {} layers. \
                             Per-op breakdown not available — use `apr profile` for real brick-level telemetry",
                            prompt_tokens, completion_tokens, num_layers
                        )),
                    },
                ],
                provenance: TraceProvenance::WallClockTotal,
            }),
            None,
            None,
        ),
        Some("step") => (
            None,
            Some(TraceData {
                level: "step".to_string(),
                operations: completion_tokens,
                total_time_us: latency_us,
                breakdown: vec![
                    TraceOperation {
                        name: "total_inference".to_string(),
                        time_us: latency_us,
                        details: Some(format!(
                            "{} prompt + {} completion tokens, {} layers. \
                             Step-level breakdown not instrumented — use `apr profile` for real timing",
                            prompt_tokens, completion_tokens, num_layers
                        )),
                    },
                ],
                provenance: TraceProvenance::WallClockTotal,
            }),
            None,
        ),
        Some("layer") => (
            None,
            None,
            Some(TraceData {
                level: "layer".to_string(),
                operations: num_layers,
                total_time_us: latency_us,
                breakdown: vec![
                    TraceOperation {
                        name: "total_inference".to_string(),
                        time_us: latency_us,
                        details: Some(format!(
                            "{} layers, {} tokens. \
                             Per-layer breakdown not instrumented — use `apr profile --granular` for real per-layer timing",
                            num_layers, prompt_tokens + completion_tokens
                        )),
                    },
                ],
                provenance: TraceProvenance::WallClockTotal,
            }),
        ),
        _ => (None, None, None),
    }
}
