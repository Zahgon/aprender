//! TRACE-001 TR-09: `apr-trace-v1`, the one trace record `apr serve`, `apr run`
//! and (TR-09b) `apr train` write.
//!
//! The record lived in `aprender-serve`'s `api` module, behind the `server`
//! feature, so nothing but the HTTP server could produce it and nothing that
//! does not depend on the server could read it. It lives here because this
//! crate has no stack dependencies: serving and training can both depend on it
//! and so emit one schema.
//!
//! [`check`] is what a reader holds a record to. It is the golden check's
//! oracle: a record that claims more than it measured fails it.

use serde::{Deserialize, Serialize};

/// The levels an `apr-trace-v1` record may carry.
pub const LEVELS: [&str; 3] = ["brick", "step", "layer"];

/// Provenance of trace timing data (GH-92: truth-in-reporting)
///
/// Distinguishes measured data from estimates to prevent fabricated trace output.
/// Every `TraceData` instance MUST declare its provenance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceProvenance {
    /// Real per-operation timing from BrickProfiler instrumentation
    Measured,
    /// Only the wall-clock total is real; no per-op breakdown available
    WallClockTotal,
    /// Values are statistical estimates (e.g., from sampling or heuristics)
    #[default]
    Estimated,
}

/// Trace data for debugging inference
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceData {
    /// Trace level that was requested
    pub level: String,
    /// Number of operations traced
    pub operations: usize,
    /// Total time in microseconds
    pub total_time_us: u64,
    /// Per-operation timing breakdown
    pub breakdown: Vec<TraceOperation>,
    /// Data provenance — how these values were obtained (GH-92)
    #[serde(default)]
    pub provenance: TraceProvenance,
}

/// Individual traced operation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceOperation {
    /// Operation name
    pub name: String,
    /// Time in microseconds
    pub time_us: u64,
    /// Additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

/// Why a record is not a valid `apr-trace-v1` record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation {
    /// `level` is not one of [`LEVELS`].
    UnknownLevel(String),
    /// The parts add up to more than the record's own total (C-TRACE-002).
    PartsExceedTotal { parts_us: u64, total_us: u64 },
    /// `measured` with no operation or no breakdown row: nothing was measured (C-TRACE-001).
    MeasuredWithoutOperations,
    /// `wall_clock_total` claims only the total, so it is exactly one row equal
    /// to it. More rows, or a row that differs, is a split nobody measured.
    WallClockIsNotOneTotalRow,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownLevel(l) => write!(f, "level {l:?} is not one of {LEVELS:?}"),
            Self::PartsExceedTotal { parts_us, total_us } => {
                write!(
                    f,
                    "breakdown sums to {parts_us} us, past the total {total_us} us"
                )
            }
            Self::MeasuredWithoutOperations => {
                write!(
                    f,
                    "provenance measured with no operation or no breakdown row"
                )
            }
            Self::WallClockIsNotOneTotalRow => write!(
                f,
                "provenance wall_clock_total with a breakdown other than one row equal to the total"
            ),
        }
    }
}

/// Holds `t` to `apr-trace-v1`. `Ok` means it claims no more than it measured.
pub fn check(t: &TraceData) -> Result<(), Violation> {
    if !LEVELS.contains(&t.level.as_str()) {
        return Err(Violation::UnknownLevel(t.level.clone()));
    }
    let parts_us = t
        .breakdown
        .iter()
        .fold(0u64, |acc, op| acc.saturating_add(op.time_us));
    if parts_us > t.total_time_us {
        return Err(Violation::PartsExceedTotal {
            parts_us,
            total_us: t.total_time_us,
        });
    }
    match t.provenance {
        TraceProvenance::Measured if t.operations == 0 || t.breakdown.is_empty() => {
            Err(Violation::MeasuredWithoutOperations)
        }
        TraceProvenance::WallClockTotal
            if t.breakdown.len() != 1 || t.breakdown[0].time_us != t.total_time_us =>
        {
            Err(Violation::WallClockIsNotOneTotalRow)
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(name: &str, time_us: u64) -> TraceOperation {
        TraceOperation {
            name: name.to_string(),
            time_us,
            details: None,
        }
    }

    fn measured() -> TraceData {
        TraceData {
            level: "step".to_string(),
            operations: 3,
            total_time_us: 1_000,
            breakdown: vec![op("prefill", 400), op("decode", 500), op("emit", 50)],
            provenance: TraceProvenance::Measured,
        }
    }

    fn wall_clock() -> TraceData {
        TraceData {
            level: "layer".to_string(),
            operations: 28,
            total_time_us: 1_000,
            breakdown: vec![op("total_inference", 1_000)],
            provenance: TraceProvenance::WallClockTotal,
        }
    }

    #[test]
    fn honest_records_pass() {
        assert_eq!(check(&measured()), Ok(()));
        assert_eq!(check(&wall_clock()), Ok(()));
    }

    /// FALSIFY-OBS-TRACE-004: each mutation of an honest record is rejected, so
    /// a golden check built on [`check`] can fail.
    #[test]
    fn falsify_obs_trace_004_mutated_records_are_rejected() {
        let mut t = measured();
        t.breakdown[1].time_us = 600;
        assert_eq!(
            check(&t),
            Err(Violation::PartsExceedTotal {
                parts_us: 1_050,
                total_us: 1_000
            })
        );

        let mut t = measured();
        t.breakdown.clear();
        assert_eq!(check(&t), Err(Violation::MeasuredWithoutOperations));

        let mut t = measured();
        t.operations = 0;
        assert_eq!(check(&t), Err(Violation::MeasuredWithoutOperations));

        let mut t = wall_clock();
        t.breakdown = vec![op("attention", 600), op("ffn", 400)];
        assert_eq!(check(&t), Err(Violation::WallClockIsNotOneTotalRow));

        let mut t = wall_clock();
        t.breakdown[0].time_us = 999;
        assert_eq!(check(&t), Err(Violation::WallClockIsNotOneTotalRow));

        let mut t = measured();
        t.level = "basic".to_string();
        assert_eq!(check(&t), Err(Violation::UnknownLevel("basic".to_string())));
    }

    /// The wire form is the one serve has always sent: snake_case provenance,
    /// `details` omitted when absent, a missing provenance read as `estimated`.
    #[test]
    fn wire_form_is_unchanged() {
        let json = serde_json::to_string(&wall_clock()).expect("serialize");
        assert_eq!(
            json,
            r#"{"level":"layer","operations":28,"total_time_us":1000,"breakdown":[{"name":"total_inference","time_us":1000}],"provenance":"wall_clock_total"}"#
        );
        let old: TraceData = serde_json::from_str(
            r#"{"level":"step","operations":0,"total_time_us":5,"breakdown":[]}"#,
        )
        .expect("deserialize");
        assert_eq!(old.provenance, TraceProvenance::Estimated);
    }
}
