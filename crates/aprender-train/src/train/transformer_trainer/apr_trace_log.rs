//! TRACE-001 TR-09b: `apr_trace.jsonl`, one `apr-trace-v1` record per optimizer
//! step, for a train loop to write beside the run's other logs in `output_dir`
//! (the CPU loop's call is TR-09c). Line N is the Nth
//! completed step; a reader holds each line to
//! `renacer_core::apr_trace::check_spans(.., &STEP_SPANS)`.

use renacer_core::apr_trace::TraceData;
use std::io::Write;
use std::path::Path;

/// The file name, under the run's `output_dir`.
pub const FILE: &str = "apr_trace.jsonl";

/// Writes a step's record once per optimizer step.
pub struct AprTraceLog {
    file: Option<std::fs::File>,
    last_step: usize,
}

impl AprTraceLog {
    /// Opens `output_dir/apr_trace.jsonl` fresh, so line N is this run's step N; a run that cannot open it
    /// says so and trains on without it, like the other run logs.
    pub fn open(output_dir: &Path) -> Self {
        let path = output_dir.join(FILE);
        let file = std::fs::create_dir_all(output_dir).and_then(|()| {
            std::fs::OpenOptions::new().create(true).write(true).truncate(true).open(&path)
        });
        let file = match file {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!("Warning: cannot open {}: {e}; no apr-trace-v1 records", path.display());
                None
            }
        };
        Self { file, last_step: 0 }
    }

    /// Writes `trace` when `step` moved past the last step written: a batch that
    /// only accumulated gradients closes no step and writes nothing.
    pub fn record(&mut self, step: usize, trace: Option<&TraceData>) {
        if step == self.last_step {
            return;
        }
        self.last_step = step;
        let (Some(f), Some(t)) = (self.file.as_mut(), trace) else { return };
        if let Ok(line) = serde_json::to_string(t) {
            let _ = writeln!(f, "{line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::step_trace::{step_record, STEP_SPANS};
    use super::*;
    use renacer_core::apr_trace::check_spans;

    #[test]
    fn one_line_per_optimizer_step_and_each_passes_check_spans() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join(FILE), "stale line from an earlier run\n").expect("seed");
        let t = step_record(100, 1, 40, 50, 9);
        let mut log = AprTraceLog::open(dir.path());
        log.record(0, None); // accumulated only: step did not move
        log.record(1, Some(&t));
        log.record(1, Some(&t)); // same step again: no second line
        log.record(2, Some(&t));
        drop(log);
        let text = std::fs::read_to_string(dir.path().join(FILE)).expect("read");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "{text}");
        for l in lines {
            let r: TraceData = serde_json::from_str(l).expect("apr-trace-v1 line");
            assert_eq!(check_spans(&r, &STEP_SPANS), Ok(()));
        }
    }
}
