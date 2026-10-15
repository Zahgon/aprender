//! The ranked gap list of one run (contracts/apr-code-eval-gap-v1.yaml, ACE-GAP-001..005).
//!
//! Each nightly posts its receipt and one list of failing tasks, worst first
//! (docs/specifications/apr-code-eval-v1.md §10, ticket #5288). The list is read off the
//! receipt alone:
//!
//! - A sample counts by its final attempt only; a retry that fixes a task removes it
//!   (ACE-GAP-001).
//! - Every task with a non-resolved final attempt is listed once, with its failure count
//!   and labels, and the counts reconcile with the samples (ACE-GAP-002).
//! - Worst first: more failing samples ranks higher, ties by task id, so row order in the
//!   receipt never changes the list (ACE-GAP-003).
//! - A receipt with no rows, another schema, a duplicated attempt or label counts that
//!   disagree with its rows is refused, never ranked (ACE-GAP-004).
//! - The rendered comment carries no timing field, so two `same_run` receipts render
//!   byte-identical text (ACE-GAP-005).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::receipt::{Backend, Receipt, Row, SCHEMA};
use crate::verdict::Verdict;

/// One failing task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gap {
    /// Task id.
    pub id: String,
    /// Samples of the task in the run.
    pub samples: usize,
    /// Samples whose final attempt is not `Resolved`; at least 1.
    pub failures: usize,
    /// Final-attempt labels of the failing samples; sums to `failures`.
    pub labels: BTreeMap<Verdict, usize>,
}

/// The receipt cannot be ranked.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GapError {
    /// The receipt has no rows: an empty list would read as "nothing failed".
    #[error("receipt has no rows; nothing was measured")]
    Empty,
    /// The receipt is not an `apr-code-eval-receipt/v1`.
    #[error("receipt schema is {0:?}, expected {SCHEMA:?}")]
    Schema(String),
    /// Two rows claim the same task, sample and attempt, or an attempt is 0.
    #[error("task {id} sample {sample}: attempt {attempt} is duplicated or out of range")]
    Attempt {
        /// Task id.
        id: String,
        /// Sample index.
        sample: usize,
        /// The offending attempt number.
        attempt: u8,
    },
    /// `counts_by_label` does not match a recount of the rows.
    #[error("counts_by_label disagrees with the rows for {0:?}")]
    Counts(Verdict),
}

/// ACE-GAP-001..003: the failing tasks of these rows, worst first.
pub fn rank(rows: &[Row]) -> Result<Vec<Gap>, GapError> {
    if rows.is_empty() {
        return Err(GapError::Empty);
    }
    // (task, sample) → (final attempt, its verdict).
    let mut finals: BTreeMap<(&str, usize), (u8, Verdict)> = BTreeMap::new();
    let mut seen = std::collections::BTreeSet::new();
    for r in rows {
        if r.attempt == 0 || !seen.insert((r.id.as_str(), r.sample, r.attempt)) {
            return Err(GapError::Attempt {
                id: r.id.clone(),
                sample: r.sample,
                attempt: r.attempt,
            });
        }
        let slot = finals
            .entry((r.id.as_str(), r.sample))
            .or_insert((r.attempt, r.verdict));
        if r.attempt > slot.0 {
            *slot = (r.attempt, r.verdict);
        }
    }
    let mut by_task: BTreeMap<&str, Gap> = BTreeMap::new();
    for (&(id, _), &(_, verdict)) in &finals {
        let gap = by_task.entry(id).or_insert_with(|| Gap {
            id: id.to_string(),
            samples: 0,
            failures: 0,
            labels: BTreeMap::new(),
        });
        gap.samples += 1;
        if verdict != Verdict::Resolved {
            gap.failures += 1;
            *gap.labels.entry(verdict).or_insert(0) += 1;
        }
    }
    let mut gaps: Vec<Gap> = by_task.into_values().filter(|g| g.failures > 0).collect();
    gaps.sort_by(|a, b| b.failures.cmp(&a.failures).then_with(|| a.id.cmp(&b.id)));
    Ok(gaps)
}

/// ACE-GAP-004: check the receipt, then rank its rows.
pub fn of_receipt(receipt: &Receipt) -> Result<Vec<Gap>, GapError> {
    if receipt.schema != SCHEMA {
        return Err(GapError::Schema(receipt.schema.clone()));
    }
    for v in Verdict::ALL {
        let recount = receipt.per_task.iter().filter(|r| r.verdict == v).count();
        if receipt.counts_by_label.get(&v).copied().unwrap_or(0) != recount {
            return Err(GapError::Counts(v));
        }
    }
    rank(&receipt.per_task)
}

/// The receipt's name for a label, e.g. `FORMAT_ERROR`.
fn label(v: Verdict) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|j| j.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{v:?}"))
}

/// ACE-GAP-005: the gap comment for one receipt, in markdown, with no timing field.
pub fn render(receipt: &Receipt, gaps: &[Gap]) -> String {
    let run = &receipt.run;
    let m = &receipt.metrics.attempt_2;
    let backend = match run.model.backend {
        Backend::Cpu => "cpu",
        Backend::Cuda => "cuda",
    };
    let sha = |s: &str| s.chars().take(12).collect::<String>();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "apr code eval `{}` on {backend}: pass@1 {:.4} (Wilson 95% {:.4}–{:.4}) over {} tasks × n={}",
        run.suite, m.pass_at_1, m.wilson95.0, m.wilson95.1, m.tasks, m.n
    );
    let _ = writeln!(
        out,
        "model sha256 `{}`, apr {} (`{}`), task set `{}`, harness `{}`, seed {}",
        sha(&run.model.sha256),
        run.apr.version,
        sha(&run.apr.git_sha),
        sha(&run.task_set_digest),
        sha(&run.harness_git_sha),
        run.seed
    );
    out.push('\n');
    if gaps.is_empty() {
        out.push_str("No failing tasks: every sample's final attempt is RESOLVED.\n");
        return out;
    }
    let _ = writeln!(out, "{} failing tasks, worst first:\n", gaps.len());
    out.push_str("| rank | task | failing / samples | final-attempt labels |\n");
    out.push_str("|---:|---|---:|---|\n");
    for (i, g) in gaps.iter().enumerate() {
        let labels = g
            .labels
            .iter()
            .map(|(v, c)| format!("{} {c}", label(*v)))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            out,
            "| {} | `{}` | {} / {} | {labels} |",
            i + 1,
            g.id,
            g.failures,
            g.samples
        );
    }
    out
}

#[cfg(test)]
#[path = "gap_tests.rs"]
mod tests;
