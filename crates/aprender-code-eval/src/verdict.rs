//! Per-task verdict (contracts/apr-code-eval-verdict-v1.yaml, ACE-VER-001..007).
//!
//! One sandbox report and one model output become exactly one closed label. The first
//! match of ACE-VER-006 wins:
//!
//! 1. the output has no edit → `Empty`
//! 2. the edit does not parse or does not apply → `FormatError`
//! 3. the run was killed by its time limit → `Timeout`
//! 4. any other non-ok sandbox status → `SandboxError`
//! 5. zero tests ran, or some FAIL_TO_PASS test did not run → `NoTests`
//! 6. every FAIL_TO_PASS and PASS_TO_PASS test ran and passed → `Resolved`
//! 7. otherwise → `Failed`
//!
//! Only `Resolved` counts toward a pass, and every label stays in the denominator.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The closed label set V of ACE-VER-006.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Verdict {
    /// Every F2P and P2P test executed and passed (the SWE-bench "resolved" rule).
    Resolved,
    /// Tests ran and at least one F2P or P2P test did not pass.
    Failed,
    /// The model's edit did not parse or did not apply. Never silently repaired.
    FormatError,
    /// The model's output carried no edit at all.
    Empty,
    /// The sandbox killed the run at its time limit.
    Timeout,
    /// Any other sandbox failure (OOM, image error, missing report, …).
    SandboxError,
    /// The run executed zero tests, or not every F2P test.
    NoTests,
}

impl Verdict {
    /// Every label, in a fixed order (used to reconcile counts, ACE-MET-008).
    pub const ALL: [Verdict; 7] = [
        Verdict::Resolved,
        Verdict::Failed,
        Verdict::FormatError,
        Verdict::Empty,
        Verdict::Timeout,
        Verdict::SandboxError,
        Verdict::NoTests,
    ];
}

/// What the edit adapter (#5287) found in the model's output: the pre(t, out) parts of
/// the verdict contract, already decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditOutcome {
    /// edits(out) = ∅: blank output, or output that parsed to zero edit blocks.
    Empty,
    /// The output is non-empty and does not parse under the task's edit format.
    Unparsable,
    /// The edits parsed but did not apply to the task snapshot.
    NotApplied,
    /// pre(t, out) holds: the edits parsed and applied.
    Applied,
}

/// Classify a model output. `parsed_edits` is the parser's result: `None` when the output
/// does not parse, `Some(n)` when it parsed to `n` edit blocks. `applied` is whether those
/// blocks applied to the snapshot.
///
/// EMPTY is decided from the raw text first, never from the parser alone: a parser that
/// fails on non-empty text must give `Unparsable` (FORMAT_ERROR), not `Empty`, or the
/// ¬parses arm of ACE-VER-005 could never fire.
#[must_use]
pub fn classify_edit(raw: &str, parsed_edits: Option<usize>, applied: bool) -> EditOutcome {
    if raw.trim().is_empty() {
        return EditOutcome::Empty;
    }
    match parsed_edits {
        None => EditOutcome::Unparsable,
        Some(0) => EditOutcome::Empty,
        Some(_) if applied => EditOutcome::Applied,
        Some(_) => EditOutcome::NotApplied,
    }
}

/// Sandbox exit status (ACE-SBX-003: `Ok` implies nothing killed the run).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxStatus {
    /// The test command finished and wrote its report.
    Ok,
    /// Anything else, with the sandbox's reason.
    Error(String),
}

/// Why the sandbox killed a run, when it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KilledBy {
    /// The wall-clock limit.
    Timeout,
    /// The memory limit.
    Oom,
    /// Any other limit (pids, disk, …).
    Other,
}

/// Outcome of one executed test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestOutcome {
    /// The test ran and passed.
    Passed,
    /// The test ran and failed.
    Failed,
    /// The test ran and errored (collection or setup error).
    Error,
    /// The test was collected and skipped. A skipped test is not a pass.
    Skipped,
}

/// The sandbox's report on one run (contracts/apr-code-eval-sandbox-v1.yaml).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxReport {
    /// Exit status of the sandbox.
    pub status: SandboxStatus,
    /// Which limit killed the run, if any.
    pub killed_by: Option<KilledBy>,
    /// Every executed test, by id. `tests_executed` is its length.
    pub outcomes: BTreeMap<String, TestOutcome>,
}

impl SandboxReport {
    /// Number of executed tests.
    #[must_use]
    pub fn tests_executed(&self) -> usize {
        self.outcomes.len()
    }

    /// pass(x) := x ∈ executed ∧ outcome(x) = passed. A test that never ran is not a pass.
    #[must_use]
    pub fn pass(&self, id: &str) -> bool {
        self.outcomes.get(id) == Some(&TestOutcome::Passed)
    }

    fn executed(&self, id: &str) -> bool {
        self.outcomes.contains_key(id)
    }
}

/// The task's two test lists (contracts/apr-code-eval-task-v1.yaml).
#[derive(Debug, Clone, Copy)]
pub struct TestLists<'a> {
    /// FAIL_TO_PASS: tests the fix must make pass.
    pub f2p: &'a [String],
    /// PASS_TO_PASS: tests that must still pass.
    pub p2p: &'a [String],
}

/// verdict(t, out, rep), the first-match function of ACE-VER-006. Total: every input gets
/// exactly one label and nothing panics.
#[must_use]
pub fn verdict(tests: TestLists<'_>, edit: EditOutcome, rep: &SandboxReport) -> Verdict {
    match edit {
        EditOutcome::Empty => return Verdict::Empty,
        EditOutcome::Unparsable | EditOutcome::NotApplied => return Verdict::FormatError,
        EditOutcome::Applied => {}
    }
    if rep.killed_by == Some(KilledBy::Timeout) {
        return Verdict::Timeout;
    }
    // ACE-SBX-003: status ok with a kill recorded is inconsistent, so it is an error too.
    if rep.status != SandboxStatus::Ok || rep.killed_by.is_some() {
        return Verdict::SandboxError;
    }
    if rep.tests_executed() == 0 || !tests.f2p.iter().all(|x| rep.executed(x)) {
        return Verdict::NoTests;
    }
    let all_pass = tests.f2p.iter().chain(tests.p2p).all(|x| rep.pass(x));
    if all_pass {
        Verdict::Resolved
    } else {
        Verdict::Failed
    }
}

#[cfg(test)]
#[path = "verdict_tests.rs"]
mod tests;
