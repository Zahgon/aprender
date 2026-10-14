//! The run receipt (contracts/apr-code-eval-receipt-v1.yaml, ACE-RCP-001..006).
//!
//! One JSON document per run, in the shape of docs/specifications/apr-code-eval-v1.md §7:
//! where the numbers came from, every attempt, and the metrics per attempt.
//!
//! - Every provenance field is required; an empty one refuses the receipt (ACE-RCP-001).
//! - The recorded seed and budget are the ones the driver ran with (ACE-RCP-002).
//! - One row per attempt; the label counts sum to the rows (ACE-RCP-003).
//! - pass@1 is reported for attempt 1 and for the final attempt (ACE-RCP-004).
//! - Two seeded runs are byte-identical once the timing fields are cleared (ACE-RCP-005).
//! - Two receipts compare only on the same suite, task set, harness and sampling
//!   (ACE-RCP-006).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::driver::{results, Config, Pass, TaskRecord};
use crate::metric::{compute, MetricError, Metrics};
use crate::verdict::Verdict;

/// The schema id every receipt carries.
pub const SCHEMA: &str = "apr-code-eval-receipt/v1";

/// Where the model ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// realizar on the CPU.
    Cpu,
    /// realizar on CUDA.
    Cuda,
}

/// The model under test.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    /// Model file.
    pub path: String,
    /// sha256 of the model file.
    pub sha256: String,
    /// Quantization, e.g. `Q4_K_M`.
    pub quant: String,
    /// Where it ran.
    pub backend: Backend,
}

/// The `apr` binary that drove the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Apr {
    /// `apr --version`.
    pub version: String,
    /// The git sha it was built from.
    pub git_sha: String,
}

/// The sandbox the tests ran in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sandbox {
    /// Container engine version.
    pub engine_version: String,
    /// Image digest per task id.
    pub image_digests: BTreeMap<String, String>,
}

/// Sampling settings. Part of the comparison key.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    /// Temperature.
    pub temperature: f64,
    /// Nucleus mass.
    pub top_p: f64,
    /// Samples per task.
    pub n: usize,
    /// Generated tokens per attempt.
    pub max_tokens: u64,
    /// Tool-loop turns per attempt.
    pub max_turns: u32,
}

/// Everything about a run except its results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    /// Suite name, e.g. `smoke`.
    pub suite: String,
    /// sha256 of the canonical task set.
    pub task_set_digest: String,
    /// The model.
    pub model: Model,
    /// The driving binary.
    pub apr: Apr,
    /// The git sha of this harness.
    pub harness_git_sha: String,
    /// The sandbox.
    pub sandbox: Sandbox,
    /// Base seed.
    pub seed: u64,
    /// Sampling settings.
    pub sampling: Sampling,
    /// RFC 3339 start time. Timing field.
    pub started_at: String,
    /// Wall time of the run in seconds. Timing field.
    pub wall_s: f64,
}

/// One attempt of one sample of one task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Row {
    /// Task id.
    pub id: String,
    /// Sample index.
    pub sample: usize,
    /// 1 or 2.
    pub attempt: u8,
    /// The label.
    pub verdict: Verdict,
    /// Tests executed.
    pub tests_executed: usize,
    /// Tests passed.
    pub tests_passed: usize,
    /// FAIL_TO_PASS tests passed.
    pub f2p_passed: usize,
    /// PASS_TO_PASS tests not passed.
    pub p2p_failed: usize,
    /// The reply was not a FORMAT_ERROR (the numerator of ACE-MET-005).
    pub edit_format_ok: bool,
    /// Prompt tokens.
    pub tokens_in: u64,
    /// Generated tokens.
    pub tokens_out: u64,
    /// Tool-loop turns.
    pub turns: u32,
    /// Wall time. Timing field.
    pub wall_ms: u64,
    /// sha256 of the sandbox report, when the tests ran.
    pub report_sha256: Option<String>,
}

/// pass@1 and the rest, per attempt (spec §8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerAttempt {
    /// Over every sample's first attempt.
    pub attempt_1: Metrics,
    /// Over every sample's final attempt.
    pub attempt_2: Metrics,
}

/// The receipt of one run (spec §7).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Receipt {
    /// Always [`SCHEMA`].
    pub schema: String,
    /// The run.
    #[serde(flatten)]
    pub run: Run,
    /// One row per attempt, in task, sample, attempt order.
    pub per_task: Vec<Row>,
    /// Labels over every row; sums to `per_task.len()`.
    pub counts_by_label: BTreeMap<Verdict, usize>,
    /// Metrics per attempt.
    pub metrics: PerAttempt,
}

/// The receipt cannot be written, or two receipts cannot be compared.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ReceiptError {
    /// ACE-RCP-001: a provenance field is empty.
    #[error("receipt field missing: {0}")]
    Missing(String),
    /// ACE-RCP-002: the recorded settings are not the ones the driver used.
    #[error("receipt field {0} does not match the driver config")]
    Mismatch(&'static str),
    /// The metric engine refused the results.
    #[error(transparent)]
    Metric(#[from] MetricError),
    /// ACE-RCP-006: the two receipts differ in a comparison key.
    #[error("receipts are not comparable: {0} differs")]
    Incomparable(&'static str),
}

fn missing(run: &Run, records: &[TaskRecord]) -> Option<String> {
    let fields = [
        ("suite", &run.suite),
        ("task_set_digest", &run.task_set_digest),
        ("model.path", &run.model.path),
        ("model.sha256", &run.model.sha256),
        ("model.quant", &run.model.quant),
        ("apr.version", &run.apr.version),
        ("apr.git_sha", &run.apr.git_sha),
        ("harness_git_sha", &run.harness_git_sha),
        ("sandbox.engine_version", &run.sandbox.engine_version),
        ("started_at", &run.started_at),
    ];
    if let Some((name, _)) = fields.iter().find(|(_, v)| v.trim().is_empty()) {
        return Some((*name).to_string());
    }
    records
        .iter()
        .find(|t| {
            run.sandbox
                .image_digests
                .get(&t.id)
                .is_none_or(|d| d.trim().is_empty())
        })
        .map(|t| format!("sandbox.image_digests.{}", t.id))
}

fn mismatch(run: &Run, cfg: &Config, records: &[TaskRecord]) -> Option<&'static str> {
    let s = &run.sampling;
    let checks = [
        ("seed", run.seed == cfg.seed),
        (
            "sampling.n",
            s.n == cfg.n && records.iter().all(|t| t.samples.len() == cfg.n),
        ),
        ("sampling.max_tokens", s.max_tokens == cfg.budget.max_tokens),
        ("sampling.max_turns", s.max_turns == cfg.budget.max_turns),
    ];
    checks.iter().find(|(_, ok)| !ok).map(|(name, _)| *name)
}

fn rows(records: &[TaskRecord]) -> Vec<Row> {
    let mut out = Vec::new();
    for t in records {
        for (sample, attempts) in t.samples.iter().enumerate() {
            for (i, a) in attempts.iter().enumerate() {
                out.push(Row {
                    id: t.id.clone(),
                    sample,
                    attempt: u8::try_from(i + 1).unwrap_or(u8::MAX),
                    verdict: a.verdict,
                    tests_executed: a.tests_executed,
                    tests_passed: a.tests_passed,
                    f2p_passed: a.f2p_passed,
                    p2p_failed: a.p2p_failed,
                    edit_format_ok: a.verdict != Verdict::FormatError,
                    tokens_in: a.tokens_in,
                    tokens_out: a.tokens_out,
                    turns: a.turns,
                    wall_ms: a.wall_ms,
                    report_sha256: a.report_sha256.clone(),
                });
            }
        }
    }
    out
}

/// Write the receipt of a driver run. `k` and `cutoff` go to the metric engine.
pub fn build(
    run: Run,
    cfg: &Config,
    records: &[TaskRecord],
    k: Option<usize>,
    cutoff: Option<&str>,
) -> Result<Receipt, ReceiptError> {
    if let Some(name) = missing(&run, records) {
        return Err(ReceiptError::Missing(name));
    }
    if let Some(name) = mismatch(&run, cfg, records) {
        return Err(ReceiptError::Mismatch(name));
    }
    let metrics = PerAttempt {
        attempt_1: compute(&results(records, Pass::Attempt1), k, cutoff)?,
        attempt_2: compute(&results(records, Pass::Attempt2), k, cutoff)?,
    };
    let per_task = rows(records);
    let mut counts_by_label: BTreeMap<Verdict, usize> =
        Verdict::ALL.iter().map(|v| (*v, 0)).collect();
    for r in &per_task {
        *counts_by_label.entry(r.verdict).or_insert(0) += 1;
    }
    Ok(Receipt {
        schema: SCHEMA.to_string(),
        run,
        per_task,
        counts_by_label,
        metrics,
    })
}

impl Receipt {
    /// ACE-RCP-005: the receipt with `started_at`, `wall_s` and every `wall_ms` cleared.
    #[must_use]
    pub fn without_timing(&self) -> Self {
        let mut out = self.clone();
        out.run.started_at = String::new();
        out.run.wall_s = 0.0;
        for r in &mut out.per_task {
            r.wall_ms = 0;
        }
        out
    }

    /// The receipt's JSON bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(self).expect("a receipt serializes")
    }

    /// ACE-RCP-005: the determinism check of spec §8.
    #[must_use]
    pub fn same_run(&self, other: &Self) -> bool {
        self.without_timing().to_bytes() == other.without_timing().to_bytes()
    }

    /// ACE-RCP-006: two receipts may be compared only on equal keys.
    pub fn comparable(&self, other: &Self) -> Result<(), ReceiptError> {
        let (a, b) = (&self.run, &other.run);
        let keys = [
            ("schema", self.schema == other.schema),
            ("suite", a.suite == b.suite),
            ("task_set_digest", a.task_set_digest == b.task_set_digest),
            ("harness_git_sha", a.harness_git_sha == b.harness_git_sha),
            ("sampling", a.sampling == b.sampling),
        ];
        match keys.iter().find(|(_, same)| !same) {
            Some((name, _)) => Err(ReceiptError::Incomparable(name)),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
#[path = "receipt_tests.rs"]
mod tests;
