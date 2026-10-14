//! Metric engine and receipt gate (contracts/apr-code-eval-metric-v1.yaml, ACE-MET-001..008).
//!
//! Every task stays in the denominator whatever its label, pass@k is the unbiased estimator
//! of Chen et al. (2021), the Wilson interval is over every sample, and a run with no evidence
//! is refused rather than scored. A metric leaves this module only inside a complete receipt.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::verdict::Verdict;

/// z for a two-sided 95% interval (ACE-MET-003).
pub const Z95: f64 = 1.959964;

/// One sample of one task: its verdict and how many tests its sandbox run executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sample {
    /// The sample's label.
    pub verdict: Verdict,
    /// Tests the sandbox executed for this sample.
    pub tests_executed: usize,
}

/// Every sample of one task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskResult {
    /// The task id.
    pub task_id: String,
    /// The task's release date, `YYYY-MM-DD` (ACE-TASK-006), for the contamination split.
    pub release_date: String,
    /// The n samples, in sample order.
    pub samples: Vec<Sample>,
}

/// Why a run produced no number (ACE-MET-004).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Refusal {
    /// |T| = 0.
    NoTasks,
    /// Σ tests_executed = 0: nothing was measured.
    NoTestsExecuted,
    /// Every sample is EMPTY: a broken driver, not a 0% model.
    AllEmpty,
}

/// Errors of the metric engine. None of them carries a number.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MetricError {
    /// The run is refused (ACE-MET-004).
    #[error("refused: {0:?}")]
    Refused(Refusal),
    /// pass@k asked for k > n, or k = 0 (ACE-MET-002).
    #[error("pass@k needs 1 <= k <= n, got k={k}, n={n}")]
    BadK {
        /// Requested k.
        k: usize,
        /// Samples per task.
        n: usize,
    },
    /// Tasks carry different sample counts, or none.
    #[error("every task needs the same n >= 1 samples; task {0} differs")]
    UnevenSamples(String),
    /// A required receipt field is missing or empty (ACE-MET-007).
    #[error("receipt field missing: {0}")]
    MissingField(&'static str),
}

/// The numbers of one run. Built only by [`compute`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// |T|.
    pub tasks: usize,
    /// Samples per task.
    pub n: usize,
    /// ACE-MET-001.
    pub pass_at_1: f64,
    /// ACE-MET-002, as (k, value), when a k was asked for.
    pub pass_at_k: Option<(usize, f64)>,
    /// ACE-MET-003, (lo, hi), over N = |T|·n samples.
    pub wilson95: (f64, f64),
    /// ACE-MET-005.
    pub format_conformance: f64,
    /// ACE-MET-006: `None` when no task is after the cutoff, never 0.
    pub post_cutoff_pass_at_1: Option<f64>,
    /// |T_post|.
    pub post_cutoff_tasks: usize,
    /// ACE-MET-008: every label, summing to |T|·n.
    pub counts_by_label: BTreeMap<Verdict, usize>,
}

/// Unbiased pass@k for one task with `c` of `n` samples resolved, in the stable product
/// form 1 − Π_{i=n−c+1}^{n} (1 − k/i), and 1 when n − c < k (ACE-MET-002).
pub fn pass_at_k_task(n: usize, c: usize, k: usize) -> Result<f64, MetricError> {
    if k == 0 || k > n || c > n {
        return Err(MetricError::BadK { k, n });
    }
    if n - c < k {
        return Ok(1.0);
    }
    let prod: f64 = (n - c + 1..=n).map(|i| 1.0 - k as f64 / i as f64).product();
    Ok(1.0 - prod)
}

/// Wilson score interval for x successes in N trials (ACE-MET-003). N = 0 is refused
/// upstream; here it returns the uninformative (0, 1).
#[must_use]
pub fn wilson(x: usize, big_n: usize, z: f64) -> (f64, f64) {
    if big_n == 0 {
        return (0.0, 1.0);
    }
    let n = big_n as f64;
    let p_hat = x as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = p_hat + z2 / (2.0 * n);
    let half = z * (p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)).sqrt();
    (
        ((centre - half) / denom).max(0.0),
        ((centre + half) / denom).min(1.0),
    )
}

fn resolved(t: &TaskResult) -> usize {
    t.samples
        .iter()
        .filter(|s| s.verdict == Verdict::Resolved)
        .count()
}

fn uniform_n(results: &[TaskResult]) -> Result<usize, MetricError> {
    let n = results[0].samples.len();
    match results.iter().find(|t| t.samples.len() != n || n == 0) {
        Some(t) => Err(MetricError::UnevenSamples(t.task_id.clone())),
        None => Ok(n),
    }
}

/// ACE-MET-004: refuse a run that measured nothing.
fn refusal(results: &[TaskResult]) -> Option<Refusal> {
    let samples = || results.iter().flat_map(|t| &t.samples);
    if results.is_empty() {
        Some(Refusal::NoTasks)
    } else if samples().all(|s| s.verdict == Verdict::Empty) {
        Some(Refusal::AllEmpty)
    } else if samples().map(|s| s.tests_executed).sum::<usize>() == 0 {
        Some(Refusal::NoTestsExecuted)
    } else {
        None
    }
}

fn mean_pass_at_1(results: &[&TaskResult], n: usize) -> f64 {
    let sum: f64 = results.iter().map(|t| resolved(t) as f64 / n as f64).sum();
    sum / results.len() as f64
}

/// Compute every metric of one run, or refuse it. `k` asks for pass@k; `cutoff` is the
/// model's training cutoff as `YYYY-MM-DD` (ISO dates order as strings).
pub fn compute(
    results: &[TaskResult],
    k: Option<usize>,
    cutoff: Option<&str>,
) -> Result<Metrics, MetricError> {
    if let Some(r) = refusal(results) {
        return Err(MetricError::Refused(r));
    }
    let n = uniform_n(results)?;
    let all: Vec<&TaskResult> = results.iter().collect();
    let pass_at_k = match k {
        Some(k) => {
            let per_task: Result<Vec<f64>, _> = results
                .iter()
                .map(|t| pass_at_k_task(n, resolved(t), k))
                .collect();
            Some((k, per_task?.iter().sum::<f64>() / results.len() as f64))
        }
        None => None,
    };
    let big_n = results.len() * n;
    let x: usize = results.iter().map(resolved).sum();
    let mut counts_by_label: BTreeMap<Verdict, usize> =
        Verdict::ALL.iter().map(|v| (*v, 0)).collect();
    for s in results.iter().flat_map(|t| &t.samples) {
        *counts_by_label.entry(s.verdict).or_insert(0) += 1;
    }
    let post: Vec<&TaskResult> = match cutoff {
        Some(c) => results
            .iter()
            .filter(|t| t.release_date.as_str() > c)
            .collect(),
        None => Vec::new(),
    };
    Ok(Metrics {
        tasks: results.len(),
        n,
        pass_at_1: mean_pass_at_1(&all, n),
        pass_at_k,
        wilson95: wilson(x, big_n, Z95),
        format_conformance: 1.0 - counts_by_label[&Verdict::FormatError] as f64 / big_n as f64,
        post_cutoff_pass_at_1: (!post.is_empty()).then(|| mean_pass_at_1(&post, n)),
        post_cutoff_tasks: post.len(),
        counts_by_label,
    })
}

/// The provenance a receipt must carry beside its numbers (ACE-MET-007). Built by the
/// driver (#5286); an empty string or list counts as missing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// sha256 of the canonical task set.
    pub task_set_digest: String,
    /// sha256 of the model file.
    pub model_sha256: String,
    /// `apr --version`.
    pub apr_version: String,
    /// The git sha `apr` was built from.
    pub apr_git_sha: String,
    /// Every sandbox image, by digest.
    pub image_digests: Vec<String>,
    /// The fixed seed.
    pub seed: Option<u64>,
    /// Sampling settings (temperature, top_p, max tokens), as one canonical string.
    pub sampling: String,
    /// The git sha of this harness.
    pub harness_git_sha: String,
}

/// A complete receipt: provenance, every per-task verdict, and the metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Receipt {
    /// Where the numbers came from.
    pub provenance: Provenance,
    /// Every task's verdicts, in sample order.
    pub per_task_verdicts: BTreeMap<String, Vec<Verdict>>,
    /// The numbers.
    pub metrics: Metrics,
}

fn missing(p: &Provenance) -> Option<&'static str> {
    let fields: [(&'static str, bool); 8] = [
        ("task_set_digest", p.task_set_digest.is_empty()),
        ("model_sha256", p.model_sha256.is_empty()),
        ("apr_version", p.apr_version.is_empty()),
        ("apr_git_sha", p.apr_git_sha.is_empty()),
        (
            "image_digests",
            p.image_digests.is_empty() || p.image_digests.iter().any(String::is_empty),
        ),
        ("seed", p.seed.is_none()),
        ("sampling", p.sampling.is_empty()),
        ("harness_git_sha", p.harness_git_sha.is_empty()),
    ];
    fields
        .iter()
        .find(|(_, absent)| *absent)
        .map(|(name, _)| *name)
}

/// Bind metrics to a receipt, or refuse naming the first missing field (ACE-MET-007).
pub fn emit(
    provenance: Provenance,
    results: &[TaskResult],
    metrics: Metrics,
) -> Result<Receipt, MetricError> {
    if let Some(name) = missing(&provenance) {
        return Err(MetricError::MissingField(name));
    }
    let per_task_verdicts = results
        .iter()
        .map(|t| {
            (
                t.task_id.clone(),
                t.samples.iter().map(|s| s.verdict).collect(),
            )
        })
        .collect();
    Ok(Receipt {
        provenance,
        per_task_verdicts,
        metrics,
    })
}

#[cfg(test)]
#[path = "metric_tests.rs"]
mod tests;
