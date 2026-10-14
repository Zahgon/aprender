//! FALSIFY-ACE-MET-001..008 (contracts/apr-code-eval-metric-v1.yaml).

use super::*;
use proptest::prelude::*;

fn task(id: &str, date: &str, verdicts: &[Verdict]) -> TaskResult {
    TaskResult {
        task_id: id.to_string(),
        release_date: date.to_string(),
        samples: verdicts
            .iter()
            .map(|v| Sample {
                verdict: *v,
                tests_executed: 3,
            })
            .collect(),
    }
}

fn suite(labels: &[(Verdict, usize)]) -> Vec<TaskResult> {
    let mut out = Vec::new();
    for (v, count) in labels {
        for _ in 0..*count {
            out.push(task(&format!("t{}", out.len()), "2025-01-01", &[*v]));
        }
    }
    out
}

fn provenance() -> Provenance {
    Provenance {
        task_set_digest: "a".repeat(64),
        model_sha256: "b".repeat(64),
        apr_version: "0.71.0".into(),
        apr_git_sha: "c".repeat(40),
        image_digests: vec![format!("sha256:{}", "d".repeat(64))],
        seed: Some(42),
        sampling: "temperature=0,top_p=1,max_tokens=4096".into(),
        harness_git_sha: "e".repeat(40),
    }
}

/// Exact binomial form 1 − C(n−c,k)/C(n,k), in u128, as the reference for the stable form.
fn binom(n: u128, k: u128) -> u128 {
    if k > n {
        return 0;
    }
    (0..k).fold(1u128, |acc, i| acc * (n - i) / (i + 1))
}

fn reference_pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    1.0 - binom((n - c) as u128, k as u128) as f64 / binom(n as u128, k as u128) as f64
}

/// FALSIFY-ACE-MET-001: 5 RESOLVED, 3 SANDBOX_ERROR, 2 FAILED of 10 → 0.5, not 5/7.
#[test]
fn falsify_ace_met_001_denominator() {
    let r = suite(&[
        (Verdict::Resolved, 5),
        (Verdict::SandboxError, 3),
        (Verdict::Failed, 2),
    ]);
    let m = compute(&r, None, None).expect("scored");
    assert!((m.pass_at_1 - 0.5).abs() < 1e-15, "{}", m.pass_at_1);
    assert!((m.pass_at_1 - 5.0 / 7.0).abs() > 0.1);
}

/// FALSIFY-ACE-MET-002: Chen et al. reference values for n = 10, k > n refused, and
/// n = 1 gives pass@k = pass@1.
#[test]
fn falsify_ace_met_002_pass_at_k() {
    for c in [0, 1, 5, 10] {
        for k in [1, 5, 10] {
            let got = pass_at_k_task(10, c, k).expect("k <= n");
            let want = reference_pass_at_k(10, c, k);
            assert!(
                (got - want).abs() < 1e-12,
                "n=10 c={c} k={k}: {got} vs {want}"
            );
        }
    }
    assert!((pass_at_k_task(10, 5, 5).expect("ok") - (1.0 - 1.0 / 252.0)).abs() < 1e-12);
    assert_eq!(
        pass_at_k_task(10, 3, 11),
        Err(MetricError::BadK { k: 11, n: 10 })
    );
    assert_eq!(
        pass_at_k_task(10, 3, 0),
        Err(MetricError::BadK { k: 0, n: 10 })
    );

    let r = suite(&[(Verdict::Resolved, 3), (Verdict::Failed, 7)]);
    let m = compute(&r, Some(1), None).expect("scored");
    assert_eq!(m.pass_at_k, Some((1, m.pass_at_1)));
    assert_eq!(
        compute(&r, Some(2), None),
        Err(MetricError::BadK { k: 2, n: 1 })
    );
}

/// FALSIFY-ACE-MET-003: x = 50, N = 164 against a reference computed outside this crate
/// (awk, double precision, same z), and the bounds hold at 0 and N.
#[test]
fn falsify_ace_met_003_wilson() {
    let (lo, hi) = wilson(50, 164, Z95);
    assert!((lo - 0.239_555_516_051_691).abs() < 1e-9, "{lo}");
    assert!((hi - 0.379_132_259_742_350).abs() < 1e-9, "{hi}");
    let (lo, hi) = wilson(0, 164, Z95);
    assert!(lo == 0.0 && hi > 0.0 && hi <= 1.0);
    let (lo, hi) = wilson(164, 164, Z95);
    assert!(hi == 1.0 && lo < 1.0 && lo >= 0.0);
}

proptest! {
    /// FALSIFY-ACE-MET-003 (bound): 0 ≤ lo ≤ p_hat ≤ hi ≤ 1 for every x ≤ N.
    #[test]
    fn prop_ace_met_003_wilson_bounds(big_n in 1usize..5000, frac in 0.0f64..=1.0) {
        let x = ((big_n as f64) * frac).floor() as usize;
        let p_hat = x as f64 / big_n as f64;
        let (lo, hi) = wilson(x, big_n, Z95);
        prop_assert!(0.0 <= lo && lo <= p_hat + 1e-12 && p_hat <= hi + 1e-12 && hi <= 1.0, "{lo} {p_hat} {hi}");
    }

    /// FALSIFY-ACE-MET-002 (equivalence): the product form equals the binomial form.
    #[test]
    fn prop_ace_met_002_product_equals_binomial(n in 1usize..40, c_frac in 0.0f64..=1.0, k_frac in 0.0f64..=1.0) {
        let c = ((n as f64) * c_frac).floor() as usize;
        let k = 1 + ((n - 1) as f64 * k_frac).floor() as usize;
        let got = pass_at_k_task(n, c, k).expect("k <= n");
        prop_assert!((got - reference_pass_at_k(n, c, k)).abs() < 1e-9);
    }
}

/// FALSIFY-ACE-MET-004: the three planted runs are refused, with no number in the error.
/// A run where every sample is FORMAT_ERROR executes no tests, so it is refused too.
#[test]
fn falsify_ace_met_004_refusals() {
    assert_eq!(
        compute(&[], None, None),
        Err(MetricError::Refused(Refusal::NoTasks))
    );

    let mut zero = suite(&[(Verdict::NoTests, 10)]);
    for t in &mut zero {
        t.samples[0].tests_executed = 0;
    }
    assert_eq!(
        compute(&zero, None, None),
        Err(MetricError::Refused(Refusal::NoTestsExecuted))
    );

    let empty = suite(&[(Verdict::Empty, 10)]);
    assert_eq!(
        compute(&empty, None, None),
        Err(MetricError::Refused(Refusal::AllEmpty))
    );

    let mut format = suite(&[(Verdict::FormatError, 10)]);
    for t in &mut format {
        t.samples[0].tests_executed = 0;
    }
    let err = compute(&format, None, None).expect_err("refused");
    assert_eq!(err, MetricError::Refused(Refusal::NoTestsExecuted));
    assert!(!err.to_string().contains('%') && !err.to_string().chars().any(|c| c.is_ascii_digit()));
}

/// FALSIFY-ACE-MET-005: 4 FORMAT_ERROR of 20 tasks (n = 1) → 0.8.
#[test]
fn falsify_ace_met_005_format_conformance() {
    let r = suite(&[
        (Verdict::FormatError, 4),
        (Verdict::Resolved, 6),
        (Verdict::Failed, 10),
    ]);
    let m = compute(&r, None, None).expect("scored");
    assert!((m.format_conformance - 0.8).abs() < 1e-15);
}

/// FALSIFY-ACE-MET-006: every task before the cutoff → null with count 0, never 0.0.
#[test]
fn falsify_ace_met_006_post_cutoff() {
    let r = suite(&[(Verdict::Resolved, 2), (Verdict::Failed, 2)]);
    let m = compute(&r, None, Some("2026-01-01")).expect("scored");
    assert_eq!((m.post_cutoff_pass_at_1, m.post_cutoff_tasks), (None, 0));

    let mut r = r;
    r[0].release_date = "2026-06-01".into();
    r[2].release_date = "2026-06-01".into();
    let m = compute(&r, None, Some("2026-01-01")).expect("scored");
    assert_eq!(
        (m.post_cutoff_pass_at_1, m.post_cutoff_tasks),
        (Some(0.5), 2)
    );
}

/// FALSIFY-ACE-MET-007: removing any one required field makes emit() name that field.
#[test]
fn falsify_ace_met_007_receipt_fields() {
    let r = suite(&[(Verdict::Resolved, 1), (Verdict::Failed, 1)]);
    let m = compute(&r, None, None).expect("scored");
    assert!(emit(provenance(), &r, m.clone()).is_ok());
    type Strip = fn(&mut Provenance);
    let rows: [(&str, Strip); 8] = [
        ("task_set_digest", |p| p.task_set_digest.clear()),
        ("model_sha256", |p| p.model_sha256.clear()),
        ("apr_version", |p| p.apr_version.clear()),
        ("apr_git_sha", |p| p.apr_git_sha.clear()),
        ("image_digests", |p| p.image_digests.clear()),
        ("seed", |p| p.seed = None),
        ("sampling", |p| p.sampling.clear()),
        ("harness_git_sha", |p| p.harness_git_sha.clear()),
    ];
    for (name, strip) in rows {
        let mut p = provenance();
        strip(&mut p);
        assert_eq!(
            emit(p, &r, m.clone()),
            Err(MetricError::MissingField(name)),
            "{name}"
        );
    }
}

/// FALSIFY-ACE-MET-008: label counts sum to |T|·n and no verdict is lost, with n = 3.
#[test]
fn falsify_ace_met_008_counts_reconcile() {
    use Verdict::*;
    let r = vec![
        task("a", "2025-01-01", &[Resolved, Failed, Timeout]),
        task("b", "2025-01-01", &[Empty, FormatError, SandboxError]),
        task("c", "2025-01-01", &[NoTests, Resolved, Resolved]),
    ];
    let m = compute(&r, Some(2), None).expect("scored");
    assert_eq!(m.counts_by_label.values().sum::<usize>(), 9);
    assert_eq!(m.counts_by_label.len(), Verdict::ALL.len());
    assert_eq!(m.counts_by_label[&Resolved], 3);
    let rec = emit(provenance(), &r, m).expect("complete");
    assert_eq!(
        rec.per_task_verdicts.values().map(Vec::len).sum::<usize>(),
        9
    );
}

/// Uneven sample counts are an error, not a silent average.
#[test]
fn uneven_samples_rejected() {
    let r = vec![
        task("a", "2025-01-01", &[Verdict::Resolved]),
        task("b", "2025-01-01", &[Verdict::Resolved, Verdict::Failed]),
    ];
    assert_eq!(
        compute(&r, None, None),
        Err(MetricError::UnevenSamples("b".into()))
    );
}

fn any_verdict() -> impl Strategy<Value = Verdict> {
    proptest::sample::select(Verdict::ALL.to_vec())
}

proptest! {
    /// FALSIFY-ACE-MET-008: for any verdict vector that is scored, Σ counts_by_label = |T|·n.
    #[test]
    fn prop_ace_met_008_counts(n in 1usize..5, labels in proptest::collection::vec(any_verdict(), 1..60)) {
        let tasks: Vec<TaskResult> = labels
            .chunks(n)
            .filter(|c| c.len() == n)
            .enumerate()
            .map(|(i, c)| task(&format!("t{i}"), "2025-01-01", c))
            .collect();
        if let Ok(m) = compute(&tasks, None, None) {
            prop_assert_eq!(m.counts_by_label.values().sum::<usize>(), tasks.len() * n);
            prop_assert_eq!(m.counts_by_label.len(), Verdict::ALL.len());
        } else {
            prop_assert!(tasks.is_empty() || tasks.iter().flat_map(|t| &t.samples).all(|s| s.verdict == Verdict::Empty));
        }
    }
}
