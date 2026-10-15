//! FALSIFY-ACE-GAP-001..005 (contracts/apr-code-eval-gap-v1.yaml).

use super::*;
use crate::contamination::Contamination;
use crate::metric::Metrics;
use crate::receipt::{Apr, Model, PerAttempt, Run, Sampling, Sandbox};
use proptest::prelude::*;

fn row(id: &str, sample: usize, attempt: u8, verdict: Verdict) -> Row {
    Row {
        id: id.to_string(),
        sample,
        attempt,
        verdict,
        tests_executed: 2,
        tests_passed: usize::from(verdict == Verdict::Resolved) * 2,
        f2p_passed: usize::from(verdict == Verdict::Resolved),
        p2p_failed: 0,
        edit_format_ok: verdict != Verdict::FormatError,
        tokens_in: 100,
        tokens_out: 20,
        turns: 1,
        wall_ms: 1000,
        report_sha256: None,
    }
}

fn counts(rows: &[Row]) -> BTreeMap<Verdict, usize> {
    Verdict::ALL
        .iter()
        .map(|&v| (v, rows.iter().filter(|r| r.verdict == v).count()))
        .collect()
}

fn metrics(tasks: usize, n: usize, rows: &[Row]) -> Metrics {
    Metrics {
        tasks,
        n,
        pass_at_1: 0.5,
        pass_at_k: None,
        wilson95: (0.25, 0.75),
        format_conformance: 1.0,
        post_cutoff_pass_at_1: None,
        post_cutoff_tasks: 0,
        counts_by_label: counts(rows),
    }
}

fn receipt(rows: Vec<Row>) -> Receipt {
    let m = metrics(3, 1, &rows);
    Receipt {
        schema: SCHEMA.to_string(),
        run: Run {
            suite: "smoke".into(),
            task_set_digest: "d".repeat(64),
            model: Model {
                path: "qwen3.5-0.8b-q4km.gguf".into(),
                sha256: "a".repeat(64),
                quant: "Q4_K_M".into(),
                backend: Backend::Cpu,
            },
            apr: Apr {
                version: "0.71.3".into(),
                git_sha: "b".repeat(40),
            },
            harness_git_sha: "c".repeat(40),
            sandbox: Sandbox {
                engine_version: "docker 27".into(),
                image_digests: BTreeMap::new(),
            },
            seed: 7,
            sampling: Sampling {
                temperature: 0.0,
                top_p: 1.0,
                n: 1,
                max_tokens: 2048,
                max_turns: 1,
            },
            started_at: "2026-10-15T02:30:00Z".into(),
            wall_s: 61.0,
        },
        counts_by_label: counts(&rows),
        per_task: rows,
        metrics: PerAttempt {
            attempt_1: m.clone(),
            attempt_2: m,
        },
        contamination: Contamination::unchecked(),
    }
}

/// FALSIFY-ACE-GAP-001: a retry that fixes a sample removes it; a retry that breaks one
/// lists it with the retry's label, whatever order the rows come in.
#[test]
fn falsify_ace_gap_001_final_attempt_only() {
    let rows = vec![
        row("fixed", 0, 1, Verdict::Failed),
        row("fixed", 0, 2, Verdict::Resolved),
        row("broke", 0, 2, Verdict::Timeout),
        row("broke", 0, 1, Verdict::FormatError),
    ];
    let gaps = rank(&rows).expect("ranks");
    assert_eq!(gaps.len(), 1, "{gaps:?}");
    assert_eq!(gaps[0].id, "broke");
    assert_eq!(gaps[0].labels, BTreeMap::from([(Verdict::Timeout, 1)]));
}

/// FALSIFY-ACE-GAP-002: every task with a failing final sample appears once, and
/// failures + resolved finals = samples, with the labels summing to the failures.
#[test]
fn falsify_ace_gap_002_complete_and_reconciled() {
    let rows = vec![
        row("a", 0, 1, Verdict::Resolved),
        row("a", 1, 1, Verdict::Failed),
        row("a", 2, 1, Verdict::Empty),
        row("b", 0, 1, Verdict::Resolved),
        row("b", 1, 1, Verdict::Resolved),
        row("b", 2, 1, Verdict::Resolved),
        row("c", 0, 1, Verdict::SandboxError),
        row("c", 1, 1, Verdict::NoTests),
        row("c", 2, 1, Verdict::SandboxError),
    ];
    let gaps = rank(&rows).expect("ranks");
    let ids: Vec<&str> = gaps.iter().map(|g| g.id.as_str()).collect();
    assert_eq!(ids, ["c", "a"]);
    let resolved = rows
        .iter()
        .filter(|r| r.verdict == Verdict::Resolved)
        .count();
    let failing: usize = gaps.iter().map(|g| g.failures).sum();
    assert_eq!(failing + resolved, rows.len());
    for g in &gaps {
        assert_eq!(g.samples, 3);
        assert_eq!(g.labels.values().sum::<usize>(), g.failures);
        assert!(!g.labels.contains_key(&Verdict::Resolved));
    }
    assert_eq!(
        gaps[0].labels,
        BTreeMap::from([(Verdict::SandboxError, 2), (Verdict::NoTests, 1)])
    );
}

/// FALSIFY-ACE-GAP-003: more failures rank first, ties by id; reversing the rows gives
/// the same list.
#[test]
fn falsify_ace_gap_003_worst_first_and_order_free() {
    let mut rows = vec![
        row("z", 0, 1, Verdict::Failed),
        row("z", 1, 1, Verdict::Failed),
        row("m", 0, 1, Verdict::Failed),
        row("m", 1, 1, Verdict::Resolved),
        row("b", 0, 1, Verdict::Resolved),
        row("b", 1, 1, Verdict::Timeout),
    ];
    let gaps = rank(&rows).expect("ranks");
    let ids: Vec<&str> = gaps.iter().map(|g| g.id.as_str()).collect();
    assert_eq!(ids, ["z", "b", "m"]);
    rows.reverse();
    assert_eq!(rank(&rows).expect("ranks"), gaps);
}

fn any_verdict() -> impl Strategy<Value = Verdict> {
    prop::sample::select(Verdict::ALL.to_vec())
}

proptest! {
    /// FALSIFY-ACE-GAP-003 (order-free): any rotation of the rows ranks the same.
    #[test]
    fn prop_ace_gap_003_rotation(labels in proptest::collection::vec(any_verdict(), 1..40), shift in 0usize..40) {
        let mut rows: Vec<Row> = labels
            .iter()
            .enumerate()
            .map(|(i, &v)| row(&format!("t{}", i % 7), i / 7, 1, v))
            .collect();
        let gaps = rank(&rows).expect("ranks");
        let len = rows.len();
        rows.rotate_left(shift % len);
        prop_assert_eq!(rank(&rows).expect("ranks"), gaps.clone());
        // ACE-GAP-002 under the same input.
        let resolved = labels.iter().filter(|&&v| v == Verdict::Resolved).count();
        prop_assert_eq!(gaps.iter().map(|g| g.failures).sum::<usize>() + resolved, len);
        for w in gaps.windows(2) {
            prop_assert!(w[0].failures > w[1].failures || (w[0].failures == w[1].failures && w[0].id < w[1].id));
        }
    }
}

/// FALSIFY-ACE-GAP-004: no rows, another schema, a duplicated or zero attempt, and
/// tampered label counts are refused.
#[test]
fn falsify_ace_gap_004_refusals() {
    assert_eq!(rank(&[]), Err(GapError::Empty));
    assert_eq!(of_receipt(&receipt(vec![])), Err(GapError::Empty));

    let mut r = receipt(vec![row("a", 0, 1, Verdict::Failed)]);
    r.schema = "apr-code-eval-receipt/v0".into();
    assert!(matches!(of_receipt(&r), Err(GapError::Schema(_))));

    let dup = vec![
        row("a", 0, 1, Verdict::Failed),
        row("a", 0, 1, Verdict::Resolved),
    ];
    assert!(matches!(
        rank(&dup),
        Err(GapError::Attempt { attempt: 1, .. })
    ));
    assert!(matches!(
        rank(&[row("a", 0, 0, Verdict::Failed)]),
        Err(GapError::Attempt { attempt: 0, .. })
    ));

    let mut r = receipt(vec![row("a", 0, 1, Verdict::Failed)]);
    r.counts_by_label.insert(Verdict::Failed, 0);
    r.counts_by_label.insert(Verdict::Resolved, 1);
    assert!(matches!(of_receipt(&r), Err(GapError::Counts(_))));

    let ok = receipt(vec![row("a", 0, 1, Verdict::Failed)]);
    assert_eq!(of_receipt(&ok).expect("ranks").len(), 1);
}

/// FALSIFY-ACE-GAP-005: the comment names the run, lists the gaps in rank order, says so
/// when nothing failed, and two receipts that differ only in timing render identical text.
#[test]
fn falsify_ace_gap_005_render() {
    let rows = vec![
        row("a", 0, 1, Verdict::Failed),
        row("b", 0, 1, Verdict::Resolved),
        row("c", 0, 1, Verdict::FormatError),
    ];
    let r = receipt(rows);
    let gaps = of_receipt(&r).expect("ranks");
    let text = render(&r, &gaps);
    assert!(
        text.starts_with("apr code eval `smoke` on cpu: pass@1 0.5000"),
        "{text}"
    );
    assert!(text.contains("model sha256 `aaaaaaaaaaaa`"), "{text}");
    assert!(text.contains("| 1 | `a` | 1 / 1 | FAILED 1 |"), "{text}");
    assert!(
        text.contains("| 2 | `c` | 1 / 1 | FORMAT_ERROR 1 |"),
        "{text}"
    );
    assert!(!text.contains("`b`"), "{text}");

    let mut later = r.clone();
    later.run.started_at = "2026-10-16T02:30:00Z".into();
    later.run.wall_s = 99.0;
    for row in &mut later.per_task {
        row.wall_ms += 17;
    }
    assert!(r.same_run(&later));
    assert_eq!(render(&later, &of_receipt(&later).expect("ranks")), text);
    assert!(!text.contains("2026-10-15"), "{text}");

    let clean = receipt(vec![row("a", 0, 1, Verdict::Resolved)]);
    let text = render(&clean, &of_receipt(&clean).expect("ranks"));
    assert!(text.contains("No failing tasks"), "{text}");
    assert!(!text.contains("| rank |"), "{text}");
}
