//! FALSIFY-ACE-RCP-001..006 (contracts/apr-code-eval-receipt-v1.yaml), on records from
//! the real driver with a planted agent and runner.

use super::*;
use crate::driver::{run_task, Agent, Budget, Reply, Request, TestRunner};
use crate::edit::{EditFormat, Files};
use crate::task::{HiddenTests, Snapshot, Task, TaskKind};
use crate::verdict::{SandboxReport, SandboxStatus, TestOutcome};

const FIX: &str =
    "src/lib.rs\n```rust\n<<<<<<< SEARCH\npub fn f() {}\n=======\npub fn f() -> u32 { 42 }\n>>>>>>> REPLACE\n```\n";
const WRONG: &str =
    "src/lib.rs\n```rust\n<<<<<<< SEARCH\npub fn f() {}\n=======\npub fn f() -> u32 { 7 }\n>>>>>>> REPLACE\n```\n";
const REFIX: &str = "src/lib.rs\n```rust\n<<<<<<< SEARCH\npub fn f() -> u32 { 7 }\n=======\npub fn f() -> u32 { 42 }\n>>>>>>> REPLACE\n```\n";
const PROSE: &str = "I would change f.";

fn task(id: &str, kind: TaskKind) -> Task {
    Task {
        id: id.to_string(),
        kind,
        release_date: Some("2025-03-01".into()),
        prompt: format!("Make `f` return 42 ({id})."),
        snapshot: Snapshot {
            files: [("src/lib.rs".to_string(), "pub fn f() {}\n".to_string())].into(),
            ..Snapshot::default()
        },
        hidden_tests: HiddenTests {
            fail_to_pass: vec!["f2p".into()],
            pass_to_pass: vec!["p2p".into()],
            files: [(
                "tests/hidden.rs".to_string(),
                "assert_eq!(f(), 42);".to_string(),
            )]
            .into(),
        },
        gold: "gold".into(),
    }
}

fn cfg(n: usize) -> Config {
    Config {
        format: EditFormat::SearchReplace,
        budget: Budget {
            max_turns: 4,
            max_tokens: 100,
        },
        seed: 7,
        n,
    }
}

/// Replies by script position, or by seed parity when `by_seed` is set.
struct Agent1 {
    script: Vec<&'static str>,
    calls: usize,
    by_seed: bool,
}

impl Agent1 {
    fn script(s: &[&'static str]) -> Self {
        Self {
            script: s.to_vec(),
            calls: 0,
            by_seed: false,
        }
    }
    fn by_seed() -> Self {
        Self {
            script: vec![],
            calls: 0,
            by_seed: true,
        }
    }
}

impl Agent for Agent1 {
    fn complete(&mut self, req: &Request<'_>) -> Result<Reply, String> {
        let text = if self.by_seed {
            match (req.seed % 2, req.attempt) {
                (0, _) | (_, 2) => FIX,
                _ => WRONG,
            }
        } else {
            self.script[self.calls % self.script.len()]
        };
        self.calls += 1;
        Ok(Reply {
            text: text.into(),
            tokens_in: 10,
            tokens_out: 20,
            turns: 1,
        })
    }
}

struct Runner;
impl TestRunner for Runner {
    fn run(&mut self, _: &Task, tree: &Files) -> Result<SandboxReport, String> {
        let ok = tree.get("src/lib.rs").is_some_and(|s| s.contains("42"));
        Ok(SandboxReport {
            status: SandboxStatus::Ok,
            killed_by: None,
            outcomes: [
                (
                    "f2p".to_string(),
                    if ok {
                        TestOutcome::Passed
                    } else {
                        TestOutcome::Failed
                    },
                ),
                ("p2p".to_string(), TestOutcome::Passed),
            ]
            .into(),
        })
    }
}

fn records(agent: &mut Agent1, tasks: &[Task], n: usize) -> Vec<TaskRecord> {
    tasks
        .iter()
        .map(|t| run_task(t, &cfg(n), agent, &mut Runner).expect("runs"))
        .collect()
}

fn run_for(ids: &[&str], n: usize) -> Run {
    Run {
        suite: "smoke".into(),
        task_set_digest: "d".repeat(64),
        model: Model {
            path: "/models/m.gguf".into(),
            sha256: "a".repeat(64),
            quant: "Q4_K_M".into(),
            backend: Backend::Cpu,
        },
        apr: Apr {
            version: "0.71.0".into(),
            git_sha: "b".repeat(40),
        },
        harness_git_sha: "c".repeat(40),
        sandbox: Sandbox {
            engine_version: "docker 27.3.1".into(),
            image_digests: ids
                .iter()
                .map(|id| ((*id).to_string(), format!("sha256:{}", "e".repeat(64))))
                .collect(),
        },
        seed: 7,
        sampling: Sampling {
            temperature: 0.0,
            top_p: 1.0,
            n,
            max_tokens: 100,
            max_turns: 4,
        },
        started_at: "2026-10-14T02:00:00Z".into(),
        wall_s: 12.5,
    }
}

fn smoke() -> (Vec<TaskRecord>, Run) {
    let tasks = [task("a", TaskKind::Edit), task("b", TaskKind::Function)];
    let recs = records(&mut Agent1::script(&[WRONG, REFIX]), &tasks, 1);
    (recs, run_for(&["a", "b"], 1))
}

#[test]
fn falsify_ace_rcp_001_every_provenance_field_is_required() {
    let (recs, run) = smoke();
    assert!(build(run.clone(), &cfg(1), &recs, None, None).is_ok());
    type Blank = fn(&mut Run);
    let rows: [(&str, Blank); 10] = [
        ("suite", |r| r.suite.clear()),
        ("task_set_digest", |r| r.task_set_digest.clear()),
        ("model.path", |r| r.model.path.clear()),
        ("model.sha256", |r| r.model.sha256 = "  ".into()),
        ("model.quant", |r| r.model.quant.clear()),
        ("apr.version", |r| r.apr.version.clear()),
        ("apr.git_sha", |r| r.apr.git_sha.clear()),
        ("harness_git_sha", |r| r.harness_git_sha.clear()),
        ("sandbox.engine_version", |r| {
            r.sandbox.engine_version.clear()
        }),
        ("started_at", |r| r.started_at.clear()),
    ];
    for (name, blank) in rows {
        let mut r = run.clone();
        blank(&mut r);
        assert_eq!(
            build(r, &cfg(1), &recs, None, None),
            Err(ReceiptError::Missing(name.to_string())),
            "{name}"
        );
    }
    let mut r = run.clone();
    r.sandbox.image_digests.remove("b");
    assert_eq!(
        build(r, &cfg(1), &recs, None, None),
        Err(ReceiptError::Missing("sandbox.image_digests.b".into()))
    );
    let mut r = run;
    r.sandbox.image_digests.insert("a".into(), String::new());
    assert_eq!(
        build(r, &cfg(1), &recs, None, None),
        Err(ReceiptError::Missing("sandbox.image_digests.a".into()))
    );
}

#[test]
fn falsify_ace_rcp_002_recorded_settings_are_the_driver_settings() {
    let (recs, run) = smoke();
    type Skew = fn(&mut Run);
    let rows: [(&str, Skew); 4] = [
        ("seed", |r| r.seed += 1),
        ("sampling.n", |r| r.sampling.n = 2),
        ("sampling.max_tokens", |r| r.sampling.max_tokens = 99),
        ("sampling.max_turns", |r| r.sampling.max_turns = 5),
    ];
    for (name, skew) in rows {
        let mut r = run.clone();
        skew(&mut r);
        assert_eq!(
            build(r, &cfg(1), &recs, None, None),
            Err(ReceiptError::Mismatch(name)),
            "{name}"
        );
    }
    // Records with fewer samples than the config says are refused too.
    let mut short = recs.clone();
    short[1].samples.clear();
    assert_eq!(
        build(run, &cfg(1), &short, None, None),
        Err(ReceiptError::Mismatch("sampling.n"))
    );
}

#[test]
fn falsify_ace_rcp_003_one_row_per_attempt_and_counts_sum() {
    let tasks = [
        task("a", TaskKind::Edit),
        task("b", TaskKind::Function),
        task("c", TaskKind::Function),
    ];
    let recs = records(&mut Agent1::script(&[PROSE, WRONG, REFIX, FIX]), &tasks, 1);
    let r = build(run_for(&["a", "b", "c"], 1), &cfg(1), &recs, None, None).expect("builds");
    let got: Vec<(&str, usize, u8, Verdict, bool)> = r
        .per_task
        .iter()
        .map(|x| {
            (
                x.id.as_str(),
                x.sample,
                x.attempt,
                x.verdict,
                x.edit_format_ok,
            )
        })
        .collect();
    assert_eq!(
        got,
        [
            ("a", 0, 1, Verdict::Empty, true),
            ("a", 0, 2, Verdict::Failed, true),
            ("b", 0, 1, Verdict::FormatError, false),
            ("c", 0, 1, Verdict::Resolved, true),
        ]
    );
    assert_eq!(r.counts_by_label.len(), Verdict::ALL.len());
    assert_eq!(r.counts_by_label.values().sum::<usize>(), r.per_task.len());
    assert_eq!(r.counts_by_label[&Verdict::Failed], 1);
    assert_eq!(r.counts_by_label[&Verdict::FormatError], 1);
    // b got REFIX against the snapshot: no edit applied, nothing ran, tokens still kept.
    assert_eq!(r.per_task[2].tokens_out, 20);
    assert!(r.per_task[2].report_sha256.is_none());
    assert!(r.per_task[1].report_sha256.is_some());
}

#[test]
fn falsify_ace_rcp_004_pass_at_1_per_attempt() {
    let (recs, run) = smoke();
    let r = build(run, &cfg(1), &recs, None, Some("2025-01-01")).expect("builds");
    // a: WRONG then REFIX (resolved on attempt 2); b: function task, WRONG, no retry.
    assert!((r.metrics.attempt_1.pass_at_1 - 0.0).abs() < 1e-12);
    assert!((r.metrics.attempt_2.pass_at_1 - 0.5).abs() < 1e-12);
    assert_eq!(r.metrics.attempt_2.post_cutoff_tasks, 2);
    let (lo, hi) = r.metrics.attempt_2.wilson95;
    assert!(lo < 0.5 && 0.5 < hi);
    // A metric refusal surfaces as a receipt error, not a number.
    let (recs, run) = smoke();
    assert!(matches!(
        build(run, &cfg(1), &recs, Some(2), None),
        Err(ReceiptError::Metric(MetricError::BadK { k: 2, n: 1 }))
    ));
}

#[test]
fn falsify_ace_rcp_005_seeded_receipts_match_apart_from_timing() {
    let tasks = [task("a", TaskKind::Edit), task("b", TaskKind::Function)];
    let one = records(&mut Agent1::by_seed(), &tasks, 3);
    let two = records(&mut Agent1::by_seed(), &tasks, 3);
    let r1 = build(run_for(&["a", "b"], 3), &cfg(3), &one, None, None).expect("builds");
    let mut run2 = run_for(&["a", "b"], 3);
    run2.started_at = "2026-10-15T02:00:00Z".into();
    run2.wall_s = 99.0;
    let mut r2 = build(run2, &cfg(3), &two, None, None).expect("builds");
    r2.per_task[0].wall_ms += 5;
    assert_ne!(r1.to_bytes(), r2.to_bytes(), "timing differs");
    assert!(r1.same_run(&r2), "and nothing else does");
    let mut r3 = r2.clone();
    r3.per_task[4].tokens_in += 1;
    assert!(!r1.same_run(&r3), "a non-timing field is compared");
    let mut r4 = r2;
    r4.run.model.sha256 = "f".repeat(64);
    assert!(!r1.same_run(&r4), "provenance is compared");
    // The §7 keys are at the top level.
    let v: serde_json::Value = serde_json::from_slice(&r1.to_bytes()).expect("json");
    for key in [
        "schema",
        "suite",
        "task_set_digest",
        "model",
        "apr",
        "harness_git_sha",
        "sandbox",
        "seed",
        "sampling",
        "started_at",
        "wall_s",
        "per_task",
        "counts_by_label",
        "metrics",
    ] {
        assert!(v.get(key).is_some(), "{key}");
    }
    assert_eq!(v["schema"], SCHEMA);
    assert_eq!(v["model"]["backend"], "cpu");
    assert!(v["metrics"]["attempt_1"].is_object() && v["metrics"]["attempt_2"].is_object());
    let back: Receipt = serde_json::from_slice(&r1.to_bytes()).expect("round trip");
    assert_eq!(back, r1);
}

#[test]
fn falsify_ace_rcp_006_only_like_receipts_compare() {
    let (recs, run) = smoke();
    let a = build(run.clone(), &cfg(1), &recs, None, None).expect("builds");
    let mut other_model = run.clone();
    other_model.model.sha256 = "f".repeat(64);
    other_model.model.backend = Backend::Cuda;
    let b = build(other_model, &cfg(1), &recs, None, None).expect("builds");
    assert_eq!(a.comparable(&b), Ok(()), "models are what gets compared");
    type Skew = fn(&mut Receipt);
    let rows: [(&str, Skew); 5] = [
        ("schema", |r| r.schema = "apr-code-eval-receipt/v0".into()),
        ("suite", |r| r.run.suite = "full".into()),
        ("task_set_digest", |r| {
            r.run.task_set_digest = "0".repeat(64)
        }),
        ("harness_git_sha", |r| {
            r.run.harness_git_sha = "1".repeat(40)
        }),
        ("sampling", |r| r.run.sampling.temperature = 0.2),
    ];
    for (name, skew) in rows {
        let mut c = b.clone();
        skew(&mut c);
        assert_eq!(
            a.comparable(&c),
            Err(ReceiptError::Incomparable(name)),
            "{name}"
        );
        assert_eq!(
            c.comparable(&a),
            Err(ReceiptError::Incomparable(name)),
            "{name}"
        );
    }
}
