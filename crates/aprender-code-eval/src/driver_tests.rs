//! FALSIFY-ACE-DRV-001..007 (contracts/apr-code-eval-driver-v1.yaml), with planted agents
//! and runners in place of the model and the sandbox.

use super::*;
use crate::metric::compute;
use crate::task::{load, HiddenTests, Resolver, Snapshot};
use crate::verdict::TestOutcome;

const FIX: &str =
    "src/lib.rs\n```rust\n<<<<<<< SEARCH\npub fn f() {}\n=======\npub fn f() -> u32 { 42 }\n>>>>>>> REPLACE\n```\n";
const MISS: &str =
    "src/lib.rs\n```rust\n<<<<<<< SEARCH\npub fn nope() {}\n=======\npub fn f() -> u32 { 42 }\n>>>>>>> REPLACE\n```\n";
const WRONG: &str =
    "src/lib.rs\n```rust\n<<<<<<< SEARCH\npub fn f() {}\n=======\npub fn f() -> u32 { 7 }\n>>>>>>> REPLACE\n```\n";
/// Attempt 2 after WRONG: it edits the tree attempt 1 left, not the snapshot.
const REFIX: &str = "src/lib.rs\n```rust\n<<<<<<< SEARCH\npub fn f() -> u32 { 7 }\n=======\npub fn f() -> u32 { 42 }\n>>>>>>> REPLACE\n```\n";
const SNEAK: &str = "tests/hidden.rs\n```rust\n<<<<<<< SEARCH\n=======\nfn always_pass() {}\n>>>>>>> REPLACE\n```\n";

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
            fail_to_pass: vec!["hidden_marker_f2p".into()],
            pass_to_pass: vec!["hidden_marker_p2p".into()],
            files: [(
                "tests/hidden.rs".to_string(),
                "assert_eq!(f(), 42); // HIDDEN_BODY".to_string(),
            )]
            .into(),
        },
        gold: "GOLD_MARKER".into(),
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

fn reply(text: &str) -> Reply {
    Reply {
        text: text.into(),
        tokens_in: 10,
        tokens_out: 20,
        turns: 1,
    }
}

/// Answers from a script, one reply per call, and keeps every request it was sent.
struct Scripted {
    replies: Vec<Reply>,
    seen: Vec<String>,
    seeds: Vec<(u64, u8)>,
}

impl Scripted {
    fn new(texts: &[&str]) -> Self {
        Self {
            replies: texts.iter().map(|t| reply(t)).collect(),
            seen: Vec::new(),
            seeds: Vec::new(),
        }
    }
}

impl Agent for Scripted {
    fn complete(&mut self, req: &Request<'_>) -> Result<Reply, String> {
        self.seen
            .push(serde_json::to_string(req).expect("a request serializes"));
        self.seeds.push((req.seed, req.attempt));
        let i = (self.seen.len() - 1) % self.replies.len();
        Ok(self.replies[i].clone())
    }
}

/// Passes the F2P test exactly when the tree returns 42 and the hidden file is intact; its
/// report names hidden tests and carries their output, which must never reach the agent.
#[derive(Default)]
struct Planted {
    trees: Vec<Files>,
}

impl TestRunner for Planted {
    fn run(&mut self, _: &Task, tree: &Files) -> Result<SandboxReport, String> {
        self.trees.push(tree.clone());
        let ok = tree.get("src/lib.rs").is_some_and(|s| s.contains("42"))
            && tree
                .get("tests/hidden.rs")
                .is_some_and(|s| s.contains("HIDDEN_BODY"));
        let f2p = if ok {
            TestOutcome::Passed
        } else {
            TestOutcome::Failed
        };
        Ok(SandboxReport {
            status: SandboxStatus::Ok,
            killed_by: None,
            outcomes: [
                ("hidden_marker_f2p".to_string(), f2p),
                ("hidden_marker_p2p".to_string(), TestOutcome::Passed),
            ]
            .into(),
        })
    }
}

fn verdicts(rec: &TaskRecord) -> Vec<Vec<Verdict>> {
    rec.samples
        .iter()
        .map(|s| s.iter().map(|a| a.verdict).collect())
        .collect()
}

#[test]
fn falsify_ace_drv_001_agent_sees_no_answer_key() {
    let t = task("t1", TaskKind::Edit);
    let mut agent = Scripted::new(&[WRONG, REFIX]);
    let rec = run_task(&t, &cfg(1), &mut agent, &mut Planted::default()).expect("runs");
    assert_eq!(
        verdicts(&rec),
        vec![vec![Verdict::Failed, Verdict::Resolved]]
    );
    assert_eq!(agent.seen.len(), 2);
    for req in &agent.seen {
        for secret in [
            "hidden_marker",
            "HIDDEN_BODY",
            "tests/hidden.rs",
            "GOLD_MARKER",
        ] {
            assert!(!req.contains(secret), "{secret} reached the agent: {req}");
        }
    }
    assert!(
        agent.seen[0].contains("Make `f` return 42 (t1)."),
        "the prompt is shown"
    );
}

#[test]
fn falsify_ace_drv_002_only_applied_edits_are_tested_with_hidden_files_last() {
    let t = task("t2", TaskKind::Function);
    let rows: [(&str, Verdict, usize); 5] = [
        ("", Verdict::Empty, 0),
        ("no edit here", Verdict::Empty, 0),
        (MISS, Verdict::FormatError, 0),
        (FIX, Verdict::Resolved, 1),
        (SNEAK, Verdict::Failed, 1),
    ];
    for (text, want, runs) in rows {
        let mut runner = Planted::default();
        let rec = run_task(&t, &cfg(1), &mut Scripted::new(&[text]), &mut runner).expect("runs");
        assert_eq!(verdicts(&rec), vec![vec![want]], "{text:?}");
        assert_eq!(runner.trees.len(), runs, "{text:?}: sandbox runs");
        let a = &rec.samples[0][0];
        assert_eq!(a.report_sha256.is_some(), runs == 1, "{text:?}");
        for tree in &runner.trees {
            assert_eq!(
                tree.get("tests/hidden.rs"),
                t.hidden_tests.files.get("tests/hidden.rs"),
                "{text:?}"
            );
        }
    }
    let rec = run_task(
        &t,
        &cfg(1),
        &mut Scripted::new(&[FIX]),
        &mut Planted::default(),
    )
    .expect("runs");
    let a = &rec.samples[0][0];
    assert_eq!(
        (a.tests_executed, a.tests_passed, a.f2p_passed, a.p2p_failed),
        (2, 2, 1, 0)
    );
}

#[test]
fn falsify_ace_drv_003_second_attempt_only_for_edit_with_leak_free_feedback() {
    let rows: [(TaskKind, &[&str], Vec<Verdict>); 5] = [
        (
            TaskKind::Edit,
            &[WRONG, REFIX],
            vec![Verdict::Failed, Verdict::Resolved],
        ),
        (TaskKind::Edit, &[FIX, WRONG], vec![Verdict::Resolved]),
        (
            TaskKind::Edit,
            &[MISS, FIX],
            vec![Verdict::FormatError, Verdict::Resolved],
        ),
        (TaskKind::Function, &[WRONG, FIX], vec![Verdict::Failed]),
        (TaskKind::Repo, &[WRONG, FIX], vec![Verdict::Failed]),
    ];
    for (kind, texts, want) in rows {
        let mut agent = Scripted::new(texts);
        let rec = run_task(
            &task("t3", kind),
            &cfg(1),
            &mut agent,
            &mut Planted::default(),
        )
        .expect("runs");
        assert_eq!(verdicts(&rec), vec![want.clone()], "{kind:?} {texts:?}");
        assert_eq!(
            agent.seeds.iter().map(|s| s.1).collect::<Vec<_>>(),
            (1..=want.len() as u8).collect::<Vec<_>>()
        );
    }
    // Attempt 2 starts from the tree attempt 1 left, and is told why without the report.
    let mut agent = Scripted::new(&[WRONG, REFIX]);
    let mut runner = Planted::default();
    run_task(
        &task("t3", TaskKind::Edit),
        &cfg(1),
        &mut agent,
        &mut runner,
    )
    .expect("runs");
    assert!(
        agent.seen[1].contains("{ 7 }"),
        "attempt 2 sees attempt 1's tree"
    );
    assert!(agent.seen[1].contains("The change does not make the tests pass yet."));
    for v in [
        Verdict::Resolved,
        Verdict::Failed,
        Verdict::FormatError,
        Verdict::Empty,
        Verdict::Timeout,
        Verdict::SandboxError,
        Verdict::NoTests,
    ] {
        let msg = feedback(v, Some("no match in src/lib.rs"));
        assert!(!msg.contains("hidden_marker") && !msg.is_empty(), "{v:?}");
    }
    assert!(
        feedback(Verdict::FormatError, Some("no match in src/lib.rs"))
            .contains("no match in src/lib.rs")
    );
}

#[test]
fn falsify_ace_drv_004_over_budget_is_timeout_and_untested() {
    let t = task("t4", TaskKind::Function);
    let at = Reply {
        turns: 4,
        tokens_out: 100,
        ..reply(FIX)
    };
    let rows: [(Reply, Verdict, usize); 3] = [
        (at.clone(), Verdict::Resolved, 1),
        (
            Reply {
                turns: 5,
                ..at.clone()
            },
            Verdict::Timeout,
            0,
        ),
        (
            Reply {
                tokens_out: 101,
                ..at
            },
            Verdict::Timeout,
            0,
        ),
    ];
    for (r, want, runs) in rows {
        let mut agent = Scripted {
            replies: vec![r.clone()],
            seen: Vec::new(),
            seeds: Vec::new(),
        };
        let mut runner = Planted::default();
        let rec = run_task(&t, &cfg(1), &mut agent, &mut runner).expect("runs");
        assert_eq!(verdicts(&rec), vec![vec![want]], "{r:?}");
        assert_eq!(runner.trees.len(), runs, "{r:?}");
    }
}

#[test]
fn falsify_ace_drv_005_backend_failure_stops_the_run_sandbox_failure_is_a_label() {
    struct Down;
    impl Agent for Down {
        fn complete(&mut self, _: &Request<'_>) -> Result<Reply, String> {
            Err("connection refused".into())
        }
    }
    struct Broken;
    impl TestRunner for Broken {
        fn run(&mut self, _: &Task, _: &Files) -> Result<SandboxReport, String> {
            Err("image missing".into())
        }
    }
    let t = task("t5", TaskKind::Function);
    assert_eq!(
        run_task(&t, &cfg(1), &mut Down, &mut Planted::default()),
        Err(DriverError::Agent {
            id: "t5".into(),
            sample: 0,
            attempt: 1,
            msg: "connection refused".into()
        })
    );
    let rec = run_task(&t, &cfg(1), &mut Scripted::new(&[FIX]), &mut Broken).expect("runs");
    assert_eq!(verdicts(&rec), vec![vec![Verdict::SandboxError]]);
    assert_eq!(
        run_task(
            &t,
            &cfg(0),
            &mut Scripted::new(&[FIX]),
            &mut Planted::default()
        ),
        Err(DriverError::NoSamples)
    );
}

/// Answers from its seed alone, the way a seeded sampler does.
struct BySeed;
impl Agent for BySeed {
    fn complete(&mut self, req: &Request<'_>) -> Result<Reply, String> {
        Ok(reply(if req.seed % 2 == 0 { FIX } else { WRONG }))
    }
}

struct AllPass;
impl Resolver for AllPass {
    fn resolved(&mut self, _: &Task, edit: Option<&str>) -> Result<bool, String> {
        Ok(edit.is_some())
    }
}

#[test]
fn falsify_ace_drv_006_seeded_runs_match_apart_from_wall_time() {
    let manifest: String = ["a", "b"]
        .iter()
        .map(|id| serde_json::to_string(&task(id, TaskKind::Edit)).expect("serializes") + "\n")
        .collect();
    let suite = Suite::build(load(&manifest, "2026-10-14").expect("loads"), &mut AllPass)
        .expect("validates");
    let one = run_suite(&suite, &cfg(3), &mut BySeed, &mut Planted::default()).expect("runs");
    let two = run_suite(&suite, &cfg(3), &mut BySeed, &mut Planted::default()).expect("runs");
    let bytes = |r: &[TaskRecord]| serde_json::to_vec(&without_timing(r)).expect("serializes");
    assert_eq!(bytes(&one), bytes(&two));
    assert_eq!(
        one.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    // Sample i uses seed + i: seeds 7, 8, 9 → WRONG, FIX, WRONG on attempt 1.
    let first: Vec<Verdict> = one[0].samples.iter().map(|s| s[0].verdict).collect();
    assert_eq!(first, [Verdict::Failed, Verdict::Resolved, Verdict::Failed]);
    let mut agent = Scripted::new(&[FIX]);
    run_task(
        &task("s", TaskKind::Function),
        &cfg(3),
        &mut agent,
        &mut Planted::default(),
    )
    .expect("runs");
    assert_eq!(agent.seeds, [(7, 1), (8, 1), (9, 1)]);
    let mut changed = one.clone();
    changed[1].samples[2][0].tokens_out += 1;
    assert_ne!(
        bytes(&one),
        bytes(&changed),
        "a non-timing field is compared"
    );
}

#[test]
fn falsify_ace_drv_007_results_per_attempt_feed_the_metric() {
    let mut agent = Scripted::new(&[WRONG, REFIX]);
    let rec = run_task(
        &task("t7", TaskKind::Edit),
        &cfg(2),
        &mut agent,
        &mut Planted::default(),
    )
    .expect("runs");
    let recs = [rec];
    let first = results(&recs, Pass::Attempt1);
    let last = results(&recs, Pass::Attempt2);
    assert_eq!(
        first[0]
            .samples
            .iter()
            .map(|s| s.verdict)
            .collect::<Vec<_>>(),
        [Verdict::Failed, Verdict::Failed]
    );
    assert_eq!(
        last[0]
            .samples
            .iter()
            .map(|s| s.verdict)
            .collect::<Vec<_>>(),
        [Verdict::Resolved, Verdict::Resolved]
    );
    assert_eq!(
        (first[0].task_id.as_str(), first[0].release_date.as_str()),
        ("t7", "2025-03-01")
    );
    let p1 = compute(&first, None, None).expect("scores").pass_at_1;
    let p2 = compute(&last, None, None).expect("scores").pass_at_1;
    assert!(p1 < p2, "attempt 2 is reported separately: {p1} vs {p2}");
}
