//! FALSIFY-ACE-HAR-001..006 (contracts/apr-code-eval-harness-v1.yaml), with a fake engine
//! that grades the tree it is shown and a scripted model.

use super::*;
use crate::driver::{Budget, Reply, Request};
use crate::edit::EditFormat;
use crate::receipt::{Apr, Backend, Model, Sampling, Sandbox};
use crate::runner::TaskEnv;
use crate::sandbox::{
    ImageRef, Limits, Observed, ReportFormat, SandboxError, REPORT_BEGIN, REPORT_END,
};
use crate::task::{load, HiddenTests, Snapshot, TaskKind};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

const DIGEST: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const SNAP: &str = "pub fn f() {}\n";
const GOLD: &str =
    "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-pub fn f() {}\n+pub fn f() -> u32 { 42 }\n";
const FIX: &str = "src/lib.rs\n```rust\n<<<<<<< SEARCH\npub fn f() {}\n=======\npub fn f() -> u32 { 42 }\n>>>>>>> REPLACE\n```\n";
const WRONG: &str = "src/lib.rs\n```rust\n<<<<<<< SEARCH\npub fn f() {}\n=======\npub fn f() -> u32 { 7 }\n>>>>>>> REPLACE\n```\n";

fn task(id: &str, snap: &str, gold: &str) -> Task {
    Task {
        id: id.to_string(),
        kind: TaskKind::Function,
        release_date: Some("2025-03-01".into()),
        prompt: "make f return 42".into(),
        snapshot: Snapshot {
            files: [("src/lib.rs".to_string(), snap.to_string())].into(),
            ..Snapshot::default()
        },
        hidden_tests: HiddenTests {
            fail_to_pass: vec!["f2p".into()],
            pass_to_pass: vec!["keep".into()],
            files: [("tests/hidden.rs".to_string(), "h".to_string())].into(),
        },
        gold: gold.to_string(),
    }
}

fn set(tasks: &[Task]) -> TaskSet {
    let manifest: String = tasks
        .iter()
        .map(|t| serde_json::to_string(t).expect("serializes") + "\n")
        .collect();
    load(&manifest, "2026-10-14").expect("loads")
}

fn env() -> TaskEnv {
    TaskEnv {
        image: ImageRef::parse(&format!("ghcr.io/paiml/eval-rust@{DIGEST}")).expect("pinned"),
        command: vec!["cargo".into(), "test".into()],
        report_path: "report.json".into(),
        report_format: ReportFormat::LibtestJson,
        limits: Limits::default(),
    }
}

/// Grades the tree it is shown: `f2p` passes when src/lib.rs returns 42 and the hidden
/// file is present. `flip_on` makes that call disagree; `down` fails every call.
#[derive(Default)]
struct Grader {
    root: PathBuf,
    calls: usize,
    flip_on: Option<usize>,
    down: bool,
}

impl Engine for Grader {
    fn ensure_image(&mut self, _: &ImageRef) -> Result<(), SandboxError> {
        Ok(())
    }

    fn run(&mut self, _: &[String], name: &str, _: Duration) -> Result<Observed, SandboxError> {
        self.calls += 1;
        if self.down {
            return Err(SandboxError::Engine("engine down".into()));
        }
        let dir = self.root.join(name);
        let lib = fs::read_to_string(dir.join("src/lib.rs")).unwrap_or_default();
        let fixed = lib.contains("42") && dir.join("tests/hidden.rs").exists();
        let pass = fixed != (self.flip_on == Some(self.calls));
        let event = if pass { "ok" } else { "failed" };
        let report = format!(
            "{{ \"type\": \"test\", \"event\": \"{event}\", \"name\": \"f2p\" }}\n{{ \"type\": \"test\", \"event\": \"ok\", \"name\": \"keep\" }}"
        );
        Ok(Observed {
            exit_code: Some(i32::from(!pass)),
            timed_out: false,
            oom_killed: false,
            stdout: format!("{REPORT_BEGIN}\n{report}\n{REPORT_END}\n"),
        })
    }
}

fn runner(root: &Path, ids: &[&str], g: Grader) -> Sandboxed<Grader> {
    let envs: BTreeMap<String, TaskEnv> = ids.iter().map(|id| ((*id).to_string(), env())).collect();
    Sandboxed::new(
        Grader {
            root: root.to_path_buf(),
            ..g
        },
        envs,
        root.to_path_buf(),
        Some(PathBuf::from("/users/eval")),
        7,
        "har",
    )
}

/// Answers every request with `text`, or fails when `text` is `None`.
struct Model1 {
    text: Option<&'static str>,
    calls: usize,
}

impl Model1 {
    fn new(text: Option<&'static str>) -> Self {
        Self { text, calls: 0 }
    }
}

impl Agent for Model1 {
    fn complete(&mut self, _: &Request<'_>) -> Result<Reply, String> {
        self.calls += 1;
        let text = self.text.ok_or("backend unreachable")?;
        Ok(Reply {
            text: text.to_string(),
            tokens_in: 10,
            tokens_out: 5,
            turns: 1,
        })
    }
}

fn cfg() -> Config {
    Config {
        format: EditFormat::SearchReplace,
        budget: Budget {
            max_turns: 4,
            max_tokens: 100,
        },
        seed: 7,
        n: 1,
    }
}

fn provenance(started_at: &str) -> Run {
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
            image_digests: [("zzz".to_string(), "sha256:bogus".to_string())].into(),
        },
        seed: 7,
        sampling: Sampling {
            temperature: 0.0,
            top_p: 1.0,
            n: 1,
            max_tokens: 100,
            max_turns: 4,
        },
        started_at: started_at.into(),
        wall_s: 0.0,
    }
}

fn eval(
    root: &Path,
    tasks: &[Task],
    ids: &[&str],
    g: Grader,
    model: &mut Model1,
) -> (Result<Receipt, HarnessError>, usize) {
    let mut r = runner(root, ids, g);
    let out = evaluate(
        set(tasks),
        &mut r,
        model,
        &cfg(),
        provenance("t0"),
        None,
        None,
    );
    (out, r.engine.calls)
}

#[test]
fn falsify_ace_har_001_gold_and_null_run_in_the_sandbox() {
    let tmp = tempfile::tempdir().expect("tmp");
    let t = task("a", SNAP, GOLD);
    let mut r = runner(tmp.path(), &["a"], Grader::default());
    assert_eq!(
        r.resolved(&t, None),
        Ok(false),
        "null edit leaves f2p failing"
    );
    assert_eq!(r.engine.calls, 1);
    assert_eq!(r.resolved(&t, Some(GOLD)), Ok(true), "gold resolves");
    assert_eq!(r.engine.calls, 1 + GOLD_RUNS);
    let stale = GOLD.replace("-pub fn f() {}", "-pub fn g() {}");
    assert_eq!(
        r.resolved(&t, Some(&stale)),
        Ok(false),
        "gold that does not apply"
    );
    assert_eq!(
        r.resolved(&t, Some(FIX)),
        Ok(false),
        "gold must be a unified diff"
    );
    assert_eq!(r.engine.calls, 1 + GOLD_RUNS, "unapplied gold never runs");

    let mut m = Model1::new(Some(FIX));
    let (out, _) = eval(
        tmp.path(),
        &[task("a", SNAP, &stale)],
        &["a"],
        Grader::default(),
        &mut m,
    );
    assert!(matches!(out, Err(HarnessError::Suite(TaskError::GoldFails(ref id))) if id == "a"));
    let solved = "pub fn f() -> u32 { 42 }\n";
    let (out, _) = eval(
        tmp.path(),
        &[task("a", solved, GOLD)],
        &["a"],
        Grader::default(),
        &mut m,
    );
    assert!(matches!(out, Err(HarnessError::Suite(TaskError::NullResolves(ref id))) if id == "a"));
    assert_eq!(m.calls, 0);
    assert!(fs::read_dir(tmp.path()).expect("reads").next().is_none());
}

#[test]
fn falsify_ace_har_002_a_flaky_gold_refuses_the_suite() {
    for flip_on in 2..=1 + GOLD_RUNS {
        let tmp = tempfile::tempdir().expect("tmp");
        let mut m = Model1::new(Some(FIX));
        let g = Grader {
            flip_on: Some(flip_on),
            ..Grader::default()
        };
        let (out, calls) = eval(tmp.path(), &[task("a", SNAP, GOLD)], &["a"], g, &mut m);
        let Err(HarnessError::Suite(TaskError::Sandbox { id, msg })) = out else {
            panic!("call {flip_on}: flake not refused: {out:?}");
        };
        assert_eq!(id, "a");
        assert!(msg.contains("flake"), "{msg}");
        assert_eq!(calls, 1 + GOLD_RUNS, "all gold runs happen");
        assert_eq!(m.calls, 0, "no model call on a flaky suite");
    }
}

#[test]
fn falsify_ace_har_003_every_task_needs_an_environment() {
    let tmp = tempfile::tempdir().expect("tmp");
    let mut m = Model1::new(Some(FIX));
    let tasks = [
        task("a", SNAP, GOLD),
        task("b", SNAP, GOLD),
        task("c", SNAP, GOLD),
    ];
    let (out, calls) = eval(tmp.path(), &tasks, &["b"], Grader::default(), &mut m);
    let Err(HarnessError::NoEnvironment(ids)) = out else {
        panic!("missing environments not refused: {out:?}");
    };
    assert_eq!(ids, ["a", "c"]);
    assert_eq!((calls, m.calls), (0, 0), "nothing runs before the check");
}

#[test]
fn falsify_ace_har_004_failures_write_no_receipt() {
    let tmp = tempfile::tempdir().expect("tmp");
    let mut m = Model1::new(Some(FIX));
    let down = Grader {
        down: true,
        ..Grader::default()
    };
    let (out, calls) = eval(tmp.path(), &[task("a", SNAP, GOLD)], &["a"], down, &mut m);
    assert!(matches!(
        out,
        Err(HarnessError::Suite(TaskError::Sandbox { .. }))
    ));
    assert_eq!((calls, m.calls), (1, 0));

    let mut dead = Model1::new(None);
    let (out, _) = eval(
        tmp.path(),
        &[task("a", SNAP, GOLD)],
        &["a"],
        Grader::default(),
        &mut dead,
    );
    assert!(
        matches!(out, Err(HarnessError::Driver(DriverError::Agent { .. }))),
        "{out:?}"
    );
    assert_eq!(dead.calls, 1);
    assert!(fs::read_dir(tmp.path()).expect("reads").next().is_none());
}

#[test]
fn falsify_ace_har_005_digests_are_those_of_the_suite_that_ran() {
    let tmp = tempfile::tempdir().expect("tmp");
    let tasks = [task("a", SNAP, GOLD), task("b", SNAP, GOLD)];
    let mut m = Model1::new(Some(FIX));
    let (out, _) = eval(
        tmp.path(),
        &tasks,
        &["a", "b", "extra"],
        Grader::default(),
        &mut m,
    );
    let rcp = out.expect("evaluates");
    assert_eq!(rcp.run.task_set_digest, set(&tasks).digest());
    assert_eq!(
        rcp.run.sandbox.image_digests,
        [
            ("a".to_string(), DIGEST.to_string()),
            ("b".to_string(), DIGEST.to_string())
        ]
        .into()
    );
    assert_eq!(rcp.metrics.attempt_1.pass_at_1, 1.0);
    assert_eq!(rcp.per_task.len(), 2);
}

#[test]
fn falsify_ace_har_006_seeded_evaluations_are_the_same_run() {
    let tasks = [task("a", SNAP, GOLD), task("b", SNAP, GOLD)];
    let receipt = |started_at: &str| {
        let tmp = tempfile::tempdir().expect("tmp");
        let mut r = runner(tmp.path(), &["a", "b"], Grader::default());
        let mut m = Model1::new(Some(FIX));
        evaluate(
            set(&tasks),
            &mut r,
            &mut m,
            &cfg(),
            provenance(started_at),
            None,
            None,
        )
        .expect("evaluates")
    };
    let one = receipt("2026-10-14T02:00:00Z");
    let two = receipt("2026-10-15T02:00:00Z");
    assert!(one.same_run(&two));
    assert!(one.comparable(&two).is_ok());
    let tmp = tempfile::tempdir().expect("tmp");
    let mut r = runner(tmp.path(), &["a", "b"], Grader::default());
    let mut wrong = Model1::new(Some(WRONG));
    let three = evaluate(
        set(&tasks),
        &mut r,
        &mut wrong,
        &cfg(),
        provenance("t"),
        None,
        None,
    )
    .expect("evaluates");
    assert!(
        !one.same_run(&three),
        "a different model output is a different run"
    );
}
