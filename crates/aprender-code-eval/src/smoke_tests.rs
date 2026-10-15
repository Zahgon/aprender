//! FALSIFY-ACE-SMK-001..005 (contracts/apr-code-eval-smoke-v1.yaml). 001..004 run with a
//! fake engine that grades a tree by comparing it with the gold tree; 005 is `#[ignore]`d
//! and needs docker with the pinned image pulled.

use super::*;
use crate::driver::{Agent, Budget, Config, Reply, Request};
use crate::edit::{adapt, EditFormat, Files};
use crate::harness::{evaluate, GOLD_FORMAT};
use crate::receipt::{Apr, Backend, Model, Receipt, Run, Sampling, Sandbox};
use crate::runner::Sandboxed;
use crate::sandbox::{Engine, ImageRef, Observed, SandboxError, REPORT_BEGIN, REPORT_END};
use crate::task::{Task, TaskKind};
use crate::verdict::EditOutcome;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

const LOAD_DATE: &str = "2026-10-15";
const HIDDEN: &str = "tests/hidden.rs";

fn all() -> Vec<Task> {
    tasks(LOAD_DATE).expect("smoke loads").tasks().to_vec()
}

/// The snapshot with the gold edit applied.
fn gold_tree(t: &Task) -> Files {
    let a = adapt(GOLD_FORMAT, &t.gold, &t.snapshot.files);
    assert_eq!(a.outcome, EditOutcome::Applied, "{}: {:?}", t.id, a.error);
    a.files.expect("applied")
}

/// The `#[test]` function names of a hidden test file, in file order.
fn test_names(src: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut armed = false;
    for line in src.lines().map(str::trim) {
        if line == "#[test]" {
            armed = true;
        } else if armed {
            let name = line
                .strip_prefix("fn ")
                .and_then(|r| r.split('(').next())
                .expect("#[test] is followed by fn");
            names.push(name.to_string());
            armed = false;
        }
    }
    names
}

/// FALSIFY-ACE-SMK-001: eight tasks, two per kind, each with an environment, all on one
/// pinned image, and no environment without a task.
#[test]
fn falsify_ace_smk_001_eight_tasks_two_per_kind_one_image() {
    let tasks = all();
    assert_eq!(tasks.len(), 8);
    for kind in [
        TaskKind::Repo,
        TaskKind::Function,
        TaskKind::Edit,
        TaskKind::Terminal,
    ] {
        let n = tasks.iter().filter(|t| t.kind == kind).count();
        assert_eq!(n, 2, "{kind:?}");
    }
    let envs = envs().expect("envs parse");
    let ids: Vec<&str> = tasks.iter().map(|t| t.id.as_str()).collect();
    let env_ids: Vec<&str> = envs.keys().map(String::as_str).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(env_ids, sorted, "one environment per task, none spare");
    let images: std::collections::BTreeSet<String> = envs
        .values()
        .map(|e| serde_json::to_string(&e.image).expect("image serializes"))
        .collect();
    assert_eq!(images.len(), 1, "{images:?}");
    for t in &tasks {
        assert!(t.id.starts_with("smoke-"), "{}", t.id);
        assert!(t.hidden_tests.files.contains_key(HIDDEN), "{}", t.id);
        assert!(t.snapshot.files.contains_key("Cargo.toml"), "{}", t.id);
    }
}

/// FALSIFY-ACE-SMK-002: every gold edit applies to its snapshot and changes it; a terminal
/// task's gold changes `solve.sh`.
#[test]
fn falsify_ace_smk_002_every_gold_applies_and_changes_the_snapshot() {
    for t in all() {
        let gold = gold_tree(&t);
        assert_ne!(gold, t.snapshot.files, "{}: gold is a no-op", t.id);
        if t.kind == TaskKind::Terminal {
            assert_ne!(
                gold.get("solve.sh"),
                t.snapshot.files.get("solve.sh"),
                "{}",
                t.id
            );
        }
    }
}

/// FALSIFY-ACE-SMK-003: FAIL_TO_PASS ∪ PASS_TO_PASS is exactly the hidden file's tests,
/// with no name in both lists.
#[test]
fn falsify_ace_smk_003_test_lists_name_exactly_the_hidden_tests() {
    for t in all() {
        let mut found = test_names(&t.hidden_tests.files[HIDDEN]);
        found.sort();
        let h = &t.hidden_tests;
        let mut listed: Vec<String> = h
            .fail_to_pass
            .iter()
            .chain(&h.pass_to_pass)
            .cloned()
            .collect();
        listed.sort();
        assert_eq!(listed, found, "{}", t.id);
        assert!(
            h.fail_to_pass.iter().all(|f| !h.pass_to_pass.contains(f)),
            "{}",
            t.id
        );
    }
}

/// Grades a tree the way the hidden tests would if the gold edit is the only correct
/// one: FAIL_TO_PASS passes when the tree's files equal the gold tree, PASS_TO_PASS
/// always passes.
struct GoldGrader {
    root: PathBuf,
    tasks: Vec<Task>,
}

impl Engine for GoldGrader {
    fn ensure_image(&mut self, _: &ImageRef) -> Result<(), SandboxError> {
        Ok(())
    }

    fn run(&mut self, _: &[String], name: &str, _: Duration) -> Result<Observed, SandboxError> {
        let dir = self.root.join(name);
        let hidden = fs::read_to_string(dir.join(HIDDEN)).unwrap_or_default();
        let task = self
            .tasks
            .iter()
            .find(|t| t.hidden_tests.files[HIDDEN] == hidden)
            .ok_or_else(|| SandboxError::Engine(format!("{name}: no task")))?;
        let solved = gold_tree(task)
            .iter()
            .all(|(p, body)| fs::read_to_string(dir.join(p)).ok().as_deref() == Some(body));
        let event = |ok: bool| if ok { "ok" } else { "failed" };
        let lines: Vec<String> = task
            .hidden_tests
            .fail_to_pass
            .iter()
            .map(|n| (n, event(solved)))
            .chain(task.hidden_tests.pass_to_pass.iter().map(|n| (n, "ok")))
            .map(|(n, e)| {
                format!("{{ \"type\": \"test\", \"event\": \"{e}\", \"name\": \"{n}\" }}")
            })
            .collect();
        Ok(Observed {
            exit_code: Some(i32::from(!solved)),
            timed_out: false,
            oom_killed: false,
            stdout: format!("{REPORT_BEGIN}\n{}\n{REPORT_END}\n", lines.join("\n")),
        })
    }
}

/// Answers each task with its gold edit when `solves` says so, else with an edit that
/// does not apply.
struct Scripted<F: Fn(&str) -> bool> {
    tasks: Vec<Task>,
    solves: F,
}

impl<F: Fn(&str) -> bool> Agent for Scripted<F> {
    fn complete(&mut self, req: &Request<'_>) -> Result<Reply, String> {
        let id = req.view.id;
        let task = self
            .tasks
            .iter()
            .find(|t| t.id == id)
            .ok_or("unknown task")?;
        let text = if (self.solves)(id) {
            task.gold.clone()
        } else {
            "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-no such line\n+x\n".to_string()
        };
        Ok(Reply {
            text,
            tokens_in: 100,
            tokens_out: 40,
            turns: 1,
        })
    }
}

fn cfg(seed: u64) -> Config {
    Config {
        format: EditFormat::Udiff,
        budget: Budget {
            max_turns: 1,
            max_tokens: 2048,
        },
        seed,
        n: 1,
    }
}

fn provenance(seed: u64, started_at: &str, engine_version: &str) -> Run {
    Run {
        suite: NAME.into(),
        task_set_digest: String::new(),
        model: Model {
            path: "scripted".into(),
            sha256: "a".repeat(64),
            quant: "none".into(),
            backend: Backend::Cpu,
        },
        apr: Apr {
            version: "0.0.0".into(),
            git_sha: "b".repeat(40),
        },
        harness_git_sha: "b".repeat(40),
        sandbox: Sandbox {
            engine_version: engine_version.into(),
            image_digests: BTreeMap::new(),
        },
        seed,
        sampling: Sampling {
            temperature: 0.0,
            top_p: 1.0,
            n: 1,
            max_tokens: 2048,
            max_turns: 1,
        },
        started_at: started_at.into(),
        wall_s: 0.0,
    }
}

/// One fake-engine run of the suite.
fn fake_run(seed: u64, started_at: &str, solves: fn(&str) -> bool) -> Receipt {
    let tmp = tempfile::tempdir().expect("tmp");
    let engine = GoldGrader {
        root: tmp.path().to_path_buf(),
        tasks: all(),
    };
    let mut runner = Sandboxed::new(
        engine,
        envs().expect("envs"),
        tmp.path().to_path_buf(),
        None,
        seed,
        "smk",
    );
    let mut agent = Scripted {
        tasks: all(),
        solves,
    };
    evaluate(
        tasks(LOAD_DATE).expect("loads"),
        &mut runner,
        &mut agent,
        &cfg(seed),
        provenance(seed, started_at, "fake"),
        None,
        None,
    )
    .expect("evaluates")
}

/// Solves the function and repo tasks, misses the edit and terminal ones.
fn half(id: &str) -> bool {
    id.starts_with("smoke-fn-") || id.starts_with("smoke-repo-")
}

/// FALSIFY-ACE-SMK-004 (§8): two seeded runs of `smoke` started at different times write
/// receipts that are byte-identical once the timing fields are cleared; one task answered
/// differently is a different run.
#[test]
fn falsify_ace_smk_004_two_seeded_runs_are_byte_identical() {
    let one = fake_run(7, "2026-10-15T02:30:00Z", half);
    let two = fake_run(7, "2026-10-16T02:30:00Z", half);
    assert_ne!(one.to_bytes(), two.to_bytes(), "started_at is in the bytes");
    assert_eq!(
        one.without_timing().to_bytes(),
        two.without_timing().to_bytes()
    );
    assert!(one.same_run(&two));
    assert_eq!(one.metrics.attempt_2.pass_at_1, 0.5);
    assert_eq!(one.run.suite, NAME);

    let other = fake_run(7, "2026-10-15T02:30:00Z", |id| {
        half(id) && id != "smoke-repo-word-frequency"
    });
    assert!(
        !one.same_run(&other),
        "one task answered differently is another run"
    );
}

/// FALSIFY-ACE-SMK-005: in the real sandbox, every null edit fails and every gold edit
/// resolves in three agreeing runs (`evaluate` validates the suite before it scores and
/// refuses on any failure), and two runs with one seed are the same run. Needs docker and
/// the pinned image: `cargo test -p aprender-code-eval --lib -- --ignored
/// falsify_ace_smk_005`.
#[test]
#[ignore = "needs docker and the pinned rust image pulled"]
fn falsify_ace_smk_005_validates_and_repeats_in_the_real_sandbox() {
    let engine = || crate::sandbox::Cli {
        bin: PathBuf::from(std::env::var_os("APR_EVAL_ENGINE").unwrap_or_else(|| "docker".into())),
    };
    let version = engine().version().expect("engine answers");
    let tmp = tempfile::tempdir().expect("tmp");
    let root = tmp.path().to_path_buf();
    let receipt = |tag: &str, started_at: &str| {
        let dir = root.join(tag);
        fs::create_dir(&dir).expect("run dir");
        let mut runner = Sandboxed::new(engine(), envs().expect("envs"), dir, None, 7, tag);
        let mut agent = Scripted {
            tasks: all(),
            solves: half,
        };
        evaluate(
            tasks(LOAD_DATE).expect("loads"),
            &mut runner,
            &mut agent,
            &cfg(7),
            provenance(7, started_at, &version),
            None,
            None,
        )
        .expect("evaluates")
    };
    let one = receipt("smka", "2026-10-15T02:30:00Z");
    let two = receipt("smkb", "2026-10-15T03:30:00Z");
    assert!(
        one.same_run(&two),
        "{}\n{}",
        String::from_utf8_lossy(&one.without_timing().to_bytes()),
        String::from_utf8_lossy(&two.without_timing().to_bytes())
    );
    assert_eq!(one.metrics.attempt_2.pass_at_1, 0.5);
}
