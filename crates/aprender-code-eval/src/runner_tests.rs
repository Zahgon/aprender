//! FALSIFY-ACE-RUN-001..005 (contracts/apr-code-eval-runner-v1.yaml), against a fake
//! engine that reads the task dir at run time. The live engine is the nightly's (#5288).

use super::*;
use crate::sandbox::{Observed, SandboxError, REPORT_BEGIN, REPORT_END};
use crate::task::{HiddenTests, Snapshot, TaskKind};
use crate::verdict::{SandboxStatus, TestOutcome};
use std::time::Duration;

const DIGEST: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn env() -> TaskEnv {
    TaskEnv {
        image: ImageRef::parse(&format!("ghcr.io/paiml/eval-rust@{DIGEST}")).expect("pinned"),
        command: vec!["cargo".into(), "test".into()],
        report_path: "report.json".into(),
        report_format: ReportFormat::LibtestJson,
        limits: Limits::default(),
    }
}

fn task(id: &str) -> Task {
    Task {
        id: id.to_string(),
        kind: TaskKind::Function,
        release_date: Some("2025-03-01".into()),
        prompt: "p".into(),
        snapshot: Snapshot::default(),
        hidden_tests: HiddenTests {
            fail_to_pass: vec!["f2p".into()],
            ..HiddenTests::default()
        },
        gold: "g".into(),
    }
}

fn tree(pairs: &[(&str, &str)]) -> Files {
    pairs
        .iter()
        .map(|(p, b)| ((*p).to_string(), (*b).to_string()))
        .collect()
}

/// Reads the run's task dir when the container would start, then answers.
#[derive(Default)]
struct Fake {
    root: PathBuf,
    seen: Vec<(String, Files, Vec<String>)>,
    bad_image: bool,
    down: bool,
}

fn walk(dir: &Path, base: &Path, out: &mut Files) {
    for e in fs::read_dir(dir).expect("reads").flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, base, out);
        } else {
            let rel = p.strip_prefix(base).expect("under base");
            out.insert(
                rel.to_string_lossy().into_owned(),
                fs::read_to_string(&p).expect("utf8"),
            );
        }
    }
}

impl Engine for Fake {
    fn ensure_image(&mut self, image: &ImageRef) -> Result<(), SandboxError> {
        if self.bad_image {
            return Err(SandboxError::DigestMismatch(image.digest().to_string()));
        }
        Ok(())
    }

    fn run(&mut self, argv: &[String], name: &str, _: Duration) -> Result<Observed, SandboxError> {
        let dir = self.root.join(name);
        let mut files = Files::new();
        walk(&dir, &dir, &mut files);
        self.seen.push((name.to_string(), files, argv.to_vec()));
        if self.down {
            return Err(SandboxError::Engine("engine down".into()));
        }
        let report = r#"{ "type": "test", "event": "ok", "name": "f2p" }"#;
        Ok(Observed {
            exit_code: Some(0),
            timed_out: false,
            oom_killed: false,
            stdout: format!("{REPORT_BEGIN}\n{report}\n{REPORT_END}\n"),
        })
    }
}

fn runner(root: &Path, fake: Fake) -> Sandboxed<Fake> {
    let envs = [("t1".to_string(), env()), ("t2".to_string(), env())].into();
    Sandboxed::new(
        Fake {
            root: root.to_path_buf(),
            ..fake
        },
        envs,
        root.to_path_buf(),
        Some(PathBuf::from("/users/eval")),
        7,
        "test",
    )
}

fn empty(root: &Path) -> bool {
    fs::read_dir(root).expect("reads").next().is_none()
}

#[test]
fn falsify_ace_run_001_fresh_dir_and_name_per_run() {
    let tmp = tempfile::tempdir().expect("tmp");
    let mut r = runner(tmp.path(), Fake::default());
    let one = tree(&[("src/lib.rs", "a"), ("tests/hidden.rs", "h")]);
    let two = tree(&[("src/lib.rs", "b")]);
    let rep = r.run(&task("t1"), &one).expect("runs");
    assert_eq!(rep.status, SandboxStatus::Ok);
    assert_eq!(rep.outcomes.get("f2p"), Some(&TestOutcome::Passed));
    r.run(&task("t1"), &two).expect("runs");
    let seen = &r.engine.seen;
    assert_eq!(seen.len(), 2);
    assert_ne!(seen[0].0, seen[1].0, "container names differ");
    assert_eq!(seen[0].1, one, "the container sees exactly the tree");
    assert_eq!(seen[1].1, two, "nothing carries over from run 1");
    let mount = format!("source={}", tmp.path().join(&seen[1].0).display());
    assert!(
        seen[1].2.iter().any(|a| a.contains(&mount)),
        "the run's own dir is mounted"
    );
    assert!(empty(tmp.path()), "run dirs are removed");
}

#[test]
fn falsify_ace_run_002_unsafe_paths_are_refused_before_writing() {
    let tmp = tempfile::tempdir().expect("tmp");
    let root = tmp.path().join("runs");
    fs::create_dir(&root).expect("mkdir");
    for bad in [
        "../escape",
        "/etc/escape",
        "a//b",
        "./x",
        "",
        "a/../../escape",
    ] {
        let mut r = runner(&root, Fake::default());
        let t = tree(&[("aaa_first.rs", "x"), (bad, "y")]);
        let err = r.run(&task("t1"), &t).expect_err(bad);
        assert!(err.contains("unsafe tree path"), "{bad:?}: {err}");
        assert!(r.engine.seen.is_empty(), "{bad:?}: engine not called");
        assert!(empty(&root), "{bad:?}: nothing left in root");
        assert!(!tmp.path().join("escape").exists(), "{bad:?}");
    }
    for good in ["a", "src/a.rs", "a/b/c.txt", ".hidden"] {
        assert!(safe(good), "{good}");
    }
}

#[test]
fn falsify_ace_run_003_failures_are_errors_never_results() {
    let tmp = tempfile::tempdir().expect("tmp");
    let t = tree(&[("src/lib.rs", "a")]);
    let mut r = runner(tmp.path(), Fake::default());
    let err = r.run(&task("nope"), &t).expect_err("no env");
    assert!(err.contains("no environment"), "{err}");
    let mut undated = task("t1");
    undated.release_date = None;
    assert!(r
        .run(&undated, &t)
        .expect_err("undated")
        .contains("release date"));
    assert!(r.engine.seen.is_empty());
    let mut r = runner(
        tmp.path(),
        Fake {
            bad_image: true,
            ..Fake::default()
        },
    );
    assert!(r.run(&task("t1"), &t).is_err(), "image digest mismatch");
    assert!(r.engine.seen.is_empty(), "no run on a bad image");
    let mut r = runner(
        tmp.path(),
        Fake {
            down: true,
            ..Fake::default()
        },
    );
    assert!(r
        .run(&task("t2"), &t)
        .expect_err("down")
        .contains("engine down"));
    assert_eq!(r.engine.seen.len(), 1);
    assert!(empty(tmp.path()), "cleaned up after every failure");
}

#[test]
fn falsify_ace_run_004_source_date_epoch_is_the_release_date() {
    let rows = [
        ("1970-01-01", Some(0)),
        ("2000-02-29", Some(951_782_400)),
        ("2024-12-31", Some(1_735_603_200)),
        ("2025-03-01", Some(1_740_787_200)),
        ("1969-12-31", Some(-86_400)),
        ("2023-02-29", None),
        ("1900-02-29", None),
        ("2025-13-01", None),
        ("2025-04-31", None),
        ("2025-00-10", None),
        ("2025-3-1", None),
        ("2025-03-01x", None),
        ("2025-03-01-02", None),
        ("", None),
    ];
    for (date, want) in rows {
        assert_eq!(epoch(date), want, "{date}");
    }
    let tmp = tempfile::tempdir().expect("tmp");
    let mut r = runner(tmp.path(), Fake::default());
    r.run(&task("t1"), &tree(&[("a", "b")])).expect("runs");
    let argv = &r.engine.seen[0].2;
    assert!(argv.iter().any(|a| a == "SOURCE_DATE_EPOCH=1740787200"));
    assert!(argv.iter().any(|a| a == "SEED=7"));
}

#[test]
fn falsify_ace_run_005_image_digests_and_pinned_envs_only() {
    let tmp = tempfile::tempdir().expect("tmp");
    let r = runner(tmp.path(), Fake::default());
    let d = r.image_digests();
    assert_eq!(d.keys().collect::<Vec<_>>(), ["t1", "t2"]);
    assert!(d.values().all(|v| v == DIGEST));
    let json = serde_json::to_string(&env()).expect("serializes");
    let back: TaskEnv = serde_json::from_str(&json).expect("round trip");
    assert_eq!(back, env());
    let unpinned = json.replace(&format!("@{DIGEST}"), ":latest");
    assert!(
        serde_json::from_str::<TaskEnv>(&unpinned).is_err(),
        "tag-only image refused"
    );
    let short = json.replace(DIGEST, "sha256:0123");
    assert!(
        serde_json::from_str::<TaskEnv>(&short).is_err(),
        "short digest refused"
    );
}
