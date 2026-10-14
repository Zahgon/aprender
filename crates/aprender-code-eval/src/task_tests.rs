//! FALSIFY-ACE-TASK-001..006 (contracts/apr-code-eval-task-v1.yaml).

use super::*;
use proptest::prelude::*;

const TODAY: &str = "2026-10-14";

fn task(id: &str) -> Task {
    Task {
        id: id.to_string(),
        kind: TaskKind::Function,
        release_date: Some("2025-03-01".into()),
        prompt: format!("Implement `{id}`."),
        snapshot: Snapshot {
            files: [("src/lib.rs".to_string(), "pub fn f() {}".to_string())].into(),
            ..Snapshot::default()
        },
        hidden_tests: HiddenTests {
            fail_to_pass: vec![format!("{id}::f2p")],
            pass_to_pass: vec![],
            files: [(
                "tests/hidden.rs".to_string(),
                "assert_eq!(f(), 42);".to_string(),
            )]
            .into(),
        },
        gold: "<<<<<<< SEARCH\npub fn f() {}\n=======\npub fn f() -> u32 { 42 }\n>>>>>>> REPLACE"
            .into(),
    }
}

fn line(t: &Task) -> String {
    serde_json::to_string(t).expect("serializes")
}

fn manifest(tasks: &[Task]) -> String {
    tasks.iter().map(line).collect::<Vec<_>>().join("\n")
}

/// A planted sandbox: resolves exactly the edits it is told to.
struct Planted {
    null_resolves: bool,
    gold_resolves: bool,
    calls: Vec<(String, bool)>,
}

impl Resolver for Planted {
    fn resolved(&mut self, task: &Task, edit: Option<&str>) -> Result<bool, String> {
        self.calls.push((task.id.clone(), edit.is_some()));
        Ok(if edit.is_some() {
            self.gold_resolves
        } else {
            self.null_resolves
        })
    }
}

fn planted(null_resolves: bool, gold_resolves: bool) -> Planted {
    Planted {
        null_resolves,
        gold_resolves,
        calls: Vec::new(),
    }
}

/// FALSIFY-ACE-TASK-001: zero tasks and a duplicate id are both refused.
#[test]
fn falsify_ace_task_001_empty_and_duplicate_refused() {
    assert_eq!(load("", TODAY), Err(TaskError::Empty));
    assert_eq!(load("\n  \n", TODAY), Err(TaskError::Empty));
    let dup = manifest(&[task("a"), task("b"), task("a")]);
    assert_eq!(load(&dup, TODAY), Err(TaskError::DuplicateId("a".into())));
    assert_eq!(
        load(&manifest(&[task("a"), task("b")]), TODAY).map(|s| s.tasks().len()),
        Ok(2)
    );
    assert!(matches!(
        load("{not json", TODAY),
        Err(TaskError::Parse { line: 1, .. })
    ));
}

/// FALSIFY-ACE-TASK-002: fail_to_pass: [] (or absent) is refused at load.
#[test]
fn falsify_ace_task_002_empty_f2p_refused() {
    let mut t = task("a");
    t.hidden_tests.fail_to_pass.clear();
    t.hidden_tests.pass_to_pass = vec!["a::p2p".into()];
    assert_eq!(
        load(&line(&t), TODAY),
        Err(TaskError::NoFailToPass("a".into()))
    );

    let absent = r#"{"id":"b","kind":"repo","release_date":"2025-01-01","prompt":"p","gold":"g"}"#;
    assert_eq!(
        load(absent, TODAY),
        Err(TaskError::NoFailToPass("b".into()))
    );
}

/// FALSIFY-ACE-TASK-003: a task the null edit resolves is rejected; so is one the gold
/// edit does not resolve; both runs always happen; a good task passes.
#[test]
fn falsify_ace_task_003_null_patch_task_rejected() {
    let mut sb = planted(true, true);
    assert_eq!(
        validate(task("a"), &mut sb),
        Err(TaskError::NullResolves("a".into()))
    );
    assert_eq!(sb.calls, vec![("a".into(), false), ("a".into(), true)]);

    assert_eq!(
        validate(task("a"), &mut planted(false, false)),
        Err(TaskError::GoldFails("a".into()))
    );
    assert!(validate(task("a"), &mut planted(false, true)).is_ok());

    // One bad task refuses the whole suite.
    struct NullResolvesFor(&'static str);
    impl Resolver for NullResolvesFor {
        fn resolved(&mut self, task: &Task, edit: Option<&str>) -> Result<bool, String> {
            Ok(edit.is_some() || task.id == self.0)
        }
    }
    let set = load(&manifest(&[task("a"), task("b"), task("c")]), TODAY).expect("loads");
    assert_eq!(
        Suite::build(set.clone(), &mut NullResolvesFor("b")),
        Err(TaskError::NullResolves("b".into()))
    );
    let suite = Suite::build(set.clone(), &mut NullResolvesFor("none")).expect("valid");
    assert_eq!(
        (suite.tasks().len(), suite.digest()),
        (3, set.digest().as_str())
    );

    struct Down;
    impl Resolver for Down {
        fn resolved(&mut self, _: &Task, _: Option<&str>) -> Result<bool, String> {
            Err("image missing".into())
        }
    }
    assert!(matches!(
        validate(task("a"), &mut Down),
        Err(TaskError::Sandbox { .. })
    ));
}

/// FALSIFY-ACE-TASK-004: a hidden test file in the snapshot is rejected naming the path;
/// so is one quoted in the prompt; the agent view never carries hidden bytes.
#[test]
fn falsify_ace_task_004_hidden_test_in_snapshot_rejected() {
    let mut t = task("a");
    t.snapshot
        .files
        .insert("tests/hidden.rs".into(), "anything".into());
    assert_eq!(
        load(&line(&t), TODAY),
        Err(TaskError::HiddenTestInSnapshot {
            id: "a".into(),
            path: "tests/hidden.rs".into()
        })
    );

    let mut t = task("a");
    t.prompt = "Make this pass:\nassert_eq!(f(), 42);\n".into();
    assert_eq!(
        load(&line(&t), TODAY),
        Err(TaskError::HiddenTestInPrompt {
            id: "a".into(),
            path: "tests/hidden.rs".into()
        })
    );

    let t = task("a");
    let view = serde_json::to_string(&t.agent_view()).expect("serializes");
    for (path, content) in &t.hidden_tests.files {
        assert!(!view.contains(path.as_str()) && !view.contains(content.as_str()));
    }
    assert!(!view.contains("f2p") && !view.contains("REPLACE"));
}

fn any_task() -> impl Strategy<Value = Task> {
    ("[a-z]{1,6}", "[ -~]{0,20}").prop_map(|(id, prompt)| Task {
        prompt,
        ..task(&id)
    })
}

proptest! {
    /// FALSIFY-ACE-TASK-005: shuffling keeps the digest; changing one byte changes it.
    #[test]
    fn prop_ace_task_005_digest(
        tasks in proptest::collection::btree_map("[a-z]{1,6}", any_task(), 1..8),
        rot in 0usize..8,
        pick in 0usize..8,
    ) {
        let mut v: Vec<Task> = tasks.into_iter().map(|(id, t)| Task { id, ..t }).collect();
        let d = digest(&v);
        prop_assert_eq!(d.len(), 64);
        let n = v.len();
        v.rotate_left(rot % n);
        v.reverse();
        prop_assert_eq!(&digest(&v), &d);
        let i = pick % n;
        v[i].prompt.push('!');
        prop_assert_ne!(&digest(&v), &d);
        v[i].prompt.pop();
        v[i].gold.push(' ');
        prop_assert_ne!(&digest(&v), &d);
    }
}

/// The id separator keeps (id, bytes) pairs from sliding into each other.
#[test]
fn digest_matches_its_formula() {
    let t = task("a");
    let inner: [u8; 32] = Sha256::digest(serde_json::to_vec(&t).expect("serializes")).into();
    let mut h = Sha256::new();
    h.update(b"a");
    h.update([0u8]);
    h.update(inner);
    let want: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(digest(&[t]), want);
}

/// FALSIFY-ACE-TASK-006: missing, malformed and future dates are refused.
#[test]
fn falsify_ace_task_006_undated_refused() {
    let mut t = task("a");
    t.release_date = None;
    assert_eq!(load(&line(&t), TODAY), Err(TaskError::Undated("a".into())));
    for bad in [
        "2025-02-29",
        "2025-13-01",
        "2025-1-01",
        "25-01-01",
        "2025/01/01",
        "2026-10-15",
        "",
        "２０２５-01-01",
    ] {
        let mut t = task("a");
        t.release_date = Some(bad.into());
        assert_eq!(
            load(&line(&t), TODAY),
            Err(TaskError::BadDate {
                id: "a".into(),
                date: bad.into()
            }),
            "{bad}"
        );
    }
    for good in ["2024-02-29", "2000-02-29", TODAY] {
        let mut t = task("a");
        t.release_date = Some(good.into());
        assert!(load(&line(&t), TODAY).is_ok(), "{good}");
    }
}
