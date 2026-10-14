//! FALSIFY-ACE-PRM-001..005 (contracts/apr-code-eval-prompt-v1.yaml).

use super::*;
use crate::driver::Budget;
use crate::edit::{apply, parse};
use crate::task::{HiddenTests, Snapshot, Task, TaskKind};

const HIDDEN: &str = "fn hidden_marker_q7() { assert_eq!(f(), 42) }";
const GOLD: &str = "gold_marker_z9";

fn task(files: &[(&str, &str)]) -> Task {
    Task {
        id: "t1".into(),
        kind: TaskKind::Function,
        release_date: Some("2025-03-01".into()),
        prompt: "Make f return 42.\n".into(),
        snapshot: Snapshot {
            files: files
                .iter()
                .map(|(p, b)| ((*p).to_string(), (*b).to_string()))
                .collect(),
            ..Snapshot::default()
        },
        hidden_tests: HiddenTests {
            files: [("tests/hidden.rs".to_string(), HIDDEN.to_string())].into(),
            fail_to_pass: vec!["hidden_marker_q7".into()],
            ..HiddenTests::default()
        },
        gold: GOLD.into(),
    }
}

fn req<'a>(t: &'a Task, format: EditFormat, attempt: u8, feedback: Option<&'a str>) -> Request<'a> {
    Request {
        view: t.agent_view(),
        files: &t.snapshot.files,
        format,
        attempt,
        feedback,
        seed: 7,
        budget: Budget {
            max_turns: 1,
            max_tokens: 100,
        },
    }
}

const TEXT: [EditFormat; 3] = [
    EditFormat::Whole,
    EditFormat::SearchReplace,
    EditFormat::Udiff,
];

#[test]
fn falsify_ace_prm_001_every_file_verbatim_in_path_order() {
    let tricky = "let s = \"```\";\nlet t = \"````\";\n";
    let t = task(&[
        ("src/z.rs", "pub fn z() {}\n"),
        ("src/a.rs", tricky),
        ("README", "no newline"),
    ]);
    let p = render(&req(&t, EditFormat::SearchReplace, 1, None)).expect("renders");
    let a = p
        .user
        .find("src/a.rs\n`````\n")
        .expect("a.rs in a 5-backtick fence");
    let r = p
        .user
        .find("README\n```\nno newline\n```\n")
        .expect("README");
    let z = p
        .user
        .find("src/z.rs\n```\npub fn z() {}\n```\n")
        .expect("z.rs");
    assert!(r < a && a < z, "path order");
    assert!(p.user.contains(tricky));
    assert!(p.user.contains("Make f return 42."));
    for (body, want) in [("", 3), ("``", 3), ("```", 4), ("a````b`", 5)] {
        assert_eq!(fence_for(body).len(), want, "{body:?}");
    }
}

#[test]
fn falsify_ace_prm_002_hidden_tests_and_gold_never_shown() {
    let t = task(&[("src/lib.rs", "pub fn f() {}\n")]);
    for format in TEXT {
        for attempt in [1, 2] {
            let p = render(&req(&t, format, attempt, Some("1 failing test"))).expect("renders");
            for leak in [HIDDEN, "hidden_marker_q7", "tests/hidden.rs", GOLD] {
                assert!(!p.user.contains(leak), "{format:?}: {leak}");
                assert!(!p.system.contains(leak), "{format:?}: {leak}");
            }
        }
    }
}

#[test]
fn falsify_ace_prm_003_the_format_example_parses_and_applies() {
    let before: Files = [(EXAMPLE_PATH.to_string(), EXAMPLE_BEFORE.to_string())].into();
    let after = "pub fn answer() -> u32 {\n    42\n}\n";
    let t = task(&[("src/lib.rs", "pub fn f() {}\n")]);
    for format in TEXT {
        let ex = example(format).expect("text format");
        let edits = parse(format, &ex).expect("example parses under its own format");
        let out = apply(&before, &edits).expect("example applies");
        assert_eq!(out[EXAMPLE_PATH], after, "{format:?}");
        let p = render(&req(&t, format, 1, None)).expect("renders");
        assert!(
            p.system.contains(&ex),
            "{format:?}: the system message has its example"
        );
        for other in TEXT.iter().filter(|o| **o != format) {
            let o = example(*other).expect("text format");
            assert!(
                !p.system.contains(&o),
                "{format:?} shows {other:?}'s example"
            );
        }
    }
}

#[test]
fn falsify_ace_prm_004_feedback_only_on_attempt_two() {
    let t = task(&[("src/lib.rs", "pub fn f() {}\n")]);
    let fb = "FAILED: f2p hidden test failed";
    let one = render(&req(&t, EditFormat::Udiff, 1, Some(fb))).expect("renders");
    assert!(!one.user.contains(fb), "attempt 1 carries no feedback");
    assert!(!one.user.contains("previous edit"));
    let two = render(&req(&t, EditFormat::Udiff, 2, Some(fb))).expect("renders");
    assert!(two.user.contains(fb));
    assert!(two.user.contains("# Your previous edit failed"));
    let bare = render(&req(&t, EditFormat::Udiff, 2, None)).expect("renders");
    assert_eq!(bare, one, "no feedback, nothing added");
}

#[test]
fn falsify_ace_prm_005_deterministic_and_agentic_refused() {
    let t = task(&[("src/lib.rs", "pub fn f() {}\n"), ("b.rs", "x\n")]);
    for format in TEXT {
        let a = render(&req(&t, format, 2, Some("fb"))).expect("renders");
        let b = render(&req(&t, format, 2, Some("fb"))).expect("renders");
        assert_eq!(a, b, "{format:?}");
    }
    assert_eq!(
        render(&req(&t, EditFormat::Agentic, 1, None)),
        Err(PromptError::NeedsToolLoop(EditFormat::Agentic))
    );
    assert_eq!(example(EditFormat::Agentic), None);
}
