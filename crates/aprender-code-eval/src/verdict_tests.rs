//! FALSIFY-ACE-VER-001..007 (contracts/apr-code-eval-verdict-v1.yaml).

use super::*;
use proptest::prelude::*;

fn ids(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| (*s).to_string()).collect()
}

fn report(pairs: &[(&str, TestOutcome)]) -> SandboxReport {
    SandboxReport {
        status: SandboxStatus::Ok,
        killed_by: None,
        outcomes: pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect(),
    }
}

const P: TestOutcome = TestOutcome::Passed;
const F: TestOutcome = TestOutcome::Failed;

/// FALSIFY-ACE-VER-001: F2P {all, some, none} pass × P2P {all pass, one fails} ×
/// executed {full, partial}. RESOLVED only on the all-pass, fully-executed row.
#[test]
fn falsify_ace_ver_001_resolved_table() {
    let f2p = ids(&["f1", "f2"]);
    let p2p = ids(&["p1", "p2"]);
    let lists = TestLists {
        f2p: &f2p,
        p2p: &p2p,
    };
    let f2p_rows: [(&str, [TestOutcome; 2]); 3] =
        [("all", [P, P]), ("some", [P, F]), ("none", [F, F])];
    let p2p_rows: [(&str, [TestOutcome; 2]); 2] = [("all", [P, P]), ("one-fails", [P, F])];
    let mut rows = 0;
    let mut resolved = 0;
    for (fname, fo) in f2p_rows {
        for (pname, po) in p2p_rows {
            for full in [true, false] {
                let mut pairs = vec![("f1", fo[0]), ("p1", po[0]), ("p2", po[1])];
                if full {
                    pairs.push(("f2", fo[1]));
                }
                let v = verdict(lists, EditOutcome::Applied, &report(&pairs));
                let want_resolved = fname == "all" && pname == "all" && full;
                assert_eq!(
                    v == Verdict::Resolved,
                    want_resolved,
                    "row f2p={fname} p2p={pname} full={full}: {v:?}"
                );
                if !full {
                    assert_eq!(v, Verdict::NoTests, "partial F2P execution is NO_TESTS");
                }
                rows += 1;
                resolved += usize::from(v == Verdict::Resolved);
            }
        }
    }
    assert_eq!((rows, resolved), (12, 1));
}

/// FALSIFY-ACE-VER-002: empty output against an all-pass report is EMPTY.
#[test]
fn falsify_ace_ver_002_empty_output() {
    let f2p = ids(&["f1"]);
    let lists = TestLists {
        f2p: &f2p,
        p2p: &[],
    };
    let rep = report(&[("f1", P)]);
    for raw in ["", "   \n\t"] {
        let edit = classify_edit(raw, Some(0), false);
        assert_eq!(edit, EditOutcome::Empty);
        assert_eq!(verdict(lists, edit, &rep), Verdict::Empty);
    }
    // Text with no edit block parses to zero edits: still EMPTY.
    assert_eq!(
        verdict(
            lists,
            classify_edit("I think it works.", Some(0), false),
            &rep
        ),
        Verdict::Empty
    );
}

/// FALSIFY-ACE-VER-003: status ok, nothing killed, zero executed tests → NO_TESTS.
#[test]
fn falsify_ace_ver_003_zero_executed() {
    let f2p = ids(&["f1"]);
    let lists = TestLists {
        f2p: &f2p,
        p2p: &[],
    };
    assert_eq!(
        verdict(lists, EditOutcome::Applied, &report(&[])),
        Verdict::NoTests
    );
    // Tests ran, but not the F2P one.
    assert_eq!(
        verdict(lists, EditOutcome::Applied, &report(&[("other", P)])),
        Verdict::NoTests
    );
}

/// FALSIFY-ACE-VER-004: a killed or failed sandbox is a label, even with passes in it.
#[test]
fn falsify_ace_ver_004_sandbox_failures() {
    let f2p = ids(&["f1"]);
    let lists = TestLists {
        f2p: &f2p,
        p2p: &[],
    };
    let mut rep = report(&[("f1", P)]);
    rep.killed_by = Some(KilledBy::Timeout);
    rep.status = SandboxStatus::Error("killed".into());
    assert_eq!(verdict(lists, EditOutcome::Applied, &rep), Verdict::Timeout);

    let mut rep = report(&[("f1", P)]);
    rep.status = SandboxStatus::Error("sandbox_error".into());
    assert_eq!(
        verdict(lists, EditOutcome::Applied, &rep),
        Verdict::SandboxError
    );

    let mut rep = report(&[("f1", P)]);
    rep.killed_by = Some(KilledBy::Oom);
    rep.status = SandboxStatus::Error("oom".into());
    assert_eq!(
        verdict(lists, EditOutcome::Applied, &rep),
        Verdict::SandboxError
    );

    // ACE-SBX-003 violated (ok status with a kill recorded): never scored.
    let mut rep = report(&[("f1", P)]);
    rep.killed_by = Some(KilledBy::Oom);
    assert_eq!(
        verdict(lists, EditOutcome::Applied, &rep),
        Verdict::SandboxError
    );
}

/// FALSIFY-ACE-VER-005: an unparsable or non-applying edit is FORMAT_ERROR, never
/// fuzzy-applied, and non-empty unparsable text is never EMPTY.
#[test]
fn falsify_ace_ver_005_format_error() {
    let f2p = ids(&["f1"]);
    let lists = TestLists {
        f2p: &f2p,
        p2p: &[],
    };
    let rep = report(&[("f1", P)]);
    // Search text not in the file: parsed one block, did not apply.
    let edit = classify_edit(
        "<<<<<<< SEARCH\nnot there\n=======\nx\n>>>>>>> REPLACE",
        Some(1),
        false,
    );
    assert_eq!(edit, EditOutcome::NotApplied);
    assert_eq!(verdict(lists, edit, &rep), Verdict::FormatError);
    // Non-empty text the parser rejects.
    let edit = classify_edit("<<<<<<< SEARCH\nunterminated", None, false);
    assert_eq!(edit, EditOutcome::Unparsable);
    assert_eq!(verdict(lists, edit, &rep), Verdict::FormatError);
}

/// FALSIFY-ACE-VER-007: all F2P pass and one P2P fails → FAILED.
#[test]
fn falsify_ace_ver_007_p2p_regression() {
    let f2p = ids(&["f1"]);
    let p2p = ids(&["p1", "p2"]);
    let lists = TestLists {
        f2p: &f2p,
        p2p: &p2p,
    };
    let rep = report(&[("f1", P), ("p1", P), ("p2", F)]);
    assert_eq!(verdict(lists, EditOutcome::Applied, &rep), Verdict::Failed);
    // A P2P test that never ran is not a pass either.
    let rep = report(&[("f1", P), ("p1", P)]);
    assert_eq!(verdict(lists, EditOutcome::Applied, &rep), Verdict::Failed);
}

fn any_outcome() -> impl Strategy<Value = TestOutcome> {
    prop_oneof![
        Just(P),
        Just(F),
        Just(TestOutcome::Error),
        Just(TestOutcome::Skipped)
    ]
}

fn any_edit() -> impl Strategy<Value = EditOutcome> {
    prop_oneof![
        Just(EditOutcome::Empty),
        Just(EditOutcome::Unparsable),
        Just(EditOutcome::NotApplied),
        Just(EditOutcome::Applied)
    ]
}

fn any_kill() -> impl Strategy<Value = Option<KilledBy>> {
    prop_oneof![
        Just(None),
        Just(Some(KilledBy::Timeout)),
        Just(Some(KilledBy::Oom)),
        Just(Some(KilledBy::Other))
    ]
}

proptest! {
    /// FALSIFY-ACE-VER-006: total over arbitrary inputs, always a member of V, and the
    /// first-match order holds (spot-checked against each consequence equation).
    #[test]
    fn prop_ace_ver_006_total(
        edit in any_edit(),
        ok in any::<bool>(),
        kill in any_kill(),
        outcomes in proptest::collection::btree_map("[a-d][0-2]", any_outcome(), 0..8),
        f2p in proptest::collection::vec("[a-d][0-2]", 0..4),
        p2p in proptest::collection::vec("[a-d][0-2]", 0..4),
    ) {
        let rep = SandboxReport {
            status: if ok { SandboxStatus::Ok } else { SandboxStatus::Error("x".into()) },
            killed_by: kill,
            outcomes,
        };
        let v = verdict(TestLists { f2p: &f2p, p2p: &p2p }, edit, &rep);
        prop_assert!(Verdict::ALL.contains(&v));
        if edit == EditOutcome::Empty { prop_assert_eq!(v, Verdict::Empty); }
        if !ok { prop_assert_ne!(v, Verdict::Resolved); }
        if rep.tests_executed() == 0 { prop_assert_ne!(v, Verdict::Resolved); }
        if p2p.iter().any(|y| !rep.pass(y)) { prop_assert_ne!(v, Verdict::Resolved); }
        if v == Verdict::Resolved {
            prop_assert!(edit == EditOutcome::Applied && ok && kill.is_none());
            prop_assert!(f2p.iter().chain(&p2p).all(|x| rep.pass(x)));
        }
    }
}
