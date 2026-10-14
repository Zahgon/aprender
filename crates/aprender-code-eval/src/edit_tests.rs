//! FALSIFY-ACE-EDIT-001..008 (contracts/apr-code-eval-edit-v1.yaml).

use super::*;
use proptest::prelude::*;

fn files(pairs: &[(&str, &str)]) -> Files {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

fn snapshot() -> Files {
    files(&[(
        "src/lib.rs",
        "pub fn f() -> u32 {\n    0\n}\n\npub fn g() {}\n",
    )])
}

fn sr(path: &str, search: &str, replace: &str) -> String {
    format!("{path}\n```rust\n{SEARCH}\n{search}\n{DIVIDER}\n{replace}\n{REPLACE}\n```\n")
}

const ALL: [EditFormat; 4] = [
    EditFormat::Whole,
    EditFormat::SearchReplace,
    EditFormat::Udiff,
    EditFormat::Agentic,
];

proptest! {
    /// FALSIFY-ACE-EDIT-001: every format parses every input to edits or an error, without
    /// panicking, and the same input gives the same answer.
    #[test]
    fn falsify_ace_edit_001_parse_total(raw in "(```|<<<<<<< SEARCH|=======|>>>>>>> REPLACE|--- a/x|\\+\\+\\+ b/x|@@ -1 \\+1 @@|[-+ ]?[a-z/.{}\"]{0,8}|\n){0,24}") {
        for format in ALL {
            prop_assert_eq!(parse(format, &raw), parse(format, &raw));
            let a = adapt(format, &raw, &snapshot());
            prop_assert_eq!(a.files.is_some(), a.outcome == EditOutcome::Applied);
        }
    }
}

/// FALSIFY-ACE-EDIT-002: a search applies only when it matches whole lines exactly once.
/// No whitespace repair, no partial-line match, no first-of-many.
#[test]
fn falsify_ace_edit_002_exact_match_no_repair() {
    let snap = snapshot();
    let ok = parse(
        EditFormat::SearchReplace,
        &sr("src/lib.rs", "    0", "    42"),
    )
    .expect("parses");
    let got = apply(&snap, &ok).expect("applies");
    assert_eq!(
        got["src/lib.rs"],
        "pub fn f() -> u32 {\n    42\n}\n\npub fn g() {}\n"
    );

    let cases = [
        ("  0", ApplyError::NoMatch("src/lib.rs".into())),
        ("    0 ", ApplyError::NoMatch("src/lib.rs".into())),
        ("\t0", ApplyError::NoMatch("src/lib.rs".into())),
        ("fn f()", ApplyError::NoMatch("src/lib.rs".into())),
    ];
    for (search, want) in cases {
        let e = parse(EditFormat::SearchReplace, &sr("src/lib.rs", search, "x")).expect("parses");
        assert_eq!(apply(&snap, &e), Err(want), "{search:?}");
    }
    let twice = files(&[("d.rs", "x\ny\nx\n")]);
    let e = parse(EditFormat::SearchReplace, &sr("d.rs", "x", "z")).expect("parses");
    assert_eq!(
        apply(&twice, &e),
        Err(ApplyError::Ambiguous("d.rs".into(), 2))
    );
    let e = parse(EditFormat::SearchReplace, &sr("d.rs", "y", "")).expect("parses");
    assert_eq!(apply(&twice, &e).expect("deletes")["d.rs"], "x\nx\n");
    let missing = parse(EditFormat::SearchReplace, &sr("src/main.rs", "x", "y")).expect("parses");
    assert_eq!(
        apply(&snap, &missing),
        Err(ApplyError::Missing("src/main.rs".into()))
    );
    let create = parse(
        EditFormat::SearchReplace,
        &sr("src/new.rs", "", "pub mod x;"),
    )
    .expect("parses");
    assert_eq!(
        apply(&snap, &create).expect("creates")["src/new.rs"],
        "pub mod x;\n"
    );
    let mut over = create.clone();
    if let Edit::Replace { path, .. } = &mut over[0] {
        *path = "src/lib.rs".into();
    }
    assert_eq!(
        apply(&snap, &over),
        Err(ApplyError::Exists("src/lib.rs".into()))
    );
}

#[test]
fn search_replace_parse_table() {
    let two = format!(
        "Change both.\n\nsrc/lib.rs\n```rust\n{SEARCH}\n    0\n{DIVIDER}\n    1\n{REPLACE}\n{SEARCH}\npub fn g() {{}}\n{DIVIDER}\n{REPLACE}\n```\n"
    );
    let e = parse(EditFormat::SearchReplace, &two).expect("parses");
    assert_eq!(e.len(), 2);
    assert!(e.iter().all(|e| e.path() == "src/lib.rs"));
    assert_eq!(
        parse(EditFormat::SearchReplace, "no blocks here\n"),
        Ok(vec![])
    );
    let bad = [
        format!("{SEARCH}\na\n{DIVIDER}\nb\n{REPLACE}\n"),
        format!("Here is the fix:\n{SEARCH}\na\n{DIVIDER}\nb\n{REPLACE}\n"),
        format!("x.rs\n{SEARCH}\na\n{DIVIDER}\nb\n"),
        format!("x.rs\n{SEARCH}\na\n{REPLACE}\n"),
        format!("x.rs\n{SEARCH}\na\n{SEARCH}\n{DIVIDER}\nb\n{REPLACE}\n"),
        format!("x.rs\n{DIVIDER}\n"),
    ];
    for raw in &bad {
        assert!(parse(EditFormat::SearchReplace, raw).is_err(), "{raw}");
    }
}

/// FALSIFY-ACE-EDIT-003: a unified diff applies by content, hunk by hunk; a context line
/// that does not match refuses the whole diff; new and deleted files are explicit.
#[test]
fn falsify_ace_edit_003_udiff() {
    let snap = snapshot();
    let diff = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1,3 +1,3 @@\n pub fn f() -> u32 {\n-    0\n+    42\n }\n@@ -5 +5 @@\n-pub fn g() {}\n+pub fn g() -> u8 { 1 }\n\nThat is all.\n";
    let got = apply(&snap, &parse(EditFormat::Udiff, diff).expect("parses")).expect("applies");
    assert_eq!(
        got["src/lib.rs"],
        "pub fn f() -> u32 {\n    42\n}\n\npub fn g() -> u8 { 1 }\n"
    );

    let wrong_numbers =
        "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -90,1 +90,1 @@\n-pub fn g() {}\n+pub fn h() {}\n";
    assert!(apply(
        &snap,
        &parse(EditFormat::Udiff, wrong_numbers).expect("parses")
    )
    .is_ok());

    let bad_context = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1,3 +1,3 @@\n pub fn f() -> u64 {\n-    0\n+    42\n }\n";
    let e = parse(EditFormat::Udiff, bad_context).expect("parses");
    assert_eq!(
        apply(&snap, &e),
        Err(ApplyError::NoMatch("src/lib.rs".into()))
    );

    let new_file = "--- /dev/null\n+++ b/src/new.rs\n@@ -0,0 +1,2 @@\n+pub mod a;\n+pub mod b;\n";
    let got = apply(&snap, &parse(EditFormat::Udiff, new_file).expect("parses")).expect("applies");
    assert_eq!(got["src/new.rs"], "pub mod a;\npub mod b;\n");
    let delete =
        "--- a/src/lib.rs\t2026-01-01\n+++ /dev/null\n@@ -1,5 +0,0 @@\n-pub fn f() -> u32 {\n";
    let got = apply(&snap, &parse(EditFormat::Udiff, delete).expect("parses")).expect("applies");
    assert!(got.is_empty());

    for bad in [
        "@@ -1 +1 @@\n-a\n+b\n",
        "--- a/x\n@@ -1 +1 @@\n-a\n+b\n",
        "--- a/x\n+++ b/x\nno hunk\n",
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n a\n",
    ] {
        assert!(parse(EditFormat::Udiff, bad).is_err(), "{bad}");
    }
}

/// FALSIFY-ACE-EDIT-004: apply is all or nothing. A later edit that fails leaves no partial
/// result, the caller's snapshot is unchanged, and the adapter reports NotApplied.
#[test]
fn falsify_ace_edit_004_atomic() {
    let snap = snapshot();
    let raw = format!(
        "{}{}",
        sr("src/lib.rs", "    0", "    42"),
        sr("src/lib.rs", "absent", "x")
    );
    let edits = parse(EditFormat::SearchReplace, &raw).expect("parses");
    assert_eq!(
        apply(&snap, &edits),
        Err(ApplyError::NoMatch("src/lib.rs".into()))
    );
    assert_eq!(snap, snapshot());
    let a = adapt(EditFormat::SearchReplace, &raw, &snap);
    assert_eq!((a.outcome, a.files), (EditOutcome::NotApplied, None));
    assert!(a.error.is_some_and(|e| e.contains("not found")));
}

/// FALSIFY-ACE-EDIT-005: an edit may only touch relative, normalized paths outside `.git`.
#[test]
fn falsify_ace_edit_005_path_confined() {
    for bad in [
        "",
        "/etc/passwd",
        "../outside",
        "src/../../outside",
        "src//lib.rs",
        "./src/lib.rs",
        "src/.",
        ".git/config",
        "vendor/.git/hooks/pre-commit",
        "src\\lib.rs",
        "a\nb",
    ] {
        assert_eq!(
            check_path(bad),
            Err(ApplyError::BadPath(bad.into())),
            "{bad:?}"
        );
        let e = [Edit::Write {
            path: bad.into(),
            content: "x".into(),
        }];
        assert!(apply(&snapshot(), &e).is_err(), "{bad:?}");
    }
    for good in ["src/lib.rs", "a.b/c..d", ".github/x.yml", "README"] {
        assert_eq!(check_path(good), Ok(()), "{good}");
    }
    let raw = "../escape.rs\n```\npwned\n```\n";
    assert_eq!(
        adapt(EditFormat::Whole, raw, &snapshot()).outcome,
        EditOutcome::NotApplied
    );
}

/// FALSIFY-ACE-EDIT-006: hidden tests are laid over the edited tree last; an edit that
/// writes a hidden test path does not change what runs.
#[test]
fn falsify_ace_edit_006_hidden_tests_win() {
    let hidden = files(&[("tests/hidden.rs", "assert_eq!(f(), 42);\n")]);
    let raw = "tests/hidden.rs\n```rust\n// nothing to see\n```\n";
    let a = adapt(EditFormat::Whole, raw, &snapshot());
    let edited = a.files.expect("applies");
    assert_eq!(edited["tests/hidden.rs"], "// nothing to see\n");
    let run = overlay(&edited, &hidden);
    assert_eq!(run["tests/hidden.rs"], hidden["tests/hidden.rs"]);
    assert_eq!(run["src/lib.rs"], snapshot()["src/lib.rs"]);
}

/// FALSIFY-ACE-EDIT-007: the adapter's outcome is the verdict's edit outcome: blank or
/// block-free output is EMPTY, non-empty unparsable output is a format error, never EMPTY.
#[test]
fn falsify_ace_edit_007_outcome_matches_verdict() {
    let snap = snapshot();
    let table = [
        (EditFormat::SearchReplace, String::new(), EditOutcome::Empty),
        (
            EditFormat::SearchReplace,
            " \n\t\n".to_string(),
            EditOutcome::Empty,
        ),
        (
            EditFormat::SearchReplace,
            "I would change f.".to_string(),
            EditOutcome::Empty,
        ),
        (
            EditFormat::SearchReplace,
            format!("src/lib.rs\n{SEARCH}\n    0\n"),
            EditOutcome::Unparsable,
        ),
        (
            EditFormat::SearchReplace,
            sr("src/lib.rs", "    7", "    42"),
            EditOutcome::NotApplied,
        ),
        (
            EditFormat::SearchReplace,
            sr("src/lib.rs", "    0", "    42"),
            EditOutcome::Applied,
        ),
        (
            EditFormat::Whole,
            "src/lib.rs\n```\nunterminated\n".to_string(),
            EditOutcome::Unparsable,
        ),
        (
            EditFormat::Whole,
            "```\nno path line\n```\n".to_string(),
            EditOutcome::Unparsable,
        ),
        (
            EditFormat::Agentic,
            "{\"path\":\"a.rs\"}".to_string(),
            EditOutcome::Unparsable,
        ),
        (
            EditFormat::Agentic,
            "{\"path\":\"a.rs\",\"content\":\"x\",\"mode\":1}".to_string(),
            EditOutcome::Unparsable,
        ),
        (
            EditFormat::Agentic,
            "{\"path\":\"a.rs\",\"content\":\"x\"}\n".to_string(),
            EditOutcome::Applied,
        ),
        (
            EditFormat::Udiff,
            "Looks fine to me.".to_string(),
            EditOutcome::Empty,
        ),
    ];
    for (format, raw, want) in table {
        let a = adapt(format, &raw, &snap);
        assert_eq!(a.outcome, want, "{format:?} {raw:?}");
        assert_eq!(
            a.outcome,
            classify_edit(
                &raw,
                parse(format, &raw).ok().map(|e| e.len()),
                a.files.is_some()
            )
        );
    }
}

#[test]
fn whole_parse_table() {
    let raw = "First:\n\nsrc/a.rs\n```rust\nfn a() {}\n```\n\nsrc/b.md\n````markdown\n```\ncode\n```\n````\n";
    let e = parse(EditFormat::Whole, raw).expect("parses");
    assert_eq!(
        e,
        vec![
            Edit::Write {
                path: "src/a.rs".into(),
                content: "fn a() {}\n".into()
            },
            Edit::Write {
                path: "src/b.md".into(),
                content: "```\ncode\n```\n".into()
            },
        ]
    );
    assert_eq!(
        parse(EditFormat::Whole, "src/a.rs\n```\n```\n").expect("parses")[0],
        Edit::Write {
            path: "src/a.rs".into(),
            content: "\n".into()
        }
    );
    assert_eq!(
        parse(EditFormat::Whole, "a fix for a.rs\n```\nx\n```\n"),
        Err(ParseError::NoPath(2))
    );
}

/// A file of distinct numbered lines, so any span of it occurs exactly once.
fn numbered(n: usize, salt: &str) -> Vec<String> {
    (0..n).map(|i| format!("line {i} {salt}")).collect()
}

fn as_text(lines: &[String]) -> String {
    let mut s = lines.join("\n");
    s.push('\n');
    s
}

proptest! {
    /// FALSIFY-ACE-EDIT-008: rendering a known change in each format and applying it gives
    /// exactly the intended file.
    #[test]
    fn falsify_ace_edit_008_round_trip(
        n in 3usize..30,
        start in 0usize..30,
        len in 1usize..5,
        new in proptest::collection::vec("[a-z ]{0,12}", 0..4),
        salt in "[a-z]{0,6}",
    ) {
        let a = numbered(n, &salt);
        let start = start % n;
        let end = (start + len).min(n);
        let new: Vec<String> = new.iter().map(|l| format!("new {l}")).collect();
        let b: Vec<String> = a[..start].iter().chain(&new).chain(&a[end..]).cloned().collect();
        prop_assume!(!b.is_empty());
        let snap = files(&[("m.rs", &as_text(&a))]);
        let want = as_text(&b);

        let whole = format!("m.rs\n```\n{}```\n", want);
        let search = sr("m.rs", &a[start..end].join("\n"), &new.join("\n"));
        let mut diff = String::from("--- a/m.rs\n+++ b/m.rs\n@@ -1 +1 @@\n");
        for l in &a[start..end] {
            diff.push_str(&format!("-{l}\n"));
        }
        for l in &new {
            diff.push_str(&format!("+{l}\n"));
        }
        let agentic = serde_json::to_string(&files(&[("path", "m.rs"), ("content", &want)])).expect("serializes");
        let raws = [
            (EditFormat::Whole, whole),
            (EditFormat::SearchReplace, search),
            (EditFormat::Udiff, diff),
            (EditFormat::Agentic, agentic),
        ];
        for (format, raw) in raws {
            let got = adapt(format, &raw, &snap);
            prop_assert_eq!(got.files.map(|f| f["m.rs"].clone()), Some(want.clone()), "{:?}", format);
        }
    }
}
