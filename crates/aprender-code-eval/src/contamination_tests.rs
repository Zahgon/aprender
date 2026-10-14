//! FALSIFY-ACE-CON-001..005 (contracts/apr-code-eval-contamination-v1.yaml).

use super::*;
use crate::task::{HiddenTests, Snapshot, TaskKind};

/// Fourteen words: two 13-grams.
const LEAK: &str =
    "let total = items.iter().map(|x| x.price * x.qty).sum::<u64>() + shipping_fee(region);";
/// Its first twelve words.
const LEAK12: &str = "let total = items.iter().map(|x| x.price * x.qty).sum::<u64>() +";

fn diff(context: &str, removed: &str, added: &str) -> String {
    format!(
        "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1,2 +1,2 @@\n {context}\n-{removed}\n+{added}\n"
    )
}

fn task(id: &str, gold: String) -> Task {
    Task {
        id: id.to_string(),
        kind: TaskKind::Function,
        release_date: Some("2025-03-01".into()),
        prompt: "p".into(),
        snapshot: Snapshot::default(),
        hidden_tests: HiddenTests::default(),
        gold,
    }
}

#[test]
fn falsify_ace_con_001_canary() {
    assert!(has_canary(&format!("{{\"canary\": \"{CANARY}\"}}\n")));
    assert!(
        !has_canary("e2d27ba8-525a-4eaa"),
        "half a GUID is not the canary"
    );
    assert_eq!(format!("{}{}", canary_prefix(), canary_secret()), CANARY);
    assert!(
        canary_secret().len() >= 32,
        "the unshown part cannot be guessed"
    );
    assert!(canary_completed(&format!("sure: {}", canary_secret())));
    assert!(
        !canary_completed(canary_prefix()),
        "echoing the prompt is not completing it"
    );
    assert!(!canary_completed("I do not know that string."));
}

#[test]
fn falsify_ace_con_002_planted_leak_flagged() {
    let tasks = vec![
        task(
            "clean",
            diff("fn a() {", "    0", "    unrelated_body(1, 2, 3)"),
        ),
        task("leaked", diff("fn a() {", "    0", &format!("    {LEAK}"))),
    ];
    // The manifest holds the leak with other spacing, case and surroundings.
    let manifest = format!(
        "// vendored\nfn price() -> u64 {{\n{}\n}}\n",
        LEAK.to_uppercase().replace(' ', "   ")
    );
    let index = NgramIndex::build([manifest.as_str()]);
    assert!(!index.is_empty());
    assert_eq!(flag_overlap(&tasks, &index), vec!["leaked".to_string()]);
    let short = NgramIndex::build([LEAK12]);
    assert!(
        flag_overlap(&tasks, &short).is_empty(),
        "a 12-word overlap is under the 13-gram bar"
    );
}

#[test]
fn falsify_ace_con_003_only_added_lines_count() {
    // The leak sits in the context and in the removed line: the model is not asked
    // to write either, so neither is evidence that it memorised the answer.
    let gold = diff(LEAK, LEAK, "    new_body()");
    assert!(!solution_text(&gold).contains("shipping_fee"));
    assert!(
        !solution_text(&gold).contains("src/lib.rs"),
        "no +++ header"
    );
    let index = NgramIndex::build([LEAK]);
    assert!(flag_overlap(&[task("t", gold)], &index).is_empty());
    // Gold that is not a diff (a whole file) counts in full.
    assert!(index.overlaps(&solution_text(LEAK)));
}

#[test]
fn falsify_ace_con_004_unchecked_is_not_clean() {
    let none = Contamination::unchecked();
    assert_eq!(none.overlap_flagged, None);
    assert_eq!(none.canary_completed, None);
    assert_eq!(none.manifest_sha256, None);
    let json = serde_json::to_value(&none).expect("serialize");
    assert!(json["overlap_flagged"].is_null(), "not an empty list");
    let clean = Contamination::unchecked().with_overlap("ab".into(), Vec::new());
    assert_ne!(none, clean, "checked-and-clean differs from unchecked");
    assert_eq!(clean.overlap_flagged, Some(Vec::new()));
    let probed = clean.with_canary("no idea");
    assert_eq!(probed.canary_completed, Some(false));
}

#[test]
fn falsify_ace_con_005_deterministic() {
    let tasks: Vec<Task> = (0..20)
        .map(|i| {
            let body = if i % 3 == 0 {
                LEAK.to_string()
            } else {
                format!("body_{i}()")
            };
            task(&format!("t{i:02}"), diff("fn a() {", "    0", &body))
        })
        .collect();
    let a = flag_overlap(&tasks, &NgramIndex::build([LEAK]));
    let b = flag_overlap(&tasks, &NgramIndex::build([LEAK, LEAK]));
    assert_eq!(a, b);
    let ids: Vec<String> = (0..20).step_by(3).map(|i| format!("t{i:02}")).collect();
    assert_eq!(a, ids, "flags come out in task order");
}
