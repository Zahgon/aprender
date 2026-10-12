//! LT-8 tests on a synthetic fixture: planted shots, exclusions, group members and
//! duplicates, no dataset text. Mutants they must kill: a shot left in the pool, a
//! duplicate kept.

use super::*;

fn row(id: &str, text: &str, label: usize) -> SourceRow {
    SourceRow {
        id: id.to_string(),
        text: text.to_string(),
        label,
    }
}

struct Fixture {
    validation: Vec<SourceRow>,
    train: Vec<SourceRow>,
    shots: Vec<SourceRow>,
    exclusions: Exclusions,
}

/// Kept: validation:0 (0), validation:1 (1), train:1 (2), train:4 (1) -> 4 rows, [1, 2, 1].
fn fixture() -> Fixture {
    Fixture {
        validation: vec![
            row("validation:0", "synthetic alpha", 0),
            row("validation:1", "synthetic bravo", 1),
            // group member -> dropped
            row("validation:2", "synthetic charlie", 1),
            // planted shot -> dropped by id
            row("validation:3", "synthetic shot one", 2),
        ],
        train: vec![
            // duplicate of validation:0 after nfc-trim-ws -> the first is kept
            row("train:0", "  synthetic \t alpha ", 0),
            row("train:1", "synthetic delta", 2),
            // excluded train id -> dropped
            row("train:2", "synthetic echo", 0),
            // group member -> dropped
            row("train:3", "synthetic foxtrot", 1),
            row("train:4", "synthetic golf", 1),
            // planted shot -> dropped by id
            row("train:5", "synthetic shot two", 0),
        ],
        shots: vec![
            row("validation:3", "synthetic shot one", 2),
            row("train:5", "synthetic shot two", 0),
        ],
        exclusions: Exclusions {
            excluded_train_ids: vec!["train:2".to_string()],
            group_members: vec![
                (Split::Train, "train:3".to_string()),
                (Split::Validation, "validation:2".to_string()),
            ],
        },
    }
}

fn run(f: &Fixture, rows: usize, counts: &[usize]) -> Result<Vec<SourceRow>, EvalSetError> {
    eval_set_in_distribution(
        &f.validation,
        &f.train,
        &f.shots,
        &f.exclusions,
        rows,
        counts,
    )
}

fn ids(rows: &[SourceRow]) -> Vec<&str> {
    rows.iter().map(|r| r.id.as_str()).collect()
}

#[test]
fn derives_the_eval_set_in_file_order() {
    let f = fixture();
    let e = run(&f, 4, &[1, 2, 1]).expect("the fixture derives");
    assert_eq!(
        ids(&e),
        ["validation:0", "validation:1", "train:1", "train:4"]
    );
}

/// Mutant "one shot left in the pool": a shot that is not dropped by id collides with
/// itself and the set is refused, so this test sees an Err.
#[test]
fn shots_are_dropped_by_id() {
    let f = fixture();
    let e = run(&f, 4, &[1, 2, 1]).expect("shots are dropped, not refused");
    for shot in &f.shots {
        assert!(!ids(&e).contains(&shot.id.as_str()), "{} kept", shot.id);
    }
}

/// Mutant "a duplicate kept": train:0 would be a fifth row.
#[test]
fn a_duplicate_keeps_only_its_first_occurrence() {
    let f = fixture();
    let e = run(&f, 4, &[1, 2, 1]).expect("the fixture derives");
    assert!(ids(&e).contains(&"validation:0"));
    assert!(!ids(&e).contains(&"train:0"));
    let hashes: HashSet<String> = e.iter().map(|r| normalized_sha256(&r.text)).collect();
    assert_eq!(hashes.len(), e.len());
}

#[test]
fn nfc_variants_are_duplicates() {
    let mut f = fixture();
    f.validation.push(row("validation:4", "caf\u{e9}", 0));
    f.train.push(row("train:6", "cafe\u{301}", 0));
    let e = run(&f, 5, &[2, 2, 1]).expect("the NFC pair collapses to one row");
    assert!(ids(&e).contains(&"validation:4"));
    assert!(!ids(&e).contains(&"train:6"));
}

#[test]
fn excluded_train_ids_and_group_members_are_dropped() {
    let f = fixture();
    let e = run(&f, 4, &[1, 2, 1]).expect("the fixture derives");
    for gone in ["train:2", "train:3", "validation:2"] {
        assert!(!ids(&e).contains(&gone), "{gone} kept");
    }
}

#[test]
fn group_membership_is_per_split() {
    let mut f = fixture();
    // The right id under the wrong split is not a member.
    f.exclusions.group_members = vec![(Split::Train, "validation:2".to_string())];
    let e = run(&f, 6, &[1, 4, 1]).expect("validation:2 and train:3 stay");
    assert!(ids(&e).contains(&"validation:2"));
    assert!(ids(&e).contains(&"train:3"));
}

#[test]
fn shot_text_under_another_id_is_refused_not_dropped() {
    let mut f = fixture();
    f.train.push(row("train:6", "synthetic   shot ONE", 2));
    f.train.push(row("train:7", " synthetic shot one", 2));
    let err = run(&f, 5, &[1, 2, 2]).expect_err("the overlap is refused");
    assert_eq!(
        err,
        EvalSetError::ShotOverlap {
            split: Split::Train,
            id: "train:7".to_string(),
            shot_id: "validation:3".to_string(),
            normsha: normalized_sha256("synthetic shot one"),
        }
    );
    assert!(err.to_string().contains("train:7"));
}

#[test]
fn normalization_does_not_casefold() {
    let mut f = fixture();
    f.train.push(row("train:6", "Synthetic Alpha", 0));
    let e = run(&f, 5, &[2, 2, 1]).expect("a case variant is a different text");
    assert!(ids(&e).contains(&"train:6"));
}

#[test]
fn a_row_count_mismatch_is_refused() {
    let f = fixture();
    assert_eq!(
        run(&f, 5, &[1, 2, 1]),
        Err(EvalSetError::RowCount {
            expected: 5,
            got: 4
        })
    );
}

#[test]
fn a_class_count_mismatch_is_refused() {
    let f = fixture();
    assert_eq!(
        run(&f, 4, &[2, 1, 1]),
        Err(EvalSetError::ClassCounts {
            expected: vec![2, 1, 1],
            got: vec![1, 2, 1]
        })
    );
}

#[test]
fn a_label_outside_the_classes_is_refused() {
    let f = fixture();
    let err = run(&f, 4, &[1, 2]).expect_err("label 2 with two classes");
    assert_eq!(
        err,
        EvalSetError::LabelOutOfRange {
            split: Split::Train,
            id: "train:1".to_string(),
            label: 2,
            classes: 2,
        }
    );
}

#[test]
fn the_function_is_pure() {
    let f = fixture();
    let a = run(&f, 4, &[1, 2, 1]);
    let b = run(&f, 4, &[1, 2, 1]);
    assert_eq!(a, b);
}
