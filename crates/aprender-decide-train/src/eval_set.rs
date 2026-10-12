//! `eval_set_in_distribution`: the demo eval set as a pure function (LT-8).
//!
//! ```text
//! E = dedup_first_nfc_trim_ws([r in validation ++ train (file order) :
//!       r.id not in shots
//!       AND r.id not in exclusions.excluded_train_ids
//!       AND (split(r), r.id) not in union(exclusions.groups[*].members)])
//! normsha(E) ∩ normsha(shots) = ∅   (overlap refused, not dropped)
//! |E| == expect_rows AND class_counts(E) == expect_class_counts
//! ```
//!
//! The order and the dedup rule are part of the definition: the first occurrence is
//! kept, in validation-then-train file order. "The same text" is the judge's
//! `nfc-trim-ws-v1` hash ([`aprender_decide::verify::normalized_sha256`]), so the
//! trainer and the verifier hash a row the same way.
//!
//! The expected counts are parameters, never constants here: they come from the
//! benchmark contract, which this crate does not copy.

use std::collections::HashSet;
use std::fmt;

use aprender_decide::verify::normalized_sha256;

/// Which source file a row came from. Group members name a row by `(split, id)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Split {
    /// The dataset's validation file, read first.
    Validation,
    /// The dataset's train file, read second.
    Train,
}

impl fmt::Display for Split {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Validation => "validation",
            Self::Train => "train",
        })
    }
}

/// One labelled row of a source split, in file order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRow {
    /// The row's id, unique across splits (e.g. `train:70`).
    pub id: String,
    /// The row's text, as read.
    pub text: String,
    /// The class index, in the task's criteria order.
    pub label: usize,
}

/// The benchmark's exclusions: excluded train ids and the members of every group.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Exclusions {
    /// `exclusions.excluded_train_ids`.
    pub excluded_train_ids: Vec<String>,
    /// `union(exclusions.groups[*].members)`, each as `(split, id)`.
    pub group_members: Vec<(Split, String)>,
}

/// Why no eval set was produced. Every variant names what it refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalSetError {
    /// A kept row has the same normalized text as a shot. Refused, never dropped.
    ShotOverlap {
        /// The kept row's split.
        split: Split,
        /// The kept row's id.
        id: String,
        /// The shot it collides with.
        shot_id: String,
        /// The shared `nfc-trim-ws-v1` sha256.
        normsha: String,
    },
    /// A row's label is not a class index.
    LabelOutOfRange {
        /// The row's split.
        split: Split,
        /// The row's id.
        id: String,
        /// Its label.
        label: usize,
        /// The number of classes.
        classes: usize,
    },
    /// `|E|` is not the expected row count.
    RowCount {
        /// Expected.
        expected: usize,
        /// Derived.
        got: usize,
    },
    /// `class_counts(E)` is not the expected class counts.
    ClassCounts {
        /// Expected.
        expected: Vec<usize>,
        /// Derived.
        got: Vec<usize>,
    },
}

impl fmt::Display for EvalSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShotOverlap {
                split,
                id,
                shot_id,
                normsha,
            } => write!(
                f,
                "eval row {split} {id} has the normalized text of shot {shot_id} (normsha {normsha}); overlap is refused"
            ),
            Self::LabelOutOfRange {
                split,
                id,
                label,
                classes,
            } => write!(f, "row {split} {id} has label {label}, not one of {classes} classes"),
            Self::RowCount { expected, got } => {
                write!(f, "eval set has {got} rows, expected {expected}")
            }
            Self::ClassCounts { expected, got } => {
                write!(f, "eval set class counts {got:?}, expected {expected:?}")
            }
        }
    }
}

impl std::error::Error for EvalSetError {}

/// Derives the demo eval set from the source splits, the shots and the exclusions.
///
/// Rows are taken in validation-then-train file order. A row is dropped when its id is
/// a shot id or an excluded train id, or when `(split, id)` is a group member. Of rows
/// sharing a normalized text, the first is kept. The result is then refused, not
/// trimmed, if any kept row shares a normalized text with a shot, or if its row count
/// or class counts differ from `expect_rows` / `expect_class_counts` (whose length is
/// the number of classes).
///
/// # Errors
///
/// [`EvalSetError`] names the overlapping row and shot, the out-of-range label, or the
/// count that does not match.
pub fn eval_set_in_distribution(
    validation: &[SourceRow],
    train: &[SourceRow],
    shots: &[SourceRow],
    exclusions: &Exclusions,
    expect_rows: usize,
    expect_class_counts: &[usize],
) -> Result<Vec<SourceRow>, EvalSetError> {
    let rule = Rule::new(shots, exclusions);
    let ordered = validation
        .iter()
        .map(|r| (Split::Validation, r))
        .chain(train.iter().map(|r| (Split::Train, r)));

    let mut seen: HashSet<String> = HashSet::new();
    let mut kept: Vec<SourceRow> = Vec::new();
    let mut counts = vec![0_usize; expect_class_counts.len()];
    for (split, row) in ordered {
        if rule.drops(split, &row.id) {
            continue;
        }
        let normsha = normalized_sha256(&row.text);
        if !seen.insert(normsha.clone()) {
            continue;
        }
        rule.refuse_overlap(split, row, normsha)?;
        count_label(&mut counts, split, row)?;
        kept.push(row.clone());
    }
    check_counts(kept.len(), &counts, expect_rows, expect_class_counts)?;
    Ok(kept)
}

/// The shot ids, exclusions and shot hashes, indexed once.
struct Rule<'a> {
    shot_ids: HashSet<&'a str>,
    excluded: HashSet<&'a str>,
    members: HashSet<(Split, &'a str)>,
    shot_hashes: Vec<(String, &'a str)>,
}

impl<'a> Rule<'a> {
    fn new(shots: &'a [SourceRow], exclusions: &'a Exclusions) -> Self {
        Self {
            shot_ids: shots.iter().map(|s| s.id.as_str()).collect(),
            excluded: exclusions
                .excluded_train_ids
                .iter()
                .map(String::as_str)
                .collect(),
            members: exclusions
                .group_members
                .iter()
                .map(|(split, id)| (*split, id.as_str()))
                .collect(),
            shot_hashes: shots
                .iter()
                .map(|s| (normalized_sha256(&s.text), s.id.as_str()))
                .collect(),
        }
    }

    /// A shot id, an excluded train id, or a group member's `(split, id)`.
    fn drops(&self, split: Split, id: &str) -> bool {
        self.shot_ids.contains(id)
            || self.excluded.contains(id)
            || self.members.contains(&(split, id))
    }

    /// A kept row with a shot's normalized text is refused, never dropped.
    fn refuse_overlap(
        &self,
        split: Split,
        row: &SourceRow,
        normsha: String,
    ) -> Result<(), EvalSetError> {
        match self.shot_hashes.iter().find(|(h, _)| *h == normsha) {
            Some((_, shot_id)) => Err(EvalSetError::ShotOverlap {
                split,
                id: row.id.clone(),
                shot_id: (*shot_id).to_string(),
                normsha,
            }),
            None => Ok(()),
        }
    }
}

fn count_label(counts: &mut [usize], split: Split, row: &SourceRow) -> Result<(), EvalSetError> {
    let classes = counts.len();
    let slot = counts
        .get_mut(row.label)
        .ok_or_else(|| EvalSetError::LabelOutOfRange {
            split,
            id: row.id.clone(),
            label: row.label,
            classes,
        })?;
    *slot += 1;
    Ok(())
}

fn check_counts(
    rows: usize,
    counts: &[usize],
    expect_rows: usize,
    expect_class_counts: &[usize],
) -> Result<(), EvalSetError> {
    if rows != expect_rows {
        return Err(EvalSetError::RowCount {
            expected: expect_rows,
            got: rows,
        });
    }
    if counts != expect_class_counts {
        return Err(EvalSetError::ClassCounts {
            expected: expect_class_counts.to_vec(),
            got: counts.to_vec(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "eval_set_tests.rs"]
mod tests;
