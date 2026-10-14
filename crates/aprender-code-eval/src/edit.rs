//! Edit-format adapters (contracts/apr-code-eval-edit-v1.yaml, ACE-EDIT-001..008).
//!
//! A model's output becomes a list of [`Edit`]s under the task's [`EditFormat`], and the
//! edits apply to the snapshot all or nothing. Nothing is repaired: a block whose search
//! text is not found exactly once, a path that escapes the work tree, or a fence that never
//! closes is a format error (ACE-VER-005), never a guess.
//!
//! Formats (docs/specifications/apr-code-eval-v1.md §8):
//! - `whole`: a path line, then a fenced block with the full new file.
//! - `search_replace`: Aider's `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` blocks, each
//!   after a path line.
//! - `udiff`: unified diff. Hunks match by content, exactly once; line numbers are ignored.
//! - `agentic`: the tool loop's write log, one JSON object `{"path", "content"}` per line.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::verdict::{classify_edit, EditOutcome};

/// File path → content.
pub type Files = BTreeMap<String, String>;

/// ACE-EDIT-001: the closed set of edit formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditFormat {
    /// The full new file, fenced, after its path.
    Whole,
    /// Aider's SEARCH/REPLACE blocks.
    SearchReplace,
    /// Unified diff.
    Udiff,
    /// The tool loop's writes, as JSON lines.
    Agentic,
}

/// One change to one file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Edit {
    /// Set the file to `content`, creating it if needed.
    Write {
        /// Target path.
        path: String,
        /// New content.
        content: String,
    },
    /// Replace the one occurrence of `search` with `replace`. An empty `search` creates a
    /// file that does not exist yet; an empty `replace` removes the lines.
    Replace {
        /// Target path.
        path: String,
        /// Whole lines to find, without the final newline.
        search: String,
        /// Whole lines to put in their place, without the final newline.
        replace: String,
    },
    /// Remove the file.
    Delete {
        /// Target path.
        path: String,
    },
}

impl Edit {
    /// The path this edit touches.
    #[must_use]
    pub fn path(&self) -> &str {
        match self {
            Edit::Write { path, .. } | Edit::Replace { path, .. } | Edit::Delete { path } => path,
        }
    }
}

/// The output does not parse under its format.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParseError {
    /// A fence or block opened and never closed.
    #[error("line {0}: {1} never closed")]
    Unterminated(usize, &'static str),
    /// A block or fence has no path line before it, or the path line is not a path.
    #[error("line {0}: no path for this block")]
    NoPath(usize),
    /// A marker or diff line appeared where it cannot.
    #[error("line {0}: unexpected {1}")]
    Unexpected(usize, &'static str),
    /// An `agentic` log line is not `{"path", "content"}`.
    #[error("line {0}: bad write record: {1}")]
    BadRecord(usize, String),
}

/// The edits parsed but do not apply.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ApplyError {
    /// ACE-EDIT-005: the path is absolute, not normalized, or inside `.git`.
    #[error("path {0:?} is outside the work tree")]
    BadPath(String),
    /// The file to change does not exist.
    #[error("{0}: no such file")]
    Missing(String),
    /// ACE-EDIT-002: the search text is not in the file.
    #[error("{0}: search text not found")]
    NoMatch(String),
    /// ACE-EDIT-002: the search text is in the file more than once.
    #[error("{0}: search text found {1} times")]
    Ambiguous(String, usize),
    /// An empty search on a file that already exists.
    #[error("{0}: empty search on an existing file")]
    Exists(String),
}

/// ACE-EDIT-005: a relative, normalized path that stays inside the work tree.
pub fn check_path(path: &str) -> Result<(), ApplyError> {
    let bad = path.is_empty()
        || path.starts_with('/')
        || path.contains(['\\', '\0', '\n', '\r'])
        || path
            .split('/')
            .any(|c| c.is_empty() || c == "." || c == ".." || c == ".git");
    if bad {
        Err(ApplyError::BadPath(path.to_string()))
    } else {
        Ok(())
    }
}

fn is_path_line(line: &str) -> bool {
    !line.is_empty() && !line.contains(char::is_whitespace) && !line.starts_with("```")
}

/// The fence that opens on `line`, i.e. its run of backticks.
fn fence(line: &str) -> Option<&str> {
    let n = line.bytes().take_while(|&b| b == b'`').count();
    (n >= 3).then(|| &line[..n])
}

fn parse_whole(raw: &str) -> Result<Vec<Edit>, ParseError> {
    let lines: Vec<&str> = raw.lines().collect();
    let mut edits = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let Some(open) = fence(lines[i]) else {
            i += 1;
            continue;
        };
        let path = lines[..i]
            .iter()
            .rev()
            .find(|l| !l.trim().is_empty())
            .map(|l| l.trim());
        let path = path
            .filter(|p| is_path_line(p))
            .ok_or(ParseError::NoPath(i + 1))?;
        let close = lines[i + 1..].iter().position(|l| l.trim_end() == open);
        let close = i + 1 + close.ok_or(ParseError::Unterminated(i + 1, "fence"))?;
        let mut content = lines[i + 1..close].join("\n");
        content.push('\n');
        edits.push(Edit::Write {
            path: path.to_string(),
            content,
        });
        i = close + 1;
    }
    Ok(edits)
}

const SEARCH: &str = "<<<<<<< SEARCH";
const DIVIDER: &str = "=======";
const REPLACE: &str = ">>>>>>> REPLACE";

fn parse_search_replace(raw: &str) -> Result<Vec<Edit>, ParseError> {
    let lines: Vec<&str> = raw
        .lines()
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect();
    let mut edits = Vec::new();
    let mut path: Option<&str> = None;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let marker = line.trim_end();
        if marker == DIVIDER || marker == REPLACE {
            return Err(ParseError::Unexpected(i + 1, "marker outside a block"));
        }
        if marker != SEARCH {
            if fence(line).is_none() && !marker.is_empty() {
                path = Some(line.trim()).filter(|p| is_path_line(p));
            }
            i += 1;
            continue;
        }
        let p = path.ok_or(ParseError::NoPath(i + 1))?;
        let (search, after) = block_part(&lines, i + 1, DIVIDER)?;
        let (replace, after) = block_part(&lines, after, REPLACE)?;
        edits.push(Edit::Replace {
            path: p.to_string(),
            search,
            replace,
        });
        i = after;
    }
    Ok(edits)
}

/// Lines from `start` up to the `end` marker, and the index after the marker.
fn block_part(
    lines: &[&str],
    start: usize,
    end: &'static str,
) -> Result<(String, usize), ParseError> {
    for (j, line) in lines.iter().enumerate().skip(start) {
        let marker = line.trim_end();
        if marker == end {
            return Ok((lines[start..j].join("\n"), j + 1));
        }
        if [SEARCH, DIVIDER, REPLACE].contains(&marker) {
            return Err(ParseError::Unexpected(j + 1, "marker inside a block"));
        }
    }
    Err(ParseError::Unterminated(start, end))
}

fn diff_path(line: &str, prefix: &str) -> Option<String> {
    let rest = line.strip_prefix(prefix)?;
    let rest = rest.split('\t').next().unwrap_or(rest).trim_end();
    if rest == "/dev/null" {
        return Some(String::new());
    }
    let git = if prefix.starts_with('-') { "a/" } else { "b/" };
    let p = rest.strip_prefix(git).unwrap_or(rest);
    Some(p.to_string())
}

/// One hunk's old and new text, as whole lines without the final newline.
fn hunk_edit(path: &str, body: &[&str]) -> Edit {
    let old: Vec<&str> = body
        .iter()
        .filter(|l| !l.starts_with('+'))
        .map(|l| l.get(1..).unwrap_or(""))
        .collect();
    let new: Vec<&str> = body
        .iter()
        .filter(|l| !l.starts_with('-'))
        .map(|l| l.get(1..).unwrap_or(""))
        .collect();
    Edit::Replace {
        path: path.to_string(),
        search: old.join("\n"),
        replace: new.join("\n"),
    }
}

fn is_hunk_line(line: &str) -> bool {
    line.is_empty()
        || (line.starts_with([' ', '-', '+'])
            && !line.starts_with("--- ")
            && !line.starts_with("+++ "))
}

/// The `@@` hunks from `j` on: each body without `\ No newline` lines and without trailing
/// blank lines (prose after a diff), and the index after the last hunk.
fn collect_hunks<'a>(
    lines: &[&'a str],
    mut j: usize,
) -> Result<(Vec<Vec<&'a str>>, usize), ParseError> {
    let mut hunks = Vec::new();
    while lines.get(j).is_some_and(|l| l.starts_with("@@")) {
        let end = (j + 1..lines.len())
            .find(|&k| !is_hunk_line(lines[k]))
            .unwrap_or(lines.len());
        let mut body: Vec<&str> = lines[j + 1..end]
            .iter()
            .copied()
            .filter(|l| !l.starts_with('\\'))
            .collect();
        while body.last().is_some_and(|l| l.is_empty()) {
            body.pop();
        }
        if !body.iter().any(|l| l.starts_with(['-', '+'])) {
            return Err(ParseError::Unexpected(j + 1, "hunk with no change"));
        }
        hunks.push(body);
        j = end;
    }
    Ok((hunks, j))
}

/// One file section of a diff, starting at its `--- ` line. Returns the edits and the
/// index after the section.
fn parse_diff_file(lines: &[&str], i: usize) -> Result<(Vec<Edit>, usize), ParseError> {
    let old = diff_path(lines[i], "--- ").unwrap_or_default();
    let new = lines
        .get(i + 1)
        .and_then(|l| diff_path(l, "+++ "))
        .ok_or(ParseError::Unexpected(i + 1, "--- without +++"))?;
    let (hunks, j) = collect_hunks(lines, i + 2)?;
    if hunks.is_empty() {
        return Err(ParseError::Unexpected(i + 1, "file header with no hunk"));
    }
    let edits = match (old.is_empty(), new.is_empty()) {
        (_, true) => vec![Edit::Delete { path: old }],
        (true, false) => vec![Edit::Write {
            path: new,
            content: hunk_text(&hunks, '+'),
        }],
        (false, false) => hunks.iter().map(|h| hunk_edit(&new, h)).collect(),
    };
    Ok((edits, j))
}

fn hunk_text(hunks: &[Vec<&str>], sign: char) -> String {
    let mut s: String = hunks
        .iter()
        .flatten()
        .filter_map(|l| l.strip_prefix(sign))
        .collect::<Vec<_>>()
        .join("\n");
    s.push('\n');
    s
}

fn parse_udiff(raw: &str) -> Result<Vec<Edit>, ParseError> {
    let lines: Vec<&str> = raw
        .lines()
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect();
    let mut edits = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].starts_with("--- ") {
            let (mut file_edits, next) = parse_diff_file(&lines, i)?;
            edits.append(&mut file_edits);
            i = next;
        } else if lines[i].starts_with("@@") {
            return Err(ParseError::NoPath(i + 1));
        } else {
            i += 1;
        }
    }
    Ok(edits)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteRecord {
    path: String,
    content: String,
}

fn parse_agentic(raw: &str) -> Result<Vec<Edit>, ParseError> {
    let records = raw
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty());
    records
        .map(|(n, l)| {
            let r: WriteRecord =
                serde_json::from_str(l).map_err(|e| ParseError::BadRecord(n + 1, e.to_string()))?;
            Ok(Edit::Write {
                path: r.path,
                content: r.content,
            })
        })
        .collect()
}

/// ACE-EDIT-001: parse a model output under `format`. Total: every input gives edits or an
/// error, and the same input always gives the same result.
pub fn parse(format: EditFormat, raw: &str) -> Result<Vec<Edit>, ParseError> {
    match format {
        EditFormat::Whole => parse_whole(raw),
        EditFormat::SearchReplace => parse_search_replace(raw),
        EditFormat::Udiff => parse_udiff(raw),
        EditFormat::Agentic => parse_agentic(raw),
    }
}

/// Byte offsets where `needle` occurs in `text` on whole-line boundaries.
fn line_matches(text: &str, needle: &str) -> Vec<usize> {
    let at_start = |i: usize| i == 0 || text.as_bytes()[i - 1] == b'\n';
    let at_end = |i: usize| i == text.len() || text.as_bytes()[i] == b'\n';
    text.match_indices(needle)
        .map(|(i, _)| i)
        .filter(|&i| at_start(i) && at_end(i + needle.len()))
        .collect()
}

fn apply_replace(
    files: &mut Files,
    path: &str,
    search: &str,
    replace: &str,
) -> Result<(), ApplyError> {
    if search.is_empty() {
        if files.contains_key(path) {
            return Err(ApplyError::Exists(path.to_string()));
        }
        files.insert(path.to_string(), format!("{replace}\n"));
        return Ok(());
    }
    let text = files
        .get_mut(path)
        .ok_or_else(|| ApplyError::Missing(path.to_string()))?;
    let hits = line_matches(text, search);
    match hits.as_slice() {
        [] => Err(ApplyError::NoMatch(path.to_string())),
        &[at] => {
            // An empty replacement removes the lines, newline included, as Aider does.
            let mut end = at + search.len();
            if replace.is_empty() && text.as_bytes().get(end) == Some(&b'\n') {
                end += 1;
            }
            text.replace_range(at..end, replace);
            Ok(())
        }
        many => Err(ApplyError::Ambiguous(path.to_string(), many.len())),
    }
}

/// ACE-EDIT-002..005: apply `edits` in order to a copy of `files`. All or nothing: on any
/// error the caller's files are untouched and no partial result exists.
pub fn apply(files: &Files, edits: &[Edit]) -> Result<Files, ApplyError> {
    let mut out = files.clone();
    for edit in edits {
        check_path(edit.path())?;
        match edit {
            Edit::Write { path, content } => {
                out.insert(path.clone(), content.clone());
            }
            Edit::Replace {
                path,
                search,
                replace,
            } => apply_replace(&mut out, path, search, replace)?,
            Edit::Delete { path } => {
                out.remove(path)
                    .ok_or_else(|| ApplyError::Missing(path.clone()))?;
            }
        }
    }
    Ok(out)
}

/// ACE-EDIT-006: what the tests run on. Hidden tests are laid over the edited tree last, so
/// an edit cannot change them.
#[must_use]
pub fn overlay(edited: &Files, hidden: &Files) -> Files {
    let mut out = edited.clone();
    out.extend(hidden.iter().map(|(k, v)| (k.clone(), v.clone())));
    out
}

/// The adapter's answer for one output: the verdict's edit outcome, the edited files when
/// they applied, and the reason when they did not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adapted {
    /// ACE-EDIT-007: the pre(t, out) part of the verdict.
    pub outcome: EditOutcome,
    /// The snapshot with the edits applied, only when `outcome` is `Applied`.
    pub files: Option<Files>,
    /// Why the output did not parse or apply.
    pub error: Option<String>,
}

/// ACE-EDIT-007: parse and apply one output against the snapshot, and classify it the way
/// the verdict does.
#[must_use]
pub fn adapt(format: EditFormat, raw: &str, snapshot: &Files) -> Adapted {
    let edits = match parse(format, raw) {
        Ok(e) => e,
        Err(e) => {
            let outcome = classify_edit(raw, None, false);
            return Adapted {
                outcome,
                files: None,
                error: Some(e.to_string()),
            };
        }
    };
    let applied = apply(snapshot, &edits);
    let outcome = classify_edit(raw, Some(edits.len()), applied.is_ok());
    match applied {
        Ok(files) if outcome == EditOutcome::Applied => Adapted {
            outcome,
            files: Some(files),
            error: None,
        },
        Ok(_) => Adapted {
            outcome,
            files: None,
            error: None,
        },
        Err(e) => Adapted {
            outcome,
            files: None,
            error: Some(e.to_string()),
        },
    }
}

#[cfg(test)]
#[path = "edit_tests.rs"]
mod tests;
