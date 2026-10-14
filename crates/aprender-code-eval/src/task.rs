//! Task loader and task-set manifest (contracts/apr-code-eval-task-v1.yaml, ACE-TASK-001..006).
//!
//! A manifest is JSON Lines, one task record per line. [`load`] refuses any set that could
//! only produce 0/0, double-count a task, score a task that cannot fail, leak its answer
//! key, or lose its date. A loaded [`TaskSet`] is not yet a [`Suite`]: a task enters a
//! suite only through [`validate`], which runs the gold edit and the null edit in the
//! sandbox (the SWE-bench Verified rule).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// The four task kinds sharing one record (spec §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    /// A real repository at a pinned commit (SWE-bench Verified style).
    Repo,
    /// A signature and docstring with extended hidden tests (EvalPlus style).
    Function,
    /// An exercise edited through a declared edit format (Aider polyglot style).
    Edit,
    /// A container task judged by a check of the final state (Terminal-Bench style).
    Terminal,
}

/// What the agent starts from. Only this and the prompt are ever shown to it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Repository URL, for repo tasks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// Pinned commit, for repo tasks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// Inline files, path → content.
    #[serde(default)]
    pub files: BTreeMap<String, String>,
}

/// The answer key. Never shown to the agent (ACE-TASK-004).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HiddenTests {
    /// FAIL_TO_PASS test ids: must fail before the edit and pass after.
    #[serde(default)]
    pub fail_to_pass: Vec<String>,
    /// PASS_TO_PASS test ids: must pass before and after.
    #[serde(default)]
    pub pass_to_pass: Vec<String>,
    /// Test files the sandbox adds after the edit, path → content.
    #[serde(default)]
    pub files: BTreeMap<String, String>,
}

/// One task record as it appears in the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    /// Unique id within the set.
    pub id: String,
    /// The task kind.
    pub kind: TaskKind,
    /// `YYYY-MM-DD`. Optional in the schema only so a missing date is refused by name.
    #[serde(default)]
    pub release_date: Option<String>,
    /// The instruction shown to the agent.
    pub prompt: String,
    /// The starting state shown to the agent.
    #[serde(default)]
    pub snapshot: Snapshot,
    /// The answer key.
    #[serde(default)]
    pub hidden_tests: HiddenTests,
    /// The reference edit, used only by [`validate`].
    pub gold: String,
}

/// The part of a task the agent may see (ACE-TASK-PO-004).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AgentView<'a> {
    /// The task id.
    pub id: &'a str,
    /// The instruction.
    pub prompt: &'a str,
    /// The starting state.
    pub snapshot: &'a Snapshot,
}

impl Task {
    /// The agent-visible view: id, prompt and snapshot, never the hidden tests or gold.
    #[must_use]
    pub fn agent_view(&self) -> AgentView<'_> {
        AgentView {
            id: &self.id,
            prompt: &self.prompt,
            snapshot: &self.snapshot,
        }
    }

    /// The release date, which [`load`] has proved present and well-formed.
    #[must_use]
    pub fn date(&self) -> &str {
        self.release_date.as_deref().unwrap_or_default()
    }

    /// sha256 of the task's canonical JSON bytes.
    fn sha256(&self) -> [u8; 32] {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        Sha256::digest(bytes).into()
    }
}

/// Load-time and validation-time refusals. Each names the task it is about.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TaskError {
    /// A manifest line is not a task record.
    #[error("manifest line {line}: {msg}")]
    Parse {
        /// 1-based line number.
        line: usize,
        /// The parser's message.
        msg: String,
    },
    /// The set has no tasks (ACE-TASK-001).
    #[error("task set is empty")]
    Empty,
    /// Two records share an id (ACE-TASK-001).
    #[error("duplicate task id {0}")]
    DuplicateId(String),
    /// A task with no FAIL_TO_PASS test cannot fail (ACE-TASK-002).
    #[error("task {0} has no fail_to_pass test")]
    NoFailToPass(String),
    /// A task without a release date (ACE-TASK-006).
    #[error("task {0} has no release_date")]
    Undated(String),
    /// A release date that is not `YYYY-MM-DD`, or is after the load date (ACE-TASK-006).
    #[error("task {id} has release_date {date:?}, not an ISO date on or before the load date")]
    BadDate {
        /// The task.
        id: String,
        /// The offending value.
        date: String,
    },
    /// A hidden test file is also in the snapshot (ACE-TASK-004).
    #[error("task {id}: hidden test file {path} is in the snapshot")]
    HiddenTestInSnapshot {
        /// The task.
        id: String,
        /// The leaked path.
        path: String,
    },
    /// A hidden test file's content appears in the prompt (ACE-TASK-004).
    #[error("task {id}: hidden test file {path} is quoted in the prompt")]
    HiddenTestInPrompt {
        /// The task.
        id: String,
        /// The leaked path.
        path: String,
    },
    /// The gold edit does not resolve the task (ACE-TASK-003).
    #[error("task {0}: the gold edit does not resolve it")]
    GoldFails(String),
    /// The null edit already resolves the task (ACE-TASK-003).
    #[error("task {0}: the null edit resolves it, so it rewards doing nothing")]
    NullResolves(String),
    /// The sandbox could not run a validation (ACE-TASK-003).
    #[error("task {id}: validation run failed: {msg}")]
    Sandbox {
        /// The task.
        id: String,
        /// The sandbox's message.
        msg: String,
    },
}

/// A loaded task set: non-empty, unique ids, every task can fail, dated, no leak.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSet {
    tasks: Vec<Task>,
}

impl TaskSet {
    /// The tasks, in manifest order.
    #[must_use]
    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    /// ACE-TASK-005: sha256 over (id ‖ 0x00 ‖ sha256(task)) in id order, as lowercase hex.
    #[must_use]
    pub fn digest(&self) -> String {
        digest(&self.tasks)
    }
}

/// ACE-TASK-005 over any slice of tasks.
#[must_use]
pub fn digest(tasks: &[Task]) -> String {
    let mut sorted: Vec<&Task> = tasks.iter().collect();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    let mut h = Sha256::new();
    for t in sorted {
        h.update(t.id.as_bytes());
        h.update([0u8]);
        h.update(t.sha256());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn is_leap(y: u32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// A strict `YYYY-MM-DD` with a real calendar day.
fn is_iso_date(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    let num = |r: std::ops::Range<usize>| -> Option<u32> {
        let p = s.get(r)?;
        if p.bytes().all(|c| c.is_ascii_digit()) {
            p.parse().ok()
        } else {
            None
        }
    };
    let (Some(y), Some(m), Some(d)) = (num(0..4), num(5..7), num(8..10)) else {
        return false;
    };
    let days = match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(y) => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days).contains(&d)
}

fn check_task(t: &Task, load_date: &str) -> Result<(), TaskError> {
    if t.hidden_tests.fail_to_pass.is_empty() {
        return Err(TaskError::NoFailToPass(t.id.clone()));
    }
    let date = t
        .release_date
        .as_deref()
        .ok_or_else(|| TaskError::Undated(t.id.clone()))?;
    if !is_iso_date(date) || date > load_date {
        return Err(TaskError::BadDate {
            id: t.id.clone(),
            date: date.to_string(),
        });
    }
    for (path, content) in &t.hidden_tests.files {
        if t.snapshot.files.contains_key(path) {
            return Err(TaskError::HiddenTestInSnapshot {
                id: t.id.clone(),
                path: path.clone(),
            });
        }
        if !content.trim().is_empty() && t.prompt.contains(content.trim()) {
            return Err(TaskError::HiddenTestInPrompt {
                id: t.id.clone(),
                path: path.clone(),
            });
        }
    }
    Ok(())
}

/// Load a JSON Lines manifest. Blank lines are skipped. `load_date` is `YYYY-MM-DD`.
pub fn load(manifest: &str, load_date: &str) -> Result<TaskSet, TaskError> {
    let mut tasks = Vec::new();
    for (i, line) in manifest.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let t: Task = serde_json::from_str(line).map_err(|e| TaskError::Parse {
            line: i + 1,
            msg: e.to_string(),
        })?;
        tasks.push(t);
    }
    if tasks.is_empty() {
        return Err(TaskError::Empty);
    }
    let mut seen = BTreeSet::new();
    for t in &tasks {
        if !seen.insert(t.id.as_str()) {
            return Err(TaskError::DuplicateId(t.id.clone()));
        }
        check_task(t, load_date)?;
    }
    Ok(TaskSet { tasks })
}

/// Runs one edit of one task in the sandbox and says whether it is resolved.
/// The sandbox crate (#5284) implements it; tests plant their own.
pub trait Resolver {
    /// `edit = None` is the null edit (the untouched snapshot).
    fn resolved(&mut self, task: &Task, edit: Option<&str>) -> Result<bool, String>;
}

/// A task that passed gold and null validation. Built only by [`validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Validated(Task);

impl Validated {
    /// The task.
    #[must_use]
    pub fn task(&self) -> &Task {
        &self.0
    }
}

/// ACE-TASK-003: the gold edit resolves the task and the null edit does not. Both runs
/// happen every time; neither is skipped when the other fails.
pub fn validate(task: Task, sandbox: &mut impl Resolver) -> Result<Validated, TaskError> {
    let mut run = |edit: Option<&str>| {
        sandbox
            .resolved(&task, edit)
            .map_err(|msg| TaskError::Sandbox {
                id: task.id.clone(),
                msg,
            })
    };
    let null = run(None)?;
    let gold = run(Some(task.gold.as_str()))?;
    if null {
        return Err(TaskError::NullResolves(task.id.clone()));
    }
    if !gold {
        return Err(TaskError::GoldFails(task.id.clone()));
    }
    Ok(Validated(task))
}

/// Tasks that are all validated (ACE-TASK-PO-003). The only input the driver scores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suite {
    tasks: Vec<Validated>,
    digest: String,
}

impl Suite {
    /// Validate every task of a set; any failure refuses the whole suite.
    pub fn build(set: TaskSet, sandbox: &mut impl Resolver) -> Result<Self, TaskError> {
        let digest = set.digest();
        let tasks = set
            .tasks
            .into_iter()
            .map(|t| validate(t, sandbox))
            .collect::<Result<_, _>>()?;
        Ok(Self { tasks, digest })
    }

    /// The validated tasks.
    #[must_use]
    pub fn tasks(&self) -> &[Validated] {
        &self.tasks
    }

    /// The set digest, for the receipt's `task_set_digest`.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }
}

#[cfg(test)]
#[path = "task_tests.rs"]
mod tests;
