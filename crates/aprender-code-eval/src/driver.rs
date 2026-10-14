//! The `apr code` driver (contracts/apr-code-eval-driver-v1.yaml, ACE-DRV-001..007).
//!
//! The driver puts a model in front of each validated task the way a user would
//! (docs/specifications/apr-code-eval-v1.md §8): the prompt and a working copy of the
//! snapshot go in, an edit comes out, the edit is applied by [`crate::edit`], and the
//! hidden tests run on the result in the sandbox. The model and the sandbox plug in
//! through [`Agent`] and [`TestRunner`], so every rule here is testable with planted ones.
//!
//! - The agent sees only [`Request`]: the task's agent view and the working tree. Hidden
//!   tests, their names and the gold edit are not reachable from it (ACE-DRV-001).
//! - Only an applied edit reaches the sandbox, and the tests run on the edited tree with
//!   the hidden files laid over it last (ACE-DRV-002).
//! - `edit` tasks get a second attempt after a failed first one (Aider practice). The
//!   feedback is built from the label and the adapter's error, never from the sandbox
//!   report (ACE-DRV-003).
//! - A reply over its turn or token budget is TIMEOUT and is not tested (ACE-DRV-004).
//! - A backend failure stops the run; it is never a label (ACE-DRV-005).
//! - Sample i uses seed + i, so the same inputs give the same records apart from wall
//!   time (ACE-DRV-006).

use std::collections::BTreeMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::edit::{adapt, overlay, EditFormat, Files};
use crate::metric::{Sample, TaskResult};
use crate::task::{AgentView, Suite, Task, TaskKind};
use crate::verdict::{verdict, EditOutcome, SandboxReport, SandboxStatus, TestLists, Verdict};

/// ACE-DRV-004: what one attempt may spend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    /// Tool-loop turns.
    pub max_turns: u32,
    /// Generated tokens.
    pub max_tokens: u64,
}

/// One run's settings. They go into the receipt unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    /// The edit format every reply is parsed under.
    pub format: EditFormat,
    /// Per-attempt budget.
    pub budget: Budget,
    /// Base seed; sample i uses `seed + i`.
    pub seed: u64,
    /// Samples per task (the n of pass@k).
    pub n: usize,
}

/// ACE-DRV-001: everything the agent is given for one attempt, and nothing else.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Request<'a> {
    /// The task as the agent may see it.
    pub view: AgentView<'a>,
    /// The working tree: the snapshot, or the tree the first attempt left.
    pub files: &'a Files,
    /// The edit format the reply must use.
    pub format: EditFormat,
    /// 1 or 2.
    pub attempt: u8,
    /// Why the first attempt failed, on attempt 2.
    pub feedback: Option<&'a str>,
    /// This sample's seed.
    pub seed: u64,
    /// This attempt's budget.
    pub budget: Budget,
}

/// The agent's answer to one request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reply {
    /// The text the edit is parsed from.
    pub text: String,
    /// Prompt tokens.
    pub tokens_in: u64,
    /// Generated tokens.
    pub tokens_out: u64,
    /// Tool-loop turns used.
    pub turns: u32,
}

/// The model under test.
pub trait Agent {
    /// Answer one request. An error means the backend failed, not the model.
    fn complete(&mut self, req: &Request<'_>) -> Result<Reply, String>;
}

/// The sandbox, as the driver uses it.
pub trait TestRunner {
    /// Run the task's tests on `tree`. An error is a sandbox failure for this attempt.
    fn run(&mut self, task: &Task, tree: &Files) -> Result<SandboxReport, String>;
}

/// The run cannot be scored.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DriverError {
    /// `n` is zero.
    #[error("n must be at least 1")]
    NoSamples,
    /// ACE-DRV-005: the model backend failed.
    #[error("{id} sample {sample} attempt {attempt}: backend failed: {msg}")]
    Agent {
        /// Task id.
        id: String,
        /// Sample index.
        sample: usize,
        /// Attempt number.
        attempt: u8,
        /// The backend's message.
        msg: String,
    },
}

/// One attempt, as the receipt records it (spec §7 `per_task`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptRecord {
    /// The attempt's label.
    pub verdict: Verdict,
    /// What the adapter found in the reply.
    pub edit: EditOutcome,
    /// The adapter's reason when the edit did not parse or apply.
    pub edit_error: Option<String>,
    /// Tests the sandbox executed.
    pub tests_executed: usize,
    /// Tests that passed.
    pub tests_passed: usize,
    /// FAIL_TO_PASS tests that passed.
    pub f2p_passed: usize,
    /// PASS_TO_PASS tests that did not pass.
    pub p2p_failed: usize,
    /// Prompt tokens.
    pub tokens_in: u64,
    /// Generated tokens.
    pub tokens_out: u64,
    /// Tool-loop turns.
    pub turns: u32,
    /// sha256 of the sandbox report, when the tests ran.
    pub report_sha256: Option<String>,
    /// Wall time of the attempt. The only field that may differ between seeded runs.
    pub wall_ms: u64,
}

/// Every sample of one task; each sample is one or two attempts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskRecord {
    /// Task id.
    pub id: String,
    /// The task's release date.
    pub release_date: String,
    /// Samples in seed order.
    pub samples: Vec<Vec<AttemptRecord>>,
}

/// ACE-DRV-003: the message for attempt 2. Its inputs are the label and the adapter's
/// error; the sandbox report is not one, so no hidden test name or output can reach it.
#[must_use]
pub fn feedback(v: Verdict, edit_error: Option<&str>) -> String {
    match v {
        Verdict::FormatError => format!(
            "Your edit did not apply: {}. Send the edit again in the same format.",
            edit_error.unwrap_or("it did not parse")
        ),
        Verdict::Empty => {
            "Your reply contained no edit. Send the edit in the required format.".into()
        }
        Verdict::Timeout => "The attempt ran out of budget or time.".into(),
        _ => "The change does not make the tests pass yet. Fix the code.".into(),
    }
}

fn not_run() -> SandboxReport {
    SandboxReport {
        status: SandboxStatus::Error("not run".into()),
        killed_by: None,
        outcomes: BTreeMap::new(),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Score one reply against the working tree: the attempt's record, and the edited tree
/// when the edit applied.
fn score(
    task: &Task,
    cfg: &Config,
    tree: &Files,
    reply: &Reply,
    runner: &mut impl TestRunner,
) -> (AttemptRecord, Option<Files>) {
    let a = adapt(cfg.format, &reply.text, tree);
    let hidden = &task.hidden_tests;
    let tests = TestLists {
        f2p: &hidden.fail_to_pass,
        p2p: &hidden.pass_to_pass,
    };
    let mut rec = AttemptRecord {
        verdict: Verdict::Timeout,
        edit: a.outcome,
        edit_error: a.error,
        tests_executed: 0,
        tests_passed: 0,
        f2p_passed: 0,
        p2p_failed: 0,
        tokens_in: reply.tokens_in,
        tokens_out: reply.tokens_out,
        turns: reply.turns,
        report_sha256: None,
        wall_ms: 0,
    };
    if reply.turns > cfg.budget.max_turns || reply.tokens_out > cfg.budget.max_tokens {
        return (rec, None);
    }
    let Some(files) = a.files else {
        rec.verdict = verdict(tests, a.outcome, &not_run());
        return (rec, None);
    };
    let rep = runner
        .run(task, &overlay(&files, &hidden.files))
        .unwrap_or_else(|msg| SandboxReport {
            status: SandboxStatus::Error(msg),
            killed_by: None,
            outcomes: BTreeMap::new(),
        });
    rec.verdict = verdict(tests, a.outcome, &rep);
    rec.tests_executed = rep.tests_executed();
    rec.tests_passed = rep.outcomes.keys().filter(|k| rep.pass(k)).count();
    rec.f2p_passed = hidden.fail_to_pass.iter().filter(|k| rep.pass(k)).count();
    rec.p2p_failed = hidden.pass_to_pass.iter().filter(|k| !rep.pass(k)).count();
    rec.report_sha256 = Some(sha256_hex(
        &serde_json::to_vec(&rep).expect("a sandbox report serializes"),
    ));
    (rec, Some(files))
}

fn run_sample(
    task: &Task,
    cfg: &Config,
    sample: usize,
    agent: &mut impl Agent,
    runner: &mut impl TestRunner,
) -> Result<Vec<AttemptRecord>, DriverError> {
    let attempts: u8 = if task.kind == TaskKind::Edit { 2 } else { 1 };
    let seed = cfg.seed.wrapping_add(sample as u64);
    let mut tree = task.snapshot.files.clone();
    let mut note: Option<String> = None;
    let mut out = Vec::new();
    for attempt in 1..=attempts {
        let start = Instant::now();
        let req = Request {
            view: task.agent_view(),
            files: &tree,
            format: cfg.format,
            attempt,
            feedback: note.as_deref(),
            seed,
            budget: cfg.budget,
        };
        let reply = agent.complete(&req).map_err(|msg| DriverError::Agent {
            id: task.id.clone(),
            sample,
            attempt,
            msg,
        })?;
        let (mut rec, edited) = score(task, cfg, &tree, &reply, runner);
        rec.wall_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        let done = rec.verdict == Verdict::Resolved;
        note = Some(feedback(rec.verdict, rec.edit_error.as_deref()));
        if let Some(files) = edited {
            tree = files;
        }
        out.push(rec);
        if done {
            break;
        }
    }
    Ok(out)
}

/// Run every sample of one task.
pub fn run_task(
    task: &Task,
    cfg: &Config,
    agent: &mut impl Agent,
    runner: &mut impl TestRunner,
) -> Result<TaskRecord, DriverError> {
    if cfg.n == 0 {
        return Err(DriverError::NoSamples);
    }
    let samples = (0..cfg.n)
        .map(|i| run_sample(task, cfg, i, agent, runner))
        .collect::<Result<_, _>>()?;
    Ok(TaskRecord {
        id: task.id.clone(),
        release_date: task.date().to_string(),
        samples,
    })
}

/// Run a validated suite, task by task in suite order.
pub fn run_suite(
    suite: &Suite,
    cfg: &Config,
    agent: &mut impl Agent,
    runner: &mut impl TestRunner,
) -> Result<Vec<TaskRecord>, DriverError> {
    suite
        .tasks()
        .iter()
        .map(|v| run_task(v.task(), cfg, agent, runner))
        .collect()
}

/// Which attempt a metric is computed over (spec §8: pass@1 per attempt).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pass {
    /// The first attempt of every sample.
    Attempt1,
    /// The last attempt of every sample: attempt 2 where one ran.
    Attempt2,
}

/// ACE-DRV-007: the metric engine's input for one attempt.
#[must_use]
pub fn results(records: &[TaskRecord], pass: Pass) -> Vec<TaskResult> {
    let pick = |s: &[AttemptRecord]| match pass {
        Pass::Attempt1 => s.first().cloned(),
        Pass::Attempt2 => s.last().cloned(),
    };
    records
        .iter()
        .map(|t| TaskResult {
            task_id: t.id.clone(),
            release_date: t.release_date.clone(),
            samples: t
                .samples
                .iter()
                .filter_map(|s| pick(s))
                .map(|a| Sample {
                    verdict: a.verdict,
                    tests_executed: a.tests_executed,
                })
                .collect(),
        })
        .collect()
}

/// ACE-DRV-006: the records with wall time zeroed, for the determinism comparison.
#[must_use]
pub fn without_timing(records: &[TaskRecord]) -> Vec<TaskRecord> {
    let mut out = records.to_vec();
    for a in out.iter_mut().flat_map(|t| t.samples.iter_mut().flatten()) {
        a.wall_ms = 0;
    }
    out
}

#[cfg(test)]
#[path = "driver_tests.rs"]
mod tests;
