//! The eval end to end (contracts/apr-code-eval-harness-v1.yaml, ACE-HAR-001..006).
//!
//! [`evaluate`] is what `apr code eval` calls: a loaded task set, a sandbox-backed runner
//! and a model go in, the §7 receipt comes out.
//!
//! - Gold and null validation run through the same sandbox as the scored runs. The gold
//!   edit is a unified diff on the snapshot; a gold edit that does not apply fails the task
//!   (ACE-HAR-001).
//! - The gold edit runs [`GOLD_RUNS`] times and every report must agree; a flake refuses
//!   the suite by name (spec §4, ACE-HAR-002).
//! - Every task has an environment before any container or model runs (ACE-HAR-003).
//! - A suite that does not validate makes no model call, and a backend failure writes no
//!   receipt (ACE-HAR-004).
//! - The receipt's task set digest and image digests are those of the suite that ran,
//!   whatever the caller passed (ACE-HAR-005).
//! - Two seeded evaluations of the same inputs give receipts that are the same run
//!   (ACE-HAR-006).

use std::time::Instant;

use thiserror::Error;

use crate::driver::{run_suite, Agent, Config, DriverError, TestRunner};
use crate::edit::{adapt, overlay, EditFormat};
use crate::receipt::{self, Receipt, ReceiptError, Run};
use crate::runner::Sandboxed;
use crate::sandbox::{agree, Engine};
use crate::task::{Resolver, Suite, Task, TaskError, TaskSet};
use crate::verdict::{verdict, EditOutcome, TestLists, Verdict};

/// Spec §4: how many times validation runs the gold edit.
pub const GOLD_RUNS: usize = 3;

/// Gold edits are unified diffs, as SWE-bench ships them.
pub const GOLD_FORMAT: EditFormat = EditFormat::Udiff;

/// The evaluation cannot produce a receipt.
#[derive(Debug, Error)]
pub enum HarnessError {
    /// ACE-HAR-003: these tasks have no environment.
    #[error("no environment for tasks {0:?}")]
    NoEnvironment(Vec<String>),
    /// ACE-HAR-001/002/004: the suite did not validate.
    #[error("suite: {0}")]
    Suite(#[from] TaskError),
    /// ACE-HAR-004: the driver stopped.
    #[error("driver: {0}")]
    Driver(#[from] DriverError),
    /// The receipt refused the run.
    #[error("receipt: {0}")]
    Receipt(#[from] ReceiptError),
}

impl<E: Engine> Resolver for Sandboxed<E> {
    fn resolved(&mut self, task: &Task, edit: Option<&str>) -> Result<bool, String> {
        let snapshot = &task.snapshot.files;
        let (outcome, files, runs) = match edit {
            None => (EditOutcome::Applied, snapshot.clone(), 1),
            Some(gold) => {
                let a = adapt(GOLD_FORMAT, gold, snapshot);
                let Some(files) = a.files else {
                    return Ok(false);
                };
                (a.outcome, files, GOLD_RUNS)
            }
        };
        let tree = overlay(&files, &task.hidden_tests.files);
        let reports = (0..runs)
            .map(|_| self.run(task, &tree))
            .collect::<Result<Vec<_>, _>>()?;
        agree(&reports).map_err(|f| {
            format!(
                "flake: gold run {} differs from run 0 on {:?}",
                f.run, f.differing
            )
        })?;
        let tests = TestLists {
            f2p: &task.hidden_tests.fail_to_pass,
            p2p: &task.hidden_tests.pass_to_pass,
        };
        Ok(verdict(tests, outcome, &reports[0]) == Verdict::Resolved)
    }
}

/// Validate `set` through `runner`, run every task through `agent`, and write the receipt.
/// `run` carries the caller's provenance; its task set digest, image digests and wall
/// time are filled in here.
pub fn evaluate<E: Engine>(
    set: TaskSet,
    runner: &mut Sandboxed<E>,
    agent: &mut impl Agent,
    cfg: &Config,
    mut run: Run,
    k: Option<usize>,
    cutoff: Option<&str>,
) -> Result<Receipt, HarnessError> {
    let started = Instant::now();
    let missing: Vec<String> = set
        .tasks()
        .iter()
        .filter(|t| !runner.has_env(&t.id))
        .map(|t| t.id.clone())
        .collect();
    if !missing.is_empty() {
        return Err(HarnessError::NoEnvironment(missing));
    }
    let suite = Suite::build(set, runner)?;
    let records = run_suite(&suite, cfg, agent, runner)?;
    let digests = runner.image_digests();
    run.task_set_digest = suite.digest().to_string();
    run.sandbox.image_digests = suite
        .tasks()
        .iter()
        .filter_map(|v| digests.get_key_value(&v.task().id))
        .map(|(id, d)| (id.clone(), d.clone()))
        .collect();
    run.wall_s = started.elapsed().as_secs_f64();
    Ok(receipt::build(run, cfg, &records, k, cutoff)?)
}

#[cfg(test)]
#[path = "harness_tests.rs"]
mod tests;
