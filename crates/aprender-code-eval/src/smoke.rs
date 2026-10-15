//! The `smoke` suite (contracts/apr-code-eval-smoke-v1.yaml, ACE-SMK-001..004): eight
//! hand-written MIT tasks, two per kind, that every nightly runs first
//! (docs/specifications/apr-code-eval-v1.md §3).
//!
//! - The manifest loads as eight tasks, two of each kind, each with an environment pinned
//!   to one image digest (ACE-SMK-001).
//! - Every gold edit applies as a unified diff and changes the snapshot (ACE-SMK-002).
//! - The FAIL_TO_PASS and PASS_TO_PASS lists name exactly the hidden tests (ACE-SMK-003).
//! - Two runs with the same seed write receipts that are byte-identical apart from timing
//!   (§8, ACE-SMK-004).

use std::collections::BTreeMap;

use crate::runner::TaskEnv;
use crate::task::{load, TaskError, TaskSet};

/// The suite name a receipt of this suite carries.
pub const NAME: &str = "smoke";

/// The task manifest, one task per line.
pub const TASKS: &str = include_str!("../suites/smoke/tasks.jsonl");

/// The environment table: task id → pinned image and test command.
pub const ENVS: &str = include_str!("../suites/smoke/envs.json");

/// The tasks, loaded as of `load_date` (`YYYY-MM-DD`).
pub fn tasks(load_date: &str) -> Result<TaskSet, TaskError> {
    load(TASKS, load_date)
}

/// The environment table.
pub fn envs() -> Result<BTreeMap<String, TaskEnv>, serde_json::Error> {
    serde_json::from_str(ENVS)
}

#[cfg(test)]
#[path = "smoke_tests.rs"]
mod tests;
