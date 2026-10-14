//! The sandbox-backed [`TestRunner`] (contracts/apr-code-eval-runner-v1.yaml,
//! ACE-RUN-001..005).
//!
//! The driver hands a runner the edited tree with the hidden files laid over it. This one
//! writes that tree into a fresh directory, runs it through [`crate::sandbox::run`] in the
//! task's pinned image, and removes the directory again.
//!
//! - Each run gets a new directory and a new container name; nothing is shared between
//!   runs (ACE-RUN-001).
//! - A tree path that is absolute or climbs out of the directory is refused before any
//!   file is written (ACE-RUN-002).
//! - A task without an environment, a missing image or an engine failure is an `Err`,
//!   which the driver records as SANDBOX_ERROR; it is never a test result (ACE-RUN-003).
//! - `SOURCE_DATE_EPOCH` is the task's release date at 00:00 UTC (ACE-RUN-004).
//! - The image digest of every task is available for the receipt (ACE-RUN-005).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::driver::TestRunner;
use crate::edit::Files;
use crate::sandbox::{self, Engine, ImageRef, Limits, ReportFormat, RunSpec};
use crate::task::Task;
use crate::verdict::SandboxReport;

/// How one task's tests run: its pinned image, test command and report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskEnv {
    /// The image, `name@sha256:<64 hex>`.
    pub image: ImageRef,
    /// The test command, argv form, run in `/work`.
    pub command: Vec<String>,
    /// Where the command writes its report, relative to `/work`.
    pub report_path: String,
    /// The report's format.
    pub report_format: ReportFormat,
    /// Resource bounds.
    #[serde(default)]
    pub limits: Limits,
}

/// Seconds from 1970-01-01 to `YYYY-MM-DD` at 00:00 UTC, or `None` if it is not a date.
#[must_use]
pub fn epoch(date: &str) -> Option<i64> {
    let mut it = date.split('-');
    let (y, m, d) = (it.next()?, it.next()?, it.next()?);
    if it.next().is_some() || y.len() != 4 || m.len() != 2 || d.len() != 2 {
        return None;
    }
    let (y, m, d): (i64, i64, i64) = (y.parse().ok()?, m.parse().ok()?, d.parse().ok()?);
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let days_in = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if !(1..=12).contains(&m) || d < 1 || d > days_in[usize::try_from(m - 1).ok()?] {
        return None;
    }
    // Days from civil (proleptic Gregorian), era-based.
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some((era * 146_097 + doe - 719_468) * 86_400)
}

/// ACE-RUN-002: a relative path that stays inside the directory.
fn safe(path: &str) -> bool {
    let p = Path::new(path);
    !path.is_empty()
        && p.components().all(|c| matches!(c, Component::Normal(_)))
        && !path.split('/').any(str::is_empty)
}

/// Write `tree` under `dir`, refusing any unsafe path before writing anything.
pub fn materialize(dir: &Path, tree: &Files) -> Result<(), String> {
    if let Some(bad) = tree.keys().find(|p| !safe(p)) {
        return Err(format!("unsafe tree path {bad:?}"));
    }
    for (path, body) in tree {
        let to = dir.join(path);
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("{path}: {e}"))?;
        }
        fs::write(&to, body).map_err(|e| format!("{path}: {e}"))?;
    }
    Ok(())
}

/// Runs each tree in its task's image through a container [`Engine`].
#[derive(Debug)]
pub struct Sandboxed<E> {
    engine: E,
    envs: BTreeMap<String, TaskEnv>,
    root: PathBuf,
    home: Option<PathBuf>,
    seed: u64,
    tag: String,
    runs: u64,
}

impl<E: Engine> Sandboxed<E> {
    /// `root` holds the per-run directories; `tag` makes container names unique to this
    /// harness process; `home` is refused as a task dir (ACE-SBX-004).
    pub fn new(
        engine: E,
        envs: BTreeMap<String, TaskEnv>,
        root: PathBuf,
        home: Option<PathBuf>,
        seed: u64,
        tag: &str,
    ) -> Self {
        Self {
            engine,
            envs,
            root,
            home,
            seed,
            tag: tag.to_string(),
            runs: 0,
        }
    }

    /// ACE-RUN-005: task id → image digest, for the receipt.
    #[must_use]
    pub fn image_digests(&self) -> BTreeMap<String, String> {
        self.envs
            .iter()
            .map(|(id, env)| (id.clone(), env.image.digest().to_string()))
            .collect()
    }

    fn spec(&self, task: &Task, dir: PathBuf) -> Result<RunSpec, String> {
        let env = self
            .envs
            .get(&task.id)
            .ok_or_else(|| format!("no environment for task {}", task.id))?;
        let date = task.release_date.as_deref().unwrap_or_default();
        let source_date_epoch =
            epoch(date).ok_or_else(|| format!("task {} has no release date", task.id))?;
        Ok(RunSpec {
            image: env.image.clone(),
            task_dir: dir,
            command: env.command.clone(),
            report_path: env.report_path.clone(),
            report_format: env.report_format,
            seed: self.seed,
            source_date_epoch,
            limits: env.limits,
        })
    }
}

impl<E: Engine> TestRunner for Sandboxed<E> {
    fn run(&mut self, task: &Task, tree: &Files) -> Result<SandboxReport, String> {
        self.runs += 1;
        let name = format!("apr-code-eval-{}-{}", self.tag, self.runs);
        let dir = self.root.join(&name);
        let spec = self.spec(task, dir.clone())?;
        fs::create_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let out = materialize(&dir, tree).and_then(|()| {
            sandbox::run(&mut self.engine, &spec, &name, self.home.as_deref())
                .map_err(|e| e.to_string())
        });
        let cleaned = fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()));
        let report = out?;
        cleaned?;
        Ok(report)
    }
}

#[cfg(test)]
#[path = "runner_tests.rs"]
mod tests;
