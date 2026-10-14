//! `apr code` evaluation harness (docs/specifications/apr-code-eval-v1.md).
//!
//! Built in-tree on the pv contracts `contracts/apr-code-eval-*-v1.yaml`:
//! - [`verdict`]: one sandbox report and one model output → one closed label (ACE-VER).
//! - [`metric`]: labels → pass@1, pass@k, Wilson interval and a complete receipt (ACE-MET).
//! - [`task`]: a JSON Lines manifest → a checked task set → a validated suite (ACE-TASK).
//! - [`sandbox`]: one hermetic, limited, offline container run → one sandbox report (ACE-SBX).
//! - [`edit`]: a model output in one edit format → edits applied all or nothing (ACE-EDIT).
//! - [`driver`]: validated tasks, a model and a sandbox → per-attempt records (ACE-DRV).
//! - [`receipt`]: per-attempt records and run provenance → the §7 run receipt (ACE-RCP).
//! - [`runner`]: a tree and its task's pinned environment → one sandbox run (ACE-RUN).
//! - [`harness`]: a task set, a runner and a model → a validated suite run and its receipt (ACE-HAR).
//! - [`prompt`]: one driver request → one chat prompt for a single-turn model (ACE-PRM).

pub mod driver;
pub mod edit;
pub mod harness;
pub mod metric;
pub mod prompt;
pub mod receipt;
pub mod runner;
pub mod sandbox;
pub mod task;
pub mod verdict;
