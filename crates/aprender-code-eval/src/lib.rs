//! `apr code` evaluation harness (docs/specifications/apr-code-eval-v1.md).
//!
//! Built in-tree on the pv contracts `contracts/apr-code-eval-*-v1.yaml`:
//! - [`verdict`]: one sandbox report and one model output → one closed label (ACE-VER).
//! - [`metric`]: labels → pass@1, pass@k, Wilson interval and a complete receipt (ACE-MET).
//! - [`task`]: a JSON Lines manifest → a checked task set → a validated suite (ACE-TASK).
//! - [`sandbox`]: one hermetic, limited, offline container run → one sandbox report (ACE-SBX).

pub mod metric;
pub mod sandbox;
pub mod task;
pub mod verdict;
