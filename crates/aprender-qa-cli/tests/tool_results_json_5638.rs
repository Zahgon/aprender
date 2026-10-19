//! `save_tool_results_json_or_exit` (reached via `apr-qa tools`): the per-tool
//! results land in `<output>/tool_tests.json`, the output dir is created on demand,
//! and an unusable output location is a clean exit 1 with a named error (#5638).
//!
//! The tools are driven against a silent fake `apr` first on PATH (verdicts depend on
//! its output, so it prints nothing), so no model is needed.

// Integration-test binary: `.expect()` is idiomatic in assertions.
#![allow(clippy::disallowed_methods)]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_apr-qa");

/// Run `apr-qa tools <model> --output <out>` with a fake `apr` (`exit <code>`) on PATH.
fn tools(scratch: &Path, out: &Path, apr_exit: i32) -> Output {
    let bin_dir = scratch.join("fakebin");
    std::fs::create_dir_all(&bin_dir).expect("bin dir");
    let apr = bin_dir.join("apr");
    std::fs::write(&apr, format!("#!/bin/sh\nexit {apr_exit}\n")).expect("apr");
    std::fs::set_permissions(&apr, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    let model = scratch.join("model.gguf");
    std::fs::write(&model, b"not a model").expect("model");
    let path = format!(
        "{}:{}",
        bin_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    Command::new(BIN)
        .arg("tools")
        .arg(&model)
        .arg("--no-gpu")
        .arg("--output")
        .arg(out)
        .env("PATH", path)
        .output()
        .expect("spawn apr-qa")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}
fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn writes_one_record_per_tool_with_the_documented_fields() {
    let tmp = tempfile::tempdir().expect("tmp");
    let out = tmp.path().join("nested/results");
    let run = tools(tmp.path(), &out, 0);
    assert!(
        stdout(&run).contains("Results saved to:"),
        "{}",
        stdout(&run)
    );
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("tool_tests.json")).expect("read"))
            .expect("valid json");
    let rows = json.as_array().expect("array");
    assert!(
        rows.len() >= 10,
        "expected the core tool set, got {}",
        rows.len()
    );
    for row in rows {
        for field in [
            "tool",
            "passed",
            "exit_code",
            "duration_ms",
            "gate_id",
            "stderr",
        ] {
            assert!(row.get(field).is_some(), "{field} missing in {row}");
        }
    }
    let row = |name: &str| rows.iter().find(|r| r["tool"] == name).expect(name);
    // Per-tool verdicts are carried through verbatim (silent fake apr, exit 0):
    assert_eq!(row("inspect")["passed"], true);
    assert_eq!(row("inspect")["exit_code"], 0);
    assert_eq!(row("profile-ci")["passed"], false);
}

#[test]
fn failing_tools_are_recorded_then_exit_nonzero() {
    let tmp = tempfile::tempdir().expect("tmp");
    let out = tmp.path().join("res");
    let run = tools(tmp.path(), &out, 1);
    assert_eq!(run.status.code(), Some(1), "{}", stdout(&run));
    // The JSON is written BEFORE the failure exit, so a red run still leaves evidence.
    // With apr exiting 1 only the expects-failure tool passes.
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("tool_tests.json")).expect("read"))
            .expect("valid json");
    let rows = json.as_array().expect("array");
    let passed: Vec<_> = rows.iter().filter(|r| r["passed"] == true).collect();
    assert_eq!(passed.len(), 1, "{json}");
    assert_eq!(passed[0]["tool"], "profile-ci-assertion");
    assert!(rows.iter().all(|r| r["exit_code"] == 1), "{json}");
}

#[test]
fn output_dir_that_is_a_file_exits_one_naming_the_cause() {
    let tmp = tempfile::tempdir().expect("tmp");
    let blocker = tmp.path().join("blocker");
    std::fs::write(&blocker, b"x").expect("blocker");
    let run = tools(tmp.path(), &blocker.join("sub"), 0);
    assert_eq!(run.status.code(), Some(1));
    assert!(
        stderr(&run).contains("Error creating output directory"),
        "{}",
        stderr(&run)
    );
}

#[test]
fn unwritable_results_file_exits_one_naming_the_cause() {
    let tmp = tempfile::tempdir().expect("tmp");
    let out = tmp.path().join("res");
    // tool_tests.json is a directory, so fs::write must fail.
    std::fs::create_dir_all(out.join("tool_tests.json")).expect("dir");
    let run = tools(tmp.path(), &out, 0);
    assert_eq!(run.status.code(), Some(1));
    assert!(
        stderr(&run).contains("Error saving tool test results"),
        "{}",
        stderr(&run)
    );
}
