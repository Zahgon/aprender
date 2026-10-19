//! `cgp compete` (`run_compete`): labels, exit codes, winner selection, failed
//! competitors and JSON mode (#5654). Drives the real binary.
#![allow(clippy::disallowed_methods)]

use std::process::{Command, Output};

fn compete(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_aprender-cgp"))
        .arg("compete")
        .args(args)
        .output()
        .expect("spawn aprender-cgp")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn faster_competitor_wins_and_custom_labels_are_used() {
    let o = compete(&[
        "gemm",
        "--ours",
        "sleep 0.2",
        "--theirs",
        "true",
        "--label",
        "slow,fast",
    ]);
    assert!(o.status.success());
    let out = stdout(&o);
    assert!(out.contains("CGP Head-to-Head: gemm"), "{out}");
    assert!(out.contains("Winner: fast"), "{out}");
    assert!(out.contains("1.00x"), "best row must read 1.00x: {out}");
    assert!(!out.contains("Winner: slow"), "{out}");
}

#[test]
fn default_labels_are_ours_and_theirs_n() {
    let o = compete(&[
        "w", "--ours", "true", "--theirs", "true", "--theirs", "true",
    ]);
    let out = stdout(&o);
    for l in ["ours", "theirs-1", "theirs-2"] {
        assert!(out.contains(l), "missing {l}: {out}");
    }
}

#[test]
fn unlaunchable_competitor_is_reported_failed_not_fatal() {
    let o = compete(&["w", "--ours", "true", "--theirs", "no_such_bin_zz9"]);
    assert!(
        o.status.success(),
        "a missing competitor must not abort the run"
    );
    let out = stdout(&o);
    assert!(
        out.contains("FAILED") && out.contains("-1") && out.contains("N/A"),
        "{out}"
    );
    assert!(out.contains("Winner: ours"), "{out}");
}

#[test]
fn json_mode_lists_every_competitor_with_exit_codes_and_no_table() {
    let o = compete(&["w", "--json", "--ours", "true", "--theirs", "false"]);
    let out = stdout(&o);
    assert!(
        !out.contains("Winner:"),
        "json mode must skip the table: {out}"
    );
    let json = &out[out.find('[').expect("json array")..];
    let v: serde_json::Value = serde_json::from_str(json).expect("valid json");
    let rows = v.as_array().expect("array");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["label"], "ours");
    assert_eq!(rows[0]["exit_code"], 0);
    assert_eq!(rows[1]["label"], "theirs-1");
    assert_eq!(rows[1]["exit_code"], 1);
    assert!(rows[1]["wall_time_ms"].as_f64().expect("ms") > 0.0);
}
