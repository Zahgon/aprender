//! `aprender-cgp profile quant --all` (`profile_quant_all`): parses the
//! `benchmark_matrix_suite` output into a per-layer table plus a summary.
//!
//! The command is driven as a binary from a throwaway cargo project whose target
//! dir holds a fake `benchmark_matrix_suite`, so every branch (no binary, failing
//! binary, unparsable line, partial and multi-layer measurement) is exercised.

// Integration-test binary: `.expect()` / `.unwrap()` are idiomatic in assertions.
#![allow(clippy::disallowed_methods)]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

/// Run `profile quant --all` in a scratch cargo project; `bench` is the body of the
/// fake benchmark script (None = no binary installed). Returns (success, stdout).
fn sweep(bench: Option<&str>) -> (bool, String) {
    let proj = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        proj.path().join("Cargo.toml"),
        "[package]\nname = \"scratch\"\nversion = \"0.0.0\"\nedition = \"2021\"\n[lib]\npath = \"lib.rs\"\n",
    )
    .expect("manifest");
    std::fs::write(proj.path().join("lib.rs"), "").expect("lib");
    let target = proj.path().join("tgt");
    if let Some(body) = bench {
        let dir = target.join("release/examples");
        std::fs::create_dir_all(&dir).expect("examples dir");
        let exe = dir.join("benchmark_matrix_suite");
        std::fs::write(&exe, format!("#!/bin/sh\n{body}\n")).expect("script");
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    }
    let out = Command::new(env!("CARGO_BIN_EXE_aprender-cgp"))
        .args(["profile", "quant", "--all"])
        .current_dir(Path::new(proj.path()))
        .env("CARGO_TARGET_DIR", &target)
        .output()
        .expect("spawn aprender-cgp");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

fn row(shape: &str, us: &str) -> String {
    format!("echo 'Q4K GEMV {shape} ... {us} us (2.5 GFLOPS, 7.5 GB/s)'")
}

#[test]
fn no_benchmark_binary_reports_no_data_and_build_hint() {
    let (ok, out) = sweep(None);
    assert!(ok, "missing bench binary must not fail the command");
    assert!(out.contains("No benchmark data available."), "{out}");
    assert!(
        out.contains("benchmark_matrix_suite --features parallel"),
        "{out}"
    );
    assert!(!out.contains("Summary ("), "{out}");
}

#[test]
fn failing_benchmark_binary_is_treated_as_no_data() {
    let body = format!("{}\nexit 3", row("1536x8960", "100.0"));
    let (ok, out) = sweep(Some(&body));
    assert!(ok);
    assert!(out.contains("No benchmark data available."), "{out}");
}

#[test]
fn one_measured_layer_fills_its_row_and_the_summary() {
    let (ok, out) = sweep(Some(&row("1536x8960", "100.0")));
    assert!(ok);
    // 1000 / (100 us * 192 / 1000) = 52.083 tok/s
    let line = out
        .lines()
        .find(|l| l.contains("ffn_up/gate (1.5B-7B)"))
        .expect("ffn_up row");
    for cell in ["100.0", "2.5", "7.5", "52.1"] {
        assert!(line.contains(cell), "{cell} missing in {line}");
    }
    assert!(out.contains("Summary (1/7 layers measured):"), "{out}");
    assert!(out.contains("Avg GEMV time: 100.0 us"), "{out}");
    assert!(out.contains("Composite tok/s estimate: 52.1"), "{out}");
    assert!(
        out.contains("Total GEMV time (measured layers): 100.0 us"),
        "{out}"
    );
    // An unmeasured layer prints dashes, not stale numbers.
    let other = out.lines().find(|l| l.contains("generic_4K")).expect("row");
    assert!(!other.contains("100.0") && other.contains('-'), "{other}");
}

#[test]
fn two_layers_are_averaged_and_summed() {
    let body = format!(
        "{}\n{}",
        row("1536x1536", "100.0"),
        row("4096x4096", "300.0")
    );
    let (_, out) = sweep(Some(&body));
    assert!(out.contains("Summary (2/7 layers measured):"), "{out}");
    assert!(out.contains("Avg GEMV time: 200.0 us"), "{out}");
    // 1000 / (200 * 192 / 1000) = 26.04
    assert!(out.contains("Composite tok/s estimate: 26.0"), "{out}");
    assert!(
        out.contains("Total GEMV time (measured layers): 400.0 us"),
        "{out}"
    );
}

#[test]
fn unparsable_or_foreign_lines_do_not_count_as_measurements() {
    let body = "echo 'Q4K GEMV 1536x8960 ... oops us (2.5 GFLOPS, 7.5 GB/s)'\n\
                echo 'Q6K GEMV 4096x4096 ... 50.0 us (2.5 GFLOPS, 7.5 GB/s)'\n\
                echo 'Q4K GEMV 1536x1536 ... 80.0 us (no throughput)'";
    let (ok, out) = sweep(Some(body));
    assert!(ok);
    assert!(out.contains("No benchmark data available."), "{out}");
}
