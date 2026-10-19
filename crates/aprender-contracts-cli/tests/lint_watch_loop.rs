//! `pv lint --watch` re-lints on a 5 s timer and never returns on its own. The empty-corpus
//! refusal is covered in `pvl_zero_contracts.rs`; this file covers the loop itself on a real
//! corpus: a report per tick, the "Watching" banner after each, and a second tick that runs.

#![allow(clippy::disallowed_methods)]

use std::io::{BufRead, BufReader, Read};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const BANNER: &str = "--- Watching for changes (Ctrl+C to stop) ---";

/// Run `pv lint <dir> --watch [extra]` until `ticks` banners were printed (or 60 s pass),
/// kill it, and return everything it wrote to stdout.
fn watch_ticks(extra: &[&str], ticks: usize) -> (String, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/relu-kernel-v1.yaml"
    );
    std::fs::copy(src, dir.path().join("relu-kernel-v1.yaml")).expect("copy fixture");
    let mut child = Command::new(env!("CARGO_BIN_EXE_pv"))
        .current_dir(dir.path())
        .args(["lint", "--watch", "--no-cache"])
        .args(extra)
        .arg(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn pv");
    let pid = child.id();
    let done = Arc::new(Mutex::new(false));
    let flag = Arc::clone(&done);
    let watchdog = std::thread::spawn(move || {
        for _ in 0..600 {
            std::thread::sleep(Duration::from_millis(100));
            if *flag.lock().expect("flag") {
                return;
            }
        }
        // Deadline hit: unblock the reader below. The pid is our own child.
        let _ = Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status();
    });
    let mut seen = 0;
    let mut out = String::new();
    for line in BufReader::new(child.stdout.take().expect("stdout")).lines() {
        let Ok(line) = line else { break };
        out.push_str(&line);
        out.push('\n');
        if line == BANNER {
            seen += 1;
            if seen == ticks {
                break;
            }
        }
    }
    *done.lock().expect("flag") = true;
    child.kill().expect("kill pv");
    child.wait().expect("reap pv");
    watchdog.join().expect("watchdog");
    let mut err = String::new();
    child
        .stderr
        .take()
        .expect("stderr")
        .read_to_string(&mut err)
        .expect("stderr");
    (out, err)
}

#[test]
fn watch_prints_report_then_banner_and_ticks_again() {
    let (out, _) = watch_ticks(&[], 2);
    assert_eq!(out.matches(BANNER).count(), 2, "{out}");
    let first_banner = out.find(BANNER).expect("banner");
    assert!(
        out[..first_banner].contains("relu-kernel-v1"),
        "report must precede the first banner: {out}"
    );
    // The second tick prints its own report between the banners.
    let second = out[first_banner + BANNER.len()..]
        .find(BANNER)
        .expect("second banner");
    assert!(
        out[first_banner + BANNER.len()..][..second].contains("relu-kernel-v1"),
        "second tick printed no report: {out}"
    );
}

#[test]
fn watch_honours_the_format_flag_each_tick() {
    let (out, _) = watch_ticks(&["--format", "json"], 1);
    let first_banner = out.find(BANNER).expect("banner");
    assert!(
        out[..first_banner].trim_start().starts_with('{'),
        "not json: {out}"
    );
}

#[test]
fn watch_cache_stats_line_is_printed_each_tick() {
    let (out, err) = watch_ticks(&["--cache-stats"], 1);
    assert!(err.contains("Cache: "), "no cache stats line: {err}");
    assert!(
        !out.contains("Cache: "),
        "cache stats belong on stderr: {out}"
    );
}
