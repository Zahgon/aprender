//! `aprender-rollup plan` prints the rollup it would build (read-only).
//! `aprender-rollup build <dir>` assembles it locally in worktree `<dir>`; it never pushes or closes.

use aprender_rollup::{build, gh_json, report, select};
use std::process::{Command, ExitCode};

fn candidates() -> Result<Vec<aprender_rollup::Candidate>, String> {
    let o = Command::new("gh")
        .args([
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            gh_json::GH_FIELDS,
        ])
        .output()
        .map_err(|e| format!("cannot run gh: {e}"))?;
    if !o.status.success() {
        return Err(format!(
            "gh failed: {}",
            String::from_utf8_lossy(&o.stderr).trim()
        ));
    }
    gh_json::parse_prs(&String::from_utf8_lossy(&o.stdout))
}

fn git(dir: &str, args: &[String]) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .status()
        .is_ok_and(|s| s.success())
}

/// Merge each picked PR; a conflicting PR is aborted and ejected (stays open).
fn assemble(dir: &str, picked: &[u64]) -> Vec<u64> {
    let mut merged = Vec::new();
    for n in picked {
        let one = build::steps("main", &[*n]);
        // steps 2..4 are this PR's fetch + merge; branch setup already ran for the full set
        if git(dir, &one[2]) && git(dir, &one[3]) {
            merged.push(*n);
        } else {
            let _ = git(dir, &build::abort_step());
            eprintln!("eject #{n}: conflicts with the rollup so far");
        }
    }
    merged
}

fn run(cmd: &str, dir: Option<&str>) -> Result<(), String> {
    let sel = select(&candidates()?);
    if cmd == "plan" {
        print!("{}", report::plan_text(&sel));
        return Ok(());
    }
    let dir = dir.ok_or("usage: aprender-rollup build <dir>")?;
    let setup = build::steps("main", &sel.picked);
    if !git(dir, &setup[0]) || !git(dir, &setup[1]) {
        return Err("cannot create rollup branch".to_string());
    }
    let merged = assemble(dir, &sel.picked);
    println!(
        "{}\n{}",
        build::branch_name(&sel.picked),
        build::message(&merged)
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    if cmd != "plan" && cmd != "build" {
        eprintln!("usage: aprender-rollup plan | build <dir>");
        return ExitCode::from(2);
    }
    match run(cmd, args.get(2).map(String::as_str)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("not_measured: {e}");
            ExitCode::FAILURE
        }
    }
}
