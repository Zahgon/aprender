//! `aprender-rollup plan`: read open PRs through `gh` (GraphQL) and print the rollup it would build.
//! Read-only; building, merging and closing are later steps.

use aprender_rollup::{gh_json, report, select};
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    if std::env::args().nth(1).as_deref() != Some("plan") {
        eprintln!("usage: aprender-rollup plan");
        return ExitCode::from(2);
    }
    let out = Command::new("gh")
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
        .output();
    let text = match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).into_owned(),
        Ok(o) => {
            eprintln!(
                "not_measured: gh failed: {}",
                String::from_utf8_lossy(&o.stderr).trim()
            );
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("not_measured: cannot run gh: {e}");
            return ExitCode::FAILURE;
        }
    };
    match gh_json::parse_prs(&text) {
        Ok(c) => {
            print!("{}", report::plan_text(&select(&c)));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("not_measured: {e}");
            ExitCode::FAILURE
        }
    }
}
