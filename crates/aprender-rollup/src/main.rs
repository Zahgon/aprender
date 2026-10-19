//! `aprender-rollup plan` prints the rollup it would build (read-only).
//! `aprender-rollup build <dir>` assembles it locally in worktree `<dir>`; it never pushes or closes.

use aprender_rollup::mg::{self, Verdict};
use aprender_rollup::{build, close, gh_json, outcome, report, select};
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

fn gh(args: &[String]) -> Result<String, String> {
    let o = Command::new("gh")
        .args(args)
        .output()
        .map_err(|e| format!("cannot run gh: {e}"))?;
    if !o.status.success() {
        return Err(format!(
            "gh failed: {}",
            String::from_utf8_lossy(&o.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&o.stdout).into_owned())
}

/// `verify <branch> <sha|-> <n,n,..> [--apply]`: read the merge-group verdict for the
/// rollup branch. Green + SHA on main closes each constituent with the SHA; red prints the
/// two halves to rebuild and test; pending touches nothing. Without `--apply`, dry run.
fn verify(a: &[String]) -> Result<(), String> {
    let [branch, sha, list, rest @ ..] = a else {
        return Err("usage: aprender-rollup verify <branch> <sha|-> <n,n,..> [--apply]".into());
    };
    let picked: Vec<u64> = list
        .split(',')
        .map(|s| {
            s.trim_start_matches('#')
                .parse()
                .map_err(|_| format!("bad PR {s}"))
        })
        .collect::<Result<_, _>>()?;
    let runs = gh(&mg::run_list_args(branch))?;
    let v = mg::parse_verdict(&runs)?;
    println!("verdict {v:?}");
    let sha = (sha != "-").then_some(sha.as_str());
    match v {
        Verdict::Pending => {}
        Verdict::Red => match aprender_rollup::split(&picked) {
            Some((x, y)) => println!("bisect: rebuild halves {x:?} and {y:?}"),
            None => println!("culprit #{}", picked[0]),
        },
        Verdict::Green => {
            for n in &picked {
                let args = close::gh_args(*n, &outcome(true, sha));
                println!("#{n}: gh {}", args.join(" "));
                if rest.iter().any(|r| r == "--apply") {
                    gh(&args)?;
                }
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    let res = match cmd {
        "plan" | "build" => run(cmd, args.get(2).map(String::as_str)),
        "verify" => verify(&args[2..]),
        _ => {
            eprintln!("usage: aprender-rollup plan | build <dir> | verify <branch> <sha|-> <n,..> [--apply]");
            return ExitCode::from(2);
        }
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("not_measured: {e}");
            ExitCode::FAILURE
        }
    }
}
