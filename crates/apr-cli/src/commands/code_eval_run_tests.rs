//! `apr code eval` wiring (#5286): flag parsing, the environment table, provenance.

use super::*;
use crate::{CodeCommands, Commands};
use clap::Parser;

fn parse(args: &[&str]) -> Result<crate::Cli, clap::error::Error> {
    let args: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || crate::Cli::try_parse_from(args))
        .expect("spawn thread")
        .join()
        .expect("join")
}

const EVAL: &[&str] = &[
    "apr",
    "code",
    "eval",
    "--tasks",
    "t.jsonl",
    "--envs",
    "e.json",
    "--model",
    "m.gguf",
    "--receipt",
    "r.json",
    "--suite",
    "code-v1",
    "--work-dir",
    "/tmp/w",
];

#[test]
fn eval_parses_as_the_code_subcommand_with_its_defaults() {
    let cli = parse(EVAL).expect("parses");
    let Commands::Code {
        action:
            Some(CodeCommands::Eval {
                format,
                n,
                k,
                seed,
                max_tokens,
                no_gpu,
                engine,
                load_date,
                ..
            }),
        prompt,
        ..
    } = *cli.command
    else {
        panic!("expected apr code eval");
    };
    assert_eq!(format, "udiff");
    assert_eq!((n, k, seed, max_tokens, no_gpu), (1, None, 0, 2048, false));
    assert_eq!(engine.to_string(), "docker");
    assert!(load_date.is_none());
    assert!(prompt.is_empty());
}

#[test]
fn a_flag_first_makes_eval_a_prompt_word() {
    let cli = parse(&["apr", "code", "-p", "eval", "this"]).expect("parses");
    let Commands::Code {
        action,
        prompt,
        print,
        ..
    } = *cli.command
    else {
        panic!("expected apr code");
    };
    assert!(action.is_none(), "eval after -p is a prompt word");
    assert!(print);
    assert_eq!(prompt, vec!["eval".to_string(), "this".to_string()]);
}

#[test]
fn eval_refuses_bad_values() {
    let mut no_model: Vec<&str> = EVAL.to_vec();
    let at = no_model.iter().position(|a| *a == "--model").expect("flag");
    no_model.drain(at..at + 2);
    assert!(parse(&no_model).is_err(), "--model is required");
    for (flag, bad) in [("--format", "agentic"), ("--n", "0"), ("--max-tokens", "0")] {
        let mut args = EVAL.to_vec();
        args.extend([flag, bad]);
        assert!(parse(&args).is_err(), "{flag} {bad}");
    }
}

#[test]
fn format_names_map_to_the_edit_formats() {
    assert_eq!(parse_format("whole").ok(), Some(EditFormat::Whole));
    assert_eq!(
        parse_format("search_replace").ok(),
        Some(EditFormat::SearchReplace)
    );
    assert_eq!(parse_format("udiff").ok(), Some(EditFormat::Udiff));
    assert!(parse_format("diff").is_err());
}

#[test]
fn the_environment_table_requires_pinned_images() {
    let digest = "a".repeat(64);
    let good = format!(
        r#"{{"t1": {{"image": "rust@sha256:{digest}", "command": ["cargo", "test"],
            "report_path": "r.json", "report_format": "libtest_json"}}}}"#
    );
    let envs = load_envs(&good).expect("loads");
    assert_eq!(envs.len(), 1);
    assert_eq!(envs["t1"].command, vec!["cargo", "test"]);
    let unpinned = good.replace(&format!("@sha256:{digest}"), ":latest");
    assert!(load_envs(&unpinned).is_err(), "a tag is not a pin");
}

#[test]
fn provenance_helpers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let f = dir.path().join("m.gguf");
    std::fs::write(&f, b"abc").expect("write");
    assert_eq!(
        sha256_file(&f).expect("hash"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    for (name, want) in [
        ("qwen2.5-coder-1.5b-instruct-Q4_K_M.gguf", "Q4_K_M"),
        ("model-f16.gguf", "F16"),
        ("model.apr", "unknown"),
    ] {
        assert_eq!(quant_from_name(Path::new(name)), want, "{name}");
    }
}

#[test]
fn contamination_flags_parse() {
    let mut args = EVAL.to_vec();
    args.extend([
        "--train-corpus",
        "a.txt",
        "--train-corpus",
        "b.txt",
        "--canary-probe",
    ]);
    let cli = parse(&args).expect("parses");
    let Commands::Code {
        action:
            Some(CodeCommands::Eval {
                train_corpus,
                canary_probe,
                ..
            }),
        ..
    } = *cli.command
    else {
        panic!("expected apr code eval");
    };
    assert_eq!(train_corpus.len(), 2);
    assert!(canary_probe);
    let Commands::Code {
        action:
            Some(CodeCommands::Eval {
                train_corpus,
                canary_probe,
                ..
            }),
        ..
    } = *parse(EVAL).expect("parses").command
    else {
        panic!("expected apr code eval");
    };
    assert!(
        train_corpus.is_empty() && !canary_probe,
        "both off by default"
    );
}

#[test]
fn a_planted_leak_is_flagged_from_the_corpus_files() {
    let leak =
        "let total = items.iter().map(|x| x.price * x.qty).sum::<u64>() + shipping_fee(region);";
    let manifest = format!(
        "{}\n{}\n",
        serde_json::json!({"id": "clean", "kind": "function", "release_date": "2025-03-01",
            "prompt": "p", "snapshot": {"files": {"src/lib.rs": "x"}},
            "hidden_tests": {"fail_to_pass": ["f"], "pass_to_pass": [], "files": {}},
            "gold": "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-x\n+other_body(1, 2)\n"}),
        serde_json::json!({"id": "leaked", "kind": "function", "release_date": "2025-03-01",
            "prompt": "p", "snapshot": {"files": {"src/lib.rs": "x"}},
            "hidden_tests": {"fail_to_pass": ["f"], "pass_to_pass": [], "files": {}},
            "gold": format!("--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-x\n+{leak}\n")}),
    );
    let set = load(&manifest, "2026-10-14").expect("tasks load");
    let dir = tempfile::tempdir().expect("tempdir");
    let clean = dir.path().join("clean.txt");
    let dirty = dir.path().join("dirty.txt");
    std::fs::write(&clean, "nothing to see here\n").expect("write");
    std::fs::write(&dirty, format!("// vendored\n{leak}\n")).expect("write");
    let (sha_a, none) = corpus_overlap(&[clean.clone()], set.tasks()).expect("reads");
    assert!(none.is_empty());
    let (sha_b, hit) = corpus_overlap(&[clean, dirty], set.tasks()).expect("reads");
    assert_eq!(hit, vec!["leaked".to_string()]);
    assert_ne!(sha_a, sha_b, "the digest names the corpus");
    assert!(corpus_overlap(&[dir.path().join("absent")], set.tasks()).is_err());
}

fn receipt_json(rows: &[(&str, &str)]) -> receipt::Receipt {
    let labels = [
        "RESOLVED",
        "FAILED",
        "FORMAT_ERROR",
        "EMPTY",
        "TIMEOUT",
        "SANDBOX_ERROR",
        "NO_TESTS",
    ];
    let counts: serde_json::Map<String, serde_json::Value> = labels
        .iter()
        .map(|l| {
            let c = rows.iter().filter(|(_, v)| v == l).count();
            ((*l).to_string(), c.into())
        })
        .collect();
    let per_task: Vec<serde_json::Value> = rows
        .iter()
        .map(|(id, v)| {
            serde_json::json!({"id": id, "sample": 0, "attempt": 1, "verdict": v,
                "tests_executed": 1, "tests_passed": 0, "f2p_passed": 0, "p2p_failed": 0,
                "edit_format_ok": true, "tokens_in": 1, "tokens_out": 1, "turns": 1,
                "wall_ms": 5, "report_sha256": null})
        })
        .collect();
    let metrics = serde_json::json!({"tasks": rows.len(), "n": 1, "pass_at_1": 0.5,
        "pass_at_k": null, "wilson95": [0.1, 0.9], "format_conformance": 1.0,
        "post_cutoff_pass_at_1": null, "post_cutoff_tasks": 0, "counts_by_label": counts});
    serde_json::from_value(serde_json::json!({
        "schema": receipt::SCHEMA, "suite": "smoke", "task_set_digest": "d",
        "model": {"path": "m.gguf", "sha256": "a", "quant": "Q4_K_M", "backend": "cpu"},
        "apr": {"version": "0.71.3", "git_sha": "b"}, "harness_git_sha": "c",
        "sandbox": {"engine_version": "docker", "image_digests": {}}, "seed": 0,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "n": 1, "max_tokens": 8, "max_turns": 1},
        "started_at": "t", "wall_s": 1.0, "per_task": per_task, "counts_by_label": counts,
        "metrics": {"attempt_1": metrics, "attempt_2": metrics}
    }))
    .expect("a receipt deserializes")
}

/// #5288: `--gaps FILE` parses, is off by default, and writes the ranked list.
#[test]
fn gaps_flag_writes_the_ranked_list() {
    let mut args = EVAL.to_vec();
    args.extend(["--gaps", "g.md"]);
    let Commands::Code {
        action: Some(CodeCommands::Eval { gaps, .. }),
        ..
    } = *parse(&args).expect("parses").command
    else {
        panic!("expected apr code eval");
    };
    assert_eq!(gaps.map(|g| g.to_path_buf()), Some(PathBuf::from("g.md")));
    let Commands::Code {
        action: Some(CodeCommands::Eval { gaps, .. }),
        ..
    } = *parse(EVAL).expect("parses").command
    else {
        panic!("expected apr code eval");
    };
    assert!(gaps.is_none(), "off by default");

    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("gaps.md");
    let r = receipt_json(&[("a", "FAILED"), ("b", "RESOLVED"), ("c", "TIMEOUT")]);
    assert_eq!(write_gaps(&r, &out).expect("writes"), 2);
    let text = std::fs::read_to_string(&out).expect("reads");
    assert!(text.contains("| 1 | `a` | 1 / 1 | FAILED 1 |"), "{text}");
    assert!(text.contains("| 2 | `c` | 1 / 1 | TIMEOUT 1 |"), "{text}");

    let empty = dir.path().join("empty.md");
    assert!(write_gaps(&receipt_json(&[]), &empty).is_err());
    assert!(!empty.exists(), "a refused receipt writes nothing");
}
