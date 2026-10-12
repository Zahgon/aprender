//! Falsification tests for contracts/apr-recipe-v1.yaml RCP-009..012 (PMAT-5230).

use super::*;
use clap::Parser;

const FINETUNE: &str = "\
recipe: apr.recipe/v1
kind: finetune
name: qwen35-lora
base: { model: base.apr }
method: lora
lora: { rank: 16, alpha: 32 }
data: { train: train.jsonl, eval: eval.jsonl }
eval: { metric: accuracy }
output: { path: out/adapter.apr }
";

const DISTILL: &str = "\
recipe: apr.recipe/v1
kind: distill
name: qwen35-kd
base: { model: student.apr }
teacher: { model: teacher.apr }
data: { train: train.jsonl }
distillation: { temperature: 2.0, alpha: 0.5 }
training: { epochs: 7, learning_rate: 0.001 }
output: { path: out/student.apr }
";

const MERGE: &str = "\
recipe: apr.recipe/v1
kind: merge
name: two-way
merge: { strategy: weighted, models: [a.apr, b.apr], weights: [0.75, 0.25] }
output: { path: out/merged.apr }
";

fn recipe(yaml: &str) -> Recipe {
    Recipe::from_yaml(yaml).expect("fixture recipe is valid")
}

fn rules(err: RecipeError) -> Vec<&'static str> {
    match err {
        RecipeError::Invalid(v) => v.iter().map(|x| x.rule).collect(),
        other => panic!("expected Invalid, got {other}"),
    }
}

fn sets(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

// ── RCP-009: the recipe's kind is the command's ────────────────────────────

#[test]
fn falsify_rcp_run_009_kind_mismatch_is_refused() {
    let ft = recipe(FINETUNE);
    assert!(check_kind(&ft, RecipeKind::Finetune).is_ok());
    for other in [RecipeKind::Distill, RecipeKind::Merge] {
        let err = check_kind(&ft, other).expect_err("finetune recipe on another command");
        assert_eq!(rules(err), vec!["RCP-009"]);
    }
    let m = recipe(MERGE);
    assert_eq!(
        rules(check_kind(&m, RecipeKind::Finetune).expect_err("merge on finetune")),
        vec!["RCP-009"]
    );
}

#[test]
fn falsify_rcp_run_009_refused_before_the_command_runs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("r.yaml");
    std::fs::write(&path, MERGE).expect("write recipe");
    let mut ran = false;
    let err = run(&path, &[], RecipeKind::Finetune, false, |_| {
        ran = true;
        Ok(())
    })
    .expect_err("kind mismatch");
    assert!(!ran, "the command must not run on a kind mismatch");
    assert!(err.to_string().contains("RCP-009"), "{err}");
    assert!(
        !receipt_path(dir.path().join("out/merged.apr").to_str().expect("utf-8")).exists(),
        "no receipt for a run that never started"
    );
}

// ── RCP-010: --set edits the recipe, is validated again and is listed ──────

#[test]
fn falsify_rcp_run_010_set_changes_the_recipe_and_its_hash() {
    let ft = recipe(FINETUNE);
    let (changed, overrides) =
        apply_sets(&ft, &sets(&["training.epochs=5", "lora.rank=8"])).expect("valid sets");
    assert_eq!(changed.training.as_ref().map(|t| t.epochs), Some(5));
    assert_eq!(changed.lora.as_ref().map(|l| l.rank), Some(8));
    assert_ne!(changed.hash(), ft.hash());
    assert_eq!(
        overrides,
        vec![
            Override {
                key: "training.epochs".into(),
                from: serde_json::json!(3),
                to: serde_json::json!(5),
            },
            Override {
                key: "lora.rank".into(),
                from: serde_json::json!(16),
                to: serde_json::json!(8),
            },
        ]
    );
}

#[test]
fn falsify_rcp_run_010_set_is_validated_again() {
    let ft = recipe(FINETUNE);
    assert_eq!(
        rules(apply_sets(&ft, &sets(&["training.epochs=0"])).expect_err("epochs 0")),
        vec!["RCP-005"]
    );
    // method full with the lora section still present breaks RCP-004.
    assert_eq!(
        rules(apply_sets(&ft, &sets(&["method=full"])).expect_err("full + lora")),
        vec!["RCP-004"]
    );
}

#[test]
fn falsify_rcp_run_010_bad_sets_do_not_parse() {
    let ft = recipe(FINETUNE);
    for bad in [
        "training.epoch=5", // typo: unknown key
        "noequals",
        "=5",
        "training..epochs=5",
        "method=auto", // not a recipe method
        "training.epochs=five",
    ] {
        let err = apply_sets(&ft, &sets(&[bad])).expect_err(bad);
        assert!(matches!(err, RecipeError::Parse(_)), "{bad}: {err}");
    }
}

#[test]
fn falsify_rcp_run_010_identity_set_keeps_the_hash() {
    let ft = recipe(FINETUNE);
    let (same, overrides) = apply_sets(&ft, &sets(&["training.epochs=3"])).expect("valid");
    assert_eq!(same.hash(), ft.hash());
    assert_eq!(overrides.len(), 1, "a no-op set is still listed");
}

// ── RCP-011: the run receipt ───────────────────────────────────────────────

fn write_fixture(dir: &Path) -> PathBuf {
    std::fs::write(dir.join("base.apr"), b"base-bytes").expect("base");
    std::fs::write(dir.join("train.jsonl"), b"{\"x\":1}\n").expect("data");
    let yaml = FINETUNE
        .replace("base.apr", dir.join("base.apr").to_str().expect("utf-8"))
        .replace(
            "train.jsonl",
            dir.join("train.jsonl").to_str().expect("utf-8"),
        )
        .replace(
            "out/adapter.apr",
            dir.join("out/adapter.apr").to_str().expect("utf-8"),
        );
    let path = dir.join("r.yaml");
    std::fs::write(&path, yaml).expect("recipe");
    path
}

fn read_receipt(dir: &Path) -> serde_json::Value {
    let p = receipt_path(dir.join("out/adapter.apr").to_str().expect("utf-8"));
    let text = std::fs::read_to_string(&p).expect("receipt written");
    serde_json::from_str(&text).expect("receipt is JSON")
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn falsify_rcp_run_011_success_receipt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path());
    run(
        &path,
        &sets(&["training.seed=7"]),
        RecipeKind::Finetune,
        false,
        |_| Ok(()),
    )
    .expect("run ok");
    let r = read_receipt(dir.path());
    let loaded = Recipe::load(&path).expect("load");
    assert_eq!(r["receipt"], RECEIPT_VERSION);
    assert_eq!(r["status"], "success");
    assert!(r.get("error").is_none());
    assert_eq!(r["kind"], "finetune");
    assert_eq!(r["recipe_name"], "qwen35-lora");
    assert_eq!(r["recipe_file_hash"], loaded.hash());
    assert_ne!(r["recipe_hash"], r["recipe_file_hash"], "--set changed it");
    assert_eq!(r["seed"], 7);
    assert_eq!(r["overrides"][0]["key"], "training.seed");
    assert_eq!(r["apr_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(r["git_sha"], env!("APR_GIT_SHA"));
    assert_eq!(r["inputs"][0]["role"], "base");
    assert_eq!(r["inputs"][0]["sha256"], sha256_hex(b"base-bytes"));
    assert_eq!(r["inputs"][1]["sha256"], sha256_hex(b"{\"x\":1}\n"));
    assert!(r["started_at"].as_str().is_some_and(|s| s.ends_with('Z')));
    assert!(r["finished_at"].as_str().is_some_and(|s| s.ends_with('Z')));
}

#[test]
fn falsify_rcp_run_011_failed_run_still_writes_a_failed_receipt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path());
    let err = run(&path, &[], RecipeKind::Finetune, false, |_| {
        Err(CliError::ValidationFailed("boom".into()))
    })
    .expect_err("command failed");
    assert!(err.to_string().contains("boom"));
    let r = read_receipt(dir.path());
    assert_eq!(r["status"], "failed");
    assert!(r["error"].as_str().is_some_and(|e| e.contains("boom")));
    assert_eq!(r["recipe_hash"], r["recipe_file_hash"], "no --set");
}

#[test]
fn falsify_rcp_run_011_missing_input_has_no_digest() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path());
    std::fs::remove_file(dir.path().join("base.apr")).expect("rm base");
    run(&path, &[], RecipeKind::Finetune, false, |_| Ok(())).expect("run ok");
    let r = read_receipt(dir.path());
    assert!(r["inputs"][0]["sha256"].is_null(), "no invented digest");
    assert!(r["inputs"][1]["sha256"].is_string());
}

#[test]
fn falsify_rcp_run_011_plan_mode_writes_no_receipt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_fixture(dir.path());
    let mut ran = false;
    run(&path, &[], RecipeKind::Finetune, true, |_| {
        ran = true;
        Ok(())
    })
    .expect("plan ok");
    assert!(ran);
    assert!(!receipt_path(dir.path().join("out/adapter.apr").to_str().expect("utf-8")).exists());
}

// ── RCP-012: the recipe → arguments mapping, field by field ────────────────

#[test]
fn falsify_rcp_run_012_finetune_mapping() {
    let p = plan(&recipe(FINETUNE)).expect("plan");
    assert_eq!(
        p,
        Planned::Finetune(FinetuneArgs {
            model: "base.apr".into(),
            method: "lora",
            rank: Some(16),
            data: "train.jsonl".into(),
            output: "out/adapter.apr".into(),
            epochs: 3,
            learning_rate: 0.0002,
            max_seq_len: 512,
        })
    );
    let (full, _) = apply_sets(
        &recipe(FINETUNE),
        &sets(&["method=full", "lora=null", "training.max_seq_len=128"]),
    )
    .expect("full fine-tune");
    let Planned::Finetune(a) = plan(&full).expect("plan") else {
        panic!("finetune plan");
    };
    assert_eq!((a.method, a.rank, a.max_seq_len), ("full", None, 128));
}

#[test]
fn falsify_rcp_run_012_distill_mapping() {
    let p = plan(&recipe(DISTILL)).expect("plan");
    assert_eq!(
        p,
        Planned::Distill(DistillArgs {
            teacher: "teacher.apr".into(),
            student: "student.apr".into(),
            data: "train.jsonl".into(),
            output: "out/student.apr".into(),
            temperature: 2.0,
            alpha: 0.5,
            epochs: 7,
        })
    );
}

#[test]
fn falsify_rcp_run_012_merge_mapping() {
    let p = plan(&recipe(MERGE)).expect("plan");
    assert_eq!(
        p,
        Planned::Merge(MergeArgs {
            models: vec!["a.apr".into(), "b.apr".into()],
            strategy: "weighted",
            weights: Some(vec![0.75, 0.25]),
            output: "out/merged.apr".into(),
        })
    );
    for (s, name) in [
        ("average", "average"),
        ("slerp", "slerp"),
        ("ties", "ties"),
        ("dare", "dare"),
    ] {
        let mut sets_for = vec![format!("merge.strategy={s}")];
        if matches!(s, "average" | "ties") {
            sets_for.push("merge.weights=null".into());
        }
        let (r, _) = apply_sets(&recipe(MERGE), &sets_for).expect(s);
        let Planned::Merge(a) = plan(&r).expect("plan") else {
            panic!("merge plan");
        };
        assert_eq!(a.strategy, name);
    }
}

#[test]
fn falsify_rcp_run_012_inputs_and_not_passed() {
    let d = plan(&recipe(DISTILL)).expect("plan");
    let roles: Vec<String> = d.inputs().into_iter().map(|(r, _)| r).collect();
    assert_eq!(roles, ["teacher", "base", "data.train"]);
    assert_eq!(
        not_passed(&recipe(FINETUNE)),
        [
            "lora.alpha",
            "lora.dropout",
            "training.batch_size",
            "training.seed",
            "data.eval",
            "eval"
        ]
    );
    assert!(not_passed(&recipe(MERGE)).is_empty());
}

// ── clap: covered flags are refused beside --recipe ────────────────────────

/// `Commands` is a very large enum; building the clap command tree needs more
/// stack than the 2 MiB a test thread gets by default.
fn parses(args: &[&str]) -> bool {
    let args: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || crate::Cli::try_parse_from(args).is_ok())
        .expect("spawn")
        .join()
        .expect("join")
}

#[test]
fn falsify_rcp_run_010_covered_flags_conflict_with_recipe() {
    assert!(parses(&["apr", "merge", "--recipe", "r.yaml"]));
    assert!(parses(&[
        "apr",
        "merge",
        "--recipe",
        "r.yaml",
        "--set",
        "merge.strategy=slerp"
    ]));
    assert!(parses(&[
        "apr", "merge", "--recipe", "r.yaml", "--seed", "1"
    ]));
    assert!(!parses(&[
        "apr", "merge", "a.apr", "b.apr", "--recipe", "r.yaml"
    ]));
    assert!(!parses(&[
        "apr", "merge", "--recipe", "r.yaml", "-o", "x.apr"
    ]));
    assert!(!parses(&[
        "apr",
        "merge",
        "--recipe",
        "r.yaml",
        "--strategy",
        "average"
    ]));
    // clap waives `requires = "recipe"` once a flag that conflicts with
    // --recipe is present, so this parses; dispatch refuses it instead.
    assert!(parses(&[
        "apr", "merge", "a.apr", "b.apr", "-o", "x", "--set", "k=v"
    ]));
    assert!(!parses(&["apr", "merge", "--plan", "--set", "k=v"]));
    assert!(parses(&["apr", "merge", "a.apr", "b.apr", "-o", "x.apr"]));
    assert!(
        !parses(&["apr", "merge", "a.apr", "b.apr"]),
        "output still required"
    );

    assert!(parses(&[
        "apr",
        "distill",
        "--recipe",
        "r.yaml",
        "--backend",
        "fixture"
    ]));
    assert!(!parses(&["apr", "distill", "t.apr", "--recipe", "r.yaml"]));
    assert!(!parses(&[
        "apr",
        "distill",
        "--recipe",
        "r.yaml",
        "--temperature",
        "3.0"
    ]));
    assert!(!parses(&[
        "apr", "distill", "--recipe", "r.yaml", "--config", "c.yaml"
    ]));
}

#[cfg(feature = "training")]
#[test]
fn falsify_rcp_run_010_finetune_flags_conflict_with_recipe() {
    assert!(parses(&[
        "apr", "finetune", "--recipe", "r.yaml", "--vram", "8"
    ]));
    for flag in [
        &["m.apr"][..],
        &["--method", "lora"],
        &["--rank", "8"],
        &["--data", "d.jsonl"],
        &["-o", "o.apr"],
        &["--epochs", "3"],
        &["--learning-rate", "0.001"],
        &["--max-seq-len", "64"],
    ] {
        let mut args = vec!["apr", "finetune", "--recipe", "r.yaml"];
        args.extend_from_slice(flag);
        assert!(!parses(&args), "{flag:?} beside --recipe must be refused");
    }
}

#[test]
fn falsify_rcp_run_010_set_without_recipe_is_refused_at_dispatch() {
    assert!(sets_need_recipe(&[]).is_ok());
    let err = sets_need_recipe(&sets(&["k=v"])).expect_err("--set alone");
    assert!(err.to_string().contains("RCP-010"), "{err}");
}
