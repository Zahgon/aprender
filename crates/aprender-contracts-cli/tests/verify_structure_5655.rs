//! `pv verify-structure` (`commands::verify_structure::run`): contract discovery (direct and
//! nested), PASS/PARTIAL verdict, and the config / model-file branches (#5655).
#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const STEMS: [&str; 3] = [
    "apr-architecture-schema-v1",
    "model-config-algebra-v1",
    "tensor-shape-flow-v1",
];

fn repo_contracts() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

fn pv(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pv"))
        .arg("verify-structure")
        .args(args)
        .output()
        .expect("spawn pv")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn seed(dir: &Path, stems: &[&str]) {
    for s in stems {
        let name = format!("{s}.yaml");
        std::fs::copy(repo_contracts().join(&name), dir.join(&name)).expect("copy contract");
    }
}

#[test]
fn all_three_contracts_found_passes() {
    let tmp = tempfile::tempdir().expect("tmp");
    seed(tmp.path(), &STEMS);
    let o = pv(&[tmp.path().to_str().expect("utf8")]);
    let s = stdout(&o);
    assert!(o.status.success(), "{s}");
    for stem in STEMS {
        assert!(s.contains(&format!("✓ {stem}:")), "{stem} not found: {s}");
    }
    assert!(
        s.contains("Result: PASS — all 3 architecture contracts found"),
        "{s}"
    );
}

#[test]
fn empty_dir_is_partial_with_three_missing() {
    let tmp = tempfile::tempdir().expect("tmp");
    let o = pv(&[tmp.path().to_str().expect("utf8")]);
    let s = stdout(&o);
    assert!(o.status.success(), "{s}");
    for stem in STEMS {
        assert!(s.contains(&format!("✗ {stem}: not found")), "{s}");
    }
    assert!(
        s.contains("Result: PARTIAL — 0/3 contracts found, 3 missing"),
        "{s}"
    );
}

#[test]
fn nested_contract_is_found_and_counted_as_one_of_three() {
    let tmp = tempfile::tempdir().expect("tmp");
    let deep = tmp.path().join("a").join("b");
    std::fs::create_dir_all(&deep).expect("mkdir");
    seed(&deep, &STEMS[..1]);
    let s = stdout(&pv(&[tmp.path().to_str().expect("utf8")]));
    assert!(s.contains("✓ apr-architecture-schema-v1:"), "{s}");
    assert!(s.contains("✗ model-config-algebra-v1: not found"), "{s}");
    assert!(
        s.contains("Result: PARTIAL — 1/3 contracts found, 2 missing"),
        "{s}"
    );
}

#[test]
fn no_config_prints_the_hint_and_a_missing_config_is_named() {
    let tmp = tempfile::tempdir().expect("tmp");
    let dir = tmp.path().to_str().expect("utf8");
    let s = stdout(&pv(&[dir]));
    assert!(s.contains("No --config provided."), "{s}");
    let s = stdout(&pv(&[dir, "--config", "/nonexistent/cfg-5655.json"]));
    assert!(
        s.contains("Config file not found: /nonexistent/cfg-5655.json"),
        "{s}"
    );
}

#[test]
fn an_unparsable_config_fails_the_command() {
    let tmp = tempfile::tempdir().expect("tmp");
    let cfg = tmp.path().join("cfg.json");
    std::fs::write(&cfg, "{not json").expect("write");
    let o = pv(&[
        tmp.path().to_str().expect("utf8"),
        "--config",
        cfg.to_str().expect("utf8"),
    ]);
    assert!(!o.status.success(), "{}", stdout(&o));
}
