//! `save_index` (reached via `aprender-rag index`) writes `index.json` always,
//! `manifest.json` only with `--manifest`, and creates missing output dirs (#5662).

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

fn index(src: &Path, out: &Path, extra: &[&str]) -> assert_cmd::assert::Assert {
    Command::cargo_bin("aprender-rag")
        .expect("bin")
        .args(["index", "--path"])
        .arg(src)
        .arg("--output")
        .arg(out)
        .args(extra)
        .assert()
}

fn corpus() -> (TempDir, std::path::PathBuf) {
    let tmp = TempDir::new().expect("tmp");
    let src = tmp.path().join("docs");
    fs::create_dir(&src).expect("mkdir");
    fs::write(
        src.join("a.txt"),
        "Rust is a systems language about ownership.",
    )
    .expect("a");
    fs::write(src.join("b.md"), "# Title\n\nMarkdown about borrowing.").expect("b");
    (tmp, src)
}

fn read_json(p: &Path) -> serde_json::Value {
    serde_json::from_str(&fs::read_to_string(p).expect("read")).expect("json")
}

#[test]
fn writes_index_json_and_creates_nested_output_dir() {
    let (tmp, src) = corpus();
    let out = tmp.path().join("deep/nested/idx");
    index(&src, &out, &[])
        .success()
        .stdout(predicate::str::contains("Index saved to:"));
    let idx = read_json(&out.join("index.json"));
    assert!(!idx["chunks"].as_array().expect("chunks").is_empty());
    assert_eq!(
        idx["chunks"].as_array().expect("c").len(),
        idx["embeddings"].as_array().expect("e").len()
    );
}

#[test]
fn no_manifest_unless_requested() {
    let (tmp, src) = corpus();
    let out = tmp.path().join("idx");
    index(&src, &out, &[])
        .success()
        .stdout(predicate::str::contains("Manifest saved to:").not());
    assert!(out.join("index.json").exists());
    assert!(!out.join("manifest.json").exists());
}

#[test]
fn manifest_flag_writes_manifest_matching_index() {
    let (tmp, src) = corpus();
    let out = tmp.path().join("idx");
    index(&src, &out, &["--manifest"])
        .success()
        .stdout(predicate::str::contains("Manifest saved to:"));
    let m = read_json(&out.join("manifest.json"));
    let idx = read_json(&out.join("index.json"));
    assert_eq!(m["file_count"], 2);
    assert_eq!(m["chunk_count"], idx["chunks"].as_array().expect("c").len());
    let exts: Vec<&str> = m["files"]
        .as_array()
        .expect("files")
        .iter()
        .filter_map(|f| f["extension"].as_str())
        .collect();
    assert!(exts.contains(&"txt") && exts.contains(&"md"), "{exts:?}");
}

#[test]
fn output_path_that_is_a_file_fails() {
    let (tmp, src) = corpus();
    let out = tmp.path().join("occupied");
    fs::write(&out, "not a dir").expect("occupy");
    index(&src, &out, &[]).failure();
}
