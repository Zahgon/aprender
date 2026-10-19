//! `load_documents_parallel` (reached via `aprender-rag index --jobs N`, N > 1):
//! every readable file becomes a document, unreadable ones are counted and
//! reported, and an all-unreadable corpus is an error (#5633).

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

fn index(src: &Path, out: &Path, jobs: &str) -> assert_cmd::assert::Assert {
    Command::cargo_bin("aprender-rag")
        .expect("bin")
        .args(["index", "--path"])
        .arg(src)
        .arg("--output")
        .arg(out)
        .args(["--jobs", jobs])
        .assert()
}

fn corpus(good: usize, bad: usize) -> (TempDir, std::path::PathBuf) {
    let tmp = TempDir::new().expect("tmp");
    let src = tmp.path().join("docs");
    fs::create_dir(&src).expect("mkdir");
    for i in 0..good {
        fs::write(
            src.join(format!("good{i}.txt")),
            format!("Document number {i} about ownership and borrowing."),
        )
        .expect("good");
    }
    for i in 0..bad {
        // Invalid UTF-8: the text loader cannot read it.
        fs::write(src.join(format!("bad{i}.txt")), [0xff, 0xfe, 0xfd, 0x80]).expect("bad");
    }
    (tmp, src)
}

fn chunk_count(out: &Path) -> usize {
    let idx: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out.join("index.json")).expect("read"))
            .expect("json");
    idx["chunks"].as_array().expect("chunks").len()
}

#[test]
fn parallel_load_ingests_every_file() {
    let (tmp, src) = corpus(6, 0);
    let out = tmp.path().join("idx");
    index(&src, &out, "4")
        .success()
        .stdout(predicate::str::contains("Loaded 6 documents"))
        .stdout(predicate::str::contains("failed").not());
    assert!(chunk_count(&out) >= 6);
}

#[test]
fn parallel_load_counts_unreadable_files_and_keeps_the_rest() {
    let (tmp, src) = corpus(3, 2);
    let out = tmp.path().join("idx");
    index(&src, &out, "4")
        .success()
        .stdout(predicate::str::contains("Loaded 3 documents (2 failed)"))
        .stderr(predicate::str::contains("Warning: failed to load"));
    assert!(out.join("index.json").exists());
}

#[test]
fn parallel_load_fails_when_nothing_is_readable() {
    let (tmp, src) = corpus(0, 3);
    let out = tmp.path().join("idx");
    index(&src, &out, "4")
        .failure()
        .stderr(predicate::str::contains(
            "All files failed to load (3 errors)",
        ));
    assert!(!out.join("index.json").exists());
}

#[test]
fn parallel_and_sequential_load_agree_on_the_document_count() {
    let (tmp, src) = corpus(5, 1);
    for jobs in ["1", "3"] {
        let out = tmp.path().join(format!("idx{jobs}"));
        index(&src, &out, jobs)
            .success()
            .stdout(predicate::str::contains("Loaded 5 documents (1 failed)"));
    }
}
