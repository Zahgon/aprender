//! `apr-qa run` stops before executing anything when the playbook, the failure
//! policy or the integrity lock is wrong, and says which (#5714).
//! Each case runs the built binary in a scratch cwd so `playbooks/playbook.lock.yaml`
//! is under test control.

use std::path::Path;
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_apr-qa");
const NAME: &str = "gpt2-medium-dim-smoke";

fn apr_qa(cwd: &Path, args: &[&str]) -> Output {
    Command::new(BIN)
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("spawn apr-qa")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

/// Copy a real playbook into `<cwd>/playbooks/`.
fn seed(cwd: &Path) -> String {
    let src = format!(
        "{}/../../playbooks/models/{NAME}.playbook.yaml",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::create_dir_all(cwd.join("playbooks")).expect("mkdir");
    let rel = format!("playbooks/{NAME}.playbook.yaml");
    std::fs::copy(&src, cwd.join(&rel)).expect("copy playbook");
    rel
}

#[test]
fn missing_playbook_exits_1_with_a_load_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let o = apr_qa(tmp.path(), &["run", "nope.playbook.yaml"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        stderr(&o).contains("Error loading playbook"),
        "{}",
        stderr(&o)
    );
}

#[test]
fn unknown_failure_policy_exits_1_naming_it() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let pb = seed(tmp.path());
    let o = apr_qa(
        tmp.path(),
        &[
            "run",
            &pb,
            "--no-integrity-check",
            "--failure-policy",
            "bogus",
        ],
    );
    assert_eq!(o.status.code(), Some(1));
    assert!(
        stderr(&o).contains("Unknown failure policy: bogus"),
        "{}",
        stderr(&o)
    );
}

#[test]
fn edited_playbook_fails_the_integrity_lock() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let pb = seed(tmp.path());
    let lock = apr_qa(tmp.path(), &["lock-playbooks"]);
    assert!(lock.status.success(), "lock-playbooks: {}", stderr(&lock));

    let path = tmp.path().join(&pb);
    let mut text = std::fs::read_to_string(&path).expect("read");
    text.push_str("# tampered\n");
    std::fs::write(&path, text).expect("write");

    let o = apr_qa(tmp.path(), &["run", &pb]);
    assert_eq!(o.status.code(), Some(1), "tampering must stop the run");
    assert!(
        !stdout(&o).contains("Running playbook"),
        "a tampered playbook must not start executing"
    );
    assert!(stderr(&o).contains("[INTEGRITY]"), "{}", stderr(&o));
    assert!(
        stderr(&o).contains("does not match lock file"),
        "{}",
        stderr(&o)
    );
}

#[test]
fn integrity_check_can_be_bypassed_only_explicitly() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let pb = seed(tmp.path());
    // Tampered lock: valid lock first, then edit; the bypass flag must skip the check
    // so the run reaches the failure-policy guard instead of [INTEGRITY].
    apr_qa(tmp.path(), &["lock-playbooks"]);
    let path = tmp.path().join(&pb);
    let mut text = std::fs::read_to_string(&path).expect("read");
    text.push_str("# tampered\n");
    std::fs::write(&path, text).expect("write");

    let o = apr_qa(
        tmp.path(),
        &[
            "run",
            &pb,
            "--no-integrity-check",
            "--failure-policy",
            "bogus",
        ],
    );
    assert!(!stderr(&o).contains("[INTEGRITY]"), "{}", stderr(&o));
    assert!(
        stderr(&o).contains("Unknown failure policy"),
        "{}",
        stderr(&o)
    );
}
