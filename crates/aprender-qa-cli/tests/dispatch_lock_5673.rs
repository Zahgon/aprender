//! `dispatch` routes `lock-playbooks` to the lock writer and reports success (#5673).
//! Only non-exiting arms are testable in-process; the exit(1) arms are not.

use aprender_qa_cli::cli::{dispatch, Cli};
use clap::Parser;

fn run(args: &[&str]) {
    let mut argv = vec!["apr-qa"];
    argv.extend_from_slice(args);
    let cli = Cli::try_parse_from(argv).expect("args parse");
    dispatch(cli.command);
}

#[test]
fn lock_playbooks_writes_one_entry_per_playbook_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("alpha.playbook.yaml"), "name: alpha\n").expect("write");
    std::fs::create_dir(dir.path().join("sub")).expect("mkdir");
    std::fs::write(dir.path().join("sub/beta.playbook.yaml"), "name: beta\n").expect("write");
    std::fs::write(dir.path().join("ignored.yaml"), "x: 1\n").expect("write");
    let out = dir.path().join("lock.yaml");

    run(&[
        "lock-playbooks",
        dir.path().to_str().expect("utf8"),
        "--output",
        out.to_str().expect("utf8"),
    ]);

    let lock = std::fs::read_to_string(&out).expect("dispatch must write the lock file");
    assert!(lock.contains("alpha"), "alpha locked: {lock}");
    assert!(lock.contains("beta"), "nested playbook locked: {lock}");
    assert!(
        !lock.contains("ignored"),
        "non-playbook yaml skipped: {lock}"
    );
    assert!(lock.contains("sha256"), "entries carry a hash: {lock}");
}

#[test]
fn lock_playbooks_hash_tracks_file_content() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pb = dir.path().join("p.playbook.yaml");
    let out = dir.path().join("lock.yaml");
    let d = dir.path().to_str().expect("utf8");
    let o = out.to_str().expect("utf8");

    std::fs::write(&pb, "name: p\nv: 1\n").expect("write");
    run(&["lock-playbooks", d, "-o", o]);
    let first = std::fs::read_to_string(&out).expect("lock 1");
    std::fs::write(&pb, "name: p\nv: 2\n").expect("write");
    run(&["lock-playbooks", d, "-o", o]);
    let second = std::fs::read_to_string(&out).expect("lock 2");
    assert_ne!(
        first, second,
        "editing a playbook must change its locked hash"
    );
}
