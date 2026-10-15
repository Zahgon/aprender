//! FALSIFY-ACE-SBX-001..008 (contracts/apr-code-eval-sandbox-v1.yaml).
//!
//! The plan, the report path and the deadline/kill/cleanup sequence are checked here,
//! against a fake engine binary. The same equations against a real container engine are
//! the nightly's live probes (#5288).

use super::*;

const DIGEST: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn image() -> ImageRef {
    ImageRef::parse(&format!("ghcr.io/paiml/eval-rust@{DIGEST}")).expect("pinned")
}

fn spec() -> RunSpec {
    RunSpec {
        image: image(),
        task_dir: PathBuf::from("/srv/eval/tasks/t1"),
        command: vec![
            "cargo".into(),
            "test".into(),
            "--".into(),
            "-Z".into(),
            "unstable-options".into(),
        ],
        report_path: "report.json".into(),
        report_format: ReportFormat::LibtestJson,
        seed: 7,
        source_date_epoch: 1_740_787_200,
        limits: Limits::default(),
    }
}

fn argv() -> Vec<String> {
    plan(&spec(), "run-1", Some(Path::new("/users/eval"))).expect("plans")
}

/// Value following `flag` in `argv`, for every occurrence.
fn values<'a>(argv: &'a [String], flag: &str) -> Vec<&'a str> {
    argv.windows(2)
        .filter(|w| w[0] == flag)
        .map(|w| w[1].as_str())
        .collect()
}

/// The engine options, i.e. everything before the image reference.
fn options(argv: &[String]) -> &[String] {
    let i = argv
        .iter()
        .position(|a| a == image().as_str())
        .expect("image in argv");
    &argv[..i]
}

fn libtest(name: &str, event: &str) -> String {
    format!(r#"{{ "type": "test", "event": "{event}", "name": "{name}" }}"#)
}

fn stdout_with(report: &str) -> String {
    format!("building...\n{REPORT_BEGIN}\n{report}\n{REPORT_END}\n")
}

/// FALSIFY-ACE-SBX-001: the only network option is `none`, and no option can widen it.
#[test]
fn falsify_ace_sbx_001_network_refused() {
    let argv = argv();
    let opts = options(&argv);
    assert_eq!(values(opts, "--network"), ["none"]);
    for banned in [
        "--net",
        "--network-alias",
        "--add-host",
        "--dns",
        "--publish",
        "-p",
        "--privileged",
    ] {
        assert!(!opts.iter().any(|a| a == banned), "{banned}");
    }
    assert!(!opts
        .iter()
        .any(|a| a.starts_with("--network=") || a.starts_with("--net=")));
}

/// FALSIFY-ACE-SBX-002: a tag or a malformed digest is refused, and so is a pulled image
/// whose digest differs; the run never starts.
#[test]
fn falsify_ace_sbx_002_unpinned_image_refused() {
    let hex = &DIGEST["sha256:".len()..];
    for bad in [
        "python:3.12".to_string(),
        "python@sha256:abc".to_string(),
        format!("python@sha256:{}", hex.to_uppercase()),
        format!("@sha256:{hex}"),
        format!("a@b@sha256:{hex}"),
        format!("python@sha512:{hex}"),
        format!("python@sha256:{hex}0"),
    ] {
        assert_eq!(
            ImageRef::parse(&bad),
            Err(SandboxError::Unpinned(bad.clone())),
            "{bad}"
        );
    }
    let img = image();
    assert_eq!(img.digest(), DIGEST);
    assert!(verify_pulled(&img, &[format!("ghcr.io/paiml/eval-rust@{DIGEST}")]).is_ok());
    let other = format!("ghcr.io/paiml/eval-rust@sha256:{}", "f".repeat(64));
    assert_eq!(
        verify_pulled(&img, &[other]),
        Err(SandboxError::DigestMismatch(DIGEST.into()))
    );
    assert!(verify_pulled(&img, &[]).is_err());
    assert!(values(&argv(), "--pull").is_empty());
}

/// FALSIFY-ACE-SBX-003: every limit is set and finite; a zero limit is refused; a run past
/// its wall clock is killed and labelled; an OOM kill is labelled.
#[test]
fn falsify_ace_sbx_003_limits_enforced() {
    let argv = argv();
    let l = Limits::default();
    assert_eq!(values(&argv, "--memory"), [l.mem_bytes.to_string()]);
    assert_eq!(values(&argv, "--memory-swap"), [l.mem_bytes.to_string()]);
    assert_eq!(values(&argv, "--cpus"), ["2.000"]);
    assert_eq!(values(&argv, "--pids-limit"), ["512"]);
    assert_eq!(
        values(&argv, "--tmpfs"),
        [format!("/work:rw,exec,size={},mode=1777", l.disk_bytes)]
    );

    let zeroed: [(&str, fn(&mut Limits)); 5] = [
        ("wall", |l| l.wall = Duration::ZERO),
        ("millicpus", |l| l.millicpus = 0),
        ("mem_bytes", |l| l.mem_bytes = 0),
        ("pids", |l| l.pids = 0),
        ("disk_bytes", |l| l.disk_bytes = 0),
    ];
    for (name, zero) in zeroed {
        let mut s = spec();
        zero(&mut s.limits);
        assert_eq!(plan(&s, "r", None), Err(SandboxError::BadLimit(name)));
    }
    let mut s = spec();
    s.limits.millicpus = 1500;
    assert_eq!(
        values(&plan(&s, "r", None).expect("plans"), "--cpus"),
        ["1.500"]
    );

    let timeout = Observed {
        timed_out: true,
        stdout: stdout_with(&libtest("t", "ok")),
        ..Observed::default()
    };
    let r = finalize(ReportFormat::LibtestJson, &timeout);
    assert_eq!(
        (r.killed_by, r.tests_executed()),
        (Some(KilledBy::Timeout), 0)
    );
    let oom = Observed {
        oom_killed: true,
        exit_code: Some(137),
        ..Observed::default()
    };
    assert_eq!(
        finalize(ReportFormat::LibtestJson, &oom).killed_by,
        Some(KilledBy::Oom)
    );
    let signal = Observed {
        exit_code: None,
        ..Observed::default()
    };
    assert_eq!(
        finalize(ReportFormat::LibtestJson, &signal).killed_by,
        Some(KilledBy::Other)
    );
}

/// FALSIFY-ACE-SBX-004: the task dir is the only bind mount, read-only, and may not be a
/// system path, the engine socket, or the host home or a parent of it.
#[test]
fn falsify_ace_sbx_004_host_isolated() {
    let argv = argv();
    let opts = options(&argv);
    assert_eq!(
        values(opts, "--mount"),
        ["type=bind,source=/srv/eval/tasks/t1,target=/task,readonly"]
    );
    assert!(opts.iter().any(|a| a == "--read-only"));
    assert_eq!(values(opts, "--cap-drop"), ["ALL"]);
    assert_eq!(values(opts, "--security-opt"), ["no-new-privileges"]);
    for banned in [
        "-v",
        "--volume",
        "--volumes-from",
        "--device",
        "--pid",
        "--ipc",
        "--userns",
        "--cap-add",
    ] {
        assert!(!opts.iter().any(|a| a == banned), "{banned}");
    }
    let home = Some(Path::new("/users/eval"));
    for bad in [
        "tasks/t1",
        "/",
        "/users",
        "/users/eval",
        "/var/run/docker.sock",
        "/run/user/1000",
        "/etc",
        "/proc/1/root",
        "/srv/../etc",
        "/srv/t1,readonly=false",
        "/srv/t1,source=/",
        "/srv/docker.sock",
    ] {
        let mut s = spec();
        s.task_dir = PathBuf::from(bad);
        assert!(
            matches!(plan(&s, "r", home), Err(SandboxError::BadTaskDir(..))),
            "{bad}"
        );
    }
    let mut s = spec();
    s.task_dir = PathBuf::from("/users/eval/tasks/t1");
    assert!(plan(&s, "r", home).is_ok(), "a dir inside home is fine");
}

/// FALSIFY-ACE-SBX-005: the environment is exactly the fixed set; nothing from the host is
/// passed through, by name or by file.
#[test]
fn falsify_ace_sbx_005_env() {
    let argv = argv();
    let opts = options(&argv);
    let env: Vec<&str> = values(opts, "--env");
    let want: Vec<String> = environment(&spec())
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    assert_eq!(env, want);
    for (k, v) in [
        ("SEED", "7"),
        ("PYTHONHASHSEED", "7"),
        ("TZ", "UTC"),
        ("LANG", "C.UTF-8"),
    ] {
        assert!(env.contains(&format!("{k}={v}").as_str()), "{k}");
    }
    assert!(env.contains(&"SOURCE_DATE_EPOCH=1740787200"));
    for banned in ["-e", "--env-file", "--env-host"] {
        assert!(!opts.iter().any(|a| a == banned), "{banned}");
    }
    let keys: Vec<&str> = env
        .iter()
        .map(|kv| kv.split_once('=').expect("k=v").0)
        .collect();
    for secret in [
        "PATH",
        "USER",
        "SSH_AUTH_SOCK",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "ANTHROPIC_API_KEY",
    ] {
        assert!(!keys.contains(&secret), "{secret}");
    }
}

/// FALSIFY-ACE-SBX-006: the same spec plans the same argv; repeat reports that differ in
/// any outcome or status are a flake, naming what differs.
#[test]
fn falsify_ace_sbx_006_deterministic() {
    assert_eq!(argv(), argv());
    let report = |pairs: &[(&str, TestOutcome)]| SandboxReport {
        status: SandboxStatus::Ok,
        killed_by: None,
        outcomes: pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect(),
    };
    let a = report(&[("t1", TestOutcome::Passed), ("t2", TestOutcome::Failed)]);
    assert_eq!(agree(&[a.clone(), a.clone(), a.clone()]), Ok(()));
    assert_eq!(agree(&[]), Ok(()));

    let b = report(&[("t1", TestOutcome::Passed), ("t2", TestOutcome::Passed)]);
    assert_eq!(
        agree(&[a.clone(), a.clone(), b]),
        Err(Flake {
            run: 2,
            differing: vec!["t2".into()]
        })
    );
    let c = report(&[
        ("t1", TestOutcome::Passed),
        ("t2", TestOutcome::Failed),
        ("t3", TestOutcome::Passed),
    ]);
    assert_eq!(
        agree(&[a.clone(), c]),
        Err(Flake {
            run: 1,
            differing: vec!["t3".into()]
        })
    );
    let mut d = a.clone();
    d.status = SandboxStatus::Error("timeout".into());
    d.killed_by = Some(KilledBy::Timeout);
    assert_eq!(
        agree(&[a, d]),
        Err(Flake {
            run: 1,
            differing: vec!["<status>".into()]
        })
    );
}

/// FALSIFY-ACE-SBX-007: a run with no report, an empty report or an unreadable one is a
/// sandbox error with zero executed tests, whatever the exit code; failing tests with a
/// report are a normal run.
#[test]
fn falsify_ace_sbx_007_no_report_is_error() {
    for exit in [0, 1, 101] {
        let obs = Observed {
            exit_code: Some(exit),
            stdout: "test result: ok. 3 passed".into(),
            ..Observed::default()
        };
        let r = finalize(ReportFormat::LibtestJson, &obs);
        assert!(
            matches!(r.status, SandboxStatus::Error(_)) && r.tests_executed() == 0,
            "exit {exit}"
        );
    }
    let empty = Observed {
        exit_code: Some(0),
        stdout: stdout_with("  "),
        ..Observed::default()
    };
    assert!(matches!(
        finalize(ReportFormat::LibtestJson, &empty).status,
        SandboxStatus::Error(_)
    ));
    let bad_xml = Observed {
        exit_code: Some(0),
        stdout: stdout_with("<html/>"),
        ..Observed::default()
    };
    assert!(matches!(
        finalize(ReportFormat::Junit, &bad_xml).status,
        SandboxStatus::Error(_)
    ));
    let setup = Observed {
        exit_code: Some(125),
        stdout: stdout_with(&libtest("t", "ok")),
        ..Observed::default()
    };
    assert!(matches!(
        finalize(ReportFormat::LibtestJson, &setup).status,
        SandboxStatus::Error(_)
    ));

    let report = [libtest("a", "ok"), libtest("b", "failed")].join("\n");
    let failing = Observed {
        exit_code: Some(101),
        stdout: stdout_with(&report),
        ..Observed::default()
    };
    let r = finalize(ReportFormat::LibtestJson, &failing);
    assert_eq!(r.status, SandboxStatus::Ok);
    assert_eq!(r.outcomes.get("b"), Some(&TestOutcome::Failed));
    assert_eq!(r.tests_executed(), 2);
}

/// FALSIFY-ACE-SBX-008: every run gets a fresh name, a read-only root and a fresh tmpfs
/// workdir seeded from the read-only task dir; nothing persists between runs.
#[test]
fn falsify_ace_sbx_008_clean_slate() {
    let argv = argv();
    let opts = options(&argv);
    assert_eq!(values(opts, "--name"), ["run-1"]);
    assert_eq!(values(opts, "--workdir"), ["/work"]);
    assert!(values(opts, "--tmpfs")[0].starts_with("/work:"));
    assert!(!opts
        .iter()
        .any(|a| a == "--rm" || a.starts_with("--restart")));
    let tail = &argv[opts.len() + 1..];
    assert_eq!(&tail[..2], ["sh", "-c"]);
    assert!(tail[2].starts_with("cp -R /task/. /work/"));
    assert_eq!(
        &tail[3..],
        ["sh", "cargo", "test", "--", "-Z", "unstable-options"]
    );
    for bad in ["", "a b", "x;rm", "../x", "n\n"] {
        assert_eq!(
            plan(&spec(), bad, None),
            Err(SandboxError::BadSpec("run name")),
            "{bad:?}"
        );
    }
    for bad in ["", "/work/r.json", "../r.json"] {
        let mut s = spec();
        s.report_path = bad.into();
        assert!(
            matches!(plan(&s, "r", None), Err(SandboxError::BadSpec(_))),
            "{bad}"
        );
    }
    let mut s = spec();
    s.command.clear();
    assert_eq!(
        plan(&s, "r", None),
        Err(SandboxError::BadSpec("empty command"))
    );
}

#[test]
fn extract_report_takes_the_last_complete_block() {
    let out = format!("{REPORT_BEGIN}\nold\n{REPORT_END}\n{REPORT_BEGIN}\nnew\n{REPORT_END}\n");
    assert_eq!(extract_report(&out), Some("new"));
    assert_eq!(extract_report(&format!("{REPORT_BEGIN}\ncut off")), None);
    assert_eq!(extract_report("no markers"), None);
}

#[test]
fn libtest_report_table() {
    let report = [
        r#"{ "type": "suite", "event": "started", "test_count": 4 }"#.to_string(),
        r#"{ "type": "test", "event": "started", "name": "a" }"#.to_string(),
        libtest("a", "ok"),
        libtest("b", "failed"),
        libtest("c", "ignored"),
        libtest("d", "timeout"),
        "not json".into(),
        r#"{ "type": "suite", "event": "failed", "passed": 1 }"#.into(),
    ]
    .join("\n");
    let got = parse_report(ReportFormat::LibtestJson, &report).expect("parses");
    let want: BTreeMap<String, TestOutcome> = [
        ("a", TestOutcome::Passed),
        ("b", TestOutcome::Failed),
        ("c", TestOutcome::Skipped),
        ("d", TestOutcome::Failed),
    ]
    .iter()
    .map(|(k, v)| ((*k).to_string(), *v))
    .collect();
    assert_eq!(got, want);
}

#[test]
fn junit_report_table() {
    let xml = r#"<?xml version="1.0"?>
<testsuites><testsuite name="pytest" tests="5">
  <testcase classname="tests.test_x" name="test_ok" time="0.1"/>
  <testcase classname="tests.test_x" name="test_fail"><failure message="x">boom</failure></testcase>
  <testcase classname="tests.test_x" name="test_err"><error message="y"/></testcase>
  <testcase classname='tests.test_x' name='test_skip'><skipped/></testcase>
  <testcase name="test_&lt;lt&gt;[a&amp;b]"></testcase>
</testsuite></testsuites>"#;
    let got = parse_report(ReportFormat::Junit, xml).expect("parses");
    let want: BTreeMap<String, TestOutcome> = [
        ("tests.test_x::test_ok", TestOutcome::Passed),
        ("tests.test_x::test_fail", TestOutcome::Failed),
        ("tests.test_x::test_err", TestOutcome::Error),
        ("tests.test_x::test_skip", TestOutcome::Skipped),
        ("test_<lt>[a&b]", TestOutcome::Passed),
    ]
    .iter()
    .map(|(k, v)| ((*k).to_string(), *v))
    .collect();
    assert_eq!(got, want);
    assert!(parse_report(ReportFormat::Junit, "<testsuite><testcase name=\"x\">").is_err());
    assert!(parse_report(
        ReportFormat::Junit,
        "<testsuite><testcase classname=\"c\"/>"
    )
    .is_err());
    assert_eq!(
        parse_report(ReportFormat::Junit, "<testsuite/>").map(|m| m.len()),
        Ok(0)
    );
}

#[cfg(unix)]
mod engine {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// A fake engine CLI: logs every call, and behaves per the `mode` file beside it.
    const FAKE: &str = r#"#!/bin/sh
d=$(dirname "$0")
printf '%s\n' "$*" >> "$d/calls"
case "$1" in
  image) [ "$3" = --format ] && cat "$d/digests"; exit 0 ;;
  pull) exit 0 ;;
  run)
    case "$(cat "$d/mode")" in
      hang) echo $$ > "$d/pid"; exec sleep 30 ;;
      report) printf 'noise\n%s\n%s\n%s\n' "$BEGIN" '{"type":"test","event":"failed","name":"t1"}' "$END"; exit 101 ;;
      silent) exit 0 ;;
    esac ;;
  kill) kill "$(cat "$d/pid")"; exit 0 ;;
  inspect) cat "$d/state"; exit 0 ;;
  rm) exit 0 ;;
esac
exit 3
"#;

    fn fake(mode: &str, digest: &str, state: &str) -> (tempfile::TempDir, Cli) {
        let dir = tempfile::tempdir().expect("tempdir");
        let script = FAKE
            .replace("$BEGIN", REPORT_BEGIN)
            .replace("$END", REPORT_END);
        let bin = dir.path().join("engine");
        std::fs::write(&bin, script).expect("write");
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        std::fs::write(dir.path().join("mode"), mode).expect("write");
        std::fs::write(
            dir.path().join("digests"),
            format!(r#"["ghcr.io/paiml/eval-rust@{digest}"]"#),
        )
        .expect("write");
        std::fs::write(dir.path().join("state"), state).expect("write");
        (dir, Cli { bin })
    }

    fn calls(dir: &tempfile::TempDir) -> Vec<String> {
        std::fs::read_to_string(dir.path().join("calls"))
            .expect("calls")
            .lines()
            .map(String::from)
            .collect()
    }

    /// SBX-003 through the executor: the deadline fires, the container is killed by name
    /// and removed, and the run is labelled a timeout.
    #[test]
    fn falsify_ace_sbx_003_executor_kills_at_deadline() {
        let (dir, mut cli) = fake("hang", DIGEST, "false 143");
        let mut s = spec();
        s.limits.wall = Duration::from_millis(300);
        let start = Instant::now();
        let r = run(&mut cli, &s, "run-hang", None).expect("runs");
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "{:?}",
            start.elapsed()
        );
        assert_eq!(r.killed_by, Some(KilledBy::Timeout));
        let calls = calls(&dir);
        assert!(calls.contains(&"kill run-hang".to_string()), "{calls:?}");
        assert_eq!(
            calls.last().map(String::as_str),
            Some("rm --force run-hang")
        );
    }

    /// SBX-007/008 through the executor: the report on stdout becomes outcomes, a run
    /// without one is an error, and every run is removed afterwards.
    #[test]
    fn falsify_ace_sbx_007_executor_report_path() {
        let (dir, mut cli) = fake("report", DIGEST, "false 101");
        let r = run(&mut cli, &spec(), "run-ok", None).expect("runs");
        assert_eq!(
            (r.status.clone(), r.outcomes.get("t1")),
            (SandboxStatus::Ok, Some(&TestOutcome::Failed))
        );
        assert_eq!(
            calls(&dir).last().map(String::as_str),
            Some("rm --force run-ok")
        );

        let (dir, mut cli) = fake("silent", DIGEST, "false 0");
        let r = run(&mut cli, &spec(), "run-silent", None).expect("runs");
        assert!(matches!(r.status, SandboxStatus::Error(_)) && r.tests_executed() == 0);
        assert_eq!(
            calls(&dir).last().map(String::as_str),
            Some("rm --force run-silent")
        );

        let (_dir, mut cli) = fake("silent", DIGEST, "true 137");
        let r = run(&mut cli, &spec(), "run-oom", None).expect("runs");
        assert_eq!(r.killed_by, Some(KilledBy::Oom));
    }

    /// SBX-002 through the executor: a digest mismatch refuses before any `run`.
    #[test]
    fn falsify_ace_sbx_002_executor_checks_digest_first() {
        let other = format!("sha256:{}", "e".repeat(64));
        let (dir, mut cli) = fake("report", &other, "false 0");
        assert_eq!(
            run(&mut cli, &spec(), "run-x", None),
            Err(SandboxError::DigestMismatch(DIGEST.into()))
        );
        assert!(!calls(&dir).iter().any(|c| c.starts_with("run ")));
    }
}
