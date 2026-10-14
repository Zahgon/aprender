//! Hermetic sandbox runner (contracts/apr-code-eval-sandbox-v1.yaml, ACE-SBX-001..008).
//!
//! Three layers, so each equation is checked where it is decided:
//! - [`plan`] turns a [`RunSpec`] into the exact container-engine argv. Network, image pin,
//!   limits, mounts and environment are all fixed here, and a spec that would break one is
//!   refused before anything starts.
//! - [`finalize`] turns what the engine saw (exit, kill, OOM, the runner's structured report)
//!   into a [`SandboxReport`]. Test counts come only from the report, never the exit code.
//! - [`Engine`] runs one plan with a wall-clock deadline. [`Cli`] drives `docker`/`podman`.
//!
//! The report travels on the container's stdout between two marker lines, because `/work`
//! is a tmpfs that does not outlive the container.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::verdict::{KilledBy, SandboxReport, SandboxStatus, TestOutcome};

/// Marker line before the runner report on stdout.
pub const REPORT_BEGIN: &str = "=====APR-CODE-EVAL-REPORT-BEGIN=====";
/// Marker line after the runner report on stdout.
pub const REPORT_END: &str = "=====APR-CODE-EVAL-REPORT-END=====";
/// Cap on captured stdout, so a candidate cannot exhaust host memory through the pipe.
pub const STDOUT_CAP: u64 = 64 << 20;

/// Refusals before a run starts, and engine failures.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SandboxError {
    /// ACE-SBX-002: the image is not `name@sha256:<64 hex>`.
    #[error("image {0:?} is not pinned by a sha256 digest")]
    Unpinned(String),
    /// ACE-SBX-002: the engine's copy of the image does not carry the declared digest.
    #[error("pulled image does not carry the declared digest {0}")]
    DigestMismatch(String),
    /// ACE-SBX-003: a limit is zero, i.e. not a usable finite bound.
    #[error("limit {0} must be finite and non-zero")]
    BadLimit(&'static str),
    /// ACE-SBX-004: the task dir is not a safe read-only mount source.
    #[error("task dir {0:?} cannot be mounted: {1}")]
    BadTaskDir(String, &'static str),
    /// The command, report path or run name is unusable.
    #[error("bad run spec: {0}")]
    BadSpec(&'static str),
    /// The engine could not be driven at all.
    #[error("engine: {0}")]
    Engine(String),
}

/// ACE-SBX-002: an image reference pinned by content digest. Deserializing goes through
/// [`ImageRef::parse`], so a loaded manifest cannot carry an unpinned image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ImageRef(String);

impl TryFrom<String> for ImageRef {
    type Error = SandboxError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::parse(&s)
    }
}

impl From<ImageRef> for String {
    fn from(i: ImageRef) -> Self {
        i.0
    }
}

impl ImageRef {
    /// Accept only `name@sha256:<64 lowercase hex>`.
    pub fn parse(s: &str) -> Result<Self, SandboxError> {
        let unpinned = || SandboxError::Unpinned(s.to_string());
        let (name, digest) = s.split_once("@sha256:").ok_or_else(unpinned)?;
        let hex_ok = digest.len() == 64
            && digest
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if name.is_empty() || name.contains('@') || !hex_ok {
            return Err(unpinned());
        }
        Ok(Self(s.to_string()))
    }

    /// The full reference.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The `sha256:<hex>` part.
    #[must_use]
    pub fn digest(&self) -> &str {
        self.0.split_once('@').map_or("", |(_, d)| d)
    }
}

/// ACE-SBX-002: the engine's `RepoDigests` for the image must include the declared one.
pub fn verify_pulled(image: &ImageRef, repo_digests: &[String]) -> Result<(), SandboxError> {
    if repo_digests
        .iter()
        .any(|d| d.ends_with(&format!("@{}", image.digest())))
    {
        Ok(())
    } else {
        Err(SandboxError::DigestMismatch(image.digest().to_string()))
    }
}

/// ACE-SBX-003: every resource bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limits {
    /// Wall-clock limit.
    pub wall: Duration,
    /// CPU limit in thousandths of a CPU.
    pub millicpus: u32,
    /// Memory limit, swap included.
    pub mem_bytes: u64,
    /// Process/thread limit.
    pub pids: u32,
    /// Size of the `/work` tmpfs, the only writable path.
    pub disk_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            wall: Duration::from_secs(900),
            millicpus: 2000,
            mem_bytes: 4 << 30,
            pids: 512,
            disk_bytes: 2 << 30,
        }
    }
}

impl Limits {
    fn check(&self) -> Result<(), SandboxError> {
        let zero = [
            ("wall", self.wall.is_zero()),
            ("millicpus", self.millicpus == 0),
            ("mem_bytes", self.mem_bytes == 0),
            ("pids", self.pids == 0),
            ("disk_bytes", self.disk_bytes == 0),
        ];
        zero.into_iter()
            .find(|&(_, z)| z)
            .map_or(Ok(()), |(name, _)| Err(SandboxError::BadLimit(name)))
    }
}

/// Which structured report the task's test command writes (ACE-SBX-007).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportFormat {
    /// libtest JSON lines (`cargo test -- --format json`, nextest `libtest-json`).
    LibtestJson,
    /// JUnit XML (pytest `--junitxml`, most other runners).
    Junit,
}

/// One run: (task dir, command, image, seed) with its limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunSpec {
    /// The image, pinned.
    pub image: ImageRef,
    /// Host directory holding snapshot ⊕ edit and the hidden tests; mounted read-only at `/task`.
    pub task_dir: PathBuf,
    /// The test command, argv form, run with `/work` as cwd.
    pub command: Vec<String>,
    /// Where the command writes its report, relative to `/work`.
    pub report_path: String,
    /// The report's format.
    pub report_format: ReportFormat,
    /// The run seed (ACE-SBX-005).
    pub seed: u64,
    /// `SOURCE_DATE_EPOCH`: the task's release date, in seconds.
    pub source_date_epoch: i64,
    /// Resource bounds.
    pub limits: Limits,
}

/// Host paths a task dir may never be, contain, or sit inside (ACE-SBX-004).
const FORBIDDEN: [&str; 6] = ["/var/run", "/run", "/proc", "/sys", "/dev", "/etc"];

fn check_task_dir(dir: &Path, home: Option<&Path>) -> Result<String, SandboxError> {
    let s = dir
        .to_str()
        .ok_or(SandboxError::BadSpec("task dir is not UTF-8"))?;
    let bad = |why| Err(SandboxError::BadTaskDir(s.to_string(), why));
    if !dir.is_absolute() || s.contains("..") {
        return bad("not an absolute normalized path");
    }
    if s.contains(',') || s.contains('=') {
        return bad("',' or '=' would inject mount options");
    }
    if dir == Path::new("/")
        || FORBIDDEN.iter().any(|f| dir.starts_with(f))
        || s.contains("docker.sock")
    {
        return bad("system path or container socket");
    }
    if home.is_some_and(|h| h.starts_with(dir) || dir == h) {
        return bad("the host home or one of its parents");
    }
    Ok(s.to_string())
}

/// The fixed environment (ACE-SBX-005). Nothing from the host is passed through.
#[must_use]
pub fn environment(spec: &RunSpec) -> BTreeMap<&'static str, String> {
    BTreeMap::from([
        ("SEED", spec.seed.to_string()),
        ("PYTHONHASHSEED", spec.seed.to_string()),
        ("TZ", "UTC".to_string()),
        ("LANG", "C.UTF-8".to_string()),
        ("SOURCE_DATE_EPOCH", spec.source_date_epoch.to_string()),
        ("HOME", "/work".to_string()),
        ("TMPDIR", "/work".to_string()),
        ("APR_EVAL_REPORT", format!("/work/{}", spec.report_path)),
        ("APR_EVAL_BEGIN", REPORT_BEGIN.to_string()),
        ("APR_EVAL_END", REPORT_END.to_string()),
    ])
}

/// Runs the task command, then prints the report between the markers and keeps its exit
/// code. The command arrives as `"$@"`, so nothing is interpolated into the script.
const WRAPPER: &str = r#"cp -a /task/. /work/ && cd /work || exit 125
"$@"; rc=$?
printf '%s\n' "$APR_EVAL_BEGIN"; cat "$APR_EVAL_REPORT" 2>/dev/null; printf '\n%s\n' "$APR_EVAL_END"
exit $rc"#;

/// The engine argv for one run (ACE-SBX-001..005, 008). `name` must be fresh per run.
pub fn plan(spec: &RunSpec, name: &str, home: Option<&Path>) -> Result<Vec<String>, SandboxError> {
    spec.limits.check()?;
    let src = check_task_dir(&spec.task_dir, home)?;
    if spec.command.is_empty() {
        return Err(SandboxError::BadSpec("empty command"));
    }
    let rp = &spec.report_path;
    if rp.is_empty() || rp.starts_with('/') || rp.contains("..") {
        return Err(SandboxError::BadSpec(
            "report path must be relative to /work",
        ));
    }
    if name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(SandboxError::BadSpec("run name"));
    }
    let l = &spec.limits;
    let mut argv: Vec<String> = [
        "run",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--workdir",
        "/work",
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    argv.extend([
        "--tmpfs".into(),
        format!("/work:rw,exec,size={},mode=1777", l.disk_bytes),
        "--mount".into(),
        format!("type=bind,source={src},target=/task,readonly"),
        "--memory".into(),
        l.mem_bytes.to_string(),
        "--memory-swap".into(),
        l.mem_bytes.to_string(),
        "--cpus".into(),
        format!("{}.{:03}", l.millicpus / 1000, l.millicpus % 1000),
        "--pids-limit".into(),
        l.pids.to_string(),
    ]);
    for (k, v) in environment(spec) {
        argv.extend(["--env".into(), format!("{k}={v}")]);
    }
    argv.extend([
        spec.image.as_str().to_string(),
        "sh".into(),
        "-c".into(),
        WRAPPER.into(),
        "sh".into(),
    ]);
    argv.extend(spec.command.iter().cloned());
    Ok(argv)
}

/// The last marked report block on stdout, if any.
#[must_use]
pub fn extract_report(stdout: &str) -> Option<&str> {
    let begin = stdout.rfind(REPORT_BEGIN)? + REPORT_BEGIN.len();
    let end = begin + stdout[begin..].find(REPORT_END)?;
    let body = stdout[begin..end].trim();
    (!body.is_empty()).then_some(body)
}

#[derive(Deserialize)]
struct LibtestEvent {
    #[serde(rename = "type")]
    kind: String,
    event: String,
    #[serde(default)]
    name: String,
}

fn parse_libtest(report: &str) -> BTreeMap<String, TestOutcome> {
    let mut out = BTreeMap::new();
    for line in report.lines().filter(|l| l.trim_start().starts_with('{')) {
        let Ok(ev) = serde_json::from_str::<LibtestEvent>(line) else {
            continue;
        };
        if ev.kind != "test" {
            continue;
        }
        let outcome = match ev.event.as_str() {
            "ok" => TestOutcome::Passed,
            "failed" | "timeout" => TestOutcome::Failed,
            "ignored" => TestOutcome::Skipped,
            _ => continue,
        };
        out.insert(ev.name, outcome);
    }
    out
}

fn unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn attr(tag: &str, key: &str) -> Option<String> {
    for quote in ['"', '\''] {
        let pat = format!(" {key}={quote}");
        if let Some(i) = tag.find(&pat) {
            let rest = &tag[i + pat.len()..];
            return rest.find(quote).map(|j| unescape(&rest[..j]));
        }
    }
    None
}

/// One `<testcase …>` element: its outcome from its first child element, if any.
fn junit_case(open_tag: &str, body: &str) -> Option<(String, TestOutcome)> {
    let name = attr(open_tag, "name")?;
    let id = match attr(open_tag, "classname") {
        Some(c) if !c.is_empty() => format!("{c}::{name}"),
        _ => name,
    };
    let outcome = if body.contains("<failure") {
        TestOutcome::Failed
    } else if body.contains("<error") {
        TestOutcome::Error
    } else if body.contains("<skipped") {
        TestOutcome::Skipped
    } else {
        TestOutcome::Passed
    };
    Some((id, outcome))
}

fn parse_junit(report: &str) -> Result<BTreeMap<String, TestOutcome>, String> {
    if !report.contains("<testsuite") {
        return Err("no <testsuite> element".into());
    }
    let mut out = BTreeMap::new();
    let mut rest = report;
    while let Some(i) = rest.find("<testcase") {
        rest = &rest[i..];
        let close = rest.find('>').ok_or("unterminated <testcase")?;
        let open_tag = &rest[..=close];
        let (body, next) = if open_tag.ends_with("/>") {
            ("", close + 1)
        } else {
            let end = rest.find("</testcase>").ok_or("missing </testcase>")?;
            (&rest[close + 1..end], end + "</testcase>".len())
        };
        let (id, outcome) = junit_case(open_tag, body).ok_or("<testcase> without name")?;
        out.insert(id, outcome);
        rest = &rest[next..];
    }
    Ok(out)
}

/// Parse a runner report into per-test outcomes.
pub fn parse_report(
    format: ReportFormat,
    report: &str,
) -> Result<BTreeMap<String, TestOutcome>, String> {
    match format {
        ReportFormat::LibtestJson => Ok(parse_libtest(report)),
        ReportFormat::Junit => parse_junit(report),
    }
}

/// What the engine observed about one finished run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Observed {
    /// The container's exit code; `None` when it died by signal.
    pub exit_code: Option<i32>,
    /// The runner hit the wall-clock deadline and killed the container.
    pub timed_out: bool,
    /// The engine reports the kernel OOM-killed the container.
    pub oom_killed: bool,
    /// Captured stdout (capped at [`STDOUT_CAP`]).
    pub stdout: String,
}

fn error_report(status: &str, killed_by: Option<KilledBy>) -> SandboxReport {
    SandboxReport {
        status: SandboxStatus::Error(status.to_string()),
        killed_by,
        outcomes: BTreeMap::new(),
    }
}

/// ACE-SBX-003/007: a kill is a status, a missing or unreadable report is a sandbox error,
/// and only a parsed report yields test outcomes.
#[must_use]
pub fn finalize(format: ReportFormat, obs: &Observed) -> SandboxReport {
    if obs.timed_out {
        return error_report("timeout", Some(KilledBy::Timeout));
    }
    if obs.oom_killed {
        return error_report("oom", Some(KilledBy::Oom));
    }
    match obs.exit_code {
        None => return error_report("killed by signal", Some(KilledBy::Other)),
        Some(125) => return error_report("engine or setup failure (exit 125)", None),
        Some(_) => {}
    }
    let Some(report) = extract_report(&obs.stdout) else {
        return error_report("no test report", None);
    };
    match parse_report(format, report) {
        Ok(outcomes) => SandboxReport {
            status: SandboxStatus::Ok,
            killed_by: None,
            outcomes,
        },
        Err(e) => error_report(&format!("unreadable test report: {e}"), None),
    }
}

/// ACE-SBX-006: the projection two runs must agree on.
fn projection(
    r: &SandboxReport,
) -> (
    &SandboxStatus,
    Option<KilledBy>,
    &BTreeMap<String, TestOutcome>,
) {
    (&r.status, r.killed_by, &r.outcomes)
}

/// A difference between repeat runs: the task is a flake and is quarantined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Flake {
    /// Index of the first run that disagrees with run 0.
    pub run: usize,
    /// Test ids whose outcome differs, or `["<status>"]` when the status does.
    pub differing: Vec<String>,
}

/// ACE-SBX-006: every report equals the first, modulo wall time (which a report omits).
pub fn agree(reports: &[SandboxReport]) -> Result<(), Flake> {
    let Some(first) = reports.first() else {
        return Ok(());
    };
    for (i, r) in reports.iter().enumerate().skip(1) {
        if projection(r) == projection(first) {
            continue;
        }
        let ids: BTreeSet<&String> = first.outcomes.keys().chain(r.outcomes.keys()).collect();
        let mut differing: Vec<String> = ids
            .into_iter()
            .filter(|id| first.outcomes.get(*id) != r.outcomes.get(*id))
            .cloned()
            .collect();
        if differing.is_empty() {
            differing.push("<status>".into());
        }
        return Err(Flake { run: i, differing });
    }
    Ok(())
}

/// Runs one planned container to completion or deadline.
pub trait Engine {
    /// Make sure the image is present and carries its digest (ACE-SBX-002).
    fn ensure_image(&mut self, image: &ImageRef) -> Result<(), SandboxError>;
    /// Run `argv` (from [`plan`]) as container `name`, killing it at `wall`.
    fn run(
        &mut self,
        argv: &[String],
        name: &str,
        wall: Duration,
    ) -> Result<Observed, SandboxError>;
}

/// A container engine driven through its CLI (`docker` or `podman`).
#[derive(Debug, Clone)]
pub struct Cli {
    /// The engine binary.
    pub bin: PathBuf,
}

fn read_capped(mut r: impl Read + Send + 'static) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = r.by_ref().take(STDOUT_CAP).read_to_end(&mut buf);
        let _ = std::io::copy(&mut r, &mut std::io::sink());
        String::from_utf8_lossy(&buf).into_owned()
    })
}

/// Wait for `child` until `wall` elapses. `Ok(None)` means the deadline passed.
pub fn wait_deadline(
    child: &mut Child,
    wall: Duration,
) -> std::io::Result<Option<std::process::ExitStatus>> {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if start.elapsed() >= wall {
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

impl Cli {
    /// `<engine> --version`, trimmed: the receipt's `sandbox.engine_version`.
    pub fn version(&self) -> Result<String, SandboxError> {
        Ok(self.output(&["--version"])?.trim().to_string())
    }

    fn output(&self, args: &[&str]) -> Result<String, SandboxError> {
        let out = Command::new(&self.bin)
            .args(args)
            .stdin(Stdio::null())
            .output();
        let out = out.map_err(|e| SandboxError::Engine(e.to_string()))?;
        if !out.status.success() {
            return Err(SandboxError::Engine(
                String::from_utf8_lossy(&out.stderr).trim().to_string(),
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }
}

impl Engine for Cli {
    fn ensure_image(&mut self, image: &ImageRef) -> Result<(), SandboxError> {
        if self.output(&["image", "inspect", image.as_str()]).is_err() {
            self.output(&["pull", "--quiet", image.as_str()])?;
        }
        let json = self.output(&[
            "image",
            "inspect",
            "--format",
            "{{json .RepoDigests}}",
            image.as_str(),
        ])?;
        let digests: Vec<String> =
            serde_json::from_str(json.trim()).map_err(|e| SandboxError::Engine(e.to_string()))?;
        verify_pulled(image, &digests)
    }

    fn run(
        &mut self,
        argv: &[String],
        name: &str,
        wall: Duration,
    ) -> Result<Observed, SandboxError> {
        let engine = |e: std::io::Error| SandboxError::Engine(e.to_string());
        let mut child = Command::new(&self.bin)
            .args(argv)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(engine)?;
        let reader = read_capped(
            child
                .stdout
                .take()
                .ok_or(SandboxError::Engine("no stdout pipe".into()))?,
        );
        let status = wait_deadline(&mut child, wall).map_err(engine)?;
        let timed_out = status.is_none();
        if timed_out {
            let _ = self.output(&["kill", name]);
            child.wait().map_err(engine)?;
        }
        let stdout = reader.join().unwrap_or_default();
        let state = self.output(&[
            "inspect",
            "--format",
            "{{.State.OOMKilled}} {{.State.ExitCode}}",
            name,
        ]);
        let _ = self.output(&["rm", "--force", name]);
        let state = state?;
        let mut it = state.split_whitespace();
        let oom_killed = it.next() == Some("true");
        let exit_code = status
            .and_then(|s| s.code())
            .or_else(|| it.next().and_then(|c| c.parse().ok()));
        Ok(Observed {
            exit_code: if timed_out { None } else { exit_code },
            timed_out,
            oom_killed,
            stdout,
        })
    }
}

/// One complete run: check the image, plan, execute, finalize (ACE-SBX-001..008).
pub fn run(
    engine: &mut impl Engine,
    spec: &RunSpec,
    name: &str,
    home: Option<&Path>,
) -> Result<SandboxReport, SandboxError> {
    let argv = plan(spec, name, home)?;
    engine.ensure_image(&spec.image)?;
    let obs = engine.run(&argv, name, spec.limits.wall)?;
    Ok(finalize(spec.report_format, &obs))
}

#[cfg(test)]
#[path = "sandbox_tests.rs"]
mod tests;
