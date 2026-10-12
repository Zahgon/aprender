//! The process and HTTP plumbing `ladder-cells-produce` needs, with no new dependency:
//! `subprocess.run(..., capture_output=True, text=True, timeout=T)`, one `apr serve run`
//! child in its own process group, and a minimal blocking HTTP/1.1 client over `TcpStream`
//! for `urllib.request.urlopen` against that child on 127.0.0.1.
//!
//! Known gaps against the Python original: output is decoded lossily where Python decodes
//! strictly (and would crash on invalid UTF-8); the serve child gets its own process GROUP
//! (`process_group(0)`) where Python starts a new SESSION (`setsid`), and is signalled with
//! the `kill` binary, as this crate forbids the `unsafe` a direct `killpg` needs; the HTTP
//! client follows no redirect and reads no proxy setting; and an error's message text is
//! close to, not identical with, `str()` of urllib's exception.

use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// `subprocess.run`'s result: returncode (`-signal` for a killed child), stdout, stderr.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ran {
    pub rc: i64,
    pub stdout: String,
    pub stderr: String,
}

/// Seconds as a `Duration`; a negative count is zero.
pub fn secs(s: i64) -> Duration {
    Duration::from_secs(u64::try_from(s).unwrap_or(0))
}

/// `now + d`, saturating at a century instead of overflowing `Instant`.
fn deadline(d: Duration) -> Instant {
    let now = Instant::now();
    now.checked_add(d)
        .or_else(|| now.checked_add(secs(100 * 365 * 86_400)))
        .unwrap_or(now)
}

/// Poll `child` until it exits or `until` passes.
fn wait_until(child: &mut Child, until: Instant) -> Option<ExitStatus> {
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) if Instant::now() < until => std::thread::sleep(Duration::from_millis(20)),
            Ok(None) | Err(_) => return None,
        }
    }
}

fn spawn_reader<R: Read + Send + 'static>(
    mut pipe: R,
    is_err: bool,
    tx: mpsc::Sender<(bool, Vec<u8>)>,
) {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = pipe.read_to_end(&mut buf);
        let _ = tx.send((is_err, buf));
    });
}

/// `subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout_s)`.
/// `stdin` `None` inherits ours, as Python does. On the timeout the child is killed and the
/// result is `(124, "", "timed out after T s")`, as the original's `Runner.call` returns.
/// `Err` when the program cannot be started (Python raises `OSError`, uncaught there).
pub fn run_with_timeout(
    argv: &[String],
    stdin: Option<&str>,
    timeout_s: i64,
) -> Result<Ran, String> {
    let (prog, rest) = argv.split_first().ok_or("an empty argv")?;
    let mut cmd = Command::new(prog);
    cmd.args(rest)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::inherit()
        });
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("FileNotFoundError: {prog}: {e}"))?;
    let until = deadline(secs(timeout_s));
    if let (Some(text), Some(mut pipe)) = (stdin, child.stdin.take()) {
        let bytes = text.as_bytes().to_vec();
        // Written from a thread so a child that never reads cannot block us past the timeout.
        std::thread::spawn(move || {
            let _ = pipe.write_all(&bytes);
        });
    }
    let (tx, rx) = mpsc::channel();
    let mut pending = 0;
    if let Some(p) = child.stdout.take() {
        spawn_reader(p, false, tx.clone());
        pending += 1;
    }
    if let Some(p) = child.stderr.take() {
        spawn_reader(p, true, tx.clone());
        pending += 1;
    }
    drop(tx);
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let mut timed_out = false;
    while pending > 0 {
        let left = until.saturating_duration_since(Instant::now());
        match rx.recv_timeout(left) {
            Ok((true, buf)) => err = buf,
            Ok((false, buf)) => out = buf,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                timed_out = true;
                break;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        pending -= 1;
    }
    let status = if timed_out {
        None
    } else {
        wait_until(&mut child, until)
    };
    let Some(status) = status else {
        let _ = child.kill();
        let _ = child.wait();
        return Ok(Ran {
            rc: 124,
            stdout: String::new(),
            stderr: format!("timed out after {timeout_s} s"),
        });
    };
    Ok(Ran {
        rc: exit_code(status),
        stdout: crate::pystr::py_read_text(&out),
        stderr: crate::pystr::py_read_text(&err),
    })
}

/// `Popen.returncode`: the exit code, or minus the signal that killed the child.
fn exit_code(status: ExitStatus) -> i64 {
    status
        .code()
        .map_or_else(|| -i64::from(status.signal().unwrap_or(0)), i64::from)
}

/// A free TCP port on 127.0.0.1, as the original's bind-to-0-and-close finds one.
pub fn free_port() -> Result<u16, String> {
    let l = TcpListener::bind(("127.0.0.1", 0)).map_err(|e| format!("OSError: bind: {e}"))?;
    l.local_addr()
        .map(|a| a.port())
        .map_err(|e| format!("OSError: getsockname: {e}"))
}

/// One HTTP response: status and the whole body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resp {
    pub status: u16,
    pub body: Vec<u8>,
}

fn invalid(msg: &str) -> std::io::Error {
    std::io::Error::new(ErrorKind::InvalidData, msg.to_owned())
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Read once more into `buf`; `false` at end of stream.
fn read_more(s: &mut TcpStream, buf: &mut Vec<u8>) -> std::io::Result<bool> {
    let mut chunk = [0u8; 16 * 1024];
    loop {
        match s.read(&mut chunk) {
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                return Ok(n > 0);
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
}

/// A chunked body, if `raw` has all of it.
fn dechunk(raw: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut at = 0;
    loop {
        let line_end = at + find(&raw[at..], b"\r\n")?;
        let line = std::str::from_utf8(&raw[at..line_end]).ok()?;
        let size = usize::from_str_radix(line.split(';').next()?.trim(), 16).ok()?;
        at = line_end + 2;
        if size == 0 {
            return Some(out);
        }
        out.extend_from_slice(raw.get(at..at + size)?);
        at += size + 2;
    }
}

/// The status and headers of a response head (`HTTP/1.x NNN reason` + `Name: value` lines).
fn parse_head(head: &str) -> std::io::Result<(u16, Vec<(String, String)>)> {
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let mut parts = status_line.splitn(3, ' ');
    let proto = parts.next().unwrap_or("");
    let code = parts.next().unwrap_or("");
    if !proto.starts_with("HTTP/") || code.len() != 3 {
        return Err(invalid("BadStatusLine"));
    }
    let status = code.parse::<u16>().map_err(|_| invalid("BadStatusLine"))?;
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_owned()))
        .collect();
    Ok((status, headers))
}

/// One request to 127.0.0.1:`port` with `Connection: close`. `timeout` bounds the connect
/// and every read and write, as a socket timeout does in urllib. The error's flag is `true`
/// when it happened while connecting or sending (where urllib wraps it in a `URLError`).
pub fn http(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    timeout: Duration,
) -> Result<Resp, (std::io::Error, bool)> {
    let t = timeout.max(Duration::from_millis(1));
    let s = send(port, method, path, body, t).map_err(|e| (e, true))?;
    receive(s).map_err(|e| (e, false))
}

fn send(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    t: Duration,
) -> std::io::Result<TcpStream> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut s = TcpStream::connect_timeout(&addr, t)?;
    s.set_read_timeout(Some(t))?;
    s.set_write_timeout(Some(t))?;
    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAccept-Encoding: identity\r\n\
         User-Agent: aprender-ci-tools\r\nConnection: close\r\n"
    );
    if let Some(b) = body {
        req.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            b.len()
        ));
    }
    req.push_str("\r\n");
    s.write_all(req.as_bytes())?;
    if let Some(b) = body {
        s.write_all(b)?;
    }
    s.flush()?;
    Ok(s)
}

fn receive(mut s: TcpStream) -> std::io::Result<Resp> {
    let mut buf = Vec::new();
    let head_end = read_head(&mut s, &mut buf)?;
    let (status, headers) = parse_head(&String::from_utf8_lossy(&buf[..head_end]))?;
    let raw = buf.split_off(head_end + 4);
    let header = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
    };
    let chunked = header("transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked"));
    let length = header("content-length").and_then(|v| v.parse::<usize>().ok());
    let body = if chunked {
        read_chunked(&mut s, raw)?
    } else if let Some(n) = length {
        read_sized(&mut s, raw, n)?
    } else {
        read_to_close(&mut s, raw)?
    };
    Ok(Resp { status, body })
}

/// Reads until the blank line that ends the response head; returns its offset in `buf`.
fn read_head(s: &mut TcpStream, buf: &mut Vec<u8>) -> std::io::Result<usize> {
    loop {
        if let Some(i) = find(buf, b"\r\n\r\n") {
            return Ok(i);
        }
        if !read_more(s, buf)? {
            return Err(invalid("Remote end closed connection without response"));
        }
    }
}

fn read_chunked(s: &mut TcpStream, mut raw: Vec<u8>) -> std::io::Result<Vec<u8>> {
    loop {
        if let Some(b) = dechunk(&raw) {
            return Ok(b);
        }
        if !read_more(s, &mut raw)? {
            return Err(invalid("IncompleteRead: chunked body cut short"));
        }
    }
}

fn read_sized(s: &mut TcpStream, mut raw: Vec<u8>, n: usize) -> std::io::Result<Vec<u8>> {
    while raw.len() < n {
        if !read_more(s, &mut raw)? {
            return Err(invalid("IncompleteRead: body shorter than Content-Length"));
        }
    }
    raw.truncate(n);
    Ok(raw)
}

fn read_to_close(s: &mut TcpStream, mut raw: Vec<u8>) -> std::io::Result<Vec<u8>> {
    while read_more(s, &mut raw)? {}
    Ok(raw)
}

/// `str()` of the exception urllib raises for `e`: `timed out` for a timeout, else the OS
/// error in Python's `[Errno N] message` form, wrapped as a `URLError` when `connect` is set.
pub fn urllib_error(e: &std::io::Error, connect: bool) -> String {
    let inner = if matches!(e.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) {
        "timed out".to_owned()
    } else if let Some(code) = e.raw_os_error() {
        let text = e.to_string();
        let msg = text
            .strip_suffix(&format!(" (os error {code})"))
            .unwrap_or(&text)
            .to_owned();
        format!("[Errno {code}] {msg}")
    } else {
        e.to_string()
    };
    if connect {
        format!("<urlopen error {inner}>")
    } else {
        inner
    }
}

/// One `apr serve run` child, started in its own process group and killed by its RECORDED
/// pid's group on drop (the original's `finally: serve.close()`), never by pattern.
pub struct Serve {
    child: Child,
    pub port: u16,
    pub up: bool,
}

impl Serve {
    /// Start `argv` (stdout and stderr to /dev/null) and wait up to `ceiling_s` for
    /// `GET /health` to answer 2xx, retrying each second while the child lives.
    pub fn start(argv: &[String], port: u16, ceiling_s: i64) -> Result<Self, String> {
        let (prog, rest) = argv.split_first().ok_or("an empty argv")?;
        let child = Command::new(prog)
            .args(rest)
            .stdin(Stdio::inherit())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .map_err(|e| format!("FileNotFoundError: {prog}: {e}"))?;
        let mut serve = Self {
            child,
            port,
            up: false,
        };
        serve.up = serve.wait_up(ceiling_s);
        Ok(serve)
    }

    fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn wait_up(&mut self, ceiling_s: i64) -> bool {
        let end = deadline(secs(ceiling_s));
        while Instant::now() < end && self.alive() {
            let ok = http(self.port, "GET", "/health", None, secs(2))
                .is_ok_and(|r| (200..300).contains(&r.status));
            if ok {
                return true;
            }
            std::thread::sleep(secs(1));
        }
        false
    }

    /// POST `body` to `path`: `Ok` with the response, or the urllib-style error text.
    pub fn post(&self, path: &str, body: &[u8], timeout_s: i64) -> Result<Resp, String> {
        http(self.port, "POST", path, Some(body), secs(timeout_s))
            .map_err(|(e, sending)| urllib_error(&e, sending))
    }

    fn signal_group(&mut self, sig: &str) {
        let group = format!("-{}", self.child.id());
        let sent = Command::new("kill")
            .args(["-s", sig, "--", &group])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if !sent {
            let _ = self.child.kill();
        }
    }
}

impl Drop for Serve {
    fn drop(&mut self) {
        if !self.alive() {
            return;
        }
        self.signal_group("TERM");
        if wait_until(&mut self.child, deadline(secs(20))).is_none() {
            self.signal_group("KILL");
            let _ = wait_until(&mut self.child, deadline(secs(10)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn run_captures_both_pipes_and_stdin() {
        let r = run_with_timeout(
            &argv(&["sh", "-c", "cat; echo err >&2; exit 3"]),
            Some("in\r\n"),
            30,
        )
        .expect("sh runs");
        let want = Ran {
            rc: 3,
            stdout: "in\n".into(),
            stderr: "err\n".into(),
        };
        assert_eq!(r, want);
    }

    #[test]
    fn run_times_out_as_124() {
        let r = run_with_timeout(&argv(&["sleep", "5"]), None, 0).expect("sleep runs");
        assert_eq!(r.rc, 124);
        assert_eq!(r.stderr, "timed out after 0 s");
    }

    #[test]
    fn run_reports_a_signal_as_negative() {
        let r = run_with_timeout(&argv(&["sh", "-c", "kill -9 $$"]), None, 30).expect("runs");
        assert_eq!(r.rc, -9);
    }

    #[test]
    fn chunked_bodies_join() {
        let raw = b"3\r\nabc\r\n2;x\r\nde\r\n0\r\n\r\n";
        assert_eq!(dechunk(raw), Some(b"abcde".to_vec()));
        assert_eq!(dechunk(b"3\r\nab"), None);
    }

    #[test]
    fn http_reads_a_content_length_body() {
        let l = TcpListener::bind(("127.0.0.1", 0)).expect("bind");
        let port = l.local_addr().expect("addr").port();
        let server = std::thread::spawn(move || {
            let (mut c, _) = l.accept().expect("accept");
            let mut req = [0u8; 4096];
            let _ = c.read(&mut req);
            let _ = c.write_all(b"HTTP/1.0 503 Busy\r\nContent-Length: 4\r\n\r\nfull");
        });
        let r = http(port, "POST", "/x", Some(b"{}"), secs(5)).expect("a response");
        let want = Resp {
            status: 503,
            body: b"full".to_vec(),
        };
        assert_eq!(r, want);
        server.join().expect("server thread");
    }
}
