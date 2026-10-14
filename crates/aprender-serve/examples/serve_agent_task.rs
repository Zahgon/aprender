//! E-2 (#5348): an OpenAI client completes one tool-calling task against a
//! running `apr serve`, and writes a receipt.
//!
//! ```text
//! cargo run -p aprender-serve --example serve_agent_task --release -- \
//!     --url http://127.0.0.1:8080 --binary target/release/apr \
//!     --model default --backend cpu --out e2.json
//! ```
//!
//! The gateway must be the binary given: its `GET /` `binary_sha256` (R-8) has
//! to equal the sha256 of `--binary`, or the run is refused before any chat.
//! The task and its judges are `realizar::api::agent_task`. The full transcript
//! (both requests and both responses) is written beside the receipt, and the
//! receipt carries the sha256 of those exact bytes.
//!
//! Exit 0 only on `pass`. `fail`, `refused` and `not_measured` exit 1; bad
//! arguments or an unreadable `--binary` exit 2.

use std::process::ExitCode;
use std::time::Duration;

use realizar::api::agent_task::{
    first_request, judge_final, judge_tool_call, second_request, sha256_hex,
};
use realizar::api::file_sha256_hex;
use serde_json::{json, Value};

struct Args {
    url: String,
    binary: String,
    model: String,
    backend: String,
    out: String,
    transcript: Option<String>,
    max_tokens: u32,
    timeout_s: u64,
}

const USAGE: &str = "usage: serve_agent_task --url URL --binary APR --model ID --backend NAME \
    --out FILE [--transcript FILE] [--max-tokens N] [--timeout-s N]";

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        url: String::new(),
        binary: String::new(),
        model: String::new(),
        backend: String::new(),
        out: String::new(),
        transcript: None,
        max_tokens: 512,
        timeout_s: 600,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--url" => a.url = value()?.trim_end_matches('/').to_string(),
            "--binary" => a.binary = value()?,
            "--model" => a.model = value()?,
            "--backend" => a.backend = value()?,
            "--out" => a.out = value()?,
            "--transcript" => a.transcript = Some(value()?),
            "--max-tokens" => {
                a.max_tokens = value()?.parse().map_err(|e| format!("--max-tokens: {e}"))?
            },
            "--timeout-s" => {
                a.timeout_s = value()?.parse().map_err(|e| format!("--timeout-s: {e}"))?
            },
            other => return Err(format!("unknown argument {other}")),
        }
    }
    for (name, v) in [
        ("--url", &a.url),
        ("--binary", &a.binary),
        ("--model", &a.model),
        ("--backend", &a.backend),
        ("--out", &a.out),
    ] {
        if v.is_empty() {
            return Err(format!("{name} is required"));
        }
    }
    Ok(a)
}

/// How the run ended, before the receipt is written.
enum Verdict {
    Pass(String),
    Fail(String),
    Refused(String),
    NotMeasured(String),
}

impl Verdict {
    fn word(&self) -> &'static str {
        match self {
            Self::Pass(_) => "pass",
            Self::Fail(_) => "fail",
            Self::Refused(_) => "refused",
            Self::NotMeasured(_) => "not_measured",
        }
    }
}

struct Http {
    client: reqwest::blocking::Client,
    base: String,
}

impl Http {
    /// GET or POST; a transport error is `Err` (not measured), any HTTP status is `Ok`.
    fn call(&self, path: &str, body: Option<&Value>) -> Result<(u16, Value), String> {
        let url = format!("{}{path}", self.base);
        let builder = match body {
            Some(b) => self.client.post(&url).json(b),
            None => self.client.get(&url),
        };
        let resp = builder.send().map_err(|e| format!("{path}: {e}"))?;
        let status = resp.status().as_u16();
        let text = resp.text().map_err(|e| format!("{path}: {e}"))?;
        let parsed = serde_json::from_str(&text).unwrap_or(Value::String(text));
        Ok((status, parsed))
    }
}

/// One chat turn, recorded into the transcript whatever happens.
fn turn(
    http: &Http,
    n: u32,
    request: Value,
    transcript: &mut Vec<Value>,
) -> Result<Value, Verdict> {
    let result = http.call("/v1/chat/completions", Some(&request));
    let (status, response) = match result {
        Ok(r) => r,
        Err(e) => {
            transcript.push(json!({"turn": n, "request": request, "error": e}));
            return Err(Verdict::NotMeasured(e));
        },
    };
    transcript.push(json!({"turn": n, "request": request, "status": status, "response": response}));
    if status == 200 {
        Ok(response)
    } else {
        Err(Verdict::Fail(format!("turn {n}: HTTP {status}")))
    }
}

/// The two-turn task. Every request and response lands in `transcript`.
fn run_task(http: &Http, a: &Args, transcript: &mut Vec<Value>) -> Verdict {
    let first = match turn(http, 1, first_request(&a.model, a.max_tokens), transcript) {
        Ok(r) => r,
        Err(v) => return v,
    };
    let call = match judge_tool_call(&first) {
        Ok(c) => c,
        Err(e) => return Verdict::Fail(format!("turn 1: {e}")),
    };
    let second = match turn(
        http,
        2,
        second_request(&a.model, a.max_tokens, &call),
        transcript,
    ) {
        Ok(r) => r,
        Err(v) => return v,
    };
    match judge_final(&second) {
        Ok(text) => Verdict::Pass(text),
        Err(e) => Verdict::Fail(format!("turn 2: {e}")),
    }
}

/// R-8: the gateway must report the sha of `--binary`, on `GET /` or else
/// `/health` (as `LlmClient::require_binary_sha256` reads it). `None` = go on.
fn check_gateway(http: &Http, expected: &str) -> (Option<String>, Option<Verdict>) {
    let mut got = None;
    for path in ["/", "/health"] {
        match http.call(path, None) {
            Ok((200, body)) => {
                got = body
                    .get("binary_sha256")
                    .and_then(Value::as_str)
                    .map(str::to_ascii_lowercase);
            },
            Ok(_) => {},
            Err(e) => return (None, Some(Verdict::NotMeasured(e))),
        }
        if got.is_some() {
            break;
        }
    }
    match got.as_deref() {
        Some(g) if g == expected => (got, None),
        _ => {
            let why =
                format!("gateway binary_sha256 {got:?} is not the sha256 of --binary ({expected})");
            (got, Some(Verdict::Refused(why)))
        },
    }
}

fn write(path: &str, text: &str) -> Result<(), String> {
    std::fs::write(path, text).map_err(|e| format!("write {path}: {e}"))
}

fn main() -> ExitCode {
    let a = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("serve_agent_task: {e}\n{USAGE}");
            return ExitCode::from(2);
        },
    };
    let expected = match file_sha256_hex(std::path::Path::new(&a.binary)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("serve_agent_task: --binary {}: {e}", a.binary);
            return ExitCode::from(2);
        },
    };
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(a.timeout_s))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("serve_agent_task: http client: {e}");
            return ExitCode::from(2);
        },
    };
    let http = Http {
        client,
        base: a.url.clone(),
    };

    let mut transcript = Vec::new();
    let (gateway_sha, refused) = check_gateway(&http, &expected);
    let verdict = match refused {
        Some(v) => v,
        None => run_task(&http, &a, &mut transcript),
    };

    let transcript_path = a
        .transcript
        .clone()
        .unwrap_or_else(|| format!("{}.transcript.json", a.out));
    let transcript_text = match serde_json::to_string_pretty(&transcript) {
        Ok(t) => t + "\n",
        Err(e) => {
            eprintln!("serve_agent_task: transcript json: {e}");
            return ExitCode::from(1);
        },
    };
    let (reason, answer) = match &verdict {
        Verdict::Pass(text) => (None, Some(text.clone())),
        Verdict::Fail(r) | Verdict::Refused(r) | Verdict::NotMeasured(r) => (Some(r.clone()), None),
    };
    let receipt = json!({
        "row": "E-2",
        "issue": 5348,
        "shape": "openai",
        "task": realizar::api::agent_task::TOOL,
        "url": a.url,
        "model": a.model,
        "backend": a.backend,
        "binary": a.binary,
        "binary_sha256": expected,
        "gateway_binary_sha256": gateway_sha,
        "transcript": transcript_path,
        "transcript_sha256": sha256_hex(transcript_text.as_bytes()),
        "turns": transcript.len(),
        "verdict": verdict.word(),
        "reason": reason,
        "final_answer": answer,
    });
    let written = serde_json::to_string_pretty(&receipt)
        .map_err(|e| format!("receipt json: {e}"))
        .and_then(|r| {
            write(&transcript_path, &transcript_text).and_then(|()| write(&a.out, &(r + "\n")))
        });
    if let Err(e) = written {
        eprintln!("serve_agent_task: {e}");
        return ExitCode::from(1);
    }
    println!(
        "{} E-2 {} ({} turns) {}",
        verdict.word().to_ascii_uppercase(),
        a.out,
        transcript.len(),
        reason.unwrap_or_default()
    );
    if matches!(verdict, Verdict::Pass(_)) {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
