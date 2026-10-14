//! Agent conformance suite over HTTP (#5349, EPIC 0.72 row R-6).
//!
//! Sends every case in `realizar::api::conformance::cases` to a running
//! `apr serve` and writes one receipt cell per case × model × backend.
//!
//! ```text
//! cargo run -p aprender-serve --example serve_conformance -- \
//!     --url http://127.0.0.1:8080 --model qwen3-1.7b-q4km --backend cpu --out receipt.json
//! ```
//!
//! Exit 0 only when every cell is `pass`. A `fail`, a `not_measured` cell or a
//! missing cell exits 1; bad arguments exit 2.

use std::process::ExitCode;
use std::time::Duration;

use realizar::api::conformance::{cases, missing_cells, run_case, CaseRequest, Observed, Verdict};

struct Args {
    url: String,
    model: String,
    backend: String,
    out: Option<String>,
    timeout_s: u64,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        url: String::new(),
        model: String::new(),
        backend: String::new(),
        out: None,
        timeout_s: 300,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--url" => args.url = value()?.trim_end_matches('/').to_string(),
            "--model" => args.model = value()?,
            "--backend" => args.backend = value()?,
            "--out" => args.out = Some(value()?),
            "--timeout-s" => {
                args.timeout_s = value()?.parse().map_err(|e| format!("--timeout-s: {e}"))?;
            },
            other => return Err(format!("unknown argument {other}")),
        }
    }
    for (name, v) in [
        ("--url", &args.url),
        ("--model", &args.model),
        ("--backend", &args.backend),
    ] {
        if v.is_empty() {
            return Err(format!("{name} is required"));
        }
    }
    Ok(args)
}

fn send(
    client: &reqwest::blocking::Client,
    base: &str,
    req: &CaseRequest,
) -> Result<Observed, String> {
    let url = format!("{base}{}", req.path);
    let builder = match req.method {
        "GET" => client.get(&url),
        "POST" => client
            .post(&url)
            .json(req.body.as_ref().unwrap_or(&serde_json::Value::Null)),
        other => return Err(format!("unsupported method {other}")),
    };
    let resp = builder.send().map_err(|e| e.to_string())?;
    let status = resp.status().as_u16();
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let body = resp.text().map_err(|e| e.to_string())?;
    Ok(Observed {
        status,
        content_type,
        body,
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("serve_conformance: {e}");
            eprintln!("usage: serve_conformance --url URL --model ID --backend NAME [--out FILE] [--timeout-s N]");
            return ExitCode::from(2);
        },
    };
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(args.timeout_s))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("serve_conformance: http client: {e}");
            return ExitCode::from(2);
        },
    };

    let suite = cases(&args.model);
    let cells: Vec<_> = suite
        .iter()
        .map(|case| {
            run_case(case, &args.model, &args.backend, |r| {
                send(&client, &args.url, r)
            })
        })
        .collect();
    let missing = missing_cells(
        &suite,
        &[args.model.as_str()],
        &[args.backend.as_str()],
        &cells,
    );

    let mut bad = 0usize;
    for cell in &cells {
        let (word, why) = match &cell.verdict {
            Verdict::Pass => ("PASS", ""),
            Verdict::Fail { reason } => ("FAIL", reason.as_str()),
            Verdict::NotMeasured { reason } => ("NOT_MEASURED", reason.as_str()),
        };
        if cell.verdict != Verdict::Pass {
            bad += 1;
        }
        println!("{word:<12} {:<30} {} {why}", cell.case, cell.row);
    }
    for m in &missing {
        println!("MISSING      {m}");
    }

    let receipt = serde_json::json!({
        "suite": "apr-serve-conformance",
        "issue": 5349,
        "url": args.url,
        "model": args.model,
        "backend": args.backend,
        "cells": cells,
        "missing": missing,
        "pass": cells.len() - bad,
        "total": cells.len(),
    });
    if let Some(path) = &args.out {
        let text = match serde_json::to_string_pretty(&receipt) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("serve_conformance: receipt json: {e}");
                return ExitCode::from(1);
            },
        };
        if let Err(e) = std::fs::write(path, text + "\n") {
            eprintln!("serve_conformance: write {path}: {e}");
            return ExitCode::from(1);
        }
    }
    println!(
        "{}/{} pass, {} missing",
        cells.len() - bad,
        cells.len(),
        missing.len()
    );
    if bad == 0 && missing.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
