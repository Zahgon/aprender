//! Case table for the V2 serve receipt (#4953). One GREEN row, and one RED row
//! per refusal: each RED row breaks exactly one input of the GREEN fixture.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::llm::benchmark::{AggregateStats, StatSummary};
use crate::llm::loadtest::LoadTestResult;

fn req(ttft_ms: f64, latency_ms: f64, prompt: u32, completion: u32) -> RequestDetail {
    RequestDetail {
        latency_ms,
        ttft_ms,
        completion_tokens: completion,
        prompt_tokens: prompt,
        itl_ms: 0.0,
        finish_reason: Some("length".to_string()),
    }
}

fn run(details: Vec<RequestDetail>, failed: u64) -> LoadTestResult {
    let ok = details.len() as u64;
    let s: LoadTestResult = serde_json::from_value(serde_json::json!({
        "total_requests": ok + failed, "successful": ok, "failed": failed,
        "throughput_rps": 1.0, "latency_p50_ms": 1.0, "latency_p95_ms": 1.0,
        "latency_p99_ms": 1.0, "ttft_p50_ms": 1.0, "tokens_per_sec": 1.0,
        "avg_tok_per_req": 128.0, "itl_p50_ms": 1.0, "decode_tok_per_sec": 1.0,
        "timestamp": "2026-10-09T00:00:00Z", "runtime_name": "apr",
        "elapsed_secs": 1.0, "concurrency": 1,
    }))
    .expect("the minimal LoadTestResult fixture must deserialize");
    LoadTestResult {
        request_details: details,
        ..s
    }
}

fn stat() -> StatSummary {
    StatSummary {
        mean: 0.0,
        stddev: 0.0,
        ci_95_lower: 0.0,
        ci_95_upper: 0.0,
        values: Vec::new(),
    }
}

fn report(runs: Vec<LoadTestResult>, load_ms: Option<f64>, alive: Option<bool>) -> BenchmarkReport {
    BenchmarkReport {
        runs,
        aggregate: AggregateStats {
            throughput_rps: stat(),
            latency_p50: stat(),
            tokens_per_sec: stat(),
            ttft_p50: stat(),
            tpot_p50: stat(),
        },
        regressions: Vec::new(),
        load_ms,
        server_alive_after: alive,
    }
}

fn meta() -> ReceiptMeta {
    ReceiptMeta {
        apr_version: "0.71.0".to_string(),
        apr_commit: "0123456789".to_string(),
        model_sha256: "ab".repeat(32),
        host: "gpu-a".to_string(),
        workload: "prompts-w1.jsonl".to_string(),
    }
}

/// Three requests: TTFT 100/120/140 ms on 512 prompt tokens, 128 completion
/// tokens decoded in 1270 ms each. load_ms 4000.
fn green_runs() -> Vec<LoadTestResult> {
    vec![run(
        vec![
            req(100.0, 1370.0, 512, 128),
            req(120.0, 1390.0, 512, 128),
            req(140.0, 1410.0, 512, 128),
        ],
        0,
    )]
}

#[test]
fn green_a_loaded_once_pp512_tg128_run_is_a_receipt() {
    let r = build(&report(green_runs(), Some(4000.0), Some(true)), meta()).unwrap();
    assert_eq!(r.schema, SCHEMA);
    assert_eq!(r.n, 3);
    assert_eq!(r.load_ms, 4000.0);
    assert_eq!(
        r.ttft_ms,
        Spread {
            p50: 120.0,
            p95: 140.0,
            max: 140.0
        }
    );
    // pp512 = 512 / 0.120 s; tg128 = 127 / 1.270 s.
    assert!(
        (r.pp512_tok_s - 512.0 / 0.120).abs() < 1e-6,
        "{}",
        r.pp512_tok_s
    );
    assert!((r.tg128_tok_s - 100.0).abs() < 1e-6, "{}", r.tg128_tok_s);
    assert!((r.max_ttft_over_load - 140.0 / 4000.0).abs() < 1e-12);
    let j = serde_json::to_value(&r).unwrap();
    for k in [
        "load_ms",
        "ttft_ms",
        "pp512_tok_s",
        "tg128_tok_s",
        "model_sha256",
        "apr_commit",
        "n",
    ] {
        assert!(j.get(k).is_some(), "receipt JSON lacks {k}: {j}");
    }
}

fn refusal(rep: BenchmarkReport, needle: &str) {
    let e = build(&rep, meta()).expect_err("this row must refuse");
    assert!(e.contains(needle), "refusal {e:?} does not name {needle:?}");
}

#[test]
fn red_no_load_ms_when_the_server_was_not_started_here() {
    refusal(report(green_runs(), None, Some(true)), "no load_ms");
}

#[test]
fn red_zero_load_ms() {
    refusal(
        report(green_runs(), Some(0.0), Some(true)),
        "not a measured load",
    );
}

#[test]
fn red_the_server_died_or_was_never_tracked() {
    refusal(report(green_runs(), Some(4000.0), Some(false)), "not alive");
    refusal(report(green_runs(), Some(4000.0), None), "not alive");
}

#[test]
fn red_a_failed_request() {
    let mut runs = green_runs();
    runs[0].failed = 1;
    refusal(report(runs, Some(4000.0), Some(true)), "failed request");
}

#[test]
fn red_zero_requests() {
    refusal(
        report(vec![run(Vec::new(), 0)], Some(4000.0), Some(true)),
        "zero measured",
    );
}

#[test]
fn red_no_streamed_ttft() {
    let runs = vec![run(vec![req(0.0, 1370.0, 512, 128)], 0)];
    refusal(report(runs, Some(4000.0), Some(true)), "no streamed TTFT");
}

#[test]
fn red_a_completion_that_is_not_128() {
    let runs = vec![run(vec![req(100.0, 1370.0, 512, 127)], 0)];
    refusal(report(runs, Some(4000.0), Some(true)), "not 128");
}

#[test]
fn red_a_prompt_that_is_not_512_and_its_edges() {
    let at = |p: u32| vec![run(vec![req(100.0, 1370.0, p, 128)], 0)];
    refusal(report(at(486), Some(4000.0), Some(true)), "not pp512");
    refusal(report(at(538), Some(4000.0), Some(true)), "not pp512");
    // The edges of ±5% (486.4 .. 537.6) are inside.
    build(&report(at(487), Some(4000.0), Some(true)), meta()).unwrap();
    build(&report(at(537), Some(4000.0), Some(true)), meta()).unwrap();
}

#[test]
fn red_a_request_that_waited_a_whole_load() {
    // A model reloaded per request makes TTFT at least the load time.
    let runs = vec![run(
        vec![req(100.0, 1370.0, 512, 128), req(4000.0, 5270.0, 512, 128)],
        0,
    )];
    refusal(report(runs, Some(4000.0), Some(true)), "stayed loaded");
}

#[test]
fn p95_is_nearest_rank() {
    let s = spread((1..=20).map(f64::from).collect());
    assert_eq!(
        s,
        Spread {
            p50: 10.5,
            p95: 19.0,
            max: 20.0
        }
    );
    let one = spread(vec![7.0]);
    assert_eq!(
        one,
        Spread {
            p50: 7.0,
            p95: 7.0,
            max: 7.0
        }
    );
}
