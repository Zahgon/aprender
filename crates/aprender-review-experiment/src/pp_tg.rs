//! llama-bench-shaped pp/tg phases from one client (#5305).
//!
//! The comparator is a server driven by the same client as apr, never the
//! llama-bench binary (§5.3, I-15). Each shot is one non-streamed chat request
//! whose server-reported shape is exactly `pp` prompt tokens and `tg` generated
//! tokens. Its server timings give one sample per phase:
//!
//! - ttft = `prompt_ms`;
//! - prefill = 1000 / `prompt_per_second` (ms per token);
//! - decode = 1000 / `predicted_per_second` (ms per token).
//!
//! A shot of another shape or with no server timing voids the whole series: a
//! figure is never the median of the shots that happened to fit.

use crate::crux_perf::Phases;
use crate::harness::Parsed;
use serde_json::{json, Value};

/// llama-bench's default prompt length.
pub const PP: u64 = 512;
/// llama-bench's default generation length.
pub const TG: u64 = 128;

/// Why a series of shots gives no phases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShotRefusal {
    NoShots,
    /// Shot `i` reported no usage, or a usage other than `pp` / `tg`.
    WrongShape {
        shot: usize,
        prompt: u64,
        completion: u64,
    },
    /// Shot `i` carried no positive, finite server timing.
    Untimed(usize),
    /// Fewer shots than [`crate::crux_perf::MIN_RUNS`].
    TooFew(usize),
}

/// The request for one shot: greedy, exactly `tg` tokens, end-of-sequence ignored
/// so a short answer cannot shrink the decode sample.
/// (`json!` expands to an `unwrap` of an infallible `to_value`, hence the allow.)
#[allow(clippy::disallowed_methods)]
#[must_use]
pub fn shot_body(model: &str, prompt: &str, tg: u64) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": tg,
        "ignore_eos": true,
        "stream": false,
    })
}

/// One shot's (ttft ms, prefill ms/token, decode ms/token), or why none.
fn sample(i: usize, s: &Parsed, pp: u64, tg: u64) -> Result<[f64; 3], ShotRefusal> {
    let (prompt, completion) = s
        .tokens
        .as_ref()
        .map_or((0, 0), |t| (t.prompt, t.completion));
    if prompt != pp || completion != tg {
        return Err(ShotRefusal::WrongShape {
            shot: i,
            prompt,
            completion,
        });
    }
    let t = s.server.as_ref().ok_or(ShotRefusal::Untimed(i))?;
    let per_tok = |tps: Option<f64>| {
        tps.filter(|x| x.is_finite() && *x > 0.0)
            .map(|x| 1000.0 / x)
    };
    match (
        t.prompt_ms,
        per_tok(t.prompt_per_second),
        per_tok(t.predicted_per_second),
    ) {
        (ttft, Some(p), Some(d)) if ttft.is_finite() && ttft > 0.0 => Ok([ttft, p, d]),
        _ => Err(ShotRefusal::Untimed(i)),
    }
}

/// One engine's phases from its shots at shape (`pp`, `tg`).
///
/// # Errors
/// No shots, any shot of another shape or untimed, or fewer than `MIN_RUNS` shots.
pub fn phases_from_shots(
    shots: &[Parsed],
    pp: u64,
    tg: u64,
    seed: u64,
) -> Result<Phases, ShotRefusal> {
    if shots.is_empty() {
        return Err(ShotRefusal::NoShots);
    }
    let mut cols: [Vec<f64>; 3] = Default::default();
    for (i, s) in shots.iter().enumerate() {
        for (col, x) in cols.iter_mut().zip(sample(i, s, pp, tg)?) {
            col.push(x);
        }
    }
    let [t, p, d] = &cols;
    Phases::from_samples(t, p, d, seed).ok_or(ShotRefusal::TooFew(shots.len()))
}

#[cfg(test)]
#[path = "pp_tg_tests.rs"]
mod tests;
