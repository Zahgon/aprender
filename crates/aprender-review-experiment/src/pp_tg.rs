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
use crate::harness::{classify, Parsed, Reply};
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

/// Probes a fitted prompt may spend before it gives up.
pub const FIT_TRIES: usize = 8;

/// `n` filler words. Each is one token on the Qwen and Llama vocabularies,
/// but the fit below never relies on that: it reads the engine's own count.
#[must_use]
pub fn filler(n: u64) -> String {
    " the".repeat(usize::try_from(n).unwrap_or(0))
}

/// A user prompt that the engine reports as exactly `pp` prompt tokens.
///
/// The chat template adds a per-engine overhead, so one fixed text is not
/// `pp` tokens on both engines. `probe` sends a prompt and returns the
/// server-reported `prompt_tokens`; the filler length is moved by the miss
/// until the count is exact.
///
/// # Errors
/// A probe error, or no exact fit within [`FIT_TRIES`] probes.
pub fn fit_prompt(
    pp: u64,
    mut probe: impl FnMut(&str) -> Result<u64, String>,
) -> Result<String, String> {
    let mut n = pp;
    let mut seen = Vec::with_capacity(FIT_TRIES);
    for _ in 0..FIT_TRIES {
        let text = filler(n);
        let got = probe(&text)?;
        if got == pp {
            return Ok(text);
        }
        seen.push((n, got));
        let next = i128::from(n) + i128::from(pp) - i128::from(got);
        n = u64::try_from(next.max(1)).unwrap_or(1);
        if seen.iter().any(|&(m, _)| m == n) {
            break;
        }
    }
    Err(format!(
        "no filler length gives {pp} prompt tokens: {seen:?}"
    ))
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

/// Why one engine gave no phases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriveError {
    /// No prompt of exactly `pp` tokens (or a probe failed).
    Fit(String),
    /// The request never got a reply.
    Transport(String),
    /// The timed shots were refused.
    Shots(ShotRefusal),
}

/// One OpenAI-compatible server: its chat-completions URL and model name.
#[derive(Debug, Clone, Copy)]
pub struct Engine<'a> {
    pub url: &'a str,
    pub model: &'a str,
}

/// One engine's phases at (`pp`, `tg`) over `runs` timed shots.
///
/// The prompt is fitted on this engine with one-token probes, then one cold
/// shot is sent and discarded, so model load and first-touch never enter the
/// series. `send` posts a body to a URL ([`crate::harness::post`] in use).
///
/// # Errors
/// See [`DriveError`].
pub fn measure(
    engine: Engine<'_>,
    runs: usize,
    pp: u64,
    tg: u64,
    seed: u64,
    mut send: impl FnMut(&str, &Value) -> Result<Reply, String>,
) -> Result<Phases, DriveError> {
    let mut shot = |prompt: &str, n: u64| {
        send(engine.url, &shot_body(engine.model, prompt, n))
            .map(|r| classify(&r))
            .map_err(DriveError::Transport)
    };
    let prompt = fit_prompt(pp, |text| {
        let p = shot(text, 1).map_err(|e| format!("{e:?}"))?;
        p.tokens
            .map(|t| t.prompt)
            .ok_or_else(|| format!("probe reply has no usage: {:?}", p.verdict))
    })
    .map_err(DriveError::Fit)?;
    shot(&prompt, tg)?;
    let shots = (0..runs)
        .map(|_| shot(&prompt, tg))
        .collect::<Result<Vec<_>, _>>()?;
    phases_from_shots(&shots, pp, tg, seed).map_err(DriveError::Shots)
}

#[cfg(test)]
#[path = "pp_tg_tests.rs"]
mod tests;
