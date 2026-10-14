use super::*;
use crate::harness::{classify, Reply};

/// A llama-server-shaped reply: usage plus the `timings` object.
fn shot(prompt: u64, completion: u64, prompt_ms: f64, pps: f64, tps: f64) -> Parsed {
    classify(&Reply {
        status: 200,
        body: format!(
            r#"{{"choices":[{{"message":{{"content":"VERDICT: PASS"}}}}],
            "usage":{{"prompt_tokens":{prompt},"completion_tokens":{completion}}},
            "timings":{{"prompt_ms":{prompt_ms},"predicted_ms":2000.0,
            "prompt_per_second":{pps},"predicted_per_second":{tps}}}}}"#
        ),
        wall_ms: 2500.0,
    })
}

fn good(n: usize) -> Vec<Parsed> {
    (0..n).map(|_| shot(PP, TG, 128.0, 4000.0, 64.0)).collect()
}

#[test]
fn shots_of_the_pp_tg_shape_give_ms_per_token_phases() {
    let p = phases_from_shots(&good(5), PP, TG, 1).expect("five timed shots");
    assert_eq!(p.ttft.median, 128.0);
    assert!((p.prefill.median - 0.25).abs() < 1e-12, "1000 / 4000 tok/s");
    assert!((p.decode.median - 15.625).abs() < 1e-12, "1000 / 64 tok/s");
    assert_eq!(p.decode.n, 5);
}

#[test]
fn falsify_crux_perf_007_a_shot_of_another_shape_voids_the_series() {
    // An engine that stopped at EOS after 40 tokens, or truncated the prompt,
    // is not measured at pp512/tg128: the series refuses rather than dropping it.
    let mut shots = good(6);
    shots[2] = shot(PP, 40, 128.0, 4000.0, 64.0);
    assert_eq!(
        phases_from_shots(&shots, PP, TG, 1),
        Err(ShotRefusal::WrongShape {
            shot: 2,
            prompt: PP,
            completion: 40
        })
    );
    shots[2] = shot(511, TG, 128.0, 4000.0, 64.0);
    assert!(matches!(
        phases_from_shots(&shots, PP, TG, 1),
        Err(ShotRefusal::WrongShape { shot: 2, .. })
    ));
    // A reply with no usage at all is a wrong shape, never a pass.
    shots[2].tokens = None;
    assert!(matches!(
        phases_from_shots(&shots, PP, TG, 1),
        Err(ShotRefusal::WrongShape {
            shot: 2,
            prompt: 0,
            completion: 0
        })
    ));
}

#[test]
fn falsify_crux_perf_008_an_untimed_shot_voids_the_series() {
    let mut shots = good(6);
    shots[4].server = None;
    assert_eq!(
        phases_from_shots(&shots, PP, TG, 1),
        Err(ShotRefusal::Untimed(4))
    );
    for bad in [0.0, -1.0, f64::NAN] {
        let mut shots = good(6);
        shots[1] = shot(PP, TG, 128.0, 4000.0, 64.0);
        if let Some(t) = shots[1].server.as_mut() {
            t.predicted_per_second = Some(bad);
        }
        assert_eq!(
            phases_from_shots(&shots, PP, TG, 1),
            Err(ShotRefusal::Untimed(1)),
            "decode tok/s {bad}"
        );
    }
    let mut shots = good(6);
    if let Some(t) = shots[0].server.as_mut() {
        t.prompt_ms = f64::NAN;
    }
    assert_eq!(
        phases_from_shots(&shots, PP, TG, 1),
        Err(ShotRefusal::Untimed(0))
    );
}

#[test]
fn too_few_or_no_shots_give_no_phases() {
    assert_eq!(phases_from_shots(&[], PP, TG, 1), Err(ShotRefusal::NoShots));
    assert_eq!(
        phases_from_shots(&good(4), PP, TG, 1),
        Err(ShotRefusal::TooFew(4))
    );
}

/// A fake engine: `overhead` template tokens plus `per_word` tokens per word.
fn engine(overhead: u64, per_word: u64) -> impl FnMut(&str) -> Result<u64, String> {
    move |p: &str| Ok(overhead + per_word * p.split_whitespace().count() as u64)
}

#[test]
fn a_fitted_prompt_is_exactly_pp_tokens_on_each_engine() {
    for overhead in [0, 9, 23, 300] {
        let text = fit_prompt(PP, engine(overhead, 1)).expect("one token per word fits");
        assert_eq!(
            overhead + text.split_whitespace().count() as u64,
            PP,
            "overhead {overhead}"
        );
    }
}

#[test]
fn falsify_crux_perf_009_a_prompt_that_cannot_be_pp_tokens_is_refused() {
    // Two tokens per word plus an odd overhead never lands on 512: the fit
    // refuses rather than handing back the nearest miss.
    assert!(fit_prompt(PP, engine(23, 2)).is_err());
    // A template longer than pp leaves no room for any prompt.
    assert!(fit_prompt(PP, engine(600, 1)).is_err());
    // A probe failure is the fit's failure.
    assert!(fit_prompt(PP, |_: &str| Err("refused".to_string())).is_err());
}

#[test]
fn the_shot_body_asks_for_exactly_tg_tokens_past_eos() {
    let b = shot_body("qwen3.5-4b", "hello", TG);
    assert_eq!(b["max_tokens"], TG);
    assert_eq!(b["ignore_eos"], true);
    assert_eq!(b["temperature"], 0.0);
    assert_eq!(b["stream"], false);
}

/// A fake server: `overhead(url)` template tokens plus one per word, exactly
/// `max_tokens` generated, and a slow first full-length shot (cold load).
fn server(overhead: fn(&str) -> u64) -> impl FnMut(&str, &Value) -> Result<Reply, String> {
    let mut warm = std::collections::HashSet::new();
    move |url: &str, body: &Value| {
        let words = body["messages"][0]["content"]
            .as_str()
            .map_or(0, |s| s.split_whitespace().count() as u64);
        let max = body["max_tokens"].as_u64().unwrap_or(0);
        let cold = max > 1 && warm.insert(url.to_string());
        let (ms, pps, tps) = if cold {
            (9999.0, 50.0, 2.0)
        } else {
            (128.0, 4000.0, 64.0)
        };
        let prompt = overhead(url) + words;
        Ok(Reply {
            status: 200,
            body: format!(
                r#"{{"choices":[{{"message":{{"content":"x"}}}}],
                "usage":{{"prompt_tokens":{prompt},"completion_tokens":{max}}},
                "timings":{{"prompt_ms":{ms},"predicted_ms":1.0,
                "prompt_per_second":{pps},"predicted_per_second":{tps}}}}}"#
            ),
            wall_ms: 1.0,
        })
    }
}

fn two_templates(url: &str) -> u64 {
    if url.contains("apr") {
        23
    } else {
        11
    }
}

#[test]
fn each_engine_is_measured_on_its_own_fitted_pp_tg_prompt() {
    let mut send = server(two_templates);
    for url in [
        "http://apr/v1/chat/completions",
        "http://llama/v1/chat/completions",
    ] {
        let p = measure(Engine { url, model: "m" }, 5, PP, TG, 1, &mut send)
            .unwrap_or_else(|e| panic!("{url}: {e:?}"));
        assert_eq!(p.decode.n, 5, "{url}");
        assert!((p.decode.median - 15.625).abs() < 1e-12, "{url}");
    }
}

#[test]
fn falsify_crux_perf_010_the_cold_shot_never_enters_the_series() {
    let p = measure(
        Engine {
            url: "http://apr/x",
            model: "m",
        },
        5,
        PP,
        TG,
        1,
        server(two_templates),
    )
    .expect("warm shots");
    // Every warm shot is 128 ms; a 9999 ms cold shot in the series moves p95.
    assert_eq!(p.ttft.p95, 128.0, "{p:?}");
    assert_eq!(p.ttft.n, 5);
}

#[test]
fn a_dead_server_or_a_reply_without_usage_gives_no_phases() {
    let dead = |_: &str, _: &Value| Err::<Reply, _>("connection refused".to_string());
    assert!(matches!(
        measure(
            Engine {
                url: "u",
                model: "m"
            },
            5,
            PP,
            TG,
            1,
            dead
        ),
        Err(DriveError::Fit(_))
    ));
    let no_usage = |_: &str, _: &Value| {
        Ok(Reply {
            status: 200,
            body: r#"{"choices":[{"message":{"content":"x"}}]}"#.to_string(),
            wall_ms: 1.0,
        })
    };
    assert!(matches!(
        measure(
            Engine {
                url: "u",
                model: "m"
            },
            5,
            PP,
            TG,
            1,
            no_usage
        ),
        Err(DriveError::Fit(_))
    ));
}

#[test]
fn another_shape_is_another_prompt_set() {
    let base = prompt_set_sha(PP, TG);
    assert_eq!(base, prompt_set_sha(PP, TG));
    assert_ne!(base, prompt_set_sha(PP, 64));
    assert_ne!(base, prompt_set_sha(256, TG));
}
