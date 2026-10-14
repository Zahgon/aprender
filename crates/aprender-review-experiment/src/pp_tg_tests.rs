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

#[test]
fn the_shot_body_asks_for_exactly_tg_tokens_past_eos() {
    let b = shot_body("qwen3.5-4b", "hello", TG);
    assert_eq!(b["max_tokens"], TG);
    assert_eq!(b["ignore_eos"], true);
    assert_eq!(b["temperature"], 0.0);
    assert_eq!(b["stream"], false);
}
