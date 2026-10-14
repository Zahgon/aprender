//! #2853: the SafeTensors serve route (`apr serve run model.safetensors`) dropped
//! `ignore_eos` and `seed` silently — `ChatCompletionRequest` had neither field.
//! Both are now honoured: `ignore_eos: true` empties the stop set, and `seed`
//! reaches the Session sampler. A wrong-typed value is a 400 naming the field.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use axum::body::to_bytes;
use realizar::apr_transformer::{AprTransformer, AprTransformerConfig};

const VOCAB: usize = 100;
const IM_END: u32 = 7;
const EOS: u32 = 9;
const MAX_TOKENS: u64 = 8;

fn vocab() -> Vec<String> {
    (0..VOCAB)
        .map(|i| match i {
            0 => "<unk>".to_string(),
            7 => "<|im_end|>".to_string(),
            _ => format!("t{i}"),
        })
        .collect()
}

/// A tiny zero-weight transformer: the logits are exactly `bias`.
fn state_with_bias(bias: Vec<f32>) -> SafeTensorsState {
    let mut t = AprTransformer::new(AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 16,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: VOCAB,
        intermediate_dim: 32,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        eos_token_id: None,
        ..Default::default()
    });
    t.lm_head_bias = Some(bias);
    let vocab = vocab();
    SafeTensorsState {
        transformer: Some(std::sync::Arc::new(std::sync::Mutex::new(t))),
        tokenizer_info: Some(SafeTensorsTokenizerInfo {
            tokenizer: std::sync::Arc::new(
                realizar::tokenizer::BPETokenizer::new(vocab.clone(), vec![], "<unk>")
                    .expect("tiny tokenizer"),
            ),
            vocab,
            bos_token_id: None,
            eos_token_id: Some(EOS),
        }),
        model_path: "test".to_string(),
    }
}

/// Greedy decoding emits `always` at every step.
fn state_always(always: u32) -> SafeTensorsState {
    let mut bias = vec![0.0; VOCAB];
    bias[always as usize] = 10.0;
    state_with_bias(bias)
}

/// A flat distribution over the ordinary tokens (stop ids pushed far down), so a
/// sampled reply depends on the sampler's seed and on nothing else.
fn state_flat() -> SafeTensorsState {
    let mut bias = vec![0.0; VOCAB];
    for stop in [0, IM_END, EOS] {
        bias[stop as usize] = -100.0;
    }
    state_with_bias(bias)
}

async fn reply(resp: axum::response::Response) -> (axum::http::StatusCode, serde_json::Value) {
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 64 * 1024).await.expect("body");
    (status, serde_json::from_slice(&bytes).expect("json"))
}

async fn chat(
    state: SafeTensorsState,
    extra: serde_json::Value,
) -> (axum::http::StatusCode, serde_json::Value) {
    let mut body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": MAX_TOKENS,
    });
    for (k, v) in extra.as_object().expect("object") {
        body[k] = v.clone();
    }
    reply(safetensors_chat_completions_handler(axum::extract::State(state), axum::Json(body)).await)
        .await
}

fn completion_tokens(v: &serde_json::Value) -> u64 {
    v["usage"]["completion_tokens"]
        .as_u64()
        .expect("completion_tokens")
}

fn content(v: &serde_json::Value) -> String {
    v["choices"][0]["message"]["content"]
        .as_str()
        .expect("content")
        .to_string()
}

// ---- ignore_eos -----------------------------------------------------------

#[tokio::test]
async fn chat_ignore_eos_true_runs_past_the_turn_end_to_max_tokens() {
    let (status, v) = chat(
        state_always(IM_END),
        serde_json::json!({"ignore_eos": true}),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{v}");
    assert_eq!(completion_tokens(&v), MAX_TOKENS, "{v}");
    // On EOS too, and the reply text carries every generated token: the last
    // one is not trimmed as if it were the stop the loop ended on.
    let (_, v) = chat(state_always(EOS), serde_json::json!({"ignore_eos": true})).await;
    assert_eq!(completion_tokens(&v), MAX_TOKENS, "{v}");
    assert_eq!(content(&v), "t9".repeat(MAX_TOKENS as usize), "{v}");
}

#[tokio::test]
async fn chat_ignore_eos_false_and_absent_stop_at_the_turn_end_as_before() {
    for extra in [
        serde_json::json!({"ignore_eos": false}),
        serde_json::json!({}),
        serde_json::json!({"ignore_eos": null}),
    ] {
        let (status, v) = chat(state_always(IM_END), extra.clone()).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{extra}: {v}");
        assert_eq!(completion_tokens(&v), 0, "{extra}: {v}");
    }
}

#[tokio::test]
async fn generate_ignore_eos_true_runs_past_eos_to_max_tokens() {
    let run = |extra: serde_json::Value| async move {
        let mut body = serde_json::json!({"prompt": "t1", "max_tokens": MAX_TOKENS});
        for (k, v) in extra.as_object().expect("object") {
            body[k] = v.clone();
        }
        reply(
            safetensors_generate_handler(axum::extract::State(state_always(EOS)), axum::Json(body))
                .await,
        )
        .await
    };
    let (_, on) = run(serde_json::json!({"ignore_eos": true})).await;
    assert_eq!(on["tokens_generated"], MAX_TOKENS, "{on}");
    let (_, off) = run(serde_json::json!({})).await;
    assert_eq!(off["tokens_generated"], 0, "{off}");
}

#[tokio::test]
async fn ollama_chat_ignore_eos_is_not_an_ollama_option_and_stops_as_before() {
    // Ollama's `options` has no ignore_eos; the Ollama route keeps today's stop.
    let req: super::super::ollama::OllamaChatRequest = serde_json::from_value(serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "stream": false,
        "options": {"num_predict": MAX_TOKENS},
    }))
    .expect("ollama request");
    let (status, v) = reply(
        safetensors_ollama_chat_handler(
            axum::extract::State(state_always(IM_END)),
            axum::Json(req),
        )
        .await,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{v}");
    assert_eq!(v["message"]["content"], "", "{v}");
}

// ---- seed -------------------------------------------------------------------

fn sampled(seed: Option<u64>) -> serde_json::Value {
    let mut extra = serde_json::json!({"temperature": 1.0, "top_p": 1.0});
    if let Some(s) = seed {
        extra["seed"] = serde_json::json!(s);
    }
    extra
}

#[tokio::test]
async fn chat_same_seed_same_reply_other_seed_other_reply() {
    let (_, a) = chat(state_flat(), sampled(Some(42))).await;
    let (_, b) = chat(state_flat(), sampled(Some(42))).await;
    let (_, c) = chat(state_flat(), sampled(Some(43))).await;
    assert_eq!(completion_tokens(&a), MAX_TOKENS, "{a}");
    assert_eq!(
        content(&a),
        content(&b),
        "same seed must reproduce the reply"
    );
    // Kills the silent-drop mutant: a dropped seed falls back to one default seed
    // and the two replies would be identical.
    assert_ne!(
        content(&a),
        content(&c),
        "a different seed must reach the sampler"
    );
}

#[tokio::test]
async fn chat_absent_seed_is_the_engine_default_seed_as_before() {
    let (_, absent) = chat(state_flat(), sampled(None)).await;
    let (_, default) = chat(
        state_flat(),
        sampled(Some(realizar::apr_transformer::DEFAULT_SEED)),
    )
    .await;
    assert_eq!(content(&absent), content(&default));
}

#[tokio::test]
async fn ollama_options_seed_reaches_the_sampler() {
    let ollama = |seed: u64| async move {
        let req: super::super::ollama::OllamaChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false,
            "options": {"num_predict": MAX_TOKENS, "temperature": 1.0, "top_p": 1.0, "seed": seed},
        }))
        .expect("ollama request");
        let (_, v) = reply(
            safetensors_ollama_chat_handler(axum::extract::State(state_flat()), axum::Json(req))
                .await,
        )
        .await;
        v["message"]["content"]
            .as_str()
            .expect("content")
            .to_string()
    };
    let (_, openai) = chat(state_flat(), sampled(Some(42))).await;
    assert_eq!(
        ollama(42).await,
        content(&openai),
        "options.seed = OpenAI seed"
    );
    assert_ne!(ollama(42).await, ollama(43).await);
}

// ---- wrong types are refused, unknown fields stay accepted -----------------

#[tokio::test]
async fn wrong_typed_ignore_eos_or_seed_is_a_400_naming_the_field() {
    for (extra, param) in [
        (serde_json::json!({"ignore_eos": "yes"}), "ignore_eos"),
        (serde_json::json!({"ignore_eos": 1}), "ignore_eos"),
        (serde_json::json!({"seed": -1}), "seed"),
        (serde_json::json!({"seed": "42"}), "seed"),
        (serde_json::json!({"seed": 1.5}), "seed"),
    ] {
        let (status, v) = chat(state_always(IM_END), extra.clone()).await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST, "{extra}: {v}");
        assert_eq!(v["error"]["param"], param, "{extra}: {v}");
        let msg = v["error"]["message"].as_str().expect("message");
        assert!(msg.contains(param) && msg.contains("SafeTensors"), "{msg}");
    }
}

#[tokio::test]
async fn unknown_fields_are_still_ignored() {
    let (status, v) = chat(state_always(11), serde_json::json!({"not_a_field": 1})).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{v}");
    assert_eq!(completion_tokens(&v), MAX_TOKENS, "{v}");
}

#[test]
fn request_type_carries_both_fields_through_both_parse_paths() {
    let structured = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "ignore_eos": true,
        "seed": 9,
    });
    let req = parse_chat_completion_request(&structured).expect("structured");
    assert_eq!((req.ignore_eos, req.seed), (Some(true), Some(9)));
    // A non-string content forces the raw-JSON fallback; it must keep them too.
    let fallback = serde_json::json!({
        "messages": [{"role": "user", "content": 5}],
        "ignore_eos": true,
        "seed": 9,
    });
    let req = parse_chat_completion_request(&fallback).expect("fallback");
    assert_eq!((req.ignore_eos, req.seed), (Some(true), Some(9)));
    let absent =
        parse_chat_completion_request(&serde_json::json!({"messages": []})).expect("absent");
    assert_eq!((absent.ignore_eos, absent.seed), (None, None));
}

#[test]
fn reply_text_is_cut_at_the_turn_end_unless_ignore_eos() {
    let text = "a<|im_end|>b".to_string();
    assert_eq!(st_reply_text(text.clone(), false), "a");
    assert_eq!(st_reply_text(text.clone(), true), text);
    assert_eq!(st_reply_text("plain".to_string(), false), "plain");
}
