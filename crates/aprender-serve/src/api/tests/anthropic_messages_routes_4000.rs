//! `POST /v1/messages` through the real router (#4000 R-1).
//!
//! The server is the `AprTransformer` fixture of aprender#2609, which loads a
//! model and generates, so a 200 here is a generation, not an empty server.
//! A build with the route removed fails every test below with 404.

use axum::http::StatusCode;
use serde_json::Value;

use super::apr_model_routes_2609::apr_transformer_state;
use super::native_routes_2376::post;

#[tokio::test]
async fn route_answers_a_messages_request() {
    let (status, body) = post(
        apr_transformer_state(),
        "/v1/messages",
        r#"{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    let v: Value = serde_json::from_str(&body).expect("json body");
    assert_eq!(v["type"], "message", "body: {body}");
    assert_eq!(v["role"], "assistant", "body: {body}");
    assert!(v["id"].as_str().is_some_and(|id| id.starts_with("msg_")), "body: {body}");
    assert!(
        v["usage"]["output_tokens"].as_u64().is_some_and(|n| n > 0),
        "a loaded server generated nothing: {body}"
    );
    assert!(v["stop_reason"].is_string(), "body: {body}");
}

#[tokio::test]
async fn route_streams_the_messages_event_sequence() {
    let (status, body) = post(
        apr_transformer_state(),
        "/v1/messages",
        r#"{"model":"m","max_tokens":4,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(body.starts_with("event: message_start\n"), "body: {body}");
    assert!(body.contains("event: message_delta\n"), "body: {body}");
    assert!(
        body.trim_end().ends_with(r#"data: {"type":"message_stop"}"#),
        "body: {body}"
    );
}

#[tokio::test]
async fn malformed_request_gets_the_messages_error_envelope() {
    let (status, body) = post(
        apr_transformer_state(),
        "/v1/messages",
        r#"{"model":"m","messages":[]}"#,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    let v: Value = serde_json::from_str(&body).expect("json body");
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn unservable_block_is_refused_with_400_not_dropped() {
    let (status, body) = post(
        apr_transformer_state(),
        "/v1/messages",
        r#"{"model":"m","max_tokens":4,"messages":[{"role":"user","content":[
            {"type":"text","text":"what is in this picture?"},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]}]}"#,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    assert!(body.contains("image"), "the refusal names the block: {body}");
}
