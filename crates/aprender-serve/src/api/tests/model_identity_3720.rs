//! #3720: `apr serve` names the model that answered. Every response carries
//! `x-apr-model-digest` (the sha256 `sha256sum` prints for the served file), and a
//! completion body carries `model_digest`, `apr_version` and `apr_git_sha`. A server
//! that was given no identity claims none.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::*;

fn identity() -> ModelIdentity {
    ModelIdentity {
        digest: "ab".repeat(32),
        apr_version: "0.71.0".into(),
        apr_git_sha: "deadbeef".into(),
    }
}

/// The demo state: a tiny real model, so a chat completion really completes.
fn app(identity: Option<ModelIdentity>) -> axum::Router {
    static DEMO: std::sync::OnceLock<AppState> = std::sync::OnceLock::new();
    let state = DEMO.get_or_init(|| AppState::demo().expect("demo state"));
    create_router(state.clone().with_model_identity(identity))
}

async fn send(app: axum::Router, method: &str, uri: &str, body: Option<serde_json::Value>) -> axum::response::Response {
    let builder = Request::builder().method(method).uri(uri);
    let request = match body {
        Some(json) => builder
            .header("content-type", "application/json")
            .body(Body::from(json.to_string())),
        None => builder.body(Body::empty()),
    };
    app.oneshot(request.expect("request builds"))
        .await
        .expect("router responds")
}

async fn json(response: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body collects");
    serde_json::from_slice(&bytes).expect("a JSON body")
}

fn chat() -> serde_json::Value {
    serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 2
    })
}

#[tokio::test]
async fn every_response_carries_the_model_digest_header() {
    for (method, uri, body) in [
        ("GET", "/health", None),
        ("GET", "/v1/models", None),
        ("GET", "/no/such/route", None),
        ("POST", "/v1/chat/completions", Some(chat())),
    ] {
        let response = send(app(Some(identity())), method, uri, body).await;
        assert_eq!(
            response
                .headers()
                .get("x-apr-model-digest")
                .and_then(|v| v.to_str().ok()),
            Some("ab".repeat(32).as_str()),
            "{method} {uri} ({})",
            response.status()
        );
    }
    let bare = send(app(None), "GET", "/health", None).await;
    assert!(
        bare.headers().get("x-apr-model-digest").is_none(),
        "a server given no identity claims none"
    );
}

#[tokio::test]
async fn a_completion_body_names_what_produced_it() {
    let response = send(app(Some(identity())), "POST", "/v1/chat/completions", Some(chat())).await;
    assert_eq!(response.status(), StatusCode::OK);
    let v = json(response).await;
    assert_eq!(v["object"], "chat.completion", "{v}");
    assert_eq!(v["model_digest"], "ab".repeat(32));
    assert_eq!(v["apr_version"], "0.71.0");
    assert_eq!(v["apr_git_sha"], "deadbeef");
    assert!(v["choices"].is_array(), "the completion itself is untouched: {v}");

    let bare = json(send(app(None), "POST", "/v1/chat/completions", Some(chat())).await).await;
    assert!(bare.get("model_digest").is_none(), "no identity, no claim: {bare}");
}

#[test]
fn only_a_completion_object_is_stamped() {
    let id = identity();
    for object in ["chat.completion", "text_completion"] {
        let v = with_model_identity(serde_json::json!({"object": object, "id": "x"}), &id)
            .expect("a completion is stamped");
        assert_eq!(v["model_digest"], "ab".repeat(32));
        assert_eq!(v["id"], "x");
    }
    for other in [
        serde_json::json!({"object": "list", "data": []}),
        serde_json::json!({"object": "chat.completion.chunk"}),
        serde_json::json!({"status": "healthy"}),
        serde_json::json!(["chat.completion"]),
    ] {
        assert!(with_model_identity(other.clone(), &id).is_none(), "{other}");
    }
}
