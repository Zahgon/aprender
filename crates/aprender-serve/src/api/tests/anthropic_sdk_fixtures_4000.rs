//! The Anthropic SDK's own `messages.create` requests, replayed against
//! `POST /v1/messages` (#4000 R-1 step 3).
//!
//! `fixtures/anthropic_sdk_messages.json` holds the request each SDK test
//! sends, with the outcome this server owes it. The minimal requests are
//! answered. The all-params requests ask for platform features (a code
//! container, tool search, strict tool input, one call per reply), and each
//! of those is refused BY NAME in one 400, never dropped. A build with the
//! route removed fails every case here with 404.

use axum::http::StatusCode;
use serde_json::Value;

use super::apr_model_routes_2609::apr_transformer_state;
use super::native_routes_2376::post;

const FIXTURES: &str = include_str!("fixtures/anthropic_sdk_messages.json");

fn cases() -> Vec<Value> {
    let v: Value = serde_json::from_str(FIXTURES).expect("fixture json");
    v["cases"].as_array().expect("cases").clone()
}

async fn replay(request: &Value) -> (StatusCode, String) {
    post(apr_transformer_state(), "/v1/messages", &request.to_string()).await
}

fn assert_message(name: &str, body: &str) {
    let v: Value = serde_json::from_str(body).unwrap_or_else(|e| panic!("{name}: {e}: {body}"));
    assert_eq!(v["type"], "message", "{name}: {body}");
    assert_eq!(v["role"], "assistant", "{name}: {body}");
    assert!(
        v["usage"]["output_tokens"].as_u64().is_some_and(|n| n > 0),
        "{name}: a loaded server generated nothing: {body}"
    );
}

#[tokio::test]
async fn sdk_requests_get_the_outcome_the_fixture_names() {
    let cases = cases();
    assert_eq!(cases.len(), 4, "the SDK's four messages.create requests");
    for case in &cases {
        let name = case["name"].as_str().expect("name");
        let (status, body) = replay(&case["request"]).await;
        match case["expect"].as_str().expect("expect") {
            "message" => {
                assert_eq!(status, StatusCode::OK, "{name}: {body}");
                assert_message(name, &body);
            },
            "stream" => {
                assert_eq!(status, StatusCode::OK, "{name}: {body}");
                assert!(body.starts_with("event: message_start\n"), "{name}: {body}");
                assert!(
                    body.trim_end().ends_with(r#"data: {"type":"message_stop"}"#),
                    "{name}: {body}"
                );
            },
            "refused" => {
                assert_eq!(status, StatusCode::BAD_REQUEST, "{name}: {body}");
                let v: Value = serde_json::from_str(&body).expect("json error");
                assert_eq!(v["error"]["type"], "invalid_request_error", "{name}");
                let msg = v["error"]["message"].as_str().expect("message");
                for field in case["refused"].as_array().expect("refused") {
                    let field = field.as_str().expect("field name");
                    assert!(msg.contains(field), "{name}: {field} not named: {msg}");
                }
            },
            other => panic!("{name}: unknown expectation {other:?}"),
        }
    }
}

/// The refusal is for the named fields only. Without them (and without
/// `output_config`, whose schema the chat path will not combine with tools),
/// the same SDK request is served: `cache_control`, citations, `service_tier`,
/// routing ids, adaptive thinking (on a model with no thinking mode) and a
/// custom tool are all accepted.
#[tokio::test]
async fn all_params_request_without_the_refused_fields_is_served() {
    let mut request = cases()[2]["request"].clone();
    let obj = request.as_object_mut().expect("request object");
    obj.remove("container");
    obj.remove("output_config");
    let tool = request["tools"][0].as_object_mut().expect("tool");
    tool.remove("strict");
    tool.remove("defer_loading");
    let choice = request["tool_choice"].as_object_mut().expect("tool_choice");
    choice.remove("disable_parallel_tool_use");
    let (status, body) = replay(&request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_message("all params, servable", &body);
}
