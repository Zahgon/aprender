//! `AprMcpServer::handle_request_with_sink`: version check, notification silence,
//! delegation of non-`tools/call` methods and the sink staying quiet without a
//! `progressToken` (#5668).
#![allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()

use aprender_mcp::{AprMcpServer, JsonRpcRequest, NotificationSink};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn req(version: &str, method: &str, params: serde_json::Value) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: version.to_string(),
        id: Some(serde_json::json!(11)),
        method: method.to_string(),
        params,
    }
}

fn counting_sink() -> (NotificationSink, Arc<AtomicUsize>) {
    let n = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&n);
    let sink: NotificationSink = Box::new(move |_| {
        seen.fetch_add(1, Ordering::SeqCst);
    });
    (sink, n)
}

#[test]
fn wrong_jsonrpc_version_is_invalid_request_even_for_notifications() {
    let mut s = AprMcpServer::new();
    let (sink, _) = counting_sink();
    for method in ["initialize", "notifications/initialized", "tools/call"] {
        let r = s
            .handle_request_with_sink(&req("1.0", method, serde_json::json!({})), &sink)
            .expect("a response, never silence, for a bad version");
        let e = r.error.expect("error");
        assert_eq!(e.code, -32600, "{method}");
        assert!(e.message.contains("got \"1.0\""), "{}", e.message);
        assert_eq!(r.id, Some(serde_json::json!(11)));
    }
}

#[test]
fn notifications_get_no_response() {
    let mut s = AprMcpServer::new();
    let (sink, n) = counting_sink();
    for method in [
        "notifications/initialized",
        "notifications/cancelled",
        "notifications/x",
    ] {
        let r = s.handle_request_with_sink(&req("2.0", method, serde_json::json!({})), &sink);
        assert!(r.is_none(), "{method}");
    }
    assert_eq!(n.load(Ordering::SeqCst), 0);
}

#[test]
fn other_methods_delegate_to_handle_request() {
    let mut s = AprMcpServer::new();
    let (sink, _) = counting_sink();
    let init = s
        .handle_request_with_sink(&req("2.0", "initialize", serde_json::json!({})), &sink)
        .expect("response");
    assert!(init
        .result
        .expect("result")
        .get("protocolVersion")
        .is_some());
    let ping = s
        .handle_request_with_sink(&req("2.0", "ping", serde_json::json!({})), &sink)
        .expect("response");
    assert_eq!(ping.result, Some(serde_json::json!({})));
    let unknown = s
        .handle_request_with_sink(&req("2.0", "no/such", serde_json::json!({})), &sink)
        .expect("response");
    assert_eq!(unknown.error.expect("error").code, -32601);
}

#[test]
fn tools_call_without_a_token_answers_once_and_notifies_nobody() {
    let mut s = AprMcpServer::new();
    let (sink, n) = counting_sink();
    let params = serde_json::json!({"name": "no.such.tool", "arguments": {}});
    let r = s
        .handle_request_with_sink(&req("2.0", "tools/call", params), &sink)
        .expect("tools/call always answers");
    assert_eq!(r.id, Some(serde_json::json!(11)));
    assert!(
        r.error.is_none(),
        "tool failures ride in the result, not a protocol error"
    );
    let result = r.result.expect("result");
    assert_eq!(result["isError"], serde_json::json!(true), "{result}");
    assert_eq!(n.load(Ordering::SeqCst), 0);
}
