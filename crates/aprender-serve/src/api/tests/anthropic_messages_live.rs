//! Falsifiers for the live `/v1/messages` stream encoder (#4000 R-1 step 2).

use super::*;

fn chunk(delta: Value, finish: Option<&str>, usage: Option<(usize, usize)>) -> String {
    let mut c = json!({
        "id": "chatcmpl-abc", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    });
    if let Some((p, n)) = usage {
        c["usage"] = json!({"prompt_tokens": p, "completion_tokens": n, "total_tokens": p + n});
    }
    c.to_string()
}

fn names(events: &[SseEvent]) -> Vec<&'static str> {
    events.iter().map(|(e, _)| *e).collect()
}

fn run(frames: &[String]) -> Vec<SseEvent> {
    let mut enc = LiveMessages::default();
    let mut out: Vec<SseEvent> = frames.iter().flat_map(|f| enc.frame(f)).collect();
    out.extend(enc.end());
    out
}

#[test]
fn text_stream_is_re_encoded_token_by_token() {
    let events = run(&[
        chunk(json!({"role": "assistant"}), None, None),
        chunk(json!({"content": "he"}), None, None),
        chunk(json!({"content": "llo"}), None, None),
        chunk(json!({}), Some("length"), Some((7, 2))),
        "[DONE]".to_string(),
    ]);
    assert_eq!(
        names(&events),
        [
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    for (name, data) in &events {
        assert_eq!(data["type"], *name, "event name and data.type agree");
    }
    assert_eq!(events[0].1["message"]["id"], "msg_abc");
    let text: String = events
        .iter()
        .filter_map(|(_, d)| d["delta"]["text"].as_str())
        .collect();
    assert_eq!(
        text, "hello",
        "each token is its own delta and they reassemble"
    );
    assert_eq!(events[6].1["delta"]["stop_reason"], "max_tokens");
    assert_eq!(
        events[6].1["usage"],
        json!({"input_tokens": 7, "output_tokens": 2})
    );
}

#[test]
fn tool_call_closes_text_and_becomes_its_own_block() {
    let call = json!([{"index": 0, "id": "call_0", "type": "function",
                       "function": {"name": "f", "arguments": "{\"a\":1}"}}]);
    let events = run(&[
        chunk(json!({"role": "assistant", "content": "ok"}), None, None),
        chunk(json!({"tool_calls": call}), None, None),
        chunk(json!({}), Some("tool_calls"), Some((3, 4))),
        "[DONE]".to_string(),
    ]);
    assert_eq!(
        names(&events),
        [
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(events[5].1["index"], 1);
    assert_eq!(events[5].1["content_block"]["name"], "f");
    let partial = events[6].1["delta"]["partial_json"]
        .as_str()
        .expect("partial_json");
    assert_eq!(
        serde_json::from_str::<Value>(partial).expect("parses"),
        json!({"a": 1})
    );
    assert_eq!(events[8].1["delta"]["stop_reason"], "tool_use");
}

/// A cut-off generation is never presented as a finished one.
#[test]
fn stream_without_done_ends_in_error_not_message_stop() {
    let events = run(&[chunk(json!({"content": "par"}), None, None)]);
    let last = events.last().expect("events");
    assert_eq!(last.0, "error");
    assert_eq!(last.1["error"]["type"], "api_error");
    assert!(!names(&events).contains(&"message_stop"));
}

#[test]
fn upstream_error_frame_becomes_error_event_and_ends_stream() {
    let mut enc = LiveMessages::default();
    let first = enc.frame(&chunk(json!({"content": "a"}), None, None));
    assert!(!first.is_empty());
    let err = enc.frame(r#"{"error":"engine fault"}"#);
    assert_eq!(names(&err), ["error"]);
    assert_eq!(err[0].1["error"]["message"], "engine fault");
    assert!(enc.frame("[DONE]").is_empty(), "nothing after an error");
    assert!(enc.end().is_empty());
}

#[test]
fn bad_tool_arguments_are_an_error_not_a_dropped_block() {
    let call = json!([{"index": 0, "id": "c", "type": "function",
                       "function": {"name": "f", "arguments": "[1]"}}]);
    let events = run(&[
        chunk(json!({"tool_calls": call}), None, None),
        "[DONE]".to_string(),
    ]);
    assert_eq!(events.last().expect("events").0, "error");
    assert!(!names(&events).contains(&"message_stop"));
}

#[test]
fn frame_data_reads_data_lines_and_skips_comments() {
    assert_eq!(
        frame_data("data: {\"a\":1}\n\n").as_deref(),
        Some("{\"a\":1}")
    );
    assert_eq!(frame_data("data:x\ndata: y\n\n").as_deref(), Some("x\ny"));
    assert_eq!(frame_data(": keep-alive\n\n"), None);
    assert_eq!(
        frame_data("event: e\ndata: [DONE]\n\n").as_deref(),
        Some("[DONE]")
    );
}

/// Body chunks split frames, and a multi-byte character, at arbitrary bytes.
#[tokio::test]
async fn live_response_reassembles_frames_split_across_body_chunks() {
    let first = chunk(
        json!({"role": "assistant", "content": "é"}),
        Some("stop"),
        Some((1, 1)),
    );
    let wire = format!(": keep-alive\n\ndata: {first}\n\ndata: [DONE]\n\n").into_bytes();
    let parts: Vec<Result<Vec<u8>, std::io::Error>> =
        wire.chunks(5).map(|c| Ok(c.to_vec())).collect();
    let inner = axum::body::Body::from_stream(futures::stream::iter(parts)).into_response();
    let resp = live_response(inner);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body");
    let text = String::from_utf8(body.to_vec()).expect("utf8");
    assert!(text.contains(r#""text":"é""#), "{text}");
    assert!(text.contains("event: message_stop"), "{text}");
    assert!(!text.contains("event: error"), "{text}");
}
