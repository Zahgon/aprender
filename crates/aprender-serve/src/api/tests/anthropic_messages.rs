//! Falsifiers for `POST /v1/messages` (#4000 R-1).
//!
//! Translation tests pin each mapping in both directions. The route tests,
//! through the real router, are in `anthropic_messages_routes_4000.rs`.

use super::*;

fn parse(body: &str) -> MessagesRequest {
    serde_json::from_str(body).expect("valid Messages request")
}

fn chat_reply(
    content: &str,
    finish: &str,
    calls: Option<Vec<ResponseToolCall>>,
) -> ChatCompletionResponse {
    serde_json::from_value(json!({
        "id": "chatcmpl-abc", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }))
    .map(|mut r: ChatCompletionResponse| {
        r.choices[0].message.tool_calls = calls;
        r
    })
    .expect("chat reply fixture")
}

// ---------------------------------------------------------------------------
// Request translation
// ---------------------------------------------------------------------------

#[test]
fn system_string_and_text_turns_map_in_order() {
    let req = parse(
        r#"{"model":"m","max_tokens":16,"system":"be brief",
            "messages":[{"role":"user","content":"hi"},
                        {"role":"assistant","content":[{"type":"text","text":"he"},{"type":"text","text":"llo"}]},
                        {"role":"user","content":"2+2?"}],
            "stop_sequences":["\n\n"],"temperature":0.0,"top_k":1,
            "metadata":{"user_id":"u1"}}"#,
    );
    let chat = to_chat_request(&req).expect("translates");
    let roles: Vec<_> = chat.messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["system", "user", "assistant", "user"]);
    assert_eq!(chat.messages[0].content, "be brief");
    assert_eq!(chat.messages[2].content, "hello");
    assert_eq!(chat.max_tokens, Some(16));
    assert_eq!(chat.stop.as_deref(), Some(&["\n\n".to_string()][..]));
    assert_eq!(chat.top_k, Some(1));
    assert_eq!(chat.user.as_deref(), Some("u1"));
    assert!(!chat.stream, "generation is always driven non-streaming");
}

#[test]
fn system_as_text_blocks_is_concatenated() {
    let req = parse(
        r#"{"model":"m","max_tokens":4,"system":[{"type":"text","text":"a"},{"type":"text","text":"b"}],
            "messages":[{"role":"user","content":"x"}]}"#,
    );
    let chat = to_chat_request(&req).expect("translates");
    assert_eq!(chat.messages[0].content, "ab");
}

#[test]
fn tool_round_trip_maps_to_openai_tool_messages() {
    let req = parse(
        r#"{"model":"m","max_tokens":16,
            "tools":[{"name":"get_weather","description":"d","input_schema":{"type":"object"}}],
            "tool_choice":{"type":"tool","name":"get_weather"},
            "messages":[
              {"role":"user","content":"weather?"},
              {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]},
              {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"},
                                        {"type":"text","text":"thanks"}]}]}"#,
    );
    let chat = to_chat_request(&req).expect("translates");
    let tools = chat.tools.expect("tools mapped");
    assert_eq!(tools[0].function.name, "get_weather");
    assert_eq!(
        tools[0].function.parameters,
        Some(json!({"type": "object"}))
    );
    assert!(matches!(
        chat.tool_choice,
        Some(OpenAiToolChoice::Specific { ref function, .. }) if function.name == "get_weather"
    ));
    let roles: Vec<_> = chat.messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "user"]);
    let call = &chat.messages[1]
        .tool_calls
        .as_ref()
        .expect("assistant call")[0];
    assert_eq!(call.id, "toolu_1");
    assert_eq!(
        serde_json::from_str::<Value>(&call.function.arguments).expect("json args"),
        json!({"city": "Paris"})
    );
    assert_eq!(chat.messages[2].tool_call_id.as_deref(), Some("toolu_1"));
    assert_eq!(chat.messages[2].content, "sunny");
    assert_eq!(chat.messages[3].content, "thanks");
}

#[test]
fn tool_result_error_is_marked_not_dropped() {
    let req = parse(
        r#"{"model":"m","max_tokens":4,"messages":[{"role":"user","content":[
            {"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"boom"}],"is_error":true}]}]}"#,
    );
    let chat = to_chat_request(&req).expect("translates");
    assert_eq!(chat.messages[0].content, "Error: boom");
}

#[test]
fn tool_choice_modes_map() {
    for (anthropic, openai) in [("auto", "auto"), ("any", "required"), ("none", "none")] {
        let choice = ToolChoice {
            choice_type: anthropic.to_string(),
            name: None,
        };
        assert!(
            matches!(to_openai_tool_choice(&choice), Ok(OpenAiToolChoice::Mode(ref m)) if m == openai),
            "{anthropic} must map to {openai}"
        );
    }
    let nameless = ToolChoice {
        choice_type: "tool".to_string(),
        name: None,
    };
    assert!(to_openai_tool_choice(&nameless).is_err());
}

#[test]
fn thinking_toggle_maps_to_think() {
    let on = parse(
        r#"{"model":"m","max_tokens":4,"thinking":{"type":"enabled","budget_tokens":1024},"messages":[{"role":"user","content":"x"}]}"#,
    );
    let off = parse(
        r#"{"model":"m","max_tokens":4,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"x"}]}"#,
    );
    assert_eq!(to_chat_request(&on).expect("on").think, Some(true));
    assert_eq!(to_chat_request(&off).expect("off").think, Some(false));
}

/// What cannot be served faithfully is refused, never dropped.
#[test]
fn unservable_input_is_refused() {
    let image = r#"{"model":"m","max_tokens":4,"messages":[{"role":"user","content":[
        {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]}]}"#;
    assert!(to_chat_request(&parse(image)).is_err(), "image block");

    let server_tool = r#"{"model":"m","max_tokens":4,"tools":[{"type":"web_search_20250305","name":"web_search"}],
        "messages":[{"role":"user","content":"x"}]}"#;
    assert!(to_chat_request(&parse(server_tool)).is_err(), "server tool");

    let system_role =
        r#"{"model":"m","max_tokens":4,"messages":[{"role":"system","content":"x"}]}"#;
    assert!(
        to_chat_request(&parse(system_role)).is_err(),
        "system role in messages"
    );

    let user_tool_use = r#"{"model":"m","max_tokens":4,"messages":[{"role":"user","content":[
        {"type":"tool_use","id":"t","name":"f","input":{}}]}]}"#;
    assert!(
        to_chat_request(&parse(user_tool_use)).is_err(),
        "tool_use in a user turn"
    );

    let missing_max = r#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#;
    assert!(
        serde_json::from_str::<MessagesRequest>(missing_max).is_err(),
        "max_tokens is required"
    );
}

// ---------------------------------------------------------------------------
// Reply translation
// ---------------------------------------------------------------------------

#[test]
fn text_reply_maps_stop_reason_and_usage() {
    for (finish, stop) in [("stop", "end_turn"), ("length", "max_tokens")] {
        let r = from_chat_response(chat_reply("4", finish, None)).expect("maps");
        assert_eq!(r.stop_reason, stop, "finish_reason {finish}");
        assert_eq!(r.id, "msg_abc");
        assert_eq!(r.content, vec![ReplyBlock::Text { text: "4".into() }]);
        assert_eq!(
            r.usage,
            MessagesUsage {
                input_tokens: 7,
                output_tokens: 3
            }
        );
    }
}

#[test]
fn tool_call_reply_becomes_tool_use_block() {
    let call = tool_call("call_0", "get_weather", &json!({"city": "Paris"}));
    let r = from_chat_response(chat_reply("", "tool_calls", Some(vec![call]))).expect("maps");
    assert_eq!(r.stop_reason, "tool_use");
    assert_eq!(
        r.content,
        vec![ReplyBlock::ToolUse {
            id: "call_0".into(),
            name: "get_weather".into(),
            input: json!({"city": "Paris"}),
        }]
    );
}

#[test]
fn non_object_tool_arguments_are_refused() {
    let mut call = tool_call("c", "f", &json!({}));
    call.function.arguments = "[1,2]".into();
    assert!(from_chat_response(chat_reply("", "tool_calls", Some(vec![call.clone()]))).is_err());
    call.function.arguments = "not json".into();
    assert!(from_chat_response(chat_reply("", "tool_calls", Some(vec![call]))).is_err());
}

#[test]
fn reply_serializes_in_messages_shape() {
    let r = from_chat_response(chat_reply("hi", "stop", None)).expect("maps");
    let v = json!(r);
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["content"][0], json!({"type": "text", "text": "hi"}));
    assert_eq!(v["stop_sequence"], Value::Null);
}

#[test]
fn error_envelope_types_follow_status() {
    assert_eq!(error_type(StatusCode::BAD_REQUEST), "invalid_request_error");
    assert_eq!(error_type(StatusCode::NOT_FOUND), "not_found_error");
    assert_eq!(
        error_type(StatusCode::SERVICE_UNAVAILABLE),
        "overloaded_error"
    );
    assert_eq!(error_type(StatusCode::INTERNAL_SERVER_ERROR), "api_error");
    assert_eq!(
        upstream_error_message(br#"{"error":"no model"}"#),
        "no model"
    );
    assert_eq!(
        upstream_error_message(br#"{"error":{"message":"bad"}}"#),
        "bad"
    );
}

// ---------------------------------------------------------------------------
// SSE encoding
// ---------------------------------------------------------------------------

#[test]
fn sse_events_follow_the_messages_sequence_and_reassemble() {
    let call = tool_call("call_0", "f", &json!({"a": 1}));
    let r = from_chat_response(chat_reply("hi", "tool_calls", Some(vec![call]))).expect("maps");
    let events = sse_events(&r);
    let names: Vec<_> = events.iter().map(|(e, _)| *e).collect();
    assert_eq!(
        names,
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
    for (name, data) in &events {
        assert_eq!(data["type"], *name, "event name and data.type agree");
    }
    assert_eq!(events[0].1["message"]["content"], json!([]));
    assert_eq!(events[0].1["message"]["stop_reason"], Value::Null);
    assert_eq!(events[3].1["delta"]["text"], "hi");
    let partial = events[6].1["delta"]["partial_json"]
        .as_str()
        .expect("partial_json");
    assert_eq!(
        serde_json::from_str::<Value>(partial).expect("parses"),
        json!({"a": 1})
    );
    assert_eq!(events[8].1["delta"]["stop_reason"], "tool_use");
    assert_eq!(events[8].1["usage"]["output_tokens"], 3);
}

#[test]
fn sse_response_is_an_event_stream() {
    let r = from_chat_response(chat_reply("hi", "stop", None)).expect("maps");
    let resp = sse_response(&r);
    assert_eq!(
        resp.headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );
}
