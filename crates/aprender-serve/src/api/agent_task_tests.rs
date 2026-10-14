use super::*;

fn call_response(name: &str, args: &str, id: &str, finish: &str) -> Value {
    json!({"choices": [{
        "message": {"role": "assistant", "content": "", "tool_calls": [{
            "id": id, "type": "function",
            "function": {"name": name, "arguments": args}
        }]},
        "finish_reason": finish
    }]})
}

fn text_response(text: &str) -> Value {
    json!({"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})
}

/// E-2: each judge passes the right shape and fails every wrong one.
#[test]
fn judge_tool_call_case_table() {
    let ok_args = r#"{"vault":"orchid"}"#;
    let rows: Vec<(&str, Value, bool)> = vec![
        (
            "call ok",
            call_response(TOOL, ok_args, "call_0", "tool_calls"),
            true,
        ),
        (
            "vault case and space",
            call_response(TOOL, r#"{"vault":" Orchid "}"#, "c", "tool_calls"),
            true,
        ),
        ("text only", text_response("The code is 1234."), false),
        (
            "other tool",
            call_response("get_weather", ok_args, "call_0", "tool_calls"),
            false,
        ),
        (
            "wrong vault",
            call_response(TOOL, r#"{"vault":"lotus"}"#, "call_0", "tool_calls"),
            false,
        ),
        (
            "no vault",
            call_response(TOOL, "{}", "call_0", "tool_calls"),
            false,
        ),
        (
            "args not json",
            call_response(TOOL, "vault=orchid", "call_0", "tool_calls"),
            false,
        ),
        (
            "no id",
            call_response(TOOL, ok_args, "", "tool_calls"),
            false,
        ),
        (
            "said stop",
            call_response(TOOL, ok_args, "call_0", "stop"),
            false,
        ),
        ("no choices", json!({"error": "x"}), false),
    ];
    for (label, response, want) in rows {
        assert_eq!(
            judge_tool_call(&response).is_ok(),
            want,
            "{label}: {:?}",
            judge_tool_call(&response)
        );
    }
}

#[test]
fn judge_final_case_table() {
    let again = call_response(TOOL, r#"{"vault":"orchid"}"#, "call_1", "tool_calls");
    let rows: Vec<(&str, Value, bool)> = vec![
        (
            "answer ok",
            text_response("The access code is 7319-QX."),
            true,
        ),
        (
            "wrong code",
            text_response("The access code is 7319."),
            false,
        ),
        ("empty", text_response(""), false),
        ("called again", again, false),
        ("no message", json!({"choices": []}), false),
    ];
    for (label, response, want) in rows {
        assert_eq!(
            judge_final(&response).is_ok(),
            want,
            "{label}: {:?}",
            judge_final(&response)
        );
    }
}

/// Turn 2 echoes the assistant message verbatim and answers its call id.
#[test]
fn second_request_echoes_the_call() {
    let first = call_response(TOOL, r#"{"vault":"orchid"}"#, "call_7", "tool_calls");
    let call = judge_tool_call(&first).expect("valid call");
    let req = second_request("m", 64, &call);
    assert_eq!(req["messages"][1], first["choices"][0]["message"]);
    assert_eq!(req["messages"][2]["role"], "tool");
    assert_eq!(req["messages"][2]["tool_call_id"], "call_7");
    assert_eq!(req["messages"][2]["content"], SECRET);
    assert_eq!(first_request("m", 64)["tools"], req["tools"]);
}

/// The secret is not in anything the model sees before the tool result.
#[test]
fn secret_reaches_the_model_only_through_the_tool() {
    let turn1 = first_request("m", 64).to_string();
    assert!(!turn1.contains(SECRET), "turn 1 leaks the secret");
}

#[test]
fn sha256_hex_known_vector() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
