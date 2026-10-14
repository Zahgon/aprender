//! E-2 (#5348): one OpenAI tool-calling task, end to end, judged.
//!
//! The task needs a fact the model cannot know: the user asks for the code of
//! a vault, and only the [`TOOL`] knows it ([`SECRET`]). A pass needs both
//! turns of a real OpenAI client:
//!
//! 1. the model answers with a `tool_calls` entry for [`TOOL`] whose
//!    arguments name [`VAULT`] ([`judge_tool_call`]);
//! 2. after the client echoes that assistant message and a `tool` message
//!    carrying the result, the model's text contains [`SECRET`]
//!    ([`judge_final`]).
//!
//! The pure parts live here so the case table can prove each judge fails on
//! the wrong shape; the HTTP driver is the `serve_agent_task` example.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// The one tool the task offers.
pub const TOOL: &str = "get_vault_code";
/// The vault the user asks about; the tool call's `vault` argument.
pub const VAULT: &str = "orchid";
/// What the tool returns. Not guessable, so it can only reach the final
/// answer through the tool result.
pub const SECRET: &str = "7319-QX";

/// The user turn that starts the task.
pub const PROMPT: &str = "What is the access code of the vault named orchid? \
    Use the get_vault_code tool to look it up, then reply with the code.";

/// The `tools` array sent on both turns.
#[must_use]
pub fn tools() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": TOOL,
            "description": "Look up the access code of a vault by its name.",
            "parameters": {
                "type": "object",
                "properties": {"vault": {"type": "string", "description": "The vault name"}},
                "required": ["vault"]
            }
        }
    }])
}

/// Turn 1: the user prompt with [`tools`].
#[must_use]
pub fn first_request(model: &str, max_tokens: u32) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": tools(),
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": false
    })
}

/// The tool call turn 1 produced, as the client needs it for turn 2.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// The call id the `tool` message must answer.
    pub id: String,
    /// The parsed `arguments` object.
    pub arguments: Value,
    /// The assistant message exactly as the server sent it, echoed in turn 2.
    pub assistant_message: Value,
}

/// Turn 1 must be a call of [`TOOL`] with `vault` = [`VAULT`].
///
/// # Errors
/// A text answer, another tool, arguments that are not a JSON object, a
/// missing call id, or a `finish_reason` other than `tool_calls`.
pub fn judge_tool_call(response: &Value) -> Result<ToolCall, String> {
    let choice = response
        .pointer("/choices/0")
        .ok_or("no choices[0] in the response")?;
    let message = choice.get("message").ok_or("no choices[0].message")?;
    let call = message
        .pointer("/tool_calls/0")
        .ok_or_else(|| format!("no tool call; the model said {:?}", message.get("content")))?;
    let name = call.pointer("/function/name").and_then(Value::as_str);
    if name != Some(TOOL) {
        return Err(format!("called {name:?}, want {TOOL}"));
    }
    let raw = call
        .pointer("/function/arguments")
        .and_then(Value::as_str)
        .ok_or("function.arguments is not a string")?;
    let arguments: Value =
        serde_json::from_str(raw).map_err(|e| format!("arguments {raw:?} are not JSON: {e}"))?;
    let vault = arguments.get("vault").and_then(Value::as_str);
    if !vault.is_some_and(|v| v.trim().eq_ignore_ascii_case(VAULT)) {
        return Err(format!("vault argument {vault:?}, want {VAULT:?}"));
    }
    let id = call
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or("tool call has no id")?;
    let finish = choice.get("finish_reason").and_then(Value::as_str);
    if finish != Some("tool_calls") {
        return Err(format!("finish_reason {finish:?}, want tool_calls"));
    }
    Ok(ToolCall {
        id: id.to_string(),
        arguments,
        assistant_message: message.clone(),
    })
}

/// Turn 2: turn 1's messages, the echoed assistant call and the tool result.
#[must_use]
pub fn second_request(model: &str, max_tokens: u32, call: &ToolCall) -> Value {
    json!({
        "model": model,
        "messages": [
            {"role": "user", "content": PROMPT},
            call.assistant_message,
            {"role": "tool", "tool_call_id": call.id, "content": SECRET}
        ],
        "tools": tools(),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": false
    })
}

/// Turn 2 must be a text answer containing [`SECRET`].
///
/// # Errors
/// Another tool call, a missing or empty text, or text without the code.
pub fn judge_final(response: &Value) -> Result<String, String> {
    let message = response
        .pointer("/choices/0/message")
        .ok_or("no choices[0].message in the response")?;
    if message
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|c| !c.is_empty())
    {
        return Err("called a tool again instead of answering".to_string());
    }
    let text = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !text.contains(SECRET) {
        return Err(format!("answer {text:?} does not contain {SECRET}"));
    }
    Ok(text.to_string())
}

/// sha256 of the exact transcript bytes the driver writes to disk.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
#[path = "agent_task_tests.rs"]
mod tests;
