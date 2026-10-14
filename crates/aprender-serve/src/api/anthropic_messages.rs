//! Anthropic Messages API adapter: `POST /v1/messages` (#4000 R-1).
//!
//! Lets a client built on the Anthropic SDK (or anything that speaks the
//! Messages wire format) talk to `apr serve` unchanged. Like the Ollama routes,
//! the handler does no generation of its own: it translates the request into a
//! [`ChatCompletionRequest`], delegates to [`openai_chat_completions_handler`],
//! and re-shapes the reply. One generation path serves all three protocols.
//!
//! Wire shape (non-streaming reply):
//! ```json
//! {"id":"msg_…","type":"message","role":"assistant","model":"…",
//!  "content":[{"type":"text","text":"…"},{"type":"tool_use","id":"…","name":"…","input":{}}],
//!  "stop_reason":"end_turn","stop_sequence":null,
//!  "usage":{"input_tokens":N,"output_tokens":M}}
//! ```
//!
//! What the adapter cannot serve it REFUSES with a 400 `invalid_request_error`
//! rather than dropping: image and document blocks, server tools (a tool
//! entry with a `type` other than `custom`), the platform's `container` and
//! `mcp_servers`, and the guarantees `tools[].strict`, `tools[].defer_loading`
//! and `tool_choice.disable_parallel_tool_use`. A silently dropped image would
//! let the model answer a question about a picture it never saw. Hints that
//! leave the reply's content unchanged (`cache_control`, `service_tier`,
//! routing ids) are ignored. `tests/fixtures/anthropic_sdk_messages.json`
//! replays the SDK's own requests against this rule.
//!
//! `stream: true` streams live: the chat path's SSE chunks are re-encoded as
//! the Messages event sequence (`message_start` … `message_stop`) while
//! generation runs (see `anthropic_messages_live.rs`). If the chat path
//! answers a streamed request with a whole JSON reply, that reply is encoded
//! as the same sequence at once.
#![allow(unreachable_pub)] // re-exported as pub(crate) from api/mod.rs

use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{
    openai_chat_completions_handler, AppState, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, ChoiceCount, JsonSchemaFormat, OpenAiFunctionDef, OpenAiTool, OpenAiToolChoice,
    OpenAiToolChoiceFunction, ResponseFormat, ResponseFunctionCall, ResponseToolCall,
};

// ============================================================================
// Request wire types
// ============================================================================

/// `POST /v1/messages` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    /// Model id. Echoed back; the server answers with the model it loaded.
    pub model: String,
    /// Required by the Messages API, so required here too.
    pub max_tokens: usize,
    /// Conversation turns, `user` and `assistant` only.
    pub messages: Vec<InputMessage>,
    /// System prompt: a string or a list of text blocks.
    #[serde(default)]
    pub system: Option<Content>,
    /// Custom stop sequences.
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k sampling.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// SSE reply when true.
    #[serde(default)]
    pub stream: bool,
    /// Client tools the model may call.
    #[serde(default)]
    pub tools: Option<Vec<ToolDef>>,
    /// `auto` | `any` | `tool` (with `name`) | `none`.
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    /// Extended-thinking toggle; mapped onto the chat template's `think`.
    #[serde(default)]
    pub thinking: Option<Thinking>,
    /// `metadata.user_id` is passed through as the OpenAI `user`.
    #[serde(default)]
    pub metadata: Option<Metadata>,
    /// `output_config.format` is served as the chat path's `response_format`.
    #[serde(default)]
    pub output_config: Option<OutputConfig>,
    /// Every other top-level field. One that only the Anthropic platform can
    /// serve ([`PLATFORM_FIELDS`]) is refused. The rest (`cache_control`,
    /// `service_tier`, routing ids) are hints that leave the reply's content
    /// unchanged, so they are ignored.
    #[serde(flatten)]
    pub other: serde_json::Map<String, Value>,
}

/// Top-level fields served only by the Anthropic platform: a code-execution
/// container with its skills, and remote MCP servers it connects to.
pub const PLATFORM_FIELDS: [&str; 2] = ["container", "mcp_servers"];

/// `output_config`. Its `effort` is a hint and is ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct OutputConfig {
    /// `{"type":"json_schema","schema":{…}}`.
    #[serde(default)]
    pub format: Option<Value>,
}

/// One conversation turn.
#[derive(Debug, Clone, Deserialize)]
pub struct InputMessage {
    /// `user` or `assistant`.
    pub role: String,
    /// A string or a list of content blocks.
    pub content: Content,
}

/// Message content: a bare string or a list of blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// Shorthand for one text block.
    Text(String),
    /// Explicit content blocks.
    Blocks(Vec<ContentBlock>),
}

/// A request content block. Unknown types fail deserialization (400).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text.
    Text {
        /// The text.
        text: String,
    },
    /// An earlier assistant tool call, echoed back by the client.
    ToolUse {
        /// Tool-call id.
        id: String,
        /// Tool name.
        name: String,
        /// Arguments object.
        input: Value,
    },
    /// The result of a tool call.
    ToolResult {
        /// Id of the `tool_use` this answers.
        tool_use_id: String,
        /// A string or text blocks; absent means empty.
        #[serde(default)]
        content: Option<Content>,
        /// The tool failed.
        #[serde(default)]
        is_error: bool,
    },
    /// Refused: no vision path behind this route.
    Image {},
    /// Refused: no document path behind this route.
    Document {},
}

/// A client tool definition.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolDef {
    /// Absent or `custom` for a client tool; anything else is a server tool.
    #[serde(rename = "type", default)]
    pub tool_type: Option<String>,
    /// Tool name.
    pub name: String,
    /// Description.
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema of the input.
    #[serde(default)]
    pub input_schema: Option<Value>,
    /// Guarantee that `input` matches the schema. Refused when true: tool
    /// arguments are not generated under a constraint.
    #[serde(default)]
    pub strict: bool,
    /// Load the tool through tool search. Refused when true: no tool search.
    #[serde(default)]
    pub defer_loading: bool,
}

/// `tool_choice`.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoice {
    /// `auto` | `any` | `tool` | `none`.
    #[serde(rename = "type")]
    pub choice_type: String,
    /// The tool name when `type` is `tool`.
    #[serde(default)]
    pub name: Option<String>,
    /// At most one tool call per reply. Refused when true: the chat path has
    /// no such limit, and cutting extra calls would drop model output.
    #[serde(default)]
    pub disable_parallel_tool_use: bool,
}

/// `thinking`.
#[derive(Debug, Clone, Deserialize)]
pub struct Thinking {
    /// `enabled` | `adaptive` | `disabled`.
    #[serde(rename = "type")]
    pub thinking_type: String,
}

/// `metadata`.
#[derive(Debug, Clone, Deserialize)]
pub struct Metadata {
    /// Opaque end-user id.
    #[serde(default)]
    pub user_id: Option<String>,
}

// ============================================================================
// Reply wire types
// ============================================================================

/// A reply content block.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReplyBlock {
    /// Generated text.
    Text {
        /// The text.
        text: String,
    },
    /// A tool call the client should run.
    ToolUse {
        /// Tool-call id; the client echoes it in `tool_result.tool_use_id`.
        id: String,
        /// Tool name.
        name: String,
        /// Arguments object.
        input: Value,
    },
}

/// Token usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MessagesUsage {
    /// Prompt tokens.
    pub input_tokens: usize,
    /// Generated tokens.
    pub output_tokens: usize,
}

/// `POST /v1/messages` reply.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MessagesResponse {
    /// `msg_…`.
    pub id: String,
    /// Always `message`.
    #[serde(rename = "type")]
    pub object_type: &'static str,
    /// Always `assistant`.
    pub role: &'static str,
    /// Model label.
    pub model: String,
    /// Text first, then tool calls.
    pub content: Vec<ReplyBlock>,
    /// `end_turn` | `max_tokens` | `tool_use`.
    pub stop_reason: &'static str,
    /// Always null: the chat path does not report which stop sequence fired.
    pub stop_sequence: Option<String>,
    /// Token usage.
    pub usage: MessagesUsage,
}

// ============================================================================
// Errors
// ============================================================================

/// An Anthropic error envelope: `{"type":"error","error":{"type":…,"message":…}}`.
fn error_response(status: StatusCode, message: &str) -> Response {
    let body = json!({
        "type": "error",
        "error": { "type": error_type(status), "message": message },
    });
    (status, Json(body)).into_response()
}

/// The Messages API error `type` for an HTTP status.
fn error_type(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        _ => "api_error",
    }
}

/// The message of an upstream error body: `{"error":"…"}` or `{"error":{"message":"…"}}`.
fn upstream_error_message(body: &[u8]) -> String {
    let parsed = serde_json::from_slice::<Value>(body).ok();
    let err = parsed.as_ref().and_then(|v| v.get("error"));
    err.and_then(|e| {
        e.as_str()
            .or_else(|| e.get("message").and_then(Value::as_str))
    })
    .map_or_else(|| "generation unavailable".to_string(), str::to_string)
}

// ============================================================================
// Request translation
// ============================================================================

/// Concatenate the text of a [`Content`]; a non-text block is an error.
fn content_text(content: &Content) -> Result<String, String> {
    match content {
        Content::Text(s) => Ok(s.clone()),
        Content::Blocks(blocks) => blocks
            .iter()
            .map(|b| match b {
                ContentBlock::Text { text } => Ok(text.as_str()),
                other => Err(format!(
                    "only text blocks are allowed here, got {}",
                    block_name(other)
                )),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|parts| parts.concat()),
    }
}

fn block_name(block: &ContentBlock) -> &'static str {
    match block {
        ContentBlock::Text { .. } => "text",
        ContentBlock::ToolUse { .. } => "tool_use",
        ContentBlock::ToolResult { .. } => "tool_result",
        ContentBlock::Image {} => "image",
        ContentBlock::Document {} => "document",
    }
}

/// One Messages turn as OpenAI chat messages.
///
/// A `tool_result` block becomes its own `role:"tool"` message, emitted before
/// the turn's text: the Messages API requires tool results to lead the user
/// turn that carries them, and the chat template expects each result right
/// after the assistant call it answers.
fn turn_to_chat(msg: &InputMessage) -> Result<Vec<ChatMessage>, String> {
    if msg.role != "user" && msg.role != "assistant" {
        return Err(format!(
            "messages: role must be \"user\" or \"assistant\", got {:?}",
            msg.role
        ));
    }
    let blocks = match &msg.content {
        Content::Text(s) => return Ok(vec![chat(&msg.role, s.clone())]),
        Content::Blocks(b) => b,
    };
    let mut out = Vec::new();
    let mut text = String::new();
    let mut calls = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text: t } => text.push_str(t),
            ContentBlock::ToolUse { id, name, input } if msg.role == "assistant" => {
                calls.push(tool_call(id, name, input));
            },
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } if msg.role == "user" => out.push(tool_result(tool_use_id, content, *is_error)?),
            other => {
                return Err(format!(
                    "content block {:?} is not supported in a {} turn",
                    block_name(other),
                    msg.role
                ))
            },
        }
    }
    if !text.is_empty() || !calls.is_empty() {
        let mut m = chat(&msg.role, text);
        m.tool_calls = (!calls.is_empty()).then_some(calls);
        out.push(m);
    }
    Ok(out)
}

fn chat(role: &str, content: String) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content,
        ..Default::default()
    }
}

fn tool_call(id: &str, name: &str, input: &Value) -> ResponseToolCall {
    ResponseToolCall {
        id: id.to_string(),
        call_type: "function".to_string(),
        function: ResponseFunctionCall {
            name: name.to_string(),
            arguments: input.to_string(),
        },
    }
}

fn tool_result(id: &str, content: &Option<Content>, is_error: bool) -> Result<ChatMessage, String> {
    let body = content.as_ref().map(content_text).transpose()?;
    let body = body.unwrap_or_default();
    let mut m = chat(
        "tool",
        if is_error {
            format!("Error: {body}")
        } else {
            body
        },
    );
    m.tool_call_id = Some(id.to_string());
    Ok(m)
}

fn to_openai_tools(tools: &[ToolDef]) -> Result<Vec<OpenAiTool>, String> {
    tools
        .iter()
        .map(|t| match t.tool_type.as_deref() {
            None | Some("custom") => Ok(OpenAiTool {
                tool_type: "function".to_string(),
                function: OpenAiFunctionDef {
                    name: t.name.clone(),
                    description: t.description.clone().unwrap_or_default(),
                    parameters: t.input_schema.clone(),
                },
            }),
            Some(other) => Err(format!(
                "tools: server tool type {other:?} is not supported by this server"
            )),
        })
        .collect()
}

fn to_openai_tool_choice(choice: &ToolChoice) -> Result<OpenAiToolChoice, String> {
    let mode = |m: &str| Ok(OpenAiToolChoice::Mode(m.to_string()));
    match (choice.choice_type.as_str(), &choice.name) {
        ("auto", _) => mode("auto"),
        ("any", _) => mode("required"),
        ("none", _) => mode("none"),
        ("tool", Some(name)) => Ok(OpenAiToolChoice::Specific {
            choice_type: "function".to_string(),
            function: OpenAiToolChoiceFunction { name: name.clone() },
        }),
        ("tool", None) => Err("tool_choice: type \"tool\" requires a name".to_string()),
        (other, _) => Err(format!("tool_choice: unknown type {other:?}")),
    }
}

/// Every field of `req` that asks for something this server cannot do, by
/// its path in the request. All of them are named in one refusal.
fn unservable(req: &MessagesRequest) -> Vec<String> {
    let mut out: Vec<String> = PLATFORM_FIELDS
        .iter()
        .filter(|f| req.other.contains_key(**f))
        .map(|f| (*f).to_string())
        .collect();
    for (i, tool) in req.tools.iter().flatten().enumerate() {
        if tool.strict {
            out.push(format!("tools[{i}].strict"));
        }
        if tool.defer_loading {
            out.push(format!("tools[{i}].defer_loading"));
        }
    }
    if req
        .tool_choice
        .as_ref()
        .is_some_and(|c| c.disable_parallel_tool_use)
    {
        out.push("tool_choice.disable_parallel_tool_use".to_string());
    }
    out
}

/// `thinking.type` as the chat template's `think`. `adaptive` asks the model
/// to reason only when it helps, so it sets nothing and the server's default
/// stands. Forcing thinking on would refuse every model whose template has no
/// thinking mode (#3723).
fn to_think(thinking: &Thinking) -> Result<Option<bool>, String> {
    match thinking.thinking_type.as_str() {
        "enabled" => Ok(Some(true)),
        "disabled" => Ok(Some(false)),
        "adaptive" => Ok(None),
        other => Err(format!("thinking: unknown type {other:?}")),
    }
}

/// `output_config.format` as the chat path's `response_format`.
fn to_response_format(format: &Value) -> Result<ResponseFormat, String> {
    match format.get("type").and_then(Value::as_str) {
        Some("json_schema") => Ok(ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat {
                name: None,
                description: None,
                schema: format.get("schema").cloned(),
                strict: None,
            },
        }),
        other => Err(format!("output_config.format: unknown type {other:?}")),
    }
}

/// Translate a Messages request into the internal chat request.
///
/// # Errors
/// A refusal message for anything the adapter cannot serve faithfully.
pub fn to_chat_request(req: &MessagesRequest) -> Result<ChatCompletionRequest, String> {
    let refused = unservable(req);
    if !refused.is_empty() {
        return Err(format!(
            "not supported by this server: {}",
            refused.join(", ")
        ));
    }
    let mut messages = Vec::with_capacity(req.messages.len() + 1);
    if let Some(system) = &req.system {
        messages.push(chat("system", content_text(system)?));
    }
    for m in &req.messages {
        messages.extend(turn_to_chat(m)?);
    }
    Ok(ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        max_tokens: Some(req.max_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        n: ChoiceCount::ONE,
        stream: req.stream,
        stop: req.stop_sequences.clone(),
        user: req.metadata.as_ref().and_then(|m| m.user_id.clone()),
        tools: req.tools.as_deref().map(to_openai_tools).transpose()?,
        tool_choice: req
            .tool_choice
            .as_ref()
            .map(to_openai_tool_choice)
            .transpose()?,
        think: req.thinking.as_ref().map(to_think).transpose()?.flatten(),
        response_format: req
            .output_config
            .as_ref()
            .and_then(|c| c.format.as_ref())
            .map(to_response_format)
            .transpose()?,
        ..Default::default()
    })
}

// ============================================================================
// Reply translation
// ============================================================================

fn stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    }
}

/// Re-shape an OpenAI chat reply into a Messages reply.
///
/// # Errors
/// A tool call whose arguments are not a JSON object: the Messages wire format
/// requires `input` to be one, and inventing a wrapper would hand the client
/// arguments the model never produced.
pub fn from_chat_response(resp: ChatCompletionResponse) -> Result<MessagesResponse, String> {
    let choice = resp.choices.into_iter().next();
    let finish = choice
        .as_ref()
        .map_or("stop", |c| c.finish_reason.as_str())
        .to_string();
    let mut content = Vec::new();
    if let Some(msg) = choice.map(|c| c.message) {
        if !msg.content.is_empty() {
            content.push(ReplyBlock::Text { text: msg.content });
        }
        for call in msg.tool_calls.unwrap_or_default() {
            content.push(tool_use_block(call)?);
        }
    }
    let has_tool_use = content
        .iter()
        .any(|b| matches!(b, ReplyBlock::ToolUse { .. }));
    Ok(MessagesResponse {
        id: format!("msg_{}", resp.id.trim_start_matches("chatcmpl-")),
        object_type: "message",
        role: "assistant",
        model: resp.model,
        content,
        stop_reason: if has_tool_use {
            "tool_use"
        } else {
            stop_reason(&finish)
        },
        stop_sequence: None,
        usage: MessagesUsage {
            input_tokens: resp.usage.prompt_tokens,
            output_tokens: resp.usage.completion_tokens,
        },
    })
}

fn tool_use_block(call: ResponseToolCall) -> Result<ReplyBlock, String> {
    let input: Value = serde_json::from_str(&call.function.arguments).map_err(|e| {
        format!(
            "tool call {:?} carried arguments that are not JSON: {e}",
            call.function.name
        )
    })?;
    if !input.is_object() {
        return Err(format!(
            "tool call {:?} carried arguments that are not a JSON object",
            call.function.name
        ));
    }
    Ok(ReplyBlock::ToolUse {
        id: call.id,
        name: call.function.name,
        input,
    })
}

// ============================================================================
// SSE encoding
// ============================================================================

/// The Messages SSE events for a completed reply, as `(event, data)` pairs.
///
/// Invariant (asserted by the falsifiers): concatenating every `text_delta`
/// of block `i` gives that block's text, and every `input_json_delta` of a
/// `tool_use` block parses back to its `input`.
pub fn sse_events(resp: &MessagesResponse) -> Vec<(&'static str, Value)> {
    let mut start = json!(resp);
    start["content"] = json!([]);
    start["stop_reason"] = Value::Null;
    start["usage"]["output_tokens"] = json!(0);
    let mut events = vec![
        (
            "message_start",
            json!({"type": "message_start", "message": start}),
        ),
        ("ping", json!({"type": "ping"})),
    ];
    for (index, block) in resp.content.iter().enumerate() {
        events.extend(block_events(index, block));
    }
    events.push((
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": resp.stop_reason, "stop_sequence": resp.stop_sequence},
            "usage": {"output_tokens": resp.usage.output_tokens},
        }),
    ));
    events.push(("message_stop", json!({"type": "message_stop"})));
    events
}

fn block_events(index: usize, block: &ReplyBlock) -> Vec<(&'static str, Value)> {
    let (start, delta) = match block {
        ReplyBlock::Text { text } => (
            json!({"type": "text", "text": ""}),
            json!({"type": "text_delta", "text": text}),
        ),
        ReplyBlock::ToolUse { id, name, input } => (
            json!({"type": "tool_use", "id": id, "name": name, "input": {}}),
            json!({"type": "input_json_delta", "partial_json": input.to_string()}),
        ),
    };
    vec![
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": index, "content_block": start}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": index, "delta": delta}),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        ),
    ]
}

fn sse_response(resp: &MessagesResponse) -> Response {
    let body: String = sse_events(resp)
        .into_iter()
        .map(|(event, data)| format!("event: {event}\ndata: {data}\n\n"))
        .collect();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/event-stream")],
        body,
    )
        .into_response()
}

// ============================================================================
// Handler
// ============================================================================

/// `POST /v1/messages` — Anthropic Messages endpoint.
///
/// The body is parsed here rather than by axum's `Json` extractor so that a
/// malformed request gets the Messages error envelope (400
/// `invalid_request_error`) instead of a plain-text 422.
pub async fn anthropic_messages_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Extension(cancel): Extension<crate::generate::CancelToken>,
    body: Bytes,
) -> Response {
    let request: MessagesRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &e.to_string()),
    };
    let chat_req = match to_chat_request(&request) {
        Ok(r) => r,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
    };
    let inner =
        openai_chat_completions_handler(State(state), headers, Extension(cancel), Json(chat_req))
            .await;
    let status = inner.status();
    let event_stream = inner
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"));
    if status.is_success() && event_stream {
        return live::live_response(inner);
    }
    let bytes = axum::body::to_bytes(inner.into_body(), usize::MAX)
        .await
        .unwrap_or_default();
    if !status.is_success() {
        return error_response(status, &upstream_error_message(&bytes));
    }
    let reply = serde_json::from_slice::<ChatCompletionResponse>(&bytes)
        .map_err(|e| format!("chat reply was not a chat completion: {e}"))
        .and_then(from_chat_response);
    match reply {
        Ok(r) if request.stream => sse_response(&r),
        Ok(r) => Json(r).into_response(),
        Err(msg) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &msg),
    }
}

#[path = "anthropic_messages_live.rs"]
mod live;

#[cfg(test)]
#[path = "tests/anthropic_messages.rs"]
mod tests;
