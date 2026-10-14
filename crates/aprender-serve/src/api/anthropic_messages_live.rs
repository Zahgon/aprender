//! Live `stream: true` for `POST /v1/messages` (#4000 R-1 step 2).
//!
//! There is one event stream and two encoders. Generation streams through the
//! OpenAI live SSE path unchanged. Each `chat.completion.chunk` that path
//! emits is re-encoded here as a Messages event while it is still in flight,
//! so a token reaches the client as soon as the OpenAI path releases it.
//!
//! A stream that ends without its `[DONE]` sentinel, or that carries an
//! error, ends with a Messages `error` event and no `message_stop`. A cut-off
//! generation is never presented as a finished one.

use std::{convert::Infallible, time::Duration};

use axum::response::{
    sse::{Event, KeepAlive, Sse},
    IntoResponse, Response,
};
use futures::StreamExt;
use serde_json::{json, Value};

use super::super::{ChatCompletionChunk, ResponseToolCall};
use super::{block_events, stop_reason, tool_use_block};

/// One Messages SSE event, as `(event name, data)`.
pub(super) type SseEvent = (&'static str, Value);

/// Re-encodes the chat chunk stream as the Messages event sequence.
///
/// `message_start` carries `input_tokens: 0` because the prompt count is only
/// known from the terminal chunk. The real count goes out on `message_delta`,
/// next to `output_tokens`.
#[derive(Debug, Default)]
pub(super) struct LiveMessages {
    started: bool,
    next_index: usize,
    open_text: Option<usize>,
    stop_reason: Option<&'static str>,
    input_tokens: usize,
    output_tokens: usize,
    finished: bool,
}

impl LiveMessages {
    /// The events for one SSE `data` payload of the chat stream.
    pub(super) fn frame(&mut self, data: &str) -> Vec<SseEvent> {
        if self.finished {
            return Vec::new();
        }
        if data == "[DONE]" {
            return self.finish();
        }
        let value: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => return self.fail(&format!("chat stream frame is not JSON: {e}")),
        };
        if let Some(err) = value.get("error") {
            let msg = err
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| err.as_str())
                .map_or_else(|| err.to_string(), str::to_string);
            return self.fail(&msg);
        }
        match serde_json::from_value::<ChatCompletionChunk>(value) {
            Ok(chunk) => self.chunk(chunk),
            Err(e) => self.fail(&format!("chat stream frame is not a chunk: {e}")),
        }
    }

    /// The events owed when the chat stream's body ends: none after `[DONE]`
    /// or an error, otherwise an `error` event for the truncated generation.
    pub(super) fn end(&mut self) -> Vec<SseEvent> {
        if self.finished {
            return Vec::new();
        }
        self.fail("generation stream ended before completion")
    }

    fn chunk(&mut self, chunk: ChatCompletionChunk) -> Vec<SseEvent> {
        let mut events = Vec::new();
        if !self.started {
            self.started = true;
            events.extend(start_events(&chunk));
        }
        if let Some(usage) = &chunk.usage {
            self.input_tokens = usage.prompt_tokens;
            self.output_tokens = usage.completion_tokens;
        }
        for choice in chunk.choices {
            if let Some(text) = choice.delta.content.filter(|t| !t.is_empty()) {
                let index = self.text_block(&mut events);
                events.push((
                    "content_block_delta",
                    json!({"type": "content_block_delta", "index": index,
                           "delta": {"type": "text_delta", "text": text}}),
                ));
            }
            for call in choice.delta.tool_calls.unwrap_or_default() {
                self.close_text(&mut events);
                let call = ResponseToolCall {
                    id: call.id,
                    call_type: call.call_type,
                    function: call.function,
                };
                match tool_use_block(call) {
                    Ok(block) => {
                        events.extend(block_events(self.next_index, &block));
                        self.next_index += 1;
                    },
                    Err(msg) => {
                        events.extend(self.fail(&msg));
                        return events;
                    },
                }
            }
            if let Some(finish) = choice.finish_reason {
                self.stop_reason = Some(stop_reason(&finish));
            }
        }
        events
    }

    fn text_block(&mut self, events: &mut Vec<SseEvent>) -> usize {
        if let Some(index) = self.open_text {
            return index;
        }
        let index = self.next_index;
        self.next_index += 1;
        self.open_text = Some(index);
        events.push((
            "content_block_start",
            json!({"type": "content_block_start", "index": index,
                   "content_block": {"type": "text", "text": ""}}),
        ));
        index
    }

    fn close_text(&mut self, events: &mut Vec<SseEvent>) {
        if let Some(index) = self.open_text.take() {
            events.push((
                "content_block_stop",
                json!({"type": "content_block_stop", "index": index}),
            ));
        }
    }

    fn finish(&mut self) -> Vec<SseEvent> {
        if !self.started {
            return self.fail("generation stream ended before its first chunk");
        }
        let mut events = Vec::new();
        self.close_text(&mut events);
        self.finished = true;
        events.push((
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": self.stop_reason.unwrap_or("end_turn"), "stop_sequence": null},
                "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens},
            }),
        ));
        events.push(("message_stop", json!({"type": "message_stop"})));
        events
    }

    fn fail(&mut self, message: &str) -> Vec<SseEvent> {
        self.finished = true;
        vec![(
            "error",
            json!({"type": "error", "error": {"type": "api_error", "message": message}}),
        )]
    }
}

fn start_events(chunk: &ChatCompletionChunk) -> Vec<SseEvent> {
    let message = json!({
        "id": format!("msg_{}", chunk.id.trim_start_matches("chatcmpl-")),
        "type": "message",
        "role": "assistant",
        "model": chunk.model,
        "content": [],
        "stop_reason": null,
        "stop_sequence": null,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    });
    vec![
        (
            "message_start",
            json!({"type": "message_start", "message": message}),
        ),
        ("ping", json!({"type": "ping"})),
    ]
}

/// The `data` payload of one SSE frame (its `data:` lines joined by `\n`),
/// or `None` for a frame with no data, such as a keep-alive comment.
pub(super) fn frame_data(frame: &str) -> Option<String> {
    let lines: Vec<&str> = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|rest| rest.strip_prefix(' ').unwrap_or(rest))
        .collect();
    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn to_event((name, data): SseEvent) -> Result<Event, Infallible> {
    Ok(Event::default().event(name).data(data.to_string()))
}

/// Re-encode the chat path's live SSE response as a Messages SSE response.
///
/// Frames are split on the blank line that ends each one, at the byte level,
/// so a multi-byte character cut across two body chunks is decoded whole.
pub(super) fn live_response(inner: Response) -> Response {
    let mut body = inner.into_body().into_data_stream();
    let stream = async_stream::stream! {
        let mut encoder = LiveMessages::default();
        let mut buf: Vec<u8> = Vec::new();
        'body: while let Some(next) = body.next().await {
            let bytes = match next {
                Ok(b) => b,
                Err(e) => {
                    for ev in encoder.fail(&format!("generation stream failed: {e}")) {
                        yield to_event(ev);
                    }
                    break;
                }
            };
            buf.extend_from_slice(&bytes);
            while let Some(end) = buf.windows(2).position(|w| w == b"\n\n") {
                let frame: Vec<u8> = buf.drain(..end + 2).collect();
                let Some(data) = frame_data(&String::from_utf8_lossy(&frame)) else {
                    continue;
                };
                for ev in encoder.frame(&data) {
                    yield to_event(ev);
                }
                if encoder.finished {
                    break 'body;
                }
            }
        }
        for ev in encoder.end() {
            yield to_event(ev);
        }
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

#[cfg(test)]
#[path = "tests/anthropic_messages_live.rs"]
mod tests;
