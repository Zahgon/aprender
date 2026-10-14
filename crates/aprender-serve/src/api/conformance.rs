//! Agent conformance suite (#5349, EPIC 0.72 row R-6).
//!
//! One fixed list of client-observable cases, sent to a running `apr serve`
//! (any model, any backend) over the OpenAI and the Anthropic shapes. Each case
//! is a list of requests plus one check over what came back, so the same cases
//! run in process (through `create_router`) and over HTTP (the
//! `serve_conformance` example).
//!
//! A run yields one [`Cell`] per case × model × backend. There is no skip: a
//! case the server cannot answer is `fail` with the observed status, and a
//! case the runner could not send is `not_measured` with the reason. An
//! absent cell is a missing measurement ([`missing_cells`]), never a shorter
//! table.

use serde::Serialize;
use serde_json::{json, Value};

/// The client API a case speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Shape {
    /// `/v1/chat/completions`, `/v1/models`.
    OpenAi,
    /// `/v1/messages`.
    Anthropic,
    /// Ollama `/api/chat`.
    Ollama,
}

/// One HTTP request of a case.
#[derive(Debug, Clone)]
pub struct CaseRequest {
    /// `GET` or `POST`.
    pub method: &'static str,
    /// Route, e.g. `/v1/chat/completions`.
    pub path: &'static str,
    /// JSON body (`None` for `GET`).
    pub body: Option<Value>,
}

/// What came back for one request.
#[derive(Debug, Clone)]
pub struct Observed {
    /// HTTP status.
    pub status: u16,
    /// `content-type` header, lowercased (empty when absent).
    pub content_type: String,
    /// Raw body text.
    pub body: String,
}

/// One conformance case.
#[derive(Debug, Clone)]
pub struct Case {
    /// Stable id, e.g. `oai.chat.finish_reason`.
    pub id: &'static str,
    /// Client shape.
    pub shape: Shape,
    /// The EPIC row the case proves (`R-1`..`R-5`).
    pub row: &'static str,
    /// Requests, sent in order.
    pub requests: Vec<CaseRequest>,
    /// Judges every response of the case together.
    pub check: fn(&[Observed]) -> Result<(), String>,
}

/// The verdict of one case on one model × backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    /// Every assertion held.
    Pass,
    /// An assertion failed; `reason` names it.
    Fail {
        /// The failing assertion.
        reason: String,
    },
    /// The case could not be sent; `reason` says why.
    NotMeasured {
        /// Why nothing was measured.
        reason: String,
    },
}

/// One receipt cell: case × model × backend.
#[derive(Debug, Clone, Serialize)]
pub struct Cell {
    /// Case id.
    pub case: &'static str,
    /// Client shape.
    pub shape: Shape,
    /// EPIC row.
    pub row: &'static str,
    /// Model under test.
    pub model: String,
    /// Backend under test (`cpu`, `cuda`, ...).
    pub backend: String,
    /// Result.
    #[serde(flatten)]
    pub verdict: Verdict,
}

/// Prompt every generation case sends.
const PROMPT: &str = "Say hello in one short sentence.";

fn oai_chat(model: &str, extra: Value) -> Value {
    let mut body = json!({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 16,
    });
    merge(&mut body, extra);
    body
}

fn anth_messages(model: &str, extra: Value) -> Value {
    let mut body = json!({
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
    });
    merge(&mut body, extra);
    body
}

/// Ollama caps generation with `options.num_predict`, not `max_tokens`.
fn ollama_chat(model: &str, stream: bool, options: Value) -> Value {
    let mut opts = json!({"num_predict": 16});
    merge(&mut opts, options);
    json!({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": stream,
        "options": opts,
    })
}

fn merge(body: &mut Value, extra: Value) {
    if let (Some(dst), Value::Object(src)) = (body.as_object_mut(), extra) {
        dst.extend(src);
    }
}

fn post(path: &'static str, body: Value) -> CaseRequest {
    CaseRequest {
        method: "POST",
        path,
        body: Some(body),
    }
}

fn weather_tool_oai() -> Value {
    json!({
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "max_tokens": 64,
    })
}

/// `weather_tool_oai` with `stream: true`, for the streamed tool-call case.
fn stream_tool_oai() -> Value {
    let mut body = weather_tool_oai();
    merge(&mut body, json!({"stream": true}));
    body
}

fn weather_tool_anth() -> Value {
    json!({
        "tools": [{"name": "get_weather", "description": "Current weather for a city",
                   "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                                    "required": ["city"]}}],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "max_tokens": 64,
    })
}

/// The suite, for the model id the server answers to.
#[must_use]
pub fn cases(model: &str) -> Vec<Case> {
    use checks::*;
    let chat = "/v1/chat/completions";
    let msgs = "/v1/messages";
    let ochat = "/api/chat";
    let seeded = |seed: u64| json!({"temperature": 0.8, "seed": seed});
    vec![
        Case {
            id: "oai.models",
            shape: Shape::OpenAi,
            row: "R-5",
            requests: vec![CaseRequest {
                method: "GET",
                path: "/v1/models",
                body: None,
            }],
            check: oai_models,
        },
        Case {
            id: "oai.chat.finish_reason",
            shape: Shape::OpenAi,
            row: "R-5",
            requests: vec![post(chat, oai_chat(model, json!({})))],
            check: oai_chat_finish,
        },
        Case {
            id: "oai.chat.length",
            shape: Shape::OpenAi,
            row: "R-5",
            requests: vec![post(chat, oai_chat(model, json!({"max_tokens": 1})))],
            check: oai_chat_length,
        },
        Case {
            id: "oai.chat.stream",
            shape: Shape::OpenAi,
            row: "R-5",
            requests: vec![post(chat, oai_chat(model, json!({"stream": true})))],
            check: oai_stream,
        },
        Case {
            id: "oai.chat.usage",
            shape: Shape::OpenAi,
            row: "R-4",
            requests: vec![post(chat, oai_chat(model, json!({})))],
            check: oai_usage,
        },
        Case {
            id: "oai.chat.seed",
            shape: Shape::OpenAi,
            row: "R-3",
            requests: vec![
                post(chat, oai_chat(model, seeded(7))),
                post(chat, oai_chat(model, seeded(7))),
                post(chat, oai_chat(model, seeded(8))),
            ],
            check: oai_seed,
        },
        Case {
            id: "oai.chat.tool_call",
            shape: Shape::OpenAi,
            row: "R-2",
            requests: vec![post(chat, oai_chat(model, weather_tool_oai()))],
            check: oai_tool_call,
        },
        Case {
            id: "oai.chat.stream_tool_call",
            shape: Shape::OpenAi,
            row: "R-2",
            requests: vec![post(chat, oai_chat(model, stream_tool_oai()))],
            check: oai_stream_tool_call,
        },
        Case {
            id: "anth.messages.stop_reason",
            shape: Shape::Anthropic,
            row: "R-1",
            requests: vec![post(msgs, anth_messages(model, json!({})))],
            check: anth_message,
        },
        Case {
            id: "anth.messages.stream",
            shape: Shape::Anthropic,
            row: "R-5",
            requests: vec![post(msgs, anth_messages(model, json!({"stream": true})))],
            check: anth_stream,
        },
        Case {
            id: "anth.messages.usage",
            shape: Shape::Anthropic,
            row: "R-4",
            requests: vec![post(msgs, anth_messages(model, json!({})))],
            check: anth_usage,
        },
        Case {
            id: "anth.messages.greedy_repeat",
            shape: Shape::Anthropic,
            row: "R-3",
            requests: vec![
                post(msgs, anth_messages(model, json!({"temperature": 0.0}))),
                post(msgs, anth_messages(model, json!({"temperature": 0.0}))),
            ],
            check: anth_greedy_repeat,
        },
        Case {
            id: "anth.messages.tool_use",
            shape: Shape::Anthropic,
            row: "R-2",
            requests: vec![post(msgs, anth_messages(model, weather_tool_anth()))],
            check: anth_tool_use,
        },
        Case {
            id: "ollama.chat.seed",
            shape: Shape::Ollama,
            row: "R-3",
            requests: vec![
                post(ochat, ollama_chat(model, false, seeded(7))),
                post(ochat, ollama_chat(model, false, seeded(7))),
                post(ochat, ollama_chat(model, false, seeded(8))),
            ],
            check: ollama_seed,
        },
        Case {
            id: "ollama.chat.usage",
            shape: Shape::Ollama,
            row: "R-4",
            requests: vec![post(ochat, ollama_chat(model, false, json!({})))],
            check: ollama_usage,
        },
        Case {
            id: "ollama.chat.done",
            shape: Shape::Ollama,
            row: "R-5",
            requests: vec![post(ochat, ollama_chat(model, true, json!({})))],
            check: ollama_done,
        },
    ]
}

/// Run `case` with `send` and judge it. `send` returns `Err` only when the
/// request could not be sent at all, which makes the cell `not_measured`.
pub fn run_case<F>(case: &Case, model: &str, backend: &str, mut send: F) -> Cell
where
    F: FnMut(&CaseRequest) -> Result<Observed, String>,
{
    let mut observed = Vec::with_capacity(case.requests.len());
    let mut verdict = None;
    for req in &case.requests {
        match send(req) {
            Ok(o) => observed.push(o),
            Err(e) => {
                verdict = Some(Verdict::NotMeasured {
                    reason: format!("{} {}: {e}", req.method, req.path),
                });
                break;
            },
        }
    }
    let verdict = verdict.unwrap_or_else(|| match (case.check)(&observed) {
        Ok(()) => Verdict::Pass,
        Err(reason) => Verdict::Fail { reason },
    });
    Cell {
        case: case.id,
        shape: case.shape,
        row: case.row,
        model: model.to_string(),
        backend: backend.to_string(),
        verdict,
    }
}

/// Every (case, model, backend) the run was asked for and has no cell for.
#[must_use]
pub fn missing_cells(
    cases: &[Case],
    models: &[&str],
    backends: &[&str],
    cells: &[Cell],
) -> Vec<String> {
    let mut missing = Vec::new();
    for case in cases {
        for model in models {
            for backend in backends {
                let found = cells
                    .iter()
                    .any(|c| c.case == case.id && c.model == *model && c.backend == *backend);
                if !found {
                    missing.push(format!("{} × {model} × {backend}", case.id));
                }
            }
        }
    }
    missing
}

/// Rows R-1..R-5 each need at least one case per shape that speaks it (R-1
/// is the Anthropic adapter, so it has no OpenAI case; the Ollama shape
/// speaks R-3..R-5 only). Returns the uncovered
/// (row, shape) pairs; an empty suite would otherwise pass.
#[must_use]
pub fn uncovered_rows(cases: &[Case]) -> Vec<String> {
    let mut gaps = Vec::new();
    for row in ["R-1", "R-2", "R-3", "R-4", "R-5"] {
        for shape in [Shape::OpenAi, Shape::Anthropic, Shape::Ollama] {
            let speaks = match shape {
                Shape::OpenAi => row != "R-1",
                Shape::Anthropic => true,
                Shape::Ollama => !matches!(row, "R-1" | "R-2"),
            };
            if !speaks {
                continue;
            }
            if !cases.iter().any(|c| c.row == row && c.shape == shape) {
                gaps.push(format!("{row} {shape:?}"));
            }
        }
    }
    gaps
}

#[path = "conformance_checks.rs"]
pub mod checks;

#[cfg(test)]
#[path = "conformance_tests.rs"]
mod tests;
