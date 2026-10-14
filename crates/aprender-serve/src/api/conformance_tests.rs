//! #5349: the conformance suite run in process over `create_router` on the
//! demo model, plus the planted `finish_reason` mutant that must go RED.

use super::checks;
use super::*;
use axum::{body::Body, http::Request};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};

const MODEL: &str = "demo";
const BACKEND: &str = "cpu";

fn demo_app() -> axum::Router {
    static DEMO: std::sync::OnceLock<AppState> = std::sync::OnceLock::new();
    let state = DEMO.get_or_init(|| AppState::demo().expect("demo AppState"));
    create_router(state.clone())
}

fn send_in_process(rt: &tokio::runtime::Runtime, req: &CaseRequest) -> Result<Observed, String> {
    let body = req
        .body
        .as_ref()
        .map_or_else(String::new, ToString::to_string);
    let request = Request::builder()
        .method(req.method)
        .uri(req.path)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .map_err(|e| e.to_string())?;
    rt.block_on(async {
        let resp = demo_app()
            .oneshot(request)
            .await
            .map_err(|e| e.to_string())?;
        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Observed {
            status,
            content_type,
            body: String::from_utf8_lossy(&bytes).into_owned(),
        })
    })
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
}

fn run_suite<F>(mut send: F) -> Vec<Cell>
where
    F: FnMut(&CaseRequest) -> Result<Observed, String>,
{
    cases(MODEL)
        .iter()
        .map(|case| run_case(case, MODEL, BACKEND, &mut send))
        .collect()
}

fn verdict_of<'a>(cells: &'a [Cell], id: &str) -> &'a Verdict {
    &cells
        .iter()
        .find(|c| c.case == id)
        .expect("cell for case")
        .verdict
}

/// The planted mutant: every `finish_reason` key removed from the response
/// body, JSON and SSE alike, as a server that dropped it would answer.
fn drop_finish_reason(o: Observed) -> Observed {
    fn strip(v: &mut serde_json::Value) {
        match v {
            serde_json::Value::Object(m) => {
                m.remove("finish_reason");
                m.values_mut().for_each(strip);
            },
            serde_json::Value::Array(a) => a.iter_mut().for_each(strip),
            _ => {},
        }
    }
    let strip_text = |s: &str| match serde_json::from_str::<serde_json::Value>(s) {
        Ok(mut v) => {
            strip(&mut v);
            v.to_string()
        },
        Err(_) => s.to_string(),
    };
    let body = if o.content_type.starts_with("text/event-stream") {
        o.body
            .lines()
            .map(|l| match l.strip_prefix("data: ") {
                Some(d) => format!("data: {}", strip_text(d)),
                None => l.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        strip_text(&o.body)
    };
    Observed { body, ..o }
}

#[test]
fn suite_covers_every_row_on_every_shape() {
    let suite = cases(MODEL);
    assert_eq!(uncovered_rows(&suite), Vec::<String>::new());
    let mut ids: Vec<_> = suite.iter().map(|c| c.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), suite.len(), "case ids must be unique");
}

#[test]
fn uncovered_rows_names_a_missing_row() {
    let suite: Vec<Case> = cases(MODEL)
        .into_iter()
        .filter(|c| c.row != "R-3")
        .collect();
    assert_eq!(uncovered_rows(&suite), vec!["R-3 OpenAi", "R-3 Anthropic"]);
}

#[test]
fn missing_cells_names_every_absent_cell() {
    let suite = cases(MODEL);
    let mut cells = run_suite(|_| Err("not sent".to_string()));
    assert!(missing_cells(&suite, &[MODEL], &[BACKEND], &cells).is_empty());
    cells.remove(0);
    let missing = missing_cells(&suite, &[MODEL], &[BACKEND, "cuda"], &cells);
    assert_eq!(missing.len(), suite.len() + 1, "{missing:?}");
    assert!(missing.contains(&format!("{} × {MODEL} × {BACKEND}", suite[0].id)));
}

#[test]
fn unsent_case_is_not_measured_never_pass() {
    for cell in run_suite(|_| Err("connection refused".to_string())) {
        assert!(
            matches!(cell.verdict, Verdict::NotMeasured { .. }),
            "{}: {:?}",
            cell.case,
            cell.verdict
        );
    }
}

/// The demo router answers the OpenAI base cases. Every case gets a measured
/// cell (pass or fail), none `not_measured`: the router was reachable.
#[test]
fn demo_router_openai_base_cases_pass() {
    let rt = runtime();
    let cells = run_suite(|r| send_in_process(&rt, r));
    for cell in &cells {
        assert!(
            !matches!(cell.verdict, Verdict::NotMeasured { .. }),
            "{} was not measured: {:?}",
            cell.case,
            cell.verdict
        );
    }
    for id in ["oai.models", "oai.chat.finish_reason", "oai.chat.stream"] {
        assert_eq!(verdict_of(&cells, id), &Verdict::Pass, "{id}");
    }
}

/// R-6 first-green proof: the same run with `finish_reason` dropped is RED on
/// both the non-streaming and the streaming case.
#[test]
fn planted_drop_finish_reason_turns_red() {
    let rt = runtime();
    let cells = run_suite(|r| send_in_process(&rt, r).map(drop_finish_reason));
    for id in ["oai.chat.finish_reason", "oai.chat.stream"] {
        assert!(
            matches!(verdict_of(&cells, id), Verdict::Fail { reason } if reason.contains("finish_reason")),
            "{id}: {:?}",
            verdict_of(&cells, id)
        );
    }
}

fn obs(status: u16, content_type: &str, body: &str) -> Observed {
    Observed {
        status,
        content_type: content_type.to_string(),
        body: body.to_string(),
    }
}

/// Must-pass / must-fail table for the checks the demo router cannot reach.
#[test]
fn check_case_table() {
    let json = "application/json";
    let sse = "text/event-stream";
    let tool_oai = r#"{"choices":[{"message":{"tool_calls":[{"function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":"tool_calls"}]}"#;
    let msg = |t: &str| {
        format!(
            r#"{{"type":"message","content":[{{"type":"text","text":"{t}"}}],"stop_reason":"end_turn","usage":{{"input_tokens":5,"output_tokens":3}}}}"#
        )
    };
    let anth_sse = "event: message_start\ndata: {}\n\nevent: content_block_delta\ndata: {}\n\nevent: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"}}\n\nevent: message_stop\ndata: {}\n\n";
    let chat = |t: &str| {
        obs(
            200,
            json,
            &format!(r#"{{"choices":[{{"message":{{"content":"{t}"}},"finish_reason":"stop"}}]}}"#),
        )
    };
    let tool_anth = r#"{"content":[{"type":"tool_use","name":"get_weather","input":{"city":"Paris"}}],"stop_reason":"tool_use"}"#;
    let tool_sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"ci\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ty\\\":\\\"Paris\\\"}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let text_sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let rows: Vec<(&str, fn(&[Observed]) -> Result<(), String>, Vec<Observed>, bool)> = vec![
        ("tool_call ok", checks::oai_tool_call, vec![obs(200, json, tool_oai)], true),
        ("tool_call stripped", checks::oai_tool_call, vec![chat("hi")], false),
        ("tool_call 404", checks::oai_tool_call, vec![obs(404, json, "{}")], false),
        ("stream tool ok", checks::oai_stream_tool_call, vec![obs(200, sse, tool_sse)], true),
        ("stream tool text only", checks::oai_stream_tool_call, vec![obs(200, sse, text_sse)], false),
        ("stream tool no done", checks::oai_stream_tool_call, vec![obs(200, sse, &tool_sse.replace("data: [DONE]\n\n", ""))], false),
        ("stream tool said stop", checks::oai_stream_tool_call, vec![obs(200, sse, &tool_sse.replace("\"tool_calls\"}", "\"stop\"}"))], false),
        ("stream tool bad args", checks::oai_stream_tool_call, vec![obs(200, sse, &tool_sse.replace("{\\\"ci", "[\\\"ci"))], false),
        ("seed ok", checks::oai_seed, vec![chat("a"), chat("a"), chat("b")], true),
        ("seed greedy", checks::oai_seed, vec![chat("a"), chat("a"), chat("a")], false),
        ("seed unstable", checks::oai_seed, vec![chat("a"), chat("b"), chat("c")], false),
        ("usage ok", checks::oai_usage, vec![obs(200, json, r#"{"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}"#)], true),
        ("usage bad total", checks::oai_usage, vec![obs(200, json, r#"{"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":9}}"#)], false),
        ("usage absent", checks::oai_usage, vec![chat("hi")], false),
        ("length ok", checks::oai_chat_length, vec![obs(200, json, r#"{"choices":[{"finish_reason":"length"}]}"#)], true),
        ("length said stop", checks::oai_chat_length, vec![chat("hi")], false),
        ("anth message ok", checks::anth_message, vec![obs(200, json, &msg("hi"))], true),
        ("anth message 404", checks::anth_message, vec![obs(404, json, "{}")], false),
        ("anth stream ok", checks::anth_stream, vec![obs(200, sse, anth_sse)], true),
        ("anth stream no stop", checks::anth_stream, vec![obs(200, sse, &anth_sse.replace("event: message_stop\ndata: {}\n\n", ""))], false),
        ("anth stream json ct", checks::anth_stream, vec![obs(200, json, anth_sse)], false),
        ("anth usage ok", checks::anth_usage, vec![obs(200, json, &msg("hi"))], true),
        ("anth usage absent", checks::anth_usage, vec![obs(200, json, r#"{"type":"message"}"#)], false),
        ("anth greedy ok", checks::anth_greedy_repeat, vec![obs(200, json, &msg("a")), obs(200, json, &msg("a"))], true),
        ("anth greedy differs", checks::anth_greedy_repeat, vec![obs(200, json, &msg("a")), obs(200, json, &msg("b"))], false),
        ("anth tool ok", checks::anth_tool_use, vec![obs(200, json, tool_anth)], true),
        ("anth tool text only", checks::anth_tool_use, vec![obs(200, json, &msg("hi"))], false),
        ("stream done twice", checks::oai_stream, vec![obs(200, sse, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\ndata: [DONE]\n\n")], false),
        ("stream ok", checks::oai_stream, vec![obs(200, sse, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")], true),
        ("stream no done", checks::oai_stream, vec![obs(200, sse, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":\"stop\"}]}\n\n")], false),
        ("stream empty text ok", checks::oai_stream, vec![obs(200, sse, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n")], true),
        ("stream no delta", checks::oai_stream, vec![obs(200, sse, "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")], false),
        ("stream content number", checks::oai_stream, vec![obs(200, sse, "data: {\"choices\":[{\"delta\":{\"content\":7},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")], false),
        ("stream only done", checks::oai_stream, vec![obs(200, sse, "data: [DONE]\n\n")], false),
        ("models empty", checks::oai_models, vec![obs(200, json, r#"{"data":[]}"#)], false),
    ];
    for (name, check, observed, want_pass) in rows {
        let got = check(&observed);
        assert_eq!(got.is_ok(), want_pass, "{name}: {got:?}");
    }
}
