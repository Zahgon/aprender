//! Checks for the conformance cases (#5349). Each takes every response of
//! its case and names the first assertion that fails. A check never passes on
//! an absent field: a missing `finish_reason` is a failure, not "unknown".

use super::Observed;
use serde_json::Value;

fn one(obs: &[Observed]) -> Result<&Observed, String> {
    obs.first().ok_or_else(|| "no response".to_string())
}

/// Status 200 and a JSON body.
fn ok_json(o: &Observed) -> Result<Value, String> {
    if o.status != 200 {
        return Err(format!(
            "status {} (want 200): {}",
            o.status,
            snippet(&o.body)
        ));
    }
    serde_json::from_str(&o.body)
        .map_err(|e| format!("body is not JSON ({e}): {}", snippet(&o.body)))
}

fn snippet(s: &str) -> String {
    s.chars().take(160).collect()
}

fn str_at<'a>(v: &'a Value, ptr: &str) -> Result<&'a str, String> {
    v.pointer(ptr)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{ptr} is absent or not a string"))
}

fn u64_at(v: &Value, ptr: &str) -> Result<u64, String> {
    v.pointer(ptr)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{ptr} is absent or not an integer"))
}

/// Status 200, an SSE content type, and the `data:` payloads in order, with
/// the `event:` name each came under (`""` when none).
fn sse(o: &Observed) -> Result<Vec<(String, String)>, String> {
    if o.status != 200 {
        return Err(format!(
            "status {} (want 200): {}",
            o.status,
            snippet(&o.body)
        ));
    }
    if !o.content_type.starts_with("text/event-stream") {
        return Err(format!(
            "content-type {:?} (want text/event-stream)",
            o.content_type
        ));
    }
    let mut event = String::new();
    let mut out = Vec::new();
    for line in o.body.lines() {
        if let Some(name) = line.strip_prefix("event:") {
            event = name.trim().to_string();
        } else if let Some(data) = line.strip_prefix("data:") {
            out.push((event.clone(), data.trim().to_string()));
        } else if line.is_empty() {
            event.clear();
        }
    }
    if out.is_empty() {
        return Err("stream carried no data events".to_string());
    }
    Ok(out)
}

/// `GET /v1/models`: a non-empty `data` list of objects with an `id`.
pub fn oai_models(obs: &[Observed]) -> Result<(), String> {
    let v = ok_json(one(obs)?)?;
    let data = v
        .get("data")
        .and_then(Value::as_array)
        .ok_or("data is absent or not a list")?;
    if data.is_empty() {
        return Err("data is empty".to_string());
    }
    for (i, m) in data.iter().enumerate() {
        str_at(m, "/id").map_err(|e| format!("data[{i}]: {e}"))?;
    }
    Ok(())
}

fn oai_finish(v: &Value) -> Result<&str, String> {
    str_at(v, "/choices/0/finish_reason")
}

/// Non-streaming chat: a string reply and `finish_reason` `stop` or `length`.
pub fn oai_chat_finish(obs: &[Observed]) -> Result<(), String> {
    let v = ok_json(one(obs)?)?;
    str_at(&v, "/choices/0/message/content")?;
    match oai_finish(&v)? {
        "stop" | "length" => Ok(()),
        other => Err(format!("finish_reason {other:?} (want stop or length)")),
    }
}

/// `max_tokens: 1` must report `finish_reason: "length"`.
pub fn oai_chat_length(obs: &[Observed]) -> Result<(), String> {
    let v = ok_json(one(obs)?)?;
    match oai_finish(&v)? {
        "length" => Ok(()),
        other => Err(format!(
            "finish_reason {other:?} at max_tokens 1 (want length)"
        )),
    }
}

/// Streaming chat: JSON chunks that each carry a `delta` object (its
/// `content`, when present, a string), a `finish_reason` on a chunk, and
/// `[DONE]` exactly once, last. Whether the text is non-empty is the model's
/// business, not the protocol's: the demo model decodes to `""`.
pub fn oai_stream(obs: &[Observed]) -> Result<(), String> {
    let events = sse(one(obs)?)?;
    let (last, chunks) = events.split_last().ok_or("no events")?;
    if last.1 != "[DONE]" {
        return Err(format!("last data is {:?} (want [DONE])", snippet(&last.1)));
    }
    if chunks.is_empty() {
        return Err("stream carried only [DONE]".to_string());
    }
    let mut finish = None;
    for (i, (_, data)) in chunks.iter().enumerate() {
        if data == "[DONE]" {
            return Err(format!("[DONE] at event {i}, before the end"));
        }
        let v: Value = serde_json::from_str(data)
            .map_err(|e| format!("event {i} is not JSON ({e}): {}", snippet(data)))?;
        let delta = v
            .pointer("/choices/0/delta")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                format!(
                    "event {i} has no choices[0].delta object: {}",
                    snippet(data)
                )
            })?;
        if delta
            .get("content")
            .is_some_and(|c| !c.is_string() && !c.is_null())
        {
            return Err(format!(
                "event {i} delta.content is not a string: {}",
                snippet(data)
            ));
        }
        if let Some(f) = v
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
        {
            finish = Some(f.to_string());
        }
    }
    match finish.as_deref() {
        Some("stop" | "length") => Ok(()),
        Some(other) => Err(format!(
            "stream finish_reason {other:?} (want stop or length)"
        )),
        None => Err("no chunk carried a finish_reason".to_string()),
    }
}

/// `usage` present and consistent: prompt > 0, 0 < completion ≤ 16, total = sum.
pub fn oai_usage(obs: &[Observed]) -> Result<(), String> {
    let v = ok_json(one(obs)?)?;
    let p = u64_at(&v, "/usage/prompt_tokens")?;
    let c = u64_at(&v, "/usage/completion_tokens")?;
    let t = u64_at(&v, "/usage/total_tokens")?;
    if p == 0 || c == 0 || c > 16 {
        return Err(format!(
            "usage prompt={p} completion={c} (want >0, completion ≤ max_tokens 16)"
        ));
    }
    if t != p + c {
        return Err(format!("total_tokens {t} != prompt {p} + completion {c}"));
    }
    Ok(())
}

fn oai_text(o: &Observed) -> Result<String, String> {
    Ok(str_at(&ok_json(o)?, "/choices/0/message/content")?.to_string())
}

/// Same seed at T=0.8 → identical text; a different seed → different text.
/// The second half is the positive control: a greedy engine passes the first.
pub fn oai_seed(obs: &[Observed]) -> Result<(), String> {
    let [a, b, c] = obs else {
        return Err(format!("{} responses (want 3)", obs.len()));
    };
    let (a, b, c) = (oai_text(a)?, oai_text(b)?, oai_text(c)?);
    if a != b {
        return Err(format!(
            "seed 7 twice gave different text: {:?} vs {:?}",
            snippet(&a),
            snippet(&b)
        ));
    }
    if a == c {
        return Err(format!(
            "seeds 7 and 8 gave the same text at T=0.8 (sampling is greedy): {:?}",
            snippet(&a)
        ));
    }
    Ok(())
}

/// Forced `get_weather`: a tool call with JSON object arguments and
/// `finish_reason: "tool_calls"`.
pub fn oai_tool_call(obs: &[Observed]) -> Result<(), String> {
    let v = ok_json(one(obs)?)?;
    let name = str_at(&v, "/choices/0/message/tool_calls/0/function/name")?;
    if name != "get_weather" {
        return Err(format!("tool_calls[0] calls {name:?} (want get_weather)"));
    }
    let args = str_at(&v, "/choices/0/message/tool_calls/0/function/arguments")?;
    match serde_json::from_str::<Value>(args) {
        Ok(Value::Object(_)) => {},
        _ => {
            return Err(format!(
                "arguments {:?} is not a JSON object",
                snippet(args)
            ))
        },
    }
    match oai_finish(&v)? {
        "tool_calls" => Ok(()),
        other => Err(format!("finish_reason {other:?} (want tool_calls)")),
    }
}

/// Anthropic message: `type: message`, a text block, a known `stop_reason`.
pub fn anth_message(obs: &[Observed]) -> Result<(), String> {
    let v = ok_json(one(obs)?)?;
    if str_at(&v, "/type")? != "message" {
        return Err("type is not \"message\"".to_string());
    }
    if str_at(&v, "/content/0/type")? != "text" {
        return Err("content[0] is not a text block".to_string());
    }
    str_at(&v, "/content/0/text")?;
    match str_at(&v, "/stop_reason")? {
        "end_turn" | "max_tokens" | "stop_sequence" => Ok(()),
        other => Err(format!(
            "stop_reason {other:?} (want end_turn, max_tokens or stop_sequence)"
        )),
    }
}

/// Anthropic SSE: `message_start` first, `message_stop` last, a text delta,
/// and a `message_delta` carrying `stop_reason`.
pub fn anth_stream(obs: &[Observed]) -> Result<(), String> {
    let events = sse(one(obs)?)?;
    let names: Vec<&str> = events.iter().map(|(e, _)| e.as_str()).collect();
    if names.first() != Some(&"message_start") {
        return Err(format!(
            "first event {:?} (want message_start)",
            names.first()
        ));
    }
    if names.last() != Some(&"message_stop") {
        return Err(format!("last event {:?} (want message_stop)", names.last()));
    }
    if !names.contains(&"content_block_delta") {
        return Err("no content_block_delta event".to_string());
    }
    let stop = events
        .iter()
        .filter(|(e, _)| e == "message_delta")
        .find_map(|(_, d)| {
            serde_json::from_str::<Value>(d)
                .ok()?
                .pointer("/delta/stop_reason")?
                .as_str()
                .map(str::to_string)
        });
    stop.map(|_| ())
        .ok_or_else(|| "no message_delta carried delta.stop_reason".to_string())
}

/// Anthropic `usage`: input > 0, 0 < output ≤ 16.
pub fn anth_usage(obs: &[Observed]) -> Result<(), String> {
    let v = ok_json(one(obs)?)?;
    let i = u64_at(&v, "/usage/input_tokens")?;
    let o = u64_at(&v, "/usage/output_tokens")?;
    if i == 0 || o == 0 || o > 16 {
        return Err(format!(
            "usage input={i} output={o} (want >0, output ≤ max_tokens 16)"
        ));
    }
    Ok(())
}

/// Temperature 0 twice → identical text.
pub fn anth_greedy_repeat(obs: &[Observed]) -> Result<(), String> {
    let [a, b] = obs else {
        return Err(format!("{} responses (want 2)", obs.len()));
    };
    let a = str_at(&ok_json(a)?, "/content/0/text")?.to_string();
    let b = str_at(&ok_json(b)?, "/content/0/text")?.to_string();
    if a == b {
        Ok(())
    } else {
        Err(format!(
            "temperature 0 twice gave different text: {:?} vs {:?}",
            snippet(&a),
            snippet(&b)
        ))
    }
}

/// Forced `get_weather`: a `tool_use` block with an object `input` and
/// `stop_reason: "tool_use"`.
pub fn anth_tool_use(obs: &[Observed]) -> Result<(), String> {
    let v = ok_json(one(obs)?)?;
    let blocks = v
        .get("content")
        .and_then(Value::as_array)
        .ok_or("content is absent or not a list")?;
    let tool = blocks
        .iter()
        .find(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
        .ok_or("no tool_use block")?;
    if str_at(tool, "/name")? != "get_weather" {
        return Err("tool_use block does not call get_weather".to_string());
    }
    if !tool.get("input").is_some_and(Value::is_object) {
        return Err("tool_use input is not an object".to_string());
    }
    match str_at(&v, "/stop_reason")? {
        "tool_use" => Ok(()),
        other => Err(format!("stop_reason {other:?} (want tool_use)")),
    }
}
