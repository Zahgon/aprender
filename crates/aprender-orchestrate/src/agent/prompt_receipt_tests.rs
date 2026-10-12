//! #5190 falsifiers: a turn records how much of its prompt reached the driver, and a
//! request that carries less than was sent ends the turn with `input_not_consumed`.

use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use super::*;
use crate::agent::code_envelope::{envelope, CodeOutcome};
use crate::agent::code_prompts::map_error_to_exit_code;
use crate::agent::driver::{CompletionResponse, LlmDriver};
use crate::agent::manifest::AgentManifest;
use crate::agent::memory::InMemorySubstrate;
use crate::agent::result::{AgentLoopResult, StopReason, TokenUsage};
use crate::agent::runtime::run_agent_loop;
use crate::agent::tool::ToolRegistry;
use crate::serve::context::{ContextConfig, ContextWindow, TruncationStrategy};

/// SHA-256 of "abc" (FIPS 180-2, appendix B.1).
const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
const WINDOW: usize = 4096;

fn context() -> ContextManager {
    ContextManager::new(ContextConfig {
        window: ContextWindow::new(WINDOW, 256),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,
        min_messages: 2,
    })
}

fn request(messages: Vec<Message>) -> CompletionRequest {
    CompletionRequest {
        model: String::new(),
        messages,
        tools: vec![],
        max_tokens: 256,
        temperature: 0.0,
        system: None,
    }
}

fn tokens(text: &str) -> usize {
    context().estimate_tokens(&[Message::User(text.into()).to_chat_message()])
}

#[test]
fn whole_prompt_is_consumed_and_hashed() {
    let r = measure("abc", &request(vec![Message::User("abc".into())]), &context());
    assert_eq!(
        r,
        PromptReceipt {
            input_bytes_sent: 3,
            input_bytes_consumed: 3,
            prompt_sha256: ABC_SHA256.into(),
            prompt_tokens: tokens("abc"),
        }
    );
}

#[test]
fn the_last_user_message_is_the_prompt() {
    let msgs = vec![Message::User("an earlier turn".into()), Message::User("abc".into())];
    assert_eq!(measure("abc", &request(msgs), &context()).input_bytes_consumed, 3);
    // an earlier turn holding the prompt does not count for the current one
    let msgs = vec![Message::User("abc".into()), Message::User("ab".into())];
    assert_eq!(measure("abc", &request(msgs), &context()).input_bytes_consumed, 2);
}

#[test]
fn a_dropped_tail_byte_is_a_short_count() {
    let r = measure("abcdef", &request(vec![Message::User("abcde".into())]), &context());
    assert_eq!((r.input_bytes_sent, r.input_bytes_consumed), (6, 5));
    assert_eq!(r.prompt_sha256, format!("{:x}", Sha256::digest(b"abcde")));
    assert_eq!(r.prompt_tokens, tokens("abcde"));
}

#[test]
fn a_dropped_middle_byte_counts_only_the_shared_prefix() {
    let r = measure("abcdef", &request(vec![Message::User("abdef".into())]), &context());
    assert_eq!(r.input_bytes_consumed, 2);
}

#[test]
fn no_user_message_consumes_nothing() {
    let r = measure("abc", &request(vec![Message::Assistant("abc".into())]), &context());
    assert_eq!(r.input_bytes_consumed, 0);
    assert_eq!(r.prompt_sha256, format!("{:x}", Sha256::digest(b"")));
}

#[test]
fn check_refuses_only_a_short_count() {
    let short = check("abc", &request(vec![Message::User("ab".into())]), &context());
    assert!(
        matches!(short, Err(AgentError::InputNotConsumed(ref r)) if r.input_bytes_consumed == 2)
    );
    assert!(check("abc", &request(vec![Message::User("abc".into())]), &context()).is_ok());
    assert!(check("", &request(vec![Message::User(String::new())]), &context()).is_ok());
}

/// Answers every request and counts how many it was sent.
struct CountingDriver {
    calls: AtomicUsize,
}

#[async_trait]
impl LlmDriver for CountingDriver {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, AgentError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(CompletionResponse {
            text: "ok".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage::default(),
        })
    }

    fn context_window(&self) -> usize {
        WINDOW
    }

    fn privacy_tier(&self) -> crate::serve::backends::PrivacyTier {
        crate::serve::backends::PrivacyTier::Sovereign
    }
}

fn manifest() -> AgentManifest {
    let mut m = AgentManifest::default();
    m.model.max_tokens = 256;
    m
}

async fn run(prompt: &str, driver: &CountingDriver) -> Result<AgentLoopResult, AgentError> {
    run_agent_loop(
        &manifest(),
        prompt,
        driver,
        &ToolRegistry::new(),
        &InMemorySubstrate::new(),
        None,
    )
    .await
}

/// Disarms the plant even when the test panics.
struct Armed;

impl Armed {
    fn new() -> Self {
        plant::arm(true);
        Self
    }
}

impl Drop for Armed {
    fn drop(&mut self) {
        plant::arm(false);
    }
}

/// FALSIFY-5190-001: one byte dropped between the loop and the driver turns the turn red:
/// `input_not_consumed`, a non-zero exit, the driver never called, the four fields in the
/// JSON document.
#[tokio::test]
async fn falsify_5190_001_planted_one_byte_drop_is_input_not_consumed() {
    let prompt = "What is 2+2? Answer with one number.";
    let driver = CountingDriver { calls: AtomicUsize::new(0) };
    let err = {
        let _armed = Armed::new();
        run(prompt, &driver).await.expect_err("a dropped byte must not be answered")
    };
    let AgentError::InputNotConsumed(ref r) = err else {
        panic!("want InputNotConsumed, got {err}")
    };
    assert_eq!((r.input_bytes_sent, r.input_bytes_consumed), (prompt.len(), prompt.len() - 1));
    assert_eq!(driver.calls.load(Ordering::SeqCst), 0, "the short request reached the driver");

    let code = map_error_to_exit_code(&err);
    assert_ne!(code, 0);
    let outcome = CodeOutcome::from_agent_error(&err, code);
    let doc: serde_json::Value =
        serde_json::from_str(&envelope(None, Some(&outcome), None, std::time::Duration::ZERO))
            .expect("json");
    assert_eq!(doc["error"]["kind"], "input_not_consumed");
    assert_eq!(doc["error"]["exit_code"], code);
    assert_eq!(doc["input_bytes_sent"], prompt.len());
    assert_eq!(doc["input_bytes_consumed"], prompt.len() - 1);
    assert_eq!(doc["prompt_sha256"], r.prompt_sha256.as_str());
    assert_eq!(doc["prompt_tokens"], r.prompt_tokens);
}

/// The plant is the only difference: the same prompt, disarmed, is answered.
#[tokio::test]
async fn falsify_5190_002_same_prompt_unplanted_is_green() {
    let prompt = "What is 2+2? Answer with one number.";
    let driver = CountingDriver { calls: AtomicUsize::new(0) };
    let r = run(prompt, &driver).await.expect("answered");
    let p = r.prompt.expect("an agent turn records its prompt");
    assert_eq!((p.input_bytes_sent, p.input_bytes_consumed), (prompt.len(), prompt.len()));
    assert_eq!(p.prompt_sha256, format!("{:x}", Sha256::digest(prompt.as_bytes())));
    let doc: serde_json::Value = serde_json::from_str(&envelope(
        Some(&AgentLoopResult { prompt: Some(p.clone()), ..r }),
        None,
        None,
        std::time::Duration::ZERO,
    ))
    .expect("json");
    assert_eq!(doc["input_bytes_consumed"], prompt.len());
    assert!(doc.get("error").is_none());
}

/// FALSIFY-5190-003: the largest prompt the window admits stays green with sent ==
/// consumed; one byte more is refused as `context_overflow` (#4599), never as a short count.
#[tokio::test]
async fn falsify_5190_003_prompt_at_the_window_limit_is_whole() {
    let driver = CountingDriver { calls: AtomicUsize::new(0) };
    let prompt = |n: usize| "x".repeat(n);
    let (mut lo, mut hi) = (1usize, WINDOW * 8);
    assert!(run(&prompt(lo), &driver).await.is_ok());
    assert!(run(&prompt(hi), &driver).await.is_err());
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if run(&prompt(mid), &driver).await.is_ok() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let at_limit = run(&prompt(lo), &driver).await.expect("the limit fits");
    let p = at_limit.prompt.expect("receipt");
    assert_eq!((p.input_bytes_sent, p.input_bytes_consumed), (lo, lo));
    assert!(lo > WINDOW, "the limit is {lo} bytes, want more than the {WINDOW}-token window");
    let over = run(&prompt(hi), &driver).await.expect_err("one byte over");
    assert!(matches!(over, AgentError::ContextOverflow { .. }), "got {over}");
}
