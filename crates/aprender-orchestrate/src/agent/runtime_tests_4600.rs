//! L22 (#4600) falsifiers: a verb result proves it consumed all of its input.
//!
//! `apr code` once answered a 212 KB prompt with rc 0 and `tokens_in 19` (#4599): nothing in
//! the result said how much of the prompt the model got. Every turn now measures the request
//! it hands the driver, refuses one that carries less than the prompt, and reports
//! `input_bytes_sent`, `input_bytes_consumed`, `prompt_sha256` and `prompt_tokens`.

use super::*;
use crate::agent::driver::mock::MockDriver;
use crate::agent::memory::InMemorySubstrate;

/// SHA-256 of the empty string: what an account that consumed nothing carries.
const SHA256_EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

fn manifest() -> AgentManifest {
    let mut m = AgentManifest::default();
    m.model.max_tokens = 4096;
    m
}

fn prompt(bytes: usize) -> String {
    let mut p = String::from("The passphrase is TANGERINE-4417. Remember it.\n");
    while p.len() < bytes {
        p.push_str("filler text for the context rung. ");
    }
    p.truncate(bytes);
    p
}

fn ctx(window: usize) -> ContextManager {
    ContextManager::new(ContextConfig {
        window: ContextWindow::new(window, 4096),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,
        min_messages: 2,
    })
}

async fn run(query: &str, window: usize) -> Result<AgentLoopResult, AgentError> {
    let driver = MockDriver::single_response("answer").with_context_window(window);
    run_agent_loop(
        &manifest(),
        query,
        &driver,
        &ToolRegistry::new(),
        &InMemorySubstrate::new(),
        None,
    )
    .await
}

/// FALSIFY-4600-001: the account is measured, not copied. Known SHA-256 vectors, and a whole
/// prompt in the request is consumed in full.
#[test]
fn falsify_4600_001_measure_hashes_the_consumed_bytes() {
    let a = InputAccount::measure("abc", &[Message::User("abc".into())], 1);
    assert_eq!(a.input_bytes_sent, 3);
    assert_eq!(a.input_bytes_consumed, 3);
    assert_eq!(a.prompt_sha256, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert_eq!(a.prompt_tokens, 1);
    assert!(a.is_complete());
}

/// FALSIFY-4600-002: a planted one-byte drop is RED. The request carries the prompt minus its
/// last byte; the turn is refused before the driver sees it, never answered with rc 0.
#[tokio::test]
async fn falsify_4600_002_planted_one_byte_drop_is_refused() {
    let query = prompt(70_678);
    let short = query[..query.len() - 1].to_string();
    let a = InputAccount::measure(&query, &[Message::User(short.clone())], 0);
    assert_eq!((a.input_bytes_sent, a.input_bytes_consumed), (70_678, 70_677));
    assert!(!a.is_complete());

    let driver = MockDriver::single_response("unreachable");
    let r =
        reason_step(&driver, &[Message::User(short)], &query, &[], &manifest(), "", &ctx(262_144))
            .await;
    assert!(
        matches!(r, Err(AgentError::InputNotConsumed { sent: 70_678, consumed: 70_677 })),
        "got {:?}",
        r.map(|(_, a)| a)
    );
}

/// FALSIFY-4600-003: an evicted prompt counts as nothing consumed. The request's last user
/// message is an older turn's, so not one byte of this prompt reached the driver.
#[test]
fn falsify_4600_003_an_evicted_prompt_consumed_nothing() {
    let a = InputAccount::measure(
        "the current prompt",
        &[Message::User("an older turn".into()), Message::Assistant("old answer".into())],
        0,
    );
    assert_eq!((a.input_bytes_sent, a.input_bytes_consumed), (18, 0));
    assert_eq!(a.prompt_sha256, SHA256_EMPTY);
    assert!(!a.is_complete());
}

/// FALSIFY-4600-004: the refusal is the envelope's named `input_not_consumed`, exit ≠ 0.
#[test]
fn falsify_4600_004_refusal_is_named_and_nonzero() {
    let e = AgentError::InputNotConsumed { sent: 10, consumed: 9 };
    let code = crate::agent::code_prompts::map_error_to_exit_code(&e);
    assert_ne!(code, 0, "a short request exits nonzero");
    let outcome = crate::agent::code_envelope::CodeOutcome::from_agent_error(&e, code);
    assert_eq!(outcome.kind, "input_not_consumed");
    assert_eq!(outcome.status, "failed");
    assert_eq!(outcome.exit_code, code);
}

/// FALSIFY-4600-005: at the window limit the prompt still goes through whole and says so. The
/// largest prompt the 32K window accepts (found by stepping down from one it refuses) reports
/// sent == consumed and the SHA-256 of the prompt itself.
#[tokio::test]
async fn falsify_4600_005_prompt_at_the_window_limit_is_consumed_whole() {
    use sha2::{Digest, Sha256};
    let mut bytes = 128 * 1024;
    assert!(matches!(run(&prompt(bytes), 32_768).await, Err(AgentError::ContextOverflow { .. })));
    let (query, result) = loop {
        bytes -= 256;
        let query = prompt(bytes);
        if let Ok(r) = run(&query, 32_768).await {
            break (query, r);
        }
    };
    let a = &result.input;
    assert_eq!(a.input_bytes_sent, query.len() as u64);
    assert_eq!(a.input_bytes_consumed, a.input_bytes_sent);
    assert_eq!(a.prompt_sha256, format!("{:x}", Sha256::digest(query.as_bytes())));
    assert_eq!(a.prompt_tokens, TokenEstimator::new().estimate(&query) as u64);
    assert!(a.prompt_tokens > 0);
}

/// FALSIFY-4600-006: the JSON result document carries the four fields, equal to the account.
#[tokio::test]
async fn falsify_4600_006_result_document_carries_the_account() {
    let result = run("What is 2+2?", 32_768).await.expect("fits");
    let doc: serde_json::Value = serde_json::from_str(&crate::agent::code_envelope::envelope(
        Some(&result),
        None,
        None,
        std::time::Duration::ZERO,
    ))
    .expect("json");
    assert_eq!(doc["input_bytes_sent"], 12);
    assert_eq!(doc["input_bytes_consumed"], 12);
    assert_eq!(doc["prompt_sha256"], result.input.prompt_sha256.as_str());
    assert_eq!(doc["prompt_tokens"], result.input.prompt_tokens);
}
