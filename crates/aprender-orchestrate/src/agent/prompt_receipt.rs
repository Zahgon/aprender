//! #5190 (L22 slice 1): how much of a turn's prompt reached the driver.
//!
//! #4601 refuses an over-budget prompt instead of dropping it, but nothing proved
//! that the whole prompt was in the request a Pass was scored on. Each turn now
//! measures its prompt in the [`CompletionRequest`] it hands to the driver, and a
//! request that carries fewer bytes than were sent ends the turn with
//! [`AgentError::InputNotConsumed`] instead of being answered.

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::driver::{CompletionRequest, Message};
use super::result::AgentError;
use crate::serve::context::ContextManager;

/// The four numbers one turn records about its prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PromptReceipt {
    /// Bytes of the prompt the turn was asked.
    pub input_bytes_sent: usize,
    /// Bytes of that prompt present in the request handed to the driver.
    pub input_bytes_consumed: usize,
    /// SHA-256 (lowercase hex) of the consumed bytes.
    pub prompt_sha256: String,
    /// Tokens of the consumed bytes, by the counter that sizes the window.
    pub prompt_tokens: usize,
}

impl PromptReceipt {
    /// The receipt as the four top-level JSON fields of a result document or a
    /// `session_end` record.
    pub fn json_fields(&self) -> [(&'static str, serde_json::Value); 4] {
        [
            ("input_bytes_sent", self.input_bytes_sent.into()),
            ("input_bytes_consumed", self.input_bytes_consumed.into()),
            ("prompt_sha256", self.prompt_sha256.clone().into()),
            ("prompt_tokens", self.prompt_tokens.into()),
        ]
    }
}

/// Measure `prompt` in `request`: the turn's prompt is the request's last `User`
/// message. It is consumed whole when that message holds it; otherwise only the
/// prefix the two share counts, so a byte dropped anywhere is a short count.
pub(super) fn measure(
    prompt: &str,
    request: &CompletionRequest,
    context: &ContextManager,
) -> PromptReceipt {
    let carried = request.messages.iter().rev().find_map(|m| match m {
        Message::User(s) => Some(s.as_str()),
        _ => None,
    });
    let consumed = match carried {
        Some(s) if s.contains(prompt) => prompt.len(),
        Some(s) => prompt.bytes().zip(s.bytes()).take_while(|(a, b)| a == b).count(),
        None => 0,
    };
    let bytes = &prompt.as_bytes()[..consumed];
    let tokens_of = String::from_utf8_lossy(bytes).into_owned();
    PromptReceipt {
        input_bytes_sent: prompt.len(),
        input_bytes_consumed: consumed,
        prompt_sha256: format!("{:x}", Sha256::digest(bytes)),
        prompt_tokens: context.estimate_tokens(&[Message::User(tokens_of).to_chat_message()]),
    }
}

/// [`measure`], refusing the turn when consumed < sent.
pub(super) fn check(
    prompt: &str,
    request: &CompletionRequest,
    context: &ContextManager,
) -> Result<PromptReceipt, AgentError> {
    let receipt = measure(prompt, request, context);
    if receipt.input_bytes_consumed < receipt.input_bytes_sent {
        return Err(AgentError::InputNotConsumed(receipt));
    }
    Ok(receipt)
}

/// The falsifier's planted defect: one byte dropped from the turn's prompt between
/// the loop and the driver. Armed per thread, so only the test that arms it sees it.
#[cfg(test)]
pub(super) mod plant {
    use std::cell::Cell;

    use super::{CompletionRequest, Message};

    thread_local! { static ARMED: Cell<bool> = const { Cell::new(false) }; }

    /// Arm (or disarm) the drop for the requests this thread builds next.
    pub(crate) fn arm(on: bool) {
        ARMED.with(|a| a.set(on));
    }

    pub(crate) fn drop_one_byte(mut request: CompletionRequest) -> CompletionRequest {
        if ARMED.with(Cell::get) {
            if let Some(Message::User(s)) =
                request.messages.iter_mut().rev().find(|m| matches!(m, Message::User(_)))
            {
                s.pop();
            }
        }
        request
    }
}

#[cfg(test)]
#[path = "prompt_receipt_tests.rs"]
mod tests;
