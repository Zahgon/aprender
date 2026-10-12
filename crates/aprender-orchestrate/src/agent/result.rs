//! Agent loop result and error types.
//!
//! Defines the outcome of a complete agent loop invocation
//! and the error taxonomy for agent failures.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::capability::Capability;

/// Outcome of a complete agent loop invocation.
#[derive(Debug, Clone, Serialize)]
pub struct AgentLoopResult {
    /// Final text response from the agent.
    pub text: String,
    /// Token usage across all iterations.
    pub usage: TokenUsage,
    /// Number of loop iterations executed.
    pub iterations: u32,
    /// Number of tool calls made.
    pub tool_calls: u32,
    /// L22 (#4600): how much of the turn's prompt reached the driver.
    pub input: InputAccount,
}

/// L22 (#4600): a turn's proof that the driver was handed its whole prompt.
///
/// Measured on the request itself, not assumed from the prompt: a result that
/// exits 0 without `input_bytes_consumed == input_bytes_sent` proves nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct InputAccount {
    /// Bytes of the prompt the turn was asked to answer.
    pub input_bytes_sent: u64,
    /// Bytes of that prompt present in the request handed to the driver.
    pub input_bytes_consumed: u64,
    /// SHA-256 (lowercase hex) of the consumed bytes.
    pub prompt_sha256: String,
    /// Tokens of the consumed prompt, by the counter that sizes the window.
    pub prompt_tokens: u64,
}

impl InputAccount {
    /// Account for `prompt` against the messages of a driver request.
    ///
    /// The consumed bytes are the request's last `User` message when that message
    /// is part of `prompt`, else nothing: an evicted or shortened prompt can never
    /// count as consumed.
    pub fn measure(prompt: &str, request: &[super::driver::Message], prompt_tokens: u64) -> Self {
        use sha2::{Digest, Sha256};
        let consumed = request
            .iter()
            .rev()
            .find_map(|m| match m {
                super::driver::Message::User(text) => Some(text.as_str()),
                _ => None,
            })
            .filter(|text| prompt.contains(*text))
            .unwrap_or("");
        Self {
            input_bytes_sent: prompt.len() as u64,
            input_bytes_consumed: consumed.len() as u64,
            prompt_sha256: format!("{:x}", Sha256::digest(consumed.as_bytes())),
            prompt_tokens,
        }
    }

    /// The driver was handed every byte of the prompt.
    pub fn is_complete(&self) -> bool {
        self.input_bytes_consumed == self.input_bytes_sent
    }
}

/// Token usage counters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Total input tokens across all completions.
    pub input_tokens: u64,
    /// Total output tokens across all completions.
    pub output_tokens: u64,
}

impl TokenUsage {
    /// Accumulate usage from another completion.
    pub fn accumulate(&mut self, other: &Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }

    /// Total tokens (input + output).
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// Stop reason from a single LLM completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Model finished naturally.
    EndTurn,
    /// Model wants to use a tool.
    ToolUse,
    /// Output truncated at `max_tokens` limit.
    MaxTokens,
    /// Hit a stop sequence.
    StopSequence,
}

/// Agent error taxonomy.
///
/// Classified by recoverability: some errors are retryable,
/// others are fatal. The agent loop uses this to decide whether
/// to retry or terminate (Jidoka: stop on defect).
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// LLM driver error (may be retryable).
    #[error("driver error: {0}")]
    Driver(#[from] DriverError),
    /// Tool execution failed.
    #[error("tool '{tool_name}' failed: {message}")]
    ToolExecution {
        /// Name of the failed tool.
        tool_name: String,
        /// Error message.
        message: String,
    },
    /// Capability denied (Poka-Yoke).
    #[error("capability denied for tool '{tool_name}': requires {required:?}")]
    CapabilityDenied {
        /// Name of the denied tool.
        tool_name: String,
        /// Required capability that was not granted.
        required: Capability,
    },
    /// Loop guard triggered (Jidoka).
    #[error("circuit break: {0}")]
    CircuitBreak(String),
    /// Max iterations reached.
    #[error("max iterations reached")]
    MaxIterationsReached,
    /// Context overflow after truncation.
    #[error("context overflow: required {required} tokens, available {available}")]
    ContextOverflow {
        /// Tokens required.
        required: usize,
        /// Tokens available.
        available: usize,
    },
    /// L22 (#4600): the driver request carried less of the prompt than was sent.
    #[error("input not consumed: the driver was handed {consumed} of the prompt's {sent} bytes")]
    InputNotConsumed {
        /// Bytes of the prompt.
        sent: u64,
        /// Bytes of the prompt in the driver request.
        consumed: u64,
    },
    /// Manifest parsing error.
    #[error("manifest error: {0}")]
    ManifestError(String),
    /// Memory substrate error.
    #[error("memory error: {0}")]
    Memory(String),
}

/// LLM driver-specific errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum DriverError {
    /// Remote API rate limited. Retryable with backoff.
    #[error("rate limited, retry after {retry_after_ms}ms")]
    RateLimited {
        /// Suggested wait time in milliseconds.
        retry_after_ms: u64,
    },
    /// Remote API overloaded. Retryable with backoff.
    #[error("overloaded, retry after {retry_after_ms}ms")]
    Overloaded {
        /// Suggested wait time in milliseconds.
        retry_after_ms: u64,
    },
    /// Model file not found. Not retryable.
    #[error("model not found: {0}")]
    ModelNotFound(PathBuf),
    /// Inference failed. Not retryable.
    #[error("inference failed: {0}")]
    InferenceFailed(String),
    /// Network error (remote driver). Retryable.
    #[error("network error: {0}")]
    Network(String),
}

impl DriverError {
    /// Whether this error is retryable with backoff.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::RateLimited { .. } | Self::Overloaded { .. } | Self::Network(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_usage_accumulate() {
        let mut total = TokenUsage::default();
        total.accumulate(&TokenUsage { input_tokens: 100, output_tokens: 50 });
        total.accumulate(&TokenUsage { input_tokens: 200, output_tokens: 75 });
        assert_eq!(total.input_tokens, 300);
        assert_eq!(total.output_tokens, 125);
        assert_eq!(total.total(), 425);
    }

    #[test]
    fn test_token_usage_default_zero() {
        let usage = TokenUsage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.total(), 0);
    }

    #[test]
    fn test_stop_reason_equality() {
        assert_eq!(StopReason::EndTurn, StopReason::EndTurn);
        assert_ne!(StopReason::EndTurn, StopReason::ToolUse);
    }

    #[test]
    fn test_driver_error_retryable() {
        assert!(DriverError::RateLimited { retry_after_ms: 1000 }.is_retryable());
        assert!(DriverError::Overloaded { retry_after_ms: 500 }.is_retryable());
        assert!(DriverError::Network("timeout".into()).is_retryable());
        assert!(!DriverError::ModelNotFound("/tmp/missing.gguf".into()).is_retryable());
        assert!(!DriverError::InferenceFailed("oom".into()).is_retryable());
    }

    #[test]
    fn test_agent_error_display() {
        let err = AgentError::CircuitBreak("cost exceeded".into());
        assert_eq!(err.to_string(), "circuit break: cost exceeded");

        let err = AgentError::MaxIterationsReached;
        assert_eq!(err.to_string(), "max iterations reached");

        let err = AgentError::ToolExecution {
            tool_name: "rag".into(),
            message: "index not found".into(),
        };
        assert!(err.to_string().contains("rag"));
    }

    #[test]
    fn test_agent_loop_result_serialize() {
        let result = AgentLoopResult {
            text: "hello".into(),
            usage: TokenUsage { input_tokens: 10, output_tokens: 5 },
            iterations: 2,
            tool_calls: 1,
            input: InputAccount::default(),
        };
        let json = serde_json::to_string(&result).expect("serialize failed");
        assert!(json.contains("\"text\":\"hello\""));
        assert!(json.contains("\"iterations\":2"));
    }

    #[test]
    fn test_stop_reason_serialization() {
        let reasons = vec![
            StopReason::EndTurn,
            StopReason::ToolUse,
            StopReason::MaxTokens,
            StopReason::StopSequence,
        ];
        for r in &reasons {
            let json = serde_json::to_string(r).expect("serialize failed");
            let back: StopReason = serde_json::from_str(&json).expect("deserialize failed");
            assert_eq!(*r, back);
        }
    }
}
