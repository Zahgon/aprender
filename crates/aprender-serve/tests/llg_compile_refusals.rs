//! `LlgEnv::compile` refusals (#5647), reached through the public `lark` and `json_schema`.
//!
//! Contract: `contracts/serve-llg-compile-refusals-v1.yaml`.
#![cfg(feature = "structured-output")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use realizar::constrain::{ConstraintError, ConstraintVocab, LlgEnv, TokenConstraint};

fn env() -> LlgEnv {
    let mut token_bytes = vec![b"<eos>".to_vec()];
    let mut special = vec![true];
    for b in 0..=255u8 {
        token_bytes.push(vec![b]);
        special.push(false);
    }
    LlgEnv::new(&ConstraintVocab {
        token_bytes,
        special,
        eos: 0,
    })
    .expect("env")
}

/// The refusal text of a compile failure, which must be `SchemaUnsupported`.
fn refusal<T>(r: Result<T, ConstraintError>) -> String {
    match r {
        Err(ConstraintError::SchemaUnsupported(why)) => why,
        Err(other) => panic!("wrong variant: {other:?}"),
        Ok(_) => panic!("compiled, expected a refusal"),
    }
}

#[test]
fn lark_unknown_rule_is_refused_with_engine_message_and_label() {
    let why = refusal(env().lark("start: foo"));
    assert!(
        why.starts_with("the Lark grammar does not compile for constrained decoding: "),
        "{why}"
    );
    assert!(why.contains("unknown name"), "{why}");
}

#[test]
fn lark_bad_regex_and_lexer_errors_are_refused() {
    for (grammar, needle) in [
        ("start: /(/", "unclosed group"),
        ("start: /(?=a)a/", "look-around"),
        ("%%%", "lexer error"),
    ] {
        let why = refusal(env().lark(grammar));
        assert!(why.contains("does not compile"), "{grammar}: {why}");
        assert!(why.contains(needle), "{grammar}: {why}");
    }
}

#[test]
fn json_schema_compile_failures_say_json_schema_not_lark() {
    for (schema, needle) in [
        (serde_json::json!({"type": "nope"}), "Invalid type"),
        (serde_json::json!({"pattern": "("}), "unclosed group"),
        (serde_json::json!({"$ref": "#/missing"}), "not found"),
    ] {
        let why = refusal(env().json_schema(&schema));
        assert!(
            why.starts_with("the JSON schema does not compile for constrained decoding: "),
            "{schema}: {why}"
        );
        assert!(why.contains(needle), "{schema}: {why}");
    }
}

#[test]
fn a_grammar_that_compiles_yields_a_fresh_constraint_at_position_zero() {
    let mut c = env().lark("start: \"a\"").expect("compiles");
    assert!(!c.is_complete());
    let mut logits = vec![0.0f32; 257];
    c.mask(&mut logits).expect("mask");
    let allowed: Vec<usize> = (0..logits.len())
        .filter(|&i| logits[i].is_finite())
        .collect();
    assert_eq!(allowed, vec![1 + usize::from(b'a')]);
    c.accept(1 + u32::from(b'a')).expect("accept a");
    assert!(c.is_complete());
}

#[test]
fn a_failed_compile_does_not_poison_the_env() {
    let env = env();
    refusal(env.lark("start: foo"));
    assert!(env.lark("start: \"b\"").is_ok());
}
