//! `ModelConfig::from_gguf` / `LlamaModel::from_gguf` metadata contract (#5600).
//!
//! Every file is built in memory: a GGUF v3 header, zero tensors, and the
//! metadata key/value pairs under test.

use trueno::inference::{GgufFile, LlamaModel, ModelConfig};
use trueno::TruenoError;

enum Kv<'a> {
    U32(u32),
    F32(f32),
    Str(&'a str),
    StrArray(usize),
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn build(pairs: &[(&str, Kv)]) -> GgufFile {
    let mut d = Vec::new();
    d.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    d.extend_from_slice(&3u32.to_le_bytes());
    d.extend_from_slice(&0u64.to_le_bytes());
    d.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
    for (k, v) in pairs {
        put_str(&mut d, k);
        match v {
            Kv::U32(x) => {
                d.extend_from_slice(&4u32.to_le_bytes());
                d.extend_from_slice(&x.to_le_bytes());
            }
            Kv::F32(x) => {
                d.extend_from_slice(&6u32.to_le_bytes());
                d.extend_from_slice(&x.to_le_bytes());
            }
            Kv::Str(s) => {
                d.extend_from_slice(&8u32.to_le_bytes());
                put_str(&mut d, s);
            }
            Kv::StrArray(n) => {
                d.extend_from_slice(&9u32.to_le_bytes());
                d.extend_from_slice(&8u32.to_le_bytes());
                d.extend_from_slice(&(*n as u64).to_le_bytes());
                for _ in 0..*n {
                    put_str(&mut d, "t");
                }
            }
        }
    }
    d.resize(d.len().div_ceil(32) * 32 + 32, 0);
    GgufFile::parse(d).expect("synthetic GGUF parses")
}

fn required<'a>() -> Vec<(&'a str, Kv<'a>)> {
    vec![
        ("llama.embedding_length", Kv::U32(64)),
        ("llama.attention.head_count", Kv::U32(8)),
        ("llama.block_count", Kv::U32(2)),
        ("llama.feed_forward_length", Kv::U32(128)),
    ]
}

fn err_msg(r: Result<ModelConfig, TruenoError>) -> String {
    match r {
        Err(TruenoError::InvalidInput(m)) => m,
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn required_keys_only_yield_documented_defaults() {
    let c = ModelConfig::from_gguf(&build(&required())).expect("config");
    assert_eq!(c.arch, "llama");
    assert_eq!((c.hidden_size, c.num_heads, c.num_layers, c.intermediate_size), (64, 8, 2, 128));
    assert_eq!(c.num_kv_heads, 8, "kv heads default to head_count");
    assert_eq!(c.head_dim, 8);
    assert_eq!(c.vocab_size, 32000);
    assert_eq!(c.rms_norm_eps, 1e-5);
    assert_eq!(c.rope_theta, 10000.0);
    assert_eq!(c.max_seq_len, 2048);
}

#[test]
fn optional_keys_override_defaults() {
    let mut p = required();
    p.push(("llama.attention.head_count_kv", Kv::U32(2)));
    p.push(("tokenizer.ggml.vocab_size", Kv::U32(500)));
    p.push(("llama.attention.layer_norm_rms_epsilon", Kv::F32(1e-6)));
    p.push(("llama.rope.freq_base", Kv::F32(500000.0)));
    p.push(("llama.context_length", Kv::U32(4096)));
    let c = ModelConfig::from_gguf(&build(&p)).expect("config");
    assert_eq!(c.num_kv_heads, 2);
    assert_eq!(c.vocab_size, 500);
    assert_eq!(c.rms_norm_eps, 1e-6);
    assert_eq!(c.rope_theta, 500000.0);
    assert_eq!(c.max_seq_len, 4096);
}

#[test]
fn architecture_key_selects_the_prefix() {
    let p = vec![
        ("general.architecture", Kv::Str("qwen2")),
        ("qwen2.embedding_length", Kv::U32(32)),
        ("qwen2.attention.head_count", Kv::U32(4)),
        ("qwen2.block_count", Kv::U32(1)),
        ("qwen2.feed_forward_length", Kv::U32(64)),
    ];
    let c = ModelConfig::from_gguf(&build(&p)).expect("config");
    assert_eq!(c.arch, "qwen2");
    assert_eq!(c.head_dim, 8);
    // llama.* keys are not consulted for a qwen2 file
    let mut q = required();
    q.push(("general.architecture", Kv::Str("qwen2")));
    assert!(err_msg(ModelConfig::from_gguf(&build(&q))).contains("embedding_length"));
}

#[test]
fn vocab_falls_back_to_token_array_length_then_explicit_wins() {
    let mut p = required();
    p.push(("tokenizer.ggml.tokens", Kv::StrArray(7)));
    assert_eq!(ModelConfig::from_gguf(&build(&p)).unwrap().vocab_size, 7);
    p.push(("tokenizer.ggml.vocab_size", Kv::U32(9)));
    assert_eq!(ModelConfig::from_gguf(&build(&p)).unwrap().vocab_size, 9);
}

#[test]
fn each_missing_required_key_names_itself() {
    for (i, want) in
        ["embedding_length", "head_count", "block_count", "feed_forward_length"].iter().enumerate()
    {
        let mut p = required();
        p.remove(i);
        let m = err_msg(ModelConfig::from_gguf(&build(&p)));
        assert_eq!(m, format!("Missing {want} in GGUF"));
    }
}

#[test]
fn llama_model_from_gguf_without_tensors_is_an_error() {
    assert!(LlamaModel::from_gguf(&build(&required())).is_err());
    // and config errors propagate through the model loader unchanged
    let mut p = required();
    p.remove(0);
    assert!(LlamaModel::from_gguf(&build(&p)).is_err());
}
