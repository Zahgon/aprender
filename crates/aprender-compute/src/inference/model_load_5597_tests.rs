//! Coverage for `ModelConfig::from_gguf`, `LlamaModel::from_gguf` and `load_weights` (#5597, #5600).
//!
//! A GGUF v3 image is assembled in memory with F32 tensors so every load path
//! runs without a model file.

use super::*;
use crate::inference::gguf::GgufFile;

const HIDDEN: usize = 4;
const HEADS: u32 = 2;
const KV_HEADS: u32 = 1;
const INTER: usize = 8;
const VOCAB: usize = 3;

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

enum Meta {
    U32(u32),
    F32(f32),
    Str(&'static str),
    StrArray(usize),
}

fn put_meta(out: &mut Vec<u8>, key: &str, v: &Meta) {
    put_str(out, key);
    match v {
        Meta::U32(x) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Meta::F32(x) => {
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Meta::Str(s) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            put_str(out, s);
        }
        Meta::StrArray(n) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(*n as u64).to_le_bytes());
            for i in 0..*n {
                put_str(out, &format!("t{i}"));
            }
        }
    }
}

/// F32 tensor named `name` with `n` elements, each equal to `fill`.
fn tensor(name: &str, n: usize, fill: f32) -> (String, usize, f32) {
    (name.to_string(), n, fill)
}

fn layer_tensors() -> Vec<(String, usize, f32)> {
    let kv = (KV_HEADS as usize) * (HIDDEN / HEADS as usize);
    vec![
        tensor("blk.0.attn_norm.weight", HIDDEN, 1.0),
        tensor("blk.0.ffn_norm.weight", HIDDEN, 1.0),
        tensor("blk.0.attn_q.weight", HIDDEN * HIDDEN, 0.5),
        tensor("blk.0.attn_k.weight", kv * HIDDEN, 0.25),
        tensor("blk.0.attn_v.weight", kv * HIDDEN, 0.125),
        tensor("blk.0.attn_output.weight", HIDDEN * HIDDEN, 0.5),
        tensor("blk.0.ffn_gate.weight", INTER * HIDDEN, 0.5),
        tensor("blk.0.ffn_up.weight", INTER * HIDDEN, 0.5),
        tensor("blk.0.ffn_down.weight", HIDDEN * INTER, 0.5),
    ]
}

fn base_tensors() -> Vec<(String, usize, f32)> {
    let mut t = vec![
        tensor("token_embd.weight", VOCAB * HIDDEN, 2.0),
        tensor("output_norm.weight", HIDDEN, 1.0),
    ];
    t.extend(layer_tensors());
    t
}

fn base_meta() -> Vec<(&'static str, Meta)> {
    vec![
        ("general.architecture", Meta::Str("llama")),
        ("llama.embedding_length", Meta::U32(HIDDEN as u32)),
        ("llama.attention.head_count", Meta::U32(HEADS)),
        ("llama.attention.head_count_kv", Meta::U32(KV_HEADS)),
        ("llama.block_count", Meta::U32(1)),
        ("llama.feed_forward_length", Meta::U32(INTER as u32)),
    ]
}

fn build(meta: &[(&str, Meta)], tensors: &[(String, usize, f32)]) -> GgufFile {
    let mut out = Vec::new();
    out.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    for (k, v) in meta {
        put_meta(&mut out, k, v);
    }
    let mut offset = 0u64;
    for (name, n, _) in tensors {
        put_str(&mut out, name);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(*n as u64).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // F32
        out.extend_from_slice(&offset.to_le_bytes());
        offset += (*n as u64) * 4;
    }
    while out.len() % 32 != 0 {
        out.push(0);
    }
    for (_, n, fill) in tensors {
        for _ in 0..*n {
            out.extend_from_slice(&fill.to_le_bytes());
        }
    }
    GgufFile::parse(out).expect("synthetic GGUF parses")
}

#[test]
fn config_reads_required_keys_and_applies_defaults() {
    let cfg = ModelConfig::from_gguf(&build(&base_meta(), &[])).expect("config");
    assert_eq!(cfg.arch, "llama");
    assert_eq!(cfg.hidden_size, HIDDEN);
    assert_eq!(cfg.num_heads, 2);
    assert_eq!(cfg.num_kv_heads, 1);
    assert_eq!(cfg.num_layers, 1);
    assert_eq!(cfg.intermediate_size, INTER);
    assert_eq!(cfg.head_dim, 2);
    assert_eq!(cfg.vocab_size, 32000);
    assert_eq!(cfg.max_seq_len, 2048);
    assert_eq!(cfg.rms_norm_eps, 1e-5);
    assert_eq!(cfg.rope_theta, 10000.0);
}

#[test]
fn config_kv_heads_default_to_head_count() {
    let meta: Vec<_> =
        base_meta().into_iter().filter(|(k, _)| !k.ends_with("head_count_kv")).collect();
    let cfg = ModelConfig::from_gguf(&build(&meta, &[])).expect("config");
    assert_eq!(cfg.num_kv_heads, cfg.num_heads);
}

#[test]
fn config_vocab_falls_back_to_token_array_length() {
    let mut meta = base_meta();
    meta.push(("tokenizer.ggml.tokens", Meta::StrArray(7)));
    let cfg = ModelConfig::from_gguf(&build(&meta, &[])).expect("config");
    assert_eq!(cfg.vocab_size, 7);
}

#[test]
fn config_explicit_vocab_and_optionals_win() {
    let mut meta = base_meta();
    meta.push(("tokenizer.ggml.vocab_size", Meta::U32(11)));
    meta.push(("tokenizer.ggml.tokens", Meta::StrArray(7)));
    meta.push(("llama.attention.layer_norm_rms_epsilon", Meta::F32(1e-6)));
    meta.push(("llama.rope.freq_base", Meta::F32(500000.0)));
    meta.push(("llama.context_length", Meta::U32(4096)));
    let cfg = ModelConfig::from_gguf(&build(&meta, &[])).expect("config");
    assert_eq!(cfg.vocab_size, 11);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.rope_theta, 500000.0);
    assert_eq!(cfg.max_seq_len, 4096);
}

#[test]
fn config_uses_architecture_prefix() {
    let meta = vec![
        ("general.architecture", Meta::Str("qwen2")),
        ("qwen2.embedding_length", Meta::U32(8)),
        ("qwen2.attention.head_count", Meta::U32(2)),
        ("qwen2.block_count", Meta::U32(3)),
        ("qwen2.feed_forward_length", Meta::U32(16)),
    ];
    let cfg = ModelConfig::from_gguf(&build(&meta, &[])).expect("config");
    assert_eq!(
        (cfg.arch.as_str(), cfg.hidden_size, cfg.num_layers, cfg.head_dim),
        ("qwen2", 8, 3, 4)
    );
}

#[test]
fn config_missing_required_key_names_the_key() {
    for (drop, want) in [
        ("llama.embedding_length", "embedding_length"),
        ("llama.attention.head_count", "head_count"),
        ("llama.block_count", "block_count"),
        ("llama.feed_forward_length", "feed_forward_length"),
    ] {
        let meta: Vec<_> = base_meta().into_iter().filter(|(k, _)| *k != drop).collect();
        let err = ModelConfig::from_gguf(&build(&meta, &[])).expect_err("missing key must fail");
        assert!(format!("{err}").contains(want), "{drop}: {err}");
    }
}

#[test]
fn from_gguf_loads_every_tensor_with_expected_shapes() {
    let model = LlamaModel::from_gguf(&build(&base_meta(), &base_tensors())).expect("model");
    let w = &model.weights;
    assert_eq!(model.config.num_layers, 1);
    assert_eq!(w.token_embd.len(), VOCAB * HIDDEN);
    assert!(w.token_embd.iter().all(|&x| x == 2.0));
    assert_eq!(w.output_norm, vec![1.0; HIDDEN]);
    assert_eq!(w.layers.len(), 1);
    let l = &w.layers[0];
    assert_eq!(l.attn_norm, vec![1.0; HIDDEN]);
    assert_eq!(l.q_weight.rows(), HIDDEN);
    assert_eq!(l.k_weight.rows(), 2); // kv_dim = 1 head * head_dim 2
    assert_eq!(l.v_weight.rows(), 2);
    assert_eq!(l.o_weight.rows(), HIDDEN);
    assert_eq!(l.gate_weight.rows(), INTER);
    assert_eq!(l.up_weight.rows(), INTER);
    assert_eq!(l.down_weight.rows(), HIDDEN);
    assert!(l.q_bias.is_none() && l.k_bias.is_none() && l.v_bias.is_none());
}

#[test]
fn missing_output_weight_ties_to_token_embeddings() {
    let mut meta = base_meta();
    meta.push(("tokenizer.ggml.vocab_size", Meta::U32(VOCAB as u32)));
    let model = LlamaModel::from_gguf(&build(&meta, &base_tensors())).expect("model");
    match &model.weights.output_weight {
        WeightMatrix::F32 { data, rows } => {
            assert_eq!(*rows, VOCAB);
            assert_eq!(data, &model.weights.token_embd);
        }
        WeightMatrix::Q4K { .. } => panic!("tied output must be F32"),
    }
}

#[test]
fn explicit_output_weight_is_loaded_not_tied() {
    let mut t = base_tensors();
    t.push(tensor("output.weight", VOCAB * HIDDEN, 7.0));
    let model = LlamaModel::from_gguf(&build(&base_meta(), &t)).expect("model");
    match &model.weights.output_weight {
        WeightMatrix::F32 { data, rows } => {
            assert_eq!(*rows, VOCAB);
            assert!(data.iter().all(|&x| x == 7.0));
        }
        WeightMatrix::Q4K { .. } => panic!("F32 tensor must stay F32"),
    }
}

#[test]
fn optional_attention_biases_are_loaded_when_present() {
    let mut t = base_tensors();
    t.push(tensor("blk.0.attn_q.bias", HIDDEN, 0.5));
    t.push(tensor("blk.0.attn_k.bias", 2, 0.25));
    t.push(tensor("blk.0.attn_v.bias", 2, 0.125));
    let model = LlamaModel::from_gguf(&build(&base_meta(), &t)).expect("model");
    let l = &model.weights.layers[0];
    assert_eq!(l.q_bias.as_deref(), Some(&[0.5; HIDDEN][..]));
    assert_eq!(l.k_bias.as_deref(), Some(&[0.25; 2][..]));
    assert_eq!(l.v_bias.as_deref(), Some(&[0.125; 2][..]));
}

#[test]
fn missing_required_tensor_is_an_error_naming_it() {
    for name in
        ["token_embd.weight", "output_norm.weight", "blk.0.attn_q.weight", "blk.0.ffn_down.weight"]
    {
        let t: Vec<_> = base_tensors().into_iter().filter(|(n, _, _)| n != name).collect();
        let err = LlamaModel::from_gguf(&build(&base_meta(), &t)).err().expect("must fail");
        let msg = format!("{err}");
        assert!(msg.contains("Missing tensor") && msg.contains(name), "{name}: {msg}");
    }
}

#[test]
fn from_gguf_propagates_config_error() {
    let meta: Vec<_> = base_meta().into_iter().filter(|(k, _)| *k != "llama.block_count").collect();
    let err = LlamaModel::from_gguf(&build(&meta, &base_tensors())).err().expect("must fail");
    assert!(format!("{err}").contains("block_count"));
}
