//! #5597: behavioural tests for `load_weights` (private), driven through
//! `LlamaModel::from_gguf` over a tiny GGUF built in memory.

#![allow(clippy::disallowed_methods)]

use trueno::inference::model::WeightMatrix;
use trueno::inference::{GgufFile, LlamaModel};

const F32: u32 = 0;
const F16: u32 = 1;
const Q8_0: u32 = 8;
const HIDDEN: u64 = 4;
const FFN: u64 = 8;
const VOCAB: u64 = 3;

struct Tensor {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    bytes: Vec<u8>,
}

fn f32_tensor(name: &str, dims: &[u64], values: Vec<f32>) -> Tensor {
    Tensor {
        name: name.to_string(),
        dims: dims.to_vec(),
        dtype: F32,
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn filled(name: &str, dims: &[u64], v: f32) -> Tensor {
    let n: u64 = dims.iter().product();
    f32_tensor(name, dims, vec![v; n as usize])
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_u32_kv(out: &mut Vec<u8>, key: &str, v: u32) {
    put_str(out, key);
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&v.to_le_bytes());
}

fn build_gguf(layers: u32, tensors: &[Tensor]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&7u64.to_le_bytes());
    put_u32_kv(&mut out, "llama.embedding_length", HIDDEN as u32);
    put_u32_kv(&mut out, "llama.attention.head_count", 2);
    put_u32_kv(&mut out, "llama.attention.head_count_kv", 1);
    put_u32_kv(&mut out, "llama.block_count", layers);
    put_u32_kv(&mut out, "llama.feed_forward_length", FFN as u32);
    put_u32_kv(&mut out, "tokenizer.ggml.vocab_size", VOCAB as u32);
    put_u32_kv(&mut out, "general.alignment", 32);
    let mut offset = 0u64;
    for t in tensors {
        put_str(&mut out, &t.name);
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.dtype.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += t.bytes.len() as u64;
    }
    while out.len() % 32 != 0 {
        out.push(0);
    }
    for t in tensors {
        out.extend_from_slice(&t.bytes);
    }
    out
}

/// Layer tensors: attn_v is F16 (1.0), ffn_up is one Q8_0 block (scale 1.0, quants 0..32).
fn layer_tensors(i: u32, with_bias: bool) -> Vec<Tensor> {
    let p = format!("blk.{i}");
    let mut q8 = vec![0x00, 0x3C];
    q8.extend((0..32u8).map(|q| q));
    let mut v = vec![
        filled(&format!("{p}.attn_norm.weight"), &[HIDDEN], 0.25),
        filled(&format!("{p}.ffn_norm.weight"), &[HIDDEN], 0.75),
        filled(&format!("{p}.attn_q.weight"), &[HIDDEN, 4], 0.5),
        filled(&format!("{p}.attn_k.weight"), &[HIDDEN, 2], 0.5),
        Tensor {
            name: format!("{p}.attn_v.weight"),
            dims: vec![HIDDEN, 2],
            dtype: F16,
            bytes: [0x00u8, 0x3C].repeat(8),
        },
        filled(&format!("{p}.attn_output.weight"), &[HIDDEN, 4], 0.5),
        filled(&format!("{p}.ffn_gate.weight"), &[HIDDEN, FFN], 0.5),
        Tensor {
            name: format!("{p}.ffn_up.weight"),
            dims: vec![HIDDEN, FFN],
            dtype: Q8_0,
            bytes: q8,
        },
        filled(&format!("{p}.ffn_down.weight"), &[FFN, HIDDEN], 0.5),
    ];
    if with_bias {
        v.push(filled(&format!("{p}.attn_q.bias"), &[HIDDEN], 1.5));
        v.push(filled(&format!("{p}.attn_k.bias"), &[2], 2.5));
        v.push(filled(&format!("{p}.attn_v.bias"), &[2], 3.5));
    }
    v
}

fn base_tensors(layers: u32, with_output: bool, with_bias: bool) -> Vec<Tensor> {
    let mut v = vec![
        f32_tensor("token_embd.weight", &[HIDDEN, VOCAB], (1..=12).map(|x| x as f32).collect()),
        filled("output_norm.weight", &[HIDDEN], 0.5),
    ];
    if with_output {
        v.push(filled("output.weight", &[HIDDEN, VOCAB], 9.0));
    }
    for i in 0..layers {
        v.extend(layer_tensors(i, with_bias));
    }
    v
}

fn load(layers: u32, tensors: &[Tensor]) -> Result<LlamaModel, String> {
    let gguf = GgufFile::parse(build_gguf(layers, tensors)).map_err(|e| e.to_string())?;
    LlamaModel::from_gguf(&gguf).map_err(|e| e.to_string())
}

fn f32_rows(w: &WeightMatrix) -> (&[f32], usize) {
    match w {
        WeightMatrix::F32 { data, rows } => (data, *rows),
        WeightMatrix::Q4K { .. } => panic!("expected F32 weights"),
    }
}

#[test]
fn loads_embeddings_norm_and_every_layer() {
    let m = load(2, &base_tensors(2, true, true)).expect("model loads");
    let expected: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    assert_eq!(m.weights.token_embd, expected);
    assert_eq!(m.weights.output_norm, vec![0.5; 4]);
    assert_eq!(m.weights.layers.len(), 2);
    for l in &m.weights.layers {
        assert_eq!(l.attn_norm, vec![0.25; 4]);
        assert_eq!(l.ffn_norm, vec![0.75; 4]);
        assert_eq!(f32_rows(&l.q_weight).1, 4);
        assert_eq!(f32_rows(&l.k_weight).1, 2);
        assert_eq!(f32_rows(&l.o_weight).1, 4);
        assert_eq!(f32_rows(&l.gate_weight).1, 8);
        assert_eq!(f32_rows(&l.down_weight).1, 4);
    }
}

#[test]
fn explicit_output_weight_is_loaded_not_tied() {
    let m = load(1, &base_tensors(1, true, false)).expect("model loads");
    let (data, rows) = f32_rows(&m.weights.output_weight);
    assert_eq!(rows, 3);
    assert_eq!(data, &[9.0f32; 12][..]);
}

#[test]
fn missing_output_weight_falls_back_to_tied_embeddings() {
    let m = load(1, &base_tensors(1, false, false)).expect("model loads");
    let (data, rows) = f32_rows(&m.weights.output_weight);
    assert_eq!(rows, 3, "tied head uses vocab_size rows");
    assert_eq!(data, &m.weights.token_embd[..]);
}

#[test]
fn per_tensor_dtypes_are_dequantized_to_f32() {
    let m = load(1, &base_tensors(1, true, false)).expect("model loads");
    let l = &m.weights.layers[0];
    let (v, v_rows) = f32_rows(&l.v_weight);
    assert_eq!((v_rows, v), (2, &[1.0f32; 8][..]), "F16 1.0 -> 1.0");
    let (up, up_rows) = f32_rows(&l.up_weight);
    assert_eq!(up_rows, 8);
    let expected: Vec<f32> = (0..32).map(|q| q as f32).collect();
    assert_eq!(up, &expected[..], "Q8_0 scale 1.0 times quants 0..32");
}

#[test]
fn attention_biases_load_when_present_and_are_none_otherwise() {
    let with = load(1, &base_tensors(1, true, true)).expect("model loads");
    let l = &with.weights.layers[0];
    assert_eq!(l.q_bias.as_deref(), Some(&[1.5f32; 4][..]));
    assert_eq!(l.k_bias.as_deref(), Some(&[2.5f32; 2][..]));
    assert_eq!(l.v_bias.as_deref(), Some(&[3.5f32; 2][..]));
    let without = load(1, &base_tensors(1, true, false)).expect("model loads");
    let l = &without.weights.layers[0];
    assert!(l.q_bias.is_none() && l.k_bias.is_none() && l.v_bias.is_none());
}

#[test]
fn each_missing_required_tensor_names_itself_in_the_error() {
    for missing in [
        "token_embd.weight",
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ] {
        let tensors: Vec<Tensor> =
            base_tensors(1, true, false).into_iter().filter(|t| t.name != missing).collect();
        let err = match load(1, &tensors) {
            Ok(_) => panic!("{missing} absent must fail"),
            Err(e) => e,
        };
        assert!(err.contains(&format!("Missing tensor: {missing}")), "{missing}: {err}");
    }
}

#[test]
fn second_layer_missing_fails_even_though_layer_zero_loads() {
    let mut tensors = base_tensors(1, true, false);
    tensors.extend(
        layer_tensors(1, false).into_iter().filter(|t| !t.name.ends_with("ffn_down.weight")),
    );
    let err = match load(2, &tensors) {
        Ok(_) => panic!("layer 1 ffn_down absent must fail"),
        Err(e) => e,
    };
    assert!(err.contains("Missing tensor: blk.1.ffn_down.weight"), "{err}");
}
