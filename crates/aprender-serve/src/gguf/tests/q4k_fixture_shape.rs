//! #4985: `build_minimal_llama_gguf` declares each Q4_K tensor as `[out_dim, in_dim]`.
//!
//! `GGUFBuilder` takes dims in the order the parser hands the loader (`[rows, cols]`, i.e.
//! `[out_dim, in_dim]`) and writes them reversed as GGUF `ne0, ne1`. The fixture declared
//! the non-square tensors transposed (`ffn_up` as `[hidden, intermediate]`), so the loader
//! sized `ffn_up` as `hidden` rows and the forward refused it once `intermediate != hidden`.
//! Tests that needed a Q4_K model worked around it with intermediate = hidden.

use std::io::Write;

use crate::gguf::model::OwnedQuantizedModel;
use crate::gguf::test_factory::build_minimal_llama_gguf;
use crate::gguf::types::{GGUF_ALIGNMENT, GGUF_TYPE_Q4_K};
use crate::gguf::{GGUFModel, MappedGGUFModel};

fn load(gguf: &[u8]) -> (tempfile::NamedTempFile, MappedGGUFModel) {
    let mut f = tempfile::NamedTempFile::new().expect("create temp gguf");
    f.write_all(gguf).expect("write gguf bytes");
    f.flush().expect("flush gguf bytes");
    let mapped = MappedGGUFModel::from_path(f.path()).expect("map gguf");
    (f, mapped)
}

/// The shape the issue names: a non-square FFN (hidden 64, intermediate 128) loads and
/// runs one forward, and the logits cover the vocabulary.
#[test]
fn a_non_square_ffn_q4k_model_runs_a_forward() {
    let (_f, mapped) = load(&build_minimal_llama_gguf(32, 64, 128, 4, 4));
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("load");
    let logits = model.forward(&[1]).expect("forward");
    assert_eq!(logits.len(), 32);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// GQA (kv_dim 32 < hidden 64) makes K and V non-square too; each is declared from its own
/// rows, so the forward runs here as well.
#[test]
fn a_gqa_q4k_model_runs_a_forward() {
    let (_f, mapped) = load(&build_minimal_llama_gguf(32, 64, 128, 4, 2));
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("load");
    let logits = model.forward(&[1, 2]).expect("forward");
    assert_eq!(logits.len(), 32);
}

/// Each Q4_K tensor's parsed shape is `[out_dim, in_dim]` for its role, and it carries
/// `out_dim` rows of `ceil(in_dim / 256)` super-blocks of 144 bytes: the gap to the next
/// tensor (or the end of the data) is that size plus less than one alignment of padding.
#[test]
fn every_q4k_tensor_is_out_by_in_and_holds_its_rows() {
    let (hidden, inter, kv) = (64u64, 128u64, 32u64);
    let bytes = build_minimal_llama_gguf(32, 64, 128, 4, 2);
    let model = GGUFModel::from_bytes(&bytes).expect("parse");
    let data_len = (bytes.len() - model.tensor_data_start) as u64;
    let mut starts: Vec<u64> = model.tensors.iter().map(|t| t.offset).collect();
    starts.push(data_len);
    starts.sort_unstable();
    let want_shape = |name: &str| match name {
        "blk.0.attn_q.weight" | "blk.0.attn_output.weight" => [hidden, hidden],
        "blk.0.attn_k.weight" | "blk.0.attn_v.weight" => [kv, hidden],
        "blk.0.ffn_up.weight" | "blk.0.ffn_gate.weight" => [inter, hidden],
        "blk.0.ffn_down.weight" => [hidden, inter],
        other => panic!("unexpected Q4_K tensor {other}"),
    };
    let q4k: Vec<_> = model
        .tensors
        .iter()
        .filter(|t| t.qtype == GGUF_TYPE_Q4_K)
        .collect();
    assert_eq!(q4k.len(), 7, "four attention and three FFN tensors");
    for t in q4k {
        assert_eq!(
            t.dims,
            want_shape(&t.name),
            "{} is [out_dim, in_dim]",
            t.name
        );
        let (rows, cols) = (t.dims[0], t.dims[1]);
        let want = rows * cols.div_ceil(256) * 144;
        let next = *starts
            .iter()
            .find(|&&s| s > t.offset)
            .expect("a next start");
        let gap = next - t.offset;
        assert!(
            gap >= want && gap < want + GGUF_ALIGNMENT as u64,
            "{} [{rows}, {cols}] holds {gap} bytes, its rows need {want}",
            t.name
        );
    }
}
