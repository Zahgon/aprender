//! #5719 — `ClassifyPipeline::from_apr` sibling-tokenizer contract.
//!
//! Training requires a BPE tokenizer, so `from_apr` must (a) load the
//! `<stem>.tokenizer.json` next to the `.apr`, (b) refuse when it is absent,
//! and (c) refuse when it is unparseable — each with a distinct error.

use super::super::*;
use aprender::serialization::apr::AprWriter;

const TOKENIZER_JSON: &str = r#"{
    "model": {"vocab": {"hello": 0, "world": 1, "<|endoftext|>": 2}, "merges": []},
    "added_tokens": []
}"#;

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("from_apr_5719_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// A complete tiny Qwen2-shaped `.apr` with F32 weights.
fn write_apr(path: &Path, c: &TransformerConfig) {
    let (h, v, i) = (c.hidden_size, c.vocab_size, c.intermediate_size);
    let (q, kv) = (c.q_dim(), c.num_kv_heads * c.head_dim());
    let mut w = AprWriter::new();
    w.add_tensor_f32("model.embed_tokens.weight", vec![v, h], &vec![0.5; v * h]);
    w.add_tensor_f32("model.norm.weight", vec![h], &vec![1.0; h]);
    w.add_tensor_f32("lm_head.weight", vec![v, h], &vec![0.25; v * h]);
    for l in 0..c.num_hidden_layers {
        let p = format!("model.layers.{l}");
        w.add_tensor_f32(&format!("{p}.input_layernorm.weight"), vec![h], &vec![1.0; h]);
        w.add_tensor_f32(&format!("{p}.post_attention_layernorm.weight"), vec![h], &vec![1.0; h]);
        w.add_tensor_f32(&format!("{p}.self_attn.q_proj.weight"), vec![q, h], &vec![0.01; q * h]);
        w.add_tensor_f32(&format!("{p}.self_attn.k_proj.weight"), vec![kv, h], &vec![0.02; kv * h]);
        w.add_tensor_f32(&format!("{p}.self_attn.v_proj.weight"), vec![kv, h], &vec![0.03; kv * h]);
        w.add_tensor_f32(&format!("{p}.self_attn.o_proj.weight"), vec![h, q], &vec![0.04; h * q]);
        w.add_tensor_f32(&format!("{p}.mlp.gate_proj.weight"), vec![i, h], &vec![0.05; i * h]);
        w.add_tensor_f32(&format!("{p}.mlp.up_proj.weight"), vec![i, h], &vec![0.06; i * h]);
        w.add_tensor_f32(&format!("{p}.mlp.down_proj.weight"), vec![h, i], &vec![0.07; h * i]);
    }
    std::fs::write(path, w.to_bytes().expect("serialize apr")).expect("write apr");
}

fn config() -> ClassifyConfig {
    ClassifyConfig { num_classes: 3, lora_rank: 4, lora_alpha: 4.0, ..ClassifyConfig::default() }
}

#[test]
fn from_apr_5719_loads_sibling_tokenizer() {
    let dir = scratch("ok");
    let mc = TransformerConfig::tiny();
    let apr = dir.join("m.apr");
    write_apr(&apr, &mc);
    std::fs::write(dir.join("m.tokenizer.json"), TOKENIZER_JSON).expect("write tokenizer");

    let p = ClassifyPipeline::from_apr(&apr, &mc, config()).expect("from_apr with sibling");
    assert_eq!(p.classifier.num_classes(), 3);
    assert!(!p.lora_layers.is_empty(), "LoRA layers must be built");
    let tok = p.tokenizer.as_ref().expect("sibling tokenizer must be loaded");
    assert_eq!(tok.vocab_size(), 3);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn from_apr_5719_missing_sibling_tokenizer_is_config_error() {
    let dir = scratch("missing");
    let mc = TransformerConfig::tiny();
    let apr = dir.join("m.apr");
    write_apr(&apr, &mc);

    let Err(err) = ClassifyPipeline::from_apr(&apr, &mc, config()) else {
        panic!("a missing sibling tokenizer must be refused");
    };
    let msg = format!("{err}");
    assert!(matches!(err, crate::Error::ConfigError(_)), "{msg}");
    assert!(msg.contains("No sibling tokenizer") && msg.contains("m.tokenizer.json"), "{msg}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn from_apr_5719_corrupt_sibling_tokenizer_is_config_error() {
    let dir = scratch("corrupt");
    let mc = TransformerConfig::tiny();
    let apr = dir.join("m.apr");
    write_apr(&apr, &mc);
    std::fs::write(dir.join("m.tokenizer.json"), "not json").expect("write tokenizer");

    let Err(err) = ClassifyPipeline::from_apr(&apr, &mc, config()) else {
        panic!("an unparseable sibling tokenizer must be refused");
    };
    let msg = format!("{err}");
    assert!(matches!(err, crate::Error::ConfigError(_)), "{msg}");
    assert!(msg.contains("Failed to load tokenizer"), "{msg}");
    let _ = std::fs::remove_dir_all(&dir);
}
