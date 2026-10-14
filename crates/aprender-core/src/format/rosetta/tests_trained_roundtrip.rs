// ============================================================================
// Falsifiers for epic #4000 row T5 ("Train What You Serve"): trained weights
// survive gguf <-> safetensors <-> .apr at per-tensor cosine >= 0.98.
//
// Contract: contracts/trained-weights-format-roundtrip-v1.yaml
//
// The pygmy fixtures carry periodic synthetic values, which a broken converter
// can reproduce by accident. These tests use LoRA-merged weights instead —
// W' = W + (alpha / r) * B * A with seeded pseudo-random W, A and B — so every
// element is distinct and a dropped, transposed or mis-scaled tensor shows up
// as a cosine collapse, not as a coincidental match.
// ============================================================================

use super::*;
use crate::format::test_factory::PygmyConfig;
use crate::nn::functional::cosine_similarity_slice;
use std::collections::BTreeMap;
use std::path::Path;

/// The T5 bar from epic #4000.
const MIN_COSINE: f32 = 0.98;

/// LoRA rank and alpha of the merged adapter.
const LORA_RANK: usize = 8;
const LORA_ALPHA: f32 = 16.0;

/// Deterministic LCG in [-scale, scale]; no `rand` so the fixture is
/// byte-identical on every host.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((self.0 >> 40) as f32) / ((1u64 << 24) as f32); // [0, 1)
        (unit * 2.0 - 1.0) * scale
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32(scale)).collect()
    }
}

/// `W [rows, cols] + (alpha / r) * B [rows, r] * A [r, cols]`, row-major.
fn lora_merge(rng: &mut Lcg, rows: usize, cols: usize) -> Vec<f32> {
    let mut w = rng.vec(rows * cols, 0.05);
    let a = rng.vec(LORA_RANK * cols, 0.05);
    let b = rng.vec(rows * LORA_RANK, 0.05);
    let scale = LORA_ALPHA / LORA_RANK as f32;
    for i in 0..rows {
        for j in 0..cols {
            let delta: f32 = (0..LORA_RANK)
                .map(|k| b[i * LORA_RANK + k] * a[k * cols + j])
                .sum();
            w[i * cols + j] += scale * delta;
        }
    }
    w
}

/// Qwen2-shaped model, large enough that every matrix spans whole Q4_K
/// super-blocks (256 elements per row), with LoRA merged into every
/// attention and MLP projection.
fn trained_config() -> PygmyConfig {
    PygmyConfig {
        vocab_size: 512,
        hidden_size: 256,
        num_layers: 2,
        num_heads: Some(4),
        num_kv_heads: Some(2),
        tied_embeddings: false,
        ..PygmyConfig::qwen2_gqa()
    }
}

fn trained_tensors(cfg: &PygmyConfig) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let (v, h, i) = (cfg.vocab_size, cfg.hidden_size, cfg.hidden_size * 4);
    let kv = cfg.kv_dim();
    let mut rng = Lcg(0x4000_7505);
    let mut t = BTreeMap::new();
    t.insert(
        "model.embed_tokens.weight".to_string(),
        (rng.vec(v * h, 0.05), vec![v, h]),
    );
    t.insert(
        "lm_head.weight".to_string(),
        (lora_merge(&mut rng, v, h), vec![v, h]),
    );
    let norm = |rng: &mut Lcg| (0..h).map(|_| 1.0 + rng.next_f32(0.1)).collect();
    t.insert("model.norm.weight".to_string(), (norm(&mut rng), vec![h]));
    for l in 0..cfg.num_layers {
        let p = format!("model.layers.{l}");
        t.insert(
            format!("{p}.input_layernorm.weight"),
            (norm(&mut rng), vec![h]),
        );
        t.insert(
            format!("{p}.post_attention_layernorm.weight"),
            (norm(&mut rng), vec![h]),
        );
        for (name, rows, cols) in [
            ("self_attn.q_proj", h, h),
            ("self_attn.k_proj", kv, h),
            ("self_attn.v_proj", kv, h),
            ("self_attn.o_proj", h, h),
            ("mlp.gate_proj", i, h),
            ("mlp.up_proj", i, h),
            ("mlp.down_proj", h, i),
        ] {
            t.insert(
                format!("{p}.{name}.weight"),
                (lora_merge(&mut rng, rows, cols), vec![rows, cols]),
            );
        }
        for (name, n) in [("q_proj", h), ("k_proj", kv), ("v_proj", kv)] {
            t.insert(
                format!("{p}.self_attn.{name}.bias"),
                (rng.vec(n, 0.02), vec![n]),
            );
        }
    }
    t
}

/// Per-tensor cosine of `got` against `want`. Refuses to pass vacuously:
/// every expected tensor must be present and the compared count is returned.
fn assert_tensors_match(
    rosetta: &RosettaStone,
    got: &Path,
    want: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    hop: &str,
) -> usize {
    let mut compared = 0;
    let mut failures = Vec::new();
    for (name, (expected, _)) in want {
        let actual = rosetta
            .load_tensor_f32(got, name)
            .unwrap_or_else(|e| panic!("{hop}: tensor {name} missing after round-trip: {e}"));
        assert_eq!(
            actual.len(),
            expected.len(),
            "{hop}: tensor {name} changed element count"
        );
        let cos = cosine_similarity_slice(&actual, expected);
        if !(cos >= MIN_COSINE) {
            failures.push(format!("{name}: cosine {cos:.4}"));
        }
        compared += 1;
    }
    assert!(
        failures.is_empty(),
        "{hop}: {} of {compared} tensors below cosine {MIN_COSINE}: {failures:?}",
        failures.len()
    );
    assert_eq!(compared, want.len(), "{hop}: compared count");
    assert!(compared > 0, "{hop}: zero tensors compared is not a pass");
    compared
}

/// Minimal BPE tokenizer.json whose vocab covers every embedding row.
fn tokenizer_json(vocab_size: usize) -> String {
    let mut vocab: Vec<String> = ["<pad>", "<eos>", "<unk>"]
        .iter()
        .enumerate()
        .map(|(id, tok)| format!("\"{tok}\": {id}"))
        .collect();
    vocab.extend((3..vocab_size).map(|id| format!("\"t{id}\": {id}")));
    format!(
        r#"{{"version": "1.0", "model": {{"type": "BPE", "vocab": {{{}}}, "merges": []}}}}"#,
        vocab.join(", ")
    )
}

/// safetensors -> apr -> gguf(`quant`) -> apr -> safetensors, checked at every
/// file that rosetta can read back.
fn run_cycle(quant: &str) {
    let cfg = trained_config();
    let want = trained_tensors(&cfg);
    let dir = tempfile::tempdir().expect("temp dir");
    let st = dir.path().join("model.safetensors");
    crate::serialization::safetensors::save_safetensors(&st, &want).expect("write safetensors");
    std::fs::write(dir.path().join("config.json"), cfg.to_config_json()).expect("config.json");
    // A trained model ships its tokenizer. The cycle must carry it through
    // every hop on its own: GGUF -> APR refuses a file without one, and no
    // hop below is handed a --tokenizer override.
    std::fs::write(
        dir.path().join("tokenizer.json"),
        tokenizer_json(cfg.vocab_size),
    )
    .expect("tokenizer.json");

    let rosetta = RosettaStone::new();
    let apr1 = dir.path().join("hop1.apr");
    let gguf = dir.path().join("hop2.gguf");
    let apr2 = dir.path().join("hop3.apr");
    let st2 = dir.path().join("hop4.safetensors");

    rosetta
        .convert(&st, &apr1, None)
        .expect("safetensors -> apr");
    assert_tensors_match(&rosetta, &apr1, &want, "safetensors->apr");

    let q = ConversionOptions {
        quantization: Some(quant.to_string()),
        ..Default::default()
    };
    rosetta
        .convert(&apr1, &gguf, Some(q))
        .unwrap_or_else(|e| panic!("apr -> gguf({quant}): {e}"));
    rosetta.convert(&gguf, &apr2, None).expect("gguf -> apr");
    rosetta
        .convert(&apr2, &st2, None)
        .expect("apr -> safetensors");

    let n = assert_tensors_match(
        &rosetta,
        &st2,
        &want,
        &format!("full cycle via gguf({quant})"),
    );
    assert_eq!(n, want.len());
}

#[test]
fn falsifier_t5_trained_weights_roundtrip_fp16() {
    run_cycle("fp16");
}

#[test]
fn falsifier_t5_trained_weights_roundtrip_q4_k() {
    run_cycle("q4_k");
}

#[test]
fn falsifier_t5_checker_rejects_a_transposed_tensor() {
    // Negative control: the checker must turn red on the defect class that
    // LAYOUT-001 has hit 100+ times. A transposed q_proj keeps every value
    // and its length, so only the cosine can catch it.
    let cfg = trained_config();
    let want = trained_tensors(&cfg);
    let (w, shape) = &want["model.layers.0.self_attn.q_proj.weight"];
    let (rows, cols) = (shape[0], shape[1]);
    let transposed: Vec<f32> = (0..rows * cols)
        .map(|k| w[(k % rows) * cols + k / rows])
        .collect();
    let cos = cosine_similarity_slice(&transposed, w);
    assert!(
        cos < MIN_COSINE,
        "a transposed LoRA-merged matrix must fall below {MIN_COSINE}, got {cos:.4}; \
         the fixture is too symmetric to detect LAYOUT-001"
    );
}
