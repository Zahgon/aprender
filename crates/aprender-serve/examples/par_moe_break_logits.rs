//! Does a named MoE break reach the logits? (E7 #4001, C358 1f)
//!
//! `cargo run --release -p aprender-serve --example par_moe_break_logits -- GOOD BROKEN IDS`
//!
//! Runs two CPU qwen3moe forwards on the same prompt ids (comma-separated)
//! against GOOD and BROKEN: the full-sequence `forward_qwen3_moe` (FP32
//! activations) and the cached per-token forward `apr run` takes (Q8_K
//! activations, the production path; 1i). For each it prints the argmax of
//! each model and the size of the logits difference. A swap the forward never
//! reads prints a zero difference.

use realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Logits of the full-sequence forward and of the cached forward.
fn logits(path: &str, ids: &[u32]) -> Res<(Vec<f32>, Vec<f32>)> {
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let meta = &mapped.model;
    let n = meta.expert_count().ok_or("no expert_count")?;
    let k = meta.expert_used_count().ok_or("no expert_used_count")?;
    let ffn = meta
        .expert_feed_forward_length()
        .ok_or("no expert_feed_forward_length")?;
    let layers = (0..model.layers().len())
        .map(|l| load_qwen3_moe_layer(meta, mapped.data(), l))
        .collect::<Result<Vec<_>, _>>()?;
    let full = model.forward_qwen3_moe(ids, &layers, n, k, ffn, mapped.data())?;
    let mut cache = OwnedQuantizedKVCache::from_config(model.config(), ids.len());
    let mut cached = Vec::new();
    for (pos, &t) in ids.iter().enumerate() {
        cached = model.forward_single_qwen3_moe_with_cache(
            t,
            &mut cache,
            pos,
            &layers,
            n,
            k,
            ffn,
            mapped.data(),
        )?;
    }
    Ok((full, cached))
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

/// Top-1 minus best other logit, at index `a`.
fn margin(v: &[f32], a: usize) -> f32 {
    v[a] - v
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != a)
        .map(|(_, x)| *x)
        .fold(f32::MIN, f32::max)
}

fn compare(g: &[f32], b: &[f32]) -> String {
    let max_abs = g
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let dot: f64 = g
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let ng: f64 = g.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let (ag, ab) = (argmax(g), argmax(b));
    format!(
        "logits len {} (good) {} (broken); argmax {ag} (good) {ab} (broken); \
         max |good - broken| {max_abs:e}; cosine {:.9}; margin at good argmax {:e} (good) {:e} (broken)",
        g.len(),
        b.len(),
        dot / (ng * nb),
        margin(g, ag),
        margin(b, ag),
    )
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [good, broken, ids] = args.as_slice() else {
        return Err("usage: par_moe_break_logits GOOD BROKEN IDS".into());
    };
    let ids: Vec<u32> = ids.split(',').map(str::parse).collect::<Result<_, _>>()?;
    let (gf, gc) = logits(good, &ids)?;
    let (bf, bc) = logits(broken, &ids)?;
    println!("full FP32: {}", compare(&gf, &bf));
    println!("cached Q8_K: {}", compare(&gc, &bc));
    Ok(())
}
