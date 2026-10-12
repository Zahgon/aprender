//! Which CPU qwen3moe path gives which first token? (E7 #4001, A5)
//!
//! `cargo run --release -p aprender-serve --example par_moe_paths -- MODEL IDS`
//!
//! On the same prompt ids (comma-separated), prints the argmax and top-2
//! margin of three CPU forwards: the full-sequence `forward_qwen3_moe`, and
//! the cached per-token `forward_single_qwen3_moe_with_cache` (the path
//! `apr run` takes) with the default Q8_K activations and with FP32 ones.

use realizar::gguf::qwen3_moe_load::{load_qwen3_moe_layer, Qwen3MoeQuantizedLayer};
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};
use realizar::quantize::with_fp32_activations;

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

struct Moe<'a> {
    model: OwnedQuantizedModel,
    layers: Vec<Qwen3MoeQuantizedLayer>,
    shape: (usize, usize, usize),
    data: &'a [u8],
}

impl Moe<'_> {
    fn full(&self, ids: &[u32]) -> Res<Vec<f32>> {
        let (n, k, ffn) = self.shape;
        Ok(self
            .model
            .forward_qwen3_moe(ids, &self.layers, n, k, ffn, self.data)?)
    }

    fn cached(&self, ids: &[u32]) -> Res<Vec<f32>> {
        let (n, k, ffn) = self.shape;
        let mut cache = OwnedQuantizedKVCache::from_config(self.model.config(), ids.len());
        let mut logits = Vec::new();
        for (pos, &t) in ids.iter().enumerate() {
            logits = self.model.forward_single_qwen3_moe_with_cache(
                t,
                &mut cache,
                pos,
                &self.layers,
                n,
                k,
                ffn,
                self.data,
            )?;
        }
        Ok(logits)
    }
}

fn top2(v: &[f32]) -> String {
    let a = (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
    let second = v
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != a)
        .map(|(_, x)| *x)
        .fold(f32::MIN, f32::max);
    format!("argmax {a} margin {:e}", v[a] - second)
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path, ids] = args.as_slice() else {
        return Err("usage: par_moe_paths MODEL IDS".into());
    };
    let ids: Vec<u32> = ids.split(',').map(str::parse).collect::<Result<_, _>>()?;
    let mapped = MappedGGUFModel::from_path(path)?;
    let meta = &mapped.model;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let layers = (0..model.layers().len())
        .map(|l| load_qwen3_moe_layer(meta, mapped.data(), l))
        .collect::<Result<Vec<_>, _>>()?;
    let shape = (
        meta.expert_count().ok_or("no expert_count")?,
        meta.expert_used_count().ok_or("no expert_used_count")?,
        meta.expert_feed_forward_length()
            .ok_or("no expert_feed_forward_length")?,
    );
    let moe = Moe {
        model,
        layers,
        shape,
        data: mapped.data(),
    };
    println!("full forward_qwen3_moe: {}", top2(&moe.full(&ids)?));
    println!("cached, Q8_K activations: {}", top2(&moe.cached(&ids)?));
    let fp32 = with_fp32_activations(|| moe.cached(&ids))?;
    println!("cached, FP32 activations: {}", top2(&fp32));
    Ok(())
}
