//! Write the named-break copy of a qwen3moe GGUF (E7 #4001, C358 1f).
//!
//! `cargo run --release -p aprender-serve --example moe_named_break -- IN OUT L I J`
//!
//! Copies IN to OUT, then swaps experts I and J of layer L in OUT's
//! `ffn_gate_exps`, `ffn_up_exps` and `ffn_down_exps`
//! (`contracts/e7-parity-arm-pins-v1.yaml` § `moe_named_break`). IN is only read.

use realizar::gguf::moe_named_break::{apply_to_file, plan};
use realizar::gguf::MappedGGUFModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output, layer, i, j] = args.as_slice() else {
        return Err("usage: moe_named_break IN OUT L I J".into());
    };
    let (layer, i, j): (usize, usize, usize) = (layer.parse()?, i.parse()?, j.parse()?);
    if std::fs::canonicalize(input).ok() == std::fs::canonicalize(output).ok() {
        return Err("OUT must not be IN: the pinned model is never edited".into());
    }
    let mapped = MappedGGUFModel::from_path(input)?;
    let swaps = plan(&mapped.model, mapped.data(), layer, i, j)?;
    std::fs::copy(input, output)?;
    let out = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(output)?;
    apply_to_file(&out, &swaps)?;
    for (name, s) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
        .iter()
        .zip(swaps)
    {
        println!(
            "blk.{layer}.{name}: swapped experts {i} and {j}: [{}, +{}) <-> [{}, +{})",
            s.a, s.len, s.b, s.len
        );
    }
    Ok(())
}
