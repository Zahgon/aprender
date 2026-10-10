//! The named break for a qwen3moe parity arm (E7 #4001, C358 1f).
//!
//! `contracts/e7-parity-arm-pins-v1.yaml` § `moe_named_break`: swap experts
//! `i` and `j` in layer `L`, in all three of `ffn_gate_exps`, `ffn_up_exps`
//! and `ffn_down_exps`; the router is untouched, so tokens routed to `i`
//! reach `j`'s weights. The byte ranges come from [`load_qwen3_moe_layer`]
//! and [`expert_byte_slice`], the rule the forward pass reads experts by, so
//! the break moves exactly the bytes the model uses for `i` and `j` and no
//! other byte. Noise proves nothing; this is the small break the contract
//! names.

#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::os::unix::fs::FileExt;

use super::qwen3_moe_load::{expert_byte_slice, load_qwen3_moe_layer, Qwen3MoeQuantizedLayer};
use crate::error::{RealizarError, Result};
use crate::gguf::GGUFModel;

/// One expert-slice swap: bytes `[a, a + len)` trade places with `[b, b + len)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceSwap {
    /// Start of expert `i`'s slice, as a file offset.
    pub a: usize,
    /// Start of expert `j`'s slice, as a file offset.
    pub b: usize,
    /// Bytes per expert in this tensor.
    pub len: usize,
}

/// The three swaps of a named break: gate, up, down, in that order.
pub fn plan_layer(
    layer: &Qwen3MoeQuantizedLayer,
    data: &[u8],
    num_experts: usize,
    i: usize,
    j: usize,
) -> Result<[SliceSwap; 3]> {
    if i == j {
        return Err(RealizarError::InvalidShape {
            reason: format!("moe_named_break: i = j = {i} swaps nothing"),
        });
    }
    let swap = |t| -> Result<SliceSwap> {
        let si = expert_byte_slice(t, data, i, num_experts)?;
        let sj = expert_byte_slice(t, data, j, num_experts)?;
        let base = data.as_ptr() as usize;
        Ok(SliceSwap {
            a: si.as_ptr() as usize - base,
            b: sj.as_ptr() as usize - base,
            len: si.len(),
        })
    };
    Ok([
        swap(&layer.gate_exps)?,
        swap(&layer.up_exps)?,
        swap(&layer.down_exps)?,
    ])
}

/// Plan the break for layer `layer_idx` of a mapped qwen3moe file.
pub fn plan(
    model: &GGUFModel,
    data: &[u8],
    layer_idx: usize,
    i: usize,
    j: usize,
) -> Result<[SliceSwap; 3]> {
    let num_experts = model
        .expert_count()
        .ok_or_else(|| RealizarError::InvalidShape {
            reason: "moe_named_break: the file declares no expert_count".to_string(),
        })?;
    let layer = load_qwen3_moe_layer(model, data, layer_idx)?;
    plan_layer(&layer, data, num_experts, i, j)
}

/// Apply the swaps to a buffer. Applying the same plan twice restores it.
pub fn apply(buf: &mut [u8], swaps: &[SliceSwap]) {
    for s in swaps {
        let (lo, hi) = (s.a.min(s.b), s.a.max(s.b));
        let (head, tail) = buf.split_at_mut(hi);
        head[lo..lo + s.len].swap_with_slice(&mut tail[..s.len]);
    }
}

/// Apply the swaps to a file in place (the caller passes a copy, never the
/// pinned model).
#[cfg(unix)]
pub fn apply_to_file(file: &File, swaps: &[SliceSwap]) -> std::io::Result<()> {
    for s in swaps {
        let mut x = vec![0u8; s.len];
        let mut y = vec![0u8; s.len];
        file.read_exact_at(&mut x, s.a as u64)?;
        file.read_exact_at(&mut y, s.b as u64)?;
        file.write_all_at(&y, s.a as u64)?;
        file.write_all_at(&x, s.b as u64)?;
    }
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::{apply, plan_layer};
    use crate::gguf::quantized::QuantizedTensorRef;
    use crate::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer;

    const N: usize = 4;
    const PER: usize = 3;

    fn t(offset: usize) -> QuantizedTensorRef {
        QuantizedTensorRef {
            offset,
            byte_size: N * PER,
            num_elements: N * PER,
            qtype: 0,
        }
    }

    /// Router at 0, gate at 10, up at 30, down at 50; 70 bytes, each its own value.
    fn fixture() -> (Qwen3MoeQuantizedLayer, Vec<u8>) {
        let layer = Qwen3MoeQuantizedLayer {
            router: t(0),
            gate_exps: t(10),
            up_exps: t(30),
            down_exps: t(50),
        };
        (layer, (0..70u8).collect())
    }

    #[test]
    fn moe_named_break_swaps_only_experts_i_and_j_in_gate_up_down() {
        let (layer, data) = fixture();
        let swaps = plan_layer(&layer, &data, N, 1, 3).expect("plan");
        let mut broken = data.clone();
        apply(&mut broken, &swaps);
        let moved: Vec<usize> = (0..data.len()).filter(|&k| broken[k] != data[k]).collect();
        let mut want = Vec::new();
        for base in [10, 30, 50] {
            for e in [1, 3] {
                want.extend(base + e * PER..base + (e + 1) * PER);
            }
        }
        want.sort_unstable();
        assert_eq!(
            moved, want,
            "only the i and j slices of the three expert tensors move"
        );
        for base in [10, 30, 50] {
            assert_eq!(
                broken[base + PER..base + 2 * PER],
                data[base + 3 * PER..base + 4 * PER]
            );
            assert_eq!(
                broken[base + 3 * PER..base + 4 * PER],
                data[base + PER..base + 2 * PER]
            );
        }
        assert_eq!(broken[..10], data[..10], "the router is untouched");
        apply(&mut broken, &swaps);
        assert_eq!(broken, data, "applying the plan twice restores the input");
    }

    #[test]
    fn moe_named_break_refuses_i_equal_j_and_ids_past_expert_count() {
        let (layer, data) = fixture();
        assert!(
            plan_layer(&layer, &data, N, 2, 2).is_err(),
            "i = j swaps nothing"
        );
        assert!(
            plan_layer(&layer, &data, N, 0, N).is_err(),
            "j past expert_count"
        );
        assert!(
            plan_layer(&layer, &data[..60], N, 0, 3).is_err(),
            "slice past the file"
        );
    }
}
