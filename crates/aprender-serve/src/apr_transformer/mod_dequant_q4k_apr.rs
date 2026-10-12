
/// Tensor lookup helper for APR v2 format parsing.
///
/// Bundles raw file data and parsed tensor index for clean access patterns.
/// Replaces closure-based lookups with method calls.
struct AprTensorLookup<'a> {
    data: &'a [u8],
    tensors: &'a std::collections::BTreeMap<String, (usize, usize, Vec<usize>, u8)>,
    /// ALB-106: entrenar checkpoints store linear weights in cuBLAS column-major
    /// convention. When true, 2D linear weight tensors are transposed on load.
    transpose_cublas_weights: bool,
}

/// Dequantize Q4K tensor data, handling per-row padding for 2D tensors (GH-202).
fn dequant_q4k_tensor(tensor_data: &[u8], dims: &[usize]) -> Vec<f32> {
    if dims.len() == 2 && !dims[1].is_multiple_of(256) {
        dequant_perrow(tensor_data, dims, 256, 144, |block, out| {
            dequant_q4k_block(block, out);
        })
    } else {
        let num_elements: usize = dims.iter().product();
        dequantize_q4_k_apr(tensor_data, num_elements)
    }
}

/// Dequantize Q5K tensor data, handling per-row padding for 2D tensors (GH-202).
///
/// #3418 (FT-GGMLT-011): Q5_K has its own 176-byte super-block. It used to share the Q4_K
/// arm, which read it with the 144-byte layout and dropped the fifth bit of every weight.
fn dequant_q5k_tensor(tensor_data: &[u8], dims: &[usize]) -> Vec<f32> {
    let num_elements: usize = dims.iter().product();
    if dims.len() == 2 && !dims[1].is_multiple_of(256) {
        dequant_perrow(tensor_data, dims, 256, 176, |block, out| {
            if let Ok(values) = crate::quantize::dequantize_q5_k(block) {
                out.copy_from_slice(&values);
            }
        })
    } else {
        let mut values = crate::quantize::dequantize_q5_k(tensor_data)
            .unwrap_or_else(|_| vec![0.0; num_elements]);
        values.resize(num_elements, 0.0);
        values
    }
}

/// Dequantize Q6K tensor data, handling per-row padding for 2D tensors (GH-202).
fn dequant_q6k_tensor(tensor_data: &[u8], dims: &[usize]) -> Vec<f32> {
    if dims.len() == 2 && !dims[1].is_multiple_of(256) {
        dequant_perrow(tensor_data, dims, 256, 210, |block, out| {
            dequant_q6k_block(block, out);
        })
    } else {
        let num_elements: usize = dims.iter().product();
        dequantize_q6_k_apr(tensor_data, num_elements)
    }
}

/// #3418: a block tensor whose block size and bytes come from the one ggml
/// table, with 2-D rows padded to whole blocks (GH-202) decoded row by row.
fn dequant_superblock_tensor(
    tensor_data: &[u8],
    dims: &[usize],
    ggml_id: u32,
    decode: impl Fn(&[u8]) -> crate::error::Result<Vec<f32>>,
) -> Vec<f32> {
    let num_elements: usize = dims.iter().product();
    let Some(t) = crate::gguf::ggml_type_table::traits(ggml_id) else {
        return vec![0.0; num_elements];
    };
    if dims.len() == 2 && !dims[1].is_multiple_of(t.blck_size) {
        dequant_perrow(tensor_data, dims, t.blck_size, t.type_size, |block, out| {
            if let Ok(values) = decode(block) {
                out.copy_from_slice(&values);
            }
        })
    } else {
        let mut values = decode(tensor_data).unwrap_or_else(|_| vec![0.0; num_elements]);
        values.resize(num_elements, 0.0);
        values
    }
}

/// #3418 (FT-GGMLT-014): the dtype bytes `dequant_by_dtype` has a decoder arm
/// for. A block-quantized id outside this list is refused at load rather than
/// read as F32; `every_admitted_block_id_has_a_decoder_arm_3418` ties the two.
fn apr_decodes(dtype: u8) -> bool {
    matches!(dtype, 0 | 1 | 2 | 3 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 30 | 128 | 129)
        || crate::quantize::iq_dispatch::iq_block_elems(u32::from(dtype)).is_some()
}

/// #3418 (FT-GGMLT-013): refuse a block-quantized tensor whose byte size does
/// not fit its dims under the one ggml table. The decoders zero-fill what they
/// cannot read, so a short tensor would otherwise load as zeros. A 2-D tensor
/// may store each row padded to whole blocks (GH-202). Ids 8 and 9 are the
/// legacy APR Q4/Q8 ids (GH-438) and are not ggml blocks here.
fn check_quant_tensor_sizes(
    tensors: &std::collections::BTreeMap<String, (usize, usize, Vec<usize>, u8)>,
) -> crate::error::Result<()> {
    for (name, (_, size, dims, dtype)) in tensors {
        if matches!(dtype, 8 | 9) {
            continue;
        }
        let Some(t) = crate::gguf::ggml_type_table::traits(u32::from(*dtype))
            .filter(|t| t.blck_size > 1)
        else {
            continue;
        };
        if !apr_decodes(*dtype) {
            return Err(crate::error::RealizarError::FormatError {
                reason: format!("tensor {name}: the APR loader has no decoder for {}", t.name),
            });
        }
        let elements: usize = dims.iter().product();
        let flat = elements.div_ceil(t.blck_size) * t.type_size;
        let padded = match dims.as_slice() {
            [rows, cols] => rows * cols.div_ceil(t.blck_size) * t.type_size,
            _ => flat,
        };
        if *size != flat && *size != padded {
            return Err(crate::error::RealizarError::FormatError {
                reason: format!(
                    "tensor {name}: {size} bytes do not fit dims {dims:?} as {} ({flat} or {padded} expected)",
                    t.name
                ),
            });
        }
    }
    Ok(())
}

/// Dequantize a single tensor's raw bytes based on GGML dtype.
/// GH-191: Match on GGML dtype values written by converter.
/// GH-88: Added Q4_0, Q4_1, Q5_0, Q5_1 support for mixed-quantization GGUF imports.
fn dequant_by_dtype(tensor_data: &[u8], dims: &[usize], dtype: u8) -> Vec<f32> {
    match dtype {
        // GH-88: Q4_0 (GGML type 2) — 32-element blocks, 18 bytes each
        2 => crate::quantize::dequantize_q4_0(tensor_data)
            .unwrap_or_else(|_| vec![0.0; dims.iter().product()]),
        // GH-88: Q4_1 (GGML type 3) — 32-element blocks, 20 bytes each
        3 => crate::quantize::dequantize_q4_1(tensor_data)
            .unwrap_or_else(|_| vec![0.0; dims.iter().product()]),
        // GH-88: Q5_0 (GGML type 6) — 32-element blocks, 22 bytes each
        6 => crate::quantize::dequantize_q5_0(tensor_data)
            .unwrap_or_else(|_| vec![0.0; dims.iter().product()]),
        // GH-88: Q5_1 (GGML type 7) — 32-element blocks, 24 bytes each
        7 => crate::quantize::dequantize_q5_1(tensor_data)
            .unwrap_or_else(|_| vec![0.0; dims.iter().product()]),
        12 => dequant_q4k_tensor(tensor_data, dims),
        13 => dequant_q5k_tensor(tensor_data, dims),
        14 => dequant_q6k_tensor(tensor_data, dims),
        // #3418: Q2_K and Q3_K have decoders; they no longer fall to the F32 arm.
        10 => dequant_superblock_tensor(tensor_data, dims, 10, crate::quantize::dequantize_q2_k),
        11 => dequant_superblock_tensor(tensor_data, dims, 11, crate::quantize::dequantize_q3_k),
        // GH-239: dtype=8 is ambiguous — either GGML Q8_0 or APR Q4 native.
        8 => {
            let num_elements: usize = dims.iter().product();
            let num_blocks = num_elements.div_ceil(32);
            if tensor_data.len() == num_blocks * 34 {
                dequantize_q8_0_apr(tensor_data, num_elements)
            } else {
                dequantize_apr_q4_native(tensor_data, num_elements)
            }
        },
        9 => {
            let num_elements: usize = dims.iter().product();
            dequantize_apr_q8_native(tensor_data, num_elements)
        },
        // #3418: the ids the APR writer emits today (GH-438); 8 and 9 above are
        // the legacy ids for the same two layouts.
        128 => dequantize_apr_q4_native(tensor_data, dims.iter().product()),
        129 => dequantize_apr_q8_native(tensor_data, dims.iter().product()),
        1 => tensor_data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        // GH-369: BF16 (GGML type 30) — 2 bytes per element, upper 16 bits of F32
        30 => tensor_data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        // #3418: IQ types decode with the IQ decoders, not the F32 arm.
        id if crate::quantize::iq_dispatch::iq_block_elems(u32::from(id)).is_some() => {
            dequant_superblock_tensor(tensor_data, dims, u32::from(id), |d| {
                crate::quantize::dequantize_iq_tensor(u32::from(id), d)
            })
        },
        _ => tensor_data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

impl AprTensorLookup<'_> {
    /// Retrieve the raw byte slice for a named tensor, or None if missing/out-of-bounds.
    fn raw_bytes(&self, name: &str) -> Option<(&[u8], &Vec<usize>, u8)> {
        self.tensors.get(name).and_then(|(offset, size, dims, dtype)| {
            let end = offset + size;
            if end > self.data.len() {
                return None;
            }
            Some((&self.data[*offset..end], dims, *dtype))
        })
    }

    /// Extract tensor as f32 values (with dequantization for Q4K/Q5K/Q6K/Q8_0/APR Q4/Q8).
    /// ALB-106: When `transpose_cublas_weights` is set, 2D linear weight tensors
    /// are transposed from cuBLAS convention [K,N] row-major to HF [N,K] row-major.
    fn get_f32(&self, name: &str) -> Option<Vec<f32>> {
        self.raw_bytes(name)
            .map(|(tensor_data, dims, dtype)| {
                let mut data = dequant_by_dtype(tensor_data, dims, dtype);
                if self.transpose_cublas_weights
                    && dims.len() == 2
                    && Self::is_linear_weight(name)
                {
                    data = Self::transpose_2d(&data, dims[0], dims[1]);
                }
                data
            })
    }

    /// PMAT-788: Probe whether a tensor is present without materializing it.
    /// Used by the tied-LM-head path to preserve the original fail-fast contract
    /// (a tied head requires a resolvable embedding) without loading a redundant
    /// copy of the embedding buffer.
    fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Check if a tensor name refers to a linear weight that needs transposing.
    fn is_linear_weight(name: &str) -> bool {
        name.contains("_proj.weight")
            || name.contains("gate_proj.weight")
            || name.contains("up_proj.weight")
            || name.contains("down_proj.weight")
            || name == "lm_head.weight"
    }

    /// Transpose a 2D matrix from [cols, rows] row-major to [rows, cols] row-major.
    /// Used to convert cuBLAS convention [K, N] to HF convention [N, K].
    fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; data.len()];
        for c in 0..cols {
            for r in 0..rows {
                out[r * cols + c] = data[c * rows + r];
            }
        }
        out
    }

    /// Extract raw Q4K bytes (no dequantization) for fused kernel.
    /// GH-191 FIX: Use GGML dtype value 12=Q4_K. #3418 (FT-GGMLT-011): Q5_K (13) is not
    /// admitted; its 176-byte super-blocks are not the Q4_K kernel's, so it takes the f32 path.
    fn get_q4k(&self, name: &str) -> Option<Vec<u8>> {
        self.raw_bytes(name).and_then(|(data, _, dtype)| {
            if dtype != 12 { return None; }
            Some(data.to_vec())
        })
    }

    /// Extract raw Q6K bytes (no dequantization) for fused kernel.
    /// GH-191 FIX: Use GGML dtype value 14=Q6_K.
    fn get_q6k(&self, name: &str) -> Option<Vec<u8>> {
        self.raw_bytes(name).and_then(|(data, _, dtype)| {
            if dtype != 14 { return None; }
            Some(data.to_vec())
        })
    }
}

/// HF and GGUF naming prefixes for a single layer.
struct LayerPrefixes {
    hf: String,
    gguf: String,
}

impl LayerPrefixes {
    fn new(layer_idx: usize) -> Self {
        Self {
            hf: format!("model.layers.{layer_idx}"),
            gguf: format!("blk.{layer_idx}"),
        }
    }
}

impl AprTensorLookup<'_> {
    /// Look up a tensor by HF name first, then GGUF name, returning f32 values.
    fn get_hf_or_gguf(&self, hf_name: &str, gguf_name: &str) -> Option<Vec<f32>> {
        self.get_f32(hf_name).or_else(|| self.get_f32(gguf_name))
    }

    /// Load fused or separate QKV weight for a layer.
    fn load_qkv_weight(
        &self,
        pfx: &LayerPrefixes,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Vec<f32> {
        if let Some(qkv) = self.get_f32(&format!("{}.self_attn.qkv_proj.weight", pfx.hf)) {
            return qkv;
        }
        let q = self.get_hf_or_gguf(
            &format!("{}.self_attn.q_proj.weight", pfx.hf),
            &format!("{}.attn_q.weight", pfx.gguf),
        ).unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);
        let k = self.get_hf_or_gguf(
            &format!("{}.self_attn.k_proj.weight", pfx.hf),
            &format!("{}.attn_k.weight", pfx.gguf),
        ).unwrap_or_else(|| vec![0.0; hidden_dim * kv_dim]);
        let v = self.get_hf_or_gguf(
            &format!("{}.self_attn.v_proj.weight", pfx.hf),
            &format!("{}.attn_v.weight", pfx.gguf),
        ).unwrap_or_else(|| vec![0.0; hidden_dim * kv_dim]);
        let mut qkv = Vec::with_capacity(q.len() + k.len() + v.len());
        qkv.extend_from_slice(&q);
        qkv.extend_from_slice(&k);
        qkv.extend_from_slice(&v);
        qkv
    }

    /// Load fused or separate QKV bias for a layer (optional, for Qwen models).
    fn load_qkv_bias(&self, pfx: &LayerPrefixes) -> Option<Vec<f32>> {
        if let Some(fused) = self.get_f32(&format!("{}.self_attn.qkv_proj.bias", pfx.hf)) {
            return Some(fused);
        }
        let q = self.get_hf_or_gguf(
            &format!("{}.self_attn.q_proj.bias", pfx.hf),
            &format!("{}.attn_q.bias", pfx.gguf),
        );
        let k = self.get_hf_or_gguf(
            &format!("{}.self_attn.k_proj.bias", pfx.hf),
            &format!("{}.attn_k.bias", pfx.gguf),
        );
        let v = self.get_hf_or_gguf(
            &format!("{}.self_attn.v_proj.bias", pfx.hf),
            &format!("{}.attn_v.bias", pfx.gguf),
        );
        match (&q, &k, &v) {
            (Some(q), Some(k), Some(v)) => {
                let mut bias = Vec::with_capacity(q.len() + k.len() + v.len());
                bias.extend_from_slice(q);
                bias.extend_from_slice(k);
                bias.extend_from_slice(v);
                Some(bias)
            },
            _ => None,
        }
    }

    /// Load Q4K/Q6K raw bytes for all layer weights (PMAT-103).
    fn load_quantized_layer_weights(&self, pfx: &LayerPrefixes) -> Q4KLayerWeights {
        let get_q4k_hf_or_gguf = |hf: &str, gguf: &str| -> Option<Vec<u8>> {
            self.get_q4k(hf).or_else(|| self.get_q4k(gguf))
        };
        let get_q6k_hf_or_gguf = |hf: &str, gguf: &str| -> Option<Vec<u8>> {
            self.get_q6k(hf).or_else(|| self.get_q6k(gguf))
        };
        Q4KLayerWeights {
            qkv_weight: None,
            attn_q_weight: get_q4k_hf_or_gguf(
                &format!("{}.self_attn.q_proj.weight", pfx.hf),
                &format!("{}.attn_q.weight", pfx.gguf),
            ),
            attn_k_weight: get_q4k_hf_or_gguf(
                &format!("{}.self_attn.k_proj.weight", pfx.hf),
                &format!("{}.attn_k.weight", pfx.gguf),
            ),
            attn_v_weight: get_q4k_hf_or_gguf(
                &format!("{}.self_attn.v_proj.weight", pfx.hf),
                &format!("{}.attn_v.weight", pfx.gguf),
            ),
            attn_v_weight_q6k: get_q6k_hf_or_gguf(
                &format!("{}.self_attn.v_proj.weight", pfx.hf),
                &format!("{}.attn_v.weight", pfx.gguf),
            ),
            attn_output_weight: get_q4k_hf_or_gguf(
                &format!("{}.self_attn.o_proj.weight", pfx.hf),
                &format!("{}.attn_output.weight", pfx.gguf),
            ),
            ffn_gate_weight: get_q4k_hf_or_gguf(
                &format!("{}.mlp.gate_proj.weight", pfx.hf),
                &format!("{}.ffn_gate.weight", pfx.gguf),
            ),
            ffn_up_weight: get_q4k_hf_or_gguf(
                &format!("{}.mlp.up_proj.weight", pfx.hf),
                &format!("{}.ffn_up.weight", pfx.gguf),
            ),
            ffn_down_weight: get_q4k_hf_or_gguf(
                &format!("{}.mlp.down_proj.weight", pfx.hf),
                &format!("{}.ffn_down.weight", pfx.gguf),
            ),
            ffn_down_weight_q6k: get_q6k_hf_or_gguf(
                &format!("{}.mlp.down_proj.weight", pfx.hf),
                &format!("{}.ffn_down.weight", pfx.gguf),
            ),
            ffn_up_weight_q6k: get_q6k_hf_or_gguf(
                &format!("{}.mlp.up_proj.weight", pfx.hf),
                &format!("{}.ffn_up.weight", pfx.gguf),
            ),
        }
    }
}

/// Check if a `Q4KLayerWeights` has any quantized data.
fn has_quantized_data(w: &Q4KLayerWeights) -> bool {
    w.attn_q_weight.is_some()
        || w.attn_k_weight.is_some()
        || w.attn_output_weight.is_some()
        || w.ffn_gate_weight.is_some()
        || w.ffn_up_weight.is_some()
        || w.ffn_down_weight.is_some()
        || w.ffn_down_weight_q6k.is_some()
        || w.ffn_up_weight_q6k.is_some()
        || w.attn_v_weight_q6k.is_some()
}

impl AprTransformer {
    /// Build transformer layers from APR tensor data.
    ///
    /// Loads Q/K/V weights, attention output, FFN weights, and norms for each layer.
    /// Also extracts Q4K/Q6K raw bytes for fused kernel inference.
    fn build_apr_layers(
        lookup: &AprTensorLookup<'_>,
        num_layers: usize,
        hidden_dim: usize,
        kv_dim: usize,
        intermediate_dim: usize,
        num_experts: Option<usize>,
        debug_enabled: bool,
    ) -> (Vec<AprTransformerLayer>, Option<Vec<Q4KLayerWeights>>) {
        let mut layers = Vec::with_capacity(num_layers);
        let mut q4k_layer_weights: Vec<Q4KLayerWeights> = Vec::with_capacity(num_layers);
        let mut has_any_q4k = false;

        for i in 0..num_layers {
            let pfx = LayerPrefixes::new(i);

            let qkv_weight = lookup.load_qkv_weight(&pfx, hidden_dim, kv_dim);
            let qkv_bias = lookup.load_qkv_bias(&pfx);
            let attn_output = lookup.get_hf_or_gguf(
                &format!("{}.self_attn.o_proj.weight", pfx.hf),
                &format!("{}.attn_output.weight", pfx.gguf),
            ).unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);
            let attn_norm = lookup.get_hf_or_gguf(
                &format!("{}.input_layernorm.weight", pfx.hf),
                &format!("{}.attn_norm.weight", pfx.gguf),
            ).unwrap_or_else(|| vec![1.0; hidden_dim]);
            let ffn_norm = lookup.get_hf_or_gguf(
                &format!("{}.post_attention_layernorm.weight", pfx.hf),
                &format!("{}.ffn_norm.weight", pfx.gguf),
            );
            let ffn_gate = lookup.get_hf_or_gguf(
                &format!("{}.mlp.gate_proj.weight", pfx.hf),
                &format!("{}.ffn_gate.weight", pfx.gguf),
            );
            let ffn_up = lookup.get_hf_or_gguf(
                &format!("{}.mlp.up_proj.weight", pfx.hf),
                &format!("{}.ffn_up.weight", pfx.gguf),
            ).unwrap_or_else(|| vec![0.0; hidden_dim * intermediate_dim]);
            let ffn_down = lookup.get_hf_or_gguf(
                &format!("{}.mlp.down_proj.weight", pfx.hf),
                &format!("{}.ffn_down.weight", pfx.gguf),
            ).unwrap_or_else(|| vec![0.0; intermediate_dim * hidden_dim]);

            let quant_weights = lookup.load_quantized_layer_weights(&pfx);
            if has_quantized_data(&quant_weights) {
                has_any_q4k = true;
            }
            q4k_layer_weights.push(quant_weights);

            // ALB-094: Load MoE weights if architecture supports it
            let moe_gate_weight = lookup.get_f32(&format!("{}.mlp.gate.weight", pfx.hf));

            let (moe_expert_gate_up, moe_expert_down, has_moe) = if let Some(n_experts) = num_experts {
                let mut all_gate_up = Vec::new();
                let mut all_down = Vec::new();
                for e in 0..n_experts {
                    let gate = lookup.get_f32(&format!("{}.mlp.experts.{e}.gate_proj.weight", pfx.hf))
                        .unwrap_or_default();
                    let up = lookup.get_f32(&format!("{}.mlp.experts.{e}.up_proj.weight", pfx.hf))
                        .unwrap_or_default();
                    let down = lookup.get_f32(&format!("{}.mlp.experts.{e}.down_proj.weight", pfx.hf))
                        .unwrap_or_default();
                    // Concatenate gate + up for fused gate_up
                    all_gate_up.extend_from_slice(&gate);
                    all_gate_up.extend_from_slice(&up);
                    all_down.extend_from_slice(&down);
                }
                let has = !all_gate_up.is_empty();
                (
                    if has { Some(all_gate_up) } else { None },
                    if has { Some(all_down) } else { None },
                    has,
                )
            } else {
                (None, None, false)
            };

            // ALB-094: Load shared expert weights (Qwen3.5 MoE)
            let moe_shared_gate = lookup.get_f32(&format!("{}.mlp.shared_expert.gate_proj.weight", pfx.hf));
            let moe_shared_up = lookup.get_f32(&format!("{}.mlp.shared_expert.up_proj.weight", pfx.hf));
            let moe_shared_down = lookup.get_f32(&format!("{}.mlp.shared_expert.down_proj.weight", pfx.hf));
            let moe_shared_expert_gate_weight = lookup.get_f32(&format!("{}.mlp.shared_expert_gate.weight", pfx.hf));

            // When MoE is active, standard FFN is replaced by experts
            let (final_ffn_gate, final_ffn_up, final_ffn_down) = if has_moe {
                (None, vec![0.0f32; 0], vec![0.0f32; 0])
            } else {
                (ffn_gate, ffn_up, ffn_down)
            };

            layers.push(AprTransformerLayer {
                attn_norm_weight: attn_norm,
                attn_norm_bias: None,
                qkv_weight,
                qkv_bias,
                attn_output_weight: attn_output,
                attn_output_bias: None,
                ffn_gate_weight: final_ffn_gate,
                ffn_gate_bias: None,
                ffn_up_weight: final_ffn_up,
                ffn_up_bias: None,
                ffn_down_weight: final_ffn_down,
                ffn_down_bias: None,
                ffn_norm_weight: ffn_norm,
                ffn_norm_bias: None,
                attn_q_norm_weight: None,
                attn_k_norm_weight: None,
                linear_attn_z_weight: None,
                linear_attn_b_weight: None,
                linear_attn_a_weight: None,
                linear_attn_conv1d_weight: None,
                linear_attn_a_log: None,
                linear_attn_dt_bias: None,
                linear_attn_norm_weight: None,
                moe_gate_weight,
                moe_expert_gate_up,
                moe_expert_down,
                moe_shared_gate,
                moe_shared_up,
                moe_shared_down,
                moe_shared_expert_gate_weight,
            });
        }

        let q4k_layers = if has_any_q4k {
            if debug_enabled {
                eprintln!("[DEBUG] Loaded Q4K raw bytes for fused kernel inference");
            }
            Some(q4k_layer_weights)
        } else {
            None
        };

        (layers, q4k_layers)
    }
}

// ============================================================================
// GH-369: BF16 dequant_by_dtype contract tests (Popperian falsification)
// ============================================================================
#[cfg(test)]
mod tests_dequant_bf16 {
    use super::dequant_by_dtype;

    /// Helper: encode f32 as BF16 bytes (truncate lower 16 bits).
    fn f32_to_bf16_bytes(val: f32) -> [u8; 2] {
        let bits = val.to_bits();
        let bf16 = (bits >> 16) as u16;
        bf16.to_le_bytes()
    }

    /// Falsify: BF16 element count matches dims, not half (old bug: 4 bytes/elem → wrong count).
    #[test]
    fn falsify_bf16_dequant_001_element_count() {
        let n = 8;
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            data.extend_from_slice(&f32_to_bf16_bytes(i as f32));
        }
        let result = dequant_by_dtype(&data, &[n], 30);
        assert_eq!(result.len(), n, "BF16 must produce exactly {n} elements from {n}*2 bytes");
    }

    /// Falsify: BF16 zeros produce f32 zeros (not garbage from misaligned reads).
    #[test]
    fn falsify_bf16_dequant_002_zeros() {
        let data = vec![0u8; 16]; // 8 BF16 zeros
        let result = dequant_by_dtype(&data, &[8], 30);
        assert!(result.iter().all(|&x| x == 0.0), "BF16 zeros must decode to f32 zeros");
    }

    /// Falsify: BF16 round-trip preserves BF16-representable values.
    #[test]
    fn falsify_bf16_dequant_003_round_trip() {
        let test_values: Vec<f32> = vec![1.0, -1.0, 0.5, 42.0, -128.0, 3.14159];
        let mut data = Vec::new();
        for &v in &test_values {
            data.extend_from_slice(&f32_to_bf16_bytes(v));
        }
        let result = dequant_by_dtype(&data, &[test_values.len()], 30);
        for (i, (&original, &decoded)) in test_values.iter().zip(result.iter()).enumerate() {
            // BF16 truncates lower 16 bits — tolerance is relative
            let expected_bf16 = f32::from_bits(original.to_bits() & 0xFFFF_0000);
            assert!(
                (decoded - expected_bf16).abs() < 1e-10,
                "BF16 round-trip mismatch at [{i}]: original={original}, expected_bf16={expected_bf16}, got={decoded}"
            );
        }
    }

    /// Falsify: BF16 1.0 decodes to exactly 1.0 (1.0 is BF16-exact).
    #[test]
    fn falsify_bf16_dequant_004_one() {
        let data = f32_to_bf16_bytes(1.0);
        let result = dequant_by_dtype(&data, &[1], 30);
        assert_eq!(result[0], 1.0, "BF16 1.0 must decode to exactly f32 1.0");
    }

    /// Falsify: F16 (dtype=1) and BF16 (dtype=30) produce DIFFERENT results for same bytes.
    /// This proves they're not aliased (old default arm treated both as F32).
    #[test]
    fn falsify_bf16_dequant_005_not_f16() {
        // 0x3F80 as F16 = 1.875, as BF16 = some other value
        let data: [u8; 2] = [0x80, 0x3F];
        let bf16_result = dequant_by_dtype(&data, &[1], 30);
        let f16_result = dequant_by_dtype(&data, &[1], 1);
        assert_ne!(
            bf16_result[0], f16_result[0],
            "BF16 and F16 must not decode identically — they're different formats"
        );
    }

    /// Falsify: BF16 dtype 30 is NOT treated as F32 (old default arm bug).
    /// F32 needs 4 bytes/element; BF16 needs 2. If treated as F32, 8 bytes → 2 elements.
    /// Correct BF16: 8 bytes → 4 elements.
    #[test]
    fn falsify_bf16_dequant_006_not_f32_default() {
        let data = vec![0u8; 8]; // 4 BF16 elements OR 2 F32 elements
        let bf16_result = dequant_by_dtype(&data, &[4], 30);
        let f32_result = dequant_by_dtype(&data, &[2], 0); // dtype 0 hits default (F32)
        assert_eq!(bf16_result.len(), 4, "BF16: 8 bytes → 4 elements");
        assert_eq!(f32_result.len(), 2, "F32: 8 bytes → 2 elements");
    }

    /// Falsify: 2D tensor dims work correctly (embedding matrix shape).
    #[test]
    fn falsify_bf16_dequant_007_2d_embedding() {
        let rows = 4;
        let cols = 8;
        let n = rows * cols;
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            data.extend_from_slice(&f32_to_bf16_bytes(i as f32));
        }
        let result = dequant_by_dtype(&data, &[rows, cols], 30);
        assert_eq!(result.len(), n, "BF16 2D tensor must produce rows*cols elements");
        // Verify first and last values
        assert_eq!(result[0], 0.0);
        let expected_last = f32::from_bits(((n - 1) as f32).to_bits() & 0xFFFF_0000);
        assert!((result[n - 1] - expected_last).abs() < 0.1);
    }
}

#[cfg(test)]
mod q5k_arm_tests_3418 {
    use super::{dequant_by_dtype, AprTensorLookup};
    use std::collections::BTreeMap;

    /// `n` Q5_K super-blocks: d = 1.0, dmin = 0.5, every other byte a fixed pattern.
    fn q5k_superblocks(n: usize) -> Vec<u8> {
        let mut data: Vec<u8> = (0..n * 176).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        for sb in data.chunks_exact_mut(176) {
            sb[..4].copy_from_slice(&[0x00, 0x3C, 0x00, 0x38]);
        }
        data
    }

    /// FT-GGMLT-011 (#3418): dtype 13 is decoded with the Q5_K layout and is
    /// never handed to a Q4_K kernel.
    #[test]
    fn apr_q5k_is_never_read_as_q4k_3418() {
        let one = q5k_superblocks(1);
        let reference = crate::quantize::dequantize_q5_k(&one).expect("one Q5_K super-block");
        assert_eq!(dequant_by_dtype(&one, &[256], 13), reference, "1-D Q5_K");

        // GH-202: two rows of 200 columns, each padded to one 256 super-block.
        let two = q5k_superblocks(2);
        let mut padded = Vec::new();
        for row in two.chunks_exact(176) {
            let r = crate::quantize::dequantize_q5_k(row).expect("one Q5_K row");
            padded.extend_from_slice(&r[..200]);
        }
        assert_eq!(dequant_by_dtype(&two, &[2, 200], 13), padded, "padded 2-D Q5_K");

        let mut tensors = BTreeMap::new();
        tensors.insert("q4k".to_string(), (0, 144, vec![256], 12u8));
        tensors.insert("q5k".to_string(), (0, 176, vec![256], 13u8));
        let lookup = AprTensorLookup {
            data: &one,
            tensors: &tensors,
            transpose_cublas_weights: false,
        };
        assert!(lookup.get_q4k("q4k").is_some(), "Q4_K bytes must reach the Q4_K kernel");
        assert!(lookup.get_q4k("q5k").is_none(), "Q5_K bytes reached the Q4_K kernel");
    }
}

/// FT-GGMLT-012 (#3418): every dtype byte the APR writer emits for a
/// quantized tensor reaches its own decoder. Before this, 128 (AprQ4),
/// 129 (AprQ8), 10 (Q2_K) and 11 (Q3_K) fell through to the F32 arm and
/// were read as raw little-endian floats.
#[cfg(test)]
mod apr_writer_bytes_tests_3418 {
    use super::dequant_by_dtype;
    use crate::apr_transformer::dequant::{dequantize_apr_q4_native, dequantize_apr_q8_native};

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| ((i * 37 + 11) % 256) as u8).collect()
    }

    /// `n` blocks of `block_bytes`, each opening with the f16 scale 1.0.
    fn blocks(n: usize, block_bytes: usize) -> Vec<u8> {
        let mut data = pattern(n * block_bytes);
        for b in data.chunks_exact_mut(block_bytes) {
            b[..2].copy_from_slice(&[0x00, 0x3C]);
        }
        data
    }

    fn type_size(id: u32) -> usize {
        crate::gguf::ggml_type_table::traits(id)
            .expect("id is in the one ggml table")
            .type_size
    }

    #[test]
    fn every_byte_the_apr_writer_emits_reaches_its_own_decoder_3418() {
        // AprQ4 = 128: per-32 block [f16 scale][16 nibble bytes].
        let q4 = blocks(2, 18);
        assert_eq!(
            dequant_by_dtype(&q4, &[64], 128),
            dequantize_apr_q4_native(&q4, 64),
            "AprQ4 (128) was not decoded as AprQ4"
        );

        // AprQ8 = 129: [f32 scale][i8 x N].
        let mut q8 = 0.5f32.to_le_bytes().to_vec();
        q8.extend(pattern(16));
        assert_eq!(
            dequant_by_dtype(&q8, &[16], 129),
            dequantize_apr_q8_native(&q8, 16),
            "AprQ8 (129) was not decoded as AprQ8"
        );

        // Q2_K = 10 and Q3_K = 11: one 256-element super-block each, sized
        // from the one ggml table.
        let q2 = pattern(type_size(10));
        let want_q2 = crate::quantize::dequantize_q2_k(&q2).expect("one Q2_K super-block");
        assert_eq!(want_q2.len(), 256);
        assert_eq!(dequant_by_dtype(&q2, &[256], 10), want_q2, "Q2_K (10) was not decoded as Q2_K");

        let q3 = pattern(type_size(11));
        let want_q3 = crate::quantize::dequantize_q3_k(&q3).expect("one Q3_K super-block");
        assert_eq!(want_q3.len(), 256);
        assert_eq!(dequant_by_dtype(&q3, &[256], 11), want_q3, "Q3_K (11) was not decoded as Q3_K");
    }
}

/// FT-GGMLT-014 (#3418): an IQ tensor reaches the IQ decoders, and no
/// block-quantized id falls to the F32 arm.
#[cfg(test)]
mod iq_and_undecoded_tests_3418 {
    use super::dequant_by_dtype;

    fn whole(id: u32, elements: usize) -> Vec<u8> {
        let t = crate::gguf::ggml_type_table::traits(id).expect("id is in the one ggml table");
        let mut data: Vec<u8> = (0..elements / t.blck_size * t.type_size)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect();
        for b in data.chunks_exact_mut(t.type_size) {
            b[..2].copy_from_slice(&[0x00, 0x3C]);
        }
        data
    }

    #[test]
    fn iq_tensors_reach_the_iq_decoders_3418() {
        for id in 0u8..=42 {
            if crate::quantize::iq_dispatch::iq_block_elems(u32::from(id)).is_none() {
                continue;
            }
            let data = whole(u32::from(id), 512);
            let want = crate::quantize::dequantize_iq_tensor(u32::from(id), &data)
                .expect("a whole IQ tensor");
            assert_eq!(
                dequant_by_dtype(&data, &[512], id),
                want,
                "IQ id {id} was not decoded by the IQ decoder"
            );
        }
    }

    #[test]
    fn every_admitted_block_id_has_a_decoder_arm_3418() {
        for id in 0u8..=255 {
            if !super::apr_decodes(id) || matches!(id, 8 | 9) {
                continue;
            }
            let Some(t) = crate::gguf::ggml_type_table::traits(u32::from(id))
                .filter(|t| t.blck_size > 1)
            else {
                continue;
            };
            let data = whole(u32::from(id), 512);
            let f32_reading: Vec<f32> = data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let got = dequant_by_dtype(&data, &[512], id);
            assert_eq!(got.len(), 512, "{} ({id}) decoded to the wrong length", t.name);
            assert_ne!(got, f32_reading, "{} ({id}) is admitted but falls to the F32 arm", t.name);
        }
    }
}
