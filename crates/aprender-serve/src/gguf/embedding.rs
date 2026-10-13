/// GH-278: Transpose a row-major f32 matrix from [rows x cols] to [cols x rows].
///
/// PMAT-285: Delegates to `contract_gate::transpose_f32` (single source of truth).
fn transpose_f32_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    crate::contract_gate::transpose_f32(data, rows, cols)
}

/// PMAT-895 (OBLIG-GGUF-LOAD-NANINF): scan the f16 scale field(s) of a quantized
/// tensor's raw bytes for a non-finite value (NaN/Inf). Returns the first offending
/// scale value if any, else `None`.
///
/// The block / super-block scale layouts mirror the dequant code
/// (`quantize/dequant.rs`, `quantize/dequant_q4k.rs`): each block leads with (or, for
/// Q6_K/Q2_K/Q3_K, ends with) one or two f16 scales. We read ONLY those f16 fields —
/// O(num_blocks), not O(num_elements). A non-finite `d`/`dmin` poisons every element
/// of its block at dequant, so checking the scales is both cheap and sufficient. F32
/// tensors (embeddings/norms) are stored separately and are not handled here.
/// Block layout for a scaled-block quant type: `(block_bytes, &[scale_offsets])`.
///
/// Each offset marks a 2-byte little-endian f16 scale field within the block.
/// `None` for F16/F32/BF16/unknown, which are not scaled-block layouts.
///
/// Split out of `quant_scale_first_nonfinite` so that function stays under the
/// cognitive-complexity gate (dtype.rs `include!()`s this file). Both columns
/// come from the one ggml table (#3418): the block size (FT-GGMLT-009) and the
/// f16 scale offsets (FT-GGMLT-018), so none are written here.
fn quant_block_layout(qtype: u32) -> Option<(usize, &'static [usize])> {
    let scale_offsets = crate::gguf::ggml_type_table::scale_offsets(qtype)?;
    let block_bytes = crate::gguf::ggml_type_table::traits(qtype)?.type_size;
    Some((block_bytes, scale_offsets))
}

fn quant_scale_first_nonfinite(data: &[u8], qtype: u32) -> Option<f32> {
    use crate::quantize::read_f16;

    if data.is_empty() {
        return None;
    }

    let (block_bytes, scale_offsets) = quant_block_layout(qtype)?;

    if block_bytes == 0 || !data.len().is_multiple_of(block_bytes) {
        // Malformed block size for this qtype: leave to the existing shape gates.
        return None;
    }

    for block in data.chunks_exact(block_bytes) {
        for &off in scale_offsets {
            let scale = read_f16(&block[off..off + 2]);
            if !scale.is_finite() {
                return Some(scale);
            }
        }
    }
    None
}

/// Dequantize token embedding from APR format to f32 based on dtype.
///
/// Refs realizar#85: Added BF16/F16 support for aprender's GH-205/GH-353 passthrough.
/// Refs realizar#86: Added all GGML quant types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q5_K, Q6_K).
///
/// The tensor is `rows` token rows of `row_len` values (`[vocab, hidden]`).
fn dequantize_embedding(
    embed_data: &[u8],
    dtype: &str,
    rows: usize,
    row_len: usize,
) -> Result<Vec<f32>> {
    let num_elements = rows * row_len;
    match dtype {
        "F32" | "f32" => Ok(embed_data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "BF16" | "bf16" => Ok(crate::inference::simd_bf16_to_f32(embed_data)),
        "F16" | "f16" => Ok(crate::apr::dequant::dequantize_f16(
            embed_data,
            num_elements,
        )),
        // APR native quant types
        "q8" => Ok(crate::apr::dequant::dequantize_apr_q8(
            embed_data,
            num_elements,
        )),
        "q4" => Ok(crate::apr::dequant::dequantize_apr_q4(
            embed_data,
            num_elements,
        )),
        // #3418 (FT-GGMLT-017, 026): GGML block types (from GGUF-sourced APR
        // files) by their table name, through the one table, each token row
        // decoded on its own so GH-202 row padding is dropped.
        name if crate::gguf::ggml_type_table::to_float_by_name(name).is_some() => {
            let id = crate::gguf::ggml_type_table::GGML_TYPES
                .iter()
                .find(|t| t.name == name)
                .expect("guarded above")
                .id;
            crate::gguf::ggml_type_table::to_float_rows(id, embed_data, row_len, rows)
        },
        other => Err(RealizarError::FormatError {
            reason: format!("APR: unsupported embedding dtype: {other}"),
        }),
    }
}

impl OwnedQuantizedModel {
    /// Create owned model from memory-mapped GGUF file
    ///
    /// # Errors
    ///
    /// Returns error if model loading fails
    pub fn from_mapped(mapped: &crate::gguf::MappedGGUFModel) -> Result<Self> {
        let data = mapped.data();
        let transformer = QuantizedGGUFTransformer::from_gguf(&mapped.model, data)?;

        // Get config for dimension calculations
        let config = &transformer.config;
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;

        // GH-279: Contract gate — validate architecture and dimensions before proceeding
        let _proof = crate::contract_gate::validate_model_load_basic(
            &config.architecture,
            config.num_layers,
            config.hidden_dim,
            config.num_heads,
            config.num_kv_heads,
            config.intermediate_dim,
            config.vocab_size,
        )
        .map_err(crate::contract_gate::gate_error)?;

        // Convert layers to owned (passing config for dimensions)
        // GH-278: Conv1D weight transpose is NOT needed for GGUF files.
        // Both llama.cpp (convert_hf_to_gguf.py) and aprender (transpose_weights: true)
        // already transpose Conv1D [in,out] -> Linear [out,in] during GGUF export.
        // Transposing again here would double-transpose F32 tensors.
        // The APR loading path (from_apr) still handles transpose for native APR formats.
        let layers: Vec<OwnedQuantizedLayer> = transformer
            .layers
            .iter()
            .map(|l| OwnedQuantizedLayer::from_borrowed(l, data, config))
            .collect();

        let model = Self {
            config: transformer.config.clone(),
            token_embedding: transformer.token_embedding,
            position_embedding: transformer.position_embedding,
            layers,
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight: transformer.output_norm_weight,
            output_norm_bias: transformer.output_norm_bias,
            // LM head: [hidden_dim] -> [vocab_size]
            lm_head_weight: OwnedQuantizedTensor::from_ref_with_dims(
                &transformer.lm_head_weight,
                data,
                hidden_dim,
                vocab_size,
            ),
            lm_head_bias: transformer.lm_head_bias,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };
        // PMAT-750: fail closed on a truncated/corrupt model (a quantized weight
        // declares real dims but has no data because the file was incomplete) instead
        // of silently running inference on a dead weight and emitting garbage.
        model.validate_quantized_tensors()?;
        Ok(model)
    }

    /// PMAT-750: reject a truncated/corrupt model at load. `from_ref_with_dims`
    /// substitutes an empty data buffer when a tensor's bytes run past the file, so a
    /// truncated GGUF would otherwise load and produce garbage at inference (apr qa's
    /// density gate catches it, but `apr run` does not run those gates). This fails the
    /// load with a clear error naming the first truncated tensor — the fail-closed
    /// guarantee from the Pillar-4 beat (PMAT-744) extended to the load path.
    ///
    /// PMAT-895 (OBLIG-GGUF-LOAD-NANINF): also reject a model whose quantized weights
    /// dequantize to NaN/Inf. A super-block whose f16 scale `d`/`dmin` is f16 +Inf
    /// (`0x7C00`) or NaN (`0x7E00`) makes EVERY element of that block non-finite at
    /// dequant; inference then emits garbage. llama.cpp / Ollama load such a model
    /// (their `check_tensors` defaults to false), so apr failing closed here is a
    /// genuine Pillar-4 BEAT. The same NaN/Inf guarantee already exists on the
    /// SafeTensors path (F-DATA-QUALITY-002, `safetensors/validation.rs`); this wires
    /// it into the quantized load path. We scan only the f16 scale field(s) per block
    /// (O(num_blocks), not O(num_elements)) — the scales are what corrupt the dequant.
    pub(crate) fn validate_quantized_tensors(&self) -> Result<()> {
        fn check(t: &OwnedQuantizedTensor, name: &str) -> Result<()> {
            if t.is_truncated() {
                return Err(crate::error::RealizarError::InvalidShape {
                    reason: format!(
                        "truncated/corrupt model: tensor '{name}' declares {}x{} but has no data (file is incomplete)",
                        t.out_dim, t.in_dim
                    ),
                });
            }
            // PMAT-895: fail closed on a non-finite quant scale (NaN/Inf dequant).
            if let Some(bad) = quant_scale_first_nonfinite(&t.data, t.qtype) {
                return Err(crate::error::RealizarError::InvalidShape {
                    reason: format!(
                        "OBLIG-GGUF-LOAD-NANINF: tensor '{name}' (qtype {}) has a non-finite \
                         f16 quant scale (value {bad}); it dequantizes to NaN/Inf and produces \
                         garbage at inference — apr fails closed at load (F-DATA-QUALITY-002)",
                        t.qtype
                    ),
                });
            }
            Ok(())
        }
        fn check_layer(layer: &OwnedQuantizedLayer, prefix: &str) -> Result<()> {
            match &layer.qkv_weight {
                OwnedQKVWeights::Fused(t) => check(t, &format!("{prefix}.qkv"))?,
                OwnedQKVWeights::Separate { q, k, v } => {
                    check(q, &format!("{prefix}.q"))?;
                    check(k, &format!("{prefix}.k"))?;
                    check(v, &format!("{prefix}.v"))?;
                },
            }
            check(&layer.attn_output_weight, &format!("{prefix}.attn_output"))?;
            check(&layer.ffn_up_weight, &format!("{prefix}.ffn_up"))?;
            check(&layer.ffn_down_weight, &format!("{prefix}.ffn_down"))?;
            if let Some(g) = &layer.ffn_gate_weight {
                check(g, &format!("{prefix}.ffn_gate"))?;
            }
            Ok(())
        }
        /// #2535: the MoE variant — attention only. The dense FFN slots are
        /// documented placeholders on a MoE model, not truncated data, so
        /// checking them misreports a complete file as corrupt.
        fn check_layer_attention_only(layer: &OwnedQuantizedLayer, prefix: &str) -> Result<()> {
            match &layer.qkv_weight {
                OwnedQKVWeights::Fused(t) => check(t, &format!("{prefix}.qkv"))?,
                OwnedQKVWeights::Separate { q, k, v } => {
                    check(q, &format!("{prefix}.q"))?;
                    check(k, &format!("{prefix}.k"))?;
                    check(v, &format!("{prefix}.v"))?;
                },
            }
            check(&layer.attn_output_weight, &format!("{prefix}.attn_output"))?;
            Ok(())
        }
        // #2535: a Mixture-of-Experts model has NO dense FFN tensors. Its FFN
        // weights live in per-expert `ffn_{up,gate,down}_exps`, routed by
        // `ffn_gate_inp`, and `QuantizedTransformer::from_gguf_for_moe`
        // deliberately fills the dense slots with PLACEHOLDER refs
        // (offset=0, byte_size=0, num_elements=0). Its doc comment states the
        // contract outright:
        //
        //     consumers MUST check `moe_layers[i].is_some()` before attempting
        //     any dense FFN dequantization
        //
        // This validator was such a consumer and did not check. It saw the
        // placeholders, concluded `data.is_empty() && dims > 0`, and failed the
        // load of a PERFECTLY GOOD file with:
        //
        //     truncated/corrupt model: tensor 'layer.0.ffn_up' declares
        //     16384x2048 but has no data (file is incomplete)
        //
        // Measured on Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf: the file is
        // complete — a direct GGUF parse showed the last tensor ending at byte
        // 18,556,689,568, exactly the file size, and `apr validate` reported
        // "VALID: 579 tensors checked, 0 contract violations". The diagnosis
        // was simply wrong, and it sends the user to re-download 18.5 GB that
        // cannot help.
        //
        // The dense-FFN checks are therefore skipped for MoE. Attention
        // (q/k/v/attn_output) and lm_head are still checked: those tensors are
        // real on MoE models, so the fail-closed truncation guarantee of
        // PMAT-750 is preserved everywhere it actually applies.
        let is_moe =
            crate::tensor_names::normalize_architecture(&self.config.architecture) == "qwen3_moe";
        for (i, layer) in self.layers.iter().enumerate() {
            if is_moe {
                check_layer_attention_only(layer, &format!("layer.{i}"))?;
            } else {
                check_layer(layer, &format!("layer.{i}"))?;
            }
        }
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            check_layer(layer, &format!("encoder_layer.{i}"))?;
        }
        check(&self.lm_head_weight, "lm_head")?;
        Ok(())
    }

    /// Create a model for testing purposes
    ///
    /// This constructor handles the internal CUDA fields automatically,
    /// allowing external tests to construct models without accessing pub(crate) fields.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `token_embedding` - Token embedding weights
    /// * `layers` - Quantized transformer layers
    /// * `output_norm_weight` - Output normalization weight
    /// * `output_norm_bias` - Optional output normalization bias
    /// * `lm_head_weight` - Language model head weight
    /// * `lm_head_bias` - Optional language model head bias
    #[must_use]
    pub fn new_for_test(
        config: GGUFConfig,
        token_embedding: Vec<f32>,
        layers: Vec<OwnedQuantizedLayer>,
        output_norm_weight: Vec<f32>,
        output_norm_bias: Option<Vec<f32>>,
        lm_head_weight: OwnedQuantizedTensor,
        lm_head_bias: Option<Vec<f32>>,
    ) -> Self {
        Self {
            config,
            token_embedding,
            position_embedding: None,
            layers,
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight,
            output_norm_bias,
            lm_head_weight,
            lm_head_bias,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }
}

#[cfg(test)]
mod quant_block_layout_tests {
    use super::{quant_block_layout, quant_scale_first_nonfinite};
    // ggml ids, as in the admitted list below.
    const Q4_0: u32 = 2;
    const Q4_K: u32 = 12;
    const Q6_K: u32 = 14;

    /// The layout table had NO coverage: changing Q4_K's block size from 144 to
    /// 999 left all 15,657 lib tests green. It is the sole input to a corruption
    /// detector, so a wrong entry silently disables that detector for one quant
    /// type. These tests exist because this table was refactored out of
    /// `quant_scale_first_nonfinite`, and refactoring untested code without
    /// leaving a test behind is how a "pure move" changes behaviour unnoticed.
    #[test]
    fn block_layout_matches_the_ggml_wire_format() {
        // Q4_0: 2-byte f16 scale + 16 bytes of packed nibbles.
        assert_eq!(quant_block_layout(Q4_0), Some((18, &[0][..])));
        // Q4_K super-block: 144 bytes, leading d and dmin.
        assert_eq!(quant_block_layout(Q4_K), Some((144, &[0, 2][..])));
        // Q6_K: 210 bytes with a TRAILING scale at 208 -- the offset most likely
        // to be transcribed as 0 by someone assuming scales lead.
        assert_eq!(quant_block_layout(Q6_K), Some((210, &[208][..])));
    }


    /// FT-GGMLT-009 (#3418): the block size comes from the one ggml table, and
    /// the scanner admits exactly the ten scaled-block types it has offsets for.
    #[test]
    fn scale_scanner_sizes_from_the_one_ggml_table_3418() {
        use crate::gguf::ggml_type_table::traits;
        const ADMITTED: [u32; 10] = [2, 3, 6, 7, 8, 10, 11, 12, 13, 14];
        for id in 0..64u32 {
            let Some((block_bytes, offsets)) = quant_block_layout(id) else {
                assert!(!ADMITTED.contains(&id), "ggml id {id} lost its layout");
                continue;
            };
            assert!(ADMITTED.contains(&id), "ggml id {id} newly admitted");
            let t = traits(id).expect("an admitted id is in the table");
            assert_eq!(block_bytes, t.type_size, "ggml id {id} block size");
            for &off in offsets {
                assert!(off + 2 <= block_bytes, "ggml id {id} scale at {off}");
            }
        }
    }
    #[test]
    fn unscaled_and_unknown_types_have_no_block_layout() {
        // F32(0), F16(1), BF16(30) are not scaled-block layouts; 99 is not a
        // GGML type at all. All must decline rather than guess a layout.
        for qtype in [0, 1, 30, 99] {
            assert_eq!(
                quant_block_layout(qtype),
                None,
                "qtype {qtype} must not report a scaled-block layout"
            );
        }
    }

    #[test]
    fn a_nonfinite_scale_is_reported_and_a_finite_one_is_not() {
        // One Q4_0 block: f16 scale then 16 quant bytes.
        let mut block = vec![0u8; 18];

        // f16 1.0 = 0x3C00.
        block[0] = 0x00;
        block[1] = 0x3C;
        assert_eq!(
            quant_scale_first_nonfinite(&block, Q4_0),
            None,
            "a finite scale must not be reported as corruption"
        );

        // f16 NaN = 0x7E00. This is the case the function exists to catch.
        block[0] = 0x00;
        block[1] = 0x7E;
        let found =
            quant_scale_first_nonfinite(&block, Q4_0).expect("a NaN scale must be detected");
        assert!(found.is_nan(), "expected the NaN scale itself, got {found}");
    }

    #[test]
    fn a_block_size_mismatch_declines_rather_than_reading_past_the_end() {
        // 17 bytes is not a whole number of 18-byte Q4_0 blocks.
        let short = vec![0u8; 17];
        assert_eq!(quant_scale_first_nonfinite(&short, Q4_0), None);
        assert_eq!(quant_scale_first_nonfinite(&[], Q4_0), None);
    }
}

#[cfg(test)]
mod to_float_site_3418 {
    use crate::gguf::ggml_type_table::{to_float, to_float_tests_3418::blocks, GGML_TYPES};

    /// FT-GGMLT-017: an APR embedding of any type the one table decodes,
    /// named by the table's type name, decodes exactly as the table does.
    #[test]
    fn embedding_reads_the_to_float_column_3418() {
        for t in GGML_TYPES.iter().filter(|t| to_float(t.id).is_some()) {
            let data = blocks(t.id, 2);
            let got = super::dequantize_embedding(&data, t.name, 1, 2 * t.blck_size)
                .unwrap_or_else(|e| panic!("embedding refused {} ({}): {e}", t.name, t.id));
            assert_eq!(got, to_float(t.id).expect("row")(&data).expect("decode"), "{}", t.name);
        }
    }

    /// FT-GGMLT-026: an APR embedding whose rows are 1.5 blocks wide (GH-202
    /// padding) keeps each token row's own hidden values, its padding dropped.
    #[test]
    fn embedding_drops_row_padding_3418() {
        for t in GGML_TYPES.iter().filter(|t| to_float(t.id).is_some()) {
            let hidden = t.blck_size + t.blck_size / 2;
            let data = blocks(t.id, 6);
            let want: Vec<u32> = data
                .chunks_exact(2 * t.type_size)
                .flat_map(|row| to_float(t.id).expect("row")(row).expect("decode")[..hidden].to_vec())
                .map(f32::to_bits)
                .collect();
            let got: Vec<u32> = super::dequantize_embedding(&data, t.name, 3, hidden)
                .unwrap_or_else(|e| panic!("embedding refused {} ({}): {e}", t.name, t.id))
                .into_iter()
                .map(f32::to_bits)
                .collect();
            assert_eq!(got, want, "{} padded rows", t.name);
        }
    }
}
