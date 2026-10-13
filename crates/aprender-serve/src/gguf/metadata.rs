impl GGUFModel {

    /// Extract tensor data by name with dequantization
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to extract
    /// * `file_data` - Complete GGUF file bytes
    ///
    /// # Returns
    ///
    /// Dequantized f32 tensor data
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tensor not found
    /// - Unsupported quantization type
    /// - Invalid data at offset
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let file_data = std::fs::read("model.gguf")?;
    /// let model = GGUFModel::from_bytes(&file_data)?;
    /// let weights = model.get_tensor_f32("layer.0.weight", &file_data)?;
    /// ```
    pub fn get_tensor_f32(&self, name: &str, file_data: &[u8]) -> Result<Vec<f32>> {
        // Find tensor info
        let tensor = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        // Calculate tensor size in elements
        let size: usize = tensor
            .dims
            .iter()
            .try_fold(1usize, |acc, &dim| {
                usize::try_from(dim).ok().and_then(|d| acc.checked_mul(d))
            })
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: format!("Tensor dimensions overflow: {:?}", tensor.dims),
            })?;

        // Convert tensor offset to usize and add tensor data start
        let tensor_offset =
            usize::try_from(tensor.offset).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "convert_offset".to_string(),
                reason: format!("Offset {} exceeds platform usize limit", tensor.offset),
            })?;
        let offset = self.tensor_data_start + tensor_offset;

        Self::dequantize_at(tensor.qtype, file_data, offset, size)
    }

    /// Dequantize `size` values of GGUF type `qtype` starting at byte `offset` of
    /// `file_data`. [`Self::get_tensor_f32`] passes a whole tensor; the Qwen3.5
    /// embedding passes one row of its still-quantized table (#4991).
    ///
    /// # Errors
    /// The byte range lies outside `file_data`, or `qtype` is not supported.
    pub(crate) fn dequantize_at(
        qtype: u32,
        file_data: &[u8],
        offset: usize,
        size: usize,
    ) -> Result<Vec<f32>> {
        let (block_values, block_bytes, dequant) = Self::dequant_kernel(qtype)?;
        let byte_size = size.div_ceil(block_values) * block_bytes;
        let bytes = offset
            .checked_add(byte_size)
            .and_then(|end| file_data.get(offset..end))
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!(
                    "Data range [{}, {}) exceeds file size {}",
                    offset,
                    offset.saturating_add(byte_size),
                    file_data.len()
                ),
            })?;
        let mut values = dequant(bytes)?;
        // Block types dequantize whole blocks; trim to the exact size.
        values.truncate(size);
        Ok(values)
    }

    /// `(values per block, bytes per block, dequantizer)` for GGUF type `qtype`.
    ///
    /// # Errors
    /// `qtype` is not supported.
    fn dequant_kernel(qtype: u32) -> Result<(usize, usize, Dequantizer)> {
        use crate::gguf::ggml_type_table::{to_float, traits};
        use crate::quantize::{dequantize_f16, dequantize_q4_k_simd, dequantize_q8_0_simd};
        // #3418 (FT-GGMLT-022): every block type takes its block size and
        // bytes from the one table and its decoder from the table's to_float
        // column, so this path decodes every type the column does (the IQ
        // types too). Q8_0 and Q4_K keep their SIMD loaders, which agree with
        // the column's scalar rows (dequantize_02.rs, dequantize_05.rs).
        if let (Some(t), Some(column)) = (traits(qtype), to_float(qtype)) {
            let decode: Dequantizer = match qtype {
                GGUF_TYPE_Q8_0 => dequantize_q8_0_simd,
                GGUF_TYPE_Q4_K => dequantize_q4_k_simd,
                _ => column,
            };
            return Ok((t.blck_size, t.type_size, decode));
        }
        // Float types stay with the site (the column holds block types only).
        Ok(match qtype {
            GGUF_TYPE_F32 => (1, 4, |b| {
                Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
            }),
            GGUF_TYPE_F16 => (1, 2, dequantize_f16),
            // BF16 (bfloat16): 2 bytes/elem, no block structure. Reuses the
            // existing SIMD converter (y = from_bits((bits as u32) << 16))
            // already used by the safetensors loaders + gguf/embedding.rs.
            // #1893-class fix: without this arm a BF16 GGUF's
            // embeddings/norms/lm_head hit the catch-all "Unsupported
            // quantization type: 30".
            GGUF_TYPE_BF16 => (1, 2, |b| Ok(crate::inference::simd_bf16_to_f32(b))),
            _ => {
                return Err(RealizarError::UnsupportedOperation {
                    operation: "get_tensor_f32".to_string(),
                    reason: format!("Unsupported quantization type: {qtype}"),
                })
            },
        })
    }

    /// Extract model architecture from metadata
    pub fn architecture(&self) -> Option<&str> {
        if let Some(GGUFValue::String(arch)) = self.metadata.get(crate::gguf::keys::GENERAL_ARCHITECTURE) {
            Some(arch.as_str())
        } else {
            None
        }
    }

    /// Get embedding dimension from metadata
    pub fn embedding_dim(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::EMBEDDING_LENGTH);
        if let Some(GGUFValue::UInt32(dim)) = self.metadata.get(&key) {
            Some(*dim as usize)
        } else {
            None
        }
    }

    /// Get number of layers from metadata
    pub fn num_layers(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::BLOCK_COUNT);
        if let Some(GGUFValue::UInt32(count)) = self.metadata.get(&key) {
            Some(*count as usize)
        } else {
            None
        }
    }

    /// Get number of attention heads from metadata
    pub fn num_heads(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ATTENTION_HEAD_COUNT);
        if let Some(GGUFValue::UInt32(count)) = self.metadata.get(&key) {
            Some(*count as usize)
        } else {
            None
        }
    }

    /// Get context length from metadata
    pub fn context_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::CONTEXT_LENGTH);
        if let Some(GGUFValue::UInt32(len)) = self.metadata.get(&key) {
            Some(*len as usize)
        } else {
            None
        }
    }

    /// Get number of key-value heads from metadata (for GQA)
    pub fn num_kv_heads(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ATTENTION_HEAD_COUNT_KV);
        if let Some(GGUFValue::UInt32(count)) = self.metadata.get(&key) {
            Some(*count as usize)
        } else {
            None
        }
    }

    /// Get attention key length (head dimension) from metadata.
    ///
    /// This is the per-head dimension for Q/K projections. For most models
    /// this equals `hidden_dim / num_heads`, but Qwen3-0.6B has `head_dim=128`
    /// while `hidden_dim=1024` and `num_heads=16` (so `q_dim=2048 ≠ hidden_dim`).
    ///
    /// GGUF key: `{arch}.attention.key_length`
    pub fn key_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ATTENTION_KEY_LENGTH);
        if let Some(GGUFValue::UInt32(len)) = self.metadata.get(&key) {
            Some(*len as usize)
        } else {
            None
        }
    }

    /// Get attention value length (value head dimension) from metadata.
    ///
    /// GGUF key: `{arch}.attention.value_length`
    pub fn value_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ATTENTION_VALUE_LENGTH);
        if let Some(GGUFValue::UInt32(len)) = self.metadata.get(&key) {
            Some(*len as usize)
        } else {
            None
        }
    }

    /// Get RoPE frequency base from metadata
    /// Different models use different bases (LLaMA: 10000, Qwen2: 1000000)
    pub fn rope_freq_base(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ROPE_FREQ_BASE);
        if let Some(GGUFValue::Float32(base)) = self.metadata.get(&key) {
            Some(*base)
        } else {
            None
        }
    }

    /// PMAT-810: Get Gemma2 attention-logit softcap (`{arch}.attn_logit_softcapping`).
    /// 50.0 for gemma-2-*. Returns None when absent (non-Gemma2, or weights-only GGUF).
    pub fn attn_logit_softcapping(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ATTN_LOGIT_SOFTCAPPING);
        if let Some(GGUFValue::Float32(cap)) = self.metadata.get(&key) {
            Some(*cap)
        } else {
            None
        }
    }

    /// PMAT-810: Get Gemma2 final-logit softcap (`{arch}.final_logit_softcapping`).
    /// 30.0 for gemma-2-*. Returns None when absent (non-Gemma2, or weights-only GGUF).
    pub fn final_logit_softcapping(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::FINAL_LOGIT_SOFTCAPPING);
        if let Some(GGUFValue::Float32(cap)) = self.metadata.get(&key) {
            Some(*cap)
        } else {
            None
        }
    }

    /// PMAT-810: Get Gemma2 pre-attention query scale denominator
    /// (`{arch}.attention.query_pre_attn_scalar`). The attention scale is
    /// `1/sqrt(query_pre_attn_scalar)` rather than `1/sqrt(head_dim)`. 256 for
    /// gemma-2-2b (== head_dim, no-op), 224 for 9b/27b. Returns None when absent
    /// (non-Gemma2, weights-only GGUF, or older converts) so callers fall back to
    /// `head_dim` — exactly llama.cpp's default (`n_embd_head_k`).
    pub fn query_pre_attn_scalar(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::QUERY_PRE_ATTN_SCALAR);
        match self.metadata.get(&key) {
            Some(GGUFValue::UInt32(v)) => Some(*v as f32),
            Some(GGUFValue::Float32(v)) => Some(*v),
            _ => None,
        }
    }

    /// M32c.2.2.2.1.2: Get MoE expert count (`{arch}.expert_count`).
    /// 128 for Qwen3-Coder-30B-A3B-Instruct. Returns None for dense models.
    pub fn expert_count(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::EXPERT_COUNT);
        if let Some(GGUFValue::UInt32(count)) = self.metadata.get(&key) {
            Some(*count as usize)
        } else {
            None
        }
    }

    /// M32c.2.2.2.1.2: Get MoE top-k expert count (`{arch}.expert_used_count`).
    /// 8 for Qwen3-Coder-30B-A3B-Instruct. Returns None for dense models.
    pub fn expert_used_count(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::EXPERT_USED_COUNT);
        if let Some(GGUFValue::UInt32(count)) = self.metadata.get(&key) {
            Some(*count as usize)
        } else {
            None
        }
    }

    /// M32c.2.2.2.1.2: Get MoE per-expert FFN intermediate dim
    /// (`{arch}.expert_feed_forward_length`). 768 for Qwen3-Coder-30B-A3B.
    /// Returns None for dense models.
    pub fn expert_feed_forward_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key =
            crate::gguf::keys::arch_key(arch, crate::gguf::keys::EXPERT_FEED_FORWARD_LENGTH);
        if let Some(GGUFValue::UInt32(len)) = self.metadata.get(&key) {
            Some(*len as usize)
        } else {
            None
        }
    }
}

/// One GGUF type's whole-buffer dequantizer, as [`GGUFModel::dequantize_at`] picks it.
type Dequantizer = fn(&[u8]) -> Result<Vec<f32>>;

#[cfg(test)]
mod dequant_kernel_tests_3418 {
    use crate::gguf::ggml_type_table::{to_float, to_float_tests_3418::blocks, GGML_TYPES};
    use crate::gguf::GGUFModel;

    /// FT-GGMLT-022: `get_tensor_f32` decodes every block type the one table
    /// decodes, as the table does (Q8_0 and Q4_K through their SIMD loaders,
    /// within 1e-4), and trims a partial last block to the asked size.
    #[test]
    fn get_tensor_f32_reads_the_to_float_column_3418() {
        for t in GGML_TYPES.iter().filter(|t| to_float(t.id).is_some()) {
            let data = blocks(t.id, 2);
            let want = to_float(t.id).expect("row")(&data).expect("decode");
            let size = 2 * t.blck_size - 1;
            let got = GGUFModel::dequantize_at(t.id, &data, 0, size)
                .unwrap_or_else(|e| panic!("get_tensor_f32 refused {} ({}): {e}", t.name, t.id));
            assert_eq!(got.len(), size, "{}", t.name);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    g.to_bits() == w.to_bits() || (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                    "{} value {i}: {g} vs the table's {w}",
                    t.name
                );
            }
        }
    }
}
