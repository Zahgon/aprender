//! Q4_0 GPU Adapter - Direct Quantized Inference (PMAT-803)
//!
//! Keeps weights in Q4_0 format on GPU, using `q4_0_gemv_into` kernels.
//! This eliminates the 8x bandwidth overhead of F32 dequantization.
//!
//! # Performance Target
//!
//! >50 tok/s on TinyLlama-1.1B (4x improvement over F32 GPU)
//!
//! # Architecture
//!
//! ```text
//! QuantizedAprTransformerQ4 (CPU, Q4_0 weights)
//!         │
//!         ▼ AprQ4ToGpuAdapter::upload_weights()
//!         │
//! CudaExecutor (GPU, Q4_0 cached)
//!         │
//!         ▼ GpuModelQ4::forward()
//!         │
//! Logits (F32)
//! ```

#![allow(clippy::similar_names)]

use crate::apr_transformer::{
    AprTransformerConfig, QuantizedAprTensorQ4, QuantizedAprTransformerQ4,
};
#[cfg(feature = "cuda")]
use crate::cuda::CudaExecutor;
use crate::error::{RealizarError, Result};
#[cfg(feature = "cuda")]
use trueno_gpu::driver::GpuBuffer;

/// Adapter for uploading Q4_0 APR models to GPU
///
/// Unlike `AprToGpuAdapter`, this keeps weights quantized on GPU.
pub struct AprQ4ToGpuAdapter;

impl AprQ4ToGpuAdapter {
    /// #3418 (FT-GGMLT-029): the GGML type to upload `tensor` as, read from
    /// the tensor's own `qtype`. The GPU Q4 forward runs only the Q4_0 gemv,
    /// so any other type is refused here, before a byte is uploaded, instead
    /// of being labelled and run as Q4_0.
    ///
    /// # Errors
    ///
    /// Returns error if `tensor` is not Q4_0.
    pub fn gpu_q4_weight_type(name: &str, tensor: &QuantizedAprTensorQ4) -> Result<u32> {
        if tensor.qtype == crate::gguf::GGUF_TYPE_Q4_0 {
            return Ok(tensor.qtype);
        }
        Err(RealizarError::FormatError {
            reason: format!(
                "GPU Q4 adapter: {name} is GGML type {} ({}); the GPU forward runs only \
                 the Q4_0 kernel (#3418)",
                tensor.qtype,
                crate::gguf::ggml_type_table::name(tensor.qtype).unwrap_or("no table row")
            ),
        })
    }

    /// Every weight `upload_weights` sends, with its cache name, in upload
    /// order, each typed by [`Self::gpu_q4_weight_type`]; the first refusal
    /// stops the whole upload.
    ///
    /// # Errors
    ///
    /// Returns error if any weight is not Q4_0.
    pub fn typed_weights(apr: &QuantizedAprTransformerQ4) -> Result<Vec<(String, &[u8], u32)>> {
        let mut named: Vec<(String, &QuantizedAprTensorQ4)> = Vec::new();
        for (layer_idx, layer) in apr.layers.iter().enumerate() {
            named.push((format!("layer_{layer_idx}.attn.qkv"), &layer.qkv_weight));
            named.push((
                format!("layer_{layer_idx}.attn.out"),
                &layer.attn_output_weight,
            ));
            named.push((format!("layer_{layer_idx}.ffn.up"), &layer.ffn_up_weight));
            named.push((
                format!("layer_{layer_idx}.ffn.down"),
                &layer.ffn_down_weight,
            ));
            if let Some(ref gate_weight) = layer.ffn_gate_weight {
                named.push((format!("layer_{layer_idx}.ffn.gate"), gate_weight));
            }
        }
        named.push(("lm_head".to_string(), &apr.lm_head_weight));
        named
            .into_iter()
            .map(|(name, t)| {
                let qtype = Self::gpu_q4_weight_type(&name, t)?;
                Ok((name, t.data.as_slice(), qtype))
            })
            .collect()
    }

    /// Upload Q4_0 weights to CUDA executor
    ///
    /// Weights are cached on GPU by layer name for reuse across forward passes.
    /// No dequantization is performed - weights stay in Q4_0 format.
    ///
    /// # Arguments
    ///
    /// * `apr` - Source APR transformer with Q4_0 weights
    /// * `executor` - CUDA executor with weight cache
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU
    ///
    /// # Errors
    ///
    /// Returns error if a weight is not Q4_0 or GPU upload fails.
    #[cfg(feature = "cuda")]
    pub fn upload_weights(
        apr: &QuantizedAprTransformerQ4,
        executor: &mut CudaExecutor,
    ) -> Result<usize> {
        let mut total_bytes = 0;

        // #3418 (FT-GGMLT-029): every weight is typed (and a non-Q4_0 one
        // refused) before anything is uploaded; norms are kept on CPU.
        for (name, data, qtype) in Self::typed_weights(apr)? {
            total_bytes += executor
                .load_quantized_weights_with_type(&name, data, qtype)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("Failed to upload {name}: {e}"),
                })?;
        }

        // PAR-023: Upload norm weights to GPU for GPU-resident RMSNorm
        // This eliminates CPU roundtrips for normalization operations
        for (layer_idx, layer) in apr.layers.iter().enumerate() {
            // Attention norm
            let attn_norm_name = format!("apr.layer_{layer_idx}.attn_norm");
            let attn_norm_bytes = executor
                .cache_rmsnorm_gamma(&attn_norm_name, &layer.attn_norm_weight)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("Failed to cache {attn_norm_name}: {e}"),
                })?;
            total_bytes += attn_norm_bytes;

            // FFN norm (use ones if not present)
            let ffn_norm_name = format!("apr.layer_{layer_idx}.ffn_norm");
            let ffn_norm = layer
                .ffn_norm_weight
                .as_ref()
                .map_or_else(|| vec![1.0f32; apr.config.hidden_dim], Clone::clone);
            let ffn_norm_bytes = executor
                .cache_rmsnorm_gamma(&ffn_norm_name, &ffn_norm)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!("Failed to cache {ffn_norm_name}: {e}"),
                })?;
            total_bytes += ffn_norm_bytes;
        }

        // Output norm
        let output_norm_bytes = executor
            .cache_rmsnorm_gamma("apr.output_norm", &apr.output_norm_weight)
            .map_err(|e| RealizarError::GpuError {
                reason: format!("Failed to cache apr.output_norm: {e}"),
            })?;
        total_bytes += output_norm_bytes;

        Ok(total_bytes)
    }

    /// Create a GPU model wrapper for Q4_0 inference
    ///
    /// # Arguments
    ///
    /// * `apr` - Source APR transformer (for config, embeddings, and norm weights)
    ///
    /// # Returns
    ///
    /// `GpuModelQ4` ready for inference
    #[must_use]
    pub fn create_model(apr: &QuantizedAprTransformerQ4) -> GpuModelQ4 {
        // Extract layer norms (small, keep on CPU)
        let layer_norms: Vec<LayerNorms> = apr
            .layers
            .iter()
            .map(|layer| LayerNorms {
                attn_norm: layer.attn_norm_weight.clone(),
                ffn_norm: layer
                    .ffn_norm_weight
                    .clone()
                    .unwrap_or_else(|| vec![1.0; apr.config.hidden_dim]),
            })
            .collect();

        GpuModelQ4 {
            config: apr.config.clone(),
            token_embedding: apr.token_embedding.clone(),
            output_norm_weight: apr.output_norm_weight.clone(),
            layer_norms,
            num_layers: apr.layers.len(),
            has_gate: apr
                .layers
                .first()
                .is_some_and(|l| l.ffn_gate_weight.is_some()),
        }
    }
}

/// Layer normalization weights (small, kept on CPU)
#[derive(Debug, Clone)]
pub struct LayerNorms {
    /// Attention norm weight
    pub attn_norm: Vec<f32>,
    /// FFN norm weight
    pub ffn_norm: Vec<f32>,
}

/// GPU model for Q4_0 inference
///
/// Lightweight wrapper that holds config and uses CudaExecutor for computation.
/// Large Q4_0 weights are stored on GPU, small norm weights on CPU.
#[derive(Debug, Clone)]
pub struct GpuModelQ4 {
    /// Model configuration
    pub config: AprTransformerConfig,
    /// Token embedding (F32, CPU copy for fast lookup)
    pub token_embedding: Vec<f32>,
    /// Output norm weight (F32, CPU)
    pub output_norm_weight: Vec<f32>,
    /// Per-layer norm weights (F32, CPU)
    pub layer_norms: Vec<LayerNorms>,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Whether model uses SwiGLU (has gate projection)
    pub has_gate: bool,
}

/// Helper to download GPU buffer to host Vec
#[cfg(feature = "cuda")]
fn gpu_to_host(buf: &GpuBuffer<f32>) -> Result<Vec<f32>> {
    let mut host = vec![0.0f32; buf.len()];
    buf.copy_to_host(&mut host)
        .map_err(|e| RealizarError::GpuError {
            reason: format!("GPU->CPU copy failed: {e}"),
        })?;
    Ok(host)
}

include!("apr_q4_apply_rope_gpu.rs");
include!("activation.rs");
