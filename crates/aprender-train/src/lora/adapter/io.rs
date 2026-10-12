//! LoRA adapter I/O convenience functions

use super::error::AdapterError;
use super::lora_adapter::LoRAAdapter;
use super::peft_export::PeftAdapterBundle;
use crate::lora::{LoRAConfig, LoRALayer};
use crate::Tensor;
use std::path::Path;

/// Adapter serialization format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterFormat {
    /// Entrenar's native JSON format (single-layer)
    EntrenarJson,
    /// HuggingFace PEFT format (adapter_config.json + adapter_model.safetensors)
    Peft,
}

/// Save LoRA adapter to file (Entrenar JSON format)
///
/// # Arguments
/// * `layer` - LoRALayer to save
/// * `rank` - LoRA rank
/// * `alpha` - LoRA alpha parameter
/// * `path` - File path to save to
pub fn save_adapter<P: AsRef<Path>>(
    layer: &LoRALayer,
    rank: usize,
    alpha: f32,
    path: P,
) -> Result<(), AdapterError> {
    let adapter = LoRAAdapter::from_layer(layer, rank, alpha);
    adapter.save(path)
}

/// Load LoRA adapter from file (Entrenar JSON format)
///
/// # Arguments
/// * `base_weight` - Frozen base weight to apply adapter to
/// * `path` - File path to load from
pub fn load_adapter<P: AsRef<Path>>(
    base_weight: Tensor,
    path: P,
) -> Result<LoRALayer, AdapterError> {
    let adapter = LoRAAdapter::load(path)?;
    adapter.to_layer(base_weight)
}

/// Save LoRA adapters in PEFT-compatible format
///
/// # Arguments
/// * `adapters` - Layer path to LoRA layer mappings
/// * `config` - LoRA configuration
/// * `base_model` - Optional base model name for adapter_config.json
/// * `output_dir` - Output directory (will contain adapter_config.json + adapter_model.safetensors)
pub fn save_adapter_peft<P: AsRef<Path>>(
    adapters: &[(&str, &LoRALayer)],
    config: &LoRAConfig,
    base_model: Option<&str>,
    output_dir: P,
) -> Result<(), AdapterError> {
    let mut bundle = PeftAdapterBundle::new(config.clone());
    if let Some(name) = base_model {
        bundle = bundle.with_base_model(name);
    }
    for (path, layer) in adapters {
        bundle.add_adapter(*path, layer);
    }
    bundle.save_peft(output_dir)
}

/// Load LoRA adapter from PEFT-compatible format
///
/// Reads `adapter_config.json` and `adapter_model.safetensors` from the given directory
/// and returns the adapter configuration along with tensor name → weight data.
pub fn load_adapter_peft<P: AsRef<Path>>(
    dir: P,
) -> Result<(super::peft_config::PeftAdapterConfig, Vec<(String, Vec<f32>)>), AdapterError> {
    let dir = dir.as_ref();

    // Read adapter_config.json
    let config_path = dir.join("adapter_config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config = super::peft_config::PeftAdapterConfig::from_json(&config_str)
        .map_err(|e| AdapterError::PeftFormatError(format!("Invalid adapter_config.json: {e}")))?;

    // Read adapter_model.safetensors
    let model_path = dir.join("adapter_model.safetensors");
    let model_data = std::fs::read(&model_path)?;
    let tensors = safetensors::SafeTensors::deserialize(&model_data).map_err(|e| {
        AdapterError::SafeTensors(format!("Failed to load adapter_model.safetensors: {e}"))
    })?;

    // Sorted so the result never depends on the safetensors header's HashMap order (#4697).
    let mut names: Vec<&str> = tensors.names();
    names.sort();
    let mut weights = Vec::with_capacity(names.len());
    for name in names {
        let tensor = tensors.tensor(name).map_err(|e| {
            AdapterError::SafeTensors(format!("Failed to read tensor '{name}': {e}"))
        })?;
        let data = decode_peft_tensor(name, &tensor)?;
        weights.push((name.to_string(), data));
    }

    Ok((config, weights))
}

/// Decode one adapter tensor to f32 by its declared dtype (#4697).
///
/// F32, BF16 and F16 are widened exactly; any other dtype is refused, and so is a
/// tensor whose element count does not match its shape. Reading the bytes as f32
/// regardless of dtype turned a bf16/f16 adapter into half-length garbage.
fn decode_peft_tensor(
    name: &str,
    tensor: &safetensors::tensor::TensorView<'_>,
) -> Result<Vec<f32>, AdapterError> {
    use safetensors::Dtype;
    let bytes = tensor.data();
    let data: Vec<f32> = match tensor.dtype() {
        Dtype::F32 => {
            bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::F16 => {
            bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
        }
        other => {
            return Err(AdapterError::PeftFormatError(format!(
                "tensor '{name}' has dtype {other:?}; only F32, BF16 and F16 adapters load"
            )))
        }
    };
    let expected: usize = tensor.shape().iter().product();
    if data.len() != expected {
        return Err(AdapterError::DimensionMismatch {
            expected: format!("{expected} elements for '{name}' (shape {:?})", tensor.shape()),
            actual: format!("{} elements", data.len()),
        });
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_adapter_format_eq() {
        assert_eq!(AdapterFormat::EntrenarJson, AdapterFormat::EntrenarJson);
        assert_eq!(AdapterFormat::Peft, AdapterFormat::Peft);
        assert_ne!(AdapterFormat::EntrenarJson, AdapterFormat::Peft);
    }

    #[test]
    fn test_save_load_peft_roundtrip() {
        let config = LoRAConfig::new(4, 8.0).target_qv_projections();

        let base = Tensor::zeros(8 * 16, false);
        let layer = LoRALayer::new(base, 8, 16, 4, 8.0);

        let tmp = TempDir::new().expect("temp file creation should succeed");
        save_adapter_peft(
            &[("model.layers.0.self_attn.q_proj", &layer)],
            &config,
            Some("test/model"),
            tmp.path(),
        )
        .expect("operation should succeed");

        let (loaded_config, weights) = load_adapter_peft(tmp.path()).expect("load should succeed");
        assert_eq!(loaded_config.r, 4);
        assert_eq!(loaded_config.lora_alpha, 8.0);
        assert_eq!(weights.len(), 2); // lora_A + lora_B
    }

    /// Writes a valid adapter_config.json via save_adapter_peft, then replaces the
    /// safetensors file with `tensors` (name, dtype, shape, raw little-endian bytes).
    fn write_adapter(tmp: &TempDir, tensors: &[(&str, safetensors::Dtype, Vec<usize>, Vec<u8>)]) {
        let config = LoRAConfig::new(4, 8.0).target_qv_projections();
        let layer = LoRALayer::new(Tensor::zeros(8 * 16, false), 8, 16, 4, 8.0);
        save_adapter_peft(
            &[("model.layers.0.self_attn.q_proj", &layer)],
            &config,
            None,
            tmp.path(),
        )
        .expect("config write should succeed");
        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(n, d, s, b)| {
                let v = safetensors::tensor::TensorView::new(*d, s.clone(), b)
                    .expect("view should be valid");
                ((*n).to_string(), v)
            })
            .collect();
        let bytes = safetensors::serialize(views, None).expect("serialize should succeed");
        std::fs::write(tmp.path().join("adapter_model.safetensors"), bytes)
            .expect("write should succeed");
    }

    const VALUES: [f32; 4] = [1.0, -2.5, 0.375, 1024.0]; // exact in bf16 and f16

    #[test]
    fn falsify_4697_bf16_adapter_loads_full_length_exact_values() {
        let bytes: Vec<u8> =
            VALUES.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect();
        let tmp = TempDir::new().expect("temp dir");
        write_adapter(&tmp, &[("a.lora_A.weight", safetensors::Dtype::BF16, vec![2, 2], bytes)]);
        let (_, weights) = load_adapter_peft(tmp.path()).expect("bf16 adapter should load");
        assert_eq!(weights, vec![("a.lora_A.weight".to_string(), VALUES.to_vec())]);
    }

    #[test]
    fn falsify_4697_f16_adapter_loads_full_length_exact_values() {
        let bytes: Vec<u8> =
            VALUES.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
        let tmp = TempDir::new().expect("temp dir");
        write_adapter(&tmp, &[("a.lora_B.weight", safetensors::Dtype::F16, vec![4], bytes)]);
        let (_, weights) = load_adapter_peft(tmp.path()).expect("f16 adapter should load");
        assert_eq!(weights, vec![("a.lora_B.weight".to_string(), VALUES.to_vec())]);
    }

    #[test]
    fn falsify_4697_f32_adapter_values_unchanged() {
        let bytes: Vec<u8> = VALUES.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tmp = TempDir::new().expect("temp dir");
        write_adapter(&tmp, &[("a.lora_A.weight", safetensors::Dtype::F32, vec![1, 4], bytes)]);
        let (_, weights) = load_adapter_peft(tmp.path()).expect("f32 adapter should load");
        assert_eq!(weights, vec![("a.lora_A.weight".to_string(), VALUES.to_vec())]);
    }

    #[test]
    fn falsify_4697_unsupported_dtype_is_refused() {
        let tmp = TempDir::new().expect("temp dir");
        write_adapter(&tmp, &[("a.lora_A.weight", safetensors::Dtype::I32, vec![2], vec![0; 8])]);
        let err = load_adapter_peft(tmp.path()).expect_err("an I32 adapter must not load");
        assert!(err.to_string().contains("I32"), "error should name the dtype: {err}");
    }

    #[test]
    fn falsify_4697_tensors_return_in_sorted_name_order() {
        let one = |v: f32| v.to_le_bytes().to_vec();
        let names = ["z.lora_B.weight", "a.lora_A.weight", "m.lora_A.weight", "b.lora_B.weight"];
        let tensors: Vec<_> =
            names.iter().map(|n| (*n, safetensors::Dtype::F32, vec![1], one(1.0))).collect();
        let tmp = TempDir::new().expect("temp dir");
        write_adapter(&tmp, &tensors);
        let (_, weights) = load_adapter_peft(tmp.path()).expect("load should succeed");
        let got: Vec<&str> = weights.iter().map(|(n, _)| n.as_str()).collect();
        let mut want = names.to_vec();
        want.sort_unstable();
        assert_eq!(got, want);
    }
}
