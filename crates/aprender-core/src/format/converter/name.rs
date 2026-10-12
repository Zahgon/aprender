/// Detect GGUF model architecture for tensor name mapping (GH-200).
/// GH-236: Added GPT-2 recognition — was falling through to Qwen2 default,
/// causing metadata key mismatch (writes "qwen2.embedding_length" but reader
/// looks for "gpt2.embedding_length") → hidden_dim=0 on reimport.
/// #4419: an unreadable header, a missing `general.architecture` or an
/// unrecognized one is refused by name. It used to become Qwen2, a plausible
/// value that maps a foreign tensor set with the wrong name table.
fn detect_gguf_architecture(path: &Path) -> Result<Architecture> {
    let reader = GgufReader::from_file(path).map_err(|e| AprenderError::FormatError {
        message: format!(
            "cannot read the GGUF header of {} to choose a tensor-name table: {e}",
            path.display()
        ),
    })?;
    gguf_name_table_architecture(reader.architecture().as_deref())
}

/// The name table for a GGUF `general.architecture` value, or a refusal.
///
/// GGUF spells some families differently from `config.json`. `qwen35`,
/// `qwen3moe` and `qwen2moe` use the Qwen2 GGUF tensor names, the table they
/// were always mapped with, so they are named here, not left to a default.
fn gguf_name_table_architecture(arch: Option<&str>) -> Result<Architecture> {
    let arch = arch.ok_or_else(|| AprenderError::FormatError {
        message: "GGUF has no general.architecture; refusing to guess a tensor-name table (#4419)"
            .to_string(),
    })?;
    let known = match arch {
        "qwen35" => Some(Architecture::Qwen3_5),
        "qwen3moe" => Some(Architecture::Qwen3),
        "qwen2moe" => Some(Architecture::Qwen2),
        other => Architecture::from_model_type(other),
    };
    known.ok_or_else(|| AprenderError::FormatError {
        message: format!(
            "GGUF general.architecture '{arch}' has no tensor-name table; \
             refusing to map it as qwen2 (#4419)"
        ),
    })
}

/// Infer vocab_size and hidden_dim from tensor shapes.
///
/// Find the first 2D tensor matching any of the given name patterns.
fn find_2d_tensor_shape<'a>(
    tensors: &'a BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    patterns: &[&str],
) -> Option<&'a [usize]> {
    tensors.iter().find_map(|(name, (_, shape))| {
        if shape.len() == 2 && patterns.iter().any(|p| name.contains(p)) {
            Some(shape.as_slice())
        } else {
            None
        }
    })
}

/// Used for contract validation during export.
fn infer_vocab_hidden(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> (usize, usize) {
    // Try embedding, then lm_head for [vocab_size, hidden_dim]
    if let Some(shape) = find_2d_tensor_shape(tensors, &["embed_tokens", "token_embd"]) {
        return (shape[0], shape[1]);
    }
    if let Some(shape) = find_2d_tensor_shape(tensors, &["lm_head", "output.weight"]) {
        return (shape[0], shape[1]);
    }
    // Fallback: get hidden_dim from q_proj
    let hidden = find_2d_tensor_shape(tensors, &["q_proj"]).map_or(0, |s| s[1]);
    (0, hidden)
}

#[cfg(test)]
#[path = "export_tests.rs"]
mod tests;

#[cfg(test)]
mod gguf_name_table_tests_4419 {
    use super::gguf_name_table_architecture as table;
    use super::Architecture;

    /// #4419: what used to become Qwen2 by default is refused by name.
    #[test]
    fn a_missing_or_unknown_gguf_architecture_is_refused_not_qwen2() {
        let missing = table(None).expect_err("no general.architecture must be refused");
        assert!(
            missing.to_string().contains("no general.architecture"),
            "{missing}"
        );
        let unknown = table(Some("command-r")).expect_err("an unknown family must be refused");
        assert!(
            unknown.to_string().contains("'command-r'"),
            "names it: {unknown}"
        );
    }

    /// The GGUF spellings that were mapped with the Qwen2 table keep it, by name.
    #[test]
    fn gguf_spellings_keep_the_table_they_always_had() {
        for (arch, want) in [
            ("qwen2", Architecture::Qwen2),
            ("qwen3", Architecture::Qwen3),
            ("qwen35", Architecture::Qwen3_5),
            ("qwen3moe", Architecture::Qwen3),
            ("qwen2moe", Architecture::Qwen2),
            ("llama", Architecture::Llama),
            ("gpt2", Architecture::Gpt2),
        ] {
            assert_eq!(table(Some(arch)).expect(arch), want, "{arch}");
        }
    }
}
