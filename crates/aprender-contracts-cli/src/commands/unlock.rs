use std::path::Path;

pub fn run(contract: &Path, reason: &str) -> Result<(), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(contract)?;

    // Parse to verify it's valid
    let mut contract_obj: serde_json::Value = serde_saphyr::from_str_with_options(
        &content,
        serde_saphyr::options! { strict_booleans: true, merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary, emit_comments: false, reject_non_finite_typeless_float: false },
    )?;

    // Remove locked_level from metadata
    if let Some(metadata) = contract_obj.get_mut("metadata") {
        if let Some(map) = metadata.as_object_mut() {
            let had_lock = map.remove("locked_level");
            if had_lock.is_none() {
                println!("Contract has no locked_level -- nothing to unlock.");
                return Ok(());
            }
        }
    }

    // Write back
    let new_content = serde_saphyr::to_string(&contract_obj)?;
    std::fs::write(contract, &new_content)?;

    println!("Unlocked: {}", contract.display());
    println!("Reason: {reason}");
    println!("Warning: The contract can now regress below its previous locked level.");

    Ok(())
}
