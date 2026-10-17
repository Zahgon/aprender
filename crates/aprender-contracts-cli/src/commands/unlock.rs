use std::path::Path;

/// The standard stage options (#5205): YAML 1.2 booleans, `<<` read as a plain key.
fn parse(text: &str) -> Result<serde_json::Value, serde_saphyr::Error> {
    serde_saphyr::from_str_with_options(
        text,
        serde_saphyr::options! { strict_booleans: true, merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary, emit_comments: false, reject_non_finite_typeless_float: false },
    )
}

/// `content` without the `locked_level:` line of its top-level `metadata:` block.
/// `None` when there is no such single line. The text is edited rather than
/// re-serialized so key order and comments stay as written (#5209); `run` re-parses
/// the result and refuses it unless it equals the parsed contract minus the key.
fn drop_locked_level_line(content: &str) -> Option<String> {
    let lines: Vec<&str> = content.split_inclusive('\n').collect();
    let start = lines.iter().position(|l| l.trim_end() == "metadata:")?;
    let block = lines[start + 1..]
        .iter()
        .take_while(|l| l.trim().is_empty() || l.starts_with([' ', '#']))
        .count();
    let hit = (start + 1..start + 1 + block)
        .find(|&i| lines[i].trim_start().starts_with("locked_level:"))?;
    let mut out = String::with_capacity(content.len());
    for (i, l) in lines.iter().enumerate() {
        if i != hit {
            out.push_str(l);
        }
    }
    Some(out)
}

pub fn run(contract: &Path, reason: &str) -> Result<(), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(contract)?;

    // Parse to verify it's valid
    let mut contract_obj = parse(&content)?;

    // Remove locked_level from metadata
    let had_lock = contract_obj
        .get_mut("metadata")
        .and_then(serde_json::Value::as_object_mut)
        .and_then(|map| map.remove("locked_level"));
    if had_lock.is_none() {
        println!("Contract has no locked_level -- nothing to unlock.");
        return Ok(());
    }

    // Write back: the same text without the one line, proven equal by a re-parse.
    let new_content = drop_locked_level_line(&content)
        .filter(|text| parse(text).is_ok_and(|v| v == contract_obj))
        .ok_or("metadata.locked_level is not a single plain line; remove it by hand")?;
    std::fs::write(contract, &new_content)?;

    println!("Unlocked: {}", contract.display());
    println!("Reason: {reason}");
    println!("Warning: The contract can now regress below its previous locked level.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::run;

    /// #5209: the contract is written back in its own key order and keeps its comments.
    #[test]
    fn unlock_keeps_the_contracts_key_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("c-v1.yaml");
        std::fs::write(
            &path,
            "# head\nmetadata:\n  version: \"1.0.0\"\n  locked_level: L3\n  description: d\nequations:\n  zeta:\n    formula: z\n  alpha:\n    formula: a\n",
        )
        .expect("write");
        run(&path, "test").expect("unlock");
        let out = std::fs::read_to_string(&path).expect("read");
        assert!(!out.contains("locked_level"), "{out}");
        let pos = |k: &str| out.find(k).unwrap_or_else(|| panic!("{k} missing: {out}"));
        assert!(pos("metadata") < pos("equations"), "{out}");
        assert!(pos("version") < pos("description"), "{out}");
        assert!(pos("zeta") < pos("alpha"), "{out}");
        assert!(out.starts_with("# head\n"), "{out}");
    }

    /// A lock that is not one plain line is refused and the file is left as it was.
    #[test]
    fn unlock_refuses_a_lock_it_cannot_drop_as_one_line() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("c-v1.yaml");
        let text = "metadata: {version: \"1.0.0\", locked_level: L3}\n";
        std::fs::write(&path, text).expect("write");
        assert!(run(&path, "test").is_err());
        assert_eq!(std::fs::read_to_string(&path).expect("read"), text);
    }
}
