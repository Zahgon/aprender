//! A4 (#3422, E7 #4001): a new architecture adds no per-arch forward file to
//! this directory; its forward is composed from the shared attention/FFN
//! blocks. A `forward_*.rs` that is not listed below turns the first test red,
//! so adding one is a visible, reviewed edit to `FORWARD_FILES`.

use std::path::Path;

/// Every non-test `forward_*.rs` in this directory at origin/main 8c73aa916f.
const FORWARD_FILES: &[&str] = &[
    "forward_cached.rs",
    "forward_decode_lean.rs",
    "forward_embedding_gemma2.rs",
    "forward_fused_q4k.rs",
    "forward_profiled.rs",
    "forward_qwen35.rs",
    "forward_qwen3_moe.rs",
    "forward_qwen3_moe_gpu.rs",
    "forward_qwen3_moe_traced.rs",
    "forward_single_profiled.rs",
];

/// The non-test `forward_*.rs` file names in `dir`, sorted.
fn forward_files_in(dir: &Path) -> Vec<String> {
    let mut found: Vec<String> = std::fs::read_dir(dir)
        .expect("forward directory is readable")
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .filter(|name| {
            name.starts_with("forward_")
                && !name.ends_with("_tests.rs")
                && Path::new(name).extension().is_some_and(|ext| ext == "rs")
        })
        .collect();
    found.sort();
    found
}

/// The names in `found` that `FORWARD_FILES` does not list.
fn unlisted(found: &[String]) -> Vec<&str> {
    found
        .iter()
        .map(String::as_str)
        .filter(|name| !FORWARD_FILES.contains(name))
        .collect()
}

#[test]
fn no_forward_file_is_added_outside_the_allow_list() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gguf/inference/forward");
    let found = forward_files_in(&dir);
    assert!(
        found.len() >= FORWARD_FILES.len(),
        "scanned {} forward files in {}, the list has {}",
        found.len(),
        dir.display(),
        FORWARD_FILES.len()
    );
    assert_eq!(
        unlisted(&found),
        Vec::<&str>::new(),
        "a new per-arch forward file: compose the forward from the shared blocks (#3422)"
    );
}

#[test]
fn a_planted_forward_file_is_reported() {
    let dir = std::env::temp_dir().join(format!("a4-forward-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    for name in [
        "forward_qwen35.rs",
        "forward_newarch.rs",
        "forward_newarch_tests.rs",
        "ffn_block.rs",
    ] {
        std::fs::write(dir.join(name), "").expect("write fixture");
    }
    let found = forward_files_in(&dir);
    std::fs::remove_dir_all(&dir).expect("clean temp dir");
    assert_eq!(found, ["forward_newarch.rs", "forward_qwen35.rs"]);
    assert_eq!(unlisted(&found), ["forward_newarch.rs"]);
}
