//! #5716 — behaviour of the public `run_roofline` entry (`cgp roofline`).
#![allow(clippy::disallowed_methods)]

use cgp::analysis::roofline::{run_roofline, MemoryLevel, Precision, RooflineModel};

fn exported(target: &str) -> RooflineModel {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.json");
    let p = path.to_str().expect("utf8 path");
    run_roofline(target, None, Some(p), false, false).expect("run_roofline");
    let text = std::fs::read_to_string(&path).expect("export file written");
    serde_json::from_str(&text).expect("export is a RooflineModel")
}

#[test]
fn unknown_target_is_an_error_naming_it_and_the_supported_set() {
    let err = run_roofline("tpu", None, None, false, false).expect_err("must reject");
    let msg = err.to_string();
    assert!(msg.contains("Unknown roofline target: tpu"), "{msg}");
    for t in ["cuda", "avx2", "avx512", "neon", "wgpu"] {
        assert!(msg.contains(t), "supported list misses {t}: {msg}");
    }
}

#[test]
fn export_writes_the_selected_model() {
    let cuda = exported("cuda");
    assert!(cuda.target.contains("RTX 4090"), "{}", cuda.target);
    assert_eq!(
        cuda.target,
        exported("wgpu").target,
        "wgpu reuses the 4090 spec"
    );
    assert!(exported("avx2").target.contains("AVX2"));
    assert!(exported("avx512").target.contains("AVX-512"));
    assert!(exported("neon").target.contains("NEON"));
}

#[test]
fn exported_model_keeps_its_ridge_point() {
    let want = RooflineModel::rtx_4090()
        .ridge_point(Precision::Fp16, MemoryLevel::Dram)
        .expect("ridge");
    let got = exported("cuda")
        .ridge_point(Precision::Fp16, MemoryLevel::Dram)
        .expect("ridge after roundtrip");
    assert!((want - got).abs() < 1e-9, "{want} vs {got}");
}

#[test]
fn json_mode_returns_before_export_so_no_file_is_written() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("never.json");
    let p = path.to_str().expect("utf8 path");
    run_roofline("cuda", None, Some(p), false, true).expect("json run");
    assert!(!path.exists(), "--json must not also export");
}

#[test]
fn gpu_targets_skip_the_empirical_probe_but_still_export() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("gpu.json");
    let p = path.to_str().expect("utf8 path");
    run_roofline("cuda", None, Some(p), true, false).expect("cuda empirical");
    assert!(path.exists());
    run_roofline("wgpu", None, None, true, true).expect("wgpu json empirical");
}

#[test]
fn export_to_an_unwritable_path_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("no_such_dir").join("m.json");
    let p = path.to_str().expect("utf8 path");
    assert!(run_roofline("avx2", None, Some(p), false, false).is_err());
}
