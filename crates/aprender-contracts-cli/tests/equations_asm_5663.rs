//! `pv equations --format asm` (`render_asm`): the ISA label, symbol suffix, register file and
//! body comments follow the contract's `simd_dispatch` keys; AVX2 is covered by the
//! ground-truth snapshots, so this pins the other three tiers and the comment fallbacks (#5663).
#![allow(clippy::disallowed_methods)]

use std::process::Command;

fn contract(dispatch: &str, structure: &str) -> String {
    format!(
        "metadata:\n  version: \"1.0.0\"\n  description: \"Probe kernel\"\n\
         equations:\n  f:\n    formula: \"y = x + 1\"\n\
         proof_obligations:\n  - type: invariant\n    property: \"monotone\"\n{structure}{dispatch}"
    )
}

fn asm(stem: &str, body: &str) -> String {
    let tmp = tempfile::tempdir().expect("tmp");
    let path = tmp.path().join(format!("{stem}.yaml"));
    std::fs::write(&path, body).expect("write");
    let o = Command::new(env!("CARGO_BIN_EXE_pv"))
        .args(["equations", "--format", "asm"])
        .arg(&path)
        .output()
        .expect("spawn pv");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn avx512_key_selects_zmm_with_32_registers() {
    let s = asm(
        "probe-kernel-v1",
        &contract("simd_dispatch:\n  f:\n    avx512: \"f_avx512\"\n", ""),
    );
    assert!(s.contains("// x86-64 AVX-512 stub: probe-kernel-v1"), "{s}");
    assert!(s.contains(".globl probe_avx512"), "{s}");
    assert!(s.contains("// AVX-512 registers: zmm x 32"), "{s}");
    assert!(s.contains("vxorps zmm3, zmm3, zmm3"), "{s}");
    assert!(
        !s.contains("vxorps zmm4"),
        "only 4 registers are initialised: {s}"
    );
}

#[test]
fn a_bare_512_key_also_selects_avx512() {
    let s = asm(
        "probe-v1",
        &contract("simd_dispatch:\n  f:\n    w512: \"x\"\n", ""),
    );
    assert!(s.contains(".globl probe_avx512"), "{s}");
}

#[test]
fn non_avx_dispatch_falls_back_to_sse41_xmm() {
    let s = asm(
        "probe-v1",
        &contract("simd_dispatch:\n  f:\n    neon: \"f_neon\"\n", ""),
    );
    assert!(s.contains("// x86-64 SSE4.1 stub"), "{s}");
    assert!(s.contains("probe_sse41:"), "{s}");
    assert!(s.contains("// SSE4.1 registers: xmm x 16"), "{s}");
}

#[test]
fn no_dispatch_is_scalar_with_no_register_init() {
    let s = asm("probe-v1", &contract("", ""));
    assert!(s.contains("// x86-64 scalar stub"), "{s}");
    assert!(s.contains("probe_scalar:"), "{s}");
    assert!(!s.contains("vxorps"), "{s}");
    assert!(s.contains("    pop rbp\n    ret"), "{s}");
}

#[test]
fn without_kernel_structure_the_equations_become_the_body_comments() {
    let s = asm("probe-v1", &contract("", ""));
    assert!(s.contains("    // Equation: f\n    // y = x + 1"), "{s}");
    assert!(
        s.contains("    // Proof obligations:\n    //   [invariant] monotone"),
        "{s}"
    );
    assert!(!s.contains("// Phase 1"), "{s}");
}

#[test]
fn kernel_structure_phases_replace_the_equation_comments() {
    let ks = "kernel_structure:\n  phases:\n    - name: add\n      description: \"add one\"\n      invariant: \"y > x\"\n";
    let s = asm("probe-v1", &contract("", ks));
    assert!(
        s.contains("    // Phase 1: add\n    // add one\n    // Invariant: y > x"),
        "{s}"
    );
    assert!(!s.contains("    // Equation: f"), "{s}");
}
