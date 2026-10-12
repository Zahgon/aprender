//! KTEST-02 (KTEST-001 §3.2): the f64 scalar oracle and the per-element margin harness.
//!
//! A kernel is judged element by element: mᵢ = |ŷᵢ − yᵢ| / Bᵢ, where yᵢ is the oracle's f64 value
//! and Bᵢ is the bound the kernel's declared error model gives for that element. The kernel passes
//! iff max mᵢ ≤ 1 (plus the NaN/Inf rules in [`margin`]). A global NMSE is reported next to it so
//! receipts stay comparable with llama.cpp's `test-backend-ops`, but NMSE never decides: one wrong
//! tile in a million outputs moves it by ~1e-11 (F-12, tested in [`margin`]).
//!
//! The oracle is independent by construction: it is std-only and names nothing outside this
//! module, so it cannot call the optimized backend it judges (§0.3 "the producer is never the
//! gate"). It is a test-only module of this crate rather than its own crate because a path-only
//! dev-dependency is stripped at publish, so `src/` may not name one;
//! `oracle_reaches_nothing_outside_itself` scans these files and refuses any other path.
//!
//! Quantization error is not the kernel's (§0.4): feed the oracle the dequantized weights and, for
//! a kernel that quantizes its activations, the round-tripped activations it actually consumed.

pub mod error_model;
pub mod margin;
pub mod oracle;

pub use error_model::{Bound, Dtype, ErrorModel, Refusal};
pub use margin::{judge, Failure, Report, Verdict};

#[cfg(test)]
mod tests {
    /// Every path in `src` (comments stripped) that could reach code outside the oracle: a
    /// `crate::` path not under `crate::kernel_oracle::`, a `super::super`, an `extern crate`, or
    /// a path rooted at any crate other than `std`/`core`/`alloc`.
    fn escapes(src: &str) -> Vec<String> {
        let mut found = Vec::new();
        for line in src.lines() {
            let code = line.split("//").next().unwrap_or("");
            let mut rest = code;
            while let Some(i) = rest.find("::") {
                let head = &rest[..i];
                let root: String = head
                    .chars()
                    .rev()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                let after = &rest[i + 2..];
                let bad = match root.as_str() {
                    "crate" => !after.starts_with("kernel_oracle::"),
                    "super" => after.starts_with("super"),
                    "" | "std" | "core" | "alloc" | "self" | "Self" => false,
                    r => r.starts_with(|c: char| c.is_ascii_lowercase()) && is_crate_root(r, head),
                };
                if bad {
                    found.push(code.trim().to_string());
                }
                rest = after;
            }
            if code.contains("extern crate") {
                found.push(code.trim().to_string());
            }
        }
        found
    }

    /// A lowercase path segment is a crate root iff nothing path-like precedes it.
    fn is_crate_root(root: &str, head: &str) -> bool {
        let before = head[..head.len() - root.len()].trim_end();
        !(before.ends_with("::") || before.ends_with('.'))
            && !LOCAL_MODULES.contains(&root)
            && !PRIMITIVES.contains(&root)
    }

    /// The oracle's own modules, which its files may name by a bare path.
    const LOCAL_MODULES: [&str; 4] = ["error_model", "margin", "oracle", "tests"];

    /// Primitive types, whose associated items (`f64::from`, `u32::MAX`) are not crate paths.
    const PRIMITIVES: [&str; 15] = [
        "f32", "f64", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64",
        "i128", "isize", "bool",
    ];

    /// §0.3: the oracle must not be able to reach an optimized backend, so no file of it may name
    /// a path outside this module.
    #[test]
    fn oracle_reaches_nothing_outside_itself() {
        for (name, src) in [
            ("mod.rs", include_str!("mod.rs")),
            ("error_model.rs", include_str!("error_model.rs")),
            ("margin.rs", include_str!("margin.rs")),
            ("oracle.rs", include_str!("oracle.rs")),
        ] {
            let found = escapes(src);
            assert!(
                found.is_empty(),
                "KTEST-02 §0.3: the oracle must stay std-only, {name} reaches out: {found:?}"
            );
        }
    }

    /// Anti-vacuity: each planted escape is seen, and the oracle's own idioms are not.
    #[test]
    fn escape_scan_sees_a_planted_path() {
        for planted in [
            "use crate::quantize::fused_q4k_parallel_matvec;",
            "let y = crate::gguf::GGUFConfig::default();",
            "use super::super::kernel_registry;",
            "use trueno::Vector;",
            "let v = realizar::quantize::dequantize(x);",
            "extern crate trueno;",
        ] {
            assert_eq!(escapes(planted), vec![planted.to_string()], "{planted}");
        }
        for clean in [
            "use super::error_model::{ErrorModel, Refusal};",
            "use crate::kernel_oracle::oracle;",
            "use super::*;",
            "let n = std::f64::consts::PI + f64::EPSILON;",
            "let v: Vec<f64> = x.iter().map(|v| f64::from(*v)).collect();",
            "let e = ErrorModel::Dot { k, acc: Dtype::F32, out: Dtype::F32, ftz: false };",
            "let r = oracle::dot(&a, &b); // crate::gguf in a comment",
        ] {
            assert!(escapes(clean).is_empty(), "{clean}: {:?}", escapes(clean));
        }
    }
}
