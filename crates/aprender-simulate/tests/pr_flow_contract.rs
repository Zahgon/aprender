//! SIM-3 (FLOW-005, #5511): the `pr-flow` scenario is a contract, run through the shipped binary.
//!
//! Each design must land inside its reference band, Little's law must close, and the same
//! seeds must give the same bytes. A scenario that returned a constant would fail the band for
//! three of the four designs; a scenario that read a clock or an unseeded RNG would fail the
//! byte comparison.

use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_aprender-simulate");

/// Design, reference merged/h (the +-5% band is the contract).
const WANT: [(&str, f64); 4] = [("D", 20.08), ("A", 4.96), ("G", 12.55), ("E", 21.37)];

fn block(design: &str) -> String {
    let out = Command::new(BIN)
        .args(["pr-flow", "--design", design, "--json", "--sha", "test"])
        .output()
        .expect("spawn the binary under test");
    assert!(
        out.status.success(),
        "pr-flow {design} exited {:?}",
        out.status
    );
    String::from_utf8(out.stdout).expect("utf-8 json")
}

fn num(json: &serde_json::Value, path: &[&str]) -> f64 {
    path.iter()
        .fold(json, |v, k| &v[*k])
        .as_f64()
        .unwrap_or(f64::NAN)
}

#[test]
fn every_design_is_inside_its_reference_band_and_littles_law_closes() {
    for (design, want) in WANT {
        let v: serde_json::Value = serde_json::from_str(&block(design)).expect("json");
        let got = num(&v, &["predicted", "merged_h"]);
        assert!(
            (got - want).abs() <= want * 0.05,
            "{design}: {got:.2} vs {want}"
        );
        let (w_min, l) = (
            num(&v, &["predicted", "W_min_mean"]),
            num(&v, &["predicted", "L"]),
        );
        let lam_w = got * w_min / 60.0;
        assert!(
            (lam_w - l).abs() <= l * 0.005,
            "{design}: lambda*W {lam_w:.3} vs L {l:.3}"
        );
        assert_eq!(v["sim3_contract"], "pass", "{design}");
    }
}

#[test]
fn same_seeds_give_the_same_bytes() {
    for (design, _) in WANT {
        assert_eq!(block(design), block(design), "{design}");
    }
}
