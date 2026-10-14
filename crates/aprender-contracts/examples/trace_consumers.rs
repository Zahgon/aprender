//! TRACE-001 TR-08 (aprender#5314): hold `contracts/trace-consumers-v1.yaml` to the
//! trace/profile/metrics surfaces of the tracked built-binary snapshot.
//!
//! A LAB check (C324): it runs at night, never on a PR, merge-queue or release
//! path. It prints the measured coverage and every finding, and exits 1 on any.
//!
//!   cargo run -p aprender-contracts --example trace_consumers [-- <repo root>]

use provable_contracts::ontology::extract::binary::{parse_snapshot, SNAPSHOT};
use provable_contracts::trace_consumers::{check, derive, parse_registry, Finding, REGISTRY};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let root = std::env::args().nth(1).map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        PathBuf::from,
    );
    let read = |rel: &str| {
        std::fs::read_to_string(root.join(rel)).map_err(|e| format!("cannot read {rel}: {e}"))
    };
    let (snapshot, registry) = match (read(SNAPSHOT), read(REGISTRY)) {
        (Ok(s), Ok(r)) => (s, r),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("NOT MEASURED: {e}");
            return ExitCode::from(2);
        }
    };
    let (targets, refused) = parse_snapshot(&snapshot);
    if !refused.is_empty() {
        eprintln!("NOT MEASURED: {SNAPSHOT} refused {} line(s)", refused.len());
        return ExitCode::from(2);
    }
    let entries = match parse_registry(&registry) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("RED: {e}");
            return ExitCode::FAILURE;
        }
    };
    let derived = derive(&targets);
    let findings = check(&derived, &entries, env!("CARGO_PKG_VERSION"));
    let unregistered = findings
        .iter()
        .filter(|f| matches!(f, Finding::Unregistered(_)))
        .count();
    let consumed = entries.iter().filter(|e| e.consumer.is_some()).count();
    println!(
        "trace-consumers-v1: {} derived surfaces from {} bin targets; registered {}/{}; with a consumer {}/{}",
        derived.len(),
        targets.len(),
        derived.len() - unregistered,
        derived.len(),
        consumed,
        derived.len(),
    );
    for f in &findings {
        println!("  RED  {f}");
    }
    if findings.is_empty() {
        println!("GREEN");
        ExitCode::SUCCESS
    } else {
        println!("RED: {} finding(s)", findings.len());
        ExitCode::FAILURE
    }
}
