//! In-process tests of `dispatch`: each arm must route to its command and surface
//! that command's verdict, not swallow it (#5675). Contract: bin-pv--pv-v1.

use aprender_contracts_cli::{dispatch, Cli};
use clap::Parser;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn repo_contract() -> String {
    format!(
        "{}/../../contracts/multinomial-head-v1.yaml",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn run(args: &[&str]) -> Result<(), String> {
    let mut argv = vec!["pv"];
    argv.extend_from_slice(args);
    let cli = Cli::try_parse_from(argv).map_err(|e| format!("parse: {e}"))?;
    dispatch(cli.command).map_err(|e| e.to_string())
}

#[test]
fn valid_contract_routes_through_validate_status_and_equations() {
    let c = repo_contract();
    run(&["validate", &c]).expect("validate accepts a well-formed contract");
    run(&["validate", &c, "--check-ids"]).expect("check-ids reports counts");
    run(&["status", &c]).expect("status");
    run(&["equations", &c]).expect("equations default format");
    run(&["equations", &c, "--format", "latex"]).expect("equations latex");
}

#[test]
fn missing_contract_is_an_error_for_every_file_command() {
    let nope = "/nonexistent/contract.yaml";
    for cmd in [
        "validate",
        "status",
        "check-parity",
        "kani",
        "lean-status",
        "coverage",
    ] {
        assert!(
            run(&[cmd, nope]).is_err(),
            "{cmd} must fail on a missing path"
        );
    }
    assert!(run(&["diff", nope, nope]).is_err());
    assert!(run(&["audit", nope]).is_err());
}

#[test]
fn unknown_output_format_is_rejected_by_graph_and_equations() {
    let c = repo_contract();
    let g = run(&["graph", &fixture(""), "--format", "nosuchformat"]);
    assert!(g.is_err(), "graph must reject an unknown format");
    let e = run(&["equations", &c, "--format", "nosuchformat"]);
    assert!(e.is_err(), "equations must reject an unknown format");
}

#[test]
fn parse_errors_never_reach_dispatch() {
    assert!(run(&["no-such-subcommand"])
        .unwrap_err()
        .starts_with("parse:"));
    assert!(run(&["validate"]).unwrap_err().starts_with("parse:"));
}
