//! `audit::run` with `--binding` fails when a contract equation has no binding
//! entry, and passes when every equation is bound and implemented (#5628).

use std::path::{Path, PathBuf};

use aprender_contracts_cli::commands::audit;

const EQUATIONS: [&str; 5] = [
    "softmax_nll_objective",
    "analytic_gradient",
    "convergence_error_mapping",
    "label_order_semantics",
    "logit_finiteness",
];

fn contract() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/multinomial-head-v1.yaml")
}

fn write_binding(dir: &Path, equations: &[&str], status: &str) -> PathBuf {
    let mut y = String::from("version: 1.0.0\ntarget_crate: aprender\nbindings:\n");
    for eq in equations {
        y.push_str(&format!(
            "- contract: multinomial-head-v1.yaml\n  equation: {eq}\n  module_path: aprender::x\n  \
             function: f\n  signature: 'fn f()'\n  status: {status}\n"
        ));
    }
    let p = dir.join("binding.yaml");
    std::fs::write(&p, y).expect("write binding");
    p
}

#[test]
fn no_binding_path_skips_the_binding_audit() {
    audit::run(&contract(), None).expect("contract-only audit passes");
}

#[test]
fn fully_bound_and_implemented_passes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let b = write_binding(dir.path(), &EQUATIONS, "implemented");
    audit::run(&contract(), Some(&b)).expect("every equation bound");
}

#[test]
fn unbound_equations_are_counted_as_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let b = write_binding(dir.path(), &EQUATIONS[..1], "implemented");
    let err = audit::run(&contract(), Some(&b)).expect_err("4 equations unbound");
    assert!(err.to_string().contains("4 error(s)"), "{err}");
}

#[test]
fn unreadable_binding_file_is_an_error_not_a_pass() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert!(audit::run(&contract(), Some(&dir.path().join("absent.yaml"))).is_err());
}
