use super::*;

fn target(name: &str, commands: &[&str], routes: &[&str], tools: &[&str]) -> Target {
    let set = |xs: &[&str]| xs.iter().map(ToString::to_string).collect();
    Target {
        name: name.to_string(),
        commands: set(commands),
        routes: set(routes),
        tools: set(tools),
        ..Target::default()
    }
}

fn surface(binary: &str, kind: SurfaceKind, s: &str) -> Surface {
    Surface {
        binary: binary.to_string(),
        kind,
        surface: s.to_string(),
    }
}

const REGISTRY_OK: &str = r#"
consumers:
  - binary: apr
    kind: command
    surface: profile
    consumer: {kind: gate, ref: ".github/workflows/qwen-story-daily.yml#story"}
    status: report-only
    platforms: [linux]
  - binary: apr
    kind: route
    surface: GET /metrics
    consumer: {kind: gate, ref: ".github/workflows/x.yml#job"}
    status: active
    first_green: {run_url: "https://github.com/paiml/aprender/actions/runs/1", sha: "abc"}
  - binary: apr
    kind: tool
    surface: apr.trace
    status: deprecated
    remove_by: "0.75.0"
"#;

fn fixture() -> BTreeSet<Surface> {
    // Two `apr` targets (the facade pair) and noise that must not be derived.
    let apr = target(
        "apr",
        &["profile", "contract verify", "extract", "run"],
        &["GET /metrics", "POST /v1/chat/completions"],
        &["apr.trace", "apr.run"],
    );
    derive(&[apr.clone(), apr])
}

#[test]
fn derive_keeps_whole_tokens_and_collapses_the_apr_pair() {
    let want: BTreeSet<Surface> = [
        surface("apr", SurfaceKind::Command, "profile"),
        surface("apr", SurfaceKind::Route, "GET /metrics"),
        surface("apr", SurfaceKind::Tool, "apr.trace"),
    ]
    .into_iter()
    .collect();
    assert_eq!(fixture(), want);
    // A binary named for tracing makes every one of its commands a surface.
    let renacer = derive(&[target("aprender-profile", &["validate"], &[], &[])]);
    assert_eq!(renacer.len(), 1);
    // Token split covers `-`: a lint command named for a trace kind is a surface.
    let lint = derive(&[target(
        "apr",
        &["hang-trace-lint", "contracts-lint"],
        &[],
        &[],
    )]);
    assert_eq!(lint.len(), 1);
}

#[test]
fn an_honest_registry_holds() {
    let reg = parse_registry(REGISTRY_OK).expect("parse");
    assert_eq!(check(&fixture(), &reg, "0.72.0"), vec![]);
}

/// FALSIFY-TCR-001: a derived surface nobody registered is RED, and so is a
/// registry entry for a surface the binaries dropped.
#[test]
fn falsify_tcr_001_unregistered_surface_is_red() {
    let mut reg = parse_registry(REGISTRY_OK).expect("parse");
    let dropped = reg.remove(0);
    assert_eq!(
        check(&fixture(), &reg, "0.72.0"),
        vec![Finding::Unregistered(dropped.key())]
    );

    let mut derived = fixture();
    derived.remove(&surface("apr", SurfaceKind::Route, "GET /metrics"));
    let reg = parse_registry(REGISTRY_OK).expect("parse");
    assert_eq!(
        check(&derived, &reg, "0.72.0"),
        vec![Finding::NotDerived(surface(
            "apr",
            SurfaceKind::Route,
            "GET /metrics"
        ))]
    );
}

/// FALSIFY-TCR-002: `active` without `first_green`, and an entry with no consumer
/// that is not deprecated, are RED.
#[test]
fn falsify_tcr_002_active_without_first_green_is_red() {
    let mut reg = parse_registry(REGISTRY_OK).expect("parse");
    reg[1].first_green = None;
    assert_eq!(
        check(&fixture(), &reg, "0.72.0"),
        vec![Finding::ActiveWithoutFirstGreen(reg[1].key())]
    );

    let mut reg = parse_registry(REGISTRY_OK).expect("parse");
    reg[0].consumer = None;
    assert_eq!(
        check(&fixture(), &reg, "0.72.0"),
        vec![Finding::NoConsumer(reg[0].key())]
    );
}

/// FALSIFY-TCR-003: a deprecated surface still exposed at or past its
/// `remove_by` version is RED; one with no usable `remove_by` is RED too.
#[test]
fn falsify_tcr_003_deprecated_past_remove_by_is_red() {
    let reg = parse_registry(REGISTRY_OK).expect("parse");
    assert_eq!(check(&fixture(), &reg, "0.74.9"), vec![]);
    for at in ["0.75.0", "0.76.1", "1.0.0"] {
        assert_eq!(
            check(&fixture(), &reg, at),
            vec![Finding::DeprecatedPastRemoveBy {
                surface: reg[2].key(),
                remove_by: "0.75.0".to_string(),
                version: at.to_string(),
            }]
        );
    }

    for bad in [None, Some("soon"), Some("0.75")] {
        let mut reg = parse_registry(REGISTRY_OK).expect("parse");
        reg[2].remove_by = bad.map(ToString::to_string);
        assert_eq!(
            check(&fixture(), &reg, "0.72.0"),
            vec![Finding::BadRemoveBy(reg[2].key())]
        );
    }
}

#[test]
fn vacuity_duplicates_and_unknown_keys_are_red() {
    let reg = parse_registry(REGISTRY_OK).expect("parse");
    assert_eq!(
        check(&BTreeSet::new(), &reg, "0.72.0"),
        vec![Finding::NothingDerived]
    );

    let mut twice = reg.clone();
    twice.push(reg[0].clone());
    assert_eq!(
        check(&fixture(), &twice, "0.72.0"),
        vec![Finding::Duplicate(reg[0].key())]
    );

    let typo = REGISTRY_OK.replace("remove_by:", "removeby:");
    assert!(parse_registry(&typo).is_err());
    let bad_kind = REGISTRY_OK.replace("{kind: gate,", "{kind: dashboard,");
    assert!(parse_registry(&bad_kind).is_err());
}
