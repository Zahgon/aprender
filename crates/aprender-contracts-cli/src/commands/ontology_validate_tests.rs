//! `pv ontology validate` end to end on the W3C core cases the repo vendors: Turtle in, report out. The writers
//! and the shapes subset have their own lib tests; this proves the command joins them.

use std::path::{Path, PathBuf};

use provable_contracts::ontology::report::{to_json, to_ntriples};
use provable_contracts::ontology::shapes::SourceShape;

use super::validate_files;

fn w3c(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/oracle/w3c")
        .join(rel)
}

fn case(rel: &str) -> Result<provable_contracts::ontology::shapes::Report, super::Refusal> {
    let p = w3c(rel);
    validate_files(&p, &p, None)
}

const EX: &str = "http://datashapes.org/sh/tests/core/property/minCount-001.test#";

#[test]
fn min_count_001_reports_the_w3c_result() {
    let r = case("property/minCount-001.ttl").expect("in the subset");
    assert!(!r.conforms());
    assert_eq!(r.results.len(), 1, "{:?}", r.results);
    let x = &r.results[0];
    assert_eq!(x.focus, format!("{EX}InvalidPerson"));
    assert_eq!(x.path.as_deref(), Some(format!("{EX}firstName").as_str()));
    assert_eq!(
        x.source,
        SourceShape::Property(format!("{EX}PersonShape-firstName"))
    );
    let nt = to_ntriples(&r);
    assert!(
        nt.contains("<http://www.w3.org/ns/shacl#MinCountConstraintComponent>"),
        "{nt}"
    );
    assert_eq!(to_json(&r)["conforms"], false);
}

/// Every vendored case either yields a report or is refused with exit 3 naming the file and the term; none panics, none is
/// refused as unreadable (2), and no fewer run than the 19 of 35 measured on 2026-10-14 (the refusals are
/// `sh:class`, `sh:datatype`, `sh:in`, `sh:maxLength`, `sh:minLength`, `sh:nodeKind`, `sh:pattern` on a node shape,
/// `sh:or`, `sh:nodeKind sh:BlankNode` and a language-tagged literal).
#[test]
fn every_vendored_case_reports_or_is_refused_by_name() {
    let (mut reported, mut refused) = (0, Vec::new());
    for dir in ["node", "property", "targets"] {
        let mut files: Vec<_> = std::fs::read_dir(w3c(dir))
            .expect("the W3C directory lists")
            .map(|e| e.expect("an entry").path())
            .filter(|p| p.extension().is_some_and(|e| e == "ttl"))
            .collect();
        files.sort();
        for p in files {
            match validate_files(&p, &p, None) {
                Ok(_) => reported += 1,
                Err(e) => {
                    assert_eq!(e.code, 3, "{}: {}", p.display(), e.message);
                    assert!(
                        e.message.starts_with(&p.display().to_string()),
                        "{}",
                        e.message
                    );
                    refused.push(e.message);
                }
            }
        }
    }
    assert!(
        reported >= 19,
        "only {reported} W3C cases ran; refused:\n{}",
        refused.join("\n")
    );
}

#[test]
fn an_unknown_extension_is_exit_2() {
    let p = w3c("property/minCount-001.ttl");
    let e = validate_files(&p, Path::new("data.xml"), None).expect_err("refused");
    assert_eq!(e.code, 2, "{}", e.message);
}
