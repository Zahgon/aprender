use super::*;
use crate::ontology::shapes_rdf::tests::run as report;

const TWO_RESULTS: &str = "
ex:S sh:targetNode ex:a
ex:S sh:targetNode ex:b
ex:S sh:property _:p
_:p sh:path ex:p
_:p sh:minCount 1
ex:S sh:property ex:S-q
ex:S-q sh:path ex:q
ex:S-q sh:datatype xsd:integer
ex:S-q sh:severity sh:Warning
ex:b ex:p 1
ex:b ex:q _:v
_:v ex:r 1
";

#[test]
fn every_core_component_maps_to_its_shacl_iri_and_pvs_own_do_not() {
    for (k, local) in CORE {
        assert_eq!(component_iri(k), format!("{SH}{local}"));
    }
    for own in ["resolves", "extract:json"] {
        let i = component_iri(own);
        assert!(
            i.starts_with(ONT_BASE) && !i.starts_with(SH),
            "{own} -> {i}"
        );
    }
}

#[test]
fn the_ntriples_report_carries_every_r_feedback_field() {
    let nt = to_ntriples(&report(TWO_RESULTS));
    let sh = |l: &str| format!("<{SH}{l}>");
    for needed in [
        format!(
            "{} {}",
            sh("conforms"),
            "\"false\"^^<http://www.w3.org/2001/XMLSchema#boolean>"
        ),
        format!("{} <http://example.org/a>", sh("focusNode")),
        format!("{} <http://example.org/p>", sh("resultPath")),
        format!(
            "{} {}",
            sh("sourceConstraintComponent"),
            sh("MinCountConstraintComponent")
        ),
        format!(
            "{} {}",
            sh("sourceConstraintComponent"),
            sh("DatatypeConstraintComponent")
        ),
        format!("{} <http://example.org/S-q>", sh("sourceShape")),
        format!("{} {}", sh("resultSeverity"), sh("Warning")),
        format!("{} {}", sh("resultSeverity"), sh("Violation")),
        format!("<http://example.org/S> {} _:ps0", sh("property")),
        format!("_:ps0 {} <http://example.org/p>", sh("path")),
    ] {
        assert!(nt.contains(&needed), "missing {needed}\n{nt}");
    }
    assert_eq!(nt.matches(&sh("resultMessage")).count(), 2, "{nt}");
}

#[test]
fn a_data_blank_node_value_is_written_back_as_a_blank_node() {
    let nt = to_ntriples(&report(TWO_RESULTS));
    let value = nt
        .lines()
        .find(|l| l.contains(&format!("<{SH}value>")))
        .expect("the datatype result has a value");
    assert!(value.contains("> _:d"), "{value}");
    assert!(
        !nt.contains(SKOLEM),
        "no skolem IRI leaks into the report\n{nt}"
    );
    assert!(
        nt.contains("\"_:dv is not"),
        "the message names the value as the report does\n{nt}"
    );
    assert!(
        !to_json(&report(TWO_RESULTS)).to_string().contains(SKOLEM),
        "nor into the JSON"
    );
}

#[test]
fn a_conforming_graph_reports_conforms_true_and_no_result() {
    let r = report(
        "ex:S sh:targetNode ex:a\nex:S sh:property _:p\n_:p sh:path ex:p\n_:p sh:maxCount 1",
    );
    let nt = to_ntriples(&r);
    assert!(nt.contains("\"true\"^^"), "{nt}");
    assert!(!nt.contains(&format!("<{SH}result>")), "{nt}");
    assert_eq!(to_json(&r)["results"].as_array().map(Vec::len), Some(0));
}

#[test]
fn the_turtle_report_is_the_ntriples_report_with_sh_prefixed() {
    let r = report(TWO_RESULTS);
    let (ttl, nt) = (to_turtle(&r), to_ntriples(&r));
    assert!(ttl.starts_with(&format!("@prefix sh: <{SH}> .")), "{ttl}");
    assert_eq!(
        ttl.lines().filter(|l| l.ends_with(" .")).count() - 1,
        nt.lines().count()
    );
    assert!(ttl.contains("_:report a sh:ValidationReport ."), "{ttl}");
}

#[test]
fn the_json_report_names_each_field_as_shacl_does() {
    let j = to_json(&report(TWO_RESULTS));
    assert_eq!(j["conforms"], false);
    let rs = j["results"].as_array().expect("results");
    assert_eq!(rs.len(), 2);
    for r in rs {
        for k in [
            "focusNode",
            "resultPath",
            "value",
            "sourceShape",
            "sourceConstraintComponent",
            "resultSeverity",
            "resultMessage",
        ] {
            assert!(r.get(k).is_some(), "{k} missing in {r}");
        }
    }
    assert!(rs
        .iter()
        .any(|r| r["sourceShape"] == "<http://example.org/S> / <http://example.org/p>"));
}
