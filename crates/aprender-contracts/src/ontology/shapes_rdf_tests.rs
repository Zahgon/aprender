use super::*;
use crate::ontology::shapes::{validate, Report, SourceShape};

/// A graph written one triple per line, `S P O`, terms separated by blanks: `ex:a` (also `sh:`, `rdf:`, `rdfs:`,
/// `xsd:`), `<iri>`, `_:b`, `a` (rdf:type), `"lex"`, `"lex"^^xsd:integer`, `"lex"@en`, or a bare integer. These
/// tests never go through the Turtle reader: this module keeps apart from it (R-15). The end-to-end path,
/// Turtle in and a report out, is `pv ontology validate`'s CLI test.
pub(crate) fn triples(text: &str) -> Vec<RdfTriple> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| {
            let t: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(t.len(), 3, "not S P O: {l}");
            let RdfTerm::Iri(p) = term(t[1]) else {
                panic!("predicate {}", t[1])
            };
            RdfTriple {
                subject: term(t[0]),
                predicate: p,
                object: term(t[2]),
            }
        })
        .collect()
}

pub(crate) fn ex(l: &str) -> String {
    format!("http://example.org/{l}")
}

fn expand(t: &str) -> String {
    for (p, ns) in [
        ("ex:", "http://example.org/"),
        ("sh:", SH),
        ("rdf:", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
        ("rdfs:", "http://www.w3.org/2000/01/rdf-schema#"),
        ("xsd:", "http://www.w3.org/2001/XMLSchema#"),
    ] {
        if let Some(l) = t.strip_prefix(p) {
            return format!("{ns}{l}");
        }
    }
    panic!("no prefix: {t}")
}

fn term(t: &str) -> RdfTerm {
    let xsd = |l: &str| format!("http://www.w3.org/2001/XMLSchema#{l}");
    if t == "a" {
        return RdfTerm::Iri(RDF_TYPE.into());
    }
    if let Some(b) = t.strip_prefix("_:") {
        return RdfTerm::Blank(b.into());
    }
    if let Some(i) = t.strip_prefix('<').and_then(|t| t.strip_suffix('>')) {
        return RdfTerm::Iri(i.into());
    }
    if let Some(rest) = t.strip_prefix('"') {
        let (lex, tail) = rest.split_once('"').expect("a closing quote");
        let (datatype, lang) = match tail {
            "" => (xsd("string"), None),
            _ if tail.starts_with("^^") => (expand(&tail[2..]), None),
            _ if tail.starts_with('@') => (
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".into(),
                Some(tail[1..].into()),
            ),
            _ => panic!("literal tail {tail}"),
        };
        return RdfTerm::Literal {
            lexical: lex.into(),
            datatype,
            lang,
        };
    }
    if t.bytes().all(|b| b.is_ascii_digit()) {
        return RdfTerm::Literal {
            lexical: t.into(),
            datatype: xsd("integer"),
            lang: None,
        };
    }
    RdfTerm::Iri(expand(t))
}

/// The RDF list `items` at `_:<head>`, as triples.
pub(crate) fn list(head: &str, items: &[&str]) -> String {
    let mut s = String::new();
    for (n, item) in items.iter().enumerate() {
        let here = if n == 0 {
            format!("_:{head}")
        } else {
            format!("_:{head}{n}")
        };
        let next = if n + 1 == items.len() {
            "rdf:nil".to_string()
        } else {
            format!("_:{head}{}", n + 1)
        };
        s.push_str(&format!(
            "{here} rdf:first {item}\n{here} rdf:rest {next}\n"
        ));
    }
    s
}

pub(crate) fn run(text: &str) -> Report {
    let g = triples(text);
    let shapes = shapes(&g).expect("shapes in the subset");
    validate(&data_graph(&g).expect("data graph"), &shapes)
}

/// W3C core property/minCount-001, shapes and data in one graph as the suite writes them.
const MIN_COUNT_001: &str = "
ex:InvalidPerson a ex:Person
ex:Person a rdfs:Class
ex:PersonShape a sh:NodeShape
ex:PersonShape sh:property ex:PersonShape-firstName
ex:PersonShape sh:targetClass ex:Person
ex:PersonShape sh:targetNode ex:ValidResource
ex:PersonShape-firstName sh:path ex:firstName
ex:PersonShape-firstName sh:datatype xsd:string
ex:PersonShape-firstName sh:minCount 1
ex:ValidResource a ex:Person
ex:ValidResource ex:firstName \"John\"
";

#[test]
fn named_property_shape_and_target_node_are_read() {
    let s = shapes(&triples(MIN_COUNT_001)).expect("in the subset");
    assert_eq!(s.len(), 1);
    assert_eq!(s[0].iri.as_deref(), Some(ex("PersonShape").as_str()));
    assert_eq!(s[0].target_class, ex("Person"));
    assert_eq!(s[0].target_nodes, vec![ex("ValidResource")]);
    let p = &s[0].properties[0];
    assert_eq!(p.iri.as_deref(), Some(ex("PersonShape-firstName").as_str()));
    assert_eq!(
        (p.min_count, p.path.as_str()),
        (Some(1), ex("firstName").as_str())
    );
}

#[test]
fn min_count_001_reports_the_w3c_tuple() {
    let r = run(MIN_COUNT_001);
    assert_eq!(r.results.len(), 1, "{:?}", r.results);
    let x = &r.results[0];
    assert_eq!(x.focus, ex("InvalidPerson"));
    assert_eq!(x.path.as_deref(), Some(ex("firstName").as_str()));
    assert_eq!(x.source, SourceShape::Property(ex("PersonShape-firstName")));
    assert_eq!(x.component, "minCount");
    assert_eq!(x.value, None, "minCount has no sh:value");
}

const BLANK_PROPERTY: &str = "
ex:S sh:targetNode ex:a
ex:S sh:property _:p
_:p sh:path ex:p
_:p sh:minCount 1
";

#[test]
fn a_target_node_outside_the_class_is_a_focus_node_and_a_blank_shape_is_keyed_by_path() {
    let r = run(BLANK_PROPERTY);
    assert_eq!(r.results.len(), 1);
    assert_eq!(r.results[0].focus, ex("a"));
    assert_eq!(
        r.results[0].source,
        SourceShape::Blank {
            node: ex("S"),
            path: ex("p")
        }
    );
}

#[test]
fn a_declared_message_heads_every_result_message() {
    let r = run(&format!("{BLANK_PROPERTY}_:p sh:message \"needs-p\"\n"));
    assert!(
        r.results[0].message.starts_with("needs-p"),
        "{}",
        r.results[0].message
    );
}

#[test]
fn sh_in_reads_an_rdf_list_of_iris_and_literals() {
    let text = format!(
        "ex:S sh:targetNode ex:a\nex:S sh:targetNode ex:b\nex:S sh:targetNode ex:c\nex:S sh:property _:p\n\
         _:p sh:path ex:p\n_:p sh:in _:l\n{}\
         ex:a ex:p ex:ok\nex:b ex:p \"ok\"\nex:c ex:p ex:bad\nex:c ex:p \"other\"\n",
        list("l", &["ex:ok", "\"ok\""])
    );
    let r = run(&text);
    assert_eq!(
        r.results.len(),
        2,
        "only ex:c's two values fail: {:?}",
        r.results
    );
    assert!(r.results.iter().all(|x| x.focus == ex("c")));
}

#[test]
fn a_data_blank_node_is_held_as_a_skolem_iri() {
    let d = data_graph(&triples("ex:a ex:p _:x\n_:x ex:q 1")).expect("no language tag");
    let held: Vec<_> = d.iter().filter(|t| t.subject.starts_with(SKOLEM)).collect();
    assert_eq!(held.len(), 1, "the blank node is a subject once");
}

#[test]
fn a_language_tagged_literal_is_refused_by_name() {
    let e = data_graph(&triples("ex:a ex:p \"hi\"@en")).expect_err("refused");
    assert!(e.contains("language-tagged") && e.contains("@en"), "{e}");
}

/// Planted falsifiers: each shapes graph uses one term outside the subset. Ignoring any of them would turn a
/// constraint the validator never checked into a "conforms"; each must be refused, naming the term.
#[test]
fn every_term_outside_the_subset_is_refused_by_name() {
    let p = "ex:S sh:targetNode ex:a\nex:S sh:property _:p\n_:p sh:path ex:p\n";
    let cases: Vec<(String, &str)> = vec![
        ("ex:S sh:targetNode ex:a\nex:S sh:datatype xsd:string".into(), "sh:datatype on a node shape"),
        (format!("{p}_:p sh:pattern \"a\"\n_:p sh:flags \"i\""), "sh:flags"),
        ("ex:S sh:targetNode ex:a\nex:S sh:property _:p\n_:p sh:path _:x\n_:x sh:inversePath ex:p".into(), "path expression"),
        ("ex:S sh:targetSubjectsOf ex:p".into(), "sh:targetSubjectsOf"),
        ("ex:S sh:targetObjectsOf ex:p".into(), "sh:targetObjectsOf"),
        (format!("{p}_:p sh:nodeKind sh:BlankNode"), "sh:nodeKind"),
        (format!("{p}_:p sh:severity sh:Info"), "sh:severity"),
        (format!("{p}_:p sh:minCount 1\n_:p sh:minCount 2"), "2 values of sh:minCount"),
        ("_:s sh:targetNode ex:a".into(), "blank-node node shape"),
        (format!("{p}_:p sh:hasValue ex:v"), "sh:hasValue"),
        ("ex:S sh:targetNode ex:a\nex:S sh:or _:l".into(), "sh:or"),
        ("ex:S a sh:NodeShape\nex:S a rdfs:Class\nex:S sh:property _:p\n_:p sh:path ex:p".into(), "implicit class target"),
        (format!("{p}_:p sh:node ex:T\nex:T sh:property _:q\n_:q sh:path ex:q\n_:q sh:node ex:U"), "nested more than one level"),
        (format!("{p}_:p sh:node ex:T\nex:T sh:targetNode ex:b"), "a target on a shape that is also the object of sh:node"),
    ];
    for (text, named) in &cases {
        let e = shapes(&triples(text)).expect_err(text).to_string();
        assert!(
            e.contains(named),
            "{text}\n  refused as {e:?}, which does not name {named:?}"
        );
    }
}

#[test]
fn non_validating_terms_are_skipped() {
    let g = triples(&format!(
        "{BLANK_PROPERTY}ex:S sh:name \"n\"\n_:p sh:description \"d\"\n_:p sh:order 1\n"
    ));
    assert_eq!(shapes(&g).expect("skipped").len(), 1);
}
