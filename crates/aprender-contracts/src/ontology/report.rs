//! CRUX §14.6 S4 — the validator's [`Report`] written as a W3C `sh:ValidationReport`, as Turtle, as N-Triples
//! and as JSON, each result carrying every field of the §14.2 R-FEEDBACK tuple: focus node, result path, value,
//! source shape, source constraint component and severity, plus the message.
//!
//! A node the reader held as a [`SKOLEM`] IRI (a data blank node) is written back as a blank node, so the report
//! names the data's own nodes. A blank-node property shape is written as one: its node shape's `sh:property`
//! points at it and it carries its `sh:path`, which is how §14.2 keys it. A component outside SHACL core (pv's
//! own `resolves`, `extract:json`) gets an IRI under `ont:component/`, never a SHACL IRI it is not.

use std::fmt::Write as _;

use crate::ontology::rdf::{escape_literal, Term, ONT_BASE, RDF_TYPE};
use crate::ontology::shapes::{Report, Severity, SourceShape, ValidationResult};
use crate::ontology::shapes_rdf::{SH, SKOLEM};

/// The SHACL core components the validator implements, by the key it reports them under.
const CORE: &[(&str, &str)] = &[
    ("class", "ClassConstraintComponent"),
    ("closed", "ClosedConstraintComponent"),
    ("datatype", "DatatypeConstraintComponent"),
    ("in", "InConstraintComponent"),
    ("lessThan", "LessThanConstraintComponent"),
    ("lessThanOrEquals", "LessThanOrEqualsConstraintComponent"),
    ("maxCount", "MaxCountConstraintComponent"),
    ("maxLength", "MaxLengthConstraintComponent"),
    ("minCount", "MinCountConstraintComponent"),
    ("minLength", "MinLengthConstraintComponent"),
    ("node", "NodeConstraintComponent"),
    ("nodeKind", "NodeKindConstraintComponent"),
    ("pattern", "PatternConstraintComponent"),
];

/// `sh:<X>ConstraintComponent` for a SHACL core component, `ont:component/<key>` for pv's own.
#[must_use]
pub fn component_iri(component: &str) -> String {
    match CORE.iter().find(|(k, _)| *k == component) {
        Some((_, local)) => format!("{SH}{local}"),
        None => format!("{ONT_BASE}component/{}", component.replace(':', "-")),
    }
}

#[must_use]
pub fn severity_iri(s: Severity) -> String {
    match s {
        Severity::Violation => format!("{SH}Violation"),
        Severity::Warning => format!("{SH}Warning"),
    }
}

/// A node of the report graph.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Iri(String),
    Blank(String),
    Literal(String, String),
}

impl Node {
    /// An IRI, or the data blank node a [`SKOLEM`] IRI holds (label prefixed `d`, so no report node shares it).
    fn iri(i: &str) -> Self {
        match i.strip_prefix(SKOLEM) {
            Some(label) => Self::Blank(format!("d{label}")),
            None => Self::Iri(i.to_string()),
        }
    }

    /// A message as written: each `<SKOLEM…>` it names becomes the blank node [`Node::iri`] writes for it.
    fn message(m: &str) -> String {
        let mut out = String::new();
        let mut rest = m;
        let open = format!("<{SKOLEM}");
        while let Some(at) = rest.find(&open) {
            let label = &rest[at + open.len()..];
            let Some(end) = label.find('>') else { break };
            out.push_str(&rest[..at]);
            out.push_str(&format!("_:d{}", &label[..end]));
            rest = &label[end + 1..];
        }
        out.push_str(rest);
        out
    }

    fn term(t: &Term) -> Self {
        match t {
            Term::Iri(i) => Self::iri(i),
            Term::Literal { value, datatype } => Self::Literal(value.clone(), datatype.clone()),
        }
    }

    fn nt(&self) -> String {
        match self {
            Self::Iri(i) => format!("<{i}>"),
            Self::Blank(b) => format!("_:{b}"),
            Self::Literal(v, d) => format!("\"{}\"^^<{d}>", escape_literal(v)),
        }
    }

    /// As Turtle: an `sh:` IRI with its prefix, everything else as N-Triples writes it.
    fn ttl(&self) -> String {
        match self {
            Self::Iri(i) => match i.strip_prefix(SH) {
                Some(local)
                    if !local.is_empty() && local.chars().all(|c| c.is_ascii_alphanumeric()) =>
                {
                    format!("sh:{local}")
                }
                _ => self.nt(),
            },
            _ => self.nt(),
        }
    }
}

/// The report as triples, in a fixed order: the report node, then each result in the validator's order.
fn triples(report: &Report) -> Vec<(Node, String, Node)> {
    let sh = |l: &str| format!("{SH}{l}");
    let xsd_bool = "http://www.w3.org/2001/XMLSchema#boolean";
    let xsd_string = "http://www.w3.org/2001/XMLSchema#string";
    let root = Node::Blank("report".into());
    let mut out = vec![
        (
            root.clone(),
            RDF_TYPE.to_string(),
            Node::Iri(sh("ValidationReport")),
        ),
        (
            root.clone(),
            sh("conforms"),
            Node::Literal(report.conforms().to_string(), xsd_bool.into()),
        ),
    ];
    let mut blank_shapes: Vec<(String, String)> = Vec::new();
    for (n, r) in report.results.iter().enumerate() {
        let res = Node::Blank(format!("r{n}"));
        out.push((root.clone(), sh("result"), res.clone()));
        out.push((
            res.clone(),
            RDF_TYPE.to_string(),
            Node::Iri(sh("ValidationResult")),
        ));
        out.push((res.clone(), sh("focusNode"), Node::iri(&r.focus)));
        if let Some(p) = &r.path {
            out.push((res.clone(), sh("resultPath"), Node::Iri(p.clone())));
        }
        if let Some(v) = &r.value {
            out.push((res.clone(), sh("value"), Node::term(v)));
        }
        let source = match &r.source {
            SourceShape::Node(i) | SourceShape::Property(i) => Node::Iri(i.clone()),
            SourceShape::Blank { node, path } => {
                let key = (node.clone(), path.clone());
                let at = blank_shapes
                    .iter()
                    .position(|k| *k == key)
                    .unwrap_or_else(|| {
                        blank_shapes.push(key);
                        blank_shapes.len() - 1
                    });
                Node::Blank(format!("ps{at}"))
            }
        };
        out.push((res.clone(), sh("sourceShape"), source));
        out.push((
            res.clone(),
            sh("sourceConstraintComponent"),
            Node::Iri(component_iri(r.component)),
        ));
        out.push((
            res.clone(),
            sh("resultSeverity"),
            Node::Iri(severity_iri(r.severity)),
        ));
        out.push((
            res,
            sh("resultMessage"),
            Node::Literal(Node::message(&r.message), xsd_string.into()),
        ));
    }
    for (at, (node, path)) in blank_shapes.iter().enumerate() {
        let ps = Node::Blank(format!("ps{at}"));
        out.push((Node::Iri(node.clone()), sh("property"), ps.clone()));
        out.push((ps, sh("path"), Node::Iri(path.clone())));
    }
    out
}

/// The report as N-Triples, one triple per line.
#[must_use]
pub fn to_ntriples(report: &Report) -> String {
    let mut s = String::new();
    for (a, p, b) in triples(report) {
        let _ = writeln!(s, "{} <{p}> {} .", a.nt(), b.nt());
    }
    s
}

/// The report as Turtle: the same triples, `sh:` terms written with their prefix.
#[must_use]
pub fn to_turtle(report: &Report) -> String {
    let mut s = format!("@prefix sh: <{SH}> .\n\n");
    for (a, p, b) in triples(report) {
        let pred = if p == RDF_TYPE {
            "a".to_string()
        } else {
            Node::Iri(p).ttl()
        };
        let _ = writeln!(s, "{} {pred} {} .", a.ttl(), b.ttl());
    }
    s
}

fn json_node(n: &Node) -> serde_json::Value {
    match n {
        Node::Iri(i) => serde_json::json!({ "iri": i }),
        Node::Blank(b) => serde_json::json!({ "blank": b }),
        Node::Literal(v, d) => serde_json::json!({ "value": v, "datatype": d }),
    }
}

fn json_result(r: &ValidationResult) -> serde_json::Value {
    serde_json::json!({
        "focusNode": json_node(&Node::iri(&r.focus)),
        "resultPath": r.path.as_ref().map(|p| json_node(&Node::Iri(p.clone()))),
        "value": r.value.as_ref().map(|v| json_node(&Node::term(v))),
        "sourceShape": r.source.key(),
        "sourceConstraintComponent": component_iri(r.component),
        "resultSeverity": severity_iri(r.severity),
        "resultMessage": Node::message(&r.message),
    })
}

/// The report as JSON: `conforms` and one object per result, every §14.2 field named as SHACL names it.
/// `sourceShape` is the §14.2 key (`<iri>`, or `<node> / <path>` for a blank-node property shape).
#[must_use]
pub fn to_json(report: &Report) -> serde_json::Value {
    serde_json::json!({
        "conforms": report.conforms(),
        "results": report.results.iter().map(json_result).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
