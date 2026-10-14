//! CRUX §14.6 S4 — a SHACL shapes graph read from RDF (as `pv ontology validate` maps S3's reader output) into [`NodeShape`]s, and a data graph into the
//! validator's [`Graph`], so `pv ontology validate` runs the in-house validator on any Turtle or N-Triples input,
//! the W3C core cases included.
//!
//! The subset is the contract dialect's (`shapes.rs`), plus `sh:targetNode`. Every other SHACL term on a shape
//! is REFUSED by name, never ignored: an ignored constraint is a shape that reports "conforms" for something it
//! never checked. The non-validating properties (`sh:name`, `sh:description`, `sh:order`, `sh:group`,
//! `sh:defaultValue`) change no result and are skipped.
//!
//! The validator's graph has no blank node (R-15), so a data blank node is held as the IRI [`SKOLEM`]`<label>`,
//! and the report writer (`report.rs`) writes it back as a blank node. A language-tagged literal has no term in
//! that graph either, and is refused.

use std::collections::{BTreeMap, BTreeSet};

use crate::ontology::rdf::{Graph, Term, RDF_TYPE};
use crate::ontology::shapes::{InEntry, NodeKind, NodeShape, PropertyShape, Severity, ShapeError};

pub const SH: &str = "http://www.w3.org/ns/shacl#";
const RDFS_CLASS: &str = "http://www.w3.org/2000/01/rdf-schema#Class";

const RDF_FIRST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
const RDF_REST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
const RDF_NIL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";

/// A term of an input graph. This module never names the Turtle reader's types (R-15 keeps the reader apart
/// from pv's graph): `pv ontology validate` maps the reader's graph into these.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RdfTerm {
    Iri(String),
    Blank(String),
    Literal {
        lexical: String,
        datatype: String,
        lang: Option<String>,
    },
}

/// A triple of an input graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdfTriple {
    pub subject: RdfTerm,
    pub predicate: String,
    pub object: RdfTerm,
}

/// The IRI prefix a data blank node is held under; the reader's labels (`b0`, `b1`, …) follow it.
pub const SKOLEM: &str = "urn:pv:blank:";

/// `sh:` terms that never change a result (SHACL §2.3); a shape may carry them.
const NON_VALIDATING: &[&str] = &["name", "description", "order", "group", "defaultValue"];

/// The keys of a node shape in the subset.
const NODE_KEYS: &[&str] = &[
    "targetClass",
    "targetNode",
    "property",
    "closed",
    "ignoredProperties",
];

/// The keys of a property shape in the subset.
const PROPERTY_KEYS: &[&str] = &[
    "path",
    "minCount",
    "maxCount",
    "datatype",
    "class",
    "nodeKind",
    "in",
    "pattern",
    "minLength",
    "maxLength",
    "node",
    "lessThan",
    "lessThanOrEquals",
    "severity",
    "message",
];

/// The data graph, with each blank node held as a [`SKOLEM`] IRI.
///
/// # Errors
/// A language-tagged literal: the validator's terms have no language tag, so the input is refused, naming it.
pub fn data_graph(input: &[RdfTriple]) -> Result<Graph, String> {
    let mut g = Graph::new();
    for t in input.iter() {
        let s = match &t.subject {
            RdfTerm::Iri(i) => i.clone(),
            RdfTerm::Blank(b) => format!("{SKOLEM}{b}"),
            RdfTerm::Literal { .. } => return Err("a literal in subject position".into()),
        };
        g.insert(s, t.predicate.clone(), term(&t.object)?);
    }
    Ok(g)
}

fn term(t: &RdfTerm) -> Result<Term, String> {
    match t {
        RdfTerm::Iri(i) => Ok(Term::Iri(i.clone())),
        RdfTerm::Blank(b) => Ok(Term::Iri(format!("{SKOLEM}{b}"))),
        RdfTerm::Literal {
            lexical,
            lang: Some(lang),
            ..
        } => Err(format!(
            "the language-tagged literal {lexical:?}@{lang}: the validator's terms carry no language tag"
        )),
        RdfTerm::Literal {
            lexical, datatype, ..
        } => Ok(Term::Literal {
            value: lexical.clone(),
            datatype: datatype.clone(),
        }),
    }
}

/// The shapes graph indexed by subject.
struct Index<'a> {
    out: BTreeMap<&'a RdfTerm, Vec<(&'a str, &'a RdfTerm)>>,
}

impl<'a> Index<'a> {
    fn new(g: &'a [RdfTriple]) -> Self {
        let mut out: BTreeMap<&RdfTerm, Vec<(&str, &RdfTerm)>> = BTreeMap::new();
        for t in g.iter() {
            out.entry(&t.subject)
                .or_default()
                .push((t.predicate.as_str(), &t.object));
        }
        Self { out }
    }

    fn edges(&self, s: &RdfTerm) -> &[(&'a str, &'a RdfTerm)] {
        self.out.get(s).map_or(&[], Vec::as_slice)
    }

    fn objs(&self, s: &RdfTerm, p: &str) -> Vec<&'a RdfTerm> {
        self.edges(s)
            .iter()
            .filter(|(q, _)| *q == p)
            .map(|(_, o)| *o)
            .collect()
    }

    /// The items of the RDF list at `head`, or `None` when it is not a well-formed list.
    fn list(&self, head: &'a RdfTerm) -> Option<Vec<&'a RdfTerm>> {
        let mut items = Vec::new();
        let mut cur = head;
        let mut seen = BTreeSet::new();
        while *cur != RdfTerm::Iri(RDF_NIL.into()) {
            if !seen.insert(cur) {
                return None;
            }
            let (first, rest) = (self.objs(cur, RDF_FIRST), self.objs(cur, RDF_REST));
            match (first.as_slice(), rest.as_slice()) {
                ([f], [r]) => {
                    items.push(*f);
                    cur = r;
                }
                _ => return None,
            }
        }
        Some(items)
    }
}

fn name(t: &RdfTerm) -> String {
    match t {
        RdfTerm::Iri(i) => i.clone(),
        RdfTerm::Blank(b) => format!("_:{b}"),
        RdfTerm::Literal { lexical, .. } => format!("{lexical:?}"),
    }
}

fn unsupported(shape: &RdfTerm, component: impl Into<String>) -> ShapeError {
    ShapeError::Unsupported {
        shape: name(shape),
        component: component.into(),
    }
}

fn malformed(shape: &RdfTerm, what: impl Into<String>) -> ShapeError {
    ShapeError::Malformed {
        shape: name(shape),
        what: what.into(),
    }
}

/// The shapes of a shapes graph: every node with `sh:targetClass` or `sh:targetNode`, and every `sh:NodeShape`
/// that is also an `rdfs:Class` (SHACL §2.1.3.3's implicit class target), in byte order of its IRI.
/// A node shape without a target selects no focus node, so it matters only as the object of an `sh:node`.
///
/// # Errors
/// [`ShapeError::Unsupported`] for a SHACL term outside the subset, [`ShapeError::Malformed`] for one written
/// wrong; either names the shape.
pub fn shapes(input: &[RdfTriple]) -> Result<Vec<NodeShape>, ShapeError> {
    let ix = Index::new(input);
    let (target_class, target_node) = (format!("{SH}targetClass"), format!("{SH}targetNode"));
    let mut roots: BTreeSet<&RdfTerm> = BTreeSet::new();
    for t in input {
        refuse_target_kind(t)?;
        if t.predicate == target_class
            || t.predicate == target_node
            || implicit_class(&ix, &t.subject)
        {
            roots.insert(&t.subject);
        }
    }
    roots.into_iter().map(|s| node_shape(&ix, s, 0)).collect()
}

/// The targets outside the subset: `sh:targetSubjectsOf` and `sh:targetObjectsOf`.
fn refuse_target_kind(t: &RdfTriple) -> Result<(), ShapeError> {
    for refused in ["targetSubjectsOf", "targetObjectsOf"] {
        if t.predicate == format!("{SH}{refused}") {
            return Err(unsupported(&t.subject, format!("sh:{refused}")));
        }
    }
    Ok(())
}

/// SHACL §2.1.3.3: a node shape that is also an `rdfs:Class` targets the instances of itself.
fn implicit_class(ix: &Index<'_>, s: &RdfTerm) -> bool {
    let types = ix.objs(s, RDF_TYPE);
    types.contains(&&RdfTerm::Iri(format!("{SH}NodeShape")))
        && types.contains(&&RdfTerm::Iri(RDFS_CLASS.into()))
}

/// The `sh:` keys of `s`, refusing any outside `allowed` by name.
fn sh_keys<'a>(
    ix: &Index<'a>,
    s: &RdfTerm,
    allowed: &[&str],
    what: &str,
) -> Result<BTreeMap<&'a str, Vec<&'a RdfTerm>>, ShapeError> {
    let mut keys: BTreeMap<&str, Vec<&RdfTerm>> = BTreeMap::new();
    for (p, o) in ix.edges(s) {
        let Some(local) = p.strip_prefix(SH) else {
            continue;
        };
        if NON_VALIDATING.contains(&local) {
            continue;
        }
        if !allowed.contains(&local) {
            return Err(unsupported(s, format!("sh:{local} on a {what}")));
        }
        keys.entry(local).or_default().push(o);
    }
    Ok(keys)
}

/// The one value of a single-valued key, or `None`.
fn one<'a>(
    s: &RdfTerm,
    keys: &BTreeMap<&str, Vec<&'a RdfTerm>>,
    k: &str,
) -> Result<Option<&'a RdfTerm>, ShapeError> {
    match keys.get(k).map(Vec::as_slice) {
        None | Some([]) => Ok(None),
        Some([x]) => Ok(Some(*x)),
        Some(many) => Err(unsupported(s, format!("{} values of sh:{k}", many.len()))),
    }
}

fn iri_of(s: &RdfTerm, k: &str, t: &RdfTerm) -> Result<String, ShapeError> {
    match t {
        RdfTerm::Iri(i) => Ok(i.clone()),
        _ => Err(unsupported(
            s,
            format!("sh:{k} {} (only an IRI is in the subset)", name(t)),
        )),
    }
}

fn lexical_of<'a>(s: &RdfTerm, k: &str, t: &'a RdfTerm) -> Result<&'a str, ShapeError> {
    match t {
        RdfTerm::Literal { lexical, .. } => Ok(lexical),
        _ => Err(malformed(s, format!("sh:{k} is not a literal"))),
    }
}

fn node_shape(ix: &Index<'_>, s: &RdfTerm, depth: usize) -> Result<NodeShape, ShapeError> {
    let RdfTerm::Iri(iri) = s else {
        return Err(unsupported(s, "a blank-node node shape"));
    };
    let keys = sh_keys(ix, s, NODE_KEYS, "node shape")?;
    let implicit = implicit_class(ix, s);
    if depth > 0
        && (implicit || keys.contains_key("targetClass") || keys.contains_key("targetNode"))
    {
        // A target on a nested shape is a second, independent shape; validating it once as nested would be wrong.
        return Err(unsupported(
            s,
            "a target on a shape that is also the object of sh:node",
        ));
    }
    let target_class = match (one(s, &keys, "targetClass")?, implicit) {
        // The validator holds one target class per shape.
        (Some(_), true) => {
            return Err(unsupported(
                s,
                "sh:targetClass beside an implicit class target",
            ))
        }
        (Some(t), false) => iri_of(s, "targetClass", t)?,
        (None, true) => iri.clone(),
        (None, false) => String::new(),
    };
    let target_nodes = keys
        .get("targetNode")
        .into_iter()
        .flatten()
        .map(|t| match t {
            RdfTerm::Iri(i) => Ok(i.clone()),
            other => Err(unsupported(s, format!("sh:targetNode {}", name(other)))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let closed = match one(s, &keys, "closed")? {
        None => false,
        Some(t) => match lexical_of(s, "closed", t)? {
            "true" | "1" => true,
            "false" | "0" => false,
            other => {
                return Err(malformed(
                    s,
                    format!("sh:closed {other:?} is not a boolean"),
                ))
            }
        },
    };
    let ignored_properties = match one(s, &keys, "ignoredProperties")? {
        None => Vec::new(),
        Some(head) => ix
            .list(head)
            .ok_or_else(|| malformed(s, "sh:ignoredProperties is not an RDF list"))?
            .into_iter()
            .map(|t| iri_of(s, "ignoredProperties", t))
            .collect::<Result<_, _>>()?,
    };
    let properties = keys
        .get("property")
        .into_iter()
        .flatten()
        .map(|p| property_shape(ix, p, depth))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(NodeShape {
        id: iri.clone(),
        target_class,
        closed,
        ignored_properties,
        properties,
        allow_empty: None,
        iri: Some(iri.clone()),
        target_nodes,
    })
}

fn count(
    s: &RdfTerm,
    keys: &BTreeMap<&str, Vec<&RdfTerm>>,
    k: &str,
) -> Result<Option<usize>, ShapeError> {
    match one(s, keys, k)? {
        None => Ok(None),
        Some(t) => lexical_of(s, k, t)?
            .parse()
            .map(Some)
            .map_err(|_| malformed(s, format!("sh:{k} is not a non-negative integer"))),
    }
}

fn opt_iri(
    s: &RdfTerm,
    keys: &BTreeMap<&str, Vec<&RdfTerm>>,
    k: &str,
) -> Result<Option<String>, ShapeError> {
    one(s, keys, k)?.map(|t| iri_of(s, k, t)).transpose()
}

/// The value of an IRI-valued key whose IRIs the subset enumerates (`sh:<local>` each); any other is refused.
fn choice<T: Clone>(
    s: &RdfTerm,
    keys: &BTreeMap<&str, Vec<&RdfTerm>>,
    k: &str,
    table: &[(&str, T)],
) -> Result<Option<T>, ShapeError> {
    let Some(iri) = opt_iri(s, keys, k)? else {
        return Ok(None);
    };
    table
        .iter()
        .find(|(local, _)| iri == format!("{SH}{local}"))
        .map(|(_, v)| Some(v.clone()))
        .ok_or_else(|| unsupported(s, format!("sh:{k} <{iri}>")))
}

fn property_shape(ix: &Index<'_>, s: &RdfTerm, depth: usize) -> Result<PropertyShape, ShapeError> {
    let keys = sh_keys(ix, s, PROPERTY_KEYS, "property shape")?;
    let path = match one(s, &keys, "path")? {
        Some(RdfTerm::Iri(i)) => i.clone(),
        Some(_) => {
            return Err(unsupported(
                s,
                "a path expression (only a single predicate is a path here)",
            ))
        }
        None => return Err(malformed(s, "a property shape has no sh:path")),
    };
    let node_kind = choice(
        s,
        &keys,
        "nodeKind",
        &[("IRI", NodeKind::Iri), ("Literal", NodeKind::Literal)],
    )?;
    let severity = choice(
        s,
        &keys,
        "severity",
        &[
            ("Violation", Severity::Violation),
            ("Warning", Severity::Warning),
        ],
    )?
    .unwrap_or(Severity::Violation);
    let r#in = match one(s, &keys, "in")? {
        None => None,
        Some(head) => Some(
            ix.list(head)
                .ok_or_else(|| malformed(s, "sh:in is not an RDF list"))?
                .into_iter()
                .map(|t| in_entry(s, t))
                .collect::<Result<Vec<_>, _>>()?,
        ),
    };
    let pattern = match one(s, &keys, "pattern")? {
        None => None,
        Some(t) => {
            let src = lexical_of(s, "pattern", t)?;
            let re = regex::Regex::new(src)
                .map_err(|e| malformed(s, format!("sh:pattern does not compile: {e}")))?;
            Some((src.to_string(), re))
        }
    };
    let node = match one(s, &keys, "node")? {
        None => None,
        Some(_) if depth >= 1 => return Err(unsupported(s, "sh:node nested more than one level")),
        Some(t) => Some(Box::new(node_shape(ix, t, depth + 1)?)),
    };
    let message = one(s, &keys, "message")?
        .map(|t| lexical_of(s, "message", t).map(String::from))
        .transpose()?;
    Ok(PropertyShape {
        path,
        min_count: count(s, &keys, "minCount")?,
        max_count: count(s, &keys, "maxCount")?,
        datatype: opt_iri(s, &keys, "datatype")?,
        class: opt_iri(s, &keys, "class")?,
        node_kind,
        r#in,
        pattern,
        min_length: count(s, &keys, "minLength")?,
        max_length: count(s, &keys, "maxLength")?,
        node,
        less_than: opt_iri(s, &keys, "lessThan")?,
        less_than_or_equals: opt_iri(s, &keys, "lessThanOrEquals")?,
        resolves: None,
        severity,
        iri: match s {
            RdfTerm::Iri(i) => Some(i.clone()),
            _ => None,
        },
        message,
    })
}

/// One `sh:in` member as a term. An IRI member has an empty datatype, so it can match only an IRI value.
fn in_entry(s: &RdfTerm, t: &RdfTerm) -> Result<InEntry, ShapeError> {
    match t {
        RdfTerm::Iri(i) => Ok(InEntry {
            lexical: i.clone(),
            datatype: String::new(),
        }),
        RdfTerm::Literal {
            lexical,
            datatype,
            lang: None,
        } => Ok(InEntry {
            lexical: lexical.clone(),
            datatype: datatype.clone(),
        }),
        other => Err(unsupported(s, format!("sh:in member {}", name(other)))),
    }
}

#[cfg(test)]
#[path = "shapes_rdf_tests.rs"]
pub(crate) mod tests;
