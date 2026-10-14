//! `pv ontology export --owl` / `pv ontology tbox` — ONT-001 §3.8, row ONT-2c; `pv ontology read` — CRUX-SHACL S3.
//!
//! `export --owl` writes Σ as OWL 2 functional syntax with the in-house writer, byte-deterministic, so CI can
//! `cmp` a fresh export against the tracked `contracts/ontology.ofn` (R-18). `tbox` writes the told-closure
//! classification, `contracts/tbox-report.json`, which is ADVISORY: the `tbox` lint gate maps it to
//! `Unknown{Advisory}` and it never arms (R-7: no inferred fact arms a merge).
//!
//! Exit codes follow `pv lint`: 3 `error:` when Σ is malformed, cannot be written as OWL, or the TBox
//! precondition fails. In that last case the report is still printed, so the refusal names its axiom.
//!
//! `read` prints a Turtle or N-Triples file as canonical N-Triples, so an outside engine's dump of the same file
//! can be compared with it (CRUX R-INPUT, R-RETURN). It is not a gate and nothing it reads reaches `contracts.nt`.
//! It exits 1 when the file is malformed, naming the line, and 2 when the file cannot be read or its syntax is
//! unknown.

use std::path::Path;

use provable_contracts::ontology::owl;
use provable_contracts::ontology::read::{self, Syntax};
use provable_contracts::ontology::report;
use provable_contracts::ontology::shapes::{self, Report};
use provable_contracts::ontology::shapes_rdf::{self, RdfTerm, RdfTriple};
use provable_contracts::ontology::sigma::Sigma;

use crate::cli::OntologyCommand;

pub fn run(command: &OntologyCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        OntologyCommand::Export {
            sigma,
            owl: as_owl,
            write,
        } => {
            if !as_owl {
                eprintln!("error: `pv ontology export` writes one format today; pass --owl");
                std::process::exit(2);
            }
            let export = load_export(sigma);
            emit(sigma, "ontology.ofn", &owl::to_ofn(&export), *write)
        }
        OntologyCommand::Tbox { sigma, write } => {
            let export = load_export(sigma);
            let report = owl::tbox(&export);
            emit(
                sigma,
                "tbox-report.json",
                &owl::report_json(&report),
                *write,
            )?;
            if !report.precondition.holds {
                for r in &report.precondition.refused {
                    eprintln!("error: told-closure precondition fails: {r}");
                }
                std::process::exit(3);
            }
            Ok(())
        }
        OntologyCommand::Read {
            input,
            syntax,
            base,
        } => {
            read_rdf(input, syntax.as_deref(), base.as_deref());
            Ok(())
        }
        OntologyCommand::Validate {
            shapes,
            data,
            format,
            base,
        } => {
            validate_rdf(shapes, data, format, base.as_deref());
            Ok(())
        }
    }
}

fn load_export(sigma_path: &Path) -> owl::OwlExport {
    let fail = |msg: String| -> ! {
        eprintln!("error: {msg}");
        std::process::exit(3);
    };
    let text = std::fs::read_to_string(sigma_path)
        .unwrap_or_else(|e| fail(format!("cannot read Σ {}: {e}", sigma_path.display())));
    let sigma = Sigma::from_yaml(&text).unwrap_or_else(|e| fail(format!("Σ: {e}")));
    if let Err(e) = sigma.check_integrity() {
        fail(format!("Σ: {e}"));
    }
    owl::export(&sigma).unwrap_or_else(|e| fail(e.to_string()))
}

fn emit(
    sigma_path: &Path,
    name: &str,
    body: &str,
    write: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if write {
        let dir = sigma_path.parent().unwrap_or_else(|| Path::new("."));
        let out = dir.join(name);
        std::fs::write(&out, body)?;
        eprintln!("wrote {}", out.display());
    } else {
        print!("{body}");
    }
    Ok(())
}

/// `pv ontology read`: the canonical N-Triples on stdout, or exit 1 (malformed) or 2 (unreadable, unknown syntax).
fn read_rdf(input: &Path, syntax: Option<&str>, base: Option<&str>) {
    let usage = |msg: String| -> ! {
        eprintln!("error: {msg}");
        std::process::exit(2);
    };
    let syntax = match syntax {
        Some(name) => Syntax::from_name(name)
            .unwrap_or_else(|| usage(format!("unknown syntax {name:?}: use turtle or ntriples"))),
        None => Syntax::for_path(input).unwrap_or_else(|| {
            usage(format!(
                "{}: not .ttl or .nt; pass --syntax",
                input.display()
            ))
        }),
    };
    let text = std::fs::read_to_string(input)
        .unwrap_or_else(|e| usage(format!("cannot read {}: {e}", input.display())));
    let base = base.map_or_else(|| file_iri(input), str::to_string);
    match read::read(&text, syntax, Some(&base)) {
        Ok(graph) => print!("{}", graph.to_ntriples()),
        Err(e) => {
            eprintln!("error: {}: {e}", input.display());
            std::process::exit(1);
        }
    }
}

/// The `file://` IRI of a file, the default base (the path is made absolute first).
fn file_iri(path: &Path) -> String {
    let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    format!("file://{}", abs.display())
}

/// Read one RDF file for `validate`: exit 2 when it cannot be read or its syntax is unknown, 3 when malformed.
/// Why `validate` stopped before a report: the exit code and the message.
#[derive(Debug)]
pub(crate) struct Refusal {
    pub code: i32,
    pub message: String,
}

fn refuse(code: i32, message: String) -> Refusal {
    Refusal { code, message }
}

/// Read one RDF file for `validate`: 2 when it cannot be read or its syntax is unknown, 3 when malformed.
fn read_for_validate(input: &Path, base: Option<&str>) -> Result<Vec<RdfTriple>, Refusal> {
    let syntax = Syntax::for_path(input)
        .ok_or_else(|| refuse(2, format!("{}: not .ttl or .nt", input.display())))?;
    let text = std::fs::read_to_string(input)
        .map_err(|e| refuse(2, format!("cannot read {}: {e}", input.display())))?;
    let base = base.map_or_else(|| file_iri(input), str::to_string);
    let graph = read::read(&text, syntax, Some(&base))
        .map_err(|e| refuse(3, format!("{}: {e}", input.display())))?;
    Ok(graph
        .iter()
        .map(|t| RdfTriple {
            subject: rdf_term(&t.subject),
            predicate: t.predicate.clone(),
            object: rdf_term(&t.object),
        })
        .collect())
}

/// The reader's term as the validator's input term. This command is the one place both are named (R-15).
fn rdf_term(t: &read::InputTerm) -> RdfTerm {
    match t {
        read::InputTerm::Iri(i) => RdfTerm::Iri(i.clone()),
        read::InputTerm::Blank(b) => RdfTerm::Blank(b.clone()),
        read::InputTerm::Literal {
            lexical,
            datatype,
            lang,
        } => RdfTerm::Literal {
            lexical: lexical.clone(),
            datatype: datatype.clone(),
            lang: lang.clone(),
        },
    }
}

/// The validator's report on `data_path` against `shapes_path`, or the refusal: 3 for a shapes graph outside the
/// subset or a data graph the validator cannot hold, naming the term.
pub(crate) fn validate_files(
    shapes_path: &Path,
    data_path: &Path,
    base: Option<&str>,
) -> Result<Report, Refusal> {
    let shapes_graph = read_for_validate(shapes_path, base)?;
    let data_graph = read_for_validate(data_path, base)?;
    let shapes = shapes_rdf::shapes(&shapes_graph)
        .map_err(|e| refuse(3, format!("{}: {e}", shapes_path.display())))?;
    let data = shapes_rdf::data_graph(&data_graph)
        .map_err(|e| refuse(3, format!("{}: {e}", data_path.display())))?;
    Ok(shapes::validate(&data, &shapes))
}

/// `pv ontology validate` (CRUX §14.6 S4): the in-house validator on any shapes and data graph, its report
/// written as a W3C `sh:ValidationReport`. A shapes graph outside the subset is refused, never half-checked.
fn validate_rdf(shapes_path: &Path, data_path: &Path, format: &str, base: Option<&str>) {
    if !matches!(format, "turtle" | "ntriples" | "json") {
        eprintln!("error: unknown format {format:?}: use turtle, ntriples or json");
        std::process::exit(2);
    }
    let r = validate_files(shapes_path, data_path, base).unwrap_or_else(|e| {
        eprintln!("error: {}", e.message);
        std::process::exit(e.code);
    });
    match format {
        "json" => println!(
            "{}",
            serde_json::to_string_pretty(&report::to_json(&r)).unwrap_or_default()
        ),
        "ntriples" => print!("{}", report::to_ntriples(&r)),
        _ => print!("{}", report::to_turtle(&r)),
    }
    if !r.conforms() {
        std::process::exit(1);
    }
}

#[cfg(test)]
#[path = "ontology_validate_tests.rs"]
mod validate_tests;
