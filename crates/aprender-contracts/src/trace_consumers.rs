//! TRACE-001 TR-08 (aprender#5314): `trace-consumers-v1`, the registry of who reads
//! each trace, profile and metrics surface the built binaries expose.
//!
//! The surface list is never written down. [`derive`] reads it off the tracked
//! built-binary snapshot ([`crate::ontology::extract::binary::SNAPSHOT`], written by
//! `scripts/dogfood_surfaces.sh --snapshot`): every command, route and MCP tool of
//! every bin target, kept when one of its tokens is in [`TOKENS`]. The `apr` pair
//! (`apr-cli` and the root facade) is keyed by bin name, so it is one surface.
//! Flags are not in the snapshot and so are not derived.
//!
//! [`check`] holds the registry ([`REGISTRY`]) to the derived set. Each
//! [`Finding`] is RED: a surface nobody registered, an entry for a surface the
//! binaries no longer expose, an entry with neither a consumer nor a deprecation,
//! an `active` entry without its first green run, a deprecation past its
//! `remove_by` version. An empty derived set is RED too, never a clean sweep.

use crate::ontology::extract::binary::Target;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// The registry, relative to the repo root.
pub const REGISTRY: &str = "contracts/trace-consumers-v1.yaml";

/// A surface is a trace/profile/metrics surface when its binary name or its own
/// name holds one of these tokens (split on space, `/`, `.`, `_`, `:` and `-`).
/// Whole tokens, so `contract` and `extract` never match `trace`.
pub const TOKENS: [&str; 14] = [
    "bench",
    "benchmark",
    "cbtop",
    "memtrace",
    "metrics",
    "monitor",
    "otlp",
    "profile",
    "profiling",
    "roofline",
    "telemetry",
    "trace",
    "traces",
    "tracing",
];

/// What kind of interface a surface is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceKind {
    /// A (sub)command path, e.g. `cgp trace`.
    Command,
    /// An HTTP route, e.g. `GET /metrics`.
    Route,
    /// An MCP tool, e.g. `apr.trace`.
    Tool,
}

/// One derived surface: the bin name, its kind, and the surface as the snapshot spells it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Surface {
    pub binary: String,
    pub kind: SurfaceKind,
    pub surface: String,
}

impl fmt::Display for Surface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {:?} {:?}", self.binary, self.kind, self.surface)
    }
}

fn is_trace_surface(binary: &str, surface: &str) -> bool {
    let lower = format!("{binary} {surface}").to_lowercase();
    lower
        .split([' ', '/', '.', '_', ':', '-'])
        .any(|tok| TOKENS.contains(&tok))
}

/// The trace/profile/metrics surfaces of the snapshot's targets, byte-ordered.
#[must_use]
pub fn derive(targets: &[Target]) -> BTreeSet<Surface> {
    let mut out = BTreeSet::new();
    for t in targets {
        let kinds = [
            (SurfaceKind::Command, &t.commands),
            (SurfaceKind::Route, &t.routes),
            (SurfaceKind::Tool, &t.tools),
        ];
        for (kind, set) in kinds {
            for s in set.iter().filter(|s| is_trace_surface(&t.name, s)) {
                out.insert(Surface {
                    binary: t.name.clone(),
                    kind,
                    surface: s.clone(),
                });
            }
        }
    }
    out
}

/// Who reads a surface and acts on it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Consumer {
    pub kind: ConsumerKind,
    /// `<file>#<job, target or heading>`.
    #[serde(rename = "ref")]
    pub reference: String,
}

/// TRACE-001 §2.2: `ledger`, `gate` or `runbook`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsumerKind {
    Ledger,
    Gate,
    Runbook,
}

/// An entry's status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    /// Its consumer runs it and has a recorded first green run.
    Active,
    /// Registered, with no first green run on record yet.
    ReportOnly,
    /// To be removed by `remove_by`.
    Deprecated,
}

/// The run that first proved the consumer green.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FirstGreen {
    pub run_url: String,
    pub sha: String,
}

/// One registry entry.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub binary: String,
    pub kind: SurfaceKind,
    pub surface: String,
    #[serde(default)]
    pub consumer: Option<Consumer>,
    pub status: Status,
    #[serde(default)]
    pub first_green: Option<FirstGreen>,
    #[serde(default)]
    pub remove_by: Option<String>,
    #[serde(default)]
    pub platforms: Vec<String>,
}

impl Entry {
    fn key(&self) -> Surface {
        Surface {
            binary: self.binary.clone(),
            kind: self.kind,
            surface: self.surface.clone(),
        }
    }
}

#[derive(Deserialize)]
struct RegistryFile {
    consumers: Vec<Entry>,
}

/// The `consumers:` block of the registry file.
///
/// # Errors
///
/// The YAML does not parse, or an entry carries a key this schema does not know.
pub fn parse_registry(yaml: &str) -> Result<Vec<Entry>, String> {
    serde_yaml::from_str::<RegistryFile>(yaml)
        .map(|r| r.consumers)
        .map_err(|e| format!("{REGISTRY}: {e}"))
}

/// Why the registry does not hold for the derived surfaces. Every variant is RED.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finding {
    /// The snapshot yielded no trace surface: a broken derivation, never a pass.
    NothingDerived,
    /// A derived surface with no entry.
    Unregistered(Surface),
    /// An entry for a surface the binaries no longer expose.
    NotDerived(Surface),
    /// Two entries for one surface.
    Duplicate(Surface),
    /// Neither a consumer nor `deprecated` with a `remove_by`.
    NoConsumer(Surface),
    /// `active` with no `first_green`.
    ActiveWithoutFirstGreen(Surface),
    /// `deprecated` with no `remove_by`, or one that is not `X.Y.Z`.
    BadRemoveBy(Surface),
    /// `deprecated` and the workspace version has reached `remove_by`.
    DeprecatedPastRemoveBy {
        surface: Surface,
        remove_by: String,
        version: String,
    },
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NothingDerived => write!(f, "the snapshot yielded no trace surface"),
            Self::Unregistered(s) => write!(f, "unregistered surface: {s}"),
            Self::NotDerived(s) => write!(f, "registered but not exposed by any binary: {s}"),
            Self::Duplicate(s) => write!(f, "registered twice: {s}"),
            Self::NoConsumer(s) => write!(f, "no consumer and not deprecated: {s}"),
            Self::ActiveWithoutFirstGreen(s) => write!(f, "active without first_green: {s}"),
            Self::BadRemoveBy(s) => write!(f, "deprecated without an X.Y.Z remove_by: {s}"),
            Self::DeprecatedPastRemoveBy {
                surface,
                remove_by,
                version,
            } => write!(
                f,
                "deprecated past remove_by {remove_by} (workspace {version}): {surface}"
            ),
        }
    }
}

fn semver(v: &str) -> Option<(u64, u64, u64)> {
    let mut it = v.trim().split('.').map(str::parse::<u64>);
    let parsed = (it.next()?.ok()?, it.next()?.ok()?, it.next()?.ok()?);
    it.next().is_none().then_some(parsed)
}

fn entry_findings(e: &Entry, version: &str, out: &mut Vec<Finding>) {
    let key = e.key();
    match e.status {
        Status::Deprecated => match e.remove_by.as_deref().map(|r| (r, semver(r))) {
            Some((r, Some(by))) => {
                if semver(version).is_some_and(|now| now >= by) {
                    out.push(Finding::DeprecatedPastRemoveBy {
                        surface: key,
                        remove_by: r.to_string(),
                        version: version.to_string(),
                    });
                }
            }
            _ => out.push(Finding::BadRemoveBy(key)),
        },
        Status::Active | Status::ReportOnly if e.consumer.is_none() => {
            out.push(Finding::NoConsumer(key));
        }
        Status::Active if e.first_green.is_none() => {
            out.push(Finding::ActiveWithoutFirstGreen(key));
        }
        _ => {}
    }
}

/// Holds `registry` to `derived` at workspace `version`. Empty means it holds.
#[must_use]
pub fn check(derived: &BTreeSet<Surface>, registry: &[Entry], version: &str) -> Vec<Finding> {
    if derived.is_empty() {
        return vec![Finding::NothingDerived];
    }
    let mut out = Vec::new();
    let mut seen: BTreeMap<Surface, usize> = BTreeMap::new();
    for e in registry {
        *seen.entry(e.key()).or_default() += 1;
        entry_findings(e, version, &mut out);
    }
    for (s, n) in &seen {
        if *n > 1 {
            out.push(Finding::Duplicate(s.clone()));
        }
        if !derived.contains(s) {
            out.push(Finding::NotDerived(s.clone()));
        }
    }
    out.extend(
        derived
            .iter()
            .filter(|s| !seen.contains_key(*s))
            .cloned()
            .map(Finding::Unregistered),
    );
    out
}

#[cfg(test)]
#[path = "trace_consumers_tests.rs"]
mod tests;
