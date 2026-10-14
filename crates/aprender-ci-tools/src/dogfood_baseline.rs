//! Port of `scripts/dogfood_baseline.py`: re-derive the G2.3 coverage baselines from the
//! ledger (`docs/audits/surface_audit.csv`), print them as the YAML block the contract
//! quotes, hold that output to the T2 pairing rule, and with `--check` require every
//! printed baseline line to appear verbatim in `contracts/apr-dogfood-coverage-v1.yaml`.
//!
//! A baseline nobody can re-derive is a number someone eventually "corrects" to a rounder
//! one. 17.1% is 142/830; it is not "about 17%".
//!
//! Exit 0 when the output pairs and (with `--check`) the contract matches, else 1. An
//! unreadable file, a short row or a non-integer `quality_1_10` exits 1, as the original's
//! traceback did.

use crate::dogfood_pairing::enforce_pairing;
use crate::pystr::py_strip;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

pub const CSV_PATH: &str = "docs/audits/surface_audit.csv";
pub const CONTRACT_PATH: &str = "contracts/apr-dogfood-coverage-v1.yaml";

const BANDS: [(&str, i64, i64); 5] = [
    ("q1_2", 1, 2),
    ("q3_4", 3, 4),
    ("q5_6", 5, 6),
    ("q7_8", 7, 8),
    ("q9_10", 9, 10),
];

/// One ledger row, reduced to the fields the baselines read.
pub struct Row {
    binary: String,
    quality: i64,
    covered: bool,
    unknown_hardware: bool,
    low_confidence: bool,
    cluster_label: String,
}

/// `(total, covered)` per key, in the original's order.
type Tally = Vec<(String, (usize, usize))>;

/// What `compute` returned.
pub struct Metrics {
    rows: usize,
    covered: usize,
    binaries: usize,
    binaries_at_zero: usize,
    unknown_hardware: usize,
    low_and_uncovered: usize,
    broken_and_ungated: usize,
    per_binary: Tally,
    per_band: Tally,
    per_cluster: Tally,
    clusters: usize,
    clusters_at_zero: usize,
}

/// `csv.DictReader` over the ledger. Blank lines are skipped, as DictReader skips them; a
/// row missing a read column is refused (the original crashed on `None.strip()`).
pub fn load(bytes: &[u8]) -> Result<Vec<Row>, String> {
    let mut rdr = csv::ReaderBuilder::new().flexible(true).from_reader(bytes);
    let headers = rdr
        .headers()
        .map_err(|e| format!("{CSV_PATH}: {e}"))?
        .clone();
    // `csv.DictReader` on an empty file has no fieldnames and no rows: zero rows, not a
    // missing column.
    if headers.is_empty() && bytes.iter().all(|b| matches!(b, b'\r' | b'\n')) {
        return Ok(Vec::new());
    }
    // The csv crate strips a UTF-8 BOM; the original opened the file as plain UTF-8, so
    // its first column was named "\u{feff}binary" and `row["binary"]` raised.
    let bom = bytes.starts_with(b"\xef\xbb\xbf");
    let col = |name: &str| {
        headers
            .iter()
            .enumerate()
            .position(|(i, h)| h == name && !(bom && i == 0))
            .ok_or_else(|| format!("{CSV_PATH}: no `{name}` column"))
    };
    let (binary, quality, hardware, skill, label, confidence) = (
        col("binary")?,
        col("quality_1_10")?,
        col("verified_hardware")?,
        col("in_dogfood_skill")?,
        col("cluster_label")?,
        col("confidence")?,
    );
    let mut rows = Vec::new();
    for (i, rec) in rdr.records().enumerate() {
        let rec = rec.map_err(|e| format!("{CSV_PATH}: {e}"))?;
        let line = i + 2;
        let field = |c: usize| {
            rec.get(c)
                .ok_or_else(|| format!("{CSV_PATH}: row {line} is short ({} fields)", rec.len()))
        };
        let q = field(quality)?;
        rows.push(Row {
            binary: field(binary)?.to_owned(),
            quality: py_strip(q).parse().map_err(|_| {
                format!("{CSV_PATH}: row {line}: quality_1_10 {q:?} is not an integer")
            })?,
            // `(row["in_dogfood_skill"] or "")`: only this column tolerated a missing value.
            covered: py_strip(rec.get(skill).unwrap_or("")).to_lowercase() == "yes",
            unknown_hardware: py_strip(field(hardware)?) == "UNKNOWN",
            low_confidence: py_strip(field(confidence)?).to_lowercase() == "low",
            cluster_label: py_strip(field(label)?).to_owned(),
        });
    }
    Ok(rows)
}

fn count_covered<'a>(rows: impl IntoIterator<Item = &'a Row>) -> usize {
    rows.into_iter().filter(|r| r.covered).count()
}

/// Group in first-seen order, then order by size descending, key ascending.
fn grouped(rows: &[Row], key: impl Fn(&Row) -> &str) -> Tally {
    let mut first_seen: Vec<String> = Vec::new();
    let mut groups: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in rows {
        let k = key(r);
        let e = groups.entry(k.to_owned()).or_insert_with(|| {
            first_seen.push(k.to_owned());
            (0, 0)
        });
        e.0 += 1;
        e.1 += usize::from(r.covered);
    }
    let mut out: Tally = first_seen
        .into_iter()
        .map(|k| {
            let v = groups[&k];
            (k, v)
        })
        .collect();
    // Python's `sorted` is stable and the key is total, so the first-seen order never shows.
    out.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then_with(|| a.0.cmp(&b.0)));
    out
}

pub fn compute(rows: &[Row]) -> Metrics {
    let per_binary = grouped(rows, |r| &r.binary);
    let per_cluster = grouped(rows, |r| &r.cluster_label);
    let per_band = BANDS
        .iter()
        .map(|&(name, lo, hi)| {
            let sel: Vec<&Row> = rows
                .iter()
                .filter(|r| lo <= r.quality && r.quality <= hi)
                .collect();
            (name.to_owned(), (sel.len(), count_covered(sel)))
        })
        .collect();
    Metrics {
        rows: rows.len(),
        covered: count_covered(rows),
        binaries: per_binary.len(),
        binaries_at_zero: per_binary.iter().filter(|(_, (_, c))| *c == 0).count(),
        unknown_hardware: rows.iter().filter(|r| r.unknown_hardware).count(),
        low_and_uncovered: rows
            .iter()
            .filter(|r| r.low_confidence && !r.covered)
            .count(),
        broken_and_ungated: rows.iter().filter(|r| r.quality <= 4 && !r.covered).count(),
        clusters: per_cluster.len(),
        clusters_at_zero: per_cluster.iter().filter(|(_, (_, c))| *c == 0).count(),
        per_binary,
        per_band,
        per_cluster,
    }
}

#[allow(clippy::cast_precision_loss)] // ledger counts are far below 2^52
fn pct(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        100.0 * num as f64 / den as f64
    }
}

/// `report_lines`.
pub fn report_lines(m: &Metrics) -> Vec<String> {
    let mut out = Vec::new();
    #[allow(clippy::cast_precision_loss)]
    let ratio = if m.rows == 0 {
        0.0
    } else {
        m.covered as f64 / m.rows as f64
    };
    out.push("# totals".to_owned());
    out.push(format!("rows: {}", m.rows));
    out.push(format!("covered: {}", m.covered));
    out.push(format!(
        "ratio: {ratio:.4}            # {}/{} -- {:.1}%",
        m.covered,
        m.rows,
        100.0 * ratio
    ));
    out.push(format!("binaries: {}", m.binaries));
    out.push(format!("binaries_at_zero_coverage: {}", m.binaries_at_zero));
    out.push(format!("clusters: {}", m.clusters));
    out.push(format!("clusters_at_zero_coverage: {}", m.clusters_at_zero));
    // The paired line. It exists so this printer is not VACUOUSLY compliant: a pairing rule
    // applied to output that states no cluster ratio proves nothing.
    out.push(format!(
        "# clusters gated {}/{}, features gated {}/{}",
        m.clusters - m.clusters_at_zero,
        m.clusters,
        m.covered,
        m.rows
    ));
    out.push(format!("unknown_hardware: {}", m.unknown_hardware));
    out.push(format!("low_and_uncovered: {}", m.low_and_uncovered));
    out.push(format!("broken_and_ungated: {}", m.broken_and_ungated));
    out.push(String::new());
    out.push("# per_binary".to_owned());
    for (b, (total, cov)) in &m.per_binary {
        out.push(format!("{b}: {{total: {total}, covered: {cov}}}"));
    }
    out.push(String::new());
    out.push("# per_band".to_owned());
    for (name, (total, cov)) in &m.per_band {
        out.push(format!(
            "{name}: {{total: {total}, covered: {cov}}}     # {:.1}%",
            pct(*cov, *total)
        ));
    }
    out.push(String::new());
    out.push("# per_cluster  -- ALLOCATION OF GATE EFFORT, the reason clustering earns".to_owned());
    out.push("#                 its place. Nobody chose this allocation; it accreted.".to_owned());
    let tot_gates = if m.covered == 0 { 1 } else { m.covered };
    for (lab, (total, cov)) in &m.per_cluster {
        out.push(format!(
            "{lab}: {{total: {total}, covered: {cov}}}     # {:.1}% of the cluster, {:.1}% of all gate effort",
            pct(*cov, *total),
            pct(*cov, tot_gates)
        ));
    }
    out
}

/// `expected_lines`: every literal the contract must carry, as `(label, line)`.
pub fn expected_lines(m: &Metrics) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (kind, tally) in [
        ("per_binary", &m.per_binary),
        ("per_band", &m.per_band),
        ("per_cluster", &m.per_cluster),
    ] {
        for (k, (total, cov)) in tally {
            out.push((
                format!("{kind} {k}"),
                format!("{k}: {{total: {total}, covered: {cov}}}"),
            ));
        }
    }
    for (label, v) in [
        ("rows", m.rows),
        ("covered", m.covered),
        ("unknown", m.unknown_hardware),
        ("low_and_uncovered", m.low_and_uncovered),
        ("count", m.broken_and_ungated),
        ("clusters", m.clusters),
        ("clusters_at_zero_coverage", m.clusters_at_zero),
    ] {
        out.push((format!("totals {label}"), format!("{label}: {v}")));
    }
    out
}

/// `check`: the drift report appended to `out`, and its exit code.
pub fn check(m: &Metrics, contract: &str, out: &mut String) -> u8 {
    let bad: Vec<String> = expected_lines(m)
        .into_iter()
        .filter(|(_, line)| !contract.contains(line.as_str()))
        .map(|(label, line)| format!("{label} -> {line}"))
        .collect();
    out.push('\n');
    if bad.is_empty() {
        out.push_str("CHECK PASSED: every baseline in the contract matches the ledger.\n");
        return 0;
    }
    out.push_str("CONTRACT DRIFT -- the ledger no longer matches the committed baseline:\n");
    for x in bad {
        let _ = writeln!(out, "    {x}");
    }
    1
}

/// The whole run from the repository root: `(stdout, exit code)`, or `(already printed,
/// reason)`. The original printed the report before it opened the contract, so an
/// unreadable contract still leaves the report on stdout.
pub fn run(root: &Path, check_contract: bool) -> Result<(String, u8), (String, String)> {
    let csv = std::fs::read(root.join(CSV_PATH))
        .map_err(|e| (String::new(), format!("{CSV_PATH}: {e}")))?;
    let m = compute(&load(&csv).map_err(|e| (String::new(), e))?);
    let lines = report_lines(&m);
    let mut findings = Vec::new();
    let paired = enforce_pairing(
        &lines,
        &mut findings,
        m.clusters as u64,
        m.rows as u64,
        "aprender-ci-tools dogfood-baseline output",
    );
    let mut out = String::new();
    for ln in lines.iter().chain(&findings) {
        out.push_str(ln);
        out.push('\n');
    }
    let mut rc = u8::from(!paired);
    if check_contract {
        let contract = std::fs::read(root.join(CONTRACT_PATH))
            .map_err(|e| format!("{CONTRACT_PATH}: {e}"))
            .and_then(|b| String::from_utf8(b).map_err(|e| format!("{CONTRACT_PATH}: {e}")));
        let contract = match contract {
            Ok(c) => c,
            Err(reason) => return Err((out, reason)),
        };
        let drift = check(&m, &contract, &mut out);
        if drift != 0 {
            rc = drift;
        }
    }
    Ok((out, rc))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEDGER: &str = "binary,feature,quality_1_10,verified_hardware,top_competitor,in_dogfood_skill,cluster_id,cluster_label,evidence_path,confidence\n\
a,a x,6,UNKNOWN,t,yes,1,lab-b,p,high\n\
\n\
a,a y,3,gx,t,no,1,lab-b,p,low\n\
b,b z,9,UNKNOWN,t,YES ,2, lab-a ,p,Low\n";

    #[test]
    fn computes_every_baseline() {
        let m = compute(&load(LEDGER.as_bytes()).expect("ledger"));
        assert_eq!(
            (m.rows, m.covered, m.binaries, m.binaries_at_zero),
            (3, 2, 2, 0)
        );
        assert_eq!(
            (
                m.unknown_hardware,
                m.low_and_uncovered,
                m.broken_and_ungated
            ),
            (2, 1, 1)
        );
        assert_eq!(m.per_binary[0], ("a".to_owned(), (2, 1)));
        assert_eq!(m.per_cluster[1], ("lab-a".to_owned(), (1, 1)));
        let lines = report_lines(&m);
        assert!(lines.contains(&"ratio: 0.6667            # 2/3 -- 66.7%".to_owned()));
        assert!(lines.contains(&"q3_4: {total: 1, covered: 0}     # 0.0%".to_owned()));
        let mut f = Vec::new();
        assert!(enforce_pairing(&lines, &mut f, 2, 3, "x"), "{f:?}");
    }

    #[test]
    fn check_reports_drift_and_passes_on_a_match() {
        let m = compute(&load(LEDGER.as_bytes()).expect("ledger"));
        let full: String = expected_lines(&m)
            .into_iter()
            .map(|(_, l)| l + "\n")
            .collect();
        let mut out = String::new();
        assert_eq!(check(&m, &full, &mut out), 0, "{out}");
        let mut out = String::new();
        assert_eq!(check(&m, &full.replace("rows: 3", "rows: 4"), &mut out), 1);
        assert!(out.contains("    totals rows -> rows: 3"), "{out}");
    }

    #[test]
    fn a_non_integer_quality_is_refused() {
        let bad = LEDGER.replace(",6,", ",six,");
        assert!(load(bad.as_bytes()).is_err());
    }
}
