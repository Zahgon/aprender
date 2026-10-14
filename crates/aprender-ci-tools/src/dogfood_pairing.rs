//! Port of the T2 pairing rule in `scripts/lib/dogfood_coverage_gate.py`
//! (`_RATIO`, `_CLUSTER_FRAC`, `_FEATURE_FRAC`, `_CLUSTER_PCT`, `_PAIR_PRAGMA`,
//! `_SCALAR_DECL`, `states_cluster_ratio`, `states_feature_ratio`, `enforce_pairing`).
//!
//! Every emitted line stating a CLUSTER-level ratio must state the underlying
//! FEATURE-level ratio beside it. The rule keys on the NUMBER (the ledger's cluster and
//! feature counts) before any phrase, because a phrase can be reworded and a denominator
//! cannot. One module so the printer (`dogfood-baseline`) and the gate share one rule.
//!
//! CPython `\s` also matches `\x1c`–`\x1f`; the classes below spell that out. Python's
//! `int()` reads any Unicode decimal digit; here a non-ASCII digit run never equals a
//! count, which only matters for a ledger count written in non-ASCII digits (none is).

use crate::pystr::is_py_space;
use regex::Regex;
use std::sync::LazyLock;

/// `_RATIO` without its lookarounds (the `regex` crate has none): `N/M` or `N of M`,
/// anchored at a digit run. [`denominators`] applies `(?<![\d./])` and `(?![\d./])`.
static RATIO_HEAD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(\d+)[\s\x1c-\x1f]*(?:/|\bof\b)[\s\x1c-\x1f]*(\d+)").expect("RATIO_HEAD")
});
static RATIO_EDGE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[\d./]$").expect("RATIO_EDGE"));
static DIGIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\d$").expect("DIGIT"));
static CLUSTER_FRAC: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)cluster[s\-]?[^,;|]*?\b(\d+)[\s\x1c-\x1f]*(?:/|\bof\b)[\s\x1c-\x1f]*(\d+)")
        .expect("CLUSTER_FRAC")
});
static FEATURE_FRAC: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)feature[s\-]?[^,;|]*?\b(\d+)[\s\x1c-\x1f]*(?:/|\bof\b)[\s\x1c-\x1f]*(\d+)")
        .expect("FEATURE_FRAC")
});
static CLUSTER_PCT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(\bclusters\b[^|]{0,30}?\d+(?:\.\d+)?[\s\x1c-\x1f]*%|\d+(?:\.\d+)?[\s\x1c-\x1f]*%[^|]{0,30}?\bclusters\b)",
    )
    .expect("CLUSTER_PCT")
});
static PAIR_PRAGMA: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)t2-pairing[: ]*allow[ \t]*\(").expect("PAIR_PRAGMA"));
static SCALAR_DECL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"^[\s\x1c-\x1f\-]*[\w.\[\]"'`]+[\s\x1c-\x1f]*[:=][\s\x1c-\x1f]*\d+[\s\x1c-\x1f]*,?[\s\x1c-\x1f]*$"#,
    )
    .expect("SCALAR_DECL")
});

fn char_matches(re: &Regex, c: char) -> bool {
    let mut buf = [0u8; 4];
    re.is_match(c.encode_utf8(&mut buf))
}

/// The ASCII value of a digit run; `None` for a non-ASCII digit or an overflow (neither
/// can equal a ledger count).
fn ascii_value(digits: &str) -> Option<u64> {
    if digits.bytes().all(|b| b.is_ascii_digit()) {
        digits.parse().ok()
    } else {
        None
    }
}

/// `{int(m.group(2)) for m in _RATIO.finditer(text)}`, as a list.
///
/// A `_RATIO` match can only start at the first digit of a run (the lookbehind refuses a
/// digit before it), and greedy `\d+` has no shorter alternative that could pass the
/// lookahead, so trying the anchored head at each run start and checking the next char is
/// the same match set. A match consumes its text, as `finditer` does.
pub fn denominators(text: &str) -> Vec<u64> {
    let mut out = Vec::new();
    let mut prev: Option<char> = None;
    let mut pos = 0;
    while let Some(c) = text[pos..].chars().next() {
        let starts_run =
            char_matches(&DIGIT, c) && prev.is_none_or(|p| !char_matches(&RATIO_EDGE, p));
        if starts_run {
            if let Some(caps) = RATIO_HEAD.captures(&text[pos..]) {
                let end = pos + caps.get(0).map_or(0, |m| m.end());
                let next = text[end..].chars().next();
                if next.is_none_or(|n| !char_matches(&RATIO_EDGE, n)) {
                    if let Some(v) = caps.get(2).and_then(|m| ascii_value(m.as_str())) {
                        out.push(v);
                    }
                    prev = text[..end].chars().next_back();
                    pos = end;
                    continue;
                }
            }
        }
        prev = Some(c);
        pos += c.len_utf8();
    }
    out
}

/// `states_cluster_ratio`: the union of four triggers, the number first.
pub fn states_cluster_ratio(line: &str, n_clusters: u64) -> bool {
    if n_clusters != 0 && denominators(line).contains(&n_clusters) {
        return true;
    }
    if n_clusters != 0 {
        let near = Regex::new(&format!(
            r"(?i)(\b{n}\b[^|]{{0,20}}?\bclusters\b|\bclusters\b[^|]{{0,20}}?\b{n}\b)",
            n = n_clusters
        ))
        .expect("cluster-count regex");
        if near.is_match(line) {
            return true;
        }
    }
    CLUSTER_FRAC.is_match(line) || CLUSTER_PCT.is_match(line)
}

/// `states_feature_ratio`.
pub fn states_feature_ratio(line: &str, n_features: u64) -> bool {
    if n_features != 0 && denominators(line).contains(&n_features) {
        return true;
    }
    FEATURE_FRAC.is_match(line)
}

/// The report's own name in `enforce_pairing`'s default `where`; only that report is held
/// to the vacuity guard.
pub const OWN_REPORT: &str = "the per-cluster report";

/// `enforce_pairing`: true when the text is admissible; otherwise one finding is pushed,
/// worded as the original's.
pub fn enforce_pairing<S: AsRef<str>>(
    lines: &[S],
    findings: &mut Vec<String>,
    n_clusters: u64,
    n_features: u64,
    place: &str,
) -> bool {
    let mut offenders: Vec<&str> = Vec::new();
    let mut seen = false;
    for ln in lines {
        let ln = ln.as_ref();
        if SCALAR_DECL.is_match(ln) || !states_cluster_ratio(ln, n_clusters) {
            continue;
        }
        seen = true;
        if states_feature_ratio(ln, n_features) || PAIR_PRAGMA.is_match(ln) {
            continue;
        }
        offenders.push(ln);
    }
    if !offenders.is_empty() {
        let shown: Vec<String> = offenders
            .iter()
            .take(10)
            .map(|ln| format!("        | {}", ln.trim_end_matches(is_py_space)))
            .collect();
        findings.push(format!(
            "G2.6 T2 FAIL ({place}): {} line(s) state a CLUSTER-level ratio with no \
             FEATURE-level ratio beside it. One gate in a 95-member cluster is 1%, not \
             \"covered\"; reporting the proxy alone is how a coverage gate becomes theatre \
             while looking stricter than before:\n{}",
            offenders.len(),
            shown.join("\n")
        ));
        return false;
    }
    if !seen && place == OWN_REPORT {
        findings.push(
            "G2.6 T2 FAIL: the per-cluster report emitted no cluster-level ratio at all, so \
             the pairing rule checked nothing. An empty report is not a satisfied one."
                .to_owned(),
        );
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denominators_follow_the_lookarounds() {
        let cases: &[(&str, &[u64])] = &[
            ("5/14", &[14]),
            ("5 of 14 clusters", &[14]),
            ("5 OF 14", &[14]),
            ("1/2/3", &[]),
            ("1.5/2", &[]),
            ("2/3.", &[]),
            ("x1/2 and 3 of 4", &[2, 4]),
            ("1 of 2 of 3", &[2]),
            ("5of 14", &[]),
            ("142/830 -- 17.1%", &[830]),
        ];
        for (text, want) in cases {
            assert_eq!(denominators(text), *want, "{text:?}");
        }
    }

    #[test]
    fn pairing_case_table() {
        // (line, admissible on its own) with 14 clusters and 904 features.
        let cases: &[(&str, bool)] = &[
            ("5 of 14 clusters gated (35.7%)", false),
            ("5 of 14 clusters gated, 144 of 904 features gated", true),
            ("# clusters gated 5/14, features gated 144/904", true),
            ("35.7% of clusters gated", false),
            ("clusters gated: 35.7%", false),
            ("14 clusters, 5 gated", false),
            ("clusters: 14", true),
            ("one gate in a 95-member cluster is 1%", true),
            ("5/14 t2-pairing allow (a dated sample)", true),
            ("nothing about it", true),
        ];
        for (line, ok) in cases {
            let mut f = Vec::new();
            assert_eq!(
                enforce_pairing(&[*line], &mut f, 14, 904, "a scanned file"),
                *ok,
                "{line:?}"
            );
            assert_eq!(f.is_empty(), *ok, "{line:?}");
        }
    }

    #[test]
    fn own_report_must_state_a_cluster_ratio() {
        let mut f = Vec::new();
        assert!(!enforce_pairing(
            &["rows: 904"],
            &mut f,
            14,
            904,
            OWN_REPORT
        ));
        assert!(f[0].contains("checked nothing"), "{f:?}");
        let mut f = Vec::new();
        assert!(enforce_pairing(
            &["rows: 904"],
            &mut f,
            14,
            904,
            "elsewhere"
        ));
    }
}
