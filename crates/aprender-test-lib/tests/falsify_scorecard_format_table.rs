//! FALSIFY-TEST-LIB-SCORECARD-TABLE-001: `format_table` picks its column set from concurrency.
#![cfg(feature = "llm")]

use jugar_probar::llm::score::{format_table, MetricScore, RuntimeScore, Scorecard};
use std::collections::HashMap;

fn ms(score: u8, best: bool) -> MetricScore {
    MetricScore {
        value: 1.0,
        score,
        best,
        jitter_penalty: None,
    }
}

fn card(concurrency: usize, metrics: &[(&str, MetricScore)]) -> Scorecard {
    let metrics: HashMap<String, MetricScore> = metrics
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect();
    Scorecard {
        contract_version: "2.0.0".into(),
        timestamp: "t".into(),
        concurrency,
        runtimes: vec![RuntimeScore {
            name: "realizr".into(),
            source_file: "r.json".into(),
            metrics,
            composite: 87.0,
            grade: "A".into(),
        }],
    }
}

#[test]
fn interactive_table_has_interactive_header_and_row() {
    let out = format_table(&card(1, &[("decode_tok_s", ms(95, true))]));
    assert!(out.contains("Interactive (c=1)"));
    assert!(out.contains("Tail"));
    assert!(!out.contains("Scale"));
    assert!(out.contains(" 95*"));
    assert!(out.contains("realizr"));
}

#[test]
fn throughput_table_names_concurrency_and_scale_column() {
    let out = format_table(&card(4, &[("aggregate_tok_s", ms(80, false))]));
    assert!(out.contains("Throughput (c=4)"));
    assert!(out.contains("Scale"));
    assert!(!out.contains("Tail"));
    assert!(out.contains(" 80 "));
}

#[test]
fn missing_metrics_render_as_dash_and_footer_is_last() {
    let out = format_table(&card(1, &[]));
    assert!(out.contains("   -"));
    assert!(out.ends_with("* = best in class"));
}
