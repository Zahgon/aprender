//! Output-shape tests for the scorecard renderers `format_table` and
//! `format_profile_markdown` (#5667, #5644). Contract: llm-scorecard-format-v1.

use jugar_probar::llm::score::{
    format_profile_markdown, format_table, ConsistencyScore, MetricScore, ProfileEntry,
    ProfileScorecard, PromptCategory, RuntimeScore, Scorecard,
};
use std::collections::HashMap;

fn cell(score: u8, best: bool) -> MetricScore {
    MetricScore {
        value: 1.0,
        score,
        best,
        jitter_penalty: None,
    }
}

fn runtime(name: &str, keys: &[&str], composite: f64, grade: &str) -> RuntimeScore {
    let mut metrics = HashMap::new();
    for (i, k) in keys.iter().enumerate() {
        metrics.insert((*k).to_string(), cell(90 - i as u8, i == 0));
    }
    RuntimeScore {
        name: name.into(),
        source_file: String::new(),
        metrics,
        composite,
        grade: grade.into(),
    }
}

fn card(concurrency: usize, runtimes: Vec<RuntimeScore>) -> Scorecard {
    Scorecard {
        contract_version: "2.0.0".into(),
        timestamp: "t".into(),
        concurrency,
        runtimes,
    }
}

#[test]
fn interactive_table_has_c1_columns_and_marks_best_cells() {
    let rt = runtime("apr", &["decode_tok_s", "ttft_p50_ms"], 87.46, "B");
    let out = format_table(&card(1, vec![rt]));
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], "Inference Runtime Scorecard — Interactive (c=1)");
    assert!(lines[2].contains("Decode") && lines[2].contains("Tail"));
    assert!(!lines[2].contains("Aggr"), "c=1 has no aggregate column");
    assert_eq!(lines[4], "-".repeat(74));
    let row = lines[5];
    assert!(row.starts_with("apr "));
    assert!(row.contains(" 90*"), "best cell carries a star: {row}");
    assert!(row.contains(" 89 "), "non-best cell has no star: {row}");
    assert!(
        row.ends_with("87.5 B"),
        "composite rounds to 1 dp then grade: {row}"
    );
    assert_eq!(lines.last(), Some(&"* = best in class"));
}

#[test]
fn throughput_table_has_aggregate_columns_and_scenario_label() {
    let rt = runtime(
        "llama",
        &["aggregate_tok_s", "throughput_scaling"],
        55.0,
        "D",
    );
    let out = format_table(&card(8, vec![rt]));
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], "Inference Runtime Scorecard — Throughput (c=8)");
    assert!(lines[2].contains("Aggr") && lines[2].contains("Scale"));
    assert_eq!(lines[4], "-".repeat(88));
    assert!(lines[5].starts_with("llama"));
    assert!(lines[5].contains(" 90*"));
    assert!(lines[5].ends_with("55.0 D"));
}

#[test]
fn missing_metric_renders_dash_and_empty_card_has_only_chrome() {
    let rt = runtime("bare", &[], 0.0, "F");
    let out = format_table(&card(1, vec![rt]));
    let row = out.lines().nth(5).expect("runtime row");
    assert_eq!(
        row.matches("   -").count(),
        5,
        "all five c=1 cells are dashes: {row}"
    );

    let empty = format_table(&card(2, vec![]));
    assert_eq!(
        empty.lines().count(),
        7,
        "title, blank, 2 headers, rule, blank, legend"
    );
}

fn entry(name: &str, profile: PromptCategory, composite: f64) -> ProfileEntry {
    ProfileEntry {
        name: name.into(),
        profile,
        avg_prompt_tokens: 127.6,
        composite,
        grade: "A".into(),
        decode_tok_s: 120.25,
        ttft_p50_ms: 33.33,
        itl_p50_ms: 8.04,
    }
}

#[test]
fn profile_markdown_rows_and_consistency_section() {
    let sc = ProfileScorecard {
        timestamp: "t".into(),
        concurrency: 1,
        entries: vec![entry("apr", PromptCategory::Medium, 91.26)],
        consistency: vec![ConsistencyScore {
            name: "apr".into(),
            best_score: 95.0,
            worst_score: 80.5,
            consistency: 84.74,
            grade: "B".into(),
        }],
    };
    let out = format_profile_markdown(&sc);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], "## Per-Prompt-Profile Scores");
    assert!(lines[2].starts_with("| Runtime | Profile |"));
    assert_eq!(
        lines[4],
        "| apr | medium | 128 | 120.2 | 33.3 | 8.0 | **91.3 (A)** |"
    );
    assert!(out.contains("### Profile Consistency"));
    assert_eq!(lines.last(), Some(&"| apr | 95.0 | 80.5 | 85% | B |"));
}

#[test]
fn profile_markdown_omits_consistency_when_empty() {
    let sc = ProfileScorecard {
        timestamp: "t".into(),
        concurrency: 1,
        entries: vec![
            entry("a", PromptCategory::Micro, 1.0),
            entry("b", PromptCategory::Long, 2.0),
        ],
        consistency: vec![],
    };
    let out = format_profile_markdown(&sc);
    assert!(!out.contains("Consistency"));
    assert!(out.contains("| a | micro |") && out.contains("| b | long |"));
    assert_eq!(
        out.lines().count(),
        6,
        "heading, blank, header, rule, two rows"
    );
}
