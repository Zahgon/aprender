// `apr qa` with every gate skipped must be NOT MEASURED and exit non-zero (#5226).
//
// THE DEFECT: `gates_pass` is `all(|g| g.passed || g.skipped)`, which is vacuously true when
// every gate skipped. The only lower bound, `check_min_executed`, returns true when
// `--min-executed` is unset, which is the CLI default. So `apr qa m.gguf` with every `--skip-*`
// flag printed "All QA gates passed (0 executed, 12 skipped)" and exited 0, having measured
// nothing. These rows drive the production finalizer over the full gate registry.
#[cfg(test)]
mod qa_all_skipped_not_measured_tests {
    use super::*;

    #[derive(Clone, Copy)]
    enum G {
        Skip,
        Pass,
        Fail,
    }

    /// Every registered gate, the first `head.len()` of them as given, the rest skipped.
    fn gates(head: &[G]) -> Vec<GateResult> {
        QA_GATES
            .iter()
            .enumerate()
            .map(|(i, name)| match head.get(i).copied().unwrap_or(G::Skip) {
                G::Skip => GateResult::skipped(name, "by the case table"),
                G::Pass => GateResult::passed(name, "ok", None, None, Duration::ZERO),
                G::Fail => GateResult::failed(name, "bad", None, None, Duration::ZERO),
            })
            .collect()
    }

    fn finalize(head: &[G], min_executed: Option<usize>) -> QaReport {
        let config = QaConfig {
            json: true,
            min_executed,
            ..QaConfig::default()
        };
        finalize_qa_report(Path::new("m.gguf"), &Instant::now(), gates(head), &config)
            .expect("finalize_qa_report")
    }

    #[test]
    fn qa_all_skipped_case_table() {
        // (label, gates, --min-executed, want passed, want summary prefix)
        let rows: [(&str, &[G], Option<usize>, bool, &str); 7] = [
            (
                "every gate skipped, no bound",
                &[],
                None,
                false,
                "NOT MEASURED",
            ),
            (
                "every gate skipped, --min-executed 0",
                &[],
                Some(0),
                false,
                "NOT MEASURED",
            ),
            (
                "every gate skipped, --min-executed 3",
                &[],
                Some(3),
                false,
                "NOT MEASURED",
            ),
            (
                "one gate passed, rest skipped",
                &[G::Pass],
                None,
                true,
                "All QA gates passed",
            ),
            (
                "one gate failed, rest skipped",
                &[G::Fail],
                None,
                false,
                "Failed gates",
            ),
            (
                "two passed under --min-executed 3",
                &[G::Pass, G::Pass],
                Some(3),
                false,
                "Insufficient",
            ),
            (
                "every gate passed",
                &[G::Pass; 12],
                None,
                true,
                "All QA gates passed",
            ),
        ];
        for (label, head, min, want_passed, want_prefix) in rows {
            let r = finalize(head, min);
            assert_eq!(
                r.passed, want_passed,
                "{label}: passed (summary: {})",
                r.summary
            );
            assert!(
                r.summary.starts_with(want_prefix),
                "{label}: summary {:?} does not start with {want_prefix:?}",
                r.summary
            );
        }
    }

    #[test]
    fn qa_measured_nothing_is_only_zero_executed() {
        assert!(run_measured_nothing(0));
        assert!(!run_measured_nothing(1));
    }
}
