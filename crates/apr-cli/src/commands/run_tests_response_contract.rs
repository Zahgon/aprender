// #3720 — the apr response contract on `apr run --json` / `--stream`: every exit writes
// one document with `status` (ok | refused | failed) and, unless ok, `error {kind,
// message, exit_code}`; an empty completion is a failure, never exit 0 with "".

fn contract_result(text: &str) -> RunResult {
    RunResult {
        text: text.to_string(),
        duration_secs: 1.0,
        cached: true,
        tokens_generated: Some(3),
        tok_per_sec: Some(3.0),
        used_gpu: Some(true),
        gpu_attempted: Some(true),
        generated_tokens: Some(vec![1, 2, 3]),
        token_texts: None,
        usage: RunUsage::default(),
        constraint_refusal: None,
    }
}

fn contract_refusal() -> crate::error::ConstraintRefusal {
    crate::error::ConstraintRefusal {
        kind: "SchemaViolation",
        message: "SchemaViolation: the output is empty".to_string(),
        removed_by: None,
        finish_reason: Some("constraint_complete"),
    }
}

/// Every variant: its kind, its status, its exit code. A new variant that is not
/// classified fails to compile (the matches are exhaustive); this pins the values.
#[test]
fn response_contract_every_cli_error_has_its_kind_status_and_exit_code() {
    use crate::error::{CliError, RunStatus};
    let p = std::path::PathBuf::from("m.gguf");
    let s = || "x".to_string();
    let table: Vec<(CliError, &str, RunStatus, u8)> = vec![
        (CliError::FileNotFound(p.clone()), "file_not_found", RunStatus::Refused, 3),
        (CliError::NotAFile(p), "not_a_file", RunStatus::Refused, 3),
        (CliError::InvalidFormat(s()), "invalid_format", RunStatus::Refused, 4),
        (CliError::InvalidInput(s()), "invalid_input", RunStatus::Refused, 4),
        (
            CliError::InvalidModelFile {
                format: "GGUF",
                message: s(),
            },
            "invalid_model_file",
            RunStatus::Refused,
            4,
        ),
        (
            CliError::MissingCompanionFile(s()),
            "missing_companion_file",
            RunStatus::Refused,
            3,
        ),
        (
            CliError::Io(std::io::Error::other("x")),
            "io_error",
            RunStatus::Failed,
            7,
        ),
        (CliError::ValidationFailed(s()), "validation_failed", RunStatus::Failed, 5),
        (CliError::Aprender(s()), "aprender_error", RunStatus::Failed, 1),
        (CliError::ModelLoadFailed(s()), "model_load_failed", RunStatus::Failed, 6),
        (CliError::InferenceFailed(s()), "inference_failed", RunStatus::Failed, 8),
        (CliError::FeatureDisabled(s()), "feature_disabled", RunStatus::Refused, 9),
        (CliError::NetworkError(s()), "network_error", RunStatus::Failed, 10),
        (CliError::HttpNotFound(s()), "http_not_found", RunStatus::Refused, 11),
        (CliError::NotImplemented(s()), "not_implemented", RunStatus::Refused, 12),
        (CliError::ParityFailed(s()), "parity_failed", RunStatus::Refused, 13),
        (
            CliError::BackendUnavailable(s()),
            "backend_unavailable",
            RunStatus::Refused,
            14,
        ),
        (
            CliError::ConstraintRefused(contract_refusal()),
            "constraint_refused",
            RunStatus::Refused,
            15,
        ),
        (CliError::EmptyCompletion(s()), "empty_completion", RunStatus::Failed, 8),
    ];
    assert_eq!(table.len(), 19, "one row per CliError variant");
    for (e, kind, status, code) in &table {
        assert_eq!(e.kind(), *kind, "{e:?}");
        assert_eq!(e.status(), *status, "{e:?}");
        assert_eq!(e.exit_code_value(), *code, "{e:?}");
        let env = e.envelope();
        assert_eq!(env["status"], status.as_str());
        assert_eq!(env["error"]["kind"], *kind);
        assert_eq!(env["error"]["exit_code"], u64::from(*code), "never re-mapped");
        assert_eq!(env["error"]["message"], e.to_string());
    }
}

#[test]
fn response_contract_status_spellings() {
    use crate::error::RunStatus;
    assert_eq!(RunStatus::Ok.as_str(), "ok");
    assert_eq!(RunStatus::Refused.as_str(), "refused");
    assert_eq!(RunStatus::Failed.as_str(), "failed");
}

#[test]
fn response_contract_a_run_that_ended_ok_says_so_and_carries_no_error() {
    let doc = final_document(&contract_result("4"), "m.gguf", 16, false, None);
    assert_eq!(doc["status"], "ok");
    assert!(doc.get("error").is_none(), "error iff status != ok: {doc}");
    assert_eq!(doc["text"], "4");
    assert_eq!(build_final_json(&contract_result("4"), "m.gguf", 16, false), doc);
}

#[test]
fn response_contract_an_empty_completion_is_a_failure_that_names_why() {
    for empty in ["", "  \n"] {
        let result = contract_result(empty);
        let err = require_answer(&result).expect_err("empty is not ok");
        assert_eq!(err.kind(), "empty_completion");
        assert_eq!(err.status(), crate::error::RunStatus::Failed);
        assert_eq!(err.exit_code_value(), 8);
        assert!(err.to_string().contains("generation produced no text"), "{err}");
        assert!(err.to_string().contains("3 tokens generated"), "{err}");
        let doc = final_document(&result, "m.gguf", 16, false, Some(&err));
        assert_eq!(doc["status"], "failed");
        assert_eq!(doc["error"]["kind"], "empty_completion");
        assert_eq!(doc["error"]["exit_code"], 8);
    }
    require_answer(&contract_result("4")).expect("an answer is ok");
}

#[test]
fn response_contract_an_empty_run_exits_non_zero_through_reconcile() {
    let err = reconcile_and_emit(&contract_result(""), "m.gguf", "text", 16, true, false, false)
        .expect_err("reconcile must refuse an empty answer");
    assert_eq!(err.kind(), "empty_completion");
    reconcile_and_emit(&contract_result("4"), "m.gguf", "text", 16, true, false, false)
        .expect("a real answer passes");
}

#[test]
fn response_contract_a_constraint_refusal_wins_over_empty() {
    let mut result = contract_result("");
    result.constraint_refusal = Some(contract_refusal());
    let err = reconcile_and_emit(&result, "m.gguf", "text", 16, true, false, false)
        .expect_err("refused");
    assert_eq!(err.kind(), "constraint_refused");
    assert_eq!(err.exit_code_value(), 15);
}

#[test]
fn response_contract_a_run_that_never_produced_a_result_still_writes_one_document() {
    let e = crate::error::CliError::ModelLoadFailed("cannot mmap m.gguf".into());
    let doc = error_document(&e, "m.gguf", false);
    assert_eq!(doc["status"], "failed");
    assert_eq!(doc["error"]["kind"], "model_load_failed");
    assert_eq!(doc["error"]["exit_code"], 6);
    assert_eq!(doc["model"], "m.gguf");
    assert!(doc["text"].is_null());
    assert!(doc.get("event").is_none());
    assert!(doc.get("refusal").is_none(), "only a constraint refusal carries one");
    assert_eq!(error_document(&e, "m", true)["event"], "final", "the stream's terminal event");
}

#[test]
fn response_contract_a_constraint_refused_before_output_keeps_its_refusal_fields() {
    let e = crate::error::CliError::ConstraintRefused(contract_refusal());
    let doc = error_document(&e, "m.gguf", false);
    assert_eq!(doc["status"], "refused");
    assert_eq!(doc["error"]["kind"], "constraint_refused");
    assert_eq!(doc["error"]["exit_code"], 15);
    assert_eq!(doc["refusal"]["kind"], "SchemaViolation");
    assert_eq!(doc["finish_reason"], "constraint_complete");
    assert!(doc["text"].is_null());
}

#[test]
fn response_contract_the_stream_final_event_carries_the_envelope() {
    let mut buf = Vec::new();
    let err = crate::error::CliError::EmptyCompletion("empty completion: x".into());
    write_stream_output_with(&mut buf, &contract_result(""), "m.gguf", 8, false, Some(&err))
        .expect("write");
    let out = String::from_utf8(buf).expect("utf8");
    let last = out.lines().last().expect("a final line");
    let v: serde_json::Value = serde_json::from_str(last).expect("json");
    assert_eq!(v["event"], "final");
    assert_eq!(v["status"], "failed");
    assert_eq!(v["error"]["kind"], "empty_completion");

    let mut ok = Vec::new();
    write_stream_output(&mut ok, &contract_result("4"), "m.gguf", 8, false).expect("write");
    let ok = String::from_utf8(ok).expect("utf8");
    let v: serde_json::Value =
        serde_json::from_str(ok.lines().last().expect("final")).expect("json");
    assert_eq!(v["status"], "ok");
    assert!(v.get("error").is_none());
}
