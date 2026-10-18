/// Setup SIGINT handler for Jidoka cleanup
///
/// Toyota Way: Stop the line, clean up, never leave orphan processes.
fn setup_signal_handler() {
    if let Err(e) = ctrlc::set_handler(move || {
        let count = aprender_qa_runner::process::kill_all_registered();
        eprintln!("\n[JIDOKA] SIGINT received. Reaping {count} child process(es)...");
        eprintln!("[JIDOKA] Toyota Way: Stop the line, clean up, exit.");
        std::process::exit(130); // 128 + SIGINT(2)
    }) {
        eprintln!("Warning: Failed to set signal handler: {e}");
    }
}

/// Print playbook execution status including model, workers, and test configuration
#[allow(clippy::fn_params_excessive_bools)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn print_run_status(
    playbook: &aprender_qa_runner::Playbook,
    effective_workers: usize,
    model_path: Option<&str>,
    dry_run: bool,
    timeout: u64,
    skip_conversion_tests: bool,
    hf_parity: bool,
    hf_corpus_path: &str,
    hf_model_family: Option<&str>,
) {
    let mut out = std::io::stdout().lock();
    // stdout write errors (closed pipe) are not actionable for a status banner
    let _ = write_run_status(
        &mut out,
        playbook,
        effective_workers,
        model_path,
        dry_run,
        timeout,
        skip_conversion_tests,
        hf_parity,
        hf_corpus_path,
        hf_model_family,
    );
}

/// Writer seam for [`print_run_status`] so the banner is testable.
#[allow(
    clippy::fn_params_excessive_bools,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn write_run_status(
    out: &mut impl std::io::Write,
    playbook: &aprender_qa_runner::Playbook,
    effective_workers: usize,
    model_path: Option<&str>,
    dry_run: bool,
    timeout: u64,
    skip_conversion_tests: bool,
    hf_parity: bool,
    hf_corpus_path: &str,
    hf_model_family: Option<&str>,
) -> std::io::Result<()> {
    writeln!(
        out,
        "{} {}",
        "Running playbook:".bold(),
        playbook.name.bold().cyan()
    )?;
    writeln!(
        out,
        "  {} {}",
        "Total tests:".dimmed(),
        playbook.total_tests()
    )?;
    writeln!(out, "  {} {dry_run}", "Dry run:".dimmed())?;
    writeln!(
        out,
        "  {} {:?}",
        "Model size:".dimmed(),
        playbook.size_category()
    )?;
    if let Some(path) = model_path {
        writeln!(out, "  {} {path}", "Model path:".dimmed())?;
    }
    writeln!(
        out,
        "  {} {} (max for size: {})",
        "Workers:".dimmed(),
        effective_workers,
        playbook.model.size_category.max_workers()
    )?;
    writeln!(out, "  {} {timeout}ms", "Timeout:".dimmed())?;

    // Conversion test status (P0 CRITICAL)
    if !skip_conversion_tests && model_path.is_some() {
        writeln!(
            out,
            "  {} {}",
            "Conversion tests:".dimmed(),
            "ENABLED (P0 CRITICAL)".bold().green()
        )?;
    } else if skip_conversion_tests {
        writeln!(
            out,
            "  {} {}",
            "Conversion tests:".dimmed(),
            "DISABLED (WARNING: P0 tests skipped)".bold().yellow()
        )?;
    }

    // HF parity status
    if hf_parity {
        writeln!(out, "  {} {}", "HF parity:".dimmed(), "ENABLED".green())?;
        writeln!(out, "    {} {hf_corpus_path}", "Corpus:".dimmed())?;
        if let Some(family) = hf_model_family {
            writeln!(out, "    {} {family}", "Model family:".dimmed())?;
        } else {
            writeln!(
                out,
                "    {} {}",
                "Model family:".dimmed(),
                "NOT SET (required for parity tests)".yellow()
            )?;
        }
    }
    Ok(())
}

/// Load, validate, and execute a playbook with the given configuration
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
fn run_playbook(
    playbook_path: &PathBuf,
    output_dir: &PathBuf,
    failure_policy: &str,
    dry_run: bool,
    workers: usize,
    model_path: Option<String>,
    timeout: u64,
    no_gpu: bool,
    skip_conversion_tests: bool,
    run_tool_tests_flag: bool,
    profile_ci: bool,
    hf_parity: bool,
    hf_corpus_path: &str,
    hf_model_family: Option<String>,
    no_integrity_check: bool,
    metadata_only: bool,
) {
    if failure_policy == "fail-fast" {
        log_environment();
    }

    let playbook = load_playbook_or_exit(playbook_path);

    if !no_integrity_check {
        verify_playbook_lock_or_exit(playbook_path, &playbook.name);
    }

    validate_failure_policy_or_exit(failure_policy);

    let effective_workers = resolve_effective_workers(&playbook, workers);

    print_run_status(
        &playbook,
        effective_workers,
        model_path.as_deref(),
        dry_run,
        timeout,
        skip_conversion_tests,
        hf_parity,
        hf_corpus_path,
        hf_model_family.as_deref(),
    );

    let run_config = build_run_config(
        failure_policy,
        dry_run,
        effective_workers,
        model_path.clone(),
        timeout,
        no_gpu,
        skip_conversion_tests,
        run_tool_tests_flag,
        profile_ci,
        hf_parity,
        hf_corpus_path,
        hf_model_family,
        metadata_only,
    );

    let config = match build_execution_config(&run_config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    if run_tool_tests_flag && !dry_run {
        run_tool_tests_or_exit(model_path.as_deref(), no_gpu, timeout);
    }

    let result = match execute_playbook(&playbook, config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    print_playbook_results(&result);
    persist_and_exit(dry_run, &result, output_dir);
}

fn load_playbook_or_exit(playbook_path: &PathBuf) -> aprender_qa_runner::Playbook {
    println!(
        "{} {}",
        "Loading playbook:".bold().cyan(),
        playbook_path.display().to_string().cyan()
    );
    match load_playbook(playbook_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e.red());
            std::process::exit(1);
        }
    }
}

fn validate_failure_policy_or_exit(failure_policy: &str) {
    if parse_failure_policy(failure_policy).is_err() {
        eprintln!("Unknown failure policy: {failure_policy}");
        std::process::exit(1);
    }
}

/// §3.4: Resource-aware scheduling — enforce worker limits based on model size.
fn resolve_effective_workers(playbook: &aprender_qa_runner::Playbook, requested: usize) -> usize {
    let effective = playbook.effective_max_workers(requested);
    if effective < requested {
        eprintln!(
            "{} Model size {:?} caps workers at {} (requested {})",
            "[RESOURCE]".yellow(),
            playbook.size_category(),
            effective,
            requested
        );
    }
    effective
}

#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
fn build_run_config(
    failure_policy: &str,
    dry_run: bool,
    workers: usize,
    model_path: Option<String>,
    timeout: u64,
    no_gpu: bool,
    skip_conversion_tests: bool,
    run_tool_tests: bool,
    run_profile_ci: bool,
    hf_parity: bool,
    hf_corpus_path: &str,
    hf_model_family: Option<String>,
    metadata_only: bool,
) -> PlaybookRunConfig {
    PlaybookRunConfig {
        failure_policy: failure_policy.to_string(),
        dry_run,
        workers,
        model_path,
        timeout,
        no_gpu,
        skip_conversion_tests,
        run_tool_tests,
        run_profile_ci,
        run_hf_parity: hf_parity,
        hf_parity_corpus_path: if hf_parity {
            Some(hf_corpus_path.to_string())
        } else {
            None
        },
        hf_parity_model_family: hf_model_family,
        metadata_only,
    }
}

fn run_tool_tests_or_exit(model_path: Option<&str>, no_gpu: bool, timeout: u64) {
    let Some(mp) = model_path else {
        eprintln!("Error: --run-tool-tests requires --model-path to be set");
        std::process::exit(1);
    };
    println!("\n{}", "=== Running APR Tool Tests ===".bold().cyan());
    let tool_executor = ToolExecutor::new(mp.to_string(), no_gpu, timeout);
    let tool_results = tool_executor.execute_all();
    let tool_passed = tool_results.iter().filter(|r| r.passed).count();
    let tool_failed = tool_results.len() - tool_passed;
    println!(
        "  Tool tests: {} passed, {} failed",
        tool_passed.to_string().green(),
        if tool_failed > 0 {
            tool_failed.to_string().red()
        } else {
            tool_failed.to_string().dimmed()
        }
    );
    if tool_failed > 0 {
        eprintln!("Aborting: {tool_failed} tool test(s) failed (Jidoka: stop the line)");
        std::process::exit(1);
    }
}

/// Persist evidence (unless dry-run) and exit non-zero on any failure.
///
/// Dry-run skips evidence persistence: output directory may not be writable and
/// the purpose of dry-run is inspection, not artifact production. Jidoka: stop
/// the line on any failed scenario or gateway blow.
fn persist_and_exit(
    dry_run: bool,
    result: &aprender_qa_runner::ExecutionResult,
    output_dir: &PathBuf,
) {
    if !dry_run && !save_playbook_evidence(result, output_dir) {
        eprintln!("Error: evidence save failed — results may be lost");
        std::process::exit(1);
    }

    if !dry_run && (result.failed > 0 || result.gateway_failed.is_some()) {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod run_status_tests {
    use super::*;

    const YAML: &str = r#"
name: status-playbook
version: "1.0.0"
description: "Status banner"
model:
  hf_repo: "test/model"
  formats:
    - gguf
  quantizations:
    - q4_k_m
  size_category: tiny
test_matrix:
  modalities:
    - run
  backends:
    - cpu
  scenario_count: 1
  seed: 42
  timeout_ms: 30000
gates:
  g1_model_loads: true
  g2_basic_inference: true
  g3_no_crashes: true
  g4_not_garbage: true
"#;

    fn banner(model: Option<&str>, skip: bool, parity: bool, family: Option<&str>) -> String {
        colored::control::set_override(false);
        let playbook = aprender_qa_runner::Playbook::from_yaml(YAML).expect("valid yaml");
        let mut buf = Vec::new();
        write_run_status(
            &mut buf, &playbook, 3, model, true, 1500, skip, parity, "corp/dir", family,
        )
        .expect("Vec write cannot fail");
        String::from_utf8(buf).expect("utf8")
    }

    #[test]
    fn header_lists_name_workers_timeout_and_dry_run() {
        let out = banner(None, false, false, None);
        assert!(
            out.starts_with("Running playbook: status-playbook\n"),
            "{out}"
        );
        assert!(out.contains("  Dry run: true\n"), "{out}");
        assert!(out.contains("  Workers: 3 (max for size: "), "{out}");
        assert!(out.contains("  Timeout: 1500ms\n"), "{out}");
        assert!(!out.contains("Model path:"), "{out}");
    }

    #[test]
    fn model_path_line_only_when_given() {
        let out = banner(Some("/m/x.gguf"), false, false, None);
        assert!(out.contains("  Model path: /m/x.gguf\n"), "{out}");
    }

    #[test]
    fn conversion_status_has_three_states() {
        let enabled = banner(Some("/m"), false, false, None);
        assert!(
            enabled.contains("Conversion tests: ENABLED (P0 CRITICAL)"),
            "{enabled}"
        );
        let disabled = banner(Some("/m"), true, false, None);
        assert!(
            disabled.contains("Conversion tests: DISABLED (WARNING: P0 tests skipped)"),
            "{disabled}"
        );
        let skipped_no_model = banner(None, true, false, None);
        assert!(skipped_no_model.contains("DISABLED"), "{skipped_no_model}");
        let silent = banner(None, false, false, None);
        assert!(!silent.contains("Conversion tests:"), "{silent}");
    }

    #[test]
    fn hf_parity_block_only_when_enabled_and_names_family_or_warns() {
        let off = banner(None, false, false, Some("llama"));
        assert!(
            !off.contains("HF parity:") && !off.contains("Corpus:"),
            "{off}"
        );
        let with = banner(None, false, true, Some("llama"));
        assert!(with.contains("HF parity: ENABLED"), "{with}");
        assert!(with.contains("    Corpus: corp/dir\n"), "{with}");
        assert!(with.contains("    Model family: llama\n"), "{with}");
        let without = banner(None, false, true, None);
        assert!(
            without.contains("Model family: NOT SET (required for parity tests)"),
            "{without}"
        );
    }

    #[test]
    fn write_error_is_propagated() {
        struct Broken;
        impl std::io::Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("closed"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        colored::control::set_override(false);
        let playbook = aprender_qa_runner::Playbook::from_yaml(YAML).expect("valid yaml");
        let r = write_run_status(
            &mut Broken,
            &playbook,
            1,
            None,
            false,
            1,
            false,
            false,
            "",
            None,
        );
        assert!(r.is_err());
    }
}
