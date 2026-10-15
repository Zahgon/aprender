//! `apr code eval` (#5286): the hermetic, receipted eval of `apr code`
//! (docs/specifications/apr-code-eval-v1.md, contracts/apr-code-eval-*-v1.yaml).
//!
//! The harness lives in `aprender-code-eval`; this module only wires it to the
//! world: the model runs in an `apr serve` child launched through batuta, the
//! same path `apr code` uses, and the tests run in a docker or podman container
//! per attempt. The receipt is written only when the whole run succeeds.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use aprender_code_eval::contamination::{canary_prefix, flag_overlap, Contamination, NgramIndex};
use aprender_code_eval::driver::{Agent, Budget, Config, Reply, Request};
use aprender_code_eval::edit::EditFormat;
use aprender_code_eval::gap;
use aprender_code_eval::harness::evaluate;
use aprender_code_eval::prompt::render;
use aprender_code_eval::receipt::{self, Apr, Backend, Model, Run, Sampling, Sandbox};
use aprender_code_eval::runner::{Sandboxed, TaskEnv};
use aprender_code_eval::sandbox::Cli;
use aprender_code_eval::task::{load, Task};
use batuta::agent::driver::apr_serve::{AprServeDriver, ServeBackend, ServeLaunchOptions};
use batuta::agent::driver::{CompletionRequest, LlmDriver, Message};
use sha2::{Digest, Sha256};

use crate::error::CliError;

/// Parsed `apr code eval` flags.
#[derive(Debug, Clone)]
pub(crate) struct EvalArgs {
    pub tasks: PathBuf,
    pub envs: PathBuf,
    pub model: PathBuf,
    pub receipt: PathBuf,
    pub suite: String,
    pub format: EditFormat,
    pub n: usize,
    pub k: Option<usize>,
    pub seed: u64,
    pub max_tokens: u32,
    pub no_gpu: bool,
    pub engine: PathBuf,
    pub work_dir: PathBuf,
    pub load_date: Option<String>,
    pub cutoff: Option<String>,
    pub train_corpus: Vec<PathBuf>,
    pub canary_probe: bool,
    pub gaps: Option<PathBuf>,
}

/// Greedy decoding: the run is seeded, so sampling must not add noise.
const TEMPERATURE: f32 = 0.0;

/// The model under test, one completion per attempt through `apr serve`.
struct ServeAgent {
    driver: AprServeDriver,
    rt: tokio::runtime::Runtime,
}

impl Agent for ServeAgent {
    fn complete(&mut self, req: &Request<'_>) -> Result<Reply, String> {
        let p = render(req).map_err(|e| e.to_string())?;
        let request = CompletionRequest {
            model: String::new(),
            messages: vec![Message::User(p.user)],
            tools: Vec::new(),
            max_tokens: u32::try_from(req.budget.max_tokens).unwrap_or(u32::MAX),
            temperature: TEMPERATURE,
            system: Some(p.system),
        };
        let out = self
            .rt
            .block_on(self.driver.complete(request))
            .map_err(|e| e.to_string())?;
        Ok(Reply {
            text: out.text,
            tokens_in: out.usage.input_tokens,
            tokens_out: out.usage.output_tokens,
            turns: 1,
        })
    }
}

impl ServeAgent {
    /// One bare completion, outside the suite: the canary probe.
    fn ask(&mut self, user: String) -> Result<String, String> {
        let request = CompletionRequest {
            model: String::new(),
            messages: vec![Message::User(user)],
            tools: Vec::new(),
            max_tokens: 64,
            temperature: TEMPERATURE,
            system: None,
        };
        self.rt
            .block_on(self.driver.complete(request))
            .map(|out| out.text)
            .map_err(|e| e.to_string())
    }
}

/// ACE-CON-002: the corpus digest and the tasks whose gold overlaps the corpus. The
/// digest is sha256 over one `<sha256>  <path>` line per file, in flag order.
pub(crate) fn corpus_overlap(
    paths: &[PathBuf],
    tasks: &[Task],
) -> Result<(String, Vec<String>), String> {
    let mut texts = Vec::with_capacity(paths.len());
    let mut digest = Sha256::new();
    for p in paths {
        let text =
            std::fs::read_to_string(p).map_err(|e| format!("train corpus {}: {e}", p.display()))?;
        let sha = format!("{:x}", Sha256::digest(text.as_bytes()));
        digest.update(format!("{sha}  {}\n", p.display()));
        texts.push(text);
    }
    let index = NgramIndex::build(texts.iter().map(String::as_str));
    Ok((
        format!("{:x}", digest.finalize()),
        flag_overlap(tasks, &index),
    ))
}

/// sha256 of a file, streamed.
pub(crate) fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// The quantization named in a GGUF file name (`...-Q4_K_M.gguf` → `Q4_K_M`),
/// else `unknown`. The sha256 identifies the model; this is a label.
pub(crate) fn quant_from_name(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy())
        .unwrap_or_default();
    stem.split(['-', '.'])
        .rev()
        .find(|part| {
            let b = part.as_bytes();
            matches!(b.first(), Some(b'Q' | b'q' | b'F' | b'f' | b'B' | b'b'))
                && b.get(1).is_some_and(u8::is_ascii_digit)
        })
        .map_or_else(|| "unknown".to_string(), str::to_uppercase)
}

fn read(path: &Path, what: &str) -> Result<String, CliError> {
    std::fs::read_to_string(path)
        .map_err(|e| CliError::Aprender(format!("code eval: {what} {}: {e}", path.display())))
}

/// The edit format named on the command line (`whole`, `search_replace`, `udiff`).
pub(crate) fn parse_format(name: &str) -> Result<EditFormat, CliError> {
    serde_json::from_value(serde_json::Value::String(name.to_string()))
        .map_err(|e| CliError::Aprender(format!("code eval: --format {name}: {e}")))
}

/// The environment table: task id → pinned image, command and report format.
pub(crate) fn load_envs(json: &str) -> Result<BTreeMap<String, TaskEnv>, String> {
    serde_json::from_str(json).map_err(|e| format!("environments: {e}"))
}

/// #5288: write the ranked failing-task list of `receipt` to `path` and return how many
/// tasks it lists. A receipt the gap engine refuses (ACE-GAP-004) writes nothing.
pub(crate) fn write_gaps(receipt: &receipt::Receipt, path: &Path) -> Result<usize, String> {
    let gaps = gap::of_receipt(receipt).map_err(|e| format!("gaps: {e}"))?;
    std::fs::write(path, gap::render(receipt, &gaps))
        .map_err(|e| format!("gaps: {}: {e}", path.display()))?;
    Ok(gaps.len())
}

/// Run the eval and write the receipt.
pub(crate) fn run(args: &EvalArgs) -> Result<(), CliError> {
    let fail = |m: String| CliError::Aprender(format!("code eval: {m}"));
    let today = chrono::Utc::now();
    let load_date = args
        .load_date
        .clone()
        .unwrap_or_else(|| today.format("%Y-%m-%d").to_string());
    let set = load(&read(&args.tasks, "tasks")?, &load_date).map_err(|e| fail(e.to_string()))?;
    let envs = load_envs(&read(&args.envs, "environments")?).map_err(fail)?;
    let mut contamination = Contamination::unchecked();
    if !args.train_corpus.is_empty() {
        let (sha, flagged) = corpus_overlap(&args.train_corpus, set.tasks()).map_err(fail)?;
        contamination = contamination.with_overlap(sha, flagged);
    }
    let engine = Cli {
        bin: args.engine.clone(),
    };
    let engine_version = engine.version().map_err(|e| fail(e.to_string()))?;
    std::fs::create_dir_all(&args.work_dir).map_err(|e| fail(e.to_string()))?;
    let tag = format!("ace{}", std::process::id());
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut runner = Sandboxed::new(engine, envs, args.work_dir.clone(), home, args.seed, &tag);
    let cfg = Config {
        format: args.format,
        budget: Budget {
            max_turns: 1,
            max_tokens: u64::from(args.max_tokens),
        },
        seed: args.seed,
        n: args.n,
    };
    let backend = if args.no_gpu {
        ServeBackend::Cpu
    } else {
        ServeBackend::Gpu
    };
    let model_sha = sha256_file(&args.model).map_err(|e| fail(e.to_string()))?;
    let run = Run {
        suite: args.suite.clone(),
        task_set_digest: String::new(),
        model: Model {
            path: args.model.display().to_string(),
            sha256: model_sha,
            quant: quant_from_name(&args.model),
            backend: if args.no_gpu {
                Backend::Cpu
            } else {
                Backend::Cuda
            },
        },
        apr: Apr {
            version: env!("CARGO_PKG_VERSION").to_string(),
            git_sha: env!("APR_GIT_SHA").to_string(),
        },
        harness_git_sha: env!("APR_GIT_SHA").to_string(),
        sandbox: Sandbox {
            engine_version,
            image_digests: BTreeMap::new(),
        },
        seed: args.seed,
        sampling: Sampling {
            temperature: f64::from(TEMPERATURE),
            top_p: 1.0,
            n: args.n,
            max_tokens: u64::from(args.max_tokens),
            max_turns: 1,
        },
        started_at: today.to_rfc3339(),
        wall_s: 0.0,
    };
    let opts = ServeLaunchOptions {
        backend,
        max_tokens: Some(args.max_tokens),
        think: Some(false),
    };
    let driver = AprServeDriver::launch_with(args.model.clone(), None, &opts)
        .map_err(|e| fail(format!("apr serve: {e}")))?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| fail(e.to_string()))?;
    let mut agent = ServeAgent { driver, rt };
    if args.canary_probe {
        let reply = agent
            .ask(format!(
                "Continue this text exactly, with nothing else: {}",
                canary_prefix()
            ))
            .map_err(|e| fail(format!("canary probe: {e}")))?;
        contamination = contamination.with_canary(&reply);
    }
    let mut receipt = evaluate(
        set,
        &mut runner,
        &mut agent,
        &cfg,
        run,
        args.k,
        args.cutoff.as_deref(),
    )
    .map_err(|e| fail(e.to_string()))?;
    receipt.contamination = contamination;
    std::fs::write(&args.receipt, receipt.to_bytes()).map_err(|e| fail(e.to_string()))?;
    if let Some(path) = &args.gaps {
        let failing = write_gaps(&receipt, path).map_err(fail)?;
        eprintln!("code eval: {failing} failing tasks → {}", path.display());
    }
    eprintln!(
        "code eval: {} → {} (schema {})",
        args.suite,
        args.receipt.display(),
        receipt::SCHEMA
    );
    Ok(())
}

#[cfg(test)]
#[path = "code_eval_run_tests.rs"]
mod tests;
