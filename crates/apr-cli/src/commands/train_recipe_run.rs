//! Drive `apr finetune`, `apr distill` and `apr merge` from a recipe (PMAT-5230,
//! contracts/apr-recipe-v1.yaml RCP-009..012).
//!
//! `--recipe FILE` loads and validates the recipe, applies each `--set key=value`
//! and validates again, refuses a recipe of another kind (RCP-009), maps the
//! recipe onto the command's own arguments (RCP-012) and runs the command. The
//! run writes `<output>.receipt.json` (RCP-011) whether it succeeds or fails.
//!
//! Every flag the recipe covers conflicts with `--recipe` in clap, so a value
//! can only change through `--set`, which edits the recipe itself. A flag left
//! at its default cannot be told apart from an omitted one, so "the flag wins"
//! would be a rule nothing can check. Each `--set` is listed on stderr and in
//! the receipt (RCP-010), and the receipt carries both hashes: the file as
//! written and the recipe that ran.

use super::train_recipe::{MergeStrategy, Method, Recipe, RecipeError, RecipeKind, Violation};
use crate::error::{CliError, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// Receipt format written next to the output.
pub const RECEIPT_VERSION: &str = "apr.recipe.receipt/v1";

/// One `--set key=value`, with the value it replaced.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Override {
    pub key: String,
    pub from: serde_json::Value,
    pub to: serde_json::Value,
}

/// RCP-012: `apr finetune` arguments taken from a finetune recipe.
#[derive(Debug, Clone, PartialEq)]
pub struct FinetuneArgs {
    pub model: String,
    pub method: &'static str,
    pub rank: Option<u32>,
    pub data: String,
    pub output: String,
    pub epochs: u32,
    pub learning_rate: f64,
    pub max_seq_len: usize,
}

/// RCP-012: `apr distill` arguments taken from a distill recipe. The student is
/// the recipe's `base`.
#[derive(Debug, Clone, PartialEq)]
pub struct DistillArgs {
    pub teacher: String,
    pub student: String,
    pub data: String,
    pub output: String,
    pub temperature: f64,
    pub alpha: f64,
    pub epochs: u32,
}

/// RCP-012: `apr merge` arguments taken from a merge recipe.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeArgs {
    pub models: Vec<String>,
    pub strategy: &'static str,
    pub weights: Option<Vec<f32>>,
    pub output: String,
}

/// The command a recipe maps to.
#[derive(Debug, Clone, PartialEq)]
pub enum Planned {
    Finetune(FinetuneArgs),
    Distill(DistillArgs),
    Merge(MergeArgs),
}

impl Planned {
    pub fn output(&self) -> &str {
        match self {
            Self::Finetune(a) => &a.output,
            Self::Distill(a) => &a.output,
            Self::Merge(a) => &a.output,
        }
    }

    /// Every input the run reads, with its role.
    pub fn inputs(&self) -> Vec<(String, String)> {
        match self {
            Self::Finetune(a) => vec![
                ("base".to_string(), a.model.clone()),
                ("data.train".to_string(), a.data.clone()),
            ],
            Self::Distill(a) => vec![
                ("teacher".to_string(), a.teacher.clone()),
                ("base".to_string(), a.student.clone()),
                ("data.train".to_string(), a.data.clone()),
            ],
            Self::Merge(a) => a
                .models
                .iter()
                .enumerate()
                .map(|(i, m)| (format!("merge.models[{i}]"), m.clone()))
                .collect(),
        }
    }
}

/// RCP-009: the recipe's kind must be the command's.
pub fn check_kind(recipe: &Recipe, expected: RecipeKind) -> std::result::Result<(), RecipeError> {
    if recipe.kind == expected {
        return Ok(());
    }
    Err(RecipeError::Invalid(vec![Violation {
        rule: "RCP-009",
        field: "kind".to_string(),
        message: format!(
            "this command runs kind {}, the recipe is kind {}",
            kind_str(expected),
            kind_str(recipe.kind)
        ),
    }]))
}

/// RCP-010: apply each `key=value` (dotted key, YAML value), then fill defaults
/// and validate the result again. Unknown keys fail as RCP-002.
pub fn apply_sets(
    recipe: &Recipe,
    sets: &[String],
) -> std::result::Result<(Recipe, Vec<Override>), RecipeError> {
    let mut value = serde_json::to_value(recipe).map_err(|e| RecipeError::Parse(e.to_string()))?;
    let mut overrides = Vec::with_capacity(sets.len());
    for set in sets {
        let (key, raw) = set
            .split_once('=')
            .ok_or_else(|| RecipeError::Parse(format!("--set {set:?}: expected key=value")))?;
        let key = key.trim();
        if key.is_empty() || key.split('.').any(str::is_empty) {
            return Err(RecipeError::Parse(format!("--set {set:?}: empty key")));
        }
        let to: serde_json::Value = serde_yaml::from_str(raw)
            .map_err(|e| RecipeError::Parse(format!("--set {key}: {e}")))?;
        let from = set_path(&mut value, key, to.clone());
        overrides.push(Override {
            key: key.to_string(),
            from,
            to,
        });
    }
    let updated = Recipe::from_value(value)?;
    Ok((updated, overrides))
}

/// Write `to` at the dotted `key`, creating objects on the way. Returns the old
/// value, or null when there was none.
fn set_path(root: &mut serde_json::Value, key: &str, to: serde_json::Value) -> serde_json::Value {
    let mut node = root;
    let mut parts = key.split('.').peekable();
    while let Some(part) = parts.next() {
        if !node.is_object() {
            *node = serde_json::Value::Object(serde_json::Map::new());
        }
        let map = node.as_object_mut().expect("node was just made an object");
        if parts.peek().is_none() {
            return map.insert(part.to_string(), to).unwrap_or_default();
        }
        node = map
            .entry(part.to_string())
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    }
    serde_json::Value::Null
}

/// RCP-012: the pure recipe → arguments mapping. The recipe has been validated,
/// so required sections are present; a missing one is still an error, not a panic.
pub fn plan(recipe: &Recipe) -> std::result::Result<Planned, RecipeError> {
    let output = recipe.output.path.clone();
    match recipe.kind {
        RecipeKind::Finetune => {
            let t = recipe.training.clone().unwrap_or_default();
            Ok(Planned::Finetune(FinetuneArgs {
                model: model_of(recipe.base.as_ref().map(|b| &b.model), "base")?,
                method: method_str(recipe.method.ok_or_else(|| missing("method"))?),
                rank: recipe
                    .lora
                    .as_ref()
                    .map(|l| to_u32(l.rank, "lora.rank"))
                    .transpose()?,
                data: train_data(recipe)?,
                output,
                epochs: to_u32(t.epochs, "training.epochs")?,
                learning_rate: t.learning_rate,
                max_seq_len: t.max_seq_len,
            }))
        }
        RecipeKind::Distill => {
            let t = recipe.training.clone().unwrap_or_default();
            let d = recipe.distillation.clone().unwrap_or_default();
            Ok(Planned::Distill(DistillArgs {
                teacher: model_of(recipe.teacher.as_ref().map(|m| &m.model), "teacher")?,
                student: model_of(recipe.base.as_ref().map(|b| &b.model), "base")?,
                data: train_data(recipe)?,
                output,
                temperature: d.temperature,
                alpha: d.alpha,
                epochs: to_u32(t.epochs, "training.epochs")?,
            }))
        }
        RecipeKind::Merge => {
            let m = recipe.merge.as_ref().ok_or_else(|| missing("merge"))?;
            // Validated weights are finite and >= 0; merge takes them as f32.
            #[allow(clippy::cast_possible_truncation)]
            let weights = m
                .weights
                .as_ref()
                .map(|w| w.iter().map(|x| *x as f32).collect());
            Ok(Planned::Merge(MergeArgs {
                models: m.models.clone(),
                strategy: strategy_str(m.strategy),
                weights,
                output,
            }))
        }
    }
}

/// Recipe fields the command has no flag for. They are hashed with the recipe
/// but do not reach the run, so the run says so instead of implying otherwise.
#[must_use]
pub fn not_passed(recipe: &Recipe) -> Vec<String> {
    let mut out: Vec<&str> = Vec::new();
    match recipe.kind {
        RecipeKind::Finetune => {
            if recipe.lora.is_some() {
                out.extend(["lora.alpha", "lora.dropout"]);
            }
            out.extend(["training.batch_size", "training.seed"]);
        }
        RecipeKind::Distill => out.extend([
            "training.batch_size",
            "training.learning_rate",
            "training.max_seq_len",
            "training.seed",
        ]),
        RecipeKind::Merge => {}
    }
    if recipe.data.as_ref().is_some_and(|d| d.eval.is_some()) {
        out.push("data.eval");
    }
    if recipe.eval.is_some() {
        out.push("eval");
    }
    out.into_iter().map(str::to_string).collect()
}

/// SHA-256 of one input. `None` when the input is not a local file (a hub id, a
/// directory): the receipt says so rather than inventing a digest.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InputDigest {
    pub role: String,
    pub path: String,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Success,
    Failed,
}

/// RCP-011: what ran, from which recipe, on which bytes, and how it ended.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunReceipt {
    pub receipt: String,
    pub recipe_name: String,
    pub kind: RecipeKind,
    /// Hash of the recipe file as loaded, before `--set`.
    pub recipe_file_hash: String,
    /// Hash of the recipe that ran, after `--set`.
    pub recipe_hash: String,
    pub apr_version: String,
    pub git_sha: String,
    /// `training.seed`; merge recipes carry no seed.
    pub seed: Option<u64>,
    pub inputs: Vec<InputDigest>,
    pub overrides: Vec<Override>,
    pub not_passed: Vec<String>,
    pub started_at: String,
    pub finished_at: String,
    pub status: RunStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// `<output>.receipt.json`.
#[must_use]
pub fn receipt_path(output: &str) -> PathBuf {
    PathBuf::from(format!("{output}.receipt.json"))
}

fn file_sha256(path: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    if !path.is_file() {
        return None;
    }
    let mut f = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut f, &mut hasher).ok()?;
    Some(format!("{:x}", hasher.finalize()))
}

#[must_use]
pub fn digest_inputs(planned: &Planned) -> Vec<InputDigest> {
    planned
        .inputs()
        .into_iter()
        .map(|(role, path)| InputDigest {
            sha256: file_sha256(Path::new(&path)),
            role,
            path,
        })
        .collect()
}

/// Load, override, kind-check and map a recipe. Everything before the run.
pub fn prepare(
    path: &Path,
    sets: &[String],
    expected: RecipeKind,
) -> std::result::Result<(Recipe, String, Vec<Override>, Planned), RecipeError> {
    let loaded = Recipe::load(path)?;
    let file_hash = loaded.hash();
    let (recipe, overrides) = if sets.is_empty() {
        (loaded, Vec::new())
    } else {
        apply_sets(&loaded, sets)?
    };
    check_kind(&recipe, expected)?;
    let planned = plan(&recipe)?;
    Ok((recipe, file_hash, overrides, planned))
}

/// Run `command` with the arguments the recipe maps to and write the receipt.
/// In plan mode nothing runs and no receipt is written.
pub(crate) fn run(
    path: &Path,
    sets: &[String],
    expected: RecipeKind,
    plan_only: bool,
    command: impl FnOnce(&Planned) -> Result<()>,
) -> Result<()> {
    let (recipe, file_hash, overrides, planned) =
        prepare(path, sets, expected).map_err(|e| CliError::ValidationFailed(e.to_string()))?;
    for o in &overrides {
        eprintln!("recipe --set {}: {} -> {}", o.key, o.from, o.to);
    }
    let skipped = not_passed(&recipe);
    if !skipped.is_empty() {
        eprintln!(
            "recipe fields with no `apr {}` flag (hashed, not passed): {}",
            kind_str(expected),
            skipped.join(", ")
        );
    }
    if plan_only {
        return command(&planned);
    }
    let inputs = digest_inputs(&planned);
    let started_at = now();
    let result = command(&planned);
    let receipt = RunReceipt {
        receipt: RECEIPT_VERSION.to_string(),
        recipe_name: recipe.name.clone(),
        kind: recipe.kind,
        recipe_file_hash: file_hash,
        recipe_hash: recipe.hash(),
        apr_version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: env!("APR_GIT_SHA").to_string(),
        seed: recipe.training.as_ref().map(|t| t.seed),
        inputs,
        overrides,
        not_passed: skipped,
        started_at,
        finished_at: now(),
        status: if result.is_ok() {
            RunStatus::Success
        } else {
            RunStatus::Failed
        },
        error: result.as_ref().err().map(ToString::to_string),
    };
    let written = write_receipt(&receipt_path(planned.output()), &receipt);
    match (result, written) {
        (Err(e), Err(w)) => {
            eprintln!("recipe receipt not written: {w}");
            Err(e)
        }
        (Err(e), Ok(())) | (Ok(()), Err(e)) => Err(e),
        (Ok(()), Ok(())) => Ok(()),
    }
}

pub fn write_receipt(path: &Path, receipt: &RunReceipt) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let text = serde_json::to_string_pretty(receipt)
        .map_err(|e| CliError::ValidationFailed(format!("recipe receipt: {e}")))?;
    std::fs::write(path, text + "\n")?;
    Ok(())
}

/// The command got a plan for another kind. `prepare` checks the kind first
/// (RCP-009), so reaching this is a dispatch bug; it is an error, not a panic.
pub(crate) fn wrong_plan(planned: &Planned) -> CliError {
    let kind = match planned {
        Planned::Finetune(_) => "finetune",
        Planned::Distill(_) => "distill",
        Planned::Merge(_) => "merge",
    };
    CliError::ValidationFailed(format!("RCP-009 internal: command got a {kind} plan"))
}

/// RCP-010: `--set` without `--recipe` is refused. clap's `requires` is not
/// enough on its own: it is waived whenever an argument that conflicts with
/// `--recipe` is present, which is every run without a recipe.
pub(crate) fn sets_need_recipe(sets: &[String]) -> Result<()> {
    if sets.is_empty() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(format!(
            "RCP-010: --set {} needs --recipe",
            sets.join(" --set ")
        )))
    }
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn missing(field: &str) -> RecipeError {
    RecipeError::Invalid(vec![Violation {
        rule: "RCP-003",
        field: field.to_string(),
        message: "required for this kind".to_string(),
    }])
}

fn model_of(model: Option<&String>, field: &str) -> std::result::Result<String, RecipeError> {
    model.cloned().ok_or_else(|| missing(field))
}

fn train_data(recipe: &Recipe) -> std::result::Result<String, RecipeError> {
    recipe
        .data
        .as_ref()
        .map(|d| d.train.clone())
        .ok_or_else(|| missing("data"))
}

fn to_u32(n: usize, field: &str) -> std::result::Result<u32, RecipeError> {
    u32::try_from(n).map_err(|_| {
        RecipeError::Invalid(vec![Violation {
            rule: "RCP-012",
            field: field.to_string(),
            message: format!("{n} does not fit the command's u32 flag"),
        }])
    })
}

fn kind_str(kind: RecipeKind) -> &'static str {
    match kind {
        RecipeKind::Finetune => "finetune",
        RecipeKind::Distill => "distill",
        RecipeKind::Merge => "merge",
    }
}

fn method_str(method: Method) -> &'static str {
    match method {
        Method::Full => "full",
        Method::Lora => "lora",
        Method::Qlora => "qlora",
    }
}

fn strategy_str(strategy: MergeStrategy) -> &'static str {
    match strategy {
        MergeStrategy::Average => "average",
        MergeStrategy::Weighted => "weighted",
        MergeStrategy::Slerp => "slerp",
        MergeStrategy::Ties => "ties",
        MergeStrategy::Dare => "dare",
    }
}

#[cfg(test)]
#[path = "train_recipe_run_tests.rs"]
mod tests;
