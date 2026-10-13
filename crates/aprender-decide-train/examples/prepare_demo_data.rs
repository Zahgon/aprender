//! Prepare the Laya demo data directory (APR-LAYA-TRAIN-001 LT-8).
//!
//! ```text
//! cargo run -p aprender-decide-train --example prepare_demo_data -- \
//!     --source <prepared dataset dir> --selection <selection-manifest.json> \
//!     --instructions <text> --expect-rows <n> --expect-class-counts <a,b,c> \
//!     --out <data dir outside the repo>
//! ```
//!
//! `--source` is the canonical prepared dataset a host receives through a
//! forjar-declared resource: `benchmark-manifest.json` (whose `dataset_attestation`
//! is the attestation) and `train.jsonl`, `validation.jsonl`, `test.jsonl`. It is
//! reopened through its attestation, so a changed byte is refused. The selection
//! manifest is replayed against it, so a selection no seed could produce is refused.
//! The expected counts come from the benchmark contract; this example holds none.
//!
//! Writes `task.json` (criteria in the dataset's label order), `train.jsonl` (the
//! shots, in selection order), `eval.jsonl` (`eval_set_in_distribution`) and
//! `shift.jsonl` (the test split). The output directory must be empty or absent, and
//! must lie outside the repository: dataset text never enters git.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::manifest::SelectionManifest;
use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
use aprender_contrastive_data::schema::LabeledExample;
use aprender_contrastive_data::select::Selection;
use aprender_decide_train::eval_set::{eval_set_in_distribution, Exclusions, SourceRow, Split};

type Res<T> = Result<T, Box<dyn Error>>;

struct Args {
    source: PathBuf,
    selection: PathBuf,
    instructions: String,
    expect_rows: usize,
    expect_class_counts: Vec<usize>,
    out: PathBuf,
}

fn main() -> Res<()> {
    let args = parse_args(std::env::args().skip(1).collect())?;
    refuse_out_dir(&args.out)?;
    let mut ledger = AccessLedger::new();
    let dataset = open_dataset(&args.source, &mut ledger)?;
    let manifest = SelectionManifest::from_bytes(&fs::read(&args.selection)?)?;
    let selection = Selection::replay(&manifest, &dataset, &mut ledger)?;

    let train: Vec<SourceRow> = dataset.train().rows().iter().map(source_row).collect();
    let validation: Vec<SourceRow> = dataset.validation().rows().iter().map(source_row).collect();
    let shots = shot_rows(&selection, &train)?;
    let eval = eval_set_in_distribution(
        &validation,
        &train,
        &shots,
        &exclusions(&dataset),
        args.expect_rows,
        &args.expect_class_counts,
    )?;

    let labels = dataset.label_names();
    let shift: Vec<SourceRow> = dataset.test().rows().iter().map(source_row).collect();
    fs::create_dir_all(&args.out)?;
    fs::write(
        args.out.join("task.json"),
        task_json(&args.instructions, labels),
    )?;
    fs::write(args.out.join("train.jsonl"), jsonl(&shots, labels)?)?;
    fs::write(args.out.join("eval.jsonl"), jsonl(&eval, labels)?)?;
    fs::write(args.out.join("shift.jsonl"), jsonl(&shift, labels)?)?;
    println!(
        "wrote {}: train {} eval {} shift {}",
        args.out.display(),
        shots.len(),
        eval.len(),
        shift.len()
    );
    Ok(())
}

fn parse_args(raw: Vec<String>) -> Res<Args> {
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    let mut it = raw.into_iter();
    while let Some(key) = it.next() {
        let name = key
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument {key:?}"))?;
        let value = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
        map.insert(name.to_string(), value);
    }
    let mut take = |k: &str| map.remove(k).ok_or_else(|| format!("--{k} is required"));
    let args = Args {
        source: take("source")?.into(),
        selection: take("selection")?.into(),
        instructions: take("instructions")?,
        expect_rows: take("expect-rows")?.parse()?,
        expect_class_counts: take("expect-class-counts")?
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?,
        out: take("out")?.into(),
    };
    if let Some(extra) = map.keys().next() {
        return Err(format!("unknown argument --{extra}").into());
    }
    Ok(args)
}

/// The output must be absent or empty, and outside this repository.
fn refuse_out_dir(out: &Path) -> Res<()> {
    if out.exists() && fs::read_dir(out)?.next().is_some() {
        return Err(format!("{} is not empty; refusing to write into it", out.display()).into());
    }
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let at = parent.canonicalize()?;
    if at.starts_with(&repo) {
        return Err(format!(
            "{} is inside the repository; data stays out of git",
            out.display()
        )
        .into());
    }
    Ok(())
}

/// Reopen the prepared dataset through its attestation (every split byte checked).
fn open_dataset(dir: &Path, ledger: &mut AccessLedger) -> Res<PreparedDataset<Canonical>> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("benchmark-manifest.json"))?)?;
    let attestation = manifest
        .get("dataset_attestation")
        .ok_or("benchmark-manifest.json has no dataset_attestation")?;
    let mut splits = BTreeMap::new();
    for role in ["train", "validation", "test"] {
        splits.insert(
            role.to_string(),
            fs::read(dir.join(format!("{role}.jsonl")))?,
        );
    }
    let bytes = serde_json::to_vec(attestation)?;
    Ok(PreparedDataset::<Canonical>::from_attested_bytes(
        &bytes, &splits, ledger,
    )?)
}

fn source_row(r: &LabeledExample) -> SourceRow {
    SourceRow {
        id: r.id.clone(),
        text: r.input.clone(),
        label: r.label,
    }
}

/// The replayed selection's shots, in selection order, resolved to train rows.
fn shot_rows(selection: &Selection, train: &[SourceRow]) -> Res<Vec<SourceRow>> {
    selection
        .ordered_ids()
        .into_iter()
        .map(|id| {
            train
                .iter()
                .find(|r| r.id == id)
                .cloned()
                .ok_or_else(|| format!("shot {id} is not a train row").into())
        })
        .collect()
}

fn exclusions(dataset: &PreparedDataset<Canonical>) -> Exclusions {
    let record = dataset.exclusions();
    let group_members = record
        .groups()
        .iter()
        .flat_map(|g| g.members.iter())
        .filter_map(|(role, id)| {
            let split = match role.as_str() {
                "train" => Split::Train,
                "validation" => Split::Validation,
                _ => return None,
            };
            Some((split, id.clone()))
        })
        .collect();
    Exclusions {
        excluded_train_ids: record.excluded_train_ids().to_vec(),
        group_members,
    }
}

/// `task.json` with criteria in label order. Written by hand because a serde map
/// would sort the keys, and the criteria's document order IS the label index.
fn task_json(instructions: &str, labels: &[String]) -> String {
    let criteria: Vec<String> = labels
        .iter()
        .map(|l| format!("{}:null", quote(l)))
        .collect();
    format!(
        "{{\"type\":\"choice\",\"instructions\":{},\"criteria\":{{{}}}}}\n",
        quote(instructions),
        criteria.join(",")
    )
}

/// One `{"text","label"}` row per line, the label as its criterion name.
fn jsonl(rows: &[SourceRow], labels: &[String]) -> Res<String> {
    let mut out = String::new();
    for r in rows {
        let label = labels
            .get(r.label)
            .ok_or_else(|| format!("row {} has label {} outside the criteria", r.id, r.label))?;
        out.push_str(&format!(
            "{{\"text\":{},\"label\":{}}}\n",
            quote(&r.text),
            quote(label)
        ));
    }
    Ok(out)
}

fn quote(s: &str) -> String {
    serde_json::Value::String(s.to_string()).to_string()
}
