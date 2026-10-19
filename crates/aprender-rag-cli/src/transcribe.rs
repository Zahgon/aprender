//! Batch transcription of media files to `.srt` sidecars.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::discover::{
    build_exclude_set, classify_media_sidecar_status, discover_and_report_media,
};
use crate::BackendType;

/// Transcription manifest for resume support.
#[derive(Serialize, Deserialize, Default)]
pub(crate) struct TranscribeManifest {
    /// Files that have been successfully transcribed.
    pub completed: Vec<String>,
    /// Files that failed transcription.
    pub failed: Vec<String>,
}

impl TranscribeManifest {
    pub(crate) fn load(root: &Path) -> Self {
        let manifest_path = root.join(".transcribe-manifest.json");
        fs::read_to_string(manifest_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    #[allow(dead_code)]
    pub(crate) fn save(&self, root: &Path) -> Result<()> {
        let manifest_path = root.join(".transcribe-manifest.json");
        let json = serde_json::to_string_pretty(self)?;
        fs::write(manifest_path, json)?;
        Ok(())
    }
}

/// Filter media files by sidecar status and manifest, returning files to process.
pub(crate) fn filter_files_for_transcription(
    media_files: &[PathBuf],
    root: &Path,
    skip_existing: bool,
) -> Vec<PathBuf> {
    let (has_sidecar, needs_transcription) = classify_media_sidecar_status(media_files);
    let manifest = TranscribeManifest::load(root);
    let previously_completed = manifest.completed.len();

    let to_process: Vec<PathBuf> = if skip_existing {
        needs_transcription
            .into_iter()
            .filter(|f| {
                !manifest
                    .completed
                    .contains(&f.to_string_lossy().to_string())
            })
            .collect()
    } else {
        media_files
            .iter()
            .filter(|f| {
                !manifest
                    .completed
                    .contains(&f.to_string_lossy().to_string())
            })
            .cloned()
            .collect()
    };

    println!(
        "\nSidecar status: {} with .srt/.vtt, {} need transcription",
        has_sidecar.len(),
        to_process.len()
    );
    if previously_completed > 0 {
        println!(
            "  {} previously completed (from manifest)",
            previously_completed
        );
    }
    to_process
}

/// Run the transcription feature gate (or print a message if not available).
#[allow(clippy::unnecessary_wraps)]
fn run_transcription_or_report(
    to_process: &[PathBuf],
    jobs: usize,
    model: Option<&str>,
    backend: BackendType,
    root: &Path,
    prompt: Option<&str>,
    hotwords: &[String],
) -> Result<()> {
    // The `transcription` feature and its implementation went with whisper-apr,
    // which is a standalone project rather than part of this framework. The
    // cfg blocks that used to live here referenced `aprender_rag::
    // TranscriptionLoader`, which no longer exists, behind a feature no
    // manifest declares -- so the code could never compile even if selected,
    // while the cfg attribute itself tripped `unexpected_cfgs`
    // under clippy's -D warnings.
    let _ = (jobs, model, backend, root, prompt, hotwords);
    println!(
        "\nTranscription is not available in this build: it lives in the\n\
         standalone whisper.apr project, not in aprender.\n\n\
         {} files need transcription. Run with --dry-run to list them.",
        to_process.len()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_transcribe(
    path: &str,
    recursive: bool,
    skip_existing: bool,
    jobs: usize,
    model: Option<&str>,
    backend: BackendType,
    dry_run: bool,
    prompt: Option<&str>,
    hotwords_file: Option<&str>,
    exclude_patterns: &[String],
) -> Result<()> {
    let root = Path::new(path);
    if !root.exists() {
        anyhow::bail!("Path not found: {}", root.display());
    }

    let exclude = build_exclude_set(exclude_patterns)?;
    let start_time = std::time::Instant::now();
    let media_files = discover_and_report_media(root, recursive, &exclude)?;
    if media_files.is_empty() {
        return Ok(());
    }

    let to_process = filter_files_for_transcription(&media_files, root, skip_existing);
    if to_process.is_empty() {
        println!("All files already have sidecars. Nothing to do.");
        return Ok(());
    }

    if dry_run {
        println!("\nDry run — files that would be transcribed:");
        for file in &to_process {
            println!("  {}", file.display());
        }
        println!("\nTotal: {} files", to_process.len());
        return Ok(());
    }

    // Load hotwords from file if provided (one word per line)
    let hotwords: Vec<String> = hotwords_file
        .map(|path| {
            fs::read_to_string(path)
                .unwrap_or_default()
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect()
        })
        .unwrap_or_default();

    if !hotwords.is_empty() {
        println!("Loaded {} hotwords for vocabulary biasing", hotwords.len());
    }
    if let Some(p) = prompt {
        println!("Using prompt: {:?}", p);
    }

    run_transcription_or_report(&to_process, jobs, model, backend, root, prompt, &hotwords)?;

    let elapsed = start_time.elapsed();
    println!(
        "\nTotal time: {:.1}s ({:.1} files/sec)",
        elapsed.as_secs_f64(),
        media_files.len() as f64 / elapsed.as_secs_f64().max(0.001)
    );
    Ok(())
}

#[cfg(test)]
mod filter_tests {
    use super::*;

    /// Three media files in a tempdir: `a` has an .srt sidecar, `b` has a .vtt
    /// sidecar, `c` has none.
    fn fixture() -> (tempfile::TempDir, Vec<PathBuf>) {
        let dir = tempfile::tempdir().expect("tempdir");
        let media: Vec<PathBuf> = ["a.mp4", "b.mp4", "c.mp4"]
            .iter()
            .map(|n| dir.path().join(n))
            .collect();
        fs::write(dir.path().join("a.srt"), "x").expect("srt");
        fs::write(dir.path().join("b.vtt"), "x").expect("vtt");
        (dir, media)
    }

    fn mark_completed(root: &Path, done: &[&PathBuf]) {
        let m = TranscribeManifest {
            completed: done
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect(),
            failed: vec![],
        };
        m.save(root).expect("save manifest");
    }

    #[test]
    fn skip_existing_keeps_only_files_without_sidecar() {
        let (dir, media) = fixture();
        let got = filter_files_for_transcription(&media, dir.path(), true);
        assert_eq!(got, vec![media[2].clone()]);
    }

    #[test]
    fn not_skipping_existing_keeps_every_file() {
        let (dir, media) = fixture();
        let got = filter_files_for_transcription(&media, dir.path(), false);
        assert_eq!(got, media);
    }

    #[test]
    fn manifest_completed_files_are_dropped_in_both_modes() {
        let (dir, media) = fixture();
        mark_completed(dir.path(), &[&media[2], &media[0]]);
        let skip = filter_files_for_transcription(&media, dir.path(), true);
        assert!(
            skip.is_empty(),
            "c is completed, a/b have sidecars: {skip:?}"
        );
        let all = filter_files_for_transcription(&media, dir.path(), false);
        assert_eq!(all, vec![media[1].clone()]);
    }

    #[test]
    fn corrupt_manifest_is_treated_as_empty() {
        let (dir, media) = fixture();
        fs::write(dir.path().join(".transcribe-manifest.json"), "{not json").expect("write");
        let got = filter_files_for_transcription(&media, dir.path(), false);
        assert_eq!(got, media);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(filter_files_for_transcription(&[], dir.path(), true).is_empty());
        assert!(filter_files_for_transcription(&[], dir.path(), false).is_empty());
    }
}
