//! `DeviceInfo::cuda_info` parses `nvidia-smi --query-gpu=name,memory.total,driver_version`
//! output and returns `None` for a failing tool, a short row or a missing binary (#5659).
//! One test body: it swaps the process-wide PATH, so cases must not run in parallel.
#![cfg(unix)]
#![allow(clippy::disallowed_methods)]

use entrenar::finetune::DeviceInfo;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

fn fake_smi(dir: &Path, body: &str) {
    let p = dir.join("nvidia-smi");
    std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).expect("write fake");
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).expect("chmod");
}

#[test]
fn cuda_info_follows_nvidia_smi_output() {
    let tmp = tempfile::tempdir().expect("tmp");
    std::env::set_var("PATH", tmp.path());

    // No nvidia-smi on PATH at all.
    assert!(DeviceInfo::cuda_info(0).is_none());

    // Well-formed row: name, MiB, driver.
    fake_smi(tmp.path(), "echo 'Fake RTX 4090, 24576, 550.54'");
    let info = DeviceInfo::cuda_info(0).expect("parsed");
    assert_eq!(info.name, "Fake RTX 4090");
    assert!((info.memory_gb - 24.0).abs() < 1e-9, "{}", info.memory_gb);
    assert_eq!(info.driver_version.as_deref(), Some("550.54"));
    assert_eq!(info.compute_capability, None);

    // The device id is passed through as --id=N.
    fake_smi(tmp.path(), "case \"$*\" in *--id=3*) echo 'Gpu3, 2048, 1.0';; *) exit 1;; esac");
    assert_eq!(DeviceInfo::cuda_info(3).expect("id 3").name, "Gpu3");
    assert!(DeviceInfo::cuda_info(4).is_none());

    // Unparsable memory degrades to 0 GB, still Some.
    fake_smi(tmp.path(), "echo 'Odd, n/a, 9.9'");
    assert_eq!(DeviceInfo::cuda_info(0).expect("some").memory_gb, 0.0);

    // Too few fields -> None.
    fake_smi(tmp.path(), "echo 'only, two'");
    assert!(DeviceInfo::cuda_info(0).is_none());

    // Non-zero exit -> None even though stdout is well-formed.
    fake_smi(tmp.path(), "echo 'Bad, 1024, 1.0'; exit 9");
    assert!(DeviceInfo::cuda_info(0).is_none());
}
