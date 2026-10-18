//! Behavioural tests for `CbtopApp::render_main_content` (Refs #5541).

use super::*;
use crate::app::{DiskMetrics, MemoryBreakdown, NetworkMetrics};
use presentar_terminal::direct::CellBuffer;

fn hw(gpu: Option<&str>) -> HardwareInfo {
    HardwareInfo {
        cpu_model: "Test CPU".to_string(),
        cpu_cores: 8,
        simd_type: "AVX2",
        gpu_name: gpu.map(str::to_string),
        memory_gb: 32.0,
    }
}

/// Run `render_main_content` into a fresh buffer and return the screen, one `String` per row.
fn screen(width: u16, height: u16, running: bool, m: &LoadMetrics, gpu: Option<&str>) -> String {
    let mut buf = CellBuffer::new(width, height);
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buf);
        CbtopApp::render_main_content(
            &mut canvas,
            width,
            height,
            running,
            m,
            &[10.0, 50.0, 90.0],
            &[100.0, 200.0],
            42.5,
            1_000_000,
            &hw(gpu),
            &Theme::tokyo_night(),
        );
    }
    (0..height)
        .map(|y| {
            (0..width)
                .map(|x| buf.get(x, y).map_or(" ", |c| c.symbol.as_str()))
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn busy_metrics() -> LoadMetrics {
    LoadMetrics {
        bricks_per_second: 5000.0,
        total_bricks: 1_234_567,
        avg_latency_us: 12.5,
        per_core_usage: vec![10.0, 90.0],
        ops_per_second: 2.5e9,
        bytes_per_second: 1_073_741_824.0,
        memory: MemoryBreakdown {
            total_kb: 16_000_000,
            used_kb: 8_000_000,
            cached_kb: 2_000_000,
            buffers_kb: 100_000,
            available_kb: 8_000_000,
        },
        network: NetworkMetrics {
            rx_rate: 1024.0,
            tx_rate: 2048.0,
            ..Default::default()
        },
        disks: vec![DiskMetrics {
            mount: "/".to_string(),
            total_bytes: 1 << 40,
            used_bytes: 1 << 39,
            usage_percent: 50.0,
        }],
        ..Default::default()
    }
}

#[test]
fn running_and_stopped_status_are_distinct() {
    let m = LoadMetrics::default();
    assert!(screen(100, 40, true, &m, None).contains("● RUNNING"));
    let stopped = screen(100, 40, false, &m, None);
    assert!(stopped.contains("○ STOPPED"));
    assert!(!stopped.contains("● RUNNING"));
}

#[test]
fn metrics_box_prints_formatted_values() {
    let s = screen(100, 40, true, &busy_metrics(), None);
    assert!(s.contains("Real-Time Metrics"));
    assert!(s.contains("42.5%"));
    // the closing " │" at box_width overwrites the last digit of the 7-wide rate
    assert!(s.contains("500"));
    assert!(s.contains("1.23M"));
    assert!(s.contains("12.5 μs"));
    assert!(s.contains("elements"));
    assert!(s.contains("2.50 GFLOP/s"));
    assert!(s.contains("1.00 GB/s"));
}

#[test]
fn tall_screen_with_all_metrics_shows_every_panel() {
    let s = screen(100, 60, true, &busy_metrics(), Some("RTX 4090"));
    for title in [
        "Per-Core CPU",
        "Memory",
        "GPU",
        "Network",
        "Disks",
        "CPU History",
    ] {
        assert!(s.contains(title), "missing panel {title}");
    }
    assert!(!s.contains("┌─ Hardware"), "per-core replaces hardware box");
}

#[test]
fn without_per_core_data_hardware_box_is_shown() {
    let m = LoadMetrics::default();
    let s = screen(100, 40, true, &m, None);
    assert!(s.contains("┌─ Hardware"));
    assert!(!s.contains("Per-Core CPU"));
}

#[test]
fn optional_panels_are_omitted_when_data_absent() {
    let s = screen(100, 40, true, &LoadMetrics::default(), None);
    for title in ["┌─ Memory", "┌─ GPU", "┌─ Network", "┌─ Disks"] {
        assert!(!s.contains(title), "unexpected panel {title}");
    }
}

#[test]
fn short_screen_drops_per_core_and_controls_hint() {
    let s = screen(100, 10, true, &busy_metrics(), None);
    assert!(
        !s.contains("Per-Core CPU"),
        "height <= 20 uses hardware box"
    );
    assert!(
        !s.contains("[Space] Toggle load"),
        "height <= 10 hides hint"
    );
    let tall = screen(100, 40, true, &busy_metrics(), None);
    assert!(tall.contains("[Space] Toggle load"));
}

#[test]
fn narrow_terminal_does_not_panic_and_keeps_title() {
    let s = screen(30, 30, true, &busy_metrics(), Some("GPU"));
    assert!(s.contains("Real-Time Metrics"));
}
