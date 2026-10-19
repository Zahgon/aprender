//! Behavioural tests for `CbtopApp::render_status_bar` (ticket #5611).

use presentar_terminal::direct::{CellBuffer, DirectTerminalCanvas};
use presentar_terminal::Theme;

use crate::app::hardware::LoadMetrics;
use crate::app::CbtopApp;
use crate::config::ComputeBackend;

const W: u16 = 100;
const H: u16 = 5;

fn metrics() -> LoadMetrics {
    LoadMetrics {
        bricks_per_second: 1234.4,
        avg_latency_us: 5.04,
        ops_per_second: 2_500_000_000.0,
        ..LoadMetrics::default()
    }
}

/// Render the status bar on a fresh buffer; return the text of the last row.
fn bar(running: bool, intensity: f64, show_fps: bool, frame_avg: f64) -> (String, CellBuffer) {
    let mut buffer = CellBuffer::new(W, H);
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        CbtopApp::render_status_bar(
            &mut canvas,
            W,
            H,
            running,
            intensity,
            ComputeBackend::Simd,
            show_fps,
            frame_avg,
            &metrics(),
            &Theme::tokyo_night(),
        );
    }
    let row = (0..W)
        .filter_map(|x| buffer.get(x, H - 1).map(|c| c.symbol.to_string()))
        .collect::<String>();
    (row, buffer)
}

#[test]
fn running_bar_shows_state_backend_intensity_and_metrics_in_order() {
    let (row, _) = bar(true, 0.5, false, 0.0);
    let parts = [
        " ●RUN ",
        "│ Simd ",
        "│ Int:50% ",
        "│ 1234 brick/s │ 5.0μs ",
        "│ 2.50 GFLOP/s ",
    ];
    let mut at = 0;
    for p in parts {
        let pos = row[at..]
            .find(p)
            .unwrap_or_else(|| panic!("{p:?} missing in {row:?}"));
        at += pos + p.len();
    }
}

#[test]
fn stopped_bar_shows_off_marker_instead_of_run() {
    let (row, _) = bar(false, 0.5, false, 0.0);
    assert!(row.starts_with(" ○OFF "), "{row:?}");
    assert!(!row.contains("RUN"));
}

#[test]
fn run_and_off_markers_are_drawn_in_different_colors() {
    let (_, on) = bar(true, 0.5, false, 0.0);
    let (_, off) = bar(false, 0.5, false, 0.0);
    assert_ne!(
        on.get(1, H - 1).map(|c| c.fg),
        off.get(1, H - 1).map(|c| c.fg)
    );
}

#[test]
fn intensity_is_rounded_to_whole_percent() {
    assert!(bar(true, 0.0, false, 0.0).0.contains("Int:0% "));
    assert!(bar(true, 1.0, false, 0.0).0.contains("Int:100% "));
    assert!(bar(true, 0.756, false, 0.0).0.contains("Int:76% "));
}

#[test]
fn fps_hidden_unless_flag_or_frame_time_present() {
    assert!(!bar(true, 0.5, false, 0.0).0.contains("FPS"));
}

#[test]
fn fps_from_frame_average_sits_ten_columns_from_the_right() {
    let (row, buf) = bar(true, 0.5, false, 16.0);
    assert!(row.contains("│ 62 FPS"), "{row:?}");
    assert_eq!(
        buf.get(W - 10, H - 1).map(|c| c.symbol.to_string()),
        Some("│".into())
    );
}

#[test]
fn show_fps_with_no_frame_time_reports_zero_fps() {
    assert!(bar(true, 0.5, true, 0.0).0.contains("│ 0 FPS"));
}
