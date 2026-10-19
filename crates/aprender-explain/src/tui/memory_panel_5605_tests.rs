//! Rendering tests for `render_memory_panel` (#5605).

use super::*;
use crate::analyzer::{AnalysisReport, MemoryPattern, RegisterUsage, RooflineMetric};
use presentar_terminal::direct::CellBuffer;

fn app_with(loads: u64, stores: u64, ratio: f32) -> TuiApp {
    let report = AnalysisReport {
        name: "k".to_string(),
        target: "PTX".to_string(),
        registers: RegisterUsage::default(),
        memory: MemoryPattern {
            global_loads: loads as _,
            global_stores: stores as _,
            coalesced_ratio: ratio as _,
            ..Default::default()
        },
        roofline: RooflineMetric {
            arithmetic_intensity: 1.0,
            theoretical_peak_gflops: 1.0,
            memory_bound: false,
        },
        warnings: vec![],
        instruction_count: 1,
        estimated_occupancy: 1.0,
    };
    TuiApp::new("// ptx".to_string(), report)
}

fn render(app: &TuiApp) -> CellBuffer {
    let mut buf = CellBuffer::new(40, 12);
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buf);
        render_memory_panel(&mut canvas, app, 0.0, 0.0, 30, 8);
    }
    buf
}

fn row(buf: &CellBuffer, y: u16) -> String {
    (0..buf.width())
        .filter_map(|x| buf.get(x, y).map(|c| c.symbol.to_string()))
        .collect()
}

fn border_color(buf: &CellBuffer) -> Color {
    buf.get(0, 0).expect("cell").fg
}

#[test]
fn draws_title_and_counters() {
    let buf = render(&app_with(123, 45, 0.95));
    assert!(row(&buf, 0).contains("Memory"));
    assert!(row(&buf, 1).contains("Global ld: 123"));
    assert!(row(&buf, 2).contains("Global st: 45"));
    assert!(row(&buf, 3).contains("Coalesced: 95.0%"));
}

#[test]
fn bottom_border_closes_panel_after_blank_line() {
    let buf = render(&app_with(1, 1, 0.5));
    assert!(row(&buf, 4).trim_matches(['│', ' ']).is_empty());
    assert!(row(&buf, 5).contains('└'));
    assert!(row(&buf, 6).trim().is_empty());
}

#[test]
fn green_at_ninety_percent() {
    let buf = render(&app_with(1, 1, 0.9));
    assert_eq!(border_color(&buf), COLOR_GREEN);
}

#[test]
fn yellow_between_seventy_and_ninety() {
    let buf = render(&app_with(1, 1, 0.75));
    assert_eq!(border_color(&buf), COLOR_YELLOW);
}

#[test]
fn red_below_seventy() {
    let buf = render(&app_with(1, 1, 0.3));
    assert_eq!(border_color(&buf), COLOR_RED);
}

#[test]
fn coalesced_percent_has_one_decimal() {
    let buf = render(&app_with(0, 0, 0.8125));
    assert!(row(&buf, 3).contains("Coalesced: 81.2%") || row(&buf, 3).contains("Coalesced: 81.3%"));
}
