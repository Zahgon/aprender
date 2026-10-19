//! `NetworkPanel::paint` read back from the cell buffer (#5703).
//!
//! Lives here, not in `network_panel.rs`: the pre-commit complexity hook judges whole
//! files and refuses on `paint` itself (cyclomatic > 30), which a test cannot fix.

use presentar_core::{Rect, Widget};
use presentar_terminal::direct::{CellBuffer, DirectTerminalCanvas};
use presentar_terminal::{NetworkInterface, NetworkPanel};

fn row_text(buffer: &CellBuffer, y: u16) -> String {
    (0..100u16)
        .map(|x| {
            buffer
                .get(x, y)
                .map_or(" ", |c| c.symbol.as_str())
                .to_string()
        })
        .collect()
}

/// Paint one compact interface row into a 100x6 buffer and return row 1 as text.
fn compact_row(iface: NetworkInterface) -> String {
    let mut panel = NetworkPanel::new().compact();
    panel.add_interface(iface);
    panel.layout(Rect::new(0.0, 0.0, 100.0, 6.0));
    let mut buffer = CellBuffer::new(100, 6);
    let mut canvas = DirectTerminalCanvas::new(&mut buffer);
    panel.paint(&mut canvas);
    row_text(&buffer, 1)
}

fn rated(errors: f64, drops: f64) -> NetworkInterface {
    let mut iface = NetworkInterface::new("eth0");
    iface.update(1024.0, 512.0);
    iface.set_rates(errors, drops);
    iface
}

#[test]
fn test_paint_compact_error_drop_severity_indicators() {
    let critical = compact_row(rated(11.0, 0.0));
    assert!(critical.contains("●E:11/D:0"), "{critical}");
    let critical_drops = compact_row(rated(0.0, 12.0));
    assert!(critical_drops.contains("●E:0/D:12"), "{critical_drops}");
    let warning = compact_row(rated(2.0, 0.0));
    assert!(warning.contains("◐E:2/D:0"), "{warning}");
    let minor = compact_row(rated(0.5, 0.0));
    assert!(minor.contains("○E:"), "{minor}");
    let clean = compact_row(rated(0.0, 0.0));
    assert!(!clean.contains("E:"), "{clean}");
}

#[test]
fn test_paint_compact_utilization_tiers() {
    let with_util = |pct: f64| {
        let mut iface = rated(0.0, 0.0);
        iface.set_utilization(Some(pct));
        compact_row(iface)
    };
    assert!(with_util(95.0).contains("●95%"));
    assert!(with_util(150.0).contains("●100%"), "capped at 100");
    assert!(with_util(60.0).contains("◐60%"));
    let normal = with_util(20.0);
    assert!(normal.contains("20%") && !normal.contains('●') && !normal.contains('◐'));
    assert!(
        !compact_row(rated(0.0, 0.0)).contains('%'),
        "None shows no gauge"
    );
}

#[test]
fn test_paint_stops_at_bottom_of_bounds() {
    let mut panel = NetworkPanel::new().compact();
    for name in ["aaa0", "bbb0", "ccc0"] {
        let mut iface = NetworkInterface::new(name);
        iface.update(10.0, 10.0);
        panel.add_interface(iface);
    }
    // Height 3: header on row 0, two interface rows fit, the third must be cut.
    panel.layout(Rect::new(0.0, 0.0, 100.0, 3.0));
    let mut buffer = CellBuffer::new(100, 6);
    let mut canvas = DirectTerminalCanvas::new(&mut buffer);
    panel.paint(&mut canvas);
    assert!(row_text(&buffer, 1).starts_with("aaa0"));
    assert!(row_text(&buffer, 2).starts_with("bbb0"));
    assert!(
        !row_text(&buffer, 3).contains("ccc0"),
        "third interface is past the bounds"
    );
}
