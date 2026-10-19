//! #5648: `render_frame` writes the title bar, data lines and status bar.

use std::path::Path;
use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use super::render_frame;
use crate::tui::{DatasetAdapter, DatasetViewer};

fn batches(rows: i32) -> (Vec<RecordBatch>, Arc<Schema>) {
    let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int32, false)]));
    let col = Int32Array::from((0..rows).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(col)]).expect("batch");
    (vec![batch], schema)
}

fn viewer(adapter: DatasetAdapter, height: u16) -> DatasetViewer {
    DatasetViewer::with_dimensions(adapter, 40, height)
}

fn frame(v: &DatasetViewer, path: &str) -> String {
    let mut out: Vec<u8> = Vec::new();
    render_frame(v, &mut out, Path::new(path)).expect("render ok");
    String::from_utf8(out).expect("utf8")
}

#[test]
fn in_memory_title_names_file_row_count_and_mode() {
    let (b, s) = batches(5);
    let v = viewer(
        DatasetAdapter::in_memory_from_batches(b, s).expect("adapter"),
        24,
    );
    let out = frame(&v, "/some/dir/data.parquet");
    assert!(out.contains(" data.parquet | 5 rows | InMemory"), "{out:?}");
    assert!(!out.contains("/some/dir"), "only the file name is shown");
    assert!(!out.contains("Streaming"));
}

#[test]
fn streaming_adapter_is_labelled_streaming() {
    let (b, s) = batches(5);
    let v = viewer(
        DatasetAdapter::streaming_from_batches(b, s).expect("adapter"),
        24,
    );
    let out = frame(&v, "x.parquet");
    assert!(out.contains(" x.parquet | 5 rows | Streaming"), "{out:?}");
    assert!(!out.contains("InMemory"));
}

#[test]
fn status_bar_reports_row_range_and_key_help() {
    let (b, s) = batches(5);
    let v = viewer(
        DatasetAdapter::in_memory_from_batches(b, s).expect("adapter"),
        24,
    );
    let out = frame(&v, "d.parquet");
    assert!(
        out.contains(" Row 1-5 of 5 | \u{2191}\u{2193} scroll | PgUp/PgDn page"),
        "{out:?}"
    );
    assert!(out.contains("/search | q quit "));
}

#[test]
fn status_range_end_is_clamped_to_row_count_and_follows_scroll() {
    let (b, s) = batches(50);
    // height 6 => 5 visible rows
    let mut v = viewer(
        DatasetAdapter::in_memory_from_batches(b, s).expect("adapter"),
        6,
    );
    assert!(frame(&v, "d").contains(" Row 1-5 of 50 "));
    v.set_scroll_offset(10);
    assert!(frame(&v, "d").contains(" Row 11-15 of 50 "));
    v.end();
    let out = frame(&v, "d");
    let offset = v.scroll_offset();
    assert!(
        out.contains(&format!(" Row {}-50 of 50 ", offset + 1)),
        "{out:?}"
    );
}

#[test]
fn every_data_line_is_emitted_between_title_and_status() {
    let (b, s) = batches(3);
    let v = viewer(
        DatasetAdapter::in_memory_from_batches(b, s).expect("adapter"),
        24,
    );
    let out = frame(&v, "d");
    let title = out.find(" d | 3 rows").expect("title");
    let status = out.find(" Row 1-3 of 3").expect("status");
    let mut at = title;
    for line in v.render_lines() {
        let pos = out[at..].find(line.as_str()).expect("line in order") + at;
        assert!(pos < status, "{line:?} must precede the status bar");
        at = pos;
    }
    assert!(
        out.starts_with("\u{1b}[2J"),
        "frame begins by clearing the screen"
    );
}
