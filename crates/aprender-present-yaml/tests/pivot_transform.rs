//! `pivot(row, col, value)` end to end through the public parser + executor
//! (#5631, covers `ExpressionExecutor::apply_pivot`).

use presentar_yaml::{DataContext, ExecutionError, ExpressionExecutor, ExpressionParser, Value};
use std::collections::HashMap;

fn obj(pairs: &[(&str, Value)]) -> Value {
    Value::Object(
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
    )
}

fn sale(region: &str, quarter: &str, amount: Option<f64>) -> Value {
    let mut pairs = vec![
        ("region", Value::String(region.into())),
        ("quarter", Value::String(quarter.into())),
    ];
    if let Some(a) = amount {
        pairs.push(("amount", Value::Number(a)));
    }
    obj(&pairs)
}

fn run(data: Value) -> Result<Value, ExecutionError> {
    let expr = ExpressionParser::new()
        .parse("{{ data | pivot(region, quarter, amount) }}")
        .expect("parse");
    let mut ctx = DataContext::new();
    ctx.insert("data", data);
    ExpressionExecutor::new().execute(&expr, &ctx)
}

fn rows_of(out: &Value) -> &Vec<Value> {
    out.as_array().expect("array result")
}

fn row<'a>(out: &'a Value, region: &str) -> &'a HashMap<String, Value> {
    rows_of(out)
        .iter()
        .filter_map(Value::as_object)
        .find(|o| matches!(o.get("region"), Some(Value::String(s)) if s == region))
        .unwrap_or_else(|| panic!("no pivot row for {region}"))
}

#[test]
fn pivot_sums_duplicates_and_spreads_columns() {
    let out = run(Value::Array(vec![
        sale("east", "q1", Some(10.0)),
        sale("east", "q1", Some(5.0)),
        sale("east", "q2", Some(1.0)),
        sale("west", "q1", Some(7.0)),
    ]))
    .expect("pivot");
    assert_eq!(rows_of(&out).len(), 2);
    let east = row(&out, "east");
    assert_eq!(east.get("q1").and_then(Value::as_number), Some(15.0));
    assert_eq!(east.get("q2").and_then(Value::as_number), Some(1.0));
    let west = row(&out, "west");
    assert_eq!(west.get("q1").and_then(Value::as_number), Some(7.0));
    assert!(west.get("q2").is_none());
}

#[test]
fn pivot_missing_value_counts_as_zero() {
    let out = run(Value::Array(vec![
        sale("east", "q1", None),
        sale("east", "q1", Some(2.0)),
    ]))
    .expect("pivot");
    assert_eq!(
        row(&out, "east").get("q1").and_then(Value::as_number),
        Some(2.0)
    );
}

#[test]
fn pivot_skips_non_objects_and_missing_keys_become_empty() {
    let out = run(Value::Array(vec![
        Value::Number(3.0),
        obj(&[("amount", Value::Number(4.0))]),
    ]))
    .expect("pivot");
    let rows = rows_of(&out);
    assert_eq!(rows.len(), 1);
    let only = rows[0].as_object().expect("object row");
    assert_eq!(only.get("").and_then(Value::as_number), Some(4.0));
}

#[test]
fn pivot_requires_array_and_accepts_empty() {
    let err = run(Value::Number(1.0)).expect_err("non-array input");
    assert!(matches!(err, ExecutionError::ExpectedArray));
    assert!(rows_of(&run(Value::Array(vec![])).expect("empty")).is_empty());
}
