//! `moving_avg(field, window)` appends a trailing-window mean as `<field>_ma<window>` (#5643).

use std::collections::HashMap;

use presentar_yaml::{DataContext, ExecutionError, ExpressionExecutor, ExpressionParser, Value};

fn row(price: f64) -> Value {
    Value::object(HashMap::from([("price".to_string(), Value::number(price))]))
}

fn run(expr: &str, data: Value) -> Result<Value, ExecutionError> {
    let mut ctx = DataContext::new();
    ctx.insert("data", data);
    let parsed = ExpressionParser::new().parse(expr).expect("parse");
    ExpressionExecutor::new().execute(&parsed, &ctx)
}

fn column(out: &Value, key: &str) -> Vec<f64> {
    match out {
        Value::Array(rows) => rows
            .iter()
            .map(|r| match r.get(key) {
                Some(Value::Number(n)) => *n,
                other => panic!("{key} missing or not a number: {other:?}"),
            })
            .collect(),
        other => panic!("expected array, got {other:?}"),
    }
}

#[test]
fn trailing_window_mean_ramps_up_then_slides() {
    let data = Value::array(vec![row(1.0), row(2.0), row(3.0), row(4.0)]);
    let out = run("{{ data | moving_avg(price, 2) }}", data).expect("run");
    assert_eq!(column(&out, "price_ma2"), vec![1.0, 1.5, 2.5, 3.5]);
}

#[test]
fn window_larger_than_the_data_is_a_running_mean() {
    let data = Value::array(vec![row(2.0), row(4.0), row(6.0)]);
    let out = run("{{ data | ma(price, 10) }}", data).expect("run");
    assert_eq!(column(&out, "price_ma10"), vec![2.0, 3.0, 4.0]);
}

#[test]
fn window_of_one_is_the_value_itself_and_originals_are_kept() {
    let data = Value::array(vec![row(5.0), row(7.0)]);
    let out = run("{{ data | moving_avg(price, 1) }}", data).expect("run");
    assert_eq!(column(&out, "price_ma1"), vec![5.0, 7.0]);
    assert_eq!(column(&out, "price"), vec![5.0, 7.0]);
}

#[test]
fn empty_input_stays_empty() {
    let out = run("{{ data | moving_avg(price, 3) }}", Value::array(vec![])).expect("run");
    assert_eq!(out, Value::array(vec![]));
}

#[test]
fn non_array_input_is_an_execution_error() {
    assert!(run("{{ data | moving_avg(price, 3) }}", Value::number(1.0)).is_err());
}
