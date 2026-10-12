//! Random forest scaling receipt: regressor vs classifier fit and predict wall time.
//!
//! Run it twice, once pinned to one core and once on all of them:
//!
//! ```bash
//! RAYON_NUM_THREADS=1 cargo run --release -p aprender-core --example rf_regressor_scaling
//! cargo run --release -p aprender-core --example rf_regressor_scaling
//! ```
//!
//! A forest whose fit or predict uses rayon gets faster with more threads; one that runs a
//! serial loop does not. Per-row predict (n one-row calls) against one batch call shows the
//! per-call overhead.

use aprender::primitives::{Matrix, Vector};
use aprender::tree::{RandomForestClassifier, RandomForestRegressor};
use std::time::Instant;

const N_ROWS: usize = 20_000;
const N_FEATURES: usize = 10;
const N_TREES: usize = 50;
const MAX_DEPTH: usize = 10;
const N_PREDICT: usize = 2_000;

/// Deterministic xorshift data, so every run fits the same forest.
fn dataset() -> (Vec<f32>, Vec<f32>) {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let x: Vec<f32> = (0..N_ROWS * N_FEATURES).map(|_| next()).collect();
    let y: Vec<f32> = (0..N_ROWS)
        .map(|r| {
            let row = &x[r * N_FEATURES..(r + 1) * N_FEATURES];
            3.0 * row[0] + 2.0 * row[1] * row[2] - row[3] + 0.1 * next()
        })
        .collect();
    (x, y)
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string());
    let cores = std::thread::available_parallelism().map_or(0, std::num::NonZero::get);
    println!(
        "rows={N_ROWS} features={N_FEATURES} trees={N_TREES} depth={MAX_DEPTH} \
         RAYON_NUM_THREADS={threads} available_parallelism={cores}"
    );

    let (xs, ys) = dataset();
    let x = Matrix::from_vec(N_ROWS, N_FEATURES, xs.clone()).expect("shape matches data");
    let y = Vector::from_vec(ys.clone());
    let x_pred = Matrix::from_vec(N_PREDICT, N_FEATURES, xs[..N_PREDICT * N_FEATURES].to_vec())
        .expect("shape matches data");
    let rows: Vec<Matrix<f32>> = (0..N_PREDICT)
        .map(|r| {
            Matrix::from_vec(
                1,
                N_FEATURES,
                xs[r * N_FEATURES..(r + 1) * N_FEATURES].to_vec(),
            )
            .expect("one row")
        })
        .collect();

    // Regressor.
    let mut reg = RandomForestRegressor::new(N_TREES)
        .with_max_depth(MAX_DEPTH)
        .with_random_state(42);
    let t = Instant::now();
    reg.fit(&x, &y).expect("regressor fit");
    let reg_fit = ms(t);
    let t = Instant::now();
    let batch = reg.predict(&x_pred);
    let reg_batch = ms(t);
    let t = Instant::now();
    let mut per_row = Vec::with_capacity(N_PREDICT);
    for row in &rows {
        per_row.push(reg.predict(row).as_slice()[0]);
    }
    let reg_rows = ms(t);
    assert_eq!(
        batch.as_slice(),
        per_row.as_slice(),
        "per-row and batch must agree"
    );
    let t = Instant::now();
    let oob = reg.oob_prediction().map_or(0, |v| v.len());
    let reg_oob = ms(t);

    // Classifier, same features, label = target above its median.
    let mut sorted = ys.clone();
    sorted.sort_by(f32::total_cmp);
    let median = sorted[N_ROWS / 2];
    let labels: Vec<usize> = ys.iter().map(|&v| usize::from(v > median)).collect();
    let mut clf = RandomForestClassifier::new(N_TREES)
        .with_max_depth(MAX_DEPTH)
        .with_random_state(42);
    let t = Instant::now();
    clf.fit(&x, &labels).expect("classifier fit");
    let clf_fit = ms(t);
    let t = Instant::now();
    let _ = clf.predict(&x_pred);
    let clf_batch = ms(t);
    let t = Instant::now();
    for row in &rows {
        let _ = clf.predict(row);
    }
    let clf_rows = ms(t);

    println!("regressor  fit_ms={reg_fit:.1} predict_batch_ms={reg_batch:.2} predict_per_row_ms={reg_rows:.2} oob_ms={reg_oob:.1} oob_rows={oob}");
    println!("classifier fit_ms={clf_fit:.1} predict_batch_ms={clf_batch:.2} predict_per_row_ms={clf_rows:.2}");
}
