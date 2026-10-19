//! `model_selection::randomized_search`: seeded sampling of `min(n_iter, len)` distinct
//! candidates, best-of-sampled selection, and error propagation (#5641).
//!
//! A constant-score estimator makes each candidate's CV mean exactly its parameter, so
//! every assertion is about WHICH candidates were sampled and how the best is picked.

use aprender::error::Result;
use aprender::model_selection::{randomized_search, KFold};
use aprender::primitives::{Matrix, Vector};
use aprender::traits::Estimator;
use std::cell::Cell;

#[derive(Clone)]
struct Const {
    score: f32,
    fail: bool,
}

impl Estimator for Const {
    fn fit(&mut self, _x: &Matrix<f32>, _y: &Vector<f32>) -> Result<()> {
        if self.fail {
            Err("boom".into())
        } else {
            Ok(())
        }
    }
    fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
        Vector::from_vec(vec![0.0; x.shape().0])
    }
    fn score(&self, _x: &Matrix<f32>, _y: &Vector<f32>) -> f32 {
        self.score
    }
}

fn data() -> (Matrix<f32>, Vector<f32>, KFold) {
    let x = Matrix::from_vec(12, 1, (0..12).map(|i| i as f32).collect()).expect("matrix");
    let y = Vector::from_vec(vec![0.0; 12]);
    (x, y, KFold::new(3))
}

fn grid() -> Vec<f32> {
    (1..=20).map(|i| i as f32).collect()
}

fn build(p: &f32) -> Const {
    Const {
        score: *p,
        fail: false,
    }
}

#[test]
fn empty_grid_is_an_error() {
    let (x, y, cv) = data();
    let err =
        randomized_search::<Const, f32, _>(&[], 3, 0, build, &x, &y, &cv).expect_err("empty grid");
    assert_eq!(err, "Grid cannot be empty");
}

#[test]
fn n_iter_beyond_the_grid_evaluates_every_candidate_once() {
    let (x, y, cv) = data();
    let g = grid();
    let r = randomized_search(&g, 99, 7, build, &x, &y, &cv).expect("search");
    let mut seen = r.mean_scores.clone();
    seen.sort_by(f32::total_cmp);
    assert_eq!(seen, g, "each grid entry sampled exactly once");
    assert_eq!(r.best_params, 20.0);
    assert_eq!(r.best_score, 20.0);
}

#[test]
fn fewer_iterations_sample_a_distinct_subset_and_pick_its_best() {
    let (x, y, cv) = data();
    let g = grid();
    let built = Cell::new(0usize);
    let r = randomized_search(
        &g,
        5,
        3,
        |p| {
            built.set(built.get() + 1);
            build(p)
        },
        &x,
        &y,
        &cv,
    )
    .expect("search");
    assert_eq!(built.get(), 5, "build called once per sampled candidate");
    assert_eq!(r.mean_scores.len(), 5);
    let mut uniq = r.mean_scores.clone();
    uniq.sort_by(f32::total_cmp);
    uniq.dedup();
    assert_eq!(uniq.len(), 5, "no candidate sampled twice");
    let best_of_sample = r.mean_scores.iter().copied().fold(f32::MIN, f32::max);
    assert_eq!(r.best_score, best_of_sample);
    assert_eq!(r.best_params, best_of_sample);
    // best_index indexes the SAMPLED list, not the grid.
    assert_eq!(r.mean_scores[r.best_index], r.best_score);
}

#[test]
fn same_seed_repeats_and_other_seeds_resample() {
    let (x, y, cv) = data();
    let g = grid();
    let run = |seed| {
        randomized_search(&g, 5, seed, build, &x, &y, &cv)
            .expect("search")
            .mean_scores
    };
    assert_eq!(run(11), run(11), "seeded search must be reproducible");
    assert!(
        (0..8).any(|s| run(s) != run(100 + s)),
        "different seeds never changed the sample"
    );
}

#[test]
fn a_failing_fit_surfaces_as_a_training_error() {
    let (x, y, cv) = data();
    let g = vec![1.0f32, 2.0, 3.0];
    let err = randomized_search(
        &g,
        3,
        0,
        |p| Const {
            score: *p,
            fail: true,
        },
        &x,
        &y,
        &cv,
    )
    .expect_err("fit fails");
    assert!(err.starts_with("Training failed:"), "{err}");
    assert!(err.contains("boom"), "{err}");
}
