//! The Rust Laya trainer (APR-LAYA-TRAIN-001).
//!
//! This crate holds the trainer's data rules and, from LT-7, its fine-tune loop. It
//! reads the judge (`aprender-decide`) through its public API only and never changes
//! it: a row is hashed with the judge's own `nfc-trim-ws-v1` normalization, so the
//! trainer and the verifier cannot disagree on what "the same text" means.
//!
//! # Modules
//!
//! - [`eval_set`]: `eval_set_in_distribution`, the demo eval set as a pure function
//!   of the source splits, the shots and the benchmark's exclusions (LT-8).

pub mod eval_set;
