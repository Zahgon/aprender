//! TRACE-001 TR-09b: `apr train` writes `apr-trace-v1`, the record `apr serve`
//! and `apr run` write, one per optimizer step.
//!
//! Each span is timed around the work it names, on the training thread, inside
//! the step's own wall clock: zeroing the gradients, the forward passes, the
//! backward passes and the optimizer (clip + update). They do not overlap, so
//! they never sum past the step's total, and nothing is split from a total: a
//! phase the step did not run is absent, not zero.

use renacer_core::apr_trace::{TraceData, TraceOperation, TraceProvenance};
use std::time::Instant;

/// The spans every record carries, in order: what a reader passes to
/// [`renacer_core::apr_trace::check_spans`].
pub const STEP_SPANS: [&str; 4] = ["zero_grad", "forward", "backward", "optimizer"];

/// The spans of the optimizer step in progress.
#[derive(Debug, Default)]
pub(super) struct StepSpans {
    started: Option<Instant>,
    zero_grad_us: u64,
    forward_us: u64,
    backward_us: u64,
}

/// Microseconds since `t`, saturating.
pub(super) fn us_since(t: Instant) -> u64 {
    u64::try_from(t.elapsed().as_micros()).unwrap_or(u64::MAX)
}

impl StepSpans {
    /// Opens the step: its wall clock starts here.
    pub(super) fn begin(&mut self) {
        *self = Self { started: Some(Instant::now()), ..Self::default() };
    }

    pub(super) fn add_zero_grad(&mut self, us: u64) {
        self.zero_grad_us = self.zero_grad_us.saturating_add(us);
    }

    pub(super) fn add_forward(&mut self, us: u64) {
        self.forward_us = self.forward_us.saturating_add(us);
    }

    pub(super) fn add_backward(&mut self, us: u64) {
        self.backward_us = self.backward_us.saturating_add(us);
    }

    /// Closes the step after its optimizer span; `None` if it never opened.
    pub(super) fn finish(&mut self, optimizer_us: u64) -> Option<TraceData> {
        let started = self.started.take()?;
        Some(step_record(
            us_since(started),
            self.zero_grad_us,
            self.forward_us,
            self.backward_us,
            optimizer_us,
        ))
    }
}

/// The `apr-trace-v1` record of one optimizer step, from its measured spans.
#[must_use]
pub fn step_record(
    total_us: u64,
    zero_grad_us: u64,
    forward_us: u64,
    backward_us: u64,
    optimizer_us: u64,
) -> TraceData {
    let breakdown: Vec<TraceOperation> = STEP_SPANS
        .into_iter()
        .zip([zero_grad_us, forward_us, backward_us, optimizer_us])
        .map(|(name, time_us)| TraceOperation { name: name.to_string(), time_us, details: None })
        .collect();
    TraceData {
        level: "step".to_string(),
        operations: breakdown.len(),
        total_time_us: total_us,
        breakdown,
        provenance: TraceProvenance::Measured,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{LMBatch, TransformerTrainConfig, TransformerTrainer};
    use super::*;
    use crate::transformer::TransformerConfig;
    use renacer_core::apr_trace::{check, check_spans, Violation};

    fn tiny_trainer(accumulation_steps: usize) -> TransformerTrainer {
        TransformerTrainer::new(
            TransformerTrainConfig::new(TransformerConfig::tiny())
                .with_accumulation_steps(accumulation_steps),
        )
    }

    fn batch() -> LMBatch {
        LMBatch::single(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 3, 4, 5, 6, 7, 8, 9])
    }

    /// TR-09b (a): a tiny-fixture training step writes a record that serve's
    /// validator accepts, with every span measured and inside the step.
    #[test]
    fn tiny_training_step_record_passes_serves_check() {
        let mut trainer = tiny_trainer(1);
        assert!(trainer.last_step_trace().is_none(), "no step ran yet");
        trainer.train_batch(&batch());
        let t = trainer.last_step_trace().expect("one optimizer step ran");
        assert_eq!(check_spans(t, &STEP_SPANS), Ok(()), "{t:?}");
        assert_eq!(t.provenance, TraceProvenance::Measured);
        assert!(t.total_time_us > 0, "a training step takes time: {t:?}");
    }

    /// One record per optimizer step, not per batch: with two batches
    /// accumulated, the first batch closes no step.
    #[test]
    fn accumulated_batches_make_one_record() {
        let mut trainer = tiny_trainer(2);
        trainer.train_batch(&batch());
        assert!(trainer.last_step_trace().is_none(), "half a step is not a record");
        trainer.train_batch(&batch());
        let t = trainer.last_step_trace().expect("the step closed");
        assert_eq!(check_spans(t, &STEP_SPANS), Ok(()), "{t:?}");
    }

    /// TR-09b (b): a mutated step record is rejected by the same validator, so
    /// the golden check can fail.
    #[test]
    fn falsify_obs_trace_005_mutated_step_records_are_rejected() {
        let honest = step_record(1_000, 10, 400, 500, 50);
        assert_eq!(check_spans(&honest, &STEP_SPANS), Ok(()));

        let mut t = honest.clone();
        t.breakdown[2].time_us = 600; // a breakdown that does not sum
        assert!(matches!(check(&t), Err(Violation::PartsExceedTotal { .. })));

        let mut t = honest.clone();
        t.breakdown.clear(); // Measured with no events
        assert_eq!(check(&t), Err(Violation::MeasuredWithoutOperations));

        let mut t = honest;
        t.breakdown.remove(1); // a dropped span: check alone cannot see it
        assert_eq!(check(&t), Ok(()));
        assert!(matches!(check_spans(&t, &STEP_SPANS), Err(Violation::SpansDiffer { .. })));
    }
}
