//! Which experts a qwen3moe run reached, of how many (E7 #4001, C358 1g).
//!
//! `contracts/e7-parity-arm-pins-v1.yaml` § `experts_reached`: the count of
//! distinct (layer, expert) pairs the router selected for any token, over
//! `layers × experts`. A line, not a bar.
//!
//! The loader names each layer by its router tensor's file offset
//! ([`note_layer`]); [`record`] runs on the CPU MoE FFN right after
//! `route_top_k`. Nothing is recorded unless `APR_MOE_EXPERTS_REACHED` names
//! a file. That file is rewritten each time a new pair is reached, so it holds
//! the final line when the run ends.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Distinct (layer, expert) pairs reached, per layer.
#[derive(Debug, Default)]
pub struct ExpertsReached {
    layer_of: HashMap<usize, usize>,
    num_layers: usize,
    num_experts: usize,
    seen: BTreeMap<usize, Vec<bool>>,
}

impl ExpertsReached {
    /// Name the layer whose router lives at `router_offset`.
    pub fn note_layer(&mut self, router_offset: usize, layer_idx: usize) {
        self.layer_of.insert(router_offset, layer_idx);
        self.num_layers = self.num_layers.max(layer_idx + 1);
    }

    /// Mark `experts` reached in the layer named by `router_offset`.
    /// Returns true when a pair was reached for the first time. A router the
    /// loader never named is not counted.
    pub fn record(&mut self, router_offset: usize, num_experts: usize, experts: &[usize]) -> bool {
        let Some(&layer) = self.layer_of.get(&router_offset) else {
            return false;
        };
        self.num_experts = self.num_experts.max(num_experts);
        let row = self
            .seen
            .entry(layer)
            .or_insert_with(|| vec![false; num_experts]);
        let mut grew = false;
        for &e in experts {
            if let Some(slot) = row.get_mut(e) {
                grew |= !*slot;
                *slot = true;
            }
        }
        grew
    }

    /// Distinct pairs reached.
    #[must_use]
    pub fn reached(&self) -> usize {
        self.seen
            .values()
            .map(|row| row.iter().filter(|&&s| s).count())
            .sum()
    }

    /// `layers × experts`; a named layer no token reached still counts.
    #[must_use]
    pub fn total(&self) -> usize {
        self.num_layers * self.num_experts
    }

    /// One summary line, then one line per layer.
    #[must_use]
    pub fn lines(&self) -> String {
        let mut out = format!(
            "experts reached: {} of {} ({} layers x {} experts)\n",
            self.reached(),
            self.total(),
            self.num_layers,
            self.num_experts
        );
        for layer in 0..self.num_layers {
            let n = self
                .seen
                .get(&layer)
                .map_or(0, |row| row.iter().filter(|&&s| s).count());
            out.push_str(&format!("layer {layer}: {n} of {}\n", self.num_experts));
        }
        out
    }
}

fn sink() -> Option<&'static PathBuf> {
    static SINK: OnceLock<Option<PathBuf>> = OnceLock::new();
    SINK.get_or_init(|| std::env::var_os("APR_MOE_EXPERTS_REACHED").map(PathBuf::from))
        .as_ref()
}

fn global() -> &'static Mutex<ExpertsReached> {
    static G: OnceLock<Mutex<ExpertsReached>> = OnceLock::new();
    G.get_or_init(|| Mutex::new(ExpertsReached::default()))
}

/// Loader hook: name the layer whose router is at `router_offset`.
pub fn note_layer(router_offset: usize, layer_idx: usize) {
    if sink().is_none() {
        return;
    }
    if let Ok(mut g) = global().lock() {
        g.note_layer(router_offset, layer_idx);
    }
}

/// FFN hook: record the routed experts; rewrite the sink file on growth.
pub fn record(router_offset: usize, num_experts: usize, experts: &[usize]) {
    let Some(path) = sink() else {
        return;
    };
    let Ok(mut g) = global().lock() else {
        return;
    };
    if g.record(router_offset, num_experts, experts) {
        if let Err(e) = std::fs::write(path, g.lines()) {
            eprintln!(
                "APR_MOE_EXPERTS_REACHED: cannot write {}: {e}",
                path.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExpertsReached;

    #[test]
    fn moe_experts_reached_counts_distinct_pairs_of_layers_times_experts() {
        let mut r = ExpertsReached::default();
        r.note_layer(1000, 0);
        r.note_layer(2000, 1);
        r.note_layer(3000, 2);
        assert!(r.record(1000, 8, &[0, 3]));
        assert!(!r.record(1000, 8, &[3, 0]), "a repeat is not new");
        assert!(r.record(2000, 8, &[3, 7]));
        assert_eq!(r.reached(), 4);
        assert_eq!(
            r.total(),
            24,
            "layer 2 was named but never reached: still in the denominator"
        );
        let lines = r.lines();
        assert!(lines.starts_with("experts reached: 4 of 24 (3 layers x 8 experts)\n"));
        assert!(lines.contains("layer 1: 2 of 8\n"));
        assert!(lines.contains("layer 2: 0 of 8\n"));
    }

    #[test]
    fn moe_experts_reached_ignores_an_unnamed_layer_and_out_of_range_ids() {
        let mut r = ExpertsReached::default();
        r.note_layer(1000, 0);
        assert!(!r.record(9999, 8, &[1]), "router the loader never named");
        assert!(!r.record(1000, 8, &[8, 99]), "ids past num_experts");
        assert_eq!(r.reached(), 0);
        assert_eq!(
            r.total(),
            8,
            "the named layer reported 8 experts; none reached"
        );
    }
}
