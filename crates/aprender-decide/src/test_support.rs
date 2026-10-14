//! Test plumbing: the tiny Laya fixture (plan 08-02) in an in-memory `.apr`, its
//! oracle, and contract values read from YAML at test time (never Rust literals).

use crate::laya::Laya;
use crate::Task;
use aprender::format::v2::{AprV2Metadata, AprV2ReaderRef, AprV2Writer, TensorDType};
use base64::Engine as _;
use std::path::{Path, PathBuf};

/// `(name, dtype, shape, raw little-endian bytes)` as stored in the checkpoint.
pub(crate) type RawTensor = (String, TensorDType, Vec<usize>, Vec<u8>);

pub(crate) fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya_tiny")
}

pub(crate) fn read(rel: &str) -> Vec<u8> {
    std::fs::read(fixture_dir().join(rel)).unwrap_or_else(|e| panic!("read laya_tiny/{rel}: {e}"))
}

/// Every checkpoint tensor with its raw bytes: F16 stays F16, the F32 `temperature`
/// buffer stays F32.
pub(crate) fn checkpoint_tensors() -> Vec<RawTensor> {
    let bytes = read("checkpoint/model.safetensors");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let mut out: Vec<RawTensor> = st
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let dtype = match view.dtype() {
                safetensors::Dtype::F16 => TensorDType::F16,
                safetensors::Dtype::F32 => TensorDType::F32,
                other => panic!("{name}: unexpected fixture dtype {other:?}"),
            };
            (name, dtype, view.shape().to_vec(), view.data().to_vec())
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Write `tensors` into an in-memory `.apr` with `AprV2Writer::add_tensor`.
pub(crate) fn write_apr(tensors: &[RawTensor]) -> Vec<u8> {
    let mut w = AprV2Writer::new(AprV2Metadata::default());
    for (name, dtype, shape, raw) in tensors {
        w.add_tensor(name.clone(), *dtype, shape.clone(), raw.clone());
    }
    w.write().expect("write in-memory .apr")
}

/// The fixture checkpoint as an in-memory `.apr`.
pub(crate) fn fixture_apr() -> Vec<u8> {
    write_apr(&checkpoint_tensors())
}

/// Laya over `apr` for `task`, with the fixture's configs and tokenizer.
pub(crate) fn load_laya(apr: &[u8], task: Task) -> Result<Laya, crate::LayaError> {
    let reader = AprV2ReaderRef::from_bytes(apr).expect("open in-memory .apr");
    Laya::from_parts(
        &reader,
        "",
        &read("checkpoint/encoder/config.json"),
        &read("checkpoint/rl_agent_config.json"),
        &read("checkpoint/tokenizer/tokenizer.json"),
        task,
    )
}

/// The fixture's own `data/task.json`.
pub(crate) fn fixture_task() -> Task {
    Task::from_slice(&read("data/task.json")).expect("fixture task parses")
}

pub(crate) fn oracle() -> serde_json::Value {
    serde_json::from_slice(&read("oracle.json")).expect("parse oracle.json")
}

/// A decimal list of exact f32 values (carried as their f64 repr).
pub(crate) fn f32_list(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("f32 list")
        .iter()
        .map(|x| x.as_f64().expect("f32 value") as f32)
        .collect()
}

pub(crate) fn u32_list(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .expect("id list")
        .iter()
        .map(|x| u32::try_from(x.as_u64().expect("id")).expect("id fits u32"))
        .collect()
}

pub(crate) fn usize_list(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .expect("position list")
        .iter()
        .map(|x| usize::try_from(x.as_u64().expect("position")).expect("fits usize"))
        .collect()
}

pub(crate) fn string_list(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .expect("string list")
        .iter()
        .map(|x| x.as_str().expect("string").to_string())
        .collect()
}

/// An `f32le_base64` block (standard padded base64 of little-endian f32 bytes).
pub(crate) fn f32_b64(v: &serde_json::Value) -> Vec<f32> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(v.as_str().expect("base64 block"))
        .expect("decode base64 block");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub(crate) fn contract_yaml(name: &str) -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../contracts/{name}"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));
    serde_saphyr::from_str_with_options(&text, serde_saphyr::options! { strict_booleans: true, merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary, emit_comments: false, reject_non_finite_typeless_float: false }).unwrap_or_else(|e| panic!("parse {name}: {e}"))
}

/// `equations.<equation>.float_tolerance` from `contracts/laya-parity-v1.yaml`.
pub(crate) fn tolerance(equation: &str) -> f64 {
    contract_yaml("laya-parity-v1.yaml")["equations"][equation]["float_tolerance"]
        .as_f64()
        .unwrap_or_else(|| panic!("laya-parity-v1 equations.{equation}.float_tolerance"))
}

/// `constants.<key>` of `contracts/<contract>` as f64.
pub(crate) fn constant_f64(contract: &str, key: &str) -> f64 {
    contract_yaml(contract)["constants"][key]
        .as_f64()
        .unwrap_or_else(|| panic!("{contract} constants.{key}"))
}

/// NaN-visible `delta <= bound`: the crate's one definition, in `artifact`.
pub(crate) use crate::artifact::within;

/// `max |a - b|` in f64; NaN-propagating, and NaN on a length mismatch: the verifier's own.
pub(crate) use crate::verify::row_max_abs as max_abs;

/// The keys of the mapping at `path` in contract `name`, in DOCUMENT order.
/// `contract_yaml` reads into `serde_json::Value`, whose map is sorted (decide never
/// turns `preserve_order` on), so a test that pins key order reads through this.
pub(crate) fn contract_keys_in_order(name: &str, path: &[&str]) -> Vec<String> {
    let path_buf = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../contracts/{name}"));
    let text = std::fs::read_to_string(&path_buf).unwrap_or_else(|e| panic!("read {name}: {e}"));
    let mut node: OrderedNode = serde_saphyr::from_str_with_options(&text, serde_saphyr::options! { strict_booleans: true, merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary, emit_comments: false, reject_non_finite_typeless_float: false }).unwrap_or_else(|e| panic!("parse {name}: {e}"));
    for key in path {
        let OrderedNode::Map(entries) = node else {
            panic!("{name}: {key} is not under a mapping");
        };
        node = entries
            .into_iter()
            .find_map(|(k, v)| (k == *key).then_some(v))
            .unwrap_or_else(|| panic!("{name}: no key {key}"));
    }
    match node {
        OrderedNode::Map(entries) => entries.into_iter().map(|(k, _)| k).collect(),
        OrderedNode::Other => panic!("{name}: {path:?} is not a mapping"),
    }
}

/// A YAML node that keeps mapping entries in document order; non-mappings are dropped.
enum OrderedNode {
    Map(Vec<(String, OrderedNode)>),
    Other,
}

impl<'de> serde::Deserialize<'de> for OrderedNode {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = OrderedNode;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("any YAML node")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut m: A,
            ) -> Result<OrderedNode, A::Error> {
                let mut entries = Vec::new();
                while let Some((k, v)) = m.next_entry::<String, OrderedNode>()? {
                    entries.push((k, v));
                }
                Ok(OrderedNode::Map(entries))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut s: A,
            ) -> Result<OrderedNode, A::Error> {
                while s.next_element::<serde::de::IgnoredAny>()?.is_some() {}
                Ok(OrderedNode::Other)
            }
            fn visit_bool<E>(self, _: bool) -> Result<OrderedNode, E> {
                Ok(OrderedNode::Other)
            }
            fn visit_i64<E>(self, _: i64) -> Result<OrderedNode, E> {
                Ok(OrderedNode::Other)
            }
            fn visit_u64<E>(self, _: u64) -> Result<OrderedNode, E> {
                Ok(OrderedNode::Other)
            }
            fn visit_f64<E>(self, _: f64) -> Result<OrderedNode, E> {
                Ok(OrderedNode::Other)
            }
            fn visit_str<E>(self, _: &str) -> Result<OrderedNode, E> {
                Ok(OrderedNode::Other)
            }
            fn visit_unit<E>(self) -> Result<OrderedNode, E> {
                Ok(OrderedNode::Other)
            }
            fn visit_none<E>(self) -> Result<OrderedNode, E> {
                Ok(OrderedNode::Other)
            }
            fn visit_some<D2: serde::Deserializer<'de>>(
                self,
                d: D2,
            ) -> Result<OrderedNode, D2::Error> {
                <OrderedNode as serde::Deserialize>::deserialize(d)
            }
        }
        d.deserialize_any(V)
    }
}

#[cfg(test)]
mod tests {
    use super::within;

    /// KANI-LAYA-PARITY-001's evidence: `within` is false whenever either argument is
    /// NaN, in both positions, against every special value; and it is exactly `<=`
    /// on ordinary values.
    #[test]
    fn within_is_nan_visible() {
        let specials = [
            f64::NAN,
            -f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            1e-5,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::from(f32::NAN),
        ];
        for &a in &specials {
            for &b in &specials {
                let want = !a.is_nan() && !b.is_nan() && a <= b;
                assert_eq!(within(a, b), want, "within({a}, {b})");
            }
        }
        assert!(within(1e-5, 1e-5), "the bound itself passes");
        assert!(!within(1.000_000_1e-5, 1e-5), "just over the bound fails");
    }
}
