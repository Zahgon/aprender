//! The one YAML reader for this crate (#5205 S4).
//!
//! serde-saphyr replaces serde_yaml 0.9 (which pulls unsafe-libyaml,
//! RUSTSEC-2024-0449). The options keep the old reader's semantics:
//! `yes`/`no` stay strings, `<<` stays an ordinary key, a long comment run is
//! not a budget refusal, and `.inf`/`.nan` load in typeless slots.
//!
//! An untyped read (`serde_json::Value` or `Map`) goes through [`Guarded`], which keeps two
//! more of serde_yaml's verdicts that a plain serde-saphyr read into `serde_json` loses.

use std::fmt;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};

fn options(last_wins: bool) -> serde_saphyr::Options {
    let mut opts = serde_saphyr::options! {
        strict_booleans: true,
        merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary,
        emit_comments: false,
        reject_non_finite_typeless_float: false,
    };
    if last_wins {
        opts.duplicate_keys = serde_saphyr::DuplicateKeyPolicy::LastWins;
    }
    opts
}

fn read<'de, T: serde::Deserialize<'de>>(
    s: &'de str,
    last_wins: bool,
) -> Result<T, serde_saphyr::Error> {
    let target = std::any::type_name::<T>();
    let untyped = target == std::any::type_name::<Value>()
        || target == std::any::type_name::<Map<String, Value>>();
    if !untyped {
        return serde_saphyr::from_str_with_options(s, options(last_wins));
    }
    let value = if last_wins {
        serde_saphyr::from_str_with_options::<Guarded<true>>(s, options(true))?.0
    } else {
        serde_saphyr::from_str_with_options::<Guarded<false>>(s, options(false))?.0
    };
    T::deserialize(value).map_err(de::Error::custom)
}

/// Parse `s` as YAML into `T` with the #5205 stage options.
///
/// # Errors
///
/// Returns the serde-saphyr error if `s` is not valid YAML or does not match `T`.
pub(crate) fn from_str<'de, T: serde::Deserialize<'de>>(
    s: &'de str,
) -> Result<T, serde_saphyr::Error> {
    read(s, false)
}

/// [`from_str`], but a duplicate mapping key keeps the last value instead of
/// refusing the document.
///
/// serde_yaml 0.9 refused a duplicate key only when it built a `Value`/`Mapping`;
/// a typed read (`Contract`, a `BTreeMap`, a probe struct) skipped or overwrote
/// it. The contract schema relies on that split: the typed parse must accept the
/// document so SCHEMA-020 can report the duplicate from the strict `Value` read
/// (`strict_yaml_error`). serde-saphyr checks duplicates in the parser for every
/// target, so the typed contract reads opt into `LastWins` here to keep it.
///
/// # Errors
///
/// Returns the serde-saphyr error if `s` is not valid YAML or does not match `T`.
pub(crate) fn from_str_lenient<'de, T: serde::Deserialize<'de>>(
    s: &'de str,
) -> Result<T, serde_saphyr::Error> {
    read(s, true)
}

/// An untyped document, read with two of serde_yaml's verdicts kept:
///
/// - An integer beyond 64 bits is refused. serde-saphyr hands one over as a float; serde_yaml's
///   `Value` had no 128-bit integer and refused it. A float literal that large (`2e20`) is refused
///   with it, since the visitor cannot tell them apart.
/// - serde_yaml kept `1:` and `"1":` as two keys. A `serde_json` map has string keys only, so
///   the second would silently replace the first. That is refused instead, unless `LAST_WINS`.
struct Guarded<const LAST_WINS: bool>(Value);

impl<'de, const LAST_WINS: bool> serde::Deserialize<'de> for Guarded<LAST_WINS> {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(GuardedVisitor::<LAST_WINS>).map(Guarded)
    }
}

impl<'de, const LAST_WINS: bool> DeserializeSeed<'de> for GuardedVisitor<LAST_WINS> {
    type Value = Value;
    fn deserialize<D: de::Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
        d.deserialize_any(self)
    }
}

fn beyond_64_bits<E: de::Error>(v: &dyn fmt::Display) -> E {
    E::custom(format_args!("integer `{v}` is beyond 64 bits"))
}

struct GuardedVisitor<const LAST_WINS: bool>;

impl<'de, const LAST_WINS: bool> Visitor<'de> for GuardedVisitor<LAST_WINS> {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any YAML value")
    }

    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Value, E> {
        Ok(Value::Bool(v))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Value, E> {
        Ok(Value::from(v))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Value, E> {
        Ok(Value::from(v))
    }

    fn visit_i128<E: de::Error>(self, v: i128) -> Result<Value, E> {
        i64::try_from(v)
            .map(Value::from)
            .map_err(|_| beyond_64_bits(&v))
    }

    fn visit_u128<E: de::Error>(self, v: u128) -> Result<Value, E> {
        u64::try_from(v)
            .map(Value::from)
            .map_err(|_| beyond_64_bits(&v))
    }

    /// Both bounds are exclusive: `-9223372036854775809` rounds to exactly -2^63, and `i64::MIN`
    /// itself arrives through `visit_i64`, never here.
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Value, E> {
        const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;
        const NEG_TWO_POW_63: f64 = -9_223_372_036_854_775_808.0;
        if v.is_finite() && v.fract() == 0.0 && (v <= NEG_TWO_POW_63 || v >= TWO_POW_64) {
            return Err(beyond_64_bits(&v));
        }
        // serde_json has no non-finite number; `.inf`/`.nan` become null, as `Value::from` does.
        Ok(Value::from(v))
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<Value, E> {
        Ok(Value::String(v.to_owned()))
    }

    fn visit_string<E: de::Error>(self, v: String) -> Result<Value, E> {
        Ok(Value::String(v))
    }

    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D: de::Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
        d.deserialize_any(self)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut out = Vec::new();
        while let Some(v) = seq.next_element_seed(GuardedVisitor::<LAST_WINS>)? {
            out.push(v);
        }
        Ok(Value::Array(out))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut out = Map::new();
        while let Some(key) = map.next_key_seed(GuardedVisitor::<LAST_WINS>)? {
            let key = match key {
                Value::String(s) => s,
                other => other.to_string(),
            };
            let value = map.next_value_seed(GuardedVisitor::<LAST_WINS>)?;
            if out.insert(key.clone(), value).is_some() && !LAST_WINS {
                return Err(de::Error::custom(format_args!(
                    "two keys read as `{key}` (an integer key and its string twin?); \
                     serde_yaml kept both, a string-keyed map cannot"
                )));
            }
        }
        Ok(Value::Object(out))
    }
}

#[cfg(test)]
mod tests {
    use super::from_str;

    /// serde_yaml 0.9 read `1: a` + `"1": b` as two keys. A `serde_json` map cannot hold both,
    /// and a plain read kept only `{"1": "b"}`. The strict read refuses the document instead of
    /// dropping a key; the lenient read keeps the last value, as it does for any duplicate.
    #[test]
    fn an_integer_key_and_its_string_twin_are_refused_not_silently_merged() {
        let text = "1: a\n\"1\": b\n";
        let err = from_str::<serde_json::Value>(text)
            .expect_err("strict")
            .to_string();
        assert!(err.contains("two keys read as `1`"), "{err}");
        let err = from_str::<serde_json::Map<String, serde_json::Value>>(text).expect_err("map");
        assert!(err.to_string().contains("two keys read as `1`"), "{err}");
        let last = super::from_str_lenient::<serde_json::Value>(text).expect("lenient");
        assert_eq!(last, serde_json::json!({"1": "b"}));
        let lone = from_str::<serde_json::Value>("1: a\n").expect("lone");
        assert_eq!(lone, serde_json::json!({"1": "a"}));
    }

    /// Parity with serde_yaml 0.9: an integer beyond 64 bits in an untyped slot is refused.
    #[test]
    fn an_integer_beyond_64_bits_is_refused_as_serde_yaml_refused_it() {
        for text in [
            "n: 184467440737095516160\n",
            "n: 18446744073709551616\n",
            "n: -9223372036854775809\n",
            // The cost of the rule: a float literal this large is refused with them.
            "n: 2e20\n",
        ] {
            let got = from_str::<serde_json::Value>(text);
            assert!(got.is_err(), "{text}: {got:?}");
        }
        for (text, want) in [
            (
                "n: 18446744073709551615\n",
                serde_json::json!({"n": u64::MAX}),
            ),
            (
                "n: -9223372036854775808\n",
                serde_json::json!({"n": i64::MIN}),
            ),
            ("n: 1.5e18\n", serde_json::json!({"n": 1.5e18})),
        ] {
            assert_eq!(
                from_str::<serde_json::Value>(text).expect(text),
                want,
                "{text}"
            );
        }
        // `.inf` reads as the guard-free read does (serde-saphyr hands it over as text here).
        let plain: serde_json::Value =
            serde_saphyr::from_str_with_options("n: .inf\n", super::options(false)).expect("plain");
        assert_eq!(
            from_str::<serde_json::Value>("n: .inf\n").expect(".inf"),
            plain
        );
        // A typed read is not guarded: serde_yaml read a bigint into an f64 field too.
        let typed: std::collections::BTreeMap<String, f64> =
            from_str("n: 184467440737095516160\n").expect("typed");
        assert!(typed["n"] > 1.8e20);
    }
}
