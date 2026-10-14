//! The crate's one YAML door (#5205): serde-saphyr underneath, so no
//! `unsafe-libyaml` in the tree.
//!
//! Every parse goes through [`from_str`], which applies the same options at
//! every call site:
//!
//! - `strict_booleans`: only `true`/`false` are booleans; `yes`/`no`/`on`/`off`
//!   stay strings (YAML 1.2, as serde_yaml 0.9 read them).
//! - `merge_keys: AsOrdinary`: `<<` is an ordinary key, as in serde_yaml 0.9.
//! - `emit_comments: false`: comments are validated, never captured.
//! - `reject_non_finite_typeless_float: false`: `.inf`/`.nan` in an untyped
//!   slot are accepted (saphyr hands them over as canonical strings, and
//!   [`Value`] turns them back into floats — see [`Value`]'s `Deserialize`).
//!
//! [`Value`] and [`Mapping`] replace `serde_yaml::Value`/`Mapping` with the
//! subset of their API this crate uses. A [`Mapping`] keeps FILE ORDER and
//! accepts any [`Value`] as a key; lookups are linear, which is fine for the
//! small maps contracts carry.
//!
//! [`from_value`] deserializes straight out of a [`Value`] (it implements
//! `serde::Deserializer`). [`to_value`] goes through text: it serializes with
//! [`to_string`] and parses the result with [`from_str`].

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};

use serde::de::value::{MapDeserializer, SeqDeserializer};
use serde::de::{
    self, DeserializeOwned, Deserializer, IntoDeserializer, MapAccess, SeqAccess, Visitor,
};
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};

/// The error a parse (or a [`from_value`]) returns.
pub type Error = serde_saphyr::Error;

/// The error [`to_string`] returns.
pub type SerializeError = serde_saphyr::SerializeError;

fn options() -> serde_saphyr::Options {
    serde_saphyr::options! {
        strict_booleans: true,
        merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary,
        emit_comments: false,
        reject_non_finite_typeless_float: false,
    }
}

/// Parse YAML text into `T` with the crate's standard options.
///
/// # Errors
/// Any YAML syntax error, or a document that does not fit `T`.
pub fn from_str<T: DeserializeOwned>(s: &str) -> Result<T, Error> {
    serde_saphyr::from_str_with_options(s, options())
}

/// [`from_str`], except that a duplicate mapping key keeps the LAST value
/// instead of failing the parse.
///
/// For the contract schema's own lenient pass only (`schema::parser`):
/// serde_yaml's derived deserializers never looked inside subtrees they
/// skipped, so a duplicate key there parsed clean, and SCHEMA-020 reports it
/// by re-reading the document strictly through [`from_str`]. serde-saphyr
/// checks every mapping, so without this policy the duplicate would fail the
/// whole parse and SCHEMA-020 (and the `metadata.kind` probe) would never run.
///
/// # Errors
/// Any YAML syntax error, or a document that does not fit `T`.
pub fn from_str_last_key_wins<T: DeserializeOwned>(s: &str) -> Result<T, Error> {
    let mut opts = options();
    opts.duplicate_keys = serde_saphyr::DuplicateKeyPolicy::LastWins;
    serde_saphyr::from_str_with_options(s, opts)
}

/// Serialize `value` as YAML text.
///
/// # Errors
/// When `T`'s `Serialize` impl fails, or emits a shape YAML cannot hold.
pub fn to_string<T: Serialize>(value: &T) -> Result<String, SerializeError> {
    serde_saphyr::to_string(value)
}

/// Deserialize `T` out of an already-parsed [`Value`].
///
/// # Errors
/// When the value does not fit `T`.
pub fn from_value<T: DeserializeOwned>(value: Value) -> Result<T, Error> {
    T::deserialize(value)
}

/// Turn any `Serialize` into a [`Value`] by serializing it to text and parsing
/// that text back ([`to_string`] then [`from_str`]).
///
/// # Errors
/// When serialization fails or the emitted text does not parse.
pub fn to_value<T: Serialize>(value: T) -> Result<Value, Error> {
    let text = to_string(&value).map_err(<Error as de::Error>::custom)?;
    from_str(&text)
}

// ---------------------------------------------------------------- Number

#[derive(Clone, Copy)]
enum N {
    PosInt(u64),
    /// Always negative.
    NegInt(i64),
    Float(f64),
}

/// A YAML number: an integer that fits `i64`/`u64`, or an `f64` (including
/// `.inf`, `-.inf` and `.nan`).
#[derive(Clone, Copy)]
pub struct Number {
    n: N,
}

impl Number {
    #[must_use]
    pub fn is_i64(&self) -> bool {
        match self.n {
            N::PosInt(v) => i64::try_from(v).is_ok(),
            N::NegInt(_) => true,
            N::Float(_) => false,
        }
    }

    #[must_use]
    pub fn is_u64(&self) -> bool {
        matches!(self.n, N::PosInt(_))
    }

    #[must_use]
    pub fn is_f64(&self) -> bool {
        matches!(self.n, N::Float(_))
    }

    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self.n {
            N::PosInt(v) => i64::try_from(v).ok(),
            N::NegInt(v) => Some(v),
            N::Float(_) => None,
        }
    }

    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self.n {
            N::PosInt(v) => Some(v),
            N::NegInt(_) | N::Float(_) => None,
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(&self) -> Option<f64> {
        match self.n {
            N::PosInt(v) => Some(v as f64),
            N::NegInt(v) => Some(v as f64),
            N::Float(v) => Some(v),
        }
    }

    #[must_use]
    pub fn is_nan(&self) -> bool {
        matches!(self.n, N::Float(v) if v.is_nan())
    }

    #[must_use]
    pub fn is_infinite(&self) -> bool {
        matches!(self.n, N::Float(v) if v.is_infinite())
    }

    #[must_use]
    pub fn is_finite(&self) -> bool {
        !matches!(self.n, N::Float(v) if !v.is_finite())
    }
}

impl From<u64> for Number {
    fn from(v: u64) -> Self {
        Self { n: N::PosInt(v) }
    }
}

impl From<i64> for Number {
    fn from(v: i64) -> Self {
        match u64::try_from(v) {
            Ok(u) => Self { n: N::PosInt(u) },
            Err(_) => Self { n: N::NegInt(v) },
        }
    }
}

impl From<f64> for Number {
    fn from(v: f64) -> Self {
        Self { n: N::Float(v) }
    }
}

macro_rules! number_from_small {
    ($($t:ty => $via:ty),*) => {$(
        impl From<$t> for Number {
            fn from(v: $t) -> Self {
                Self::from(<$via>::from(v))
            }
        }
    )*};
}
number_from_small!(u8 => u64, u16 => u64, u32 => u64, i8 => i64, i16 => i64, i32 => i64, f32 => f64);

impl From<usize> for Number {
    fn from(v: usize) -> Self {
        Self::from(u64::try_from(v).unwrap_or(u64::MAX))
    }
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        match (self.n, other.n) {
            (N::PosInt(a), N::PosInt(b)) => a == b,
            (N::NegInt(a), N::NegInt(b)) => a == b,
            // NaN equals NaN, as in serde_yaml, so `Value` can be `Eq`.
            #[allow(clippy::float_cmp)]
            (N::Float(a), N::Float(b)) => a == b || (a.is_nan() && b.is_nan()),
            _ => false,
        }
    }
}

impl Eq for Number {}

impl Hash for Number {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.n {
            N::PosInt(v) => {
                0u8.hash(state);
                v.hash(state);
            }
            N::NegInt(v) => {
                1u8.hash(state);
                v.hash(state);
            }
            N::Float(v) => {
                2u8.hash(state);
                // One bit pattern per NaN, and 0.0 == -0.0.
                let bits = if v.is_nan() {
                    f64::NAN.to_bits()
                } else if v == 0.0 {
                    0
                } else {
                    v.to_bits()
                };
                bits.hash(state);
            }
        }
    }
}

/// serde_yaml's spelling: integers as-is, non-finite floats as `.nan`,
/// `.inf`, `-.inf`, finite floats in Rust's shortest round-trip form
/// (`1.0`, `0.5`, `1e20`).
impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.n {
            N::PosInt(v) => write!(f, "{v}"),
            N::NegInt(v) => write!(f, "{v}"),
            N::Float(v) if v.is_nan() => f.write_str(".nan"),
            N::Float(v) if v.is_infinite() => f.write_str(if v.is_sign_negative() {
                "-.inf"
            } else {
                ".inf"
            }),
            N::Float(v) => write!(f, "{v:?}"),
        }
    }
}

impl fmt::Debug for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Number({self})")
    }
}

impl Serialize for Number {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.n {
            N::PosInt(v) => s.serialize_u64(v),
            N::NegInt(v) => s.serialize_i64(v),
            N::Float(v) => s.serialize_f64(v),
        }
    }
}

// ---------------------------------------------------------------- Value

/// A YAML sequence.
pub type Sequence = Vec<Value>;

/// Any YAML value. Tags are dropped at parse time (serde-saphyr hands untyped
/// visitors the untagged node).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Value {
    #[default]
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Sequence(Sequence),
    Mapping(Mapping),
}

static NULL: Value = Value::Null;

impl Value {
    /// Index into a sequence (by `usize`) or a mapping (by key).
    pub fn get<I: Index>(&self, index: I) -> Option<&Value> {
        index.index_into(self)
    }

    pub fn get_mut<I: Index>(&mut self, index: I) -> Option<&mut Value> {
        index.index_into_mut(self)
    }

    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    #[must_use]
    pub fn as_null(&self) -> Option<()> {
        self.is_null().then_some(())
    }

    #[must_use]
    pub fn is_bool(&self) -> bool {
        matches!(self, Value::Bool(_))
    }

    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    #[must_use]
    pub fn is_number(&self) -> bool {
        matches!(self, Value::Number(_))
    }

    #[must_use]
    pub fn is_i64(&self) -> bool {
        matches!(self, Value::Number(n) if n.is_i64())
    }

    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Number(n) => n.as_i64(),
            _ => None,
        }
    }

    #[must_use]
    pub fn is_u64(&self) -> bool {
        matches!(self, Value::Number(n) if n.is_u64())
    }

    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Number(n) => n.as_u64(),
            _ => None,
        }
    }

    #[must_use]
    pub fn is_f64(&self) -> bool {
        matches!(self, Value::Number(n) if n.is_f64())
    }

    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(n) => n.as_f64(),
            _ => None,
        }
    }

    #[must_use]
    pub fn is_string(&self) -> bool {
        matches!(self, Value::String(_))
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    #[must_use]
    pub fn is_sequence(&self) -> bool {
        matches!(self, Value::Sequence(_))
    }

    #[must_use]
    pub fn as_sequence(&self) -> Option<&Sequence> {
        match self {
            Value::Sequence(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_sequence_mut(&mut self) -> Option<&mut Sequence> {
        match self {
            Value::Sequence(s) => Some(s),
            _ => None,
        }
    }

    #[must_use]
    pub fn is_mapping(&self) -> bool {
        matches!(self, Value::Mapping(_))
    }

    #[must_use]
    pub fn as_mapping(&self) -> Option<&Mapping> {
        match self {
            Value::Mapping(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_mapping_mut(&mut self) -> Option<&mut Mapping> {
        match self {
            Value::Mapping(m) => Some(m),
            _ => None,
        }
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::String(v.to_owned())
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::String(v)
    }
}

impl From<Number> for Value {
    fn from(v: Number) -> Self {
        Value::Number(v)
    }
}

impl From<Mapping> for Value {
    fn from(v: Mapping) -> Self {
        Value::Mapping(v)
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::Sequence(v.into_iter().map(Into::into).collect())
    }
}

macro_rules! value_from_number {
    ($($t:ty),*) => {$(
        impl From<$t> for Value {
            fn from(v: $t) -> Self {
                Value::Number(Number::from(v))
            }
        }
    )*};
}
value_from_number!(u8, u16, u32, u64, usize, i8, i16, i32, i64, f32, f64);

impl PartialEq<str> for Value {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == Some(other)
    }
}

impl PartialEq<&str> for Value {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == Some(*other)
    }
}

impl PartialEq<String> for Value {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == Some(other.as_str())
    }
}

impl PartialEq<bool> for Value {
    fn eq(&self, other: &bool) -> bool {
        self.as_bool() == Some(*other)
    }
}

impl<I: Index> std::ops::Index<I> for Value {
    type Output = Value;

    /// A missing key or index yields `Value::Null`, as in serde_yaml.
    fn index(&self, index: I) -> &Value {
        index.index_into(self).unwrap_or(&NULL)
    }
}

// ---------------------------------------------------------------- Index

mod private {
    pub trait Sealed {}
    impl Sealed for usize {}
    impl Sealed for str {}
    impl Sealed for String {}
    impl Sealed for super::Value {}
    impl<T: Sealed + ?Sized> Sealed for &T {}
}

/// What [`Value::get`] and [`Mapping::get`] accept: `usize` (a sequence
/// position, or an integer mapping key), `&str`/`String` (a string key), or a
/// whole [`Value`] key.
pub trait Index: private::Sealed {
    #[doc(hidden)]
    fn position_in(&self, map: &Mapping) -> Option<usize>;

    #[doc(hidden)]
    fn index_into<'v>(&self, v: &'v Value) -> Option<&'v Value> {
        match v {
            Value::Mapping(m) => self.position_in(m).map(|i| &m.entries[i].1),
            _ => None,
        }
    }

    #[doc(hidden)]
    fn index_into_mut<'v>(&self, v: &'v mut Value) -> Option<&'v mut Value> {
        match v {
            Value::Mapping(m) => self.position_in(m).map(move |i| &mut m.entries[i].1),
            _ => None,
        }
    }
}

impl Index for usize {
    fn position_in(&self, map: &Mapping) -> Option<usize> {
        let key = Value::Number(Number::from(*self));
        map.entries.iter().position(|(k, _)| *k == key)
    }

    fn index_into<'v>(&self, v: &'v Value) -> Option<&'v Value> {
        match v {
            Value::Sequence(s) => s.get(*self),
            Value::Mapping(m) => self.position_in(m).map(|i| &m.entries[i].1),
            _ => None,
        }
    }

    fn index_into_mut<'v>(&self, v: &'v mut Value) -> Option<&'v mut Value> {
        match v {
            Value::Sequence(s) => s.get_mut(*self),
            Value::Mapping(m) => self.position_in(m).map(move |i| &mut m.entries[i].1),
            _ => None,
        }
    }
}

impl Index for str {
    fn position_in(&self, map: &Mapping) -> Option<usize> {
        map.entries
            .iter()
            .position(|(k, _)| k.as_str() == Some(self))
    }
}

impl Index for String {
    fn position_in(&self, map: &Mapping) -> Option<usize> {
        self.as_str().position_in(map)
    }
}

impl Index for Value {
    fn position_in(&self, map: &Mapping) -> Option<usize> {
        map.entries.iter().position(|(k, _)| k == self)
    }
}

impl<T: Index + ?Sized> Index for &T {
    fn position_in(&self, map: &Mapping) -> Option<usize> {
        (**self).position_in(map)
    }

    fn index_into<'v>(&self, v: &'v Value) -> Option<&'v Value> {
        (**self).index_into(v)
    }

    fn index_into_mut<'v>(&self, v: &'v mut Value) -> Option<&'v mut Value> {
        (**self).index_into_mut(v)
    }
}

// ---------------------------------------------------------------- Mapping

/// A YAML mapping in file order. Keys may be any [`Value`]. Equality and
/// hashing ignore order, as serde_yaml's did.
#[derive(Clone, Default)]
pub struct Mapping {
    entries: Vec<(Value, Value)>,
}

impl Mapping {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert `k → v`. An existing key keeps its position and its old value
    /// is returned; a new key goes last.
    pub fn insert(&mut self, k: Value, v: Value) -> Option<Value> {
        if let Some(i) = k.position_in(self) {
            return Some(std::mem::replace(&mut self.entries[i].1, v));
        }
        self.entries.push((k, v));
        None
    }

    pub fn get<I: Index>(&self, index: I) -> Option<&Value> {
        index.position_in(self).map(|i| &self.entries[i].1)
    }

    pub fn get_mut<I: Index>(&mut self, index: I) -> Option<&mut Value> {
        index.position_in(self).map(move |i| &mut self.entries[i].1)
    }

    pub fn contains_key<I: Index>(&self, index: I) -> bool {
        index.position_in(self).is_some()
    }

    /// Remove a key, keeping the order of the rest.
    pub fn remove<I: Index>(&mut self, index: I) -> Option<Value> {
        index.position_in(self).map(|i| self.entries.remove(i).1)
    }

    pub fn retain<F: FnMut(&Value, &mut Value) -> bool>(&mut self, mut keep: F) {
        self.entries.retain_mut(|(k, v)| keep(k, v));
    }

    pub fn iter(&self) -> Iter<'_> {
        Iter {
            inner: self.entries.iter(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_> {
        IterMut {
            inner: self.entries.iter_mut(),
        }
    }

    pub fn keys(&self) -> impl DoubleEndedIterator<Item = &Value> + ExactSizeIterator + '_ {
        self.entries.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl DoubleEndedIterator<Item = &Value> + ExactSizeIterator + '_ {
        self.entries.iter().map(|(_, v)| v)
    }

    pub fn values_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = &mut Value> + ExactSizeIterator + '_ {
        self.entries.iter_mut().map(|(_, v)| v)
    }
}

impl fmt::Debug for Mapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.entries.iter().map(|(k, v)| (k, v)))
            .finish()
    }
}

impl PartialEq for Mapping {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|(k, v)| other.get(k) == Some(v))
    }
}

impl Eq for Mapping {}

impl Hash for Mapping {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Order-independent: XOR of the per-entry hashes.
        let mut acc = 0u64;
        for entry in &self.entries {
            let mut h = DefaultHasher::new();
            entry.hash(&mut h);
            acc ^= h.finish();
        }
        acc.hash(state);
    }
}

impl<I: Index> std::ops::Index<I> for Mapping {
    type Output = Value;

    /// Panics on a missing key, as serde_yaml's did.
    fn index(&self, index: I) -> &Value {
        self.get(index).expect("key not found in yaml::Mapping")
    }
}

impl FromIterator<(Value, Value)> for Mapping {
    fn from_iter<T: IntoIterator<Item = (Value, Value)>>(iter: T) -> Self {
        let mut m = Mapping::new();
        m.extend(iter);
        m
    }
}

impl Extend<(Value, Value)> for Mapping {
    fn extend<T: IntoIterator<Item = (Value, Value)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

/// Borrowing iterator over a [`Mapping`], in file order.
pub struct Iter<'a> {
    inner: std::slice::Iter<'a, (Value, Value)>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a Value, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for Iter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(k, v)| (k, v))
    }
}

impl ExactSizeIterator for Iter<'_> {}

/// Mutably borrowing iterator over a [`Mapping`], in file order.
pub struct IterMut<'a> {
    inner: std::slice::IterMut<'a, (Value, Value)>,
}

impl<'a> Iterator for IterMut<'a> {
    type Item = (&'a Value, &'a mut Value);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (&*k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for IterMut<'_> {}

impl<'a> IntoIterator for &'a Mapping {
    type Item = (&'a Value, &'a Value);
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut Mapping {
    type Item = (&'a Value, &'a mut Value);
    type IntoIter = IterMut<'a>;

    fn into_iter(self) -> IterMut<'a> {
        self.iter_mut()
    }
}

impl IntoIterator for Mapping {
    type Item = (Value, Value);
    type IntoIter = std::vec::IntoIter<(Value, Value)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

// ---------------------------------------------------------------- Serialize

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::Null => s.serialize_unit(),
            Value::Bool(b) => s.serialize_bool(*b),
            Value::Number(n) => n.serialize(s),
            Value::String(v) => s.serialize_str(v),
            Value::Sequence(v) => v.serialize(s),
            Value::Mapping(m) => m.serialize(s),
        }
    }
}

impl Serialize for Mapping {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(self.len()))?;
        for (k, v) in self {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

// ---------------------------------------------------------------- Deserialize

fn beyond_64_bits<E: de::Error>(v: &dyn fmt::Display) -> E {
    E::custom(format_args!("integer `{v}` is beyond 64 bits"))
}

struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any YAML value")
    }

    fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
        Ok(Value::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }

    fn visit_i128<E: de::Error>(self, v: i128) -> Result<Value, E> {
        match i64::try_from(v) {
            Ok(i) => Ok(Value::Number(i.into())),
            Err(_) => match u64::try_from(v) {
                Ok(u) => Ok(Value::Number(u.into())),
                Err(_) => Err(beyond_64_bits(&v)),
            },
        }
    }

    fn visit_u128<E: de::Error>(self, v: u128) -> Result<Value, E> {
        u64::try_from(v)
            .map(|u| Value::Number(u.into()))
            .map_err(|_| beyond_64_bits(&v))
    }

    /// serde-saphyr hands an integer literal beyond 64 bits over as a float. serde_yaml refused
    /// it in an untyped slot (its `Value` has no 128-bit integer), and `pv validate` keeps that
    /// verdict, so an integral float outside the 64-bit range is refused here. A float literal
    /// that large in an untyped slot (`2e20`) is refused with it: the visitor cannot tell them apart.
    /// Both bounds are exclusive: `-9223372036854775809` rounds to exactly -2^63, and `i64::MIN`
    /// itself arrives through `visit_i64`, never here.
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Value, E> {
        const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;
        const NEG_TWO_POW_63: f64 = -9_223_372_036_854_775_808.0;
        if v.is_finite() && v.fract() == 0.0 && (v <= NEG_TWO_POW_63 || v >= TWO_POW_64) {
            return Err(beyond_64_bits(&v));
        }
        Ok(Value::Number(v.into()))
    }

    fn visit_str<E>(self, v: &str) -> Result<Value, E> {
        Ok(Value::String(v.to_owned()))
    }

    /// serde-saphyr hands a typeless non-finite float over as an OWNED
    /// canonical string (`.nan`, `.inf`, `-.inf`), and a quoted scalar as a
    /// borrowed one; an owned canonical spelling is therefore the float. The
    /// one string that would be misread is a quoted `".inf"`-style scalar the
    /// parser had to copy (an escape inside it), which no contract carries.
    fn visit_string<E>(self, v: String) -> Result<Value, E> {
        Ok(match v.as_str() {
            ".nan" => Value::Number(f64::NAN.into()),
            ".inf" => Value::Number(f64::INFINITY.into()),
            "-.inf" => Value::Number(f64::NEG_INFINITY.into()),
            _ => Value::String(v),
        })
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
        Value::deserialize(d)
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
        Value::deserialize(d)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(4096));
        while let Some(v) = seq.next_element()? {
            out.push(v);
        }
        Ok(Value::Sequence(out))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut out = Mapping::with_capacity(map.size_hint().unwrap_or(0).min(4096));
        // Hash each key so a duplicate (later one wins, as an insert would)
        // costs a linear scan only on a hash hit, not on every key.
        let mut seen = HashSet::new();
        while let Some((k, v)) = map.next_entry::<Value, Value>()? {
            let mut h = DefaultHasher::new();
            k.hash(&mut h);
            if seen.insert(h.finish()) {
                out.entries.push((k, v));
            } else {
                out.insert(k, v);
            }
        }
        Ok(Value::Mapping(out))
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Value, D::Error> {
        d.deserialize_any(ValueVisitor)
    }
}

impl<'de> Deserialize<'de> for Mapping {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Mapping, D::Error> {
        match Value::deserialize(d)? {
            Value::Mapping(m) => Ok(m),
            Value::Null => Ok(Mapping::new()),
            other => Err(de::Error::invalid_type(
                other.unexpected(),
                &"a YAML mapping",
            )),
        }
    }
}

// ---------------------------------------------------------------- Value as a Deserializer

impl Value {
    fn unexpected(&self) -> de::Unexpected<'_> {
        match self {
            Value::Null => de::Unexpected::Unit,
            Value::Bool(b) => de::Unexpected::Bool(*b),
            Value::Number(n) => match n.n {
                N::PosInt(v) => de::Unexpected::Unsigned(v),
                N::NegInt(v) => de::Unexpected::Signed(v),
                N::Float(v) => de::Unexpected::Float(v),
            },
            Value::String(s) => de::Unexpected::Str(s),
            Value::Sequence(_) => de::Unexpected::Seq,
            Value::Mapping(_) => de::Unexpected::Map,
        }
    }
}

impl<'de> IntoDeserializer<'de, Error> for Value {
    type Deserializer = Value;

    fn into_deserializer(self) -> Value {
        self
    }
}

impl<'de> Deserializer<'de> for Value {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self {
            Value::Null => visitor.visit_unit(),
            Value::Bool(b) => visitor.visit_bool(b),
            Value::Number(n) => match n.n {
                N::PosInt(v) => visitor.visit_u64(v),
                N::NegInt(v) => visitor.visit_i64(v),
                N::Float(v) => visitor.visit_f64(v),
            },
            Value::String(s) => visitor.visit_string(s),
            Value::Sequence(v) => {
                let mut seq = SeqDeserializer::new(v.into_iter());
                let out = visitor.visit_seq(&mut seq)?;
                seq.end()?;
                Ok(out)
            }
            Value::Mapping(m) => {
                let mut map = MapDeserializer::new(m.entries.into_iter());
                let out = visitor.visit_map(&mut map)?;
                map.end()?;
                Ok(out)
            }
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self {
            Value::Null => visitor.visit_none(),
            other => visitor.visit_some(other),
        }
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        match self {
            // A unit variant: `kind: kernel`.
            Value::String(s) => visitor.visit_enum(s.into_deserializer()),
            // Any other variant: a one-entry map `{Variant: payload}`.
            Value::Mapping(m) if m.len() == 1 => visitor.visit_enum(
                de::value::MapAccessDeserializer::new(MapDeserializer::new(m.entries.into_iter())),
            ),
            other => Err(de::Error::invalid_type(
                other.unexpected(),
                &"a string or a one-entry mapping (an enum)",
            )),
        }
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf unit unit_struct seq tuple tuple_struct map struct
        identifier ignored_any
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(m: &Mapping) -> Vec<String> {
        m.keys()
            .map(|k| k.as_str().unwrap_or("?").to_owned())
            .collect()
    }

    #[test]
    fn key_order_survives_parse_and_serialize() {
        let text = "zeta: 1\nalpha: 2\nmid: 3\nbeta: 4\n";
        let v: Value = from_str(text).expect("parse");
        let m = v.as_mapping().expect("mapping");
        assert_eq!(keys(m), ["zeta", "alpha", "mid", "beta"]);
        let out = to_string(&v).expect("serialize");
        let order: Vec<&str> = out.lines().filter_map(|l| l.split(':').next()).collect();
        assert_eq!(order, ["zeta", "alpha", "mid", "beta"], "{out}");
        let back: Value = from_str(&out).expect("reparse");
        assert_eq!(back, v);
        assert_eq!(keys(back.as_mapping().expect("mapping")), keys(m));
    }

    #[test]
    fn an_integer_beyond_64_bits_is_refused_as_serde_yaml_did() {
        for text in [
            "n: 184467440737095516160\n",
            "n: 18446744073709551616\n",
            "n: -9223372036854775809\n",
            // The cost of the rule: a float literal this large is refused with them.
            "n: 2e20\n",
        ] {
            let err = from_str::<Value>(text).expect_err(text).to_string();
            assert!(err.contains("beyond 64 bits"), "{text}: {err}");
        }
        for (text, want) in [
            ("n: 18446744073709551615\n", Value::from(u64::MAX)),
            ("n: -9223372036854775808\n", Value::from(i64::MIN)),
            ("n: 1.5e18\n", Value::from(1.5e18)),
            ("n: 0.5\n", Value::from(0.5)),
        ] {
            assert_eq!(from_str::<Value>(text).expect(text)["n"], want, "{text}");
        }
    }

    #[test]
    fn integer_key_is_a_number_key() {
        let v: Value = from_str("1: one\nname: two\n").expect("parse");
        assert_eq!(v.get(1usize).and_then(Value::as_str), Some("one"));
        assert_eq!(
            v.get(Value::from(1u64)).and_then(Value::as_str),
            Some("one")
        );
        assert_eq!(v["name"].as_str(), Some("two"));
        assert!(
            v.get("1").is_none(),
            "an integer key is not the string \"1\""
        );
        assert!(v["missing"].is_null());
    }

    #[test]
    fn non_finite_floats_survive_an_untyped_slot() {
        let v: Value = from_str("a: .inf\nb: -.inf\nc: .nan\nd: 1.5\ne: '.inf'\n").expect("parse");
        assert_eq!(v["a"].as_f64(), Some(f64::INFINITY));
        assert_eq!(v["b"].as_f64(), Some(f64::NEG_INFINITY));
        assert!(v["c"].as_f64().is_some_and(f64::is_nan));
        assert_eq!(v["d"].as_f64(), Some(1.5));
        assert_eq!(
            v["e"].as_str(),
            Some(".inf"),
            "a quoted .inf stays a string"
        );
        let Value::Number(n) = &v["a"] else {
            panic!("a is not a number")
        };
        assert_eq!(n.to_string(), ".inf");
        let back: Value = from_str(&to_string(&v).expect("serialize")).expect("reparse");
        assert_eq!(back, v);
    }

    #[test]
    fn yes_and_no_stay_strings() {
        let v: Value = from_str("a: yes\nb: no\nc: on\nd: true\ne: false\n").expect("parse");
        assert_eq!(v["a"], "yes");
        assert_eq!(v["b"], "no");
        assert_eq!(v["c"], "on");
        assert_eq!(v["d"], true);
        assert_eq!(v["e"], false);
    }

    #[test]
    fn merge_key_is_an_ordinary_key() {
        let v: Value = from_str("base: &b {x: 1}\nchild:\n  <<: *b\n  y: 2\n").expect("parse");
        let child = v["child"].as_mapping().expect("mapping");
        assert_eq!(keys(child), ["<<", "y"]);
        assert!(child.get("x").is_none(), "<< was not expanded");
        assert_eq!(child["<<"]["x"].as_u64(), Some(1));
    }

    #[test]
    fn integers_keep_their_kind() {
        let v: Value = from_str("p: 18446744073709551615\nn: -3\nf: 2.0\n").expect("parse");
        assert_eq!(v["p"].as_u64(), Some(u64::MAX));
        assert_eq!(v["p"].as_i64(), None);
        assert_eq!(v["n"].as_i64(), Some(-3));
        assert_eq!(v["n"].as_u64(), None);
        assert!(v["f"].is_f64());
        let Value::Number(f) = &v["f"] else {
            panic!("f is not a number")
        };
        assert_eq!(f.to_string(), "2.0");
    }

    #[test]
    fn duplicate_key_is_an_error() {
        assert!(from_str::<Value>("a: 1\na: 2\n").is_err());
    }

    #[test]
    fn mapping_equality_ignores_order() {
        let a: Value = from_str("x: 1\ny: 2\n").expect("parse");
        let b: Value = from_str("y: 2\nx: 1\n").expect("parse");
        assert_eq!(a, b);
    }

    #[test]
    fn from_value_and_to_value_round_trip() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        enum Kind {
            Unit,
            Pair(u8, u8),
        }
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct S {
            name: String,
            kinds: Vec<Kind>,
            opt: Option<i32>,
            ratio: f64,
        }
        let s = S {
            name: "n".into(),
            kinds: vec![Kind::Unit, Kind::Pair(1, 2)],
            opt: None,
            ratio: 0.5,
        };
        let v = to_value(&s).expect("to_value");
        assert_eq!(v["name"], "n");
        let back: S = from_value(v).expect("from_value");
        assert_eq!(back, s);
    }

    #[test]
    fn mapping_insert_keeps_position_and_remove_keeps_order() {
        let mut m = Mapping::new();
        m.insert("a".into(), 1u64.into());
        m.insert("b".into(), 2u64.into());
        m.insert("c".into(), 3u64.into());
        assert_eq!(m.insert("a".into(), 9u64.into()), Some(Value::from(1u64)));
        assert_eq!(keys(&m), ["a", "b", "c"]);
        assert_eq!(m.remove("b"), Some(Value::from(2u64)));
        assert_eq!(keys(&m), ["a", "c"]);
        assert!(m.contains_key("a") && !m.contains_key("b"));
    }
}
