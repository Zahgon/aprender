//! The one YAML reader for this crate (#5205 S4).
//!
//! serde-saphyr replaces serde_yaml 0.9 (which pulls unsafe-libyaml,
//! RUSTSEC-2024-0449). The options keep the old reader's semantics:
//! `yes`/`no` stay strings, `<<` stays an ordinary key, a long comment run is
//! not a budget refusal, and `.inf`/`.nan` load in typeless slots.

/// Parse `s` as YAML into `T` with the #5205 stage options.
///
/// # Errors
///
/// Returns the serde-saphyr error if `s` is not valid YAML or does not match `T`.
pub(crate) fn from_str<'de, T: serde::Deserialize<'de>>(
    s: &'de str,
) -> Result<T, serde_saphyr::Error> {
    serde_saphyr::from_str_with_options(
        s,
        serde_saphyr::options! {
            strict_booleans: true,
            merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary,
            emit_comments: false,
            reject_non_finite_typeless_float: false,
        },
    )
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
    serde_saphyr::from_str_with_options(
        s,
        serde_saphyr::options! {
            strict_booleans: true,
            merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary,
            emit_comments: false,
            reject_non_finite_typeless_float: false,
            duplicate_keys: serde_saphyr::DuplicateKeyPolicy::LastWins,
        },
    )
}
