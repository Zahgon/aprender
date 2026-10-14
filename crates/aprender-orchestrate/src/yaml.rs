//! YAML reading for this crate (#5205).

/// The standard stage options (#5205): every YAML parse in this crate goes
/// through `serde-saphyr` with these options.
pub(crate) fn yaml_from_str<T: serde::de::DeserializeOwned>(
    s: &str,
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
