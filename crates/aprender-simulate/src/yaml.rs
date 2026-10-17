//! The one YAML read: serde-saphyr with the #5205 stage options (YAML 1.2 booleans, merge keys as ordinary keys).

/// Parse `s` as YAML into `T`.
///
/// # Errors
/// Returns the parser's error when `s` is not valid YAML for `T`.
pub fn from_str<T: serde::de::DeserializeOwned>(s: &str) -> Result<T, serde_saphyr::Error> {
    serde_saphyr::from_str_with_options(
        s,
        serde_saphyr::options! { strict_booleans: true, merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary, emit_comments: false, reject_non_finite_typeless_float: false },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yes_stays_a_string_and_true_is_a_bool() {
        let v: serde_json::Value = from_str("a: yes\nb: true\n").expect("valid yaml");
        assert_eq!(v["a"], "yes");
        assert_eq!(v["b"], true);
    }

    #[test]
    fn invalid_yaml_is_an_error() {
        assert!(from_str::<serde_json::Value>("a: [1, 2").is_err());
    }
}
