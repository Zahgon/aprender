//! #3432: ONE ggml type table, mirroring llama.cpp's `ggml.c` `type_traits[]`.
//!
//! Every place that needs "how many bytes does a tensor of ggml type N occupy"
//! used to answer it with its own hand-written `match`, and each of those
//! matches listed only the types its author had met. The loader's match
//! (`QuantizedGGUFTransformer::tensor_byte_size`) carried 11 arms, so real
//! unsloth files — `Qwen3.5-0.8B-UD-IQ2_XXS.gguf` (type 16) and
//! `Qwen3.5-0.8B-IQ4_XS.gguf` (type 23) — were refused at *size* time with
//! "Unsupported quantization type" before any kernel could be consulted.
//!
//! Sizing a type is NOT a claim that a kernel can run it: the dequant/matmul
//! refusal is downstream and unchanged (`fused_matmul_into.rs` fails loud for
//! any qtype it has no dequantizer for). This table answers one question only,
//! the one the GGUF byte layout already fixes.
//!
//! #3418: the rows are no longer written here. They are read at compile time
//! from `trueno_quant::TRAITS`, which is extracted from upstream ggml.
//!
//! Each row is `(id, name, blck_size, type_size)` where `type_size` is
//! `sizeof(block_*)` for a block of `blck_size` elements, exactly as in ggml.

use trueno_quant::TRAITS;

/// Static layout traits of one ggml tensor type.
///
/// Mirrors `ggml_type_traits` in llama.cpp's `ggml.c`: a block of `blck_size`
/// elements is stored in `type_size` bytes. Non-quantized types have
/// `blck_size == 1`, so `type_size` is simply bytes-per-element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlTypeTraits {
    /// The ggml type id as written in the GGUF tensor info record.
    pub id: u32,
    /// The ggml name of the type (`Q4_K`, `IQ2_XXS`, …), for error messages.
    pub name: &'static str,
    /// Elements per stored block (1 for the plain scalar types).
    pub blck_size: usize,
    /// Bytes occupied by one block of `blck_size` elements.
    pub type_size: usize,
}

/// How many ggml ids have a block layout: the live rows of [`TRAITS`].
const SIZED_COUNT: usize = {
    let mut n = 0;
    let mut i = 0;
    while i < TRAITS.len() {
        if TRAITS[i].blck_size > 0 {
            n += 1;
        }
        i += 1;
    }
    n
};

/// [`TRAITS`]'s live rows in id order, read at compile time (#3418).
const SIZED_ROWS: [GgmlTypeTraits; SIZED_COUNT] = {
    let mut rows = [GgmlTypeTraits {
        id: 0,
        name: "",
        blck_size: 0,
        type_size: 0,
    }; SIZED_COUNT];
    let mut n = 0;
    let mut id: u32 = 0;
    while (id as usize) < TRAITS.len() {
        let t = &TRAITS[id as usize];
        if t.blck_size > 0 {
            rows[n] = GgmlTypeTraits {
                id,
                name: t.name,
                blck_size: t.blck_size as usize,
                type_size: t.type_size as usize,
            };
            n += 1;
        }
        id += 1;
    }
    rows
};

/// Every ggml type id with a defined block layout, in id order.
///
/// #3418: derived from `trueno_quant::TRAITS`, the one table that
/// `contracts/ggml-type-v1.yaml` pins to upstream ggml's compiled
/// `type_traits` at the comparator commit. This file carries no sizes of its
/// own, so it cannot drift from upstream (FT-GGMLT-006).
pub const GGML_TYPES: &[GgmlTypeTraits] = &SIZED_ROWS;

/// Layout traits for a ggml type id, or `None` if this table has no size for it.
#[must_use]
pub fn traits(id: u32) -> Option<&'static GgmlTypeTraits> {
    GGML_TYPES.iter().find(|t| t.id == id)
}

/// The ggml name of a type id, including the removed ids that carry no size.
#[must_use]
pub fn name(id: u32) -> Option<&'static str> {
    usize::try_from(id)
        .ok()
        .and_then(|i| TRAITS.get(i))
        .map(|t| t.name)
}

/// A ggml block decoder: whole blocks of one type in, f32 values out. This is
/// ggml's `type_traits[id].to_float`.
pub type ToFloat = fn(&[u8]) -> crate::error::Result<Vec<f32>>;

/// #3418 (FT-GGMLT-017): the `to_float` column of the one table, one row per
/// ggml block type serve can decode. Every site that decodes a block tensor by
/// id reads this instead of keeping its own arms, so no path can refuse a type
/// another path decodes (#3869). Float types (F32/F16/BF16) stay with each site.
const TO_FLOAT: &[(u32, ToFloat)] = {
    use crate::gguf::types::{
        GGUF_TYPE_Q2_K, GGUF_TYPE_Q3_K, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K,
        GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0,
    };
    use crate::quantize::iq2_s::GGML_TYPE_IQ2_S;
    use crate::quantize::iq2_xxs::GGML_TYPE_IQ2_XXS;
    use crate::quantize::iq3_s::GGML_TYPE_IQ3_S;
    use crate::quantize::iq3_xxs::GGML_TYPE_IQ3_XXS;
    use crate::quantize::iq4_nl::GGML_TYPE_IQ4_NL;
    use crate::quantize::iq4_xs::GGML_TYPE_IQ4_XS;
    use crate::quantize::{self as q, dequantize_iq_tensor as iq};
    &[
        (GGUF_TYPE_Q4_0, q::dequantize_q4_0),
        (GGUF_TYPE_Q4_1, q::dequantize_q4_1),
        (GGUF_TYPE_Q5_0, q::dequantize_q5_0),
        (GGUF_TYPE_Q5_1, q::dequantize_q5_1),
        (GGUF_TYPE_Q8_0, q::dequantize_q8_0),
        (GGUF_TYPE_Q2_K, q::dequantize_q2_k),
        (GGUF_TYPE_Q3_K, q::dequantize_q3_k),
        (GGUF_TYPE_Q4_K, q::dequantize_q4_k),
        (GGUF_TYPE_Q5_K, q::dequantize_q5_k),
        (GGUF_TYPE_Q6_K, q::dequantize_q6_k),
        (GGML_TYPE_IQ2_XXS, |d| iq(GGML_TYPE_IQ2_XXS, d)),
        (GGML_TYPE_IQ3_XXS, |d| iq(GGML_TYPE_IQ3_XXS, d)),
        (GGML_TYPE_IQ4_NL, |d| iq(GGML_TYPE_IQ4_NL, d)),
        (GGML_TYPE_IQ3_S, |d| iq(GGML_TYPE_IQ3_S, d)),
        (GGML_TYPE_IQ2_S, |d| iq(GGML_TYPE_IQ2_S, d)),
        (GGML_TYPE_IQ4_XS, |d| iq(GGML_TYPE_IQ4_XS, d)),
    ]
};

/// The block decoder for ggml type `id`, or `None` if serve has none.
#[must_use]
pub fn to_float(id: u32) -> Option<ToFloat> {
    TO_FLOAT.iter().find(|(row, _)| *row == id).map(|(_, f)| *f)
}

/// [`to_float`] by the table's type name (`"Q4_K"`, `"IQ4_XS"`, ...), for
/// sites that carry a dtype string instead of an id.
#[must_use]
pub fn to_float_by_name(name: &str) -> Option<ToFloat> {
    GGML_TYPES
        .iter()
        .find(|t| t.name == name)
        .and_then(|t| to_float(t.id))
}

/// #3418 (FT-GGMLT-025): a 2-D `[out_dim, in_dim]` weight of block type `id`,
/// decoded through its [`to_float`] row. Each GGUF row is padded to whole
/// blocks (GH-202), so a row whose `in_dim` is not a whole number of blocks is
/// decoded on its own and its padding dropped. Returns exactly
/// `in_dim * out_dim` values.
///
/// # Errors
/// `id` has no `to_float` row, `data` is shorter than `out_dim` padded rows,
/// or a block fails to decode.
pub fn to_float_rows(
    id: u32,
    data: &[u8],
    in_dim: usize,
    out_dim: usize,
) -> crate::error::Result<Vec<f32>> {
    let decode = to_float(id).ok_or_else(|| crate::error::RealizarError::UnsupportedOperation {
        operation: "to_float_rows".to_string(),
        reason: format!("no to_float row for ggml type {id}"),
    })?;
    to_float_rows_with(id, decode, data, in_dim, out_dim)
}

/// [`to_float_rows`] with the caller's `decode` for `id` (a SIMD decoder that
/// agrees with the column), the block size and bytes still from the table.
///
/// # Errors
/// As [`to_float_rows`].
pub fn to_float_rows_with(
    id: u32,
    decode: ToFloat,
    data: &[u8],
    in_dim: usize,
    out_dim: usize,
) -> crate::error::Result<Vec<f32>> {
    let t = traits(id).ok_or_else(|| crate::error::RealizarError::UnsupportedOperation {
        operation: "to_float_rows".to_string(),
        reason: format!("ggml type {id} is not a sized block type"),
    })?;
    let row_bytes = in_dim.div_ceil(t.blck_size) * t.type_size;
    let Some(data) = data.get(..row_bytes * out_dim) else {
        return Err(crate::error::RealizarError::FormatError {
            reason: format!(
                "{} weight [{out_dim}, {in_dim}] needs {} bytes, has {}",
                t.name,
                row_bytes * out_dim,
                data.len()
            ),
        });
    };
    if in_dim.is_multiple_of(t.blck_size) {
        return decode(data);
    }
    let mut output = Vec::with_capacity(in_dim * out_dim);
    for row in data.chunks_exact(row_bytes) {
        output.extend_from_slice(&decode(row)?[..in_dim]);
    }
    Ok(output)
}

/// #3418 (FT-GGMLT-018): where each scaled-block type keeps its f16 scale
/// field(s), as byte offsets inside one block. The load gate's non-finite
/// scale scan (PMAT-895) reads this instead of its own match. The tests tie
/// every row to its [`to_float`] decoder: zero scales decode to zeros, and a
/// NaN at any listed offset reaches the output.
const SCALES: &[(u32, &[usize])] = {
    use crate::gguf::types::{
        GGUF_TYPE_Q2_K, GGUF_TYPE_Q3_K, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K,
        GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0,
    };
    &[
        (GGUF_TYPE_Q4_0, &[0]),
        (GGUF_TYPE_Q4_1, &[0, 2]),
        (GGUF_TYPE_Q5_0, &[0]),
        (GGUF_TYPE_Q5_1, &[0, 2]),
        (GGUF_TYPE_Q8_0, &[0]),
        (GGUF_TYPE_Q2_K, &[80, 82]), // d, dmin (trailing)
        (GGUF_TYPE_Q3_K, &[108]),    // d_all (trailing)
        (GGUF_TYPE_Q4_K, &[0, 2]),   // d, dmin
        (GGUF_TYPE_Q5_K, &[0, 2]),   // d, dmin
        (GGUF_TYPE_Q6_K, &[208]),    // d (trailing)
    ]
};

/// The f16 scale offsets inside one block of ggml type `id`, or `None` for
/// a type this table has no scale row for.
#[must_use]
pub fn scale_offsets(id: u32) -> Option<&'static [usize]> {
    SCALES.iter().find(|(row, _)| *row == id).map(|(_, o)| *o)
}

/// A ggml fused matvec: one row of f32 activations against a whole block
/// weight of one type, written into `output[..out_dim]`. Arguments are
/// (weight, activations, in_dim, out_dim, output).
pub type MatvecInto = fn(&[u8], &[f32], usize, usize, &mut [f32]) -> crate::error::Result<()>;

/// #3418 (FT-GGMLT-020): the fused matvec column of the one table, one row per
/// ggml block type serve has a fused (quantized-activation) kernel for. The
/// single-token matmul and the K-quant matcher read this instead of keeping
/// per-id matches; every row is the kernel those matches already called, so
/// the switch is bitwise. A type with no row takes the dequantize fallback.
const MATVEC_INTO: &[(u32, MatvecInto)] = {
    use crate::gguf::types::{
        GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0,
    };
    use crate::quantize as q;
    &[
        // The Q4_0 kernel takes its out_dim from the output slice.
        (GGUF_TYPE_Q4_0, |w, a, k, _, out| {
            q::fused_q4_0_q8_0_parallel_matvec_into(w, a, k, out)
        }),
        (GGUF_TYPE_Q8_0, q::fused_q8_0_q8_0_parallel_matvec_into),
        (GGUF_TYPE_Q4_K, q::fused_q4k_parallel_matvec_into),
        (GGUF_TYPE_Q5_K, q::fused_q5k_parallel_matvec_into),
        (GGUF_TYPE_Q6_K, q::fused_q6k_parallel_matvec_into),
    ]
};

/// The fused matvec kernel for ggml type `id`, or `None` if serve has none.
#[must_use]
pub fn matvec_into(id: u32) -> Option<MatvecInto> {
    MATVEC_INTO
        .iter()
        .find(|(row, _)| *row == id)
        .map(|(_, f)| *f)
}

/// A ggml block decoder that writes into a caller's buffer: whole blocks of
/// one type in, `out` filled, nothing allocated.
pub type ToFloatInto = fn(&[u8], &mut [f32]) -> crate::error::Result<()>;

/// #3418 (FT-GGMLT-023): the in-place `to_float` column, one row per ggml
/// block type serve has an in-place decoder for. Each row is the body its
/// [`TO_FLOAT`] row wraps, so the two decode bit for bit. A per-lookup row
/// decode (the Qwen3.5 embedding, #4991) reads this instead of its own arms;
/// a type with no row goes through [`to_float`] and a copy.
const TO_FLOAT_INTO: &[(u32, ToFloatInto)] = {
    use crate::gguf::types::{GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0};
    use crate::quantize::dequant as q;
    &[
        (GGUF_TYPE_Q8_0, q::dequantize_q8_0_into),
        (GGUF_TYPE_Q4_K, q::dequantize_q4_k_into),
        (GGUF_TYPE_Q6_K, q::dequantize_q6_k_into),
    ]
};

/// The in-place block decoder for ggml type `id`, or `None` if serve has none.
#[must_use]
pub fn to_float_into(id: u32) -> Option<ToFloatInto> {
    TO_FLOAT_INTO
        .iter()
        .find(|(row, _)| *row == id)
        .map(|(_, f)| *f)
}

#[cfg(test)]
mod to_float_into_tests_3418 {
    use super::{to_float, to_float_into, to_float_tests_3418::blocks, GGML_TYPES};

    /// FT-GGMLT-023: every in-place row decodes exactly as the same type's
    /// to_float row, and every type with an in-place row has a to_float row.
    #[test]
    fn every_into_row_agrees_with_its_to_float_row_3418() {
        let mut rows = 0;
        for t in GGML_TYPES {
            let Some(into) = to_float_into(t.id) else {
                continue;
            };
            rows += 1;
            let data = blocks(t.id, 2);
            let want = to_float(t.id).expect("an into row has a to_float row")(&data)
                .expect("whole blocks");
            let mut got = vec![f32::NAN; want.len()];
            into(&data, &mut got).expect("whole blocks");
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(g, w)| g.to_bits() == w.to_bits()),
                "{}: in-place decode differs from the to_float row",
                t.name
            );
        }
        assert_eq!(rows, 3, "Q8_0, Q4_K and Q6_K have in-place rows");
    }

    /// FT-GGMLT-025: rows 1.5 blocks wide keep each row's own in_dim values;
    /// whole-block rows decode as one flat run; short data is refused.
    #[test]
    fn to_float_rows_keeps_each_rows_values_3418() {
        for t in GGML_TYPES.iter().filter(|t| to_float(t.id).is_some()) {
            let decode = to_float(t.id).expect("row");
            let data = blocks(t.id, 6);
            let in_dim = t.blck_size + t.blck_size / 2;
            let want: Vec<u32> = data
                .chunks_exact(2 * t.type_size)
                .flat_map(|row| decode(row).expect("decode")[..in_dim].to_vec())
                .map(f32::to_bits)
                .collect();
            let got: Vec<u32> = super::to_float_rows(t.id, &data, in_dim, 3)
                .expect("padded rows")
                .into_iter()
                .map(f32::to_bits)
                .collect();
            assert_eq!(got, want, "{} padded rows", t.name);

            let flat = decode(&data).expect("decode");
            let whole = super::to_float_rows(t.id, &data, 2 * t.blck_size, 3).expect("whole");
            assert!(
                whole
                    .iter()
                    .zip(&flat)
                    .all(|(g, w)| g.to_bits() == w.to_bits())
                    && whole.len() == flat.len(),
                "{} whole-block rows",
                t.name
            );
            assert!(
                super::to_float_rows(t.id, &data, in_dim, 4).is_err(),
                "{} short data accepted",
                t.name
            );
        }
    }
}

/// The refusal text for an id with no size in this table. Every id that
/// [`TRAITS`] names but does not size is one upstream removed.
fn unsupported(id: u32) -> String {
    match name(id) {
        Some(known) => format!("Unsupported quantization type: {id} ({known}, removed from ggml)"),
        None => format!("Unsupported quantization type: {id}"),
    }
}

/// Bytes a tensor of ggml type `id` with shape `dims` occupies in a GGUF file.
///
/// A 2-D tensor pads EACH ROW to a whole block, which is what the file layout
/// does and what the loader's `k_quant_bytes` has always done; 1-D (and higher)
/// shapes are sized from the flat element count. All arithmetic is checked, so
/// a corrupt header claiming absurd dims returns `Err` instead of wrapping into
/// a small, plausible, wrong byte count.
///
/// # Errors
///
/// Returns the refusal text when `id` has no layout here, or when the size
/// overflows `usize`.
pub fn byte_size(id: u32, dims: &[u64]) -> Result<usize, String> {
    let t = traits(id).ok_or_else(|| unsupported(id))?;
    if dims.len() == 2 {
        let rows = usize::try_from(dims[0]).map_err(|_| overflow(id, dims))?;
        let cols = usize::try_from(dims[1]).map_err(|_| overflow(id, dims))?;
        return cols
            .div_ceil(t.blck_size)
            .checked_mul(t.type_size)
            .and_then(|row_bytes| row_bytes.checked_mul(rows))
            .ok_or_else(|| overflow(id, dims));
    }
    let mut n: usize = 1;
    for &d in dims {
        let d = usize::try_from(d).map_err(|_| overflow(id, dims))?;
        n = n.checked_mul(d).ok_or_else(|| overflow(id, dims))?;
    }
    flat_byte_size(id, n)
}

/// Bytes for `num_elements` elements of type `id`, ignoring row structure.
///
/// This is the pre-#3432 sizing of the legacy arms (Q4_0/Q4_1/Q5_0/Q8_0/Q2_K
/// and the scalar types) — kept as its own entry point so the loader can
/// preserve those numbers exactly rather than silently re-sizing every model.
///
/// # Errors
///
/// Returns the refusal text when `id` has no layout here, or on overflow.
pub fn flat_byte_size(id: u32, num_elements: usize) -> Result<usize, String> {
    let t = traits(id).ok_or_else(|| unsupported(id))?;
    num_elements
        .div_ceil(t.blck_size)
        .checked_mul(t.type_size)
        .ok_or_else(|| {
            format!(
                "Tensor of ggml type {} with {num_elements} elements overflows usize",
                t.name
            )
        })
}

/// Overflow refusal naming the shape that produced it.
fn overflow(id: u32, dims: &[u64]) -> String {
    format!("Tensor of ggml type {id} with dims {dims:?} overflows usize")
}

#[cfg(test)]
mod ggml_type_table_tests {
    use super::*;

    /// A duplicate id would make `traits()` return whichever row came first —
    /// a silently wrong size for one of the two types.
    #[test]
    fn ids_are_unique_and_ascending() {
        let mut previous = None;
        for t in GGML_TYPES {
            if let Some(prev) = previous {
                assert!(
                    t.id > prev,
                    "table must be unique and ascending by id: {} after {prev}",
                    t.id
                );
            }
            previous = Some(t.id);
        }
    }

    /// A zero in either column would produce a zero-byte tensor (or a division
    /// by zero) rather than a refusal.
    #[test]
    fn every_row_has_a_nonzero_block_and_size() {
        for t in GGML_TYPES {
            assert!(t.blck_size > 0, "{} has blck_size 0", t.name);
            assert!(t.type_size > 0, "{} has type_size 0", t.name);
            assert!(!t.name.is_empty(), "id {} has no name", t.id);
        }
    }

    /// The rows this repo's existing kernels already agree on — if a transcription
    /// slip changed one of these, every model of that type would mis-load.
    #[test]
    fn spot_checks_against_ggml_sizes() {
        let cases: [(u32, &str, usize, usize); 8] = [
            (0, "F32", 1, 4),
            (1, "F16", 1, 2),
            (2, "Q4_0", 32, 18),
            (12, "Q4_K", 256, 144),
            (14, "Q6_K", 256, 210),
            (16, "IQ2_XXS", 256, 66),
            (23, "IQ4_XS", 256, 136),
            (30, "BF16", 1, 2),
        ];
        for (id, name, blck, size) in cases {
            let t = traits(id).expect("id must be in the table");
            assert_eq!(t.name, name, "id {id} name");
            assert_eq!(t.blck_size, blck, "{name} block size");
            assert_eq!(t.type_size, size, "{name} type size");
        }
    }

    /// 2-D tensors pad each ROW, which is the file layout — sizing them from the
    /// flat element count under-counts whenever a row is not a whole block.
    #[test]
    fn two_d_tensors_pad_each_row() {
        // 4 rows x 300 cols of Q4_K: each row is 2 super-blocks (300 -> 512).
        assert_eq!(byte_size(12, &[4, 300]).expect("Q4_K"), 4 * 2 * 144);
        // Aligned rows: row padding and flat sizing agree.
        assert_eq!(byte_size(12, &[4, 512]).expect("Q4_K"), 4 * 2 * 144);
    }

    #[test]
    fn one_d_tensors_are_sized_from_the_element_count() {
        assert_eq!(byte_size(0, &[1000]).expect("F32"), 4000);
        assert_eq!(byte_size(16, &[512]).expect("IQ2_XXS"), 2 * 66);
        assert_eq!(byte_size(23, &[512]).expect("IQ4_XS"), 2 * 136);
    }

    /// An id outside the table must refuse, naming the id — never fall back to
    /// a "close enough" size.
    #[test]
    fn an_unknown_id_is_refused_by_id() {
        let err = byte_size(4242, &[16]).expect_err("4242 is not a ggml type");
        assert!(err.contains("4242"), "got: {err}");
        assert!(err.contains("Unsupported quantization type"), "got: {err}");
        assert_eq!(name(4242), None);
    }

    /// A removed/unverified id is refused too, but by NAME — the message is the
    /// only thing that tells a user their file uses a repacked legacy type.
    #[test]
    fn a_named_but_unsized_id_is_refused_by_name() {
        let err = byte_size(32, &[16]).expect_err("Q4_0_4_8 carries no size here");
        assert!(err.contains("Q4_0_4_8"), "got: {err}");
        assert!(traits(32).is_none(), "32 must not be sized");
        assert!(err.contains("removed from ggml"), "got: {err}");
        // #3418: TRAITS sizes MXFP4 from upstream, so serve does too.
        assert_eq!(
            traits(39).map(|t| (t.blck_size, t.type_size)),
            Some((32, 17))
        );
    }

    /// Checked arithmetic: a corrupt header must produce an error, not a small
    /// wrapped byte count that then reads inside the file.
    #[test]
    fn absurd_dims_overflow_into_an_error() {
        let err = byte_size(12, &[u64::MAX, u64::MAX]).expect_err("must not wrap");
        assert!(err.contains("overflows usize"), "got: {err}");
        // Q8_K: 292 bytes per 256 elements, the widest row in the table — the
        // only one whose flat size overflows at `usize::MAX` elements.
        let err = flat_byte_size(15, usize::MAX).expect_err("must not wrap");
        assert!(err.contains("overflows usize"), "got: {err}");
    }

    /// #3418: serve's table against the one TRAITS table (trueno_quant), id by id.
    #[test]
    fn serve_table_matches_trueno_quant_traits() {
        let mut diffs = Vec::new();
        for (id, q) in trueno_quant::TRAITS.iter().enumerate() {
            let id = u32::try_from(id).expect("43 ids fit u32");
            let sized = q.blck_size > 0;
            match traits(id) {
                Some(t) if !sized => {
                    diffs.push(format!("{id} {}: serve sizes a removed id", t.name));
                },
                Some(t) => {
                    if t.name != q.name
                        || t.blck_size != q.blck_size as usize
                        || t.type_size != q.type_size as usize
                    {
                        diffs.push(format!(
                            "{id}: serve ({}, {}, {}) vs TRAITS ({}, {}, {})",
                            t.name, t.blck_size, t.type_size, q.name, q.blck_size, q.type_size
                        ));
                    }
                },
                None if sized => diffs.push(format!(
                    "{id} {}: TRAITS sizes it, serve does not ({:?})",
                    q.name,
                    name(id)
                )),
                None => {},
            }
        }
        for t in GGML_TYPES {
            if t.id as usize >= trueno_quant::TRAITS.len() {
                diffs.push(format!(
                    "{} {}: serve sizes an id TRAITS lacks",
                    t.id, t.name
                ));
            }
        }
        assert!(
            diffs.is_empty(),
            "{} rows differ:\n{}",
            diffs.len(),
            diffs.join("\n")
        );
    }
}

#[cfg(test)]
pub(crate) mod to_float_tests_3418 {
    use super::{to_float, traits, TO_FLOAT};

    /// `k` whole blocks of type `id` with a 1.0 f16 scale at the head of
    /// each block and a fixed byte pattern after it.
    pub(crate) fn blocks(id: u32, k: usize) -> Vec<u8> {
        let t = traits(id).expect("a to_float row is a sized ggml type");
        let mut data: Vec<u8> = (0..k * t.type_size)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect();
        for b in data.chunks_exact_mut(t.type_size) {
            b[..2].copy_from_slice(&[0x00, 0x3C]);
        }
        data
    }

    #[test]
    fn every_to_float_row_decodes_whole_blocks_3418() {
        for &(id, decode) in TO_FLOAT {
            let t = traits(id).expect("a to_float row is a sized ggml type");
            assert!(t.blck_size > 1, "{} ({id}) is not a block type", t.name);
            let got = decode(&blocks(id, 3)).unwrap_or_else(|e| panic!("{} ({id}): {e}", t.name));
            assert_eq!(
                got.len(),
                3 * t.blck_size,
                "{} ({id}) decoded to the wrong length",
                t.name
            );
        }
    }

    #[test]
    fn to_float_k_quants_match_the_quant_crate_3418() {
        let independent: [(u32, fn(&[u8], usize) -> Vec<f32>); 3] = [
            (12, trueno_quant::dequantize_q4_k_to_f32),
            (13, trueno_quant::dequantize_q5_k_to_f32),
            (14, trueno_quant::dequantize_q6_k_to_f32),
        ];
        for (id, reference) in independent {
            let data = blocks(id, 2);
            let got = to_float(id).expect("K-quant row")(&data).expect("whole blocks");
            assert_eq!(
                got,
                reference(&data, 512),
                "ggml id {id} differs from aprender-quant"
            );
        }
    }

    #[test]
    fn to_float_iq_rows_pass_their_own_id_3418() {
        for &(id, decode) in TO_FLOAT {
            if crate::quantize::iq_block_bytes(id).is_none() {
                continue;
            }
            let data = blocks(id, 2);
            assert_eq!(
                decode(&data).expect("whole IQ blocks"),
                crate::quantize::dequantize_iq_tensor(id, &data).expect("whole IQ blocks"),
                "IQ row {id} does not decode as IQ type {id}"
            );
        }
    }
}

#[cfg(test)]
mod scale_tests_3418 {
    use super::to_float_tests_3418::blocks;
    use super::{scale_offsets, to_float, traits, SCALES};

    /// f16 bit patterns.
    const ZERO: [u8; 2] = [0x00, 0x00];
    const NAN: [u8; 2] = [0x00, 0x7E];

    fn decode(id: u32, data: &[u8]) -> Vec<f32> {
        to_float(id).expect("a scale row has a to_float row")(data).expect("whole blocks")
    }

    #[test]
    fn scale_rows_are_the_decoders_scales_3418() {
        for &(id, offsets) in SCALES {
            let t = traits(id).expect("a scale row is a sized ggml type");
            let mut zeroed = blocks(id, 2);
            for b in zeroed.chunks_exact_mut(t.type_size) {
                for &off in offsets {
                    assert!(off + 2 <= t.type_size, "{} scale at {off}", t.name);
                    b[off..off + 2].copy_from_slice(&ZERO);
                }
            }
            assert!(
                decode(id, &zeroed).iter().all(|v| *v == 0.0),
                "{} ({id}): zeroing {offsets:?} leaves a non-zero value, so a scale sits elsewhere",
                t.name
            );
            for &off in offsets {
                let mut poisoned = blocks(id, 2);
                poisoned[off..off + 2].copy_from_slice(&NAN);
                assert!(
                    decode(id, &poisoned).iter().any(|v| v.is_nan()),
                    "{} ({id}): a NaN at {off} does not reach the output, so {off} is not a scale",
                    t.name
                );
            }
        }
    }

    #[test]
    fn scale_rows_cover_the_ten_scaled_ids_3418() {
        const SCALED: [u32; 10] = [2, 3, 6, 7, 8, 10, 11, 12, 13, 14];
        for id in 0..64u32 {
            assert_eq!(
                scale_offsets(id).is_some(),
                SCALED.contains(&id),
                "ggml id {id} scale row"
            );
        }
    }
}

#[cfg(test)]
mod matvec_tests_3418 {
    use super::to_float_tests_3418::blocks;
    use super::{matvec_into, scale_offsets, to_float, traits, MATVEC_INTO};

    /// One row of `in_dim` weights of type `id` per output, every f16 scale
    /// set to 1/16 so the decode is finite and small.
    fn weight(id: u32, in_dim: usize, out_dim: usize) -> Vec<u8> {
        let t = traits(id).expect("a matvec row is a sized ggml type");
        let mut data = blocks(id, out_dim * in_dim / t.blck_size);
        for b in data.chunks_exact_mut(t.type_size) {
            for &off in scale_offsets(id).unwrap_or(&[0]) {
                b[off..off + 2].copy_from_slice(&[0x00, 0x2C]);
            }
        }
        data
    }

    #[test]
    fn every_matvec_row_agrees_with_its_to_float_row_3418() {
        let (in_dim, out_dim) = (512, 6);
        let x: Vec<f32> = (0..in_dim)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.0)
            .collect();
        for &(id, matvec) in MATVEC_INTO {
            let name = traits(id).expect("sized").name;
            let w = weight(id, in_dim, out_dim);
            let decoded = to_float(id).expect("a matvec row has a to_float row")(&w)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut got = vec![f32::NAN; out_dim];
            matvec(&w, &x, in_dim, out_dim, &mut got).unwrap_or_else(|e| panic!("{name}: {e}"));
            for (o, row) in decoded.chunks_exact(in_dim).enumerate() {
                let exact: f64 = row
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                let scale: f64 = row
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| f64::from((a * b).abs()))
                    .sum();
                assert!(
                    (f64::from(got[o]) - exact).abs() <= 0.02 * scale + 1e-3,
                    "{name} ({id}) row {o}: matvec {} vs to_float dot {exact}",
                    got[o]
                );
            }
        }
    }

    #[test]
    fn matvec_rows_are_the_five_fused_kernels_3418() {
        const FUSED: [u32; 5] = [2, 8, 12, 13, 14];
        for id in 0..64u32 {
            assert_eq!(
                matvec_into(id).is_some(),
                FUSED.contains(&id),
                "ggml id {id} matvec row"
            );
        }
    }
}
