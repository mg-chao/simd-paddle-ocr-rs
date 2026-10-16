//! Portable SIMD arithmetic with guarded, per-function CPU specialization.
//!
//! Scalar, Portable4, and AVX retain separate F32 multiply/add operations.
//! FMA-capable policies deliberately fuse eligible accumulations. Runtime
//! feature detection guards every target-feature entry, including forced
//! policies; AVX-512 selection remains operator- and shape-dependent.

use super::packing::{MATRIX_BLOCK_COLUMNS, PackedMatrix};
use crate::math::exp_f32;
use serde::{Deserialize, Serialize};
use std::simd::{Select, Simd, StdFloat, cmp::SimdPartialOrd, num::SimdFloat};
mod argmax;
mod columns;
pub(crate) mod matrix;
pub(crate) mod parallel;
pub(crate) use columns::{ArgMaxWorkspace, try_compact_columns};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelPolicy {
    #[default]
    Auto,
    Scalar,
    Portable4,
    Avx,
    AvxFma,
    Avx2Fma,
    Avx512,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Scalar,
    Portable4,
    Avx,
    AvxFma,
    Avx2Fma,
    Avx512,
}

impl Backend {
    #[cfg(test)]
    pub(crate) fn uses_fma(self) -> bool {
        matches!(self, Self::AvxFma | Self::Avx2Fma | Self::Avx512)
    }
}

/// Transform an output accumulator before its first store. Indices refer to
/// the complete physical output, even when a worker owns only one band.
/// Implementations borrow their operands; no intermediate tensor is required.
pub(crate) trait StoreTransform: Sync {
    fn scalar(&self, value: f32, index: usize) -> f32;
    fn vector<const LANES: usize>(&self, value: Simd<f32, LANES>, index: usize)
    -> Simd<f32, LANES>;

    /// NHWC kernels already know the first output channel. Keep the complete
    /// output index for residuals while allowing bound channel operands to
    /// avoid recovering it with integer division at every vector store.
    #[inline(always)]
    fn vector_channel<const LANES: usize>(
        &self,
        value: Simd<f32, LANES>,
        index: usize,
        _channel: usize,
    ) -> Simd<f32, LANES> {
        self.vector(value, index)
    }
}

pub(crate) struct IdentityStore;

impl StoreTransform for IdentityStore {
    #[inline(always)]
    fn scalar(&self, value: f32, _index: usize) -> f32 {
        value
    }
    #[inline(always)]
    fn vector<const LANES: usize>(
        &self,
        value: Simd<f32, LANES>,
        _index: usize,
    ) -> Simd<f32, LANES> {
        value
    }
}

/// Bit-preserving baseline 8x8 transpose, including safe partial tiles.
pub(crate) fn transpose_matrix(input: &[f32], output: &mut [f32], rows: usize, columns: usize) {
    assert_eq!(
        input.len(),
        rows.checked_mul(columns).expect("transpose size overflow")
    );
    assert_eq!(input.len(), output.len());
    transpose_columns_impl(input, output, rows, columns, 0);
}

/// Transpose source columns into a contiguous exclusive destination band.
/// The reference's AVX2/FMA and AVX-512 policies use the same eight-lane AVX tile.
pub(crate) fn transpose_matrix_columns(
    backend: Backend,
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    columns: usize,
    first_column: usize,
) {
    assert_eq!(
        input.len(),
        rows.checked_mul(columns).expect("transpose size overflow")
    );
    assert!(first_column <= columns);
    if rows == 0 {
        assert!(output.is_empty());
        return;
    }
    assert!(output.len().is_multiple_of(rows));
    assert!(
        first_column
            .checked_add(output.len() / rows)
            .is_some_and(|last| last <= columns)
    );
    match backend {
        Backend::Avx2Fma | Backend::Avx512 => {
            if backend == Backend::Avx512 {
                checked_avx512();
            } else {
                checked_avx2();
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: the selected policy's runtime guard includes AVX.
            unsafe {
                transpose_columns_avx(input, output, rows, columns, first_column)
            }
        }
        _ => {
            if first_column == 0 && output.len() == input.len() {
                transpose_matrix(input, output, rows, columns);
            } else {
                transpose_columns_impl(input, output, rows, columns, first_column);
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn transpose_columns_avx(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    columns: usize,
    first_column: usize,
) {
    transpose_columns_impl(input, output, rows, columns, first_column);
}

#[inline(always)]
fn transpose_columns_impl(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    columns: usize,
    first_column: usize,
) {
    let count = output.len().checked_div(rows).unwrap_or(0);
    for local_col in (0..count).step_by(8) {
        let col = first_column + local_col;
        let nc = (count - local_col).min(8);
        for row in (0..rows).step_by(8) {
            let nr = (rows - row).min(8);
            if nr == 8 && nc == 8 {
                let tile =
                    std::array::from_fn(|r| Simd::from_slice(&input[(row + r) * columns + col..]));
                for (c, value) in transpose_tile(tile).into_iter().enumerate() {
                    value.copy_to_slice(&mut output[(local_col + c) * rows + row..][..8]);
                }
            } else {
                for r in 0..nr {
                    for c in 0..nc {
                        output[(local_col + c) * rows + row + r] =
                            input[(row + r) * columns + col + c];
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn transpose_tile(r: [Simd<f32, 8>; 8]) -> [Simd<f32, 8>; 8] {
    use std::simd::simd_swizzle;
    let a = std::array::from_fn::<_, 8, _>(|i| {
        if i % 2 == 0 {
            simd_swizzle!(r[i], r[i + 1], [0, 8, 1, 9, 2, 10, 3, 11])
        } else {
            simd_swizzle!(r[i - 1], r[i], [4, 12, 5, 13, 6, 14, 7, 15])
        }
    });
    let b = [
        simd_swizzle!(a[0], a[2], [0, 1, 8, 9, 2, 3, 10, 11]),
        simd_swizzle!(a[0], a[2], [4, 5, 12, 13, 6, 7, 14, 15]),
        simd_swizzle!(a[1], a[3], [0, 1, 8, 9, 2, 3, 10, 11]),
        simd_swizzle!(a[1], a[3], [4, 5, 12, 13, 6, 7, 14, 15]),
        simd_swizzle!(a[4], a[6], [0, 1, 8, 9, 2, 3, 10, 11]),
        simd_swizzle!(a[4], a[6], [4, 5, 12, 13, 6, 7, 14, 15]),
        simd_swizzle!(a[5], a[7], [0, 1, 8, 9, 2, 3, 10, 11]),
        simd_swizzle!(a[5], a[7], [4, 5, 12, 13, 6, 7, 14, 15]),
    ];
    std::array::from_fn(|i| {
        let half = i / 2;
        if i % 2 == 0 {
            simd_swizzle!(b[half], b[half + 4], [0, 1, 2, 3, 8, 9, 10, 11])
        } else {
            simd_swizzle!(b[half], b[half + 4], [4, 5, 6, 7, 12, 13, 14, 15])
        }
    })
}

/// Preserve sequential LayerNorm statistics; vectorize only its independent
/// affine stage with the graph's separate subtraction/division/multiply/add.
/// `None` reads each row from `output` for exact in-place execution.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_norm_affine(
    backend: Backend,
    input: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    mean: f32,
    deviation: f32,
    output: &mut [f32],
) {
    assert!(input.is_none_or(|input| input.len() == output.len()));
    assert!(gamma.len() >= output.len() && beta.len() >= output.len());
    match backend {
        Backend::Scalar => layer_norm_affine_simd::<1>(input, gamma, beta, mean, deviation, output),
        Backend::Portable4 => {
            layer_norm_affine_simd::<4>(input, gamma, beta, mean, deviation, output)
        }
        Backend::Avx2Fma => {
            checked_avx2();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime detection verifies the enabled features.
            unsafe {
                layer_norm_affine_avx2(input, gamma, beta, mean, deviation, output)
            }
        }
        Backend::Avx => {
            checked_avx();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime detection verifies the enabled features.
            unsafe {
                layer_norm_affine_avx(input, gamma, beta, mean, deviation, output)
            }
        }
        Backend::AvxFma => {
            checked_avx_fma();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime detection verifies the enabled features.
            unsafe {
                layer_norm_affine_avx_fma(input, gamma, beta, mean, deviation, output)
            }
        }
        Backend::Avx512 => {
            checked_avx512();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime detection verifies the enabled features.
            unsafe {
                layer_norm_affine_avx512(input, gamma, beta, mean, deviation, output)
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn layer_norm_affine_avx2(
    input: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    mean: f32,
    deviation: f32,
    output: &mut [f32],
) {
    layer_norm_affine_simd::<8>(input, gamma, beta, mean, deviation, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn layer_norm_affine_avx(
    input: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    mean: f32,
    deviation: f32,
    output: &mut [f32],
) {
    layer_norm_affine_simd::<8>(input, gamma, beta, mean, deviation, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn layer_norm_affine_avx_fma(
    input: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    mean: f32,
    deviation: f32,
    output: &mut [f32],
) {
    layer_norm_affine_simd::<8>(input, gamma, beta, mean, deviation, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn layer_norm_affine_avx512(
    input: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    mean: f32,
    deviation: f32,
    output: &mut [f32],
) {
    layer_norm_affine_simd::<16>(input, gamma, beta, mean, deviation, output);
}

#[inline(always)]
fn layer_norm_affine_simd<const LANES: usize>(
    input: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    mean: f32,
    deviation: f32,
    output: &mut [f32],
) {
    let end = output.len() / LANES * LANES;
    for start in (0..end).step_by(LANES) {
        let value = Simd::<f32, LANES>::from_slice(&input.unwrap_or(output)[start..start + LANES]);
        let gamma = Simd::from_slice(&gamma[start..start + LANES]);
        let beta = Simd::from_slice(&beta[start..start + LANES]);
        let value = ((value - Simd::splat(mean)) / Simd::splat(deviation)) * gamma + beta;
        value.copy_to_slice(&mut output[start..start + LANES]);
    }
    for index in end..output.len() {
        output[index] =
            ((input.unwrap_or(output)[index] - mean) / deviation) * gamma[index] + beta[index];
    }
}

pub(crate) fn avx2_fma_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

pub(crate) fn avx_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

pub(crate) fn avx_fma_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        avx_available() && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

pub(crate) fn avx512_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        avx2_fma_available() && std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

pub(crate) fn select_backend(policy: KernelPolicy) -> Result<Backend, &'static str> {
    select_backend_with_features(
        policy,
        avx_available(),
        avx_fma_available(),
        avx2_fma_available(),
        avx512_available(),
    )
}

#[cfg(test)]
fn select_backend_with_avx2(policy: KernelPolicy, avx2_fma: bool) -> Result<Backend, &'static str> {
    select_backend_with_features(policy, avx2_fma, avx2_fma, avx2_fma, false)
}

fn select_backend_with_features(
    policy: KernelPolicy,
    avx: bool,
    avx_fma: bool,
    avx2_fma: bool,
    avx512: bool,
) -> Result<Backend, &'static str> {
    match policy {
        KernelPolicy::Auto => Ok(if avx512 {
            Backend::Avx512
        } else if avx2_fma {
            Backend::Avx2Fma
        } else if avx_fma {
            Backend::AvxFma
        } else if avx {
            Backend::Avx
        } else {
            Backend::Portable4
        }),
        KernelPolicy::Scalar => Ok(Backend::Scalar),
        KernelPolicy::Portable4 => Ok(Backend::Portable4),
        KernelPolicy::Avx if avx => Ok(Backend::Avx),
        KernelPolicy::Avx => Err("AVX kernel policy is unsupported on this CPU"),
        KernelPolicy::AvxFma if avx_fma => Ok(Backend::AvxFma),
        KernelPolicy::AvxFma => Err("AVX/FMA kernel policy is unsupported on this CPU"),
        KernelPolicy::Avx2Fma if avx2_fma => Ok(Backend::Avx2Fma),
        KernelPolicy::Avx2Fma => Err("AVX2/FMA kernel policy is unsupported on this CPU"),
        KernelPolicy::Avx512 if avx512 => Ok(Backend::Avx512),
        KernelPolicy::Avx512 => {
            Err("AVX-512 kernel policy requires AVX-512F, AVX2 and FMA on this CPU")
        }
    }
}

#[inline]
fn checked_avx2() {
    assert!(
        avx2_fma_available(),
        "AVX2/FMA backend requires AVX2 and FMA support"
    );
}

#[inline]
fn checked_avx() {
    assert!(avx_available(), "AVX backend requires AVX support");
}
#[inline]
fn checked_avx_fma() {
    assert!(
        avx_fma_available(),
        "AVX/FMA backend requires AVX and FMA support"
    );
}
#[inline]
fn checked_avx512() {
    assert!(
        avx512_available(),
        "AVX-512 backend requires AVX-512F, AVX2 and FMA support"
    );
}

#[inline(always)]
fn multiply_add<const LANES: usize, const FUSED: bool>(
    a: Simd<f32, LANES>,
    b: Simd<f32, LANES>,
    c: Simd<f32, LANES>,
) -> Simd<f32, LANES> {
    if FUSED { a.mul_add(b, c) } else { a * b + c }
}

#[inline(always)]
fn store_partial<const LANES: usize>(value: Simd<f32, LANES>, destination: &mut [f32]) {
    if destination.len() == LANES {
        value.copy_to_slice(destination);
    } else {
        destination.copy_from_slice(&value.to_array()[..destination.len()]);
    }
}

fn matrix_lengths(
    input: &[f32],
    weights: &[f32],
    output_len: usize,
    rows: usize,
    inner: usize,
    columns: usize,
) {
    assert!(
        inner > 0 && columns > 0,
        "matrix dimensions must be positive"
    );
    assert_eq!(
        input.len(),
        rows.checked_mul(inner).expect("matrix size overflow")
    );
    assert_eq!(
        weights.len(),
        inner.checked_mul(columns).expect("matrix size overflow")
    );
    assert_eq!(
        output_len,
        rows.checked_mul(columns).expect("matrix size overflow")
    );
}

/// Row-major `[rows,inner] × [inner,columns]`. A compatible shared panel is
/// optional: dynamic attention matrices use the same SIMD implementation
/// directly without a temporary packing allocation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul(
    backend: Backend,
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    matrix_lengths(input, weights, output.len(), rows, inner, columns);
    if let Some(panel) = packed {
        assert!(panel.matches(inner, columns), "matrix panel shape mismatch");
    }
    if matrix::try_dynamic(
        backend,
        input,
        weights,
        packed.is_some(),
        output,
        rows,
        inner,
        columns,
    ) {
        return;
    }
    match backend {
        Backend::Scalar => scalar_matmul(input, weights, output, rows, inner, columns),
        Backend::Portable4 => {
            matmul_simd::<4, 4, false>(input, weights, packed, output, rows, inner, columns)
        }
        Backend::Avx2Fma => {
            checked_avx2();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: the runtime check above verifies this branch's enabled
            // features; safe slices validate complete matrix lengths.
            unsafe {
                matmul_avx2(input, weights, packed, output, rows, inner, columns)
            }
        }
        Backend::Avx => {
            checked_avx();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: the runtime check above verifies this branch's enabled
            // features; safe slices validate complete matrix lengths.
            unsafe {
                matmul_avx(input, weights, packed, output, rows, inner, columns)
            }
        }
        Backend::AvxFma => {
            checked_avx_fma();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: the runtime check above verifies this branch's enabled
            // features; safe slices validate complete matrix lengths.
            unsafe {
                matmul_avx_fma(input, weights, packed, output, rows, inner, columns)
            }
        }
        Backend::Avx512 => {
            checked_avx512();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: the runtime check above verifies this branch's enabled
            // features; safe slices validate complete matrix lengths.
            unsafe {
                matmul_avx512(input, weights, packed, output, rows, inner, columns)
            }
        }
    }
}

/// Bias-first dense output using the same four-row/sixteen-column core.
/// With a panel, `weights` may be empty: every backend, including Scalar,
/// reads `[column_block][inner][16]` directly. Without a panel, coefficients
/// must be row-major `[inner][columns]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_bias(
    backend: Backend,
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    bias: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    matmul_bias_store(
        backend,
        input,
        weights,
        packed,
        bias,
        output,
        rows,
        inner,
        columns,
        &IdentityStore,
        0,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_bias_store(
    backend: Backend,
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    bias: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
    offset: usize,
) {
    assert!(inner > 0 && columns > 0);
    assert_eq!(
        input.len(),
        rows.checked_mul(inner).expect("matrix size overflow")
    );
    assert_eq!(
        output.len(),
        rows.checked_mul(columns).expect("matrix size overflow")
    );
    let weight_count = inner.checked_mul(columns).expect("matrix size overflow");
    assert!(bias.is_empty() || bias.len() == columns);
    if let Some(panel) = packed {
        assert!(panel.matches(inner, columns), "matrix panel shape mismatch");
        assert!(weights.is_empty() || weights.len() == weight_count);
    } else {
        assert_eq!(weights.len(), weight_count);
    }
    match backend {
        Backend::Scalar => {
            for row in 0..rows {
                for col in 0..columns {
                    let mut sum = if bias.is_empty() { 0.0 } else { bias[col] };
                    for k in 0..inner {
                        let weight = if let Some(panel) = packed {
                            panel.data()[((col / 16) * inner + k) * 16 + col % 16]
                        } else {
                            weights[k * columns + col]
                        };
                        sum += input[row * inner + k] * weight;
                    }
                    output[row * columns + col] = store.scalar(sum, offset + row * columns + col);
                }
            }
        }
        Backend::Portable4 => matmul_bias_simd::<4, 4, false>(
            input, weights, packed, bias, output, rows, inner, columns, store, offset,
        ),
        Backend::Avx2Fma => {
            checked_avx2();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime CPU detection guards both enabled features.
            unsafe {
                matmul_bias_avx2(
                    input, weights, packed, bias, output, rows, inner, columns, store, offset,
                )
            }
        }
        Backend::Avx => {
            checked_avx();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime CPU detection guards both enabled features.
            unsafe {
                matmul_bias_avx(
                    input, weights, packed, bias, output, rows, inner, columns, store, offset,
                )
            }
        }
        Backend::AvxFma => {
            checked_avx_fma();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime CPU detection guards both enabled features.
            unsafe {
                matmul_bias_avx_fma(
                    input, weights, packed, bias, output, rows, inner, columns, store, offset,
                )
            }
        }
        Backend::Avx512 => {
            checked_avx512();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime CPU detection guards both enabled features.
            unsafe {
                matmul_bias_avx512(
                    input, weights, packed, bias, output, rows, inner, columns, store, offset,
                )
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_bias_avx2(
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    bias: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
    offset: usize,
) {
    matmul_bias_simd::<8, 2, true>(
        input, weights, packed, bias, output, rows, inner, columns, store, offset,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_bias_avx(
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    bias: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
    offset: usize,
) {
    matmul_bias_simd::<8, 2, false>(
        input, weights, packed, bias, output, rows, inner, columns, store, offset,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_bias_avx_fma(
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    bias: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
    offset: usize,
) {
    matmul_bias_simd::<8, 2, true>(
        input, weights, packed, bias, output, rows, inner, columns, store, offset,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_bias_avx512(
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    bias: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
    offset: usize,
) {
    matmul_bias_simd::<16, 1, true>(
        input, weights, packed, bias, output, rows, inner, columns, store, offset,
    );
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn matmul_bias_simd<const LANES: usize, const VECTORS: usize, const FUSED: bool>(
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    bias: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
    offset: usize,
) {
    if let Some(panel) = packed {
        matmul_simd_impl::<LANES, VECTORS, FUSED, true>(
            input,
            panel.data(),
            bias,
            output,
            rows,
            inner,
            columns,
            store,
            offset,
        );
    } else {
        matmul_simd_impl::<LANES, VECTORS, FUSED, false>(
            input, weights, bias, output, rows, inner, columns, store, offset,
        );
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_avx2(
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    matmul_simd::<8, 2, true>(input, weights, packed, output, rows, inner, columns);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_avx(
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    matmul_simd::<8, 2, false>(input, weights, packed, output, rows, inner, columns);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_avx_fma(
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    matmul_simd::<8, 2, true>(input, weights, packed, output, rows, inner, columns);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_avx512(
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    if wide_projection_eligible(rows, inner, columns, packed.is_some()) {
        matmul_simd::<16, 1, true>(input, weights, packed, output, rows, inner, columns);
    } else {
        // The reference deliberately keeps attention and light projections
        // on AVX; broad AVX-512 MatMul is not its production selection rule.
        matmul_simd::<8, 2, true>(input, weights, packed, output, rows, inner, columns);
    }
}

pub(crate) fn wide_projection_eligible(
    rows: usize,
    inner: usize,
    columns: usize,
    packed: bool,
) -> bool {
    packed && rows >= 8 && rows.is_multiple_of(8) && inner >= 64 && columns >= 1024
}

fn scalar_matmul(
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    // Share the production reference's four-row/four-column implementation,
    // both to keep fallback performance and to measure an honest baseline.
    super::kernels::matmul(input, weights, output, rows, inner, columns);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn projection_tile<
    const LANES: usize,
    const VECTORS: usize,
    const FUSED: bool,
    const ROWS: usize,
    const PACKED: bool,
>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    first_row: usize,
    block: usize,
    inner: usize,
    columns: usize,
) -> [[Simd<f32, LANES>; VECTORS]; ROWS] {
    let first_column = block * MATRIX_BLOCK_COLUMNS;
    // Establish each row's bounds before K. Indexing k in these exact-length
    // slices lets LLVM remove the repeated flattened input bounds checks.
    let input_rows: [&[f32]; ROWS] =
        std::array::from_fn(|row| &input[(first_row + row) * inner..(first_row + row + 1) * inner]);
    let packed_panel = if PACKED {
        &weights[block * inner * MATRIX_BLOCK_COLUMNS..(block + 1) * inner * MATRIX_BLOCK_COLUMNS]
    } else {
        &[]
    };
    let mut sums = [[Simd::splat(0.0); VECTORS]; ROWS];
    if !bias.is_empty() {
        for vector in 0..VECTORS {
            let column = first_column + vector * LANES;
            let value = if column < columns {
                Simd::load_or_default(&bias[column..(column + LANES).min(columns)])
            } else {
                Simd::splat(0.0)
            };
            for row in &mut sums {
                row[vector] = value;
            }
        }
    }
    for k in 0..inner {
        let weight_row = if PACKED {
            &packed_panel[k * MATRIX_BLOCK_COLUMNS..(k + 1) * MATRIX_BLOCK_COLUMNS]
        } else {
            &weights[k * columns..(k + 1) * columns]
        };
        for vector in 0..VECTORS {
            let weight = if PACKED {
                Simd::from_slice(&weight_row[vector * LANES..(vector + 1) * LANES])
            } else {
                let column = first_column + vector * LANES;
                if column < columns {
                    Simd::load_or_default(&weight_row[column..(column + LANES).min(columns)])
                } else {
                    Simd::splat(0.0)
                }
            };
            for row in 0..ROWS {
                let value = Simd::splat(input_rows[row][k]);
                sums[row][vector] = multiply_add::<LANES, FUSED>(value, weight, sums[row][vector]);
            }
        }
    }
    sums
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn matmul_simd<const LANES: usize, const VECTORS: usize, const FUSED: bool>(
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    if let Some(panel) = packed {
        matmul_simd_impl::<LANES, VECTORS, FUSED, true>(
            input,
            panel.data(),
            &[],
            output,
            rows,
            inner,
            columns,
            &IdentityStore,
            0,
        );
    } else {
        matmul_simd_impl::<LANES, VECTORS, FUSED, false>(
            input,
            weights,
            &[],
            output,
            rows,
            inner,
            columns,
            &IdentityStore,
            0,
        );
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn matmul_simd_impl<
    const LANES: usize,
    const VECTORS: usize,
    const FUSED: bool,
    const PACKED: bool,
>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
    offset: usize,
) {
    let wide = wide_projection_eligible(rows, inner, columns, PACKED);
    let mut row = 0;
    if wide && LANES == 16 {
        while row + 16 <= rows {
            matmul_rows::<LANES, VECTORS, FUSED, PACKED, 16>(
                input, weights, bias, output, row, inner, columns, store, offset,
            );
            row += 16;
        }
    }
    if wide {
        while row + 8 <= rows {
            matmul_rows::<LANES, VECTORS, FUSED, PACKED, 8>(
                input, weights, bias, output, row, inner, columns, store, offset,
            );
            row += 8;
        }
    }
    while row + 4 <= rows {
        matmul_rows::<LANES, VECTORS, FUSED, PACKED, 4>(
            input, weights, bias, output, row, inner, columns, store, offset,
        );
        row += 4;
    }
    while row < rows {
        matmul_rows::<LANES, VECTORS, FUSED, PACKED, 1>(
            input, weights, bias, output, row, inner, columns, store, offset,
        );
        row += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn matmul_rows<
    const LANES: usize,
    const VECTORS: usize,
    const FUSED: bool,
    const PACKED: bool,
    const ROWS: usize,
>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    row: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
    offset: usize,
) {
    for block in 0..columns.div_ceil(MATRIX_BLOCK_COLUMNS) {
        let sums = projection_tile::<LANES, VECTORS, FUSED, ROWS, PACKED>(
            input, weights, bias, row, block, inner, columns,
        );
        for (r, row_sums) in sums.into_iter().enumerate() {
            for (v, sum) in row_sums.into_iter().enumerate() {
                let column = block * MATRIX_BLOCK_COLUMNS + v * LANES;
                if column < columns {
                    let start = (row + r) * columns + column;
                    store_partial(
                        store.vector(sum, offset + start),
                        &mut output[start..start + (columns - column).min(LANES)],
                    );
                }
            }
        }
    }
}

/// Compact projection with no dense vocabulary output. Complete packed tiles
/// add bias after K; scalar tails, Scalar and unpacked paths stay bias-first.
/// Equal logits keep the first class, including ties across vectors and tails.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compact_argmax(
    backend: Backend,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    indices: &mut [usize],
    scores: &mut [f32],
    inner: usize,
    columns: usize,
) -> Result<(), &'static str> {
    let rows = indices.len();
    matrix_lengths(
        input,
        weights,
        rows.checked_mul(columns)
            .ok_or("projection size overflow")?,
        rows,
        inner,
        columns,
    );
    if scores.len() != rows || (!bias.is_empty() && bias.len() != columns) {
        return Err("compact projection buffer shape mismatch");
    }
    if packed.is_some_and(|panel| !panel.matches(inner, columns)) {
        return Err("compact projection panel shape mismatch");
    }
    match backend {
        Backend::Scalar => scalar_argmax(input, weights, bias, indices, scores, inner, columns),
        Backend::Portable4 => argmax_simd::<4, 4, false>(
            input, weights, bias, packed, indices, scores, inner, columns,
        ),
        Backend::Avx2Fma => {
            if !avx2_fma_available() {
                return Err("AVX2/FMA backend requires AVX2 and FMA support");
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime feature detection immediately above guards entry.
            unsafe {
                argmax_avx2(
                    input, weights, bias, packed, indices, scores, inner, columns,
                )
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            Err("AVX2/FMA backend is unavailable on this architecture")
        }
        Backend::Avx => {
            if !avx_available() {
                return Err("AVX backend requires AVX support");
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime feature detection immediately above guards entry.
            unsafe {
                argmax_avx(
                    input, weights, bias, packed, indices, scores, inner, columns,
                )
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            Err("Avx backend is unavailable on this architecture")
        }
        Backend::AvxFma => {
            if !avx_fma_available() {
                return Err("AVX/FMA backend requires AVX and FMA support");
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime feature detection immediately above guards entry.
            unsafe {
                argmax_avx_fma(
                    input, weights, bias, packed, indices, scores, inner, columns,
                )
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            Err("AvxFma backend is unavailable on this architecture")
        }
        Backend::Avx512 => {
            if !avx512_available() {
                return Err("AVX-512 backend requires AVX-512F, AVX2 and FMA support");
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime feature detection immediately above guards entry.
            unsafe {
                argmax_avx512(
                    input, weights, bias, packed, indices, scores, inner, columns,
                )
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            Err("Avx512 backend is unavailable on this architecture")
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn scalar_argmax(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    indices: &mut [usize],
    scores: &mut [f32],
    inner: usize,
    columns: usize,
) -> Result<(), &'static str> {
    // Match the existing public recognition head's four-row/four-column
    // scalar tile, including bias-first accumulation and strict first ties.
    let full_rows = indices.len() / 4 * 4;
    let full_columns = columns / 4 * 4;
    for row in (0..full_rows).step_by(4) {
        let f0 = &input[row * inner..(row + 1) * inner];
        let f1 = &input[(row + 1) * inner..(row + 2) * inner];
        let f2 = &input[(row + 2) * inner..(row + 3) * inner];
        let f3 = &input[(row + 3) * inner..(row + 4) * inner];
        let mut best = [0usize; 4];
        let mut maximum = [f32::NEG_INFINITY; 4];
        for col in (0..full_columns).step_by(4) {
            let (b0, b1, b2, b3) = if bias.is_empty() {
                (0.0, 0.0, 0.0, 0.0)
            } else {
                (bias[col], bias[col + 1], bias[col + 2], bias[col + 3])
            };
            let (mut a00, mut a01, mut a02, mut a03) = (b0, b1, b2, b3);
            let (mut a10, mut a11, mut a12, mut a13) = (b0, b1, b2, b3);
            let (mut a20, mut a21, mut a22, mut a23) = (b0, b1, b2, b3);
            let (mut a30, mut a31, mut a32, mut a33) = (b0, b1, b2, b3);
            for k in 0..inner {
                let w = &weights[k * columns + col..k * columns + col + 4];
                let (w0, w1, w2, w3) = (w[0], w[1], w[2], w[3]);
                let (v0, v1, v2, v3) = (f0[k], f1[k], f2[k], f3[k]);
                a00 += v0 * w0;
                a01 += v0 * w1;
                a02 += v0 * w2;
                a03 += v0 * w3;
                a10 += v1 * w0;
                a11 += v1 * w1;
                a12 += v1 * w2;
                a13 += v1 * w3;
                a20 += v2 * w0;
                a21 += v2 * w1;
                a22 += v2 * w2;
                a23 += v2 * w3;
                a30 += v3 * w0;
                a31 += v3 * w1;
                a32 += v3 * w2;
                a33 += v3 * w3;
            }
            for (r, block) in [
                [a00, a01, a02, a03],
                [a10, a11, a12, a13],
                [a20, a21, a22, a23],
                [a30, a31, a32, a33],
            ]
            .into_iter()
            .enumerate()
            {
                for (lane, value) in block.into_iter().enumerate() {
                    update_winner(value, col + lane, &mut maximum[r], &mut best[r])?;
                }
            }
        }
        for col in full_columns..columns {
            let mut sums = [if bias.is_empty() { 0.0 } else { bias[col] }; 4];
            for k in 0..inner {
                let weight = weights[k * columns + col];
                sums[0] += f0[k] * weight;
                sums[1] += f1[k] * weight;
                sums[2] += f2[k] * weight;
                sums[3] += f3[k] * weight;
            }
            for r in 0..4 {
                update_winner(sums[r], col, &mut maximum[r], &mut best[r])?;
            }
        }
        indices[row..row + 4].copy_from_slice(&best);
        scores[row..row + 4].copy_from_slice(&maximum);
    }
    for row in full_rows..indices.len() {
        let input = &input[row * inner..(row + 1) * inner];
        let mut best = 0;
        let mut maximum = f32::NEG_INFINITY;
        for col in 0..columns {
            let mut sum = if bias.is_empty() { 0.0 } else { bias[col] };
            for k in 0..inner {
                sum += input[k] * weights[k * columns + col];
            }
            update_winner(sum, col, &mut maximum, &mut best)?;
        }
        indices[row] = best;
        scores[row] = maximum;
    }
    Ok(())
}

#[inline]
fn update_winner(
    value: f32,
    column: usize,
    maximum: &mut f32,
    best: &mut usize,
) -> Result<(), &'static str> {
    if !value.is_finite() {
        return Err("nonfinite compact recognition logits");
    }
    if value > *maximum {
        *maximum = value;
        *best = column;
    }
    Ok(())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn argmax_avx2(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    indices: &mut [usize],
    scores: &mut [f32],
    inner: usize,
    columns: usize,
) -> Result<(), &'static str> {
    argmax_simd::<8, 2, true>(
        input, weights, bias, packed, indices, scores, inner, columns,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn argmax_avx(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    indices: &mut [usize],
    scores: &mut [f32],
    inner: usize,
    columns: usize,
) -> Result<(), &'static str> {
    argmax_simd::<8, 2, false>(
        input, weights, bias, packed, indices, scores, inner, columns,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn argmax_avx_fma(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    indices: &mut [usize],
    scores: &mut [f32],
    inner: usize,
    columns: usize,
) -> Result<(), &'static str> {
    argmax_simd::<8, 2, true>(
        input, weights, bias, packed, indices, scores, inner, columns,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn argmax_avx512(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    indices: &mut [usize],
    scores: &mut [f32],
    inner: usize,
    columns: usize,
) -> Result<(), &'static str> {
    if !wide_projection_eligible(indices.len(), inner, columns, packed.is_some()) {
        return argmax_simd::<8, 2, true>(
            input, weights, bias, packed, indices, scores, inner, columns,
        );
    }
    argmax_simd::<16, 1, true>(
        input, weights, bias, packed, indices, scores, inner, columns,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn argmax_simd<const LANES: usize, const VECTORS: usize, const FUSED: bool>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    indices: &mut [usize],
    scores: &mut [f32],
    inner: usize,
    columns: usize,
) -> Result<(), &'static str> {
    if let Some(panel) = packed {
        argmax_simd_impl::<LANES, VECTORS, FUSED, true>(
            input,
            panel.data(),
            bias,
            indices,
            scores,
            inner,
            columns,
        )
    } else if LANES >= 8 {
        // The reference takes its scalar ArgMax implementation without a
        // packed vocabulary matrix, including its separate multiply/adds.
        scalar_argmax(input, weights, bias, indices, scores, inner, columns)
    } else {
        argmax_simd_impl::<LANES, VECTORS, FUSED, false>(
            input, weights, bias, indices, scores, inner, columns,
        )
    }
}

/// Compact packed tiles have a different bias epilogue from the public
/// low-level bias-first MatMul. Keep that distinction outside projection_tile
/// so an ArgMax correction cannot silently change the MatMul contract.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn compact_projection_tile<
    const LANES: usize,
    const VECTORS: usize,
    const FUSED: bool,
    const ROWS: usize,
    const PACKED: bool,
>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    first_row: usize,
    block: usize,
    inner: usize,
    columns: usize,
) -> [[Simd<f32, LANES>; VECTORS]; ROWS] {
    let mut sums = projection_tile::<LANES, VECTORS, FUSED, ROWS, PACKED>(
        input,
        weights,
        if PACKED { &[] } else { bias },
        first_row,
        block,
        inner,
        columns,
    );
    if PACKED && !bias.is_empty() {
        // Every packed caller passes complete 16-column tiles. The pinned
        // Vector, AVX, AVX/FMA and AVX-512 paths add bias after the final K.
        for vector in 0..VECTORS {
            let column = block * MATRIX_BLOCK_COLUMNS + vector * LANES;
            let bias = Simd::from_slice(&bias[column..column + LANES]);
            for row in &mut sums {
                row[vector] += bias;
            }
        }
    }
    sums
}

/// The reference's FinishScalarTail starts from bias and uses separate scalar
/// operations, even when preceding full tiles used FMA and a bias epilogue.
/// Packed and original coefficients have identical bits; sharing this helper
/// also keeps column-shard tails consistent without another weight traversal.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn compact_scalar_tail<const PACKED: bool>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    row: usize,
    inner: usize,
    columns: usize,
    first_column: usize,
    maximum: &mut f32,
    best: &mut usize,
) -> Result<(), &'static str> {
    let input = &input[row * inner..(row + 1) * inner];
    for column in first_column..columns {
        let mut sum = if bias.is_empty() { 0. } else { bias[column] };
        for k in 0..inner {
            let weight = if PACKED {
                weights[(column / MATRIX_BLOCK_COLUMNS * inner + k) * MATRIX_BLOCK_COLUMNS
                    + column % MATRIX_BLOCK_COLUMNS]
            } else {
                weights[k * columns + column]
            };
            sum += input[k] * weight;
        }
        update_winner(sum, column, maximum, best)?;
    }
    Ok(())
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn argmax_simd_impl<
    const LANES: usize,
    const VECTORS: usize,
    const FUSED: bool,
    const PACKED: bool,
>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    indices: &mut [usize],
    scores: &mut [f32],
    inner: usize,
    columns: usize,
) -> Result<(), &'static str> {
    if PACKED && columns <= u32::MAX as usize {
        return argmax::blocked::<LANES, VECTORS, FUSED>(
            input, weights, bias, indices, scores, inner, columns,
        );
    }
    let rows = indices.len();
    let wide = wide_projection_eligible(rows, inner, columns, PACKED);
    let mut row = 0;
    if wide && LANES == 16 {
        while row + 16 <= rows {
            argmax_rows::<LANES, VECTORS, FUSED, PACKED, 16>(
                input, weights, bias, indices, scores, row, inner, columns,
            )?;
            row += 16;
        }
    }
    if wide {
        while row + 8 <= rows {
            argmax_rows::<LANES, VECTORS, FUSED, PACKED, 8>(
                input, weights, bias, indices, scores, row, inner, columns,
            )?;
            row += 8;
        }
    }
    while row + 4 <= rows {
        argmax_rows::<LANES, VECTORS, FUSED, PACKED, 4>(
            input, weights, bias, indices, scores, row, inner, columns,
        )?;
        row += 4;
    }
    while row < rows {
        argmax_rows::<LANES, VECTORS, FUSED, PACKED, 1>(
            input, weights, bias, indices, scores, row, inner, columns,
        )?;
        row += 1;
    }
    Ok(())
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn argmax_rows<
    const LANES: usize,
    const VECTORS: usize,
    const FUSED: bool,
    const PACKED: bool,
    const ROWS: usize,
>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    indices: &mut [usize],
    scores: &mut [f32],
    row: usize,
    inner: usize,
    columns: usize,
) -> Result<(), &'static str> {
    let mut best = [0usize; ROWS];
    let mut maximum = [f32::NEG_INFINITY; ROWS];
    let tiles = if PACKED || LANES >= 8 {
        columns / MATRIX_BLOCK_COLUMNS
    } else {
        columns.div_ceil(MATRIX_BLOCK_COLUMNS)
    };
    for block in 0..tiles {
        let sums = compact_projection_tile::<LANES, VECTORS, FUSED, ROWS, PACKED>(
            input, weights, bias, row, block, inner, columns,
        );
        for r in 0..ROWS {
            for (v, vector) in sums[r].iter().enumerate() {
                let column = block * MATRIX_BLOCK_COLUMNS + v * LANES;
                for (lane, value) in vector
                    .to_array()
                    .into_iter()
                    .enumerate()
                    .take(columns.saturating_sub(column).min(LANES))
                {
                    update_winner(value, column + lane, &mut maximum[r], &mut best[r])?;
                }
            }
        }
    }
    if PACKED || LANES >= 8 {
        for r in 0..ROWS {
            compact_scalar_tail::<PACKED>(
                input,
                weights,
                bias,
                row + r,
                inner,
                columns,
                tiles * MATRIX_BLOCK_COLUMNS,
                &mut maximum[r],
                &mut best[r],
            )?;
        }
    }
    indices[row..row + ROWS].copy_from_slice(&best);
    scores[row..row + ROWS].copy_from_slice(&maximum);
    Ok(())
}

/// Validated logical NCHW convolution dimensions; filters are OIHW.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Conv2d {
    pub batch: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub height: usize,
    pub width: usize,
    pub output_height: usize,
    pub output_width: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
    pub stride_height: usize,
    pub stride_width: usize,
    pub dilation_height: usize,
    pub dilation_width: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub groups: usize,
}

impl Conv2d {
    fn avx512_spatial(&self) -> bool {
        self.dilation_height == 1
            && self.dilation_width == 1
            && ((self.groups == self.input_channels && self.output_channels == self.input_channels)
                || (self.groups == 1
                    && self.kernel_height == 3
                    && self.kernel_width == 3
                    && self.pad_top == 1
                    && self.pad_left == 1
                    && self.stride_height == self.stride_width
                    && matches!(self.stride_width, 1 | 2)
                    && self.output_channels.is_multiple_of(4)))
    }
    pub(crate) fn is_direct_pointwise(&self) -> bool {
        self.kernel_height == 1
            && self.kernel_width == 1
            && self.stride_height == 1
            && self.stride_width == 1
            && self.pad_top == 0
            && self.pad_left == 0
            && self.output_height == self.height
            && self.output_width == self.width
    }

    fn horizontal_interior(&self) -> std::ops::Range<usize> {
        let left = self
            .pad_left
            .div_ceil(self.stride_width)
            .min(self.output_width);
        let right = (self.width + self.pad_left)
            .saturating_sub((self.kernel_width - 1) * self.dilation_width)
            .div_ceil(self.stride_width)
            .min(self.output_width)
            .max(left);
        left..right
    }

    /// Preserve the original scalar loop when this dense geometry cannot
    /// execute even one selected-width vector. Depthwise/groups retain their
    /// independent selectors; direct pointwise keeps its earlier measured gate.
    pub(crate) fn uses_original_convolution(&self, backend: Backend) -> bool {
        if self.is_direct_pointwise() {
            return true;
        }
        let lanes = match backend {
            Backend::Scalar => return false,
            Backend::Portable4 => 4,
            Backend::Avx | Backend::AvxFma | Backend::Avx2Fma => 8,
            Backend::Avx512 => {
                if self.avx512_spatial() {
                    16
                } else {
                    8
                }
            }
        };
        self.groups == 1 && self.input_channels > 1 && self.horizontal_interior().len() < lanes
    }

    fn original_plan(&self) -> super::kernels::Conv {
        super::kernels::Conv {
            batch: self.batch,
            cin: self.input_channels,
            cout: self.output_channels,
            height: self.height,
            width: self.width,
            oh: self.output_height,
            ow: self.output_width,
            cpg: self.input_channels / self.groups,
            opg: self.output_channels / self.groups,
            kh: self.kernel_height,
            kw: self.kernel_width,
            sh: self.stride_height,
            sw: self.stride_width,
            dh: self.dilation_height,
            dw: self.dilation_width,
            pt: self.pad_top,
            pl: self.pad_left,
        }
    }

    fn validate(&self, input: &[f32], weights: &[f32], bias: &[f32], output_len: usize) {
        let p = self;
        assert!(
            [
                p.batch,
                p.input_channels,
                p.output_channels,
                p.height,
                p.width,
                p.output_height,
                p.output_width,
                p.kernel_height,
                p.kernel_width,
                p.stride_height,
                p.stride_width,
                p.dilation_height,
                p.dilation_width,
                p.groups
            ]
            .into_iter()
            .all(|v| v > 0),
            "convolution dimensions must be positive"
        );
        assert_eq!(p.input_channels % p.groups, 0);
        assert_eq!(p.output_channels % p.groups, 0);
        let product = |values: &[usize]| {
            values
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b))
                .expect("convolution size overflow")
        };
        assert_eq!(
            input.len(),
            product(&[p.batch, p.input_channels, p.height, p.width])
        );
        assert_eq!(
            weights.len(),
            product(&[
                p.output_channels,
                p.input_channels / p.groups,
                p.kernel_height,
                p.kernel_width
            ])
        );
        assert_eq!(
            output_len,
            product(&[p.batch, p.output_channels, p.output_height, p.output_width])
        );
        assert!(bias.is_empty() || bias.len() == p.output_channels);
        // Coordinate arithmetic is checked once, outside all kernel loops.
        (p.output_height - 1)
            .checked_mul(p.stride_height)
            .and_then(|v| {
                (p.kernel_height - 1)
                    .checked_mul(p.dilation_height)
                    .and_then(|k| v.checked_add(k))
            })
            .expect("convolution height coordinate overflow");
        (p.output_width - 1)
            .checked_mul(p.stride_width)
            .and_then(|v| {
                (p.kernel_width - 1)
                    .checked_mul(p.dilation_width)
                    .and_then(|k| v.checked_add(k))
            })
            .expect("convolution width coordinate overflow");
        p.height
            .checked_add(p.pad_top)
            .expect("convolution padded height overflow");
        p.width
            .checked_add(p.pad_left)
            .expect("convolution padded width overflow");
    }
}

/// Dense, pointwise and depthwise/grouped NCHW convolution. Interior SIMD
/// vectors share each source load across four independent output channels;
/// boundaries retain the scalar tap-skipping and bias-first contract.
pub(crate) fn conv2d(
    backend: Backend,
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    p.validate(input, weights, bias, output.len());
    match backend {
        Backend::Avx => checked_avx(),
        Backend::AvxFma => checked_avx_fma(),
        Backend::Avx2Fma => checked_avx2(),
        Backend::Avx512 => checked_avx512(),
        _ => {}
    }
    // Both fallbacks are measured production choices. In particular, a short
    // dense interior otherwise executes only the generic scalar pixel tail,
    // which is slower than the original loop's pre-sliced coefficient planes.
    // NHWC keeps its packed MatMul path; these gates only concern NCHW.
    if p.uses_original_convolution(backend) {
        let plan = p.original_plan();
        if p.is_direct_pointwise() {
            super::kernels::conv1x1(&plan, input, weights, bias, output);
        } else {
            super::kernels::conv(&plan, input, weights, bias, output);
        }
        return;
    }
    match backend {
        Backend::Scalar => convolution::<1, false>(p, input, weights, bias, output),
        Backend::Portable4 => convolution::<4, false>(p, input, weights, bias, output),
        Backend::Avx2Fma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime CPU detection guards both enabled features.
            unsafe {
                convolution_avx2(p, input, weights, bias, output)
            }
        }
        Backend::Avx => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime CPU detection guards both enabled features.
            unsafe {
                convolution_avx(p, input, weights, bias, output)
            }
        }
        Backend::AvxFma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime CPU detection guards both enabled features.
            unsafe {
                convolution_avx_fma(p, input, weights, bias, output)
            }
        }
        Backend::Avx512 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime CPU detection guards both enabled features.
            unsafe {
                convolution_avx512(p, input, weights, bias, output)
            }
        }
    }
}

/// Compute one batch's contiguous output-channel range into a disjoint slice.
/// The original input/filter/bias slices stay shared, including grouped cases.
/// This interface permits output-plane sharding without copying or pointer
/// reconstruction in the caller's worker pool.
#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "parallel", test))]
pub(crate) fn conv2d_output_range(
    backend: Backend,
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    first_channel: usize,
    count: usize,
) {
    let plane = p
        .output_height
        .checked_mul(p.output_width)
        .expect("convolution size overflow");
    let full_length = p
        .batch
        .checked_mul(p.output_channels)
        .and_then(|v| v.checked_mul(plane))
        .expect("convolution size overflow");
    p.validate(input, weights, bias, full_length);
    assert!(batch < p.batch);
    assert!(
        first_channel
            .checked_add(count)
            .is_some_and(|end| end <= p.output_channels)
    );
    assert_eq!(
        output.len(),
        count
            .checked_mul(plane)
            .expect("convolution range overflow")
    );
    let origin = (batch * p.output_channels + first_channel) * plane;
    match backend {
        Backend::Scalar => convolution_range::<1, false>(
            p,
            input,
            weights,
            bias,
            output,
            batch,
            first_channel,
            count,
            origin,
        ),
        Backend::Portable4 => convolution_range::<4, false>(
            p,
            input,
            weights,
            bias,
            output,
            batch,
            first_channel,
            count,
            origin,
        ),
        Backend::Avx2Fma => {
            checked_avx2();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime CPU detection guards both target features.
            unsafe {
                convolution_range_avx2(
                    p,
                    input,
                    weights,
                    bias,
                    output,
                    batch,
                    first_channel,
                    count,
                    origin,
                )
            }
        }
        Backend::Avx => {
            checked_avx();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime CPU detection guards both target features.
            unsafe {
                convolution_range_avx(
                    p,
                    input,
                    weights,
                    bias,
                    output,
                    batch,
                    first_channel,
                    count,
                    origin,
                )
            }
        }
        Backend::AvxFma => {
            checked_avx_fma();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime CPU detection guards both target features.
            unsafe {
                convolution_range_avx_fma(
                    p,
                    input,
                    weights,
                    bias,
                    output,
                    batch,
                    first_channel,
                    count,
                    origin,
                )
            }
        }
        Backend::Avx512 => {
            checked_avx512();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime CPU detection guards both target features.
            unsafe {
                convolution_range_avx512(
                    p,
                    input,
                    weights,
                    bias,
                    output,
                    batch,
                    first_channel,
                    count,
                    origin,
                )
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "parallel", test))]
unsafe fn convolution_range_avx2(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    first_channel: usize,
    count: usize,
    origin: usize,
) {
    convolution_range::<8, true>(
        p,
        input,
        weights,
        bias,
        output,
        batch,
        first_channel,
        count,
        origin,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "parallel", test))]
unsafe fn convolution_range_avx(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    first_channel: usize,
    count: usize,
    origin: usize,
) {
    convolution_range::<8, false>(
        p,
        input,
        weights,
        bias,
        output,
        batch,
        first_channel,
        count,
        origin,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "parallel", test))]
unsafe fn convolution_range_avx_fma(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    first_channel: usize,
    count: usize,
    origin: usize,
) {
    convolution_range::<8, true>(
        p,
        input,
        weights,
        bias,
        output,
        batch,
        first_channel,
        count,
        origin,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "parallel", test))]
unsafe fn convolution_range_avx512(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    first_channel: usize,
    count: usize,
    origin: usize,
) {
    if !p.avx512_spatial() {
        convolution_range::<8, true>(
            p,
            input,
            weights,
            bias,
            output,
            batch,
            first_channel,
            count,
            origin,
        );
        return;
    }
    convolution_range::<16, true>(
        p,
        input,
        weights,
        bias,
        output,
        batch,
        first_channel,
        count,
        origin,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn convolution_avx2(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    convolution::<8, true>(p, input, weights, bias, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn convolution_avx(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    convolution::<8, false>(p, input, weights, bias, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn convolution_avx_fma(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    convolution::<8, true>(p, input, weights, bias, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn convolution_avx512(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    if p.avx512_spatial() {
        convolution::<16, true>(p, input, weights, bias, output);
    } else {
        // DenseStride1's plain AVX-512 route is disabled in the reference.
        convolution::<8, true>(p, input, weights, bias, output);
    }
}

#[inline(always)]
fn valid_taps(
    origin: usize,
    padding: usize,
    input: usize,
    dilation: usize,
    kernel: usize,
) -> std::ops::Range<usize> {
    padding
        .saturating_sub(origin)
        .div_ceil(dilation)
        .min(kernel)
        ..(input + padding)
            .saturating_sub(origin)
            .div_ceil(dilation)
            .min(kernel)
}

#[inline(always)]
fn convolution<const LANES: usize, const FUSED: bool>(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    for batch in 0..p.batch {
        convolution_range::<LANES, FUSED>(
            p,
            input,
            weights,
            bias,
            output,
            batch,
            0,
            p.output_channels,
            0,
        );
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn convolution_range<const LANES: usize, const FUSED: bool>(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    first_channel: usize,
    count: usize,
    origin: usize,
) {
    let channels_per_group = p.output_channels / p.groups;
    let end = first_channel + count;
    let mut first = first_channel;
    while first < end {
        let group = first / channels_per_group;
        let group_end = ((group + 1) * channels_per_group).min(end);
        let tiled_end = first + (group_end - first) / 4 * 4;
        for channel in (first..tiled_end).step_by(4) {
            convolution_outputs::<LANES, FUSED, 4>(
                p, input, weights, bias, output, batch, group, channel, origin,
            );
        }
        for channel in tiled_end..group_end {
            convolution_outputs::<LANES, FUSED, 1>(
                p, input, weights, bias, output, batch, group, channel, origin,
            );
        }
        first = group_end;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn convolution_outputs<const LANES: usize, const FUSED: bool, const OUTPUTS: usize>(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    group: usize,
    channel: usize,
    output_origin: usize,
) {
    let input_plane = p.height * p.width;
    let output_plane = p.output_height * p.output_width;
    let input_channels = p.input_channels / p.groups;
    let weight_plane = input_channels * p.kernel_height * p.kernel_width;
    let source_base = (batch * p.input_channels + group * input_channels) * input_plane;
    let initial = std::array::from_fn::<_, OUTPUTS, _>(|c| {
        if bias.is_empty() {
            0.0
        } else {
            bias[channel + c]
        }
    });
    // A vector is interior only when its first and last pixels can use every
    // horizontal tap. Padding rows are skipped identically for every lane.
    let interior = p.horizontal_interior();
    let (left, right) = (interior.start, interior.end);
    for y in 0..p.output_height {
        let rows = valid_taps(
            y * p.stride_height,
            p.pad_top,
            p.height,
            p.dilation_height,
            p.kernel_height,
        );
        for x in 0..left {
            convolution_pixel::<OUTPUTS>(
                p,
                input,
                weights,
                output,
                batch,
                channel,
                source_base,
                weight_plane,
                y,
                x,
                &initial,
                rows.clone(),
                output_origin,
            );
        }
        let mut x = left;
        while x + LANES <= right {
            let mut sums =
                std::array::from_fn::<_, OUTPUTS, _>(|c| Simd::<f32, LANES>::splat(initial[c]));
            for ci in 0..input_channels {
                for ky in rows.clone() {
                    let source_row = source_base
                        + ci * input_plane
                        + (y * p.stride_height + ky * p.dilation_height - p.pad_top) * p.width;
                    let weight_row = (ci * p.kernel_height + ky) * p.kernel_width;
                    for kx in 0..p.kernel_width {
                        let source =
                            source_row + x * p.stride_width + kx * p.dilation_width - p.pad_left;
                        let value = if p.stride_width == 1 {
                            Simd::from_slice(&input[source..source + LANES])
                        } else {
                            // Fixed-width strided loads avoid gathers reading
                            // padding or the next channel at narrow tails.
                            Simd::from_array(std::array::from_fn(|lane| {
                                input[source + lane * p.stride_width]
                            }))
                        };
                        for c in 0..OUTPUTS {
                            let weight = Simd::splat(
                                weights[(channel + c) * weight_plane + weight_row + kx],
                            );
                            sums[c] = multiply_add::<LANES, FUSED>(value, weight, sums[c]);
                        }
                    }
                }
            }
            for (c, sum) in sums.into_iter().enumerate() {
                let start = (batch * p.output_channels + channel + c) * output_plane
                    + y * p.output_width
                    + x
                    - output_origin;
                sum.copy_to_slice(&mut output[start..start + LANES]);
            }
            x += LANES;
        }
        for x in x..p.output_width {
            convolution_pixel::<OUTPUTS>(
                p,
                input,
                weights,
                output,
                batch,
                channel,
                source_base,
                weight_plane,
                y,
                x,
                &initial,
                rows.clone(),
                output_origin,
            );
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn convolution_pixel<const OUTPUTS: usize>(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
    batch: usize,
    channel: usize,
    source_base: usize,
    weight_plane: usize,
    y: usize,
    x: usize,
    initial: &[f32; OUTPUTS],
    rows: std::ops::Range<usize>,
    output_origin: usize,
) {
    let mut sums = *initial;
    let columns = valid_taps(
        x * p.stride_width,
        p.pad_left,
        p.width,
        p.dilation_width,
        p.kernel_width,
    );
    for ci in 0..p.input_channels / p.groups {
        for ky in rows.clone() {
            let source_row = source_base
                + ci * p.height * p.width
                + (y * p.stride_height + ky * p.dilation_height - p.pad_top) * p.width;
            let weight_row = (ci * p.kernel_height + ky) * p.kernel_width;
            for kx in columns.clone() {
                let value =
                    input[source_row + x * p.stride_width + kx * p.dilation_width - p.pad_left];
                for c in 0..OUTPUTS {
                    sums[c] += value * weights[(channel + c) * weight_plane + weight_row + kx];
                }
            }
        }
    }
    let plane = p.output_height * p.output_width;
    for c in 0..OUTPUTS {
        output[(batch * p.output_channels + channel + c) * plane + y * p.output_width + x
            - output_origin] = sums[c];
    }
}

/// Specialization of the detector's non-overlapping 2×2 stride-two transpose
/// convolution, with SIMD across source pixels and interleaved stores.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose2x2(
    backend: Backend,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    cin: usize,
    cout: usize,
    height: usize,
    width: usize,
) {
    assert!([batch, cin, cout, height, width].into_iter().all(|v| v > 0));
    let product = |values: &[usize]| {
        values
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .expect("transpose convolution size overflow")
    };
    assert_eq!(input.len(), product(&[batch, cin, height, width]));
    assert_eq!(weights.len(), product(&[cin, cout, 4]));
    assert_eq!(output.len(), product(&[batch, cout, height, width, 4]));
    assert!(bias.is_empty() || bias.len() == cout);
    match backend {
        Backend::Scalar => transpose_convolution::<1, false>(
            input, weights, bias, output, batch, cin, cout, height, width,
        ),
        Backend::Portable4 => transpose_convolution::<4, false>(
            input, weights, bias, output, batch, cin, cout, height, width,
        ),
        Backend::Avx2Fma => {
            checked_avx2();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked immediately above.
            unsafe {
                transpose_convolution_avx2(
                    input, weights, bias, output, batch, cin, cout, height, width,
                )
            }
        }
        Backend::Avx => {
            checked_avx();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked immediately above.
            unsafe {
                transpose_convolution_avx(
                    input, weights, bias, output, batch, cin, cout, height, width,
                )
            }
        }
        Backend::AvxFma => {
            checked_avx_fma();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked immediately above.
            unsafe {
                transpose_convolution_avx_fma(
                    input, weights, bias, output, batch, cin, cout, height, width,
                )
            }
        }
        Backend::Avx512 => {
            checked_avx512();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked immediately above.
            unsafe {
                transpose_convolution_avx512(
                    input, weights, bias, output, batch, cin, cout, height, width,
                )
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_convolution_avx2(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    cin: usize,
    cout: usize,
    height: usize,
    width: usize,
) {
    transpose_convolution::<8, true>(
        input, weights, bias, output, batch, cin, cout, height, width,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_convolution_avx(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    cin: usize,
    cout: usize,
    height: usize,
    width: usize,
) {
    transpose_convolution::<8, false>(
        input, weights, bias, output, batch, cin, cout, height, width,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_convolution_avx_fma(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    cin: usize,
    cout: usize,
    height: usize,
    width: usize,
) {
    transpose_convolution::<8, true>(
        input, weights, bias, output, batch, cin, cout, height, width,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_convolution_avx512(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    cin: usize,
    cout: usize,
    height: usize,
    width: usize,
) {
    transpose_convolution::<16, true>(
        input, weights, bias, output, batch, cin, cout, height, width,
    );
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn transpose_convolution<const LANES: usize, const FUSED: bool>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    cin: usize,
    cout: usize,
    height: usize,
    width: usize,
) {
    let input_plane = height * width;
    let output_plane = input_plane * 4;
    let output_width = width * 2;
    let tiled_width = width / LANES * LANES;
    for b in 0..batch {
        for co in 0..cout {
            let initial = if bias.is_empty() { 0.0 } else { bias[co] };
            for y in 0..height {
                for x in (0..tiled_width).step_by(LANES) {
                    let mut sums = [Simd::<f32, LANES>::splat(initial); 4];
                    for ci in 0..cin {
                        let source = (b * cin + ci) * input_plane + y * width + x;
                        let value = Simd::from_slice(&input[source..source + LANES]);
                        let w = &weights[(ci * cout + co) * 4..(ci * cout + co + 1) * 4];
                        for tap in 0..4 {
                            sums[tap] =
                                multiply_add::<LANES, FUSED>(value, Simd::splat(w[tap]), sums[tap]);
                        }
                    }
                    let base = (b * cout + co) * output_plane + y * 2 * output_width + x * 2;
                    // interleave is also defined for one lane, so the scalar
                    // backend follows exactly the same coefficient order.
                    let (top0, top1) = sums[0].interleave(sums[1]);
                    let (bottom0, bottom1) = sums[2].interleave(sums[3]);
                    top0.copy_to_slice(&mut output[base..base + LANES]);
                    top1.copy_to_slice(&mut output[base + LANES..base + 2 * LANES]);
                    bottom0.copy_to_slice(
                        &mut output[base + output_width..base + output_width + LANES],
                    );
                    bottom1.copy_to_slice(
                        &mut output[base + output_width + LANES..base + output_width + 2 * LANES],
                    );
                }
                for x in tiled_width..width {
                    let mut sums = [initial; 4];
                    for ci in 0..cin {
                        let value = input[(b * cin + ci) * input_plane + y * width + x];
                        for tap in 0..4 {
                            sums[tap] += value * weights[(ci * cout + co) * 4 + tap];
                        }
                    }
                    let base = (b * cout + co) * output_plane + y * 2 * output_width + x * 2;
                    output[base] = sums[0];
                    output[base + 1] = sums[1];
                    output[base + output_width] = sums[2];
                    output[base + output_width + 1] = sums[3];
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum UnaryOp {
    Square,
    Relu,
    Sqrt,
    Erf,
    Sigmoid,
    HardSwish,
    Gelu,
    HardSigmoid { alpha: f32, beta: f32 },
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BinaryOp {
    Add,
    Mul,
    Div,
    Sub,
}

#[inline]
fn erf_value(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let magnitude = value.abs();
    let t = 1.0 / (1.0 + 0.3275911 * magnitude);
    let polynomial =
        ((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449672) * t + 0.2548296) * t;
    sign * (1.0 - polynomial * exp_f32(-magnitude * magnitude))
}

#[cfg(test)]
mod portable_softmax_tests;
mod transcendental;
#[cfg(test)]
mod unary_policy_tests;

// The unfused scalar diagnostic path deliberately keeps the original
// exponential-based formula. Optimized tails and convolution store epilogues
// use unary_value below, exactly matching their vector lanes.
#[inline]
fn scalar_unary_value(operation: UnaryOp, value: f32) -> f32 {
    match operation {
        UnaryOp::Erf => erf_value(value),
        UnaryOp::Gelu => ((erf_value(value / std::f32::consts::SQRT_2) + 1.0) * value) * 0.5,
        _ => unary_value(operation, value),
    }
}

#[inline]
pub(super) fn unary_value(operation: UnaryOp, value: f32) -> f32 {
    match operation {
        UnaryOp::Square => value * value,
        UnaryOp::Relu => {
            if value > 0.0 {
                value
            } else {
                0.0
            }
        }
        UnaryOp::Sqrt => value.sqrt(),
        UnaryOp::Erf => unary_vector(UnaryOp::Erf, Simd::<f32, 1>::splat(value))[0],
        UnaryOp::Sigmoid => {
            if value >= 0.0 {
                1.0 / (1.0 + exp_f32(-value))
            } else {
                let e = exp_f32(value);
                e / (1.0 + e)
            }
        }
        UnaryOp::HardSigmoid { alpha, beta } => (alpha * value + beta).clamp(0.0, 1.0),
        UnaryOp::HardSwish => value * (value * 0.16666667 + 0.5).clamp(0.0, 1.0),
        UnaryOp::Gelu => {
            let v = value / std::f32::consts::SQRT_2;
            ((transcendental::erf(Simd::<f32, 1>::splat(v))[0] + 1.0) * value) * 0.5
        }
    }
}

/// Arithmetic selected from the forced backend, never rediscovered hardware.
/// Only Erf Horner steps fuse. Exp and hard activations remain nonfused.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct UnaryMath {
    erf_fma: bool,
    inverse_gelu_scale: bool,
    sigmoid_vector_exp: bool,
}
impl UnaryMath {
    pub(super) const fn for_backend(backend: Backend) -> Self {
        Self {
            erf_fma: matches!(
                backend,
                Backend::AvxFma | Backend::Avx2Fma | Backend::Avx512
            ),
            inverse_gelu_scale: !matches!(backend, Backend::Scalar),
            sigmoid_vector_exp: !matches!(backend, Backend::Scalar),
        }
    }
}

// A single scalar activation program must not expand into every unoptimized
// native family frame. Release retains the caller's guarded ISA context.
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn unary_value_with_math(operation: UnaryOp, value: f32, math: UnaryMath) -> f32 {
    match operation {
        UnaryOp::Erf | UnaryOp::Gelu | UnaryOp::Sigmoid => {
            unary_vector_with_math(operation, Simd::<f32, 1>::splat(value), math)[0]
        }
        _ => unary_value(operation, value),
    }
}

#[inline(always)]
fn erf_vector<const LANES: usize>(value: Simd<f32, LANES>, math: UnaryMath) -> Simd<f32, LANES> {
    transcendental::erf_with_fma(value, math.erf_fma)
}

#[inline(always)]
pub(super) fn unary_vector<const LANES: usize>(
    operation: UnaryOp,
    value: Simd<f32, LANES>,
) -> Simd<f32, LANES> {
    unary_vector_with_math(operation, value, UnaryMath::default())
}

// Unoptimized common and generic store paths share this large activation
// program instead of reserving it in every nested native dispatcher frame.
// Release inlining keeps the surrounding kernel's guarded ISA context.
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn unary_vector_with_math<const LANES: usize>(
    operation: UnaryOp,
    value: Simd<f32, LANES>,
    math: UnaryMath,
) -> Simd<f32, LANES> {
    let zero = Simd::splat(0.0);
    match operation {
        UnaryOp::Square => value * value,
        UnaryOp::Relu => value.simd_gt(zero).select(value, zero),
        UnaryOp::Sqrt => value.sqrt(),
        UnaryOp::Erf => erf_vector(value, math),
        UnaryOp::Sigmoid => {
            let one = Simd::splat(1.0);
            if math.sigmoid_vector_exp {
                // Pinned native and generic Vector Sigmoid use nonfused exp
                // and one division. Optimized scalar tails use the same math.
                let exponent = transcendental::exp(-value.abs());
                let positive = one / (one + exponent);
                value.simd_lt(zero).select(one - positive, positive)
            } else {
                let exponent = Simd::from_array((-value.abs()).to_array().map(exp_f32));
                let denominator = one + exponent;
                value
                    .simd_ge(zero)
                    .select(one / denominator, exponent / denominator)
            }
        }
        UnaryOp::HardSigmoid { alpha, beta } => {
            // Keep this expression unfused, matching the graph primitive.
            // Comparisons/select preserve scalar clamp's NaN behavior.
            let value = Simd::splat(alpha) * value + Simd::splat(beta);
            let value = value.simd_lt(zero).select(zero, value);
            let one = Simd::splat(1.0);
            value.simd_gt(one).select(one, value)
        }
        UnaryOp::HardSwish => {
            let gate = value * Simd::splat(0.16666667) + Simd::splat(0.5);
            let gate = gate.simd_lt(zero).select(zero, gate);
            let one = Simd::splat(1.0);
            value * gate.simd_gt(one).select(one, gate)
        }
        UnaryOp::Gelu => {
            #[allow(clippy::excessive_precision, clippy::approx_constant)]
            const REFERENCE_INVERSE_SQRT_TWO: f32 = 0.70710678118654752;
            let scaled = if math.inverse_gelu_scale {
                value * Simd::splat(REFERENCE_INVERSE_SQRT_TWO)
            } else {
                value / Simd::splat(std::f32::consts::SQRT_2)
            };
            let erf = erf_vector(scaled, math);
            ((erf + Simd::splat(1.0)) * value) * Simd::splat(0.5)
        }
    }
}

pub(crate) fn unary(backend: Backend, operation: UnaryOp, input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    match backend {
        Backend::Scalar => {
            for (&value, out) in input.iter().zip(output) {
                *out = scalar_unary_value(operation, value);
            }
        }
        Backend::Portable4 => unary_simd::<4>(
            operation,
            input,
            output,
            UnaryMath::for_backend(Backend::Portable4),
        ),
        Backend::Avx2Fma => {
            checked_avx2();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked above.
            unsafe {
                unary_avx2(operation, input, output)
            }
        }
        Backend::Avx => {
            checked_avx();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked above.
            unsafe {
                unary_avx(operation, input, output)
            }
        }
        Backend::AvxFma => {
            checked_avx_fma();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked above.
            unsafe {
                unary_avx_fma(operation, input, output)
            }
        }
        Backend::Avx512 => {
            checked_avx512();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked above.
            unsafe {
                unary_avx512(operation, input, output)
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn unary_avx2(operation: UnaryOp, input: &[f32], output: &mut [f32]) {
    unary_simd::<8>(
        operation,
        input,
        output,
        UnaryMath::for_backend(Backend::Avx2Fma),
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn unary_avx(operation: UnaryOp, input: &[f32], output: &mut [f32]) {
    unary_simd::<8>(
        operation,
        input,
        output,
        UnaryMath::for_backend(Backend::Avx),
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn unary_avx_fma(operation: UnaryOp, input: &[f32], output: &mut [f32]) {
    unary_simd::<8>(
        operation,
        input,
        output,
        UnaryMath::for_backend(Backend::AvxFma),
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn unary_avx512(operation: UnaryOp, input: &[f32], output: &mut [f32]) {
    unary_simd::<16>(
        operation,
        input,
        output,
        UnaryMath::for_backend(Backend::Avx512),
    );
}

#[inline(always)]
fn unary_simd<const LANES: usize>(
    operation: UnaryOp,
    input: &[f32],
    output: &mut [f32],
    math: UnaryMath,
) {
    let tiled = input.len() / LANES * LANES;
    for start in (0..tiled).step_by(LANES) {
        unary_vector_with_math(
            operation,
            Simd::<f32, LANES>::from_slice(&input[start..start + LANES]),
            math,
        )
        .copy_to_slice(&mut output[start..start + LANES]);
    }
    for (&value, out) in input[tiled..].iter().zip(&mut output[tiled..]) {
        *out = unary_value_with_math(operation, value, math);
    }
}

pub(crate) fn unary_inplace(backend: Backend, operation: UnaryOp, output: &mut [f32]) {
    match backend {
        Backend::Scalar => {
            for value in output {
                *value = scalar_unary_value(operation, *value);
            }
        }
        Backend::Portable4 => unary_inplace_simd::<4>(
            operation,
            output,
            UnaryMath::for_backend(Backend::Portable4),
        ),
        Backend::Avx2Fma => {
            checked_avx2();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features are checked before this call.
            unsafe {
                unary_inplace_avx2(operation, output)
            }
        }
        Backend::Avx => {
            checked_avx();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features are checked before this call.
            unsafe {
                unary_inplace_avx(operation, output)
            }
        }
        Backend::AvxFma => {
            checked_avx_fma();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features are checked before this call.
            unsafe {
                unary_inplace_avx_fma(operation, output)
            }
        }
        Backend::Avx512 => {
            checked_avx512();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features are checked before this call.
            unsafe {
                unary_inplace_avx512(operation, output)
            }
        }
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn unary_inplace_avx2(operation: UnaryOp, output: &mut [f32]) {
    unary_inplace_simd::<8>(operation, output, UnaryMath::for_backend(Backend::Avx2Fma));
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn unary_inplace_avx(operation: UnaryOp, output: &mut [f32]) {
    unary_inplace_simd::<8>(operation, output, UnaryMath::for_backend(Backend::Avx));
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn unary_inplace_avx_fma(operation: UnaryOp, output: &mut [f32]) {
    unary_inplace_simd::<8>(operation, output, UnaryMath::for_backend(Backend::AvxFma));
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn unary_inplace_avx512(operation: UnaryOp, output: &mut [f32]) {
    unary_inplace_simd::<16>(operation, output, UnaryMath::for_backend(Backend::Avx512));
}
#[inline(always)]
fn unary_inplace_simd<const LANES: usize>(operation: UnaryOp, output: &mut [f32], math: UnaryMath) {
    let (vectors, tail) = output.as_chunks_mut::<LANES>();
    for values in vectors {
        let value = Simd::<f32, LANES>::from_slice(values);
        unary_vector_with_math(operation, value, math).copy_to_slice(values);
    }
    for value in tail {
        *value = unary_value_with_math(operation, *value, math);
    }
}

#[inline]
fn binary_value(operation: BinaryOp, a: f32, b: f32) -> f32 {
    match operation {
        BinaryOp::Add => a + b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Sub => a - b,
    }
}

#[inline(always)]
fn binary_vector<const LANES: usize>(
    operation: BinaryOp,
    a: Simd<f32, LANES>,
    b: Simd<f32, LANES>,
) -> Simd<f32, LANES> {
    match operation {
        BinaryOp::Add => a + b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Sub => a - b,
    }
}

pub(crate) fn binary(
    backend: Backend,
    operation: BinaryOp,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    match backend {
        Backend::Scalar => {
            for ((&a, &b), out) in a.iter().zip(b).zip(output) {
                *out = binary_value(operation, a, b);
            }
        }
        Backend::Portable4 => binary_simd::<4>(operation, a, b, output),
        Backend::Avx2Fma => {
            checked_avx2();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked above.
            unsafe {
                binary_avx2(operation, a, b, output)
            }
        }
        Backend::Avx => {
            checked_avx();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked above.
            unsafe {
                binary_avx(operation, a, b, output)
            }
        }
        Backend::AvxFma => {
            checked_avx_fma();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked above.
            unsafe {
                binary_avx_fma(operation, a, b, output)
            }
        }
        Backend::Avx512 => {
            checked_avx512();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked above.
            unsafe {
                binary_avx512(operation, a, b, output)
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn binary_avx2(operation: BinaryOp, a: &[f32], b: &[f32], output: &mut [f32]) {
    binary_simd::<8>(operation, a, b, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn binary_avx(operation: BinaryOp, a: &[f32], b: &[f32], output: &mut [f32]) {
    binary_simd::<8>(operation, a, b, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn binary_avx_fma(operation: BinaryOp, a: &[f32], b: &[f32], output: &mut [f32]) {
    binary_simd::<8>(operation, a, b, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn binary_avx512(operation: BinaryOp, a: &[f32], b: &[f32], output: &mut [f32]) {
    binary_simd::<16>(operation, a, b, output);
}

#[inline(always)]
fn binary_simd<const LANES: usize>(operation: BinaryOp, a: &[f32], b: &[f32], output: &mut [f32]) {
    let tiled = a.len() / LANES * LANES;
    for start in (0..tiled).step_by(LANES) {
        let left = Simd::<f32, LANES>::from_slice(&a[start..start + LANES]);
        let right = Simd::<f32, LANES>::from_slice(&b[start..start + LANES]);
        binary_vector(operation, left, right).copy_to_slice(&mut output[start..start + LANES]);
    }
    for ((&a, &b), out) in a[tiled..].iter().zip(&b[tiled..]).zip(&mut output[tiled..]) {
        *out = binary_value(operation, a, b);
    }
}

/// `None` uses the current output value as both operands. A slice may be
/// scalar or full-sized. Loading both operands precedes each in-place store.
pub(crate) fn binary_inplace(
    backend: Backend,
    operation: BinaryOp,
    other: Option<&[f32]>,
    other_left: bool,
    output: &mut [f32],
) {
    if let Some(values) = other {
        assert!(values.len() == 1 || values.len() == output.len());
    }
    match backend {
        Backend::Scalar => binary_inplace_simd::<1>(operation, other, other_left, output),
        Backend::Portable4 => binary_inplace_simd::<4>(operation, other, other_left, output),
        Backend::Avx2Fma => {
            checked_avx2();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime detection guards both enabled features.
            unsafe {
                binary_inplace_avx2(operation, other, other_left, output)
            }
        }
        Backend::Avx => {
            checked_avx();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime detection guards both enabled features.
            unsafe {
                binary_inplace_avx(operation, other, other_left, output)
            }
        }
        Backend::AvxFma => {
            checked_avx_fma();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime detection guards both enabled features.
            unsafe {
                binary_inplace_avx_fma(operation, other, other_left, output)
            }
        }
        Backend::Avx512 => {
            checked_avx512();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime detection guards both enabled features.
            unsafe {
                binary_inplace_avx512(operation, other, other_left, output)
            }
        }
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn binary_inplace_avx2(
    operation: BinaryOp,
    other: Option<&[f32]>,
    other_left: bool,
    output: &mut [f32],
) {
    binary_inplace_simd::<8>(operation, other, other_left, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn binary_inplace_avx(
    operation: BinaryOp,
    other: Option<&[f32]>,
    other_left: bool,
    output: &mut [f32],
) {
    binary_inplace_simd::<8>(operation, other, other_left, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn binary_inplace_avx_fma(
    operation: BinaryOp,
    other: Option<&[f32]>,
    other_left: bool,
    output: &mut [f32],
) {
    binary_inplace_simd::<8>(operation, other, other_left, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn binary_inplace_avx512(
    operation: BinaryOp,
    other: Option<&[f32]>,
    other_left: bool,
    output: &mut [f32],
) {
    binary_inplace_simd::<16>(operation, other, other_left, output);
}
#[inline(always)]
fn binary_inplace_simd<const LANES: usize>(
    operation: BinaryOp,
    other: Option<&[f32]>,
    other_left: bool,
    output: &mut [f32],
) {
    let tiled = output.len() / LANES * LANES;
    for start in (0..tiled).step_by(LANES) {
        let value = Simd::<f32, LANES>::from_slice(&output[start..start + LANES]);
        let rhs = match other {
            None => value,
            Some([scalar]) => Simd::splat(*scalar),
            Some(values) => Simd::from_slice(&values[start..start + LANES]),
        };
        let result = if other_left {
            binary_vector(operation, rhs, value)
        } else {
            binary_vector(operation, value, rhs)
        };
        result.copy_to_slice(&mut output[start..start + LANES]);
    }
    for (index, value) in output[tiled..].iter_mut().enumerate() {
        let rhs = match other {
            None => *value,
            Some([scalar]) => *scalar,
            Some(values) => values[tiled + index],
        };
        *value = if other_left {
            binary_value(operation, rhs, *value)
        } else {
            binary_value(operation, *value, rhs)
        };
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ChannelBroadcast {
    pub channels: usize,
    pub plane: usize,
    pub batched: bool,
    pub gate_left: bool,
}
impl ChannelBroadcast {
    fn validate(self, gate: &[f32], length: usize) {
        assert!(self.channels > 0 && self.plane > 0);
        let span = self
            .channels
            .checked_mul(self.plane)
            .expect("channel broadcast span overflow");
        assert!(length.is_multiple_of(span));
        assert_eq!(
            gate.len(),
            if self.batched {
                length / self.plane
            } else {
                self.channels
            }
        );
    }
}

/// Broadcast logical `[1|N,C,1,1]` coefficients over contiguous NHWC pixels.
pub(crate) fn binary_channel(
    backend: Backend,
    operation: BinaryOp,
    full: &[f32],
    gate: &[f32],
    plan: ChannelBroadcast,
    output: &mut [f32],
) {
    assert_eq!(full.len(), output.len());
    plan.validate(gate, output.len());
    channel_dispatch::<false>(backend, operation, full, gate, plan, true, output);
}
pub(crate) fn binary_channel_inplace(
    backend: Backend,
    operation: BinaryOp,
    gate: &[f32],
    plan: ChannelBroadcast,
    nhwc: bool,
    output: &mut [f32],
) {
    plan.validate(gate, output.len());
    channel_dispatch::<true>(backend, operation, &[], gate, plan, nhwc, output);
}
#[allow(clippy::too_many_arguments)]
fn channel_dispatch<const INPLACE: bool>(
    backend: Backend,
    operation: BinaryOp,
    full: &[f32],
    gate: &[f32],
    plan: ChannelBroadcast,
    nhwc: bool,
    output: &mut [f32],
) {
    match backend {
        Backend::Scalar => channel_simd::<1, INPLACE>(operation, full, gate, plan, nhwc, output),
        Backend::Portable4 => channel_simd::<4, INPLACE>(operation, full, gate, plan, nhwc, output),
        Backend::Avx2Fma => {
            checked_avx2();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both CPU target features are checked outside all rows.
            unsafe {
                channel_avx2::<INPLACE>(operation, full, gate, plan, nhwc, output)
            }
        }
        Backend::Avx => {
            checked_avx();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both CPU target features are checked outside all rows.
            unsafe {
                channel_avx::<INPLACE>(operation, full, gate, plan, nhwc, output)
            }
        }
        Backend::AvxFma => {
            checked_avx_fma();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both CPU target features are checked outside all rows.
            unsafe {
                channel_avx_fma::<INPLACE>(operation, full, gate, plan, nhwc, output)
            }
        }
        Backend::Avx512 => {
            checked_avx512();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both CPU target features are checked outside all rows.
            unsafe {
                channel_avx512::<INPLACE>(operation, full, gate, plan, nhwc, output)
            }
        }
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn channel_avx2<const INPLACE: bool>(
    operation: BinaryOp,
    full: &[f32],
    gate: &[f32],
    plan: ChannelBroadcast,
    nhwc: bool,
    output: &mut [f32],
) {
    channel_simd::<8, INPLACE>(operation, full, gate, plan, nhwc, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn channel_avx<const INPLACE: bool>(
    operation: BinaryOp,
    full: &[f32],
    gate: &[f32],
    plan: ChannelBroadcast,
    nhwc: bool,
    output: &mut [f32],
) {
    channel_simd::<8, INPLACE>(operation, full, gate, plan, nhwc, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn channel_avx_fma<const INPLACE: bool>(
    operation: BinaryOp,
    full: &[f32],
    gate: &[f32],
    plan: ChannelBroadcast,
    nhwc: bool,
    output: &mut [f32],
) {
    channel_simd::<8, INPLACE>(operation, full, gate, plan, nhwc, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn channel_avx512<const INPLACE: bool>(
    operation: BinaryOp,
    full: &[f32],
    gate: &[f32],
    plan: ChannelBroadcast,
    nhwc: bool,
    output: &mut [f32],
) {
    channel_simd::<16, INPLACE>(operation, full, gate, plan, nhwc, output);
}
#[inline(always)]
fn channel_simd<const LANES: usize, const INPLACE: bool>(
    operation: BinaryOp,
    full: &[f32],
    gate: &[f32],
    plan: ChannelBroadcast,
    nhwc: bool,
    output: &mut [f32],
) {
    let span = plan.channels * plan.plane;
    for (batch, destination) in output.chunks_exact_mut(span).enumerate() {
        let coefficients = if plan.batched {
            &gate[batch * plan.channels..(batch + 1) * plan.channels]
        } else {
            gate
        };
        let row_length = if nhwc { plan.channels } else { plan.plane };
        for (row, values) in destination.chunks_exact_mut(row_length).enumerate() {
            let input_offset = batch * span + row * row_length;
            let tiled = row_length / LANES * LANES;
            for start in (0..tiled).step_by(LANES) {
                let value = if INPLACE {
                    Simd::<f32, LANES>::from_slice(&values[start..start + LANES])
                } else {
                    Simd::from_slice(&full[input_offset + start..input_offset + start + LANES])
                };
                let coefficient = if nhwc {
                    Simd::from_slice(&coefficients[start..start + LANES])
                } else {
                    Simd::splat(coefficients[row])
                };
                let result = if plan.gate_left {
                    binary_vector(operation, coefficient, value)
                } else {
                    binary_vector(operation, value, coefficient)
                };
                result.copy_to_slice(&mut values[start..start + LANES]);
            }
            for (index, value) in values[tiled..].iter_mut().enumerate() {
                let at = tiled + index;
                let source = if INPLACE {
                    *value
                } else {
                    full[input_offset + at]
                };
                let coefficient = coefficients[if nhwc { at } else { row }];
                *value = if plan.gate_left {
                    binary_value(operation, coefficient, source)
                } else {
                    binary_value(operation, source, coefficient)
                };
            }
        }
    }
}

pub(crate) fn binary_scalar(
    backend: Backend,
    operation: BinaryOp,
    input: &[f32],
    scalar: f32,
    scalar_left: bool,
    output: &mut [f32],
) {
    assert_eq!(input.len(), output.len());
    match backend {
        Backend::Scalar => {
            for (&value, out) in input.iter().zip(output) {
                *out = if scalar_left {
                    binary_value(operation, scalar, value)
                } else {
                    binary_value(operation, value, scalar)
                };
            }
        }
        Backend::Portable4 => {
            binary_scalar_simd::<4>(operation, input, scalar, scalar_left, output)
        }
        Backend::Avx2Fma => {
            checked_avx2();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked above.
            unsafe {
                binary_scalar_avx2(operation, input, scalar, scalar_left, output)
            }
        }
        Backend::Avx => {
            checked_avx();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked above.
            unsafe {
                binary_scalar_avx(operation, input, scalar, scalar_left, output)
            }
        }
        Backend::AvxFma => {
            checked_avx_fma();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked above.
            unsafe {
                binary_scalar_avx_fma(operation, input, scalar, scalar_left, output)
            }
        }
        Backend::Avx512 => {
            checked_avx512();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both target features were checked above.
            unsafe {
                binary_scalar_avx512(operation, input, scalar, scalar_left, output)
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn binary_scalar_avx2(
    operation: BinaryOp,
    input: &[f32],
    scalar: f32,
    scalar_left: bool,
    output: &mut [f32],
) {
    binary_scalar_simd::<8>(operation, input, scalar, scalar_left, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn binary_scalar_avx(
    operation: BinaryOp,
    input: &[f32],
    scalar: f32,
    scalar_left: bool,
    output: &mut [f32],
) {
    binary_scalar_simd::<8>(operation, input, scalar, scalar_left, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn binary_scalar_avx_fma(
    operation: BinaryOp,
    input: &[f32],
    scalar: f32,
    scalar_left: bool,
    output: &mut [f32],
) {
    binary_scalar_simd::<8>(operation, input, scalar, scalar_left, output);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn binary_scalar_avx512(
    operation: BinaryOp,
    input: &[f32],
    scalar: f32,
    scalar_left: bool,
    output: &mut [f32],
) {
    binary_scalar_simd::<16>(operation, input, scalar, scalar_left, output);
}

#[inline(always)]
fn binary_scalar_simd<const LANES: usize>(
    operation: BinaryOp,
    input: &[f32],
    scalar: f32,
    scalar_left: bool,
    output: &mut [f32],
) {
    let tiled = input.len() / LANES * LANES;
    let value = Simd::splat(scalar);
    for start in (0..tiled).step_by(LANES) {
        let chunk = Simd::<f32, LANES>::from_slice(&input[start..start + LANES]);
        let result = if scalar_left {
            binary_vector(operation, value, chunk)
        } else {
            binary_vector(operation, chunk, value)
        };
        result.copy_to_slice(&mut output[start..start + LANES]);
    }
    for (&value, out) in input[tiled..].iter().zip(&mut output[tiled..]) {
        *out = if scalar_left {
            binary_value(operation, scalar, value)
        } else {
            binary_value(operation, value, scalar)
        };
    }
}

/// Last-axis softmax. Scalar keeps its exact exponential; optimized vectors
/// use the pinned reference's float32 approximation and fixed lane-order
/// denominator. Row partitioning never changes that order.
pub(crate) fn softmax(
    backend: Backend,
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    columns: usize,
) {
    assert!(columns > 0);
    assert_eq!(
        input.len(),
        rows.checked_mul(columns).expect("Softmax size overflow")
    );
    assert_eq!(input.len(), output.len());
    match backend {
        Backend::Scalar => softmax_simd::<1>(input, output, rows, columns),
        Backend::Portable4 => softmax_simd::<4>(input, output, rows, columns),
        Backend::Avx2Fma => {
            checked_avx2();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime detection guards both enabled CPU features.
            unsafe {
                softmax_avx2(input, output, rows, columns)
            }
        }
        Backend::Avx => {
            checked_avx();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime detection guards both enabled CPU features.
            unsafe {
                softmax_avx(input, output, rows, columns)
            }
        }
        Backend::AvxFma => {
            checked_avx_fma();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime detection guards both enabled CPU features.
            unsafe {
                softmax_avx_fma(input, output, rows, columns)
            }
        }
        Backend::Avx512 => {
            checked_avx512();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: runtime detection guards both enabled CPU features.
            unsafe {
                softmax_avx512(input, output, rows, columns)
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_avx2(input: &[f32], output: &mut [f32], rows: usize, columns: usize) {
    softmax_simd::<8>(input, output, rows, columns);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn softmax_avx(input: &[f32], output: &mut [f32], rows: usize, columns: usize) {
    softmax_simd::<8>(input, output, rows, columns);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn softmax_avx_fma(input: &[f32], output: &mut [f32], rows: usize, columns: usize) {
    softmax_simd::<8>(input, output, rows, columns);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn softmax_avx512(input: &[f32], output: &mut [f32], rows: usize, columns: usize) {
    if columns >= 16 {
        softmax_simd::<16>(input, output, rows, columns);
    } else {
        // The reference still uses its AVX2 softmax for axes of 8..15
        // elements on an AVX-512 host.
        softmax_simd::<8>(input, output, rows, columns);
    }
}

#[inline(always)]
fn softmax_simd<const LANES: usize>(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    columns: usize,
) {
    if LANES >= 4 && columns >= LANES {
        softmax_native::<LANES>(input, output, rows, columns);
        return;
    }
    let tiled = columns / LANES * LANES;
    for row in 0..rows {
        let input = &input[row * columns..(row + 1) * columns];
        let output = &mut output[row * columns..(row + 1) * columns];
        let mut maximum = input[0];
        for &value in &input[1..] {
            maximum = maximum.max(value);
        }
        let vector_maximum = Simd::splat(maximum);
        for offset in (0..tiled).step_by(LANES) {
            let shifted =
                Simd::<f32, LANES>::from_slice(&input[offset..offset + LANES]) - vector_maximum;
            Simd::from_array(shifted.to_array().map(exp_f32))
                .copy_to_slice(&mut output[offset..offset + LANES]);
        }
        for (&input, output) in input[tiled..].iter().zip(&mut output[tiled..]) {
            *output = exp_f32(input - maximum);
        }
        let mut sum = 0.0f32;
        for &value in output.iter() {
            sum += value;
        }
        let denominator = Simd::splat(sum);
        for offset in (0..tiled).step_by(LANES) {
            (Simd::<f32, LANES>::from_slice(&output[offset..offset + LANES]) / denominator)
                .copy_to_slice(&mut output[offset..offset + LANES]);
        }
        for value in &mut output[tiled..] {
            *value /= sum;
        }
    }
}

#[inline(always)]
fn softmax_native<const LANES: usize>(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    columns: usize,
) {
    let tiled = columns / LANES * LANES;
    for row in 0..rows {
        let input = &input[row * columns..(row + 1) * columns];
        let output = &mut output[row * columns..(row + 1) * columns];
        let mut maxima = Simd::<f32, LANES>::splat(f32::NEG_INFINITY);
        for values in input[..tiled].as_chunks::<LANES>().0 {
            maxima = maxima.simd_max(Simd::from_slice(values));
        }
        let mut maximum = maxima.reduce_max();
        for &value in &input[tiled..] {
            maximum = maximum.max(value);
        }
        let mut sums = Simd::<f32, LANES>::splat(0.0);
        for offset in (0..tiled).step_by(LANES) {
            let values = transcendental::exp(
                Simd::from_slice(&input[offset..offset + LANES]) - Simd::splat(maximum),
            );
            values.copy_to_slice(&mut output[offset..offset + LANES]);
            sums += values;
        }
        // Match the reference AVX lane-order horizontal reduction, independent
        // of worker count and without requesting reassociation from LLVM.
        let mut sum = sums.to_array().into_iter().fold(0.0, |a, b| a + b);
        for (&value, out) in input[tiled..].iter().zip(&mut output[tiled..]) {
            *out = exp_f32(value - maximum);
            sum += *out;
        }
        let inverse = Simd::<f32, LANES>::splat(1.0 / sum);
        for values in output[..tiled].as_chunks_mut::<LANES>().0 {
            (Simd::from_slice(values) * inverse).copy_to_slice(values);
        }
        for value in &mut output[tiled..] {
            *value /= sum;
        }
    }
}

mod tensor;
pub(crate) use tensor::{average_pool, max_pool, spatial_mean};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_transpose_preserves_bits_for_tiles_tails_and_roundtrips() {
        for rows in [1, 3, 7, 8, 9, 16, 17, 33] {
            for columns in [1, 5, 7, 8, 9, 16, 31] {
                let input = (0..rows * columns)
                    .map(|i| match i % 13 {
                        0 => -0.0,
                        1 => f32::from_bits(0x7fc0_0001 + i as u32),
                        _ => f32::from_bits(0x3f00_0000 + i as u32),
                    })
                    .collect::<Vec<_>>();
                let mut output = vec![0.; input.len()];
                transpose_matrix(&input, &mut output, rows, columns);
                for row in 0..rows {
                    for col in 0..columns {
                        assert_eq!(
                            output[col * rows + row].to_bits(),
                            input[row * columns + col].to_bits()
                        );
                    }
                }
                let mut restored = vec![0.; input.len()];
                transpose_matrix(&output, &mut restored, columns, rows);
                assert_bits(&restored, &input);
            }
        }
    }

    #[test]
    fn layer_norm_affine_keeps_separate_ops_inplace_and_out_of_place() {
        for length in [1, 3, 4, 7, 8, 9, 17, 31, 128] {
            let input = values(length);
            let gamma = (0..length)
                .map(|i| (i as f32 - 3.) / 7.)
                .collect::<Vec<_>>();
            let beta = (0..length)
                .map(|i| (i as f32 - 1.) / 13.)
                .collect::<Vec<_>>();
            let mean = input.iter().sum::<f32>() / length as f32;
            let sd = 0.371;
            let expected = (0..length)
                .map(|i| ((input[i] - mean) / sd) * gamma[i] + beta[i])
                .collect::<Vec<_>>();
            for backend in backends() {
                let mut output = vec![0.; length];
                layer_norm_affine(backend, Some(&input), &gamma, &beta, mean, sd, &mut output);
                assert_bits(&output, &expected);
                let mut inplace = input.clone();
                layer_norm_affine(backend, None, &gamma, &beta, mean, sd, &mut inplace);
                assert_bits(&inplace, &expected);
            }
        }
    }

    #[test]
    fn nchw_pool_vectors_preserve_scalar_edges_and_strided_average_tails() {
        for (height, width) in [(1, 1), (2, 7), (6, 66)] {
            let shape = crate::tensor::Shape::new(&[2, 3, height, width]).unwrap();
            let input = values(shape.element_count());
            let (output_shape, kind) = super::super::kernels::bind_max_pool(shape).unwrap();
            let mut expected = vec![0.; output_shape.element_count()];
            super::super::kernels::execute(
                &kind,
                &[crate::model::TensorId(0)],
                |_| input.as_slice(),
                &mut expected,
            );
            for backend in backends() {
                let mut actual = vec![f32::NAN; expected.len()];
                max_pool(backend, &input, &mut actual, height, width);
                assert_bits(&actual, &expected);
            }
            if height == 6 {
                let (output_shape, kind) = super::super::kernels::bind_average_pool(shape).unwrap();
                let mut expected = vec![0.; output_shape.element_count()];
                super::super::kernels::execute(
                    &kind,
                    &[crate::model::TensorId(0)],
                    |_| input.as_slice(),
                    &mut expected,
                );
                for backend in backends() {
                    let mut actual = vec![f32::NAN; expected.len()];
                    average_pool(
                        backend,
                        &input,
                        &mut actual,
                        height,
                        width,
                        height / 3,
                        width / 2,
                    );
                    assert_bits(&actual, &expected);
                }
            }
        }
    }

    #[test]
    fn spatial_mean_matches_reference_isa_reduction_order_including_tails() {
        for spatial in [1, 7, 8, 15, 16, 17, 31, 32, 33, 65] {
            let input = values(6 * spatial);
            for backend in backends() {
                let (lanes, pairs) = match backend {
                    Backend::Scalar => (1, false),
                    Backend::Portable4 => (4, false),
                    Backend::Avx512 => (16, true),
                    _ => (8, true),
                };
                let mut expected = vec![0.; 6];
                for (source, output) in input.chunks_exact(spatial).zip(&mut expected) {
                    let mut a = [0f32; 16];
                    let mut b = [0f32; 16];
                    let mut i = 0;
                    let stride = lanes * if pairs { 2 } else { 1 };
                    while i + stride <= spatial {
                        for lane in 0..lanes {
                            a[lane] += source[i + lane];
                            if pairs {
                                b[lane] += source[i + lanes + lane];
                            }
                        }
                        i += stride;
                    }
                    let mut sum = 0.;
                    for lane in 0..lanes {
                        sum += if pairs { a[lane] + b[lane] } else { a[lane] };
                    }
                    for &value in &source[i..] {
                        sum += value;
                    }
                    *output = sum / spatial as f32;
                }
                let mut actual = vec![f32::NAN; 6];
                spatial_mean(backend, &input, &mut actual, spatial);
                assert_bits(&actual, &expected);
            }
        }
    }
    fn backends() -> Vec<Backend> {
        let mut values = vec![Backend::Scalar, Backend::Portable4];
        for (available, backend) in [
            (avx_available(), Backend::Avx),
            (avx_fma_available(), Backend::AvxFma),
            (avx2_fma_available(), Backend::Avx2Fma),
            (avx512_available(), Backend::Avx512),
        ] {
            if available {
                values.push(backend);
            }
        }
        values
    }

    fn values(length: usize) -> Vec<f32> {
        let mut state = 0x1234_5678u32;
        (0..length)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0
            })
            .collect()
    }

    fn assert_bits(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a} vs {b}");
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= 5e-5 + 5e-5 * a.abs().max(b.abs()),
                "element {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn forcing_unsupported_avx2_fails_and_auto_has_a_portable_fallback() {
        assert!(select_backend_with_avx2(KernelPolicy::Avx2Fma, false).is_err());
        assert_eq!(
            select_backend_with_avx2(KernelPolicy::Auto, false).unwrap(),
            Backend::Portable4
        );
        assert_eq!(
            select_backend_with_avx2(KernelPolicy::Auto, true).unwrap(),
            Backend::Avx2Fma
        );
        assert_eq!(
            select_backend(KernelPolicy::Scalar).unwrap(),
            Backend::Scalar
        );
    }

    #[test]
    fn isa_ladder_forcing_and_projection_shape_guards_are_explicit() {
        for (features, expected) in [
            ((false, false, false, false), Backend::Portable4),
            ((true, false, false, false), Backend::Avx),
            ((true, true, false, false), Backend::AvxFma),
            ((true, true, true, false), Backend::Avx2Fma),
            ((true, true, true, true), Backend::Avx512),
        ] {
            let (avx, fma, avx2, avx512) = features;
            assert_eq!(
                select_backend_with_features(KernelPolicy::Auto, avx, fma, avx2, avx512).unwrap(),
                expected
            );
            for (policy, available) in [
                (KernelPolicy::Avx, avx),
                (KernelPolicy::AvxFma, fma),
                (KernelPolicy::Avx2Fma, avx2),
                (KernelPolicy::Avx512, avx512),
            ] {
                assert_eq!(
                    select_backend_with_features(policy, avx, fma, avx2, avx512).is_ok(),
                    available
                );
            }
        }
        for (policy, available) in [
            (KernelPolicy::Avx, avx_available()),
            (KernelPolicy::AvxFma, avx_fma_available()),
            (KernelPolicy::Avx2Fma, avx2_fma_available()),
            (KernelPolicy::Avx512, avx512_available()),
        ] {
            assert_eq!(select_backend(policy).is_ok(), available);
        }
        assert!(wide_projection_eligible(8, 64, 1024, true));
        for (rows, inner, columns, packed) in [
            (7, 64, 1024, true),
            (9, 64, 1024, true),
            (8, 63, 1024, true),
            (8, 64, 1023, true),
            (8, 64, 1024, false),
        ] {
            assert!(!wide_projection_eligible(rows, inner, columns, packed));
        }
        let mut dense = narrow_test_plan(5, 64);
        assert!(!dense.avx512_spatial());
        dense.kernel_height = 3;
        dense.kernel_width = 3;
        dense.pad_top = 1;
        dense.pad_left = 1;
        dense.output_channels = 16;
        assert!(dense.avx512_spatial());
        dense.output_channels = 17;
        assert!(!dense.avx512_spatial());
    }

    #[test]
    fn wide_projection_tiles_and_argmax_match_disjoint_four_row_calls() {
        let (inner, columns) = (64, 1025);
        let weights = values(inner * columns);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let bias = values(columns);
        for rows in [8, 16, 24] {
            let input = values(rows * inner);
            for backend in backends() {
                let mut dense = vec![0.; rows * columns];
                matmul_bias(
                    backend,
                    &input,
                    &weights,
                    Some(&packed),
                    &bias,
                    &mut dense,
                    rows,
                    inner,
                    columns,
                );
                let mut split = vec![0.; dense.len()];
                for (r, out) in split.chunks_mut(4 * columns).enumerate() {
                    matmul_bias(
                        backend,
                        &input[r * 4 * inner..(r + 1) * 4 * inner],
                        &weights,
                        Some(&packed),
                        &bias,
                        out,
                        4,
                        inner,
                        columns,
                    );
                }
                assert_bits(&dense, &split);
                let mut indices = vec![usize::MAX; rows];
                let mut scores = vec![0.; rows];
                compact_argmax(
                    backend,
                    &input,
                    &weights,
                    &bias,
                    Some(&packed),
                    &mut indices,
                    &mut scores,
                    inner,
                    columns,
                )
                .unwrap();
                let compact_expected = argmax::reference_logits(
                    backend, &input, &weights, &bias, rows, inner, columns, true,
                );
                for row in 0..rows {
                    let expected = compact_expected[row * columns..(row + 1) * columns]
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                        .unwrap();
                    assert_eq!(indices[row], expected.0);
                    assert_eq!(scores[row].to_bits(), expected.1.to_bits());
                }
                let mut split_indices = vec![usize::MAX; rows];
                let mut split_scores = vec![f32::NAN; rows];
                for row in (0..rows).step_by(4) {
                    compact_argmax(
                        backend,
                        &input[row * inner..(row + 4) * inner],
                        &weights,
                        &bias,
                        Some(&packed),
                        &mut split_indices[row..row + 4],
                        &mut split_scores[row..row + 4],
                        inner,
                        columns,
                    )
                    .unwrap();
                }
                assert_eq!(indices, split_indices);
                assert_bits(&scores, &split_scores);
            }
        }
    }

    #[test]
    fn matrix_panels_and_unpacked_attention_cover_rows_columns_and_k_tails() {
        for rows in [1, 3, 4, 5, 9] {
            for inner in [1, 3, 17] {
                for columns in [1, 3, 7, 8, 15, 16, 17, 31, 32, 33] {
                    let input = values(rows * inner);
                    let weights = values(inner * columns);
                    let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
                    for backend in backends() {
                        let mut expected = vec![0.0; rows * columns];
                        for row in 0..rows {
                            for col in 0..columns {
                                let mut sum = 0.0f32;
                                for k in 0..inner {
                                    sum = if backend.uses_fma() {
                                        input[row * inner + k]
                                            .mul_add(weights[k * columns + col], sum)
                                    } else {
                                        sum + input[row * inner + k] * weights[k * columns + col]
                                    };
                                }
                                expected[row * columns + col] = sum;
                            }
                        }
                        for packed in [None, Some(&packed)] {
                            let mut actual = vec![f32::NAN; expected.len()];
                            matmul(
                                backend,
                                &input,
                                &weights,
                                packed,
                                &mut actual,
                                rows,
                                inner,
                                columns,
                            );
                            assert_bits(&actual, &expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn biased_dense_matrix_supports_panel_only_input_and_preserves_tails() {
        for (rows, inner, columns) in [
            (1, 3, 5),
            (3, 17, 17),
            (4, 192, 65),
            (7, 3, 16),
            (32, 17, 33),
            (33, 3, 17),
        ] {
            let input = values(rows * inner);
            let weights = values(inner * columns);
            let bias = values(columns);
            let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
            for backend in backends() {
                let mut expected = vec![0.0; rows * columns];
                for row in 0..rows {
                    for col in 0..columns {
                        let mut sum = bias[col];
                        for k in 0..inner {
                            sum = if backend.uses_fma() {
                                input[row * inner + k].mul_add(weights[k * columns + col], sum)
                            } else {
                                sum + input[row * inner + k] * weights[k * columns + col]
                            };
                        }
                        expected[row * columns + col] = sum;
                    }
                }
                for (source, panel) in [
                    (&weights[..], None),
                    (&weights[..], Some(&packed)),
                    (&[][..], Some(&packed)),
                ] {
                    let mut actual = vec![f32::NAN; expected.len()];
                    matmul_bias(
                        backend,
                        &input,
                        source,
                        panel,
                        &bias,
                        &mut actual,
                        rows,
                        inner,
                        columns,
                    );
                    assert_bits(&actual, &expected);
                }
            }
        }
    }

    #[test]
    fn biased_dense_matrix_keeps_bias_first_cancellation_and_negative_zero() {
        let weights = [1e8, 0.0, -1e8, 0.0];
        let packed = PackedMatrix::new(&weights, 2, 2).unwrap();
        for backend in backends() {
            let mut output = [f32::NAN; 10];
            matmul_bias(
                backend,
                &[1.0; 10],
                &[],
                Some(&packed),
                &[1.0, 0.5],
                &mut output,
                5,
                2,
                2,
            );
            assert_eq!(output, [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5]);
            let packed = PackedMatrix::new(&[1.0], 1, 1).unwrap();
            let mut negative_zero = [0.0; 5];
            matmul_bias(
                backend,
                &[-0.0; 5],
                &[],
                Some(&packed),
                &[-0.0],
                &mut negative_zero,
                5,
                1,
                1,
            );
            assert_bits(&negative_zero, &[-0.0; 5]);
        }
    }

    #[test]
    fn compact_scalar_tails_are_bias_first_and_keep_first_ties_across_panels() {
        for backend in backends() {
            let mut indices = [usize::MAX; 5];
            let mut scores = [f32::NAN; 5];
            let weights = [1e8, 0.0, -1e8, 0.0];
            let packed = PackedMatrix::new(&weights, 2, 2).unwrap();
            compact_argmax(
                backend,
                &[1.0; 10],
                &weights,
                &[1.0, 0.5],
                Some(&packed),
                &mut indices,
                &mut scores,
                2,
                2,
            )
            .unwrap();
            assert_eq!(indices, [1; 5]);
            assert_eq!(scores, [0.5; 5]);
            let mut bias = [0.0; 33];
            bias[4] = 2.0;
            bias[16] = 2.0;
            bias[32] = 2.0;
            let packed = PackedMatrix::new(&[0.0; 231], 7, 33).unwrap();
            compact_argmax(
                backend,
                &[1.0; 35],
                &[0.0; 231],
                &bias,
                Some(&packed),
                &mut indices,
                &mut scores,
                7,
                33,
            )
            .unwrap();
            assert_eq!(indices, [4; 5]);
            assert_eq!(scores, [2.0; 5]);
            assert!(
                compact_argmax(
                    backend,
                    &[f32::INFINITY; 5],
                    &[1.0; 3],
                    &[],
                    None,
                    &mut indices,
                    &mut scores,
                    1,
                    3
                )
                .is_err()
            );
        }
    }

    #[test]
    fn compact_projection_matches_each_backends_accumulation_with_real_tails() {
        for (rows, inner, columns) in [(1, 65, 33), (4, 192, 129), (9, 17, 31)] {
            let input = values(rows * inner);
            let weights = values(inner * columns);
            let bias = values(columns);
            let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
            for backend in backends() {
                for packed in [None, Some(&packed)] {
                    let expected = argmax::reference_logits(
                        backend,
                        &input,
                        &weights,
                        &bias,
                        rows,
                        inner,
                        columns,
                        packed.is_some(),
                    );
                    let mut expected_indices = vec![0; rows];
                    let mut expected_scores = vec![f32::NEG_INFINITY; rows];
                    for row in 0..rows {
                        for col in 0..columns {
                            let value = expected[row * columns + col];
                            if value > expected_scores[row] {
                                expected_scores[row] = value;
                                expected_indices[row] = col;
                            }
                        }
                    }
                    let mut indices = vec![usize::MAX; rows];
                    let mut scores = vec![f32::NAN; rows];
                    compact_argmax(
                        backend,
                        &input,
                        &weights,
                        &bias,
                        packed,
                        &mut indices,
                        &mut scores,
                        inner,
                        columns,
                    )
                    .unwrap();
                    assert_eq!(indices, expected_indices);
                    assert_bits(&scores, &expected_scores);
                }
            }
        }
    }

    fn reference_convolution(p: &Conv2d, input: &[f32], weights: &[f32], bias: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; p.batch * p.output_channels * p.output_height * p.output_width];
        let cpg = p.input_channels / p.groups;
        let opg = p.output_channels / p.groups;
        for b in 0..p.batch {
            for co in 0..p.output_channels {
                let group = co / opg;
                for y in 0..p.output_height {
                    for x in 0..p.output_width {
                        let mut sum = if bias.is_empty() { 0.0 } else { bias[co] };
                        for ci in 0..cpg {
                            for ky in 0..p.kernel_height {
                                let sy = (y * p.stride_height + ky * p.dilation_height) as isize
                                    - p.pad_top as isize;
                                if sy < 0 || sy >= p.height as isize {
                                    continue;
                                }
                                for kx in 0..p.kernel_width {
                                    let sx = (x * p.stride_width + kx * p.dilation_width) as isize
                                        - p.pad_left as isize;
                                    if sx < 0 || sx >= p.width as isize {
                                        continue;
                                    }
                                    let value = input[((b * p.input_channels + group * cpg + ci)
                                        * p.height
                                        + sy as usize)
                                        * p.width
                                        + sx as usize];
                                    sum += value
                                        * weights[((co * cpg + ci) * p.kernel_height + ky)
                                            * p.kernel_width
                                            + kx];
                                }
                            }
                        }
                        output[((b * p.output_channels + co) * p.output_height + y)
                            * p.output_width
                            + x] = sum;
                    }
                }
            }
        }
        output
    }

    #[test]
    fn convolution_handles_groups_depthwise_stride_dilation_padding_and_tails() {
        for (batch, cin, cout, groups, height, width, kh, kw, sh, sw, dh, dw, pt, pl) in [
            (2, 6, 10, 2, 5, 19, 3, 3, 1, 1, 1, 1, 1, 1),
            (1, 3, 3, 3, 3, 17, 5, 5, 1, 1, 1, 1, 2, 2),
            (1, 3, 5, 1, 7, 35, 3, 3, 2, 2, 1, 1, 1, 1),
            (1, 4, 8, 2, 5, 35, 2, 3, 1, 1, 2, 2, 1, 2),
            (2, 3, 7, 1, 2, 35, 1, 1, 1, 1, 1, 1, 0, 0),
            (2, 6, 10, 2, 3, 9, 1, 1, 1, 1, 1, 1, 0, 0),
            (1, 2, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3),
        ] {
            let oh = (height + 2 * pt - ((kh - 1) * dh + 1)) / sh + 1;
            let ow = (width + 2 * pl - ((kw - 1) * dw + 1)) / sw + 1;
            let p = Conv2d {
                batch,
                input_channels: cin,
                output_channels: cout,
                height,
                width,
                output_height: oh,
                output_width: ow,
                kernel_height: kh,
                kernel_width: kw,
                stride_height: sh,
                stride_width: sw,
                dilation_height: dh,
                dilation_width: dw,
                pad_top: pt,
                pad_left: pl,
                groups,
            };
            let input = values(batch * cin * height * width);
            let weights = values(cout * (cin / groups) * kh * kw);
            let bias = values(cout);
            for bias in [bias.as_slice(), &[]] {
                let expected = reference_convolution(&p, &input, &weights, bias);
                for backend in backends() {
                    let mut actual = vec![f32::NAN; expected.len()];
                    conv2d(backend, &p, &input, &weights, bias, &mut actual);
                    if backend.uses_fma() && !p.is_direct_pointwise() {
                        assert_close(&actual, &expected);
                    } else {
                        assert_bits(&actual, &expected);
                    }
                }
            }
        }
    }

    #[test]
    fn transpose_convolution_interleaves_and_preserves_batched_width_tails() {
        for width in [1, 3, 4, 7, 8, 9, 17] {
            let (batch, cin, cout, height) = (2, 3, 5, 3);
            let input = values(batch * cin * height * width);
            let weights = values(cin * cout * 4);
            let bias = values(cout);
            let mut expected = vec![0.0; batch * cout * height * width * 4];
            for b in 0..batch {
                for co in 0..cout {
                    for y in 0..height {
                        for x in 0..width {
                            for ky in 0..2 {
                                for kx in 0..2 {
                                    let mut sum = bias[co];
                                    for ci in 0..cin {
                                        sum += input[((b * cin + ci) * height + y) * width + x]
                                            * weights[(ci * cout + co) * 4 + ky * 2 + kx];
                                    }
                                    expected[((b * cout + co) * height * 2 + y * 2 + ky)
                                        * width
                                        * 2
                                        + x * 2
                                        + kx] = sum;
                                }
                            }
                        }
                    }
                }
            }
            for backend in backends() {
                let mut actual = vec![f32::NAN; expected.len()];
                conv_transpose2x2(
                    backend,
                    &input,
                    &weights,
                    &bias,
                    &mut actual,
                    batch,
                    cin,
                    cout,
                    height,
                    width,
                );
                if backend.uses_fma() {
                    assert_close(&actual, &expected);
                } else {
                    assert_bits(&actual, &expected);
                }
            }
        }
    }

    #[test]
    fn output_channel_ranges_match_full_convolution_across_group_boundaries() {
        let p = Conv2d {
            batch: 2,
            input_channels: 6,
            output_channels: 14,
            groups: 2,
            height: 5,
            width: 19,
            output_height: 5,
            output_width: 19,
            kernel_height: 3,
            kernel_width: 3,
            stride_height: 1,
            stride_width: 1,
            dilation_height: 1,
            dilation_width: 1,
            pad_top: 1,
            pad_left: 1,
        };
        let input = values(2 * 6 * 5 * 19);
        let weights = values(14 * 3 * 9);
        let bias = values(14);
        let plane = p.output_height * p.output_width;
        for backend in backends() {
            let mut expected = vec![f32::NAN; 2 * 14 * plane];
            conv2d(backend, &p, &input, &weights, &bias, &mut expected);
            for chunk_channels in [1, 3, 4, 5, 8, 14] {
                let mut actual = vec![f32::NAN; expected.len()];
                for batch in 0..p.batch {
                    let mut first = 0;
                    while first < p.output_channels {
                        let count = chunk_channels.min(p.output_channels - first);
                        let start = (batch * p.output_channels + first) * plane;
                        conv2d_output_range(
                            backend,
                            &p,
                            &input,
                            &weights,
                            &bias,
                            &mut actual[start..start + count * plane],
                            batch,
                            first,
                            count,
                        );
                        first += count;
                    }
                }
                assert_bits(&actual, &expected);
            }
        }
    }

    #[test]
    fn elementary_vectors_preserve_scalar_order_nan_and_signed_zero_semantics() {
        let input = [
            -0.0,
            0.0,
            -1.0,
            1.0,
            2.0,
            1e-20,
            -3.5,
            4.75,
            17.0,
            -8.0,
            0.25,
            f32::NAN,
            f32::INFINITY,
        ];
        let right = [
            1.0, -1.0, 0.0, 2.0, -2.0, 1e-10, 1.5, 0.125, 5.0, 8.0, 1.25, 2.0, -1.0,
        ];
        for backend in backends() {
            for operation in [
                UnaryOp::Square,
                UnaryOp::Relu,
                UnaryOp::Sqrt,
                UnaryOp::Erf,
                UnaryOp::Sigmoid,
                UnaryOp::HardSigmoid {
                    alpha: 0.2,
                    beta: 0.5,
                },
            ] {
                let expected = input
                    .iter()
                    .map(|&x| {
                        if backend == Backend::Scalar {
                            scalar_unary_value(operation, x)
                        } else {
                            unary_value_with_math(operation, x, UnaryMath::for_backend(backend))
                        }
                    })
                    .collect::<Vec<_>>();
                let mut actual = [0.0; 13];
                unary(backend, operation, &input, &mut actual);
                assert_bits(&actual, &expected);
            }
            for operation in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
                let expected = input
                    .iter()
                    .zip(right)
                    .map(|(&a, b)| binary_value(operation, a, b))
                    .collect::<Vec<_>>();
                let mut actual = [0.0; 13];
                binary(backend, operation, &input, &right, &mut actual);
                assert_bits(&actual, &expected);
                for scalar_left in [false, true] {
                    let expected = input
                        .iter()
                        .map(|&x| {
                            if scalar_left {
                                binary_value(operation, 2.5, x)
                            } else {
                                binary_value(operation, x, 2.5)
                            }
                        })
                        .collect::<Vec<_>>();
                    binary_scalar(backend, operation, &input, 2.5, scalar_left, &mut actual);
                    assert_bits(&actual, &expected);
                }
            }
        }
    }

    #[test]
    fn unary_vectors_and_tails_preserve_the_selected_polynomial_bits() {
        let mut input = values(1027)
            .into_iter()
            .map(|x| x * 12.0)
            .collect::<Vec<_>>();
        input[..8].copy_from_slice(&[0.0, -0.0, 1e-20, -1e-20, 0.01, -0.01, 4.0, -4.0]);
        for operation in [UnaryOp::Square, UnaryOp::Erf, UnaryOp::Sigmoid] {
            for backend in backends() {
                let expected = input
                    .iter()
                    .map(|&x| {
                        if backend == Backend::Scalar {
                            scalar_unary_value(operation, x)
                        } else {
                            unary_value_with_math(operation, x, UnaryMath::for_backend(backend))
                        }
                    })
                    .collect::<Vec<_>>();
                let mut actual = vec![f32::NAN; input.len()];
                unary(backend, operation, &input, &mut actual);
                assert_bits(&actual, &expected);
            }
        }
    }

    #[test]
    fn softmax_matches_probabilities_and_preserves_row_partition_bits() {
        for rows in [1, 3] {
            for columns in [1, 3, 4, 7, 8, 15, 16, 17, 33, 129] {
                let input = values(rows * columns)
                    .into_iter()
                    .map(|x| x * 12.0)
                    .collect::<Vec<_>>();
                let mut expected = vec![0.0; input.len()];
                for row in 0..rows {
                    let values = &input[row * columns..(row + 1) * columns];
                    let mut maximum = values[0];
                    for &value in &values[1..] {
                        maximum = maximum.max(value);
                    }
                    let mut sum = 0.0;
                    for col in 0..columns {
                        let value = exp_f32(values[col] - maximum);
                        expected[row * columns + col] = value;
                        sum += value;
                    }
                    for value in &mut expected[row * columns..(row + 1) * columns] {
                        *value /= sum;
                    }
                }
                for backend in backends() {
                    let mut actual = vec![f32::NAN; input.len()];
                    softmax(backend, &input, &mut actual, rows, columns);
                    let native = match backend {
                        Backend::Portable4 => columns >= 4,
                        Backend::Avx | Backend::AvxFma | Backend::Avx2Fma => columns >= 8,
                        Backend::Avx512 => columns >= 8,
                        _ => false,
                    };
                    if native {
                        for (&value, &reference) in actual.iter().zip(&expected) {
                            assert!((value - reference).abs() < 2e-6);
                        }
                    } else {
                        assert_bits(&actual, &expected);
                    }
                    let mut partitioned = vec![f32::NAN; input.len()];
                    for (src, dst) in input
                        .chunks_exact(columns)
                        .zip(partitioned.chunks_exact_mut(columns))
                    {
                        softmax(backend, src, dst, 1, columns);
                    }
                    assert_bits(&actual, &partitioned);
                }
            }
        }
    }

    #[test]
    fn inplace_unary_preserves_primitive_and_fused_scalar_order() {
        let mut input = values(1031)
            .into_iter()
            .map(|x| x * 12.0)
            .collect::<Vec<_>>();
        input[..10].copy_from_slice(&[
            -0.0,
            0.0,
            -1e-30,
            1e-30,
            -3.0,
            3.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            std::f32::consts::SQRT_2,
        ]);
        for operation in [
            UnaryOp::Square,
            UnaryOp::Relu,
            UnaryOp::Sqrt,
            UnaryOp::Erf,
            UnaryOp::Sigmoid,
            UnaryOp::HardSigmoid {
                alpha: 0.16666667,
                beta: 0.5,
            },
            UnaryOp::HardSwish,
            UnaryOp::Gelu,
        ] {
            for backend in backends() {
                let expected = input
                    .iter()
                    .map(|&x| {
                        if backend == Backend::Scalar {
                            scalar_unary_value(operation, x)
                        } else {
                            unary_value_with_math(operation, x, UnaryMath::for_backend(backend))
                        }
                    })
                    .collect::<Vec<_>>();
                let mut actual = input.clone();
                unary_inplace(backend, operation, &mut actual);
                assert_bits(&actual, &expected);
            }
        }
    }

    #[test]
    fn optimized_erf_and_gelu_are_independent_of_tile_peers_and_worker_boundaries() {
        let mut input = values(1031)
            .into_iter()
            .map(|v| v * 9.0)
            .collect::<Vec<_>>();
        input[..12].copy_from_slice(&[
            -0.0, 0.0, -1e-30, 1e-30, -1.0, 1.0, -2.0, 2.0, -4.0, 4.0, -20.0, 20.0,
        ]);
        for operation in [UnaryOp::Erf, UnaryOp::Gelu] {
            for backend in backends() {
                let mut whole = vec![0.0; input.len()];
                unary(backend, operation, &input, &mut whole);
                for chunk in [1, 3, 7, 16, 31, 129] {
                    let mut partitioned = vec![0.0; input.len()];
                    for (src, dst) in input.chunks(chunk).zip(partitioned.chunks_mut(chunk)) {
                        unary(backend, operation, src, dst);
                    }
                    assert_bits(&whole, &partitioned);
                }
            }
        }
    }

    #[test]
    fn inplace_binary_preserves_full_scalar_self_and_operand_order() {
        let mut input = values(1031);
        input[..7].copy_from_slice(&[
            -0.0,
            0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1e-30,
            -1e-30,
        ]);
        let full = values(input.len());
        let scalar = [-0.0];
        for operation in [BinaryOp::Add, BinaryOp::Mul, BinaryOp::Div, BinaryOp::Sub] {
            for other in [None, Some(scalar.as_slice()), Some(full.as_slice())] {
                for left in [false, true] {
                    let expected = input
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| {
                            let rhs = other.map_or(x, |v| v[if v.len() == 1 { 0 } else { i }]);
                            if left {
                                binary_value(operation, rhs, x)
                            } else {
                                binary_value(operation, x, rhs)
                            }
                        })
                        .collect::<Vec<_>>();
                    for backend in backends() {
                        let mut actual = input.clone();
                        binary_inplace(backend, operation, other, left, &mut actual);
                        assert_bits(&actual, &expected);
                    }
                }
            }
        }
    }

    #[test]
    fn channel_broadcast_and_inplace_match_logical_batches_tails_and_left_gates() {
        for channels in [1, 3, 7, 8, 17, 24] {
            for plane in [1, 3, 9, 33] {
                for batches in [1, 2] {
                    for batched in [false, true] {
                        let full = values(batches * channels * plane);
                        let gate = values(channels * if batched { batches } else { 1 });
                        for gate_left in [false, true] {
                            let plan = ChannelBroadcast {
                                channels,
                                plane,
                                batched,
                                gate_left,
                            };
                            for operation in
                                [BinaryOp::Add, BinaryOp::Mul, BinaryOp::Div, BinaryOp::Sub]
                            {
                                for nhwc in [false, true] {
                                    let expected = full
                                        .iter()
                                        .enumerate()
                                        .map(|(i, &x)| {
                                            let n = i / (channels * plane);
                                            let c = if nhwc {
                                                i % channels
                                            } else {
                                                i / plane % channels
                                            };
                                            let rhs =
                                                gate[if batched { n * channels + c } else { c }];
                                            if gate_left {
                                                binary_value(operation, rhs, x)
                                            } else {
                                                binary_value(operation, x, rhs)
                                            }
                                        })
                                        .collect::<Vec<_>>();
                                    for backend in backends() {
                                        let mut actual = full.clone();
                                        binary_channel_inplace(
                                            backend,
                                            operation,
                                            &gate,
                                            plan,
                                            nhwc,
                                            &mut actual,
                                        );
                                        assert_bits(&actual, &expected);
                                        if nhwc {
                                            actual.fill(f32::NAN);
                                            binary_channel(
                                                backend,
                                                operation,
                                                &full,
                                                &gate,
                                                plan,
                                                &mut actual,
                                            );
                                            assert_bits(&actual, &expected);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn narrow_test_plan(kernel: usize, width: usize) -> Conv2d {
        Conv2d {
            batch: 1,
            input_channels: 3,
            output_channels: 5,
            groups: 1,
            height: 8,
            width,
            output_height: 8,
            output_width: width,
            kernel_height: kernel,
            kernel_width: kernel,
            stride_height: 1,
            stride_width: 1,
            dilation_height: 1,
            dilation_width: 1,
            pad_top: kernel / 2,
            pad_left: kernel / 2,
        }
    }

    #[test]
    fn narrow_dense_selection_respects_vector_boundaries_and_excludes_groups() {
        for (kernel, width, portable, avx) in [
            (3, 8, false, true),
            (3, 9, false, true),
            (3, 10, false, false),
            (5, 8, false, true),
            (5, 9, false, true),
            (5, 12, false, false),
            (7, 8, true, true),
            (7, 9, true, true),
            (7, 10, false, true),
            (7, 14, false, false),
        ] {
            let p = narrow_test_plan(kernel, width);
            assert_eq!(p.uses_original_convolution(Backend::Portable4), portable);
            assert_eq!(p.uses_original_convolution(Backend::Avx2Fma), avx);
            assert!(!p.uses_original_convolution(Backend::Scalar));
        }
        for (width, fallback) in [(17, true), (19, false)] {
            let mut p = narrow_test_plan(3, width);
            p.stride_width = 2;
            p.output_width = width.div_ceil(2);
            assert_eq!(p.uses_original_convolution(Backend::Avx2Fma), fallback);
        }
        for (width, fallback) in [(11, true), (12, false)] {
            let mut p = narrow_test_plan(3, width);
            p.dilation_width = 2;
            p.pad_left = 2;
            assert_eq!(p.uses_original_convolution(Backend::Avx2Fma), fallback);
        }
        let mut p = narrow_test_plan(7, 8);
        p.input_channels = 6;
        p.output_channels = 10;
        p.groups = 2;
        assert!(!p.uses_original_convolution(Backend::Avx2Fma));
        p.input_channels = 1;
        p.output_channels = 1;
        p.groups = 1;
        assert!(!p.uses_original_convolution(Backend::Avx2Fma));
    }

    #[test]
    fn narrow_dense_fallback_preserves_bias_order_batches_and_odd_widths() {
        for kernel in [3, 5, 7] {
            for width in [7, 8, 9, 10, 11, 15, 16, 17] {
                for batch in [1, 2] {
                    let mut p = narrow_test_plan(kernel, width);
                    p.batch = batch;
                    let input = values(batch * p.input_channels * p.height * width);
                    let weights = values(p.output_channels * p.input_channels * kernel * kernel);
                    let bias = values(p.output_channels);
                    for bias in [bias.as_slice(), &[]] {
                        let expected = reference_convolution(&p, &input, &weights, bias);
                        for backend in backends() {
                            let mut output = vec![f32::NAN; expected.len()];
                            conv2d(backend, &p, &input, &weights, bias, &mut output);
                            if backend.uses_fma() && !p.uses_original_convolution(backend) {
                                assert_close(&output, &expected);
                            } else {
                                assert_bits(&output, &expected);
                            }
                        }
                    }
                }
            }
        }
        // The fallback must not make an unavailable forced policy acceptable.
        assert!(select_backend_with_avx2(KernelPolicy::Avx2Fma, false).is_err());
        if !avx2_fma_available() {
            let p = narrow_test_plan(3, 8);
            let input = values(3 * 8 * 8);
            let weights = values(5 * 3 * 3 * 3);
            let mut output = vec![0.; 5 * 8 * 8];
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    conv2d(Backend::Avx2Fma, &p, &input, &weights, &[], &mut output)
                }))
                .is_err()
            );
        }
    }

    #[test]
    fn actual_narrow_detector_conv_uses_whole_output_before_parallel_banding() {
        use crate::model::TensorId;
        let mut p = narrow_test_plan(3, 8);
        p.batch = 2;
        p.input_channels = 256;
        p.output_channels = 64;
        let input = values(2 * 256 * 8 * 8);
        let weights = values(64 * 256 * 3 * 3);
        let bias = values(64);
        let kind = crate::runtime::kernels::BoundKind::Conv(p.original_plan());
        let inputs = [TensorId(0), TensorId(1), TensorId(2)];
        let data = |id: TensorId| match id.0 {
            0 => input.as_slice(),
            1 => weights.as_slice(),
            2 => bias.as_slice(),
            _ => unreachable!(),
        };
        let mut expected = vec![0.; 2 * 64 * 8 * 8];
        crate::runtime::kernels::execute(&kind, &inputs, data, &mut expected);
        let cpu =
            crate::cpu::CpuContext::new(if cfg!(feature = "parallel") { 4 } else { 1 }).unwrap();
        for backend in backends() {
            let mut output = vec![f32::NAN; expected.len()];
            crate::runtime::kernels::execute_optimized(
                &kind,
                &inputs,
                data,
                &mut output,
                backend,
                None,
                &cpu,
                cpu.capacity(),
            );
            if backend.uses_fma() || backend == Backend::Scalar {
                assert_bits(&output, &expected);
            } else {
                assert_close(&output, &expected);
            }
        }
    }
}
