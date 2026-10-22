//! Ordered NHWC spatial convolution with bounded worker storage.
//! Eligible macOS ARM64 2x2, 3x3, 5x5 and 7x7 layers pack sixty-four
//! receptive fields for SME.
//! The fallback uses six-row / sixteen-column tiles and 512-K reduction blocks;
//! neither path materializes an image-sized im2col tensor.
use super::*;
mod interior;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod sme;

/// Reuse one bounded receptive-field panel per worker. Small channels retain
/// the direct NEON kernel, where packing cannot be amortized across outputs.
pub(in crate::runtime) fn sme_scratch_elements(backend: Backend, p: &Conv) -> usize {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    if backend == Backend::NeonFma
        && p.cpg == p.cin
        && p.cin >= 32
        && p.cout >= 16
        && matches!(p.kh, 2 | 3 | 5 | 7)
        && p.kw == p.kh
        && p.batch.saturating_mul(p.oh).saturating_mul(p.ow) >= 32
        && super::pointwise::sme::available()
    {
        return p
            .cin
            .checked_mul(p.kh * p.kw * super::pointwise::sme::GROUP)
            .unwrap_or(0);
    }
    let _ = (backend, p);
    0
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_sme(
    backend: Backend,
    p: &Conv,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
    workspaces: &mut [Vec<f32>],
    store: &impl StoreTransform,
) -> bool {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    if sme_scratch_elements(backend, p) > 0 {
        sme::execute_bands(
            p, input, bias, packed, output, cpu, threads, workspaces, store,
        );
        return true;
    }
    let _ = (
        backend, p, input, bias, packed, output, cpu, threads, workspaces, store,
    );
    false
}

const ROWS: usize = 6;
const GROUP_ROWS: usize = ROWS * 8;

#[allow(clippy::too_many_arguments)]
pub(super) fn execute(
    backend: Backend,
    p: &Conv,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    assert!(packed.matches(p.cin * p.kh * p.kw, p.cout));
    assert_eq!(p.cpg, p.cin);
    if p.kh == 1
        && p.kw == 1
        && p.sh == 1
        && p.sw == 1
        && p.pt == 0
        && p.pl == 0
        && p.oh == p.height
        && p.ow == p.width
    {
        super::pointwise::execute(
            backend, input, bias, packed, output, first, p.cin, p.cout, store,
        );
        return;
    }
    match backend {
        Backend::Scalar => kernel::<1, 16, false>(p, input, bias, packed, output, first, store),
        Backend::Portable4 => kernel::<4, 4, false>(p, input, bias, packed, output, first, store),
        Backend::NeonFma => {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: CpuCompiledModel validates the selected NEON/FMA policy.
            unsafe {
                neon(p, input, bias, packed, output, first, store);
            }
        }
        Backend::Avx => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CpuCompiledModel validated the selected AVX backend.
            unsafe {
                avx(p, input, bias, packed, output, first, store);
            }
        }
        Backend::AvxFma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CpuCompiledModel validated AVX and FMA.
            unsafe {
                avx_fma(p, input, bias, packed, output, first, store);
            }
        }
        Backend::Avx2Fma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CpuCompiledModel validated AVX2 and FMA.
            unsafe {
                avx2(p, input, bias, packed, output, first, store);
            }
        }
        Backend::Avx512 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CpuCompiledModel validated AVX-512F, AVX2 and FMA.
            unsafe {
                avx512(p, input, bias, packed, output, first, store);
            }
        }
    }
}

macro_rules! x86_kernel {
    ($name:ident,$features:literal,$lanes:literal,$vectors:literal,$fused:literal) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable=$features)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name(
            p: &Conv,
            input: &[f32],
            bias: &[f32],
            packed: &PackedMatrix,
            output: &mut [f32],
            first: usize,
            store: &impl StoreTransform,
        ) {
            kernel::<$lanes, $vectors, $fused>(p, input, bias, packed, output, first, store);
        }
    };
}
x86_kernel!(avx, "avx", 8, 2, false);
x86_kernel!(avx_fma, "avx,fma", 8, 2, true);
x86_kernel!(avx2, "avx2,fma", 8, 2, true);
x86_kernel!(avx512, "avx512f,avx2,fma", 16, 1, true);

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn neon(
    p: &Conv,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    kernel::<4, 4, true>(p, input, bias, packed, output, first, store);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn kernel<const LANES: usize, const VECTORS: usize, const FUSED: bool>(
    p: &Conv,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    let pixels = output.len() / p.cout;
    let inner = p.cin * p.kh * p.kw;
    let kc = inner.min(512);
    let taps = if (LANES == 8 && VECTORS == 2) || (LANES == 4 && VECTORS == 4) {
        interior::Taps::new(p)
    } else {
        None
    };
    for group in (0..pixels).step_by(GROUP_ROWS) {
        let count = (pixels - group).min(GROUP_ROWS);
        // Every row's source geometry is computed once for the group, including
        // worker bands beginning in the middle of a row or a batch.
        let origins = std::array::from_fn::<_, GROUP_ROWS, _>(|r| {
            let pixel = first + group + r.min(count - 1);
            (pixel / (p.oh * p.ow), pixel / p.ow % p.oh, pixel % p.ow)
        });
        let fields = std::array::from_fn::<_, 8, _>(|tile| {
            taps.as_ref()
                .and_then(|taps| taps.inputs(p, input, &origins[tile * ROWS..(tile + 1) * ROWS]))
        });
        for block in 0..p.cout.div_ceil(16) {
            let mut partial = [[Simd::<f32, LANES>::splat(0.); VECTORS]; GROUP_ROWS];
            for v in 0..VECTORS {
                let column = block * 16 + v * LANES;
                if !bias.is_empty() && column < p.cout {
                    let initial =
                        Simd::load_or_default(&bias[column..(column + LANES).min(p.cout)]);
                    for row in &mut partial {
                        row[v] = initial;
                    }
                }
            }
            for k0 in (0..inner).step_by(kc) {
                let end = (k0 + kc).min(inner);
                for tile in (0..count).step_by(ROWS) {
                    let mut sums = std::array::from_fn::<_, ROWS, _>(|r| partial[tile + r]);
                    #[cfg(target_arch = "aarch64")]
                    if LANES == 4 && VECTORS == 4 && FUSED {
                        let panel =
                            &packed.data()[(block * inner + k0) * 16..(block * inner + end) * 16];
                        // SAFETY: only the guarded NEON entry instantiates
                        // four fused vectors. Descriptors or checked general
                        // gathers prove the complete scalar input interval.
                        sums = unsafe {
                            interior::accumulate_neon(
                                p,
                                input,
                                &origins[tile..tile + ROWS],
                                taps.as_ref(),
                                fields[tile / ROWS].as_ref(),
                                panel,
                                k0,
                                sums,
                            )
                        };
                        partial[tile..tile + ROWS].copy_from_slice(&sums);
                        continue;
                    }
                    if LANES == 4 && VECTORS == 4 && !FUSED {
                        let panel =
                            &packed.data()[(block * inner + k0) * 16..(block * inner + end) * 16];
                        for pair in 0..2 {
                            let seeds =
                                std::array::from_fn(|r| [sums[r][pair * 2], sums[r][pair * 2 + 1]]);
                            // SAFETY: origins contain six valid output rows.
                            // Optional descriptors were built from this same
                            // convolution/input; general gathers stay checked.
                            let paired = unsafe {
                                if pair == 0 {
                                    interior::accumulate_portable::<LANES, 0>(
                                        p,
                                        input,
                                        &origins[tile..tile + ROWS],
                                        taps.as_ref(),
                                        fields[tile / ROWS].as_ref(),
                                        panel,
                                        k0,
                                        seeds,
                                    )
                                } else {
                                    interior::accumulate_portable::<LANES, 8>(
                                        p,
                                        input,
                                        &origins[tile..tile + ROWS],
                                        taps.as_ref(),
                                        fields[tile / ROWS].as_ref(),
                                        panel,
                                        k0,
                                        seeds,
                                    )
                                }
                            };
                            for (sum, values) in sums.iter_mut().zip(&paired) {
                                sum[pair * 2] = values[0];
                                sum[pair * 2 + 1] = values[1];
                            }
                        }
                        partial[tile..tile + ROWS].copy_from_slice(&sums);
                        continue;
                    }
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if let (Some(taps), Some(inputs)) = (&taps, fields[tile / ROWS].as_ref()) {
                        let panel =
                            &packed.data()[(block * inner + k0) * 16..(block * inner + end) * 16];
                        // SAFETY: descriptors are only constructed for the
                        // eight-lane guarded AVX/FMA instantiations. Interior
                        // spans and border indices prevalidate scalar loads.
                        sums = unsafe {
                            interior::accumulate::<LANES, VECTORS, FUSED>(
                                inputs, taps, panel, k0, sums,
                            )
                        };
                        partial[tile..tile + ROWS].copy_from_slice(&sums);
                        continue;
                    }
                    for k in k0..end {
                        let tap = k % (p.kh * p.kw);
                        let ci = k / (p.kh * p.kw);
                        let (ky, kx) = (tap / p.kw, tap % p.kw);
                        let values = std::array::from_fn::<_, ROWS, _>(|r| {
                            let (n, y, x) = origins[tile + r];
                            let sy = y * p.sh + ky * p.dh;
                            let sx = x * p.sw + kx * p.dw;
                            if sy < p.pt
                                || sy - p.pt >= p.height
                                || sx < p.pl
                                || sx - p.pl >= p.width
                            {
                                0.
                            } else {
                                input[((n * p.height + sy - p.pt) * p.width + sx - p.pl) * p.cin
                                    + ci]
                            }
                        });
                        let weights = &packed.data()[(block * inner + k) * 16..][..16];
                        for v in 0..VECTORS {
                            let w = Simd::from_slice(&weights[v * LANES..(v + 1) * LANES]);
                            for r in 0..ROWS {
                                let value = Simd::splat(values[r]);
                                sums[r][v] = if FUSED {
                                    value.mul_add(w, sums[r][v])
                                } else {
                                    value * w + sums[r][v]
                                };
                            }
                        }
                    }
                    partial[tile..tile + ROWS].copy_from_slice(&sums);
                }
            }
            for (row, sums) in partial[..count].iter().enumerate() {
                if cfg!(target_arch = "aarch64") && LANES == 4 && VECTORS == 4 && FUSED {
                    // The complete panel shares one activation/residual
                    // recipe. Widen only the store, preserving every ordered
                    // NEON reduction and the existing masked channel tail.
                    let sum = Simd::<f32, 16>::from_array(std::array::from_fn(|channel| {
                        sums[channel / 4][channel % 4]
                    }));
                    super::pointwise::store_row(
                        sum,
                        output,
                        group + row,
                        block * 16,
                        p.cout,
                        first,
                        store,
                    );
                    continue;
                }
                for (v, &sum) in sums.iter().enumerate() {
                    let column = block * 16 + v * LANES;
                    if column < p.cout {
                        let start = (group + row) * p.cout + column;
                        let count = (p.cout - column).min(LANES);
                        let values = store.vector_channel(sum, first * p.cout + start, column);
                        if count == LANES {
                            values.copy_to_slice(&mut output[start..start + LANES]);
                        } else {
                            output[start..start + count]
                                .copy_from_slice(&values.to_array()[..count]);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod neon_tests;
#[cfg(test)]
mod portable_tests;
#[cfg(test)]
mod selection_tests;
