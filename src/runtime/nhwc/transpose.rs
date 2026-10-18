//! Non-overlapping 2x2/stride-two transpose convolutions. Packed tiles write
//! directly to exclusive output-row pairs; single-channel lanes own pixels.
use super::*;

/// Local macOS graph measurements favor direct stores for the larger ARM64 head.
/// Narrow heads retain the existing matrix path; their benefit was inconclusive.
pub(super) fn neon_direct_eligible(backend: Backend, p: &kernels::ConvTranspose) -> bool {
    cfg!(all(target_arch = "aarch64", target_os = "macos"))
        && backend == Backend::NeonFma
        && p.cin >= 64
        && p.cout >= 64
}

/// Reuse the native four-row GEMM reduction, writing its channel panels
/// directly to their interleaved 2x2 destinations without output scratch.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn neon_packed_direct(
    p: &kernels::ConvTranspose,
    input: &[f32],
    bias: &[f32],
    panels: &[PackedMatrix; 4],
    output: &mut [f32],
    first_row: usize,
    store: &impl StoreTransform,
) {
    assert!(p.cin > 0 && p.cout > 0 && p.width > 0);
    assert!(output.len().is_multiple_of(4 * p.width * p.cout));
    let pixels = output.len() / (4 * p.cout);
    let first = first_row * p.width;
    assert!(first + pixels <= p.batch * p.height * p.width);
    assert_eq!(input.len(), p.batch * p.height * p.width * p.cin);
    assert!(bias.is_empty() || bias.len() == p.cout);
    assert!(panels.iter().all(|panel| panel.matches(p.cin, p.cout)));
    let input = &input[first * p.cin..(first + pixels) * p.cin];
    let mut pixel = 0;
    while pixel + 4 <= pixels {
        // SAFETY: four complete input rows and all destination extents exist.
        unsafe { neon_direct_tile::<4>(p, input, bias, panels, output, first_row, pixel, store) };
        pixel += 4;
    }
    while pixel < pixels {
        // SAFETY: one complete input row and its destination extent exist.
        unsafe { neon_direct_tile::<1>(p, input, bias, panels, output, first_row, pixel, store) };
        pixel += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn neon_direct_tile<const ROWS: usize>(
    p: &kernels::ConvTranspose,
    input: &[f32],
    bias: &[f32],
    panels: &[PackedMatrix; 4],
    output: &mut [f32],
    first_row: usize,
    pixel: usize,
    store: &impl StoreTransform,
) {
    let destinations: [usize; ROWS] = std::array::from_fn(|r| {
        let local = pixel + r;
        (local / p.width * 4 * p.width + local % p.width * 2) * p.cout
    });
    for (tap, panel) in panels.iter().enumerate() {
        let tap_offset = (tap / 2 * 2 * p.width + tap % 2) * p.cout;
        for channel in (0..p.cout).step_by(16) {
            // SAFETY: the entry enables NEON and validates all matrix spans.
            let sums = unsafe {
                simd::packed_bias_tile::<ROWS>(input, panel, bias, pixel, channel, p.cin, p.cout)
            };
            for (r, sum) in sums.into_iter().enumerate() {
                let at = destinations[r] + tap_offset + channel;
                let global = first_row * 4 * p.width * p.cout + at;
                if channel + 16 <= p.cout {
                    store
                        .vector_channel(sum, global, channel)
                        .copy_to_slice(&mut output[at..at + 16]);
                } else {
                    // Never read residual/scale operands beyond a channel tail.
                    for (lane, value) in sum
                        .to_array()
                        .into_iter()
                        .take(p.cout - channel)
                        .enumerate()
                    {
                        output[at + lane] = store.scalar(value, global + lane);
                    }
                }
            }
        }
    }
}

/// Four source pixels share each packed coefficient walk. The existing GEMM
/// reduction is inlined here, with its bias-first ascending-channel FMA order;
/// only the destination changes from scratch to the final interleaved image.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn packed_direct(
    p: &kernels::ConvTranspose,
    input: &[f32],
    bias: &[f32],
    panels: &[PackedMatrix; 4],
    output: &mut [f32],
    first_row: usize,
    store: &impl StoreTransform,
) {
    assert!(p.cin > 0 && p.cout > 0 && p.width > 0);
    assert!(output.len().is_multiple_of(4 * p.width * p.cout));
    let pixels = output.len() / (4 * p.cout);
    let first = first_row * p.width;
    assert!(first + pixels <= p.batch * p.height * p.width);
    assert_eq!(input.len(), p.batch * p.height * p.width * p.cin);
    assert!(bias.is_empty() || bias.len() == p.cout);
    assert!(panels.iter().all(|panel| panel.matches(p.cin, p.cout)));
    let input = &input[first * p.cin..(first + pixels) * p.cin];
    let mut pixel = 0;
    while pixel + 4 <= pixels {
        direct_tile::<4>(p, input, bias, panels, output, first_row, pixel, store);
        pixel += 4;
    }
    while pixel < pixels {
        direct_tile::<1>(p, input, bias, panels, output, first_row, pixel, store);
        pixel += 1;
    }
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn direct_tile<const ROWS: usize>(
    p: &kernels::ConvTranspose,
    input: &[f32],
    bias: &[f32],
    panels: &[PackedMatrix; 4],
    output: &mut [f32],
    first_row: usize,
    pixel: usize,
    store: &impl StoreTransform,
) {
    let destinations: [usize; ROWS] = std::array::from_fn(|r| {
        let local = pixel + r;
        (local / p.width * 4 * p.width + local % p.width * 2) * p.cout
    });
    for (tap, panel) in panels.iter().enumerate() {
        let tap_offset = (tap / 2 * 2 * p.width + tap % 2) * p.cout;
        for block in 0..p.cout.div_ceil(16) {
            let sums = simd::projection_tile::<8, 2, true, ROWS, true>(
                input,
                panel.data(),
                bias,
                pixel,
                block,
                p.cin,
                p.cout,
            );
            for (r, vectors) in sums.into_iter().enumerate() {
                for (v, sum) in vectors.into_iter().enumerate() {
                    let channel = block * 16 + v * 8;
                    if channel >= p.cout {
                        continue;
                    }
                    let at = destinations[r] + tap_offset + channel;
                    let global = first_row * 4 * p.width * p.cout + at;
                    if channel + 8 <= p.cout {
                        store
                            .vector_channel(sum, global, channel)
                            .copy_to_slice(&mut output[at..at + 8]);
                    } else {
                        // A partial vector must not read a fused residual or
                        // channel operand beyond this pixel's actual extent.
                        for (lane, value) in sum
                            .to_array()
                            .into_iter()
                            .take(p.cout - channel)
                            .enumerate()
                        {
                            output[at + lane] = store.scalar(value, global + lane);
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn single(
    backend: Backend,
    p: &kernels::ConvTranspose,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    first_row: usize,
    store: &impl StoreTransform,
) {
    assert_eq!(p.cout, 1);
    assert!(output.len().is_multiple_of(4 * p.width));
    match backend {
        Backend::Scalar => kernel::<1, false>(p, input, weights, bias, output, first_row, store),
        Backend::Portable4 => kernel::<4, false>(p, input, weights, bias, output, first_row, store),
        Backend::NeonFma => kernel::<4, true>(p, input, weights, bias, output, first_row, store),
        Backend::Avx => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: the compiled model validated AVX.
            unsafe {
                avx(p, input, weights, bias, output, first_row, store);
            }
        }
        Backend::AvxFma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: the compiled model validated AVX/FMA.
            unsafe {
                avx_fma(p, input, weights, bias, output, first_row, store);
            }
        }
        Backend::Avx2Fma | Backend::Avx512 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both compiled policies require AVX2 and FMA.
            unsafe {
                avx2(p, input, weights, bias, output, first_row, store);
            }
        }
    }
}
macro_rules! x86_kernel {
    ($name:ident,$features:literal,$fused:literal) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable=$features)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name(
            p: &kernels::ConvTranspose,
            input: &[f32],
            weights: &[f32],
            bias: &[f32],
            output: &mut [f32],
            first_row: usize,
            store: &impl StoreTransform,
        ) {
            kernel::<8, $fused>(p, input, weights, bias, output, first_row, store);
        }
    };
}
x86_kernel!(avx, "avx", false);
x86_kernel!(avx_fma, "avx,fma", true);
x86_kernel!(avx2, "avx2,fma", true);

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn kernel<const LANES: usize, const FUSED: bool>(
    p: &kernels::ConvTranspose,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    first_row: usize,
    store: &impl StoreTransform,
) {
    let rows = output.len() / (4 * p.width);
    assert!(first_row + rows <= p.batch * p.height);
    let initial = Simd::<f32, LANES>::splat(bias.first().copied().unwrap_or(0.));
    for row in 0..rows {
        for x in (0..p.width).step_by(LANES) {
            let count = (p.width - x).min(LANES);
            let mut sums = [initial; 4];
            for ci in 0..p.cin {
                let values = Simd::from_array(std::array::from_fn(|lane| {
                    input[((first_row + row) * p.width + x + lane.min(count - 1)) * p.cin + ci]
                }));
                for (tap, sum) in sums.iter_mut().enumerate() {
                    let weight = Simd::splat(weights[ci * 4 + tap]);
                    *sum = if FUSED {
                        values.mul_add(weight, *sum)
                    } else {
                        values * weight + *sum
                    };
                }
            }
            for (tap, sum) in sums.into_iter().enumerate() {
                for (lane, value) in sum.to_array().into_iter().enumerate().take(count) {
                    let at = (row * 2 + tap / 2) * p.width * 2 + (x + lane) * 2 + tap % 2;
                    output[at] = store.scalar(value, first_row * 4 * p.width + at);
                }
            }
        }
    }
}

#[cfg(all(
    test,
    any(
        target_arch = "aarch64",
        all(target_os = "windows", target_arch = "x86_64")
    )
))]
mod tests;
