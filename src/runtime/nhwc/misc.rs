//! Channel-vector pooling and spatial means. Reductions keep the original
//! per-channel pixel order; changing vector width never reassociates sums.
use super::*;
use std::simd::num::SimdFloat;

#[allow(clippy::too_many_arguments)]
pub(super) fn batch_norm(
    backend: Backend,
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    scale: &[f32],
    bias: &[f32],
    mean: &[f32],
    variance: &[f32],
    epsilon: f32,
    workspace: &mut [f32],
) {
    assert!(workspace.len() >= channels);
    let multiplier = &mut workspace[..channels];
    for c in 0..channels {
        multiplier[c] = scale[c] / (variance[c] + epsilon).sqrt();
    }
    match backend {
        Backend::Scalar => batch_norm_kernel::<1>(input, output, channels, multiplier, bias, mean),
        Backend::Portable4 => {
            batch_norm_kernel::<4>(input, output, channels, multiplier, bias, mean)
        }
        Backend::Avx => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: the compiled model validated this AVX backend.
            unsafe {
                batch_norm_avx(input, output, channels, multiplier, bias, mean);
            }
        }
        Backend::AvxFma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: the compiled model validated AVX and FMA.
            unsafe {
                batch_norm_avx_fma(input, output, channels, multiplier, bias, mean);
            }
        }
        Backend::Avx2Fma | Backend::Avx512 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both compiled policies require AVX2 and FMA.
            unsafe {
                batch_norm_avx2(input, output, channels, multiplier, bias, mean);
            }
        }
    }
}

macro_rules! x86_batch_norm {
    ($name:ident,$features:literal) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable=$features)]
        unsafe fn $name(
            input: &[f32],
            output: &mut [f32],
            channels: usize,
            multiplier: &[f32],
            bias: &[f32],
            mean: &[f32],
        ) {
            batch_norm_kernel::<8>(input, output, channels, multiplier, bias, mean);
        }
    };
}
x86_batch_norm!(batch_norm_avx, "avx");
x86_batch_norm!(batch_norm_avx_fma, "avx,fma");
x86_batch_norm!(batch_norm_avx2, "avx2,fma");

#[inline(always)]
fn batch_norm_kernel<const LANES: usize>(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    multiplier: &[f32],
    bias: &[f32],
    mean: &[f32],
) {
    let end = channels / LANES * LANES;
    for (src, dst) in input
        .chunks_exact(channels)
        .zip(output.chunks_exact_mut(channels))
    {
        for c in (0..end).step_by(LANES) {
            let x = Simd::<f32, LANES>::from_slice(&src[c..c + LANES]);
            let mean = Simd::from_slice(&mean[c..c + LANES]);
            let k = Simd::from_slice(&multiplier[c..c + LANES]);
            let bias = Simd::from_slice(&bias[c..c + LANES]);
            ((x - mean) * k + bias).copy_to_slice(&mut dst[c..c + LANES]);
        }
        for c in end..channels {
            dst[c] = (src[c] - mean[c]) * multiplier[c] + bias[c];
        }
    }
}

pub(super) fn execute(
    backend: Backend,
    kind: &BoundKind,
    input: &[f32],
    shape: Shape,
    output: &mut [f32],
) {
    execute_range(backend, kind, input, shape, output, 0);
}

pub(super) fn execute_range(
    backend: Backend,
    kind: &BoundKind,
    input: &[f32],
    shape: Shape,
    output: &mut [f32],
    first: usize,
) {
    match backend {
        Backend::Scalar => kernel::<1>(kind, input, shape, output, first),
        Backend::Portable4 => kernel::<4>(kind, input, shape, output, first),
        Backend::Avx => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validated AVX for this backend.
            unsafe {
                avx(kind, input, shape, output, first);
            }
        }
        Backend::AvxFma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validated AVX and FMA.
            unsafe {
                avx_fma(kind, input, shape, output, first);
            }
        }
        Backend::Avx2Fma | Backend::Avx512 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both policies require AVX2+FMA. The reference keeps
            // these miscellaneous NHWC kernels at 256 bits on AVX-512 hosts.
            unsafe {
                avx2(kind, input, shape, output, first);
            }
        }
    }
}

macro_rules! x86_kernel {
    ($name:ident,$features:literal) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable=$features)]
        unsafe fn $name(
            kind: &BoundKind,
            input: &[f32],
            shape: Shape,
            output: &mut [f32],
            first: usize,
        ) {
            kernel::<8>(kind, input, shape, output, first);
        }
    };
}
x86_kernel!(avx, "avx");
x86_kernel!(avx_fma, "avx,fma");
x86_kernel!(avx2, "avx2,fma");

#[inline(always)]
fn kernel<const LANES: usize>(
    kind: &BoundKind,
    input: &[f32],
    shape: Shape,
    output: &mut [f32],
    first: usize,
) {
    let dims = shape.dimensions();
    let channels = dims[1];
    let vector_end = channels / LANES * LANES;
    match kind {
        BoundKind::MaxPool { height, width } => {
            for (local, destination) in output.chunks_exact_mut(channels).enumerate() {
                let pixel = first + local;
                let (n, y, x) = (
                    pixel / (height * width),
                    pixel / width % height,
                    pixel % width,
                );
                let source = |yy: usize, xx: usize| ((n * height + yy) * width + xx) * channels;
                let at = source(y, x);
                let right = source(y, (x + 1).min(width - 1));
                let bottom = source((y + 1).min(height - 1), x);
                let corner = source((y + 1).min(height - 1), (x + 1).min(width - 1));
                for c in (0..vector_end).step_by(LANES) {
                    let a = Simd::<f32, LANES>::from_slice(&input[at + c..at + c + LANES]);
                    let b = Simd::from_slice(&input[right + c..right + c + LANES]);
                    let d = Simd::from_slice(&input[bottom + c..bottom + c + LANES]);
                    let e = Simd::from_slice(&input[corner + c..corner + c + LANES]);
                    a.simd_max(b)
                        .simd_max(d.simd_max(e))
                        .copy_to_slice(&mut destination[c..c + LANES]);
                }
                for c in vector_end..channels {
                    destination[c] = input[at + c]
                        .max(input[right + c])
                        .max(input[bottom + c].max(input[corner + c]));
                }
            }
        }
        BoundKind::AveragePool {
            height,
            width,
            output_height,
            output_width,
        } => {
            for (local, destination) in output.chunks_exact_mut(channels).enumerate() {
                let pixel = first + local;
                let (n, y, x) = (
                    pixel / (output_height * output_width),
                    pixel / output_width % output_height,
                    pixel % output_width,
                );
                for c in (0..vector_end).step_by(LANES) {
                    let mut sum = Simd::<f32, LANES>::splat(0.);
                    for ky in 0..3 {
                        for kx in 0..2 {
                            let source =
                                ((n * height + y * 3 + ky) * width + x * 2 + kx) * channels + c;
                            sum += Simd::from_slice(&input[source..source + LANES]);
                        }
                    }
                    (sum / Simd::splat(6.)).copy_to_slice(&mut destination[c..c + LANES]);
                }
                for c in vector_end..channels {
                    let mut sum = 0.;
                    for ky in 0..3 {
                        for kx in 0..2 {
                            sum += input
                                [((n * height + y * 3 + ky) * width + x * 2 + kx) * channels + c];
                        }
                    }
                    destination[c] = sum / 6.;
                }
            }
        }
        BoundKind::Resize {
            height,
            width,
            factor,
        } => {
            resize_nearest_range(input, output, channels, *height, *width, *factor, first);
        }
        BoundKind::Reduce(p) => {
            let plane = dims[2] * dims[3];
            for n in 0..dims[0] {
                for c in (0..vector_end).step_by(LANES) {
                    let mut sum = Simd::<f32, LANES>::splat(0.);
                    for pixel in 0..plane {
                        let source = (n * plane + pixel) * channels + c;
                        sum += Simd::from_slice(&input[source..source + LANES]);
                    }
                    (sum / Simd::splat(p.reduction as f32))
                        .copy_to_slice(&mut output[n * channels + c..n * channels + c + LANES]);
                }
                for c in vector_end..channels {
                    let mut sum = 0.;
                    for pixel in 0..plane {
                        sum += input[(n * plane + pixel) * channels + c];
                    }
                    output[n * channels + c] = sum / p.reduction as f32;
                }
            }
        }
        _ => unreachable!("selected NHWC miscellaneous operator"),
    }
}

/// Each complete source row is expanded horizontally once per worker band;
/// following output rows with that same source identity copy the completed row.
/// Partial leading/trailing rows remain bounded pixel copies. All scratch is
/// already-owned destination storage, and each band stays inside its own slice.
fn resize_nearest_range(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    height: usize,
    width: usize,
    factor: usize,
    first: usize,
) {
    let (oh, ow) = (height * factor, width * factor);
    let pixels = output.len() / channels;
    let row_elements = ow * channels;
    let mut local = 0;
    while local < pixels {
        let pixel = first + local;
        let (n, y, x) = (pixel / (oh * ow), pixel / ow % oh, pixel % ow);
        if x != 0 || pixels - local < ow {
            let count = (ow - x).min(pixels - local);
            for column in 0..count {
                let source = ((n * height + y / factor) * width + (x + column) / factor) * channels;
                let start = (local + column) * channels;
                output[start..start + channels].copy_from_slice(&input[source..source + channels]);
            }
            local += count;
            continue;
        }
        let source = (n * height + y / factor) * width * channels;
        let start = local * channels;
        let source_row = &input[source..source + width * channels];
        let first_row = &mut output[start..start + row_elements];
        for (pixel, repeated) in source_row
            .chunks_exact(channels)
            .zip(first_row.chunks_exact_mut(factor * channels))
        {
            for destination in repeated.chunks_exact_mut(channels) {
                destination.copy_from_slice(pixel);
            }
        }
        // Stop at the next source row or this worker's destination boundary.
        // This also prevents a repeat from crossing into the following batch.
        let copies = ((pixels - local) / ow).min(factor - y % factor);
        for row in 1..copies {
            output.copy_within(start..start + row_elements, start + row * row_elements);
        }
        local += copies * ow;
    }
}

#[cfg(test)]
mod resize_tests;
