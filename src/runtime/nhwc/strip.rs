//! Large depthwise kernels accumulate one kernel row per strip pass. Each
//! fixed-size strip holds up to sixteen pixels and four channel vectors;
//! AVX uses the reference's 32-channel block without per-request allocation.
use super::*;

const PIXELS: usize = 16;

/// Output columns whose complete horizontal receptive field is in bounds.
/// Empty ranges retain the checked path, including kernels wider than input.
fn horizontal_interior(p: &Conv) -> std::ops::Range<usize> {
    let first = p.pl.div_ceil(p.sw).min(p.ow);
    let Some(last_origin) = (p.kw - 1)
        .checked_mul(p.dw)
        .and_then(|extent| p.width.checked_sub(extent)?.checked_sub(1))
        .and_then(|last| last.checked_add(p.pl))
    else {
        return first..first;
    };
    let end = (last_origin / p.sw).saturating_add(1).min(p.ow).max(first);
    first..end
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn execute<const LANES: usize, const FUSED: bool>(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    execute_core::<LANES, FUSED, true>(p, input, weights, bias, packed, output, first, store);
}

/// Paired diagnostic control: preserve the previous strip boundaries while
/// sharing every reduction, packed-panel read and final store with production.
#[cfg(test)]
pub(super) fn execute_control<const SEGMENT: bool>(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
) {
    assert!(simd::avx2_fma_available());
    assert!(packed.matches(p.kh * p.kw, p.cout));
    assert_eq!(output.len(), p.batch * p.oh * p.ow * p.cout);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // SAFETY: both controls enter the same AVX2/FMA wrapper, preserving the
    // production ISA context for checked border arithmetic as well as leaves.
    unsafe {
        execute_control_avx2::<SEGMENT>(p, input, weights, bias, packed, output);
    }
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2,fma")]
unsafe fn execute_control_avx2<const SEGMENT: bool>(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
) {
    execute_core::<8, true, SEGMENT>(
        p,
        input,
        weights,
        bias,
        Some(packed),
        output,
        0,
        &IdentityStore,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn execute_core<const LANES: usize, const FUSED: bool, const SEGMENT: bool>(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    let pixels = output.len() / p.cout;
    let vector_channels = p.cout / LANES * LANES;
    let interior_x = if SEGMENT
        && cfg!(target_arch = "x86_64")
        && LANES == 8
        && packed.is_some()
        && vector_channels >= 32
    {
        horizontal_interior(p)
    } else {
        0..0
    };
    let mut local = 0;
    while local < pixels {
        let pixel = first + local;
        let (n, y, x) = (pixel / (p.oh * p.ow), pixel / p.ow % p.oh, pixel % p.ow);
        // Keep true border pixels out of otherwise interior strips. Each band
        // still owns its exact output range, even when it starts within a row.
        let segment_end = if x < interior_x.start {
            interior_x.start
        } else if x < interior_x.end {
            interior_x.end
        } else {
            p.ow
        };
        let count = (pixels - local).min(segment_end - x).min(PIXELS);
        for channel in (0..vector_channels).step_by(LANES * 4) {
            let vectors = ((vector_channels - channel) / LANES).min(4);
            let mut partial = [[Simd::<f32, LANES>::splat(0.0); 4]; PIXELS];
            if !bias.is_empty() {
                for v in 0..vectors {
                    let c = channel + v * LANES;
                    let initial = Simd::from_slice(&bias[c..c + LANES]);
                    for row in &mut partial[..count] {
                        row[v] = initial;
                    }
                }
            }
            let fast = vectors == 4
                && interior::<LANES, FUSED>(
                    p,
                    input,
                    packed,
                    n,
                    y,
                    x,
                    count,
                    channel,
                    &mut partial,
                );
            for ky in 0..if fast { 0 } else { p.kh } {
                let sy = y * p.sh + ky * p.dh;
                if sy < p.pt || sy - p.pt >= p.height {
                    continue;
                }
                for row in (0..count).step_by(2) {
                    let second = (row + 1).min(count - 1);
                    let mut sums = [partial[row], partial[second]];
                    for kx in 0..p.kw {
                        let tap = ky * p.kw + kx;
                        for v in 0..vectors {
                            let c = channel + v * LANES;
                            let weight = if let Some(panel) = packed {
                                let at = ((c / 16) * p.kh * p.kw + tap) * 16 + c % 16;
                                Simd::from_slice(&panel.data()[at..at + LANES])
                            } else {
                                Simd::from_array(std::array::from_fn(|lane| {
                                    weights[(c + lane) * p.kh * p.kw + tap]
                                }))
                            };
                            for (r, sum) in sums.iter_mut().enumerate() {
                                let px = if r == 0 { row } else { second };
                                let sx = (x + px) * p.sw + kx * p.dw;
                                if sx < p.pl || sx - p.pl >= p.width {
                                    continue;
                                }
                                let at =
                                    ((n * p.height + sy - p.pt) * p.width + sx - p.pl) * p.cin + c;
                                let value = Simd::from_slice(&input[at..at + LANES]);
                                sum[v] = if FUSED {
                                    value.mul_add(weight, sum[v])
                                } else {
                                    value * weight + sum[v]
                                };
                            }
                        }
                    }
                    partial[row] = sums[0];
                    if row + 1 < count {
                        partial[row + 1] = sums[1];
                    }
                }
            }
            for (row, sums) in partial[..count].iter().enumerate() {
                for (v, &sum) in sums[..vectors].iter().enumerate() {
                    let at = (local + row) * p.cout + channel + v * LANES;
                    store
                        .vector(sum, first * p.cout + at)
                        .copy_to_slice(&mut output[at..at + LANES]);
                }
            }
        }
        // Match the original optimized scalar arithmetic for the final
        // non-vector channels, including each row's skipped padding taps.
        for row in 0..count {
            for channel in vector_channels..p.cout {
                depthwise_pixels::<1, false, 1>(
                    p,
                    input,
                    weights,
                    bias,
                    packed,
                    output,
                    local + row,
                    n,
                    y,
                    x + row,
                    channel,
                    first,
                    store,
                );
            }
        }
        local += count;
    }
}

/// Bounds and packed-panel geometry are established before the tap reduction.
/// Border strips, non-native policies and short channel blocks keep the general
/// path above. A strip may begin in the middle of a worker band or image row.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn interior<const LANES: usize, const FUSED: bool>(
    p: &Conv,
    input: &[f32],
    packed: Option<&PackedMatrix>,
    n: usize,
    y: usize,
    x: usize,
    count: usize,
    channel: usize,
    partial: &mut [[Simd<f32, LANES>; 4]; PIXELS],
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if LANES == 8
        && let Some(packed) = packed
    {
        let left = x * p.sw;
        let right = (x + count - 1) * p.sw + (p.kw - 1) * p.dw;
        if left < p.pl || right < p.pl || right - p.pl >= p.width {
            return false;
        }
        let taps = p.kh * p.kw;
        let base = channel / 16 * taps * 16;
        let low = packed.data()[base..base + taps * 16].as_chunks::<16>().0;
        let high = packed.data()[base + taps * 16..base + taps * 32]
            .as_chunks::<16>()
            .0;
        let pixel_stride = p.sw * p.cin;
        let tap_stride = p.dw * p.cin;
        let span = (count - 1) * pixel_stride + (p.kw - 1) * tap_stride + 32;
        for ky in 0..p.kh {
            let sy = y * p.sh + ky * p.dh;
            if sy < p.pt || sy - p.pt >= p.height {
                continue;
            }
            let at = ((n * p.height + sy - p.pt) * p.width + left - p.pl) * p.cin + channel;
            let input = &input[at..at + span];
            let low = &low[ky * p.kw..(ky + 1) * p.kw];
            let high = &high[ky * p.kw..(ky + 1) * p.kw];
            // SAFETY: native depthwise dispatch checked AVX (and FMA when
            // FUSED). LANES is eight; slices cover every pixel/tap/32-channel
            // read and both complete OC16 weight panels for this kernel row.
            unsafe {
                if FUSED {
                    row_fma(input, low, high, pixel_stride, tap_stride, count, partial);
                } else {
                    row_avx(input, low, high, pixel_stride, tap_stride, count, partial);
                }
            }
        }
        return true;
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (p, input, packed, n, y, x, count, channel, partial);
    false
}

macro_rules! native_row {
    ($name:ident, $features:literal, $fused:literal) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $features)]
        #[inline(never)]
        unsafe fn $name<const LANES: usize>(
            input: &[f32],
            low: &[[f32; 16]],
            high: &[[f32; 16]],
            pixel_stride: usize,
            tap_stride: usize,
            count: usize,
            partial: &mut [[Simd<f32, LANES>; 4]; PIXELS],
        ) {
            for row in (0..count).step_by(2) {
                let second = (row + 1).min(count - 1);
                let [mut a0, mut a1, mut a2, mut a3] = partial[row];
                let [mut b0, mut b1, mut b2, mut b3] = partial[second];
                for (kx, (lo, hi)) in low.iter().zip(high).enumerate() {
                    let at0 = row * pixel_stride + kx * tap_stride;
                    let at1 = second * pixel_stride + kx * tap_stride;
                    macro_rules! update {
                        ($a:ident, $b:ident, $weights:expr, $offset:expr) => {{
                            let w = Simd::<f32, LANES>::from_slice($weights);
                            // SAFETY: interior's span includes both pixel
                            // offsets, every tap, and all four native vectors.
                            let (v0, v1) = unsafe {
                                (
                                    Simd::from_array(
                                        input
                                            .as_ptr()
                                            .add(at0 + $offset)
                                            .cast::<[f32; LANES]>()
                                            .read_unaligned(),
                                    ),
                                    Simd::from_array(
                                        input
                                            .as_ptr()
                                            .add(at1 + $offset)
                                            .cast::<[f32; LANES]>()
                                            .read_unaligned(),
                                    ),
                                )
                            };
                            if $fused {
                                $a = v0.mul_add(w, $a);
                                $b = v1.mul_add(w, $b);
                            } else {
                                $a = v0 * w + $a;
                                $b = v1 * w + $b;
                            }
                        }};
                    }
                    update!(a0, b0, &lo[..8], 0);
                    update!(a1, b1, &lo[8..], 8);
                    update!(a2, b2, &hi[..8], 16);
                    update!(a3, b3, &hi[8..], 24);
                }
                partial[row] = [a0, a1, a2, a3];
                if row + 1 < count {
                    partial[row + 1] = [b0, b1, b2, b3];
                }
            }
        }
    };
}
native_row!(row_avx, "avx", false);
native_row!(row_fma, "avx,fma", true);

#[cfg(test)]
mod tests;
