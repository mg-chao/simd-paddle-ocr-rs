//! Keep the eight AVX accumulation chains independent of fused epilogue state.
//! Inlining PreparedStore into the reduction's caller otherwise keeps recipe
//! values live across ci/ky/kx and spills active accumulators on every row.
//! One completed packet crosses this ISA-specific boundary before its ordinary
//! ordered stores; arithmetic, weight panels and scalar borders stay unchanged.
use super::*;

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn eight<const FMA: bool, const STRIDE: usize>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    batch: usize,
    co: usize,
    y: usize,
    x: usize,
) -> [Simd<f32, 8>; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: only guarded NCHW native N8 entries call this reduction.
        // They already require AVX, and additionally FMA when FMA is true.
        unsafe {
            if FMA {
                avx_fma::<STRIDE>(p, input, bias, packed, batch, co, y, x)
            } else {
                avx::<STRIDE>(p, input, bias, packed, batch, co, y, x)
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    reduce::<FMA, STRIDE>(p, input, bias, packed, batch, co, y, x)
}

/// Scalar spatial positions still have eight contiguous coefficients. Group
/// their independent output chains without multiplying any skipped padding
/// tap, and keep separate multiply/add even under a fused native policy.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn border<const STRIDE: usize>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    batch: usize,
    co: usize,
    y: usize,
    x: usize,
) -> Simd<f32, 8> {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: only the same guarded native N8 entries call this helper.
        // The border intentionally requires AVX alone, including FMA policies.
        unsafe { border_avx::<STRIDE>(p, input, bias, packed, batch, co, y, x) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    border_impl::<STRIDE>(p, input, bias, packed, batch, co, y, x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn border_avx<const STRIDE: usize>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    batch: usize,
    co: usize,
    y: usize,
    x: usize,
) -> Simd<f32, 8> {
    border_impl::<STRIDE>(p, input, bias, packed, batch, co, y, x)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn border_impl<const STRIDE: usize>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    batch: usize,
    co: usize,
    y: usize,
    x: usize,
) -> Simd<f32, 8> {
    const {
        assert!(STRIDE == 1 || STRIDE == 2);
    }
    let inner = p.input_channels * 9;
    let mut sum = initial::<8>(bias, co, p.output_channels);
    for ci in 0..p.input_channels {
        for ky in 0..3 {
            let iy = (y * STRIDE + ky) as isize - p.pad_top as isize;
            if iy < 0 || iy >= p.height as isize {
                continue;
            }
            for kx in 0..3 {
                let ix = (x * STRIDE + kx) as isize - p.pad_left as isize;
                if ix < 0 || ix >= p.width as isize {
                    continue;
                }
                let value = input[((batch * p.input_channels + ci) * p.height + iy as usize)
                    * p.width
                    + ix as usize];
                let weights =
                    coefficient::<8>(packed, inner, (ci * 3 + ky) * 3 + kx, co, p.output_channels);
                // edge() is sum + value*weight, with no contraction and with
                // skipped taps absent from the chain (also for Inf and -0).
                sum += Simd::splat(value) * weights;
            }
        }
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn avx<const STRIDE: usize>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    batch: usize,
    co: usize,
    y: usize,
    x: usize,
) -> [Simd<f32, 8>; 8] {
    reduce::<false, STRIDE>(p, input, bias, packed, batch, co, y, x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn avx_fma<const STRIDE: usize>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    batch: usize,
    co: usize,
    y: usize,
    x: usize,
) -> [Simd<f32, 8>; 8] {
    reduce::<true, STRIDE>(p, input, bias, packed, batch, co, y, x)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn reduce<const FMA: bool, const STRIDE: usize>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    batch: usize,
    co: usize,
    y: usize,
    x: usize,
) -> [Simd<f32, 8>; 8] {
    const {
        assert!(STRIDE == 1 || STRIDE == 2);
    }
    let inner = p.input_channels * 9;
    let low = (co / 16) * inner * 16 + co % 16;
    let mut s0 = Simd::splat(if bias.is_empty() { 0. } else { bias[co] });
    let mut s1 = Simd::splat(if bias.is_empty() { 0. } else { bias[co + 1] });
    let mut s2 = Simd::splat(if bias.is_empty() { 0. } else { bias[co + 2] });
    let mut s3 = Simd::splat(if bias.is_empty() { 0. } else { bias[co + 3] });
    let mut s4 = Simd::splat(if bias.is_empty() { 0. } else { bias[co + 4] });
    let mut s5 = Simd::splat(if bias.is_empty() { 0. } else { bias[co + 5] });
    let mut s6 = Simd::splat(if bias.is_empty() { 0. } else { bias[co + 6] });
    let mut s7 = Simd::splat(if bias.is_empty() { 0. } else { bias[co + 7] });
    for ci in 0..p.input_channels {
        for ky in 0..3 {
            let row = ((batch * p.input_channels + ci) * p.height + y * STRIDE + ky - 1) * p.width;
            for kx in 0..3 {
                let at = x * STRIDE - 1 + kx;
                let value = if STRIDE == 1 {
                    Simd::<f32, 8>::from_slice(&input[row + at..][..8])
                } else {
                    // Retain stride2's exact row-bounded gather, including its
                    // 15-element source tail: no cross-row extra read is added.
                    stride2::load_even::<8>(&input[row..row + p.width][at..])
                };
                let k = (ci * 3 + ky) * 3 + kx;
                let weights = &packed.data()[low + k * 16..low + k * 16 + 8];
                s0 = madd::<8, FMA>(value, Simd::splat(weights[0]), s0);
                s1 = madd::<8, FMA>(value, Simd::splat(weights[1]), s1);
                s2 = madd::<8, FMA>(value, Simd::splat(weights[2]), s2);
                s3 = madd::<8, FMA>(value, Simd::splat(weights[3]), s3);
                s4 = madd::<8, FMA>(value, Simd::splat(weights[4]), s4);
                s5 = madd::<8, FMA>(value, Simd::splat(weights[5]), s5);
                s6 = madd::<8, FMA>(value, Simd::splat(weights[6]), s6);
                s7 = madd::<8, FMA>(value, Simd::splat(weights[7]), s7);
            }
        }
    }
    [s0, s1, s2, s3, s4, s5, s6, s7]
}
