//! ARM64 packed depthwise convolution with two-pixel, 32-channel FMA tiles.
//!
//! Horizontal interiors are proved once per output pair. Each native tile
//! retains its sixteen accumulators through the ordered KY/KX reduction;
//! border pixels and short channel blocks use the checked shared readers.
use super::*;
use std::arch::aarch64::*;

/// A safe, optional fast path. The caller retains the general depthwise
/// implementation whenever geometry, storage, or packed weights do not fit.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_execute(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) -> bool {
    if !simd::neon_available()
        || p.cin != p.cout
        || p.cpg != 1
        || p.opg != 1
        || p.cout < 32
        || [
            p.batch, p.height, p.width, p.oh, p.ow, p.kh, p.kw, p.sh, p.sw, p.dh, p.dw,
        ]
        .contains(&0)
        || !output.len().is_multiple_of(p.cout)
        || (!bias.is_empty() && bias.len() < p.cout)
    {
        return false;
    }
    let Some(taps) = p.kh.checked_mul(p.kw) else {
        return false;
    };
    let Some(packed) = packed.filter(|panel| panel.matches(taps, p.cout)) else {
        return false;
    };
    let input_length = p
        .batch
        .checked_mul(p.height)
        .and_then(|n| n.checked_mul(p.width))
        .and_then(|n| n.checked_mul(p.cin));
    let output_pixels = p
        .batch
        .checked_mul(p.oh)
        .and_then(|n| n.checked_mul(p.ow))
        .filter(|&n| n.checked_mul(p.cout).is_some());
    if input_length.is_none_or(|n| n > input.len())
        || output_pixels.is_none_or(|n| {
            first
                .checked_add(output.len() / p.cout)
                .is_none_or(|end| end > n)
        })
        || (p.oh - 1)
            .checked_mul(p.sh)
            .and_then(|n| n.checked_add((p.kh - 1).checked_mul(p.dh)?))
            .is_none()
        || (p.ow - 1)
            .checked_mul(p.sw)
            .and_then(|n| n.checked_add((p.kw - 1).checked_mul(p.dw)?))
            .is_none()
    {
        return false;
    }
    let interior = horizontal_interior(p);
    let pixels = output.len() / p.cout;
    let full_channels = p.cout / 32 * 32;
    let vector_channels = p.cout / 4 * 4;
    let mut local = 0;
    while local < pixels {
        let pixel = first + local;
        let (batch, y, x) = (pixel / (p.oh * p.ow), pixel / p.ow % p.oh, pixel % p.ow);
        let count = (pixels - local).min(p.ow - x).min(2);
        let fast = x >= interior.start && x + count <= interior.end;
        let mut channel = 0;
        if fast {
            while channel < full_channels {
                let base = channel / 16 * taps * 16;
                let low = &packed.data()[base..base + taps * 16];
                let high = &packed.data()[base + taps * 16..base + taps * 32];
                let seed = if bias.is_empty() {
                    &[]
                } else {
                    &bias[channel..channel + 32]
                };
                // SAFETY: NEON and exact input/panel/bias/output spans were
                // checked on entry. The horizontal interior proves both
                // pixels' complete receptive fields; vertical padding skips
                // remain in the leaf. KY/KX coordinate products cannot overflow.
                let sums = unsafe {
                    if count == 2 {
                        interior_tile::<true>(p, input, low, high, seed, batch, y, x, channel)
                    } else {
                        interior_tile::<false>(p, input, low, high, seed, batch, y, x, channel)
                    }
                };
                for (row, values) in sums.iter().take(count).enumerate() {
                    for (vector, &sum) in values.iter().enumerate() {
                        let c = channel + vector * 4;
                        let at = (local + row) * p.cout + c;
                        store
                            .vector_channel(sum, first * p.cout + at, c)
                            .copy_to_slice(&mut output[at..at + 4]);
                    }
                }
                channel += 32;
            }
        }
        for c in (channel..vector_channels).step_by(4) {
            if count == 2 {
                depthwise_pixels::<4, true, 2>(
                    p,
                    input,
                    weights,
                    bias,
                    Some(packed),
                    output,
                    local,
                    batch,
                    y,
                    x,
                    c,
                    first,
                    store,
                );
            } else {
                depthwise_pixels::<4, true, 1>(
                    p,
                    input,
                    weights,
                    bias,
                    Some(packed),
                    output,
                    local,
                    batch,
                    y,
                    x,
                    c,
                    first,
                    store,
                );
            }
        }
        // Keep separate MUL/ADD for the final one to three channels, matching
        // the existing optimized CPU arithmetic contract exactly.
        for c in vector_channels..p.cout {
            if count == 2 {
                depthwise_pixels::<1, false, 2>(
                    p,
                    input,
                    weights,
                    bias,
                    Some(packed),
                    output,
                    local,
                    batch,
                    y,
                    x,
                    c,
                    first,
                    store,
                );
            } else {
                depthwise_pixels::<1, false, 1>(
                    p,
                    input,
                    weights,
                    bias,
                    Some(packed),
                    output,
                    local,
                    batch,
                    y,
                    x,
                    c,
                    first,
                    store,
                );
            }
        }
        local += count;
    }
    true
}

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

#[inline(always)]
fn portable(value: float32x4_t) -> Simd<f32, 4> {
    let mut result = [0.; 4];
    // SAFETY: the local result owns four floats; all callers entered NEON.
    unsafe { vst1q_f32(result.as_mut_ptr(), value) };
    Simd::from_array(result)
}

#[target_feature(enable = "neon")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn interior_tile<const TWO: bool>(
    p: &Conv,
    input: &[f32],
    low: &[f32],
    high: &[f32],
    bias: &[f32],
    batch: usize,
    y: usize,
    x: usize,
    channel: usize,
) -> [[Simd<f32, 4>; 8]; 2] {
    let mut a0 = if bias.is_empty() {
        vdupq_n_f32(0.)
    } else {
        // SAFETY: the caller supplies exactly 32 bias coefficients.
        unsafe { vld1q_f32(bias.as_ptr().add(0)) }
    };
    let mut b0 = a0;
    let mut a1 = if bias.is_empty() {
        vdupq_n_f32(0.)
    } else {
        // SAFETY: the caller supplies exactly 32 bias coefficients.
        unsafe { vld1q_f32(bias.as_ptr().add(4)) }
    };
    let mut b1 = a1;
    let mut a2 = if bias.is_empty() {
        vdupq_n_f32(0.)
    } else {
        // SAFETY: the caller supplies exactly 32 bias coefficients.
        unsafe { vld1q_f32(bias.as_ptr().add(8)) }
    };
    let mut b2 = a2;
    let mut a3 = if bias.is_empty() {
        vdupq_n_f32(0.)
    } else {
        // SAFETY: the caller supplies exactly 32 bias coefficients.
        unsafe { vld1q_f32(bias.as_ptr().add(12)) }
    };
    let mut b3 = a3;
    let mut a4 = if bias.is_empty() {
        vdupq_n_f32(0.)
    } else {
        // SAFETY: the caller supplies exactly 32 bias coefficients.
        unsafe { vld1q_f32(bias.as_ptr().add(16)) }
    };
    let mut b4 = a4;
    let mut a5 = if bias.is_empty() {
        vdupq_n_f32(0.)
    } else {
        // SAFETY: the caller supplies exactly 32 bias coefficients.
        unsafe { vld1q_f32(bias.as_ptr().add(20)) }
    };
    let mut b5 = a5;
    let mut a6 = if bias.is_empty() {
        vdupq_n_f32(0.)
    } else {
        // SAFETY: the caller supplies exactly 32 bias coefficients.
        unsafe { vld1q_f32(bias.as_ptr().add(24)) }
    };
    let mut b6 = a6;
    let mut a7 = if bias.is_empty() {
        vdupq_n_f32(0.)
    } else {
        // SAFETY: the caller supplies exactly 32 bias coefficients.
        unsafe { vld1q_f32(bias.as_ptr().add(28)) }
    };
    let mut b7 = a7;
    let pixel_stride = if TWO { p.sw * p.cin } else { 0 };
    let tap_stride = if p.kw > 1 { p.dw * p.cin } else { 0 };
    let span = pixel_stride + (p.kw - 1) * tap_stride + 32;
    let left = x * p.sw - p.pl;
    for ky in 0..p.kh {
        let sy = y * p.sh + ky * p.dh;
        if sy < p.pt || sy - p.pt >= p.height {
            continue;
        }
        let at = ((batch * p.height + sy - p.pt) * p.width + left) * p.cin + channel;
        let row = &input[at..at + span];
        for kx in 0..p.kw {
            let tap = (ky * p.kw + kx) * 16;
            let at0 = kx * tap_stride;
            let at1 = at0 + pixel_stride;
            {
                // SAFETY: row spans every tap and both complete 32-channel
                // pixels; packed OC16 panels span every validated KY/KX tap.
                let (w, v0, v1) = unsafe {
                    (
                        vld1q_f32(low.as_ptr().add(tap)),
                        vld1q_f32(row.as_ptr().add(at0)),
                        vld1q_f32(row.as_ptr().add(at1)),
                    )
                };
                a0 = vfmaq_f32(a0, v0, w);
                if TWO {
                    b0 = vfmaq_f32(b0, v1, w);
                }
            }
            {
                // SAFETY: row spans every tap and both complete 32-channel
                // pixels; packed OC16 panels span every validated KY/KX tap.
                let (w, v0, v1) = unsafe {
                    (
                        vld1q_f32(low.as_ptr().add(tap + 4)),
                        vld1q_f32(row.as_ptr().add(at0 + 4)),
                        vld1q_f32(row.as_ptr().add(at1 + 4)),
                    )
                };
                a1 = vfmaq_f32(a1, v0, w);
                if TWO {
                    b1 = vfmaq_f32(b1, v1, w);
                }
            }
            {
                // SAFETY: row spans every tap and both complete 32-channel
                // pixels; packed OC16 panels span every validated KY/KX tap.
                let (w, v0, v1) = unsafe {
                    (
                        vld1q_f32(low.as_ptr().add(tap + 8)),
                        vld1q_f32(row.as_ptr().add(at0 + 8)),
                        vld1q_f32(row.as_ptr().add(at1 + 8)),
                    )
                };
                a2 = vfmaq_f32(a2, v0, w);
                if TWO {
                    b2 = vfmaq_f32(b2, v1, w);
                }
            }
            {
                // SAFETY: row spans every tap and both complete 32-channel
                // pixels; packed OC16 panels span every validated KY/KX tap.
                let (w, v0, v1) = unsafe {
                    (
                        vld1q_f32(low.as_ptr().add(tap + 12)),
                        vld1q_f32(row.as_ptr().add(at0 + 12)),
                        vld1q_f32(row.as_ptr().add(at1 + 12)),
                    )
                };
                a3 = vfmaq_f32(a3, v0, w);
                if TWO {
                    b3 = vfmaq_f32(b3, v1, w);
                }
            }
            {
                // SAFETY: row spans every tap and both complete 32-channel
                // pixels; packed OC16 panels span every validated KY/KX tap.
                let (w, v0, v1) = unsafe {
                    (
                        vld1q_f32(high.as_ptr().add(tap)),
                        vld1q_f32(row.as_ptr().add(at0 + 16)),
                        vld1q_f32(row.as_ptr().add(at1 + 16)),
                    )
                };
                a4 = vfmaq_f32(a4, v0, w);
                if TWO {
                    b4 = vfmaq_f32(b4, v1, w);
                }
            }
            {
                // SAFETY: row spans every tap and both complete 32-channel
                // pixels; packed OC16 panels span every validated KY/KX tap.
                let (w, v0, v1) = unsafe {
                    (
                        vld1q_f32(high.as_ptr().add(tap + 4)),
                        vld1q_f32(row.as_ptr().add(at0 + 20)),
                        vld1q_f32(row.as_ptr().add(at1 + 20)),
                    )
                };
                a5 = vfmaq_f32(a5, v0, w);
                if TWO {
                    b5 = vfmaq_f32(b5, v1, w);
                }
            }
            {
                // SAFETY: row spans every tap and both complete 32-channel
                // pixels; packed OC16 panels span every validated KY/KX tap.
                let (w, v0, v1) = unsafe {
                    (
                        vld1q_f32(high.as_ptr().add(tap + 8)),
                        vld1q_f32(row.as_ptr().add(at0 + 24)),
                        vld1q_f32(row.as_ptr().add(at1 + 24)),
                    )
                };
                a6 = vfmaq_f32(a6, v0, w);
                if TWO {
                    b6 = vfmaq_f32(b6, v1, w);
                }
            }
            {
                // SAFETY: row spans every tap and both complete 32-channel
                // pixels; packed OC16 panels span every validated KY/KX tap.
                let (w, v0, v1) = unsafe {
                    (
                        vld1q_f32(high.as_ptr().add(tap + 12)),
                        vld1q_f32(row.as_ptr().add(at0 + 28)),
                        vld1q_f32(row.as_ptr().add(at1 + 28)),
                    )
                };
                a7 = vfmaq_f32(a7, v0, w);
                if TWO {
                    b7 = vfmaq_f32(b7, v1, w);
                }
            }
        }
    }
    [
        [
            portable(a0),
            portable(a1),
            portable(a2),
            portable(a3),
            portable(a4),
            portable(a5),
            portable(a6),
            portable(a7),
        ],
        [
            portable(b0),
            portable(b1),
            portable(b2),
            portable(b3),
            portable(b4),
            portable(b5),
            portable(b6),
            portable(b7),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::simd::num::SimdFloat;

    fn values(length: usize, seed: usize) -> Vec<f32> {
        (0..length)
            .map(|i| ((i * 37 + seed * 17) % 101) as f32 / 79. - 0.625)
            .collect()
    }

    fn plan(
        dimensions: [usize; 4],
        kernel: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        padding: [usize; 4],
    ) -> Conv {
        let [batch, channels, height, width] = dimensions;
        let [kh, kw] = kernel;
        let [sh, sw] = stride;
        let [dh, dw] = dilation;
        Conv {
            batch,
            cin: channels,
            cout: channels,
            height,
            width,
            oh: (height + padding[0] + padding[2] - (kh - 1) * dh - 1) / sh + 1,
            ow: (width + padding[1] + padding[3] - (kw - 1) * dw - 1) / sw + 1,
            cpg: 1,
            opg: 1,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            pt: padding[0],
            pl: padding[1],
        }
    }

    fn pack(p: &Conv, weights: &[f32]) -> PackedMatrix {
        let taps = p.kh * p.kw;
        let matrix = (0..taps * p.cout)
            .map(|i| weights[i % p.cout * taps + i / p.cout])
            .collect::<Vec<_>>();
        PackedMatrix::new(&matrix, taps, p.cout).unwrap()
    }

    // Independent logical oracle: bias first, ascending KY/KX, skipped padding
    // taps, FMA for complete vectors and separate MUL/ADD for scalar channels.
    fn oracle(p: &Conv, input: &[f32], weights: &[f32], bias: &[f32]) -> Vec<f32> {
        let mut output = vec![0.; p.batch * p.oh * p.ow * p.cout];
        for (pixel, row) in output.chunks_exact_mut(p.cout).enumerate() {
            let (n, y, x) = (pixel / (p.oh * p.ow), pixel / p.ow % p.oh, pixel % p.ow);
            for (c, result) in row.iter_mut().enumerate() {
                let mut sum = bias.get(c).copied().unwrap_or(0.);
                for ky in 0..p.kh {
                    let sy = y * p.sh + ky * p.dh;
                    if sy < p.pt || sy - p.pt >= p.height {
                        continue;
                    }
                    for kx in 0..p.kw {
                        let sx = x * p.sw + kx * p.dw;
                        if sx < p.pl || sx - p.pl >= p.width {
                            continue;
                        }
                        let value =
                            input[((n * p.height + sy - p.pt) * p.width + sx - p.pl) * p.cin + c];
                        let weight = weights[(c * p.kh + ky) * p.kw + kx];
                        sum = if c < p.cout / 4 * 4 {
                            value.mul_add(weight, sum)
                        } else {
                            value * weight + sum
                        };
                    }
                }
                *result = sum;
            }
        }
        output
    }

    fn assert_bits(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "index {i}: {a} != {e}");
        }
    }

    fn check(p: &Conv, bias_enabled: bool, store: &impl StoreTransform) {
        let input = values(p.batch * p.height * p.width * p.cin, 3);
        let weights = values(p.cout * p.kh * p.kw, 5);
        let bias = if bias_enabled {
            values(p.cout, 7)
        } else {
            vec![]
        };
        let panel = pack(p, &weights);
        let expected = oracle(p, &input, &weights, &bias)
            .into_iter()
            .enumerate()
            .map(|(i, value)| store.scalar(value, i))
            .collect::<Vec<_>>();
        let mut output = vec![f32::NAN; expected.len()];
        assert!(try_execute(
            p,
            &input,
            &weights,
            &bias,
            Some(&panel),
            &mut output,
            0,
            store
        ));
        assert_bits(&output, &expected);
        for band in [1, 3, 8, 19] {
            output.fill(f32::NAN);
            for (chunk, destination) in output.chunks_mut(band * p.cout).enumerate() {
                assert!(try_execute(
                    p,
                    &input,
                    &weights,
                    &bias,
                    Some(&panel),
                    destination,
                    chunk * band,
                    store
                ));
            }
            assert_bits(&output, &expected);
        }
    }

    #[test]
    fn neon_depthwise_preserves_order_padding_stride_dilation_and_band_channel_tails() {
        assert!(simd::neon_available());
        for p in [
            plan([2, 32, 5, 17], [3, 3], [1, 1], [1, 1], [1; 4]),
            plan([2, 35, 7, 19], [5, 5], [2, 2], [1, 1], [2; 4]),
            plan([1, 52, 5, 17], [3, 3], [1, 1], [2, 2], [2; 4]),
            plan([2, 67, 9, 31], [7, 7], [2, 2], [1, 2], [3, 7, 3, 5]),
            plan([1, 40, 1, 5], [9, 9], [1, 1], [1, 1], [4; 4]),
            plan([1, 36, 3, 1], [3, 3], [1, 1], [1, 1], [1; 4]),
            plan([1, 64, 3, 2], [1, 1], [1, 1], [1, 1], [0; 4]),
        ] {
            check(&p, true, &IdentityStore);
            check(&p, false, &IdentityStore);
        }
    }

    struct ResidualRelu<'a>(&'a [f32]);
    impl StoreTransform for ResidualRelu<'_> {
        fn scalar(&self, value: f32, index: usize) -> f32 {
            (value + self.0[index]).max(0.)
        }
        fn vector<const N: usize>(&self, value: Simd<f32, N>, index: usize) -> Simd<f32, N> {
            (value + Simd::from_slice(&self.0[index..index + N])).simd_max(Simd::splat(0.))
        }
    }

    #[test]
    fn neon_depthwise_preserves_global_residual_indices_and_unused_single_tap_dilation() {
        for mut p in [
            plan([2, 35, 5, 17], [3, 3], [1, 1], [1, 1], [1; 4]),
            plan([1, 67, 3, 17], [1, 1], [1, 1], [1, 1], [0; 4]),
        ] {
            if p.kh == 1 && p.kw == 1 {
                p.dh = usize::MAX;
                p.dw = usize::MAX;
            }
            let residual = values(p.batch * p.oh * p.ow * p.cout, 11);
            check(&p, true, &ResidualRelu(&residual));
        }
    }

    #[test]
    fn neon_depthwise_rejects_unsupported_or_unbounded_storage_without_writing() {
        let p = plan([1, 35, 3, 5], [3, 3], [1, 1], [1, 1], [1; 4]);
        let input = values(p.batch * p.height * p.width * p.cin, 13);
        let weights = values(p.cout * p.kh * p.kw, 17);
        let bias = values(p.cout, 19);
        let panel = pack(&p, &weights);
        let mut output = vec![123.; p.oh * p.ow * p.cout];
        assert!(!try_execute(
            &p,
            &input,
            &weights,
            &bias,
            None,
            &mut output,
            0,
            &IdentityStore
        ));
        assert!(!try_execute(
            &p,
            &input[..input.len() - 1],
            &weights,
            &bias,
            Some(&panel),
            &mut output,
            0,
            &IdentityStore
        ));
        assert!(!try_execute(
            &p,
            &input,
            &weights,
            &bias[..bias.len() - 1],
            Some(&panel),
            &mut output,
            0,
            &IdentityStore
        ));
        assert!(!try_execute(
            &p,
            &input,
            &weights,
            &bias,
            Some(&panel),
            &mut output,
            1,
            &IdentityStore
        ));
        let partial = output.len() - 1;
        assert!(!try_execute(
            &p,
            &input,
            &weights,
            &bias,
            Some(&panel),
            &mut output[..partial],
            0,
            &IdentityStore
        ));
        assert!(output.iter().all(|&value| value == 123.));
        let mut overflow = p.clone();
        overflow.oh = usize::MAX / p.cout + 1;
        overflow.ow = 1;
        assert!(!try_execute(
            &overflow,
            &input,
            &weights,
            &bias,
            Some(&panel),
            &mut [],
            0,
            &IdentityStore
        ));
        assert!(try_execute(
            &p,
            &input,
            &weights,
            &bias,
            Some(&panel),
            &mut [],
            p.batch * p.oh * p.ow,
            &IdentityStore
        ));
    }
}
