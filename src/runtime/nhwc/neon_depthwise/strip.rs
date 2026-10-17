//! Tall depthwise kernels traverse one kernel row across a contiguous output
//! strip before advancing KY. Scratch carries only raw sums; the fused store
//! transform is applied once after the complete, ordered KY/KX reduction.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn execute<const PIXELS: usize>(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    assert!(PIXELS == 0 || (PIXELS >= 2 && PIXELS.is_multiple_of(2)));
    let pixels = output.len() / p.cout;
    if pixels == 0 {
        return;
    }
    let interior = horizontal_interior(p);
    #[cfg(test)]
    let capacity = (interior.end - interior.start).min(pixels);
    // At most one output row and one 32-channel block are live. Initialization
    // is explicit, and every valid strip cell is seeded before its first read.
    let mut bounded = [[0.; 32]; PIXELS];
    #[cfg(test)]
    let mut full_row = if PIXELS == 0 {
        vec![[0.; 32]; capacity]
    } else {
        vec![]
    };
    #[cfg(test)]
    let partial: &mut [[f32; 32]] = if PIXELS == 0 {
        &mut full_row
    } else {
        &mut bounded
    };
    #[cfg(not(test))]
    let partial: &mut [[f32; 32]] = &mut bounded;
    let full_channels = p.cout / 32 * 32;
    let vector_channels = p.cout / 4 * 4;
    let taps = p.kh * p.kw;
    let mut local = 0;
    while local < pixels {
        let pixel = first + local;
        let (batch, y, x) = (pixel / (p.oh * p.ow), pixel / p.ow % p.oh, pixel % p.ow);
        let row_end = x + (pixels - local).min(p.ow - x);
        let start = x.max(interior.start).min(row_end);
        let end = row_end.min(interior.end).max(start);
        let count = (end - start).min(partial.len()) / 2 * 2;
        let prefix = start - x;
        execute_current(
            p,
            input,
            weights,
            bias,
            packed,
            &mut output[local * p.cout..(local + prefix) * p.cout],
            pixel,
            store,
        );
        let strip = &mut partial[..count];
        for channel in (0..full_channels).step_by(32) {
            let mut seed = [0.; 32];
            if !bias.is_empty() {
                seed.copy_from_slice(&bias[channel..channel + 32]);
            }
            strip.fill(seed);
            let base = channel / 16 * taps * 16;
            let low = &packed.data()[base..base + taps * 16];
            let high = &packed.data()[base + taps * 16..base + taps * 32];
            for ky in 0..p.kh {
                let sy = y * p.sh + ky * p.dh;
                if sy < p.pt || sy - p.pt >= p.height || count == 0 {
                    continue;
                }
                let pixel_stride = p.sw * p.cin;
                let tap_stride = if p.kw > 1 { p.dw * p.cin } else { 0 };
                let span = (count - 1) * pixel_stride + (p.kw - 1) * tap_stride + 32;
                let at = ((batch * p.height + sy - p.pt) * p.width + start * p.sw - p.pl) * p.cin
                    + channel;
                let row = &input[at..at + span];
                let tap = ky * p.kw * 16;
                // SAFETY: the entry point established NEON and all geometry.
                // The proved horizontal interior bounds this complete row
                // span, each OC16 panel has all KW taps, and count is even.
                unsafe {
                    accumulate_row(
                        row,
                        &low[tap..tap + p.kw * 16],
                        &high[tap..tap + p.kw * 16],
                        pixel_stride,
                        tap_stride,
                        p.kw,
                        strip,
                    );
                }
            }
            for (row, values) in strip.iter().enumerate() {
                for (vector, lanes) in values.chunks_exact(4).enumerate() {
                    let c = channel + vector * 4;
                    let at = (local + prefix + row) * p.cout + c;
                    store
                        .vector_channel(Simd::<f32, 4>::from_slice(lanes), first * p.cout + at, c)
                        .copy_to_slice(&mut output[at..at + 4]);
                }
            }
        }
        // The existing checked readers retain packed channel tails and the
        // scalar MUL/ADD contract, including global fused residual indices.
        for row in (0..count).step_by(2) {
            for channel in (full_channels..vector_channels).step_by(4) {
                depthwise_pixels::<4, true, 2>(
                    p,
                    input,
                    weights,
                    bias,
                    Some(packed),
                    output,
                    local + prefix + row,
                    batch,
                    y,
                    start + row,
                    channel,
                    first,
                    store,
                );
            }
            for channel in vector_channels..p.cout {
                depthwise_pixels::<1, false, 2>(
                    p,
                    input,
                    weights,
                    bias,
                    Some(packed),
                    output,
                    local + prefix + row,
                    batch,
                    y,
                    start + row,
                    channel,
                    first,
                    store,
                );
            }
        }
        let suffix = local + prefix + count;
        let row_count = if count > 0 && start + count < end {
            prefix + count
        } else {
            row_end - x
        };
        execute_current(
            p,
            input,
            weights,
            bias,
            packed,
            &mut output[suffix * p.cout..(local + row_count) * p.cout],
            first + suffix,
            store,
        );
        local += row_count;
    }
}

#[target_feature(enable = "neon")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn accumulate_row(
    input: &[f32],
    low: &[f32],
    high: &[f32],
    pixel_stride: usize,
    tap_stride: usize,
    width: usize,
    partial: &mut [[f32; 32]],
) {
    for pixel in (0..partial.len()).step_by(2) {
        let pair = &mut partial[pixel..pixel + 2];
        // SAFETY: pair contains two complete, initialized 32-channel cells.
        let mut a0 = unsafe { vld1q_f32(pair[0].as_ptr()) };
        let mut b0 = unsafe { vld1q_f32(pair[1].as_ptr()) };
        // SAFETY: pair contains two complete, initialized 32-channel cells.
        let mut a1 = unsafe { vld1q_f32(pair[0].as_ptr().add(4)) };
        let mut b1 = unsafe { vld1q_f32(pair[1].as_ptr().add(4)) };
        // SAFETY: pair contains two complete, initialized 32-channel cells.
        let mut a2 = unsafe { vld1q_f32(pair[0].as_ptr().add(8)) };
        let mut b2 = unsafe { vld1q_f32(pair[1].as_ptr().add(8)) };
        // SAFETY: pair contains two complete, initialized 32-channel cells.
        let mut a3 = unsafe { vld1q_f32(pair[0].as_ptr().add(12)) };
        let mut b3 = unsafe { vld1q_f32(pair[1].as_ptr().add(12)) };
        // SAFETY: pair contains two complete, initialized 32-channel cells.
        let mut a4 = unsafe { vld1q_f32(pair[0].as_ptr().add(16)) };
        let mut b4 = unsafe { vld1q_f32(pair[1].as_ptr().add(16)) };
        // SAFETY: pair contains two complete, initialized 32-channel cells.
        let mut a5 = unsafe { vld1q_f32(pair[0].as_ptr().add(20)) };
        let mut b5 = unsafe { vld1q_f32(pair[1].as_ptr().add(20)) };
        // SAFETY: pair contains two complete, initialized 32-channel cells.
        let mut a6 = unsafe { vld1q_f32(pair[0].as_ptr().add(24)) };
        let mut b6 = unsafe { vld1q_f32(pair[1].as_ptr().add(24)) };
        // SAFETY: pair contains two complete, initialized 32-channel cells.
        let mut a7 = unsafe { vld1q_f32(pair[0].as_ptr().add(28)) };
        let mut b7 = unsafe { vld1q_f32(pair[1].as_ptr().add(28)) };
        for kx in 0..width {
            let tap = kx * 16;
            let at0 = pixel * pixel_stride + kx * tap_stride;
            let at1 = at0 + pixel_stride;
            // SAFETY: the caller proved both pixel spans and all packed taps.
            let (weight, v0, v1) = unsafe {
                (
                    vld1q_f32(low.as_ptr().add(tap)),
                    vld1q_f32(input.as_ptr().add(at0)),
                    vld1q_f32(input.as_ptr().add(at1)),
                )
            };
            a0 = vfmaq_f32(a0, v0, weight);
            b0 = vfmaq_f32(b0, v1, weight);
            // SAFETY: the caller proved both pixel spans and all packed taps.
            let (weight, v0, v1) = unsafe {
                (
                    vld1q_f32(low.as_ptr().add(tap + 4)),
                    vld1q_f32(input.as_ptr().add(at0 + 4)),
                    vld1q_f32(input.as_ptr().add(at1 + 4)),
                )
            };
            a1 = vfmaq_f32(a1, v0, weight);
            b1 = vfmaq_f32(b1, v1, weight);
            // SAFETY: the caller proved both pixel spans and all packed taps.
            let (weight, v0, v1) = unsafe {
                (
                    vld1q_f32(low.as_ptr().add(tap + 8)),
                    vld1q_f32(input.as_ptr().add(at0 + 8)),
                    vld1q_f32(input.as_ptr().add(at1 + 8)),
                )
            };
            a2 = vfmaq_f32(a2, v0, weight);
            b2 = vfmaq_f32(b2, v1, weight);
            // SAFETY: the caller proved both pixel spans and all packed taps.
            let (weight, v0, v1) = unsafe {
                (
                    vld1q_f32(low.as_ptr().add(tap + 12)),
                    vld1q_f32(input.as_ptr().add(at0 + 12)),
                    vld1q_f32(input.as_ptr().add(at1 + 12)),
                )
            };
            a3 = vfmaq_f32(a3, v0, weight);
            b3 = vfmaq_f32(b3, v1, weight);
            // SAFETY: the caller proved both pixel spans and all packed taps.
            let (weight, v0, v1) = unsafe {
                (
                    vld1q_f32(high.as_ptr().add(tap)),
                    vld1q_f32(input.as_ptr().add(at0 + 16)),
                    vld1q_f32(input.as_ptr().add(at1 + 16)),
                )
            };
            a4 = vfmaq_f32(a4, v0, weight);
            b4 = vfmaq_f32(b4, v1, weight);
            // SAFETY: the caller proved both pixel spans and all packed taps.
            let (weight, v0, v1) = unsafe {
                (
                    vld1q_f32(high.as_ptr().add(tap + 4)),
                    vld1q_f32(input.as_ptr().add(at0 + 20)),
                    vld1q_f32(input.as_ptr().add(at1 + 20)),
                )
            };
            a5 = vfmaq_f32(a5, v0, weight);
            b5 = vfmaq_f32(b5, v1, weight);
            // SAFETY: the caller proved both pixel spans and all packed taps.
            let (weight, v0, v1) = unsafe {
                (
                    vld1q_f32(high.as_ptr().add(tap + 8)),
                    vld1q_f32(input.as_ptr().add(at0 + 24)),
                    vld1q_f32(input.as_ptr().add(at1 + 24)),
                )
            };
            a6 = vfmaq_f32(a6, v0, weight);
            b6 = vfmaq_f32(b6, v1, weight);
            // SAFETY: the caller proved both pixel spans and all packed taps.
            let (weight, v0, v1) = unsafe {
                (
                    vld1q_f32(high.as_ptr().add(tap + 12)),
                    vld1q_f32(input.as_ptr().add(at0 + 28)),
                    vld1q_f32(input.as_ptr().add(at1 + 28)),
                )
            };
            a7 = vfmaq_f32(a7, v0, weight);
            b7 = vfmaq_f32(b7, v1, weight);
        }
        // SAFETY: pair owns two complete 32-channel accumulator cells.
        unsafe {
            vst1q_f32(pair[0].as_mut_ptr(), a0);
            vst1q_f32(pair[1].as_mut_ptr(), b0);
        }
        // SAFETY: pair owns two complete 32-channel accumulator cells.
        unsafe {
            vst1q_f32(pair[0].as_mut_ptr().add(4), a1);
            vst1q_f32(pair[1].as_mut_ptr().add(4), b1);
        }
        // SAFETY: pair owns two complete 32-channel accumulator cells.
        unsafe {
            vst1q_f32(pair[0].as_mut_ptr().add(8), a2);
            vst1q_f32(pair[1].as_mut_ptr().add(8), b2);
        }
        // SAFETY: pair owns two complete 32-channel accumulator cells.
        unsafe {
            vst1q_f32(pair[0].as_mut_ptr().add(12), a3);
            vst1q_f32(pair[1].as_mut_ptr().add(12), b3);
        }
        // SAFETY: pair owns two complete 32-channel accumulator cells.
        unsafe {
            vst1q_f32(pair[0].as_mut_ptr().add(16), a4);
            vst1q_f32(pair[1].as_mut_ptr().add(16), b4);
        }
        // SAFETY: pair owns two complete 32-channel accumulator cells.
        unsafe {
            vst1q_f32(pair[0].as_mut_ptr().add(20), a5);
            vst1q_f32(pair[1].as_mut_ptr().add(20), b5);
        }
        // SAFETY: pair owns two complete 32-channel accumulator cells.
        unsafe {
            vst1q_f32(pair[0].as_mut_ptr().add(24), a6);
            vst1q_f32(pair[1].as_mut_ptr().add(24), b6);
        }
        // SAFETY: pair owns two complete 32-channel accumulator cells.
        unsafe {
            vst1q_f32(pair[0].as_mut_ptr().add(28), a7);
            vst1q_f32(pair[1].as_mut_ptr().add(28), b7);
        }
    }
}
