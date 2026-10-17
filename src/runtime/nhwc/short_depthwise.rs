//! Packed short-depthwise interiors retain eight independent pixel sums.
//! Complete receptive fields are proved once per tile; borders and scalar
//! channel tails keep the original checked arithmetic and output indexing.
use super::*;

const PIXELS: usize = 8;
const LANES: usize = 8;

/// Prove native storage bounds and that a complete interior tile can exist.
/// Thin/padded maps with no such receptive field retain the checked traversal.
pub(super) fn eligible(
    p: &Conv,
    input: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    output_len: usize,
    first: usize,
) -> bool {
    if p.cin != p.cout
        || p.cpg != 1
        || p.opg != 1
        || p.cout < LANES
        || p.kh > 5
        || p.kw > 5
        || [
            p.batch, p.height, p.width, p.oh, p.ow, p.kh, p.kw, p.sh, p.sw, p.dh, p.dw,
        ]
        .contains(&0)
        || !output_len.is_multiple_of(p.cout)
        || output_len / p.cout < PIXELS
        || (!bias.is_empty() && bias.len() < p.cout)
    {
        return false;
    }
    let taps = p.kh * p.kw;
    if !packed.is_some_and(|panel| panel.matches(taps, p.cout)) {
        return false;
    }
    let input_len = p
        .batch
        .checked_mul(p.height)
        .and_then(|n| n.checked_mul(p.width))
        .and_then(|n| n.checked_mul(p.cin));
    let output_pixels = p
        .batch
        .checked_mul(p.oh)
        .and_then(|n| n.checked_mul(p.ow))
        .filter(|&n| n.checked_mul(p.cout).is_some());
    if input_len.is_none_or(|n| n > input.len())
        || output_pixels.is_none_or(|n| {
            first
                .checked_add(output_len / p.cout)
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
    interior_axis(p.height, p.oh, p.kh, p.sh, p.dh, p.pt).is_some()
        && interior_axis(p.width, p.ow, p.kw, p.sw, p.dw, p.pl)
            .is_some_and(|range| range.len() >= PIXELS)
}

/// Output positions whose entire receptive field lies in the input. Geometry
/// is positive and its maximum source arithmetic was checked by eligible().
fn interior_axis(
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    pad: usize,
) -> Option<std::ops::Range<usize>> {
    let last_source_origin = input.checked_sub(1)?.checked_sub((kernel - 1) * dilation)?;
    let first = pad.div_ceil(stride).min(output);
    let end = last_source_origin
        .checked_add(pad)?
        .checked_div(stride)?
        .checked_add(1)?
        .min(output);
    (first < end).then_some(first..end)
}

/// The caller has validated all geometry arithmetic and the full input extent.
/// Returning a channel-zero origin proves every tap for eight adjacent outputs.
pub(super) fn interior_origin(p: &Conv, batch: usize, y: usize, x: usize) -> Option<usize> {
    let top = (y * p.sh).checked_sub(p.pt)?;
    let left = (x * p.sw).checked_sub(p.pl)?;
    let bottom = top + (p.kh - 1) * p.dh;
    let right = left + (PIXELS - 1) * p.sw + (p.kw - 1) * p.dw;
    if bottom >= p.height || right >= p.width {
        return None;
    }
    Some(((batch * p.height + top) * p.width + left) * p.cin)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn interior_tile(
    p: &Conv,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    origin: usize,
    local: usize,
    channel: usize,
    first: usize,
    store: &impl StoreTransform,
) {
    let taps = p.kh * p.kw;
    let panel_start = channel / 16 * taps * 16;
    let panel = &packed.data()[panel_start..panel_start + taps * 16];
    let initial = if bias.is_empty() {
        Simd::<f32, LANES>::splat(0.)
    } else {
        Simd::from_slice(&bias[channel..channel + LANES])
    };
    let mut sums = [initial; PIXELS];
    let input = input.as_ptr();
    let weights = panel.as_ptr();
    for ky in 0..p.kh {
        for kx in 0..p.kw {
            let tap = ky * p.kw + kx;
            // SAFETY: channel%16 is zero or eight, and panel spans taps*16.
            let weight = Simd::from_array(unsafe {
                weights
                    .add(tap * 16 + channel % 16)
                    .cast::<[f32; LANES]>()
                    .read_unaligned()
            });
            let tap_origin = origin + (ky * p.dh * p.width + kx * p.dw) * p.cin + channel;
            for (pixel, sum) in sums.iter_mut().enumerate() {
                // SAFETY: the tile's complete receptive-field proof includes
                // this pixel/tap and channel+LANES <= cin. Unaligned reads
                // support arbitrary input starts and non-vector channel strides.
                let value = Simd::from_array(unsafe {
                    input
                        .add(tap_origin + pixel * p.sw * p.cin)
                        .cast::<[f32; LANES]>()
                        .read_unaligned()
                });
                *sum = value.mul_add(weight, *sum);
            }
        }
    }
    for (pixel, sum) in sums.into_iter().enumerate() {
        let at = (local + pixel) * p.cout + channel;
        store
            .vector_channel(sum, first * p.cout + at, channel)
            .copy_to_slice(&mut output[at..at + LANES]);
    }
}

/// Malformed-call contract for focused tests; production shares the existing
/// depthwise loop instead of adding a second complete-layer call boundary.
#[cfg(test)]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn try_execute(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) -> bool {
    if !eligible(p, input, bias, packed, output.len(), first) {
        return false;
    }
    depthwise_core::<LANES, true, true>(p, input, weights, bias, packed, output, first, store);
    true
}

/// Exact paired control keeps production's checked eight-pixel traversal.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_control<const FAST: bool>(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    assert!(simd::avx2_fma_available());
    // SAFETY: the feature check covers both paired policies and their stores.
    unsafe {
        control_avx2::<FAST>(p, input, weights, bias, packed, output, first, store);
    }
}

#[cfg(test)]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn control_avx2<const FAST: bool>(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    depthwise_core::<LANES, true, FAST>(
        p,
        input,
        weights,
        bias,
        Some(packed),
        output,
        first,
        store,
    );
}

#[cfg(test)]
mod tests;
