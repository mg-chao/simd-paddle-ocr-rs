//! Spatial ReduceMean. Two four-element packets are accumulated independently.
//! The channel alignment is that of the logical contiguous NCHW plane, including
//! when the executor keeps the activation in channels-last storage.
use std::simd::Simd;

#[inline(always)]
fn channel<const L: usize>(
    plane: usize,
    prefix: usize,
    sample: impl Fn(usize) -> Simd<f32, L>,
) -> Simd<f32, L> {
    let packets = plane.saturating_sub(prefix) / 4;
    let divisor = Simd::splat(plane as f32);
    if packets == 0 {
        let mut sum = sample(0);
        for i in 1..plane {
            sum += sample(i);
        }
        return sum / divisor;
    }
    let mut a: [_; 4] = std::array::from_fn(|j| sample(prefix + j));
    if packets >= 2 {
        let mut b: [_; 4] = std::array::from_fn(|j| sample(prefix + 4 + j));
        for k in 1..packets / 2 {
            let start = prefix + 8 * k;
            for j in 0..4 {
                a[j] += sample(start + j);
                b[j] += sample(start + j + 4);
            }
        }
        for j in 0..4 {
            a[j] += b[j];
            if packets % 2 == 1 {
                a[j] += sample(prefix + (packets - 1) * 4 + j);
            }
        }
    }
    let mut sum = (a[0] + a[2]) + (a[1] + a[3]);
    for i in 0..prefix {
        sum += sample(i);
    }
    for i in prefix + packets * 4..plane {
        sum += sample(i);
    }
    sum / divisor
}

/// Channels can have different packet boundaries when the logical plane is
/// not divisible by four. Accumulate common interior pixels by index modulo
/// eight; rotate the eight partial sums into each channel's packet order.
#[inline(always)]
fn unaligned_channels<const L: usize>(
    input: &[f32],
    dst: &mut [f32],
    first: usize,
    channels: usize,
    plane: usize,
) {
    use std::simd::{Mask, Select};
    let offsets: [usize; L] =
        std::array::from_fn(|lane| ((first + lane) * plane).wrapping_neg() % 4);
    let sample = |pixel: usize| Simd::<f32, L>::from_slice(&input[pixel * channels..]);
    let mut sums: [_; 8] = std::array::from_fn(|r| {
        let second = sample(8 + r);
        let aligned = Mask::<i32, L>::from_array(offsets.map(|offset| offset <= r));
        aligned.select(sample(r) + second, second)
    });
    // Every lane's paired-packet interval includes the complete interior.
    let interior_end = (plane - 7) / 8 * 8;
    for pixel in (16..interior_end).step_by(8) {
        for (r, sum) in sums.iter_mut().enumerate() {
            *sum += sample(pixel + r);
        }
    }
    for (lane, &prefix) in offsets.iter().enumerate() {
        let at = |pixel: usize| input[pixel * channels + lane];
        let mut by_pixel = sums.map(|v| v[lane]);
        let paired_end = prefix + (plane - prefix) / 8 * 8;
        for pixel in interior_end..paired_end {
            by_pixel[pixel % 8] += at(pixel);
        }
        let mut packet: [f32; 4] =
            std::array::from_fn(|j| by_pixel[(prefix + j) % 8] + by_pixel[(prefix + j + 4) % 8]);
        let end = prefix + (plane - prefix) / 4 * 4;
        if end != paired_end {
            for (j, sum) in packet.iter_mut().enumerate() {
                *sum += at(paired_end + j);
            }
        }
        let mut total = (packet[0] + packet[2]) + (packet[1] + packet[3]);
        for pixel in 0..prefix {
            total += at(pixel);
        }
        for pixel in end..plane {
            total += at(pixel);
        }
        dst[lane] = total / plane as f32;
    }
}

/// Keep a spatial tile hot while all channel groups revisit it. Each of the
/// eight sums follows one independent chain of the contiguous NCHW reduction.
#[inline(always)]
fn aligned_tiles<const L: usize>(
    values: &[f32],
    dst: &mut [f32],
    channels: usize,
    plane: usize,
) -> usize {
    const GROUPS: usize = 8;
    const TILE_PIXELS: usize = 128;
    let groups = channels / L;
    let paired_end = plane / 8 * 8;
    for first in (0..groups).step_by(GROUPS) {
        let count = GROUPS.min(groups - first);
        let mut saved = [[Simd::<f32, L>::splat(0.); 8]; GROUPS];
        for (g, sums) in saved[..count].iter_mut().enumerate() {
            let c = (first + g) * L;
            *sums = std::array::from_fn(|k| Simd::from_slice(&values[k * channels + c..]));
        }
        for start in (8..paired_end).step_by(TILE_PIXELS) {
            let end = (start + TILE_PIXELS).min(paired_end);
            for (g, state) in saved[..count].iter_mut().enumerate() {
                let c = (first + g) * L;
                let mut sums = *state;
                for pixel in (start..end).step_by(8) {
                    for (k, sum) in sums.iter_mut().enumerate() {
                        *sum += Simd::from_slice(&values[(pixel + k) * channels + c..]);
                    }
                }
                *state = sums;
            }
        }
        for (g, sums) in saved[..count].iter().enumerate() {
            let c = (first + g) * L;
            let mut merged = std::array::from_fn::<_, 4, _>(|j| sums[j] + sums[j + 4]);
            if paired_end != plane {
                for (j, value) in merged.iter_mut().enumerate() {
                    *value += Simd::from_slice(&values[(paired_end + j) * channels + c..]);
                }
            }
            let total = (merged[0] + merged[2]) + (merged[1] + merged[3]);
            (total / Simd::splat(plane as f32)).copy_to_slice(&mut dst[c..c + L]);
        }
    }
    groups * L
}

pub(super) fn channels_first(input: &[f32], output: &mut [f32], plane: usize) {
    for (i, (values, dst)) in input.chunks_exact(plane).zip(output).enumerate() {
        *dst = channel(plane, (i * plane).wrapping_neg() % 4, |k| {
            Simd::<_, 1>::splat(values[k])
        })[0];
    }
}

#[inline(always)]
pub(super) fn channels_last<const L: usize>(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    plane: usize,
) {
    for (n, (values, dst)) in input
        .chunks_exact(channels * plane)
        .zip(output.chunks_exact_mut(channels))
        .enumerate()
    {
        let mut c = 0;
        if L > 1 && plane >= 4096 && plane.is_multiple_of(4) && channels >= 2 * L {
            c = aligned_tiles::<L>(values, dst, channels, plane);
        } else if plane.is_multiple_of(4) {
            while c + L <= channels {
                channel::<L>(plane, 0, |i| Simd::from_slice(&values[i * channels + c..]))
                    .copy_to_slice(&mut dst[c..c + L]);
                c += L;
            }
        } else if plane >= 32 && L > 1 {
            while c + L <= channels {
                unaligned_channels::<L>(
                    &values[c..],
                    &mut dst[c..c + L],
                    n * channels + c,
                    channels,
                    plane,
                );
                c += L;
            }
        }
        for (c, dst) in dst.iter_mut().enumerate().skip(c) {
            let prefix = ((n * channels + c) * plane).wrapping_neg() % 4;
            *dst = channel(plane, prefix, |i| {
                Simd::<_, 1>::splat(values[i * channels + c])
            })[0];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_mean_preserves_packet_order_and_logical_channel_alignment() {
        let pattern = [
            16777216., 1., -16777216., 1., -1., 2., -1., -4., 5., 7., -5., 3., 1., 2., 4., -3.,
        ];
        // ORT CPU ReduceMean over axes [2,3] on [2,3,1,plane].
        for (plane, expected) in [
            (
                13,
                [
                    1062772106, 1065998494, 1047285445, 1065353216, 1235061215, 3382544850,
                ],
            ),
            (16, [1062207488; 6]),
        ] {
            let input: Vec<_> = pattern.iter().copied().cycle().take(6 * plane).collect();
            let mut result = [0.; 6];
            channels_first(&input, &mut result, plane);
            assert_eq!(result.map(f32::to_bits), expected);
            let mut packed = vec![0.; input.len()];
            for n in 0..2 {
                for c in 0..3 {
                    for i in 0..plane {
                        packed[(n * plane + i) * 3 + c] = input[(n * 3 + c) * plane + i];
                    }
                }
            }
            channels_last::<2>(&packed, &mut result, 3, plane);
            assert_eq!(result.map(f32::to_bits), expected);
        }
    }
    #[test]
    fn misaligned_channel_vectors_match_the_contiguous_packet_order() {
        for plane in (30..56).chain([173, 525, 1050, 4096, 4100]) {
            for channels in [5, 8, 19, 75] {
                let input: Vec<_> = (0..2 * channels * plane)
                    .map(|i| {
                        let bits = (i as u32).wrapping_mul(2654435761);
                        let power = (bits >> 24) % 31 + 100;
                        f32::from_bits((bits & 0x807fffff) | (power << 23))
                    })
                    .collect();
                let mut packed = vec![0.; input.len()];
                for n in 0..2 {
                    for c in 0..channels {
                        for i in 0..plane {
                            packed[(n * plane + i) * channels + c] =
                                input[(n * channels + c) * plane + i];
                        }
                    }
                }
                let mut expected = vec![0.; channels * 2];
                channels_first(&input, &mut expected, plane);
                let mut actual = vec![0.; channels * 2];
                channels_last::<8>(&packed, &mut actual, channels, plane);
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "plane {plane} channels {channels}"
                );
            }
        }
    }
}
