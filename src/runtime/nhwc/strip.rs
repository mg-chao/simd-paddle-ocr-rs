//! Large depthwise kernels accumulate one kernel row per strip pass. Each
//! fixed-size strip holds up to sixteen pixels and four channel vectors;
//! AVX uses the reference's 32-channel block without per-request allocation.
use super::*;

const PIXELS: usize = 16;

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
    let pixels = output.len() / p.cout;
    let vector_channels = p.cout / LANES * LANES;
    let mut local = 0;
    while local < pixels {
        let pixel = first + local;
        let (n, y, x) = (pixel / (p.oh * p.ow), pixel / p.ow % p.oh, pixel % p.ow);
        let count = (pixels - local).min(p.ow - x).min(PIXELS);
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
            for ky in 0..p.kh {
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
