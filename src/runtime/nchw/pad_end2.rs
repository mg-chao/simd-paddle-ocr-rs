//! Bounded eight-output PadEnd2 reduction. Border/tail arithmetic is unchanged.
use super::{Conv2d, edge, interior, madd, simd};
use std::simd::{Select, Simd};

/// Constructed from valid full-width spatial vectors only. Every input view
/// includes all channel strides and the ninth value needed by the second tap.
struct Inputs<'a> {
    top: &'a [f32],
    bottom: &'a [f32],
    weights: [&'a [[f32; 4]]; 8],
    plane: usize,
    channels: usize,
    seeds: [f32; 8],
}

impl<'a> Inputs<'a> {
    fn new(
        top: &'a [f32],
        bottom: &'a [f32],
        weights: [&'a [[f32; 4]]; 8],
        plane: usize,
        channels: usize,
        seeds: [f32; 8],
    ) -> Self {
        assert!(channels > 0);
        let span = (channels - 1)
            .checked_mul(plane)
            .and_then(|value| value.checked_add(9))
            .expect("PadEnd2 input span overflow");
        assert!(top.len() >= span);
        assert!(bottom.is_empty() || bottom.len() >= span);
        assert!(weights.iter().all(|values| values.len() == channels));
        Self {
            top,
            bottom,
            weights,
            plane,
            channels,
            seeds,
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute<const FMA: bool>(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    out: &mut [f32],
    batch: usize,
    co: usize,
    first: usize,
    store: &impl simd::StoreTransform,
) {
    // This additional safe-entry proof is outside every pixel/reduction loop.
    assert!(std::is_x86_feature_detected!("avx"));
    assert!(!FMA || std::is_x86_feature_detected!("fma"));
    debug_assert!(p.groups == 1 && p.kernel_height == 2 && p.kernel_width == 2);
    debug_assert!(p.stride_height == 1 && p.stride_width == 1);
    debug_assert!(p.pad_top == 0 && p.pad_left == 0);
    let input_plane = p.height * p.width;
    let output_plane = p.output_height * p.output_width;
    let weight_plane = p.input_channels * 4;
    let panels = std::array::from_fn(|lane| {
        weights[(co + lane) * weight_plane..][..weight_plane]
            .as_chunks::<4>()
            .0
    });
    let seeds = std::array::from_fn(|lane| if bias.is_empty() { 0. } else { bias[co + lane] });
    let span = (p.input_channels - 1)
        .checked_mul(input_plane)
        .and_then(|value| value.checked_add(9))
        .expect("PadEnd2 input span overflow");
    let valid = interior(p);
    for y in 0..p.output_height {
        let mut x = 0;
        while x < valid.start {
            for lane in 0..8 {
                let index = y * p.output_width + x;
                out[(co + lane - first) * output_plane + index] = store.scalar(
                    edge(p, input, weights, bias, batch, co + lane, y, x),
                    (batch * p.output_channels + co + lane) * output_plane + index,
                );
            }
            x += 1;
        }
        while x + 8 <= valid.end {
            let origin = (batch * p.input_channels * p.height + y) * p.width + x;
            let top = &input[origin..][..span];
            let bottom = if y + 1 < p.height {
                &input[origin + p.width..][..span]
            } else {
                &[]
            };
            let descriptor = Inputs::new(top, bottom, panels, input_plane, p.input_channels, seeds);
            // SAFETY: CPU features were checked before the loops; the private
            // descriptor proves each selected row's full load/weight bounds.
            let sums = unsafe {
                if FMA {
                    if bottom.is_empty() {
                        tile_fma::<false>(&descriptor)
                    } else {
                        tile_fma::<true>(&descriptor)
                    }
                } else if bottom.is_empty() {
                    tile_avx::<false>(&descriptor)
                } else {
                    tile_avx::<true>(&descriptor)
                }
            };
            for (lane, &sum) in sums.iter().enumerate() {
                let index = y * p.output_width + x;
                store
                    .vector(
                        sum,
                        (batch * p.output_channels + co + lane) * output_plane + index,
                    )
                    .copy_to_slice(&mut out[(co + lane - first) * output_plane + index..][..8]);
            }
            x += 8;
        }
        for x in (x..p.output_width).step_by(4) {
            // The old edge loop used separate multiply/add, even under FMA.
            // Vectorize independent spatial outputs without changing that
            // arithmetic or the scalar output transformation on these tails.
            let count = (p.output_width - x).min(4);
            // SAFETY: execute checked AVX; the helper uses checked slices.
            let sums = unsafe { tail_avx(p, input, panels, seeds, batch, y, x) };
            for (lane, sum) in sums.iter().enumerate() {
                for column in 0..count {
                    let index = y * p.output_width + x + column;
                    out[(co + lane - first) * output_plane + index] = store.scalar(
                        sum[column],
                        (batch * p.output_channels + co + lane) * output_plane + index,
                    );
                }
            }
        }
    }
}

macro_rules! reduction {
    ($name:ident, $features:literal, $fused:literal) => {
        #[inline(never)]
        #[target_feature(enable = $features)]
        unsafe fn $name<const BOTTOM: bool>(d: &Inputs<'_>) -> [Simd<f32, 8>; 8] {
            assert!(!BOTTOM || !d.bottom.is_empty());
            let mut a0 = Simd::splat(d.seeds[0]);
            let mut a1 = Simd::splat(d.seeds[1]);
            let mut a2 = Simd::splat(d.seeds[2]);
            let mut a3 = Simd::splat(d.seeds[3]);
            let mut a4 = Simd::splat(d.seeds[4]);
            let mut a5 = Simd::splat(d.seeds[5]);
            let mut a6 = Simd::splat(d.seeds[6]);
            let mut a7 = Simd::splat(d.seeds[7]);
            let [w0, w1, w2, w3, w4, w5, w6, w7] = d.weights.map(|values| values.as_ptr());
            let top = d.top.as_ptr();
            let bottom = d.bottom.as_ptr();
            for ci in 0..d.channels {
                macro_rules! tap {
                    ($row:ident, $offset:literal, $weight:literal) => {{
                        // SAFETY: ci<channels, offset<=1. Inputs::new proves
                        // (channels-1)*plane+9 available floats for each selected
                        // row, so this unaligned eight-float load ends in bounds.
                        // All eight coefficient slices contain channels entries;
                        // each fixed tap index is in the typed [f32;4] record.
                        let value = Simd::from_array(unsafe {
                            $row.add(ci * d.plane + $offset)
                                .cast::<[f32; 8]>()
                                .read_unaligned()
                        });
                        a0 = madd::<8, $fused>(
                            value,
                            Simd::splat(unsafe { (*w0.add(ci))[$weight] }),
                            a0,
                        );
                        a1 = madd::<8, $fused>(
                            value,
                            Simd::splat(unsafe { (*w1.add(ci))[$weight] }),
                            a1,
                        );
                        a2 = madd::<8, $fused>(
                            value,
                            Simd::splat(unsafe { (*w2.add(ci))[$weight] }),
                            a2,
                        );
                        a3 = madd::<8, $fused>(
                            value,
                            Simd::splat(unsafe { (*w3.add(ci))[$weight] }),
                            a3,
                        );
                        a4 = madd::<8, $fused>(
                            value,
                            Simd::splat(unsafe { (*w4.add(ci))[$weight] }),
                            a4,
                        );
                        a5 = madd::<8, $fused>(
                            value,
                            Simd::splat(unsafe { (*w5.add(ci))[$weight] }),
                            a5,
                        );
                        a6 = madd::<8, $fused>(
                            value,
                            Simd::splat(unsafe { (*w6.add(ci))[$weight] }),
                            a6,
                        );
                        a7 = madd::<8, $fused>(
                            value,
                            Simd::splat(unsafe { (*w7.add(ci))[$weight] }),
                            a7,
                        );
                    }};
                }
                tap!(top, 0, 0);
                tap!(top, 1, 1);
                if BOTTOM {
                    tap!(bottom, 0, 2);
                    tap!(bottom, 1, 3);
                }
            }
            [a0, a1, a2, a3, a4, a5, a6, a7]
        }
    };
}
reduction!(tile_avx, "avx", false);
reduction!(tile_fma, "avx,fma", true);

#[target_feature(enable = "avx")]
#[inline(never)]
// The same channel index addresses eight coefficient rows and an input plane.
#[allow(clippy::needless_range_loop)]
unsafe fn tail_avx(
    p: &Conv2d,
    input: &[f32],
    weights: [&[[f32; 4]]; 8],
    seeds: [f32; 8],
    batch: usize,
    y: usize,
    x: usize,
) -> [Simd<f32, 4>; 8] {
    let mut sums = seeds.map(Simd::splat);
    for ci in 0..p.input_channels {
        for ky in 0..2 {
            if y + ky >= p.height {
                continue;
            }
            let origin = ((batch * p.input_channels + ci) * p.height + y + ky) * p.width;
            for kx in 0..2 {
                let available = p.width.saturating_sub(x + kx).min(4);
                if available == 0 {
                    continue;
                }
                let value = Simd::<f32, 4>::load_or_default(&input[origin + x + kx..][..available]);
                let valid = std::simd::Mask::<i32, 4>::from_array(std::array::from_fn(|lane| {
                    lane < available
                }));
                for lane in 0..8 {
                    let next = sums[lane] + value * Simd::splat(weights[lane][ci][ky * 2 + kx]);
                    sums[lane] = valid.select(next, sums[lane]);
                }
            }
        }
    }
    sums
}
