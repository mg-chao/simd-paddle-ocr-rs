//! Isolate the OC4 spatial reduction from fused stores and channel geometry.
use super::*;

struct Inputs<'a> {
    input: &'a [f32],
    panel: &'a [[f32; 16]],
    plane: usize,
    lane: usize,
    seeds: [f32; 4],
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn packet<const W: usize, const FMA: bool, const PAIR: bool>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    batch: usize,
    co: usize,
    x: usize,
) -> [[Simd<f32, W>; 2]; 4] {
    let plane = p.height * p.width;
    assert!(p.input_channels > 0);
    let width = W * if PAIR { 2 } else { 1 };
    assert!(co.is_multiple_of(4) && co + 4 <= p.output_channels && x + width <= plane);
    let start = batch * p.input_channels * plane + x;
    let span = (p.input_channels - 1) * plane + width;
    let panel_start = (co / 16) * p.input_channels * 16;
    let d = Inputs {
        input: &input[start..start + span],
        panel: packed.data()[panel_start..panel_start + p.input_channels * 16]
            .as_chunks::<16>()
            .0,
        plane,
        lane: co % 16,
        seeds: initial::<4>(bias, co, p.output_channels).to_array(),
    };
    #[cfg(target_arch = "x86_64")]
    // SAFETY: only the guarded native N8 pointwise path calls this helper.
    unsafe {
        if FMA {
            fused::<W, PAIR>(&d)
        } else {
            avx::<W, PAIR>(&d)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    reduce::<W, FMA, PAIR>(&d)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline(never)]
unsafe fn fused<const W: usize, const PAIR: bool>(d: &Inputs<'_>) -> [[Simd<f32, W>; 2]; 4] {
    reduce::<W, true, PAIR>(d)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline(never)]
unsafe fn avx<const W: usize, const PAIR: bool>(d: &Inputs<'_>) -> [[Simd<f32, W>; 2]; 4] {
    reduce::<W, false, PAIR>(d)
}

#[inline(always)]
fn reduce<const W: usize, const FMA: bool, const PAIR: bool>(
    d: &Inputs<'_>,
) -> [[Simd<f32, W>; 2]; 4] {
    let [mut s0, mut s1, mut s2, mut s3] = d.seeds.map(Simd::splat);
    let (mut t0, mut t1, mut t2, mut t3) = (s0, s1, s2, s3);
    for (ci, panel) in d.panel.iter().enumerate() {
        // SAFETY: packet proves a complete W (or 2W) input packet at every
        // channel stride, and a four-aligned lane in the typed OC16 panel.
        let (value, weights) = unsafe {
            (
                Simd::from_array(
                    d.input
                        .as_ptr()
                        .add(ci * d.plane)
                        .cast::<[f32; W]>()
                        .read_unaligned(),
                ),
                panel
                    .as_ptr()
                    .add(d.lane)
                    .cast::<[f32; 4]>()
                    .read_unaligned(),
            )
        };
        let [w0, w1, w2, w3] = weights.map(Simd::splat);
        s0 = madd::<W, FMA>(value, w0, s0);
        s1 = madd::<W, FMA>(value, w1, s1);
        s2 = madd::<W, FMA>(value, w2, s2);
        s3 = madd::<W, FMA>(value, w3, s3);
        if PAIR {
            // SAFETY: the paired span was included in packet's bound above.
            let value = Simd::from_array(unsafe {
                d.input
                    .as_ptr()
                    .add(ci * d.plane + W)
                    .cast::<[f32; W]>()
                    .read_unaligned()
            });
            t0 = madd::<W, FMA>(value, w0, t0);
            t1 = madd::<W, FMA>(value, w1, t1);
            t2 = madd::<W, FMA>(value, w2, t2);
            t3 = madd::<W, FMA>(value, w3, t3);
        }
    }
    [[s0, t0], [s1, t1], [s2, t2], [s3, t3]]
}
