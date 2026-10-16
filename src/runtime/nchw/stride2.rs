//! Pinned packed stride-two 3x3 tiles. Shared OC8 coefficient panels feed
//! eight or sixteen ordered output chains. Portable4 prefers two adjacent
//! spatial packets sharing each coefficient. Every source load stays in-row.
use super::*;
use std::simd::Swizzle;

struct Even;
impl<const N: usize> Swizzle<N> for Even {
    const INDEX: [usize; N] = {
        let mut indices = [0; N];
        let mut lane = 0;
        while lane < N {
            indices[lane] = lane * 2;
            lane += 1;
        }
        indices
    };
}

#[inline(always)]
pub(super) fn load_even<const N: usize>(source: &[f32]) -> Simd<f32, N> {
    assert!(source.len() >= 2 * N - 1);
    let low = Simd::<f32, N>::from_slice(source);
    let high = if source.len() >= 2 * N {
        Simd::<f32, N>::from_slice(&source[N..])
    } else {
        // The last odd element is not selected. A complete row ending at the
        // last even lane needs no read into the next row, channel or allocation.
        Simd::<f32, N>::from_array(std::array::from_fn(|lane| {
            source.get(N + lane).copied().unwrap_or(0.)
        }))
    };
    Even::concat_swizzle(low, high)
}

#[inline(always)]
fn full_width(p: &Conv2d, x: usize, lanes: usize) -> bool {
    x > 0 && x + lanes <= p.output_width && (x + lanes - 1) * 2 + 1 < p.width
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn execute_impl<const N: usize, const FMA: bool, const PAIR16: bool>(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    out: &mut [f32],
    batch: usize,
    first: usize,
    count: usize,
    store: &impl simd::StoreTransform,
) {
    let outputs = if PAIR16 { 16 } else { 8 };
    assert!(first.is_multiple_of(8) && count.is_multiple_of(outputs));
    let plane = p.output_height * p.output_width;
    for co in (first..first + count).step_by(outputs) {
        for y in 0..p.output_height {
            let full_height = y > 0 && y * 2 + 1 < p.height;
            let mut x = 0;
            while x < p.output_width {
                if N == 4 && !PAIR16 && full_height && full_width(p, x, 8) {
                    dual4::<FMA>(p, input, bias, packed, out, batch, co, first, y, x, store);
                    x += 8;
                } else if full_height && full_width(p, x, N) {
                    if PAIR16 {
                        sixteen::<N, FMA>(
                            p, input, bias, packed, out, batch, co, first, y, x, store,
                        );
                    } else {
                        eight::<N, FMA>(p, input, bias, packed, out, batch, co, first, y, x, store);
                    }
                    x += N;
                } else {
                    let border = if N == 8 && !PAIR16 {
                        Some(reduce3::border::<2>(
                            p, input, bias, packed, batch, co, y, x,
                        ))
                    } else {
                        None
                    };
                    for lane in 0..outputs {
                        let local = (co + lane - first) * plane + y * p.output_width + x;
                        let global = (batch * p.output_channels + co + lane) * plane
                            + y * p.output_width
                            + x;
                        out[local] = store.scalar(
                            border.map_or_else(
                                || edge(p, input, weights, bias, batch, co + lane, y, x),
                                |values| values[lane],
                            ),
                            global,
                        );
                    }
                    x += 1;
                }
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn eight<const N: usize, const FMA: bool>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    out: &mut [f32],
    batch: usize,
    co: usize,
    first: usize,
    y: usize,
    x: usize,
    store: &impl simd::StoreTransform,
) {
    let plane = p.output_height * p.output_width;
    if N == 8 {
        let sums = reduce3::eight::<FMA, 2>(p, input, bias, packed, batch, co, y, x);
        let local = (co - first) * plane + y * p.output_width + x;
        let global = (batch * p.output_channels + co) * plane + y * p.output_width + x;
        for (lane, value) in sums.into_iter().enumerate() {
            store
                .vector(value, global + lane * plane)
                .copy_to_slice(&mut out[local + lane * plane..][..8]);
        }
        return;
    }
    let inner = p.input_channels * 9;
    let low = (co / 16) * inner * 16 + co % 16;
    let mut s0 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co] });
    let mut s1 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 1] });
    let mut s2 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 2] });
    let mut s3 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 3] });
    let mut s4 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 4] });
    let mut s5 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 5] });
    let mut s6 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 6] });
    let mut s7 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 7] });
    for ci in 0..p.input_channels {
        for ky in 0..3 {
            let row = ((batch * p.input_channels + ci) * p.height + y * 2 + ky - 1) * p.width;
            let source = &input[row..row + p.width];
            for kx in 0..3 {
                let at = x * 2 - 1 + kx;
                let value = load_even::<N>(&source[at..]);
                let k = (ci * 3 + ky) * 3 + kx;
                let w0 = &packed.data()[low + k * 16..low + k * 16 + 8];
                s0 = madd::<N, FMA>(value, Simd::splat(w0[0]), s0);
                s1 = madd::<N, FMA>(value, Simd::splat(w0[1]), s1);
                s2 = madd::<N, FMA>(value, Simd::splat(w0[2]), s2);
                s3 = madd::<N, FMA>(value, Simd::splat(w0[3]), s3);
                s4 = madd::<N, FMA>(value, Simd::splat(w0[4]), s4);
                s5 = madd::<N, FMA>(value, Simd::splat(w0[5]), s5);
                s6 = madd::<N, FMA>(value, Simd::splat(w0[6]), s6);
                s7 = madd::<N, FMA>(value, Simd::splat(w0[7]), s7);
            }
        }
    }
    let local = (co - first) * plane + y * p.output_width + x;
    let global = (batch * p.output_channels + co) * plane + y * p.output_width + x;
    store
        .vector(s0, global)
        .copy_to_slice(&mut out[local..][..N]);
    store
        .vector(s1, global + plane)
        .copy_to_slice(&mut out[local + plane..][..N]);
    store
        .vector(s2, global + 2 * plane)
        .copy_to_slice(&mut out[local + 2 * plane..][..N]);
    store
        .vector(s3, global + 3 * plane)
        .copy_to_slice(&mut out[local + 3 * plane..][..N]);
    store
        .vector(s4, global + 4 * plane)
        .copy_to_slice(&mut out[local + 4 * plane..][..N]);
    store
        .vector(s5, global + 5 * plane)
        .copy_to_slice(&mut out[local + 5 * plane..][..N]);
    store
        .vector(s6, global + 6 * plane)
        .copy_to_slice(&mut out[local + 6 * plane..][..N]);
    store
        .vector(s7, global + 7 * plane)
        .copy_to_slice(&mut out[local + 7 * plane..][..N]);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn sixteen<const N: usize, const FMA: bool>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    out: &mut [f32],
    batch: usize,
    co: usize,
    first: usize,
    y: usize,
    x: usize,
    store: &impl simd::StoreTransform,
) {
    let plane = p.output_height * p.output_width;
    let inner = p.input_channels * 9;
    let low = (co / 16) * inner * 16 + co % 16;
    let high = ((co + 8) / 16) * inner * 16 + (co + 8) % 16;
    let mut s0 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co] });
    let mut s1 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 1] });
    let mut s2 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 2] });
    let mut s3 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 3] });
    let mut s4 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 4] });
    let mut s5 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 5] });
    let mut s6 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 6] });
    let mut s7 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 7] });
    let mut s8 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 8] });
    let mut s9 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 9] });
    let mut s10 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 10] });
    let mut s11 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 11] });
    let mut s12 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 12] });
    let mut s13 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 13] });
    let mut s14 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 14] });
    let mut s15 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 15] });
    for ci in 0..p.input_channels {
        for ky in 0..3 {
            let row = ((batch * p.input_channels + ci) * p.height + y * 2 + ky - 1) * p.width;
            let source = &input[row..row + p.width];
            for kx in 0..3 {
                let at = x * 2 - 1 + kx;
                let value = load_even::<N>(&source[at..]);
                let k = (ci * 3 + ky) * 3 + kx;
                let w0 = &packed.data()[low + k * 16..low + k * 16 + 8];
                let w8 = &packed.data()[high + k * 16..high + k * 16 + 8];
                s0 = madd::<N, FMA>(value, Simd::splat(w0[0]), s0);
                s1 = madd::<N, FMA>(value, Simd::splat(w0[1]), s1);
                s2 = madd::<N, FMA>(value, Simd::splat(w0[2]), s2);
                s3 = madd::<N, FMA>(value, Simd::splat(w0[3]), s3);
                s4 = madd::<N, FMA>(value, Simd::splat(w0[4]), s4);
                s5 = madd::<N, FMA>(value, Simd::splat(w0[5]), s5);
                s6 = madd::<N, FMA>(value, Simd::splat(w0[6]), s6);
                s7 = madd::<N, FMA>(value, Simd::splat(w0[7]), s7);
                s8 = madd::<N, FMA>(value, Simd::splat(w8[0]), s8);
                s9 = madd::<N, FMA>(value, Simd::splat(w8[1]), s9);
                s10 = madd::<N, FMA>(value, Simd::splat(w8[2]), s10);
                s11 = madd::<N, FMA>(value, Simd::splat(w8[3]), s11);
                s12 = madd::<N, FMA>(value, Simd::splat(w8[4]), s12);
                s13 = madd::<N, FMA>(value, Simd::splat(w8[5]), s13);
                s14 = madd::<N, FMA>(value, Simd::splat(w8[6]), s14);
                s15 = madd::<N, FMA>(value, Simd::splat(w8[7]), s15);
            }
        }
    }
    let local = (co - first) * plane + y * p.output_width + x;
    let global = (batch * p.output_channels + co) * plane + y * p.output_width + x;
    store
        .vector(s0, global)
        .copy_to_slice(&mut out[local..][..N]);
    store
        .vector(s1, global + plane)
        .copy_to_slice(&mut out[local + plane..][..N]);
    store
        .vector(s2, global + 2 * plane)
        .copy_to_slice(&mut out[local + 2 * plane..][..N]);
    store
        .vector(s3, global + 3 * plane)
        .copy_to_slice(&mut out[local + 3 * plane..][..N]);
    store
        .vector(s4, global + 4 * plane)
        .copy_to_slice(&mut out[local + 4 * plane..][..N]);
    store
        .vector(s5, global + 5 * plane)
        .copy_to_slice(&mut out[local + 5 * plane..][..N]);
    store
        .vector(s6, global + 6 * plane)
        .copy_to_slice(&mut out[local + 6 * plane..][..N]);
    store
        .vector(s7, global + 7 * plane)
        .copy_to_slice(&mut out[local + 7 * plane..][..N]);
    store
        .vector(s8, global + 8 * plane)
        .copy_to_slice(&mut out[local + 8 * plane..][..N]);
    store
        .vector(s9, global + 9 * plane)
        .copy_to_slice(&mut out[local + 9 * plane..][..N]);
    store
        .vector(s10, global + 10 * plane)
        .copy_to_slice(&mut out[local + 10 * plane..][..N]);
    store
        .vector(s11, global + 11 * plane)
        .copy_to_slice(&mut out[local + 11 * plane..][..N]);
    store
        .vector(s12, global + 12 * plane)
        .copy_to_slice(&mut out[local + 12 * plane..][..N]);
    store
        .vector(s13, global + 13 * plane)
        .copy_to_slice(&mut out[local + 13 * plane..][..N]);
    store
        .vector(s14, global + 14 * plane)
        .copy_to_slice(&mut out[local + 14 * plane..][..N]);
    store
        .vector(s15, global + 15 * plane)
        .copy_to_slice(&mut out[local + 15 * plane..][..N]);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dual4<const FMA: bool>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    out: &mut [f32],
    batch: usize,
    co: usize,
    first: usize,
    y: usize,
    x: usize,
    store: &impl simd::StoreTransform,
) {
    let plane = p.output_height * p.output_width;
    let inner = p.input_channels * 9;
    let low = (co / 16) * inner * 16 + co % 16;
    let mut s0 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co] });
    let mut s1 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 1] });
    let mut s2 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 2] });
    let mut s3 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 3] });
    let mut s4 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 4] });
    let mut s5 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 5] });
    let mut s6 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 6] });
    let mut s7 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 7] });
    let mut s8 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co] });
    let mut s9 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 1] });
    let mut s10 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 2] });
    let mut s11 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 3] });
    let mut s12 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 4] });
    let mut s13 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 5] });
    let mut s14 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 6] });
    let mut s15 = Simd::<f32, 4>::splat(if bias.is_empty() { 0. } else { bias[co + 7] });
    for ci in 0..p.input_channels {
        for ky in 0..3 {
            let row = ((batch * p.input_channels + ci) * p.height + y * 2 + ky - 1) * p.width;
            let source = &input[row..row + p.width];
            for kx in 0..3 {
                let at = x * 2 - 1 + kx;
                let value = load_even::<4>(&source[at..]);
                let next = load_even::<4>(&source[at + 8..]);
                let k = (ci * 3 + ky) * 3 + kx;
                let w0 = &packed.data()[low + k * 16..low + k * 16 + 8];
                let w = Simd::splat(w0[0]);
                s0 = madd::<4, FMA>(value, w, s0);
                s8 = madd::<4, FMA>(next, w, s8);
                let w = Simd::splat(w0[1]);
                s1 = madd::<4, FMA>(value, w, s1);
                s9 = madd::<4, FMA>(next, w, s9);
                let w = Simd::splat(w0[2]);
                s2 = madd::<4, FMA>(value, w, s2);
                s10 = madd::<4, FMA>(next, w, s10);
                let w = Simd::splat(w0[3]);
                s3 = madd::<4, FMA>(value, w, s3);
                s11 = madd::<4, FMA>(next, w, s11);
                let w = Simd::splat(w0[4]);
                s4 = madd::<4, FMA>(value, w, s4);
                s12 = madd::<4, FMA>(next, w, s12);
                let w = Simd::splat(w0[5]);
                s5 = madd::<4, FMA>(value, w, s5);
                s13 = madd::<4, FMA>(next, w, s13);
                let w = Simd::splat(w0[6]);
                s6 = madd::<4, FMA>(value, w, s6);
                s14 = madd::<4, FMA>(next, w, s14);
                let w = Simd::splat(w0[7]);
                s7 = madd::<4, FMA>(value, w, s7);
                s15 = madd::<4, FMA>(next, w, s15);
            }
        }
    }
    let local = (co - first) * plane + y * p.output_width + x;
    let global = (batch * p.output_channels + co) * plane + y * p.output_width + x;
    store
        .vector(s0, global)
        .copy_to_slice(&mut out[local..][..4]);
    store
        .vector(s8, global + 4)
        .copy_to_slice(&mut out[local + 4..][..4]);
    store
        .vector(s1, global + plane)
        .copy_to_slice(&mut out[local + plane..][..4]);
    store
        .vector(s9, global + plane + 4)
        .copy_to_slice(&mut out[local + plane + 4..][..4]);
    store
        .vector(s2, global + 2 * plane)
        .copy_to_slice(&mut out[local + 2 * plane..][..4]);
    store
        .vector(s10, global + 2 * plane + 4)
        .copy_to_slice(&mut out[local + 2 * plane + 4..][..4]);
    store
        .vector(s3, global + 3 * plane)
        .copy_to_slice(&mut out[local + 3 * plane..][..4]);
    store
        .vector(s11, global + 3 * plane + 4)
        .copy_to_slice(&mut out[local + 3 * plane + 4..][..4]);
    store
        .vector(s4, global + 4 * plane)
        .copy_to_slice(&mut out[local + 4 * plane..][..4]);
    store
        .vector(s12, global + 4 * plane + 4)
        .copy_to_slice(&mut out[local + 4 * plane + 4..][..4]);
    store
        .vector(s5, global + 5 * plane)
        .copy_to_slice(&mut out[local + 5 * plane..][..4]);
    store
        .vector(s13, global + 5 * plane + 4)
        .copy_to_slice(&mut out[local + 5 * plane + 4..][..4]);
    store
        .vector(s6, global + 6 * plane)
        .copy_to_slice(&mut out[local + 6 * plane..][..4]);
    store
        .vector(s14, global + 6 * plane + 4)
        .copy_to_slice(&mut out[local + 6 * plane + 4..][..4]);
    store
        .vector(s7, global + 7 * plane)
        .copy_to_slice(&mut out[local + 7 * plane..][..4]);
    store
        .vector(s15, global + 7 * plane + 4)
        .copy_to_slice(&mut out[local + 7 * plane + 4..][..4]);
}

super::native::outline! {
    pub(super) fn execute => execute_impl <
        const N: usize,
        const FMA: bool,
        const PAIR16: bool,
    > (
        p: &Conv2d,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        packed: &PackedMatrix,
        out: &mut [f32],
        batch: usize,
        first: usize,
        count: usize,
    )
    store: Store
}
