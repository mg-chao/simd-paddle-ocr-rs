//! ARM64 six-row, sixteen-channel FMA kernels.
//!
//! Pointwise groups four K steps per input load. Two weight vectors at a
//! time leave room for the six input vectors alongside 24 accumulators in
//! the 32 NEON registers. Dense reductions reuse the checked scalar readers.
use super::*;
use std::arch::aarch64::*;

#[inline(always)]
fn native(value: Simd<f32, 4>) -> float32x4_t {
    // SAFETY: the local array contains four initialized, unaligned-loadable
    // floats. This module is reached only through the NEON entry points.
    unsafe { vld1q_f32(value.as_array().as_ptr()) }
}

#[inline(always)]
fn portable(value: float32x4_t) -> Simd<f32, 4> {
    let mut result = [0.; 4];
    // SAFETY: result owns four writable floats; NEON was checked on entry.
    unsafe { vst1q_f32(result.as_mut_ptr(), value) };
    Simd::from_array(result)
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn execute(
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
) {
    let rows = output.len() / columns;
    for group in (0..rows).step_by(GROUP_ROWS) {
        let count = (rows - group).min(GROUP_ROWS);
        let input = &input[group * inner..(group + count) * inner];
        for (block, panel) in packed.data().chunks_exact(inner * 16).enumerate() {
            let column = block * 16;
            let seed = std::array::from_fn(|v| initial::<4>(bias, column + v * 4));
            for tile in (0..count).step_by(ROWS) {
                // SAFETY: row_inputs binds complete K spans and duplicates
                // the last valid row for a short tile. The panel covers K.
                let sums =
                    unsafe { pointwise_tile(row_inputs(input, tile, count, inner), panel, seed) };
                for (r, values) in sums.iter().take(count - tile).enumerate() {
                    // Bind and apply the fused store once for the complete
                    // channel panel; the reduction remains four NEON vectors.
                    let sum = Simd::<f32, 16>::from_array(std::array::from_fn(|channel| {
                        values[channel / 4][channel % 4]
                    }));
                    store_row(sum, output, group + tile + r, column, columns, first, store);
                }
            }
        }
    }
}

// Keep activation/residual stores outside the register-critical K loop.
#[target_feature(enable = "neon")]
#[inline(never)]
unsafe fn pointwise_tile(
    rows: [&[f32]; ROWS],
    panel: &[f32],
    seed: [Simd<f32, 4>; 4],
) -> [[Simd<f32, 4>; 4]; ROWS] {
    assert_eq!(panel.len() % 16, 0);
    let inner = panel.len() / 16;
    assert!(rows.iter().all(|row| row.len() == inner));
    let (mut a00, mut a01, mut a02, mut a03) = (
        native(seed[0]),
        native(seed[1]),
        native(seed[2]),
        native(seed[3]),
    );
    let (mut a10, mut a11, mut a12, mut a13) = (
        native(seed[0]),
        native(seed[1]),
        native(seed[2]),
        native(seed[3]),
    );
    let (mut a20, mut a21, mut a22, mut a23) = (
        native(seed[0]),
        native(seed[1]),
        native(seed[2]),
        native(seed[3]),
    );
    let (mut a30, mut a31, mut a32, mut a33) = (
        native(seed[0]),
        native(seed[1]),
        native(seed[2]),
        native(seed[3]),
    );
    let (mut a40, mut a41, mut a42, mut a43) = (
        native(seed[0]),
        native(seed[1]),
        native(seed[2]),
        native(seed[3]),
    );
    let (mut a50, mut a51, mut a52, mut a53) = (
        native(seed[0]),
        native(seed[1]),
        native(seed[2]),
        native(seed[3]),
    );
    for block in 0..inner / 4 {
        let k = block * 4;
        // SAFETY: k..k+4 lies within every checked input span.
        let v0 = unsafe { vld1q_f32(rows[0].as_ptr().add(k)) };
        let v1 = unsafe { vld1q_f32(rows[1].as_ptr().add(k)) };
        let v2 = unsafe { vld1q_f32(rows[2].as_ptr().add(k)) };
        let v3 = unsafe { vld1q_f32(rows[3].as_ptr().add(k)) };
        let v4 = unsafe { vld1q_f32(rows[4].as_ptr().add(k)) };
        let v5 = unsafe { vld1q_f32(rows[5].as_ptr().add(k)) };
        // Ascending K preserves the bias-first FMA chain (lane 0).
        {
            // SAFETY: each full packed K row contains sixteen floats.
            let w0 = unsafe { vld1q_f32(panel.as_ptr().add(k * 16)) };
            let w1 = unsafe { vld1q_f32(panel.as_ptr().add(k * 16 + 4)) };
            a00 = vfmaq_laneq_f32::<0>(a00, w0, v0);
            a01 = vfmaq_laneq_f32::<0>(a01, w1, v0);
            a10 = vfmaq_laneq_f32::<0>(a10, w0, v1);
            a11 = vfmaq_laneq_f32::<0>(a11, w1, v1);
            a20 = vfmaq_laneq_f32::<0>(a20, w0, v2);
            a21 = vfmaq_laneq_f32::<0>(a21, w1, v2);
            a30 = vfmaq_laneq_f32::<0>(a30, w0, v3);
            a31 = vfmaq_laneq_f32::<0>(a31, w1, v3);
            a40 = vfmaq_laneq_f32::<0>(a40, w0, v4);
            a41 = vfmaq_laneq_f32::<0>(a41, w1, v4);
            a50 = vfmaq_laneq_f32::<0>(a50, w0, v5);
            a51 = vfmaq_laneq_f32::<0>(a51, w1, v5);
        }
        {
            // SAFETY: each full packed K row contains sixteen floats.
            let w2 = unsafe { vld1q_f32(panel.as_ptr().add(k * 16 + 8)) };
            let w3 = unsafe { vld1q_f32(panel.as_ptr().add(k * 16 + 12)) };
            a02 = vfmaq_laneq_f32::<0>(a02, w2, v0);
            a03 = vfmaq_laneq_f32::<0>(a03, w3, v0);
            a12 = vfmaq_laneq_f32::<0>(a12, w2, v1);
            a13 = vfmaq_laneq_f32::<0>(a13, w3, v1);
            a22 = vfmaq_laneq_f32::<0>(a22, w2, v2);
            a23 = vfmaq_laneq_f32::<0>(a23, w3, v2);
            a32 = vfmaq_laneq_f32::<0>(a32, w2, v3);
            a33 = vfmaq_laneq_f32::<0>(a33, w3, v3);
            a42 = vfmaq_laneq_f32::<0>(a42, w2, v4);
            a43 = vfmaq_laneq_f32::<0>(a43, w3, v4);
            a52 = vfmaq_laneq_f32::<0>(a52, w2, v5);
            a53 = vfmaq_laneq_f32::<0>(a53, w3, v5);
        }
        // Ascending K preserves the bias-first FMA chain (lane 1).
        {
            // SAFETY: each full packed K row contains sixteen floats.
            let w0 = unsafe { vld1q_f32(panel.as_ptr().add((k + 1) * 16)) };
            let w1 = unsafe { vld1q_f32(panel.as_ptr().add((k + 1) * 16 + 4)) };
            a00 = vfmaq_laneq_f32::<1>(a00, w0, v0);
            a01 = vfmaq_laneq_f32::<1>(a01, w1, v0);
            a10 = vfmaq_laneq_f32::<1>(a10, w0, v1);
            a11 = vfmaq_laneq_f32::<1>(a11, w1, v1);
            a20 = vfmaq_laneq_f32::<1>(a20, w0, v2);
            a21 = vfmaq_laneq_f32::<1>(a21, w1, v2);
            a30 = vfmaq_laneq_f32::<1>(a30, w0, v3);
            a31 = vfmaq_laneq_f32::<1>(a31, w1, v3);
            a40 = vfmaq_laneq_f32::<1>(a40, w0, v4);
            a41 = vfmaq_laneq_f32::<1>(a41, w1, v4);
            a50 = vfmaq_laneq_f32::<1>(a50, w0, v5);
            a51 = vfmaq_laneq_f32::<1>(a51, w1, v5);
        }
        {
            // SAFETY: each full packed K row contains sixteen floats.
            let w2 = unsafe { vld1q_f32(panel.as_ptr().add((k + 1) * 16 + 8)) };
            let w3 = unsafe { vld1q_f32(panel.as_ptr().add((k + 1) * 16 + 12)) };
            a02 = vfmaq_laneq_f32::<1>(a02, w2, v0);
            a03 = vfmaq_laneq_f32::<1>(a03, w3, v0);
            a12 = vfmaq_laneq_f32::<1>(a12, w2, v1);
            a13 = vfmaq_laneq_f32::<1>(a13, w3, v1);
            a22 = vfmaq_laneq_f32::<1>(a22, w2, v2);
            a23 = vfmaq_laneq_f32::<1>(a23, w3, v2);
            a32 = vfmaq_laneq_f32::<1>(a32, w2, v3);
            a33 = vfmaq_laneq_f32::<1>(a33, w3, v3);
            a42 = vfmaq_laneq_f32::<1>(a42, w2, v4);
            a43 = vfmaq_laneq_f32::<1>(a43, w3, v4);
            a52 = vfmaq_laneq_f32::<1>(a52, w2, v5);
            a53 = vfmaq_laneq_f32::<1>(a53, w3, v5);
        }
        // Ascending K preserves the bias-first FMA chain (lane 2).
        {
            // SAFETY: each full packed K row contains sixteen floats.
            let w0 = unsafe { vld1q_f32(panel.as_ptr().add((k + 2) * 16)) };
            let w1 = unsafe { vld1q_f32(panel.as_ptr().add((k + 2) * 16 + 4)) };
            a00 = vfmaq_laneq_f32::<2>(a00, w0, v0);
            a01 = vfmaq_laneq_f32::<2>(a01, w1, v0);
            a10 = vfmaq_laneq_f32::<2>(a10, w0, v1);
            a11 = vfmaq_laneq_f32::<2>(a11, w1, v1);
            a20 = vfmaq_laneq_f32::<2>(a20, w0, v2);
            a21 = vfmaq_laneq_f32::<2>(a21, w1, v2);
            a30 = vfmaq_laneq_f32::<2>(a30, w0, v3);
            a31 = vfmaq_laneq_f32::<2>(a31, w1, v3);
            a40 = vfmaq_laneq_f32::<2>(a40, w0, v4);
            a41 = vfmaq_laneq_f32::<2>(a41, w1, v4);
            a50 = vfmaq_laneq_f32::<2>(a50, w0, v5);
            a51 = vfmaq_laneq_f32::<2>(a51, w1, v5);
        }
        {
            // SAFETY: each full packed K row contains sixteen floats.
            let w2 = unsafe { vld1q_f32(panel.as_ptr().add((k + 2) * 16 + 8)) };
            let w3 = unsafe { vld1q_f32(panel.as_ptr().add((k + 2) * 16 + 12)) };
            a02 = vfmaq_laneq_f32::<2>(a02, w2, v0);
            a03 = vfmaq_laneq_f32::<2>(a03, w3, v0);
            a12 = vfmaq_laneq_f32::<2>(a12, w2, v1);
            a13 = vfmaq_laneq_f32::<2>(a13, w3, v1);
            a22 = vfmaq_laneq_f32::<2>(a22, w2, v2);
            a23 = vfmaq_laneq_f32::<2>(a23, w3, v2);
            a32 = vfmaq_laneq_f32::<2>(a32, w2, v3);
            a33 = vfmaq_laneq_f32::<2>(a33, w3, v3);
            a42 = vfmaq_laneq_f32::<2>(a42, w2, v4);
            a43 = vfmaq_laneq_f32::<2>(a43, w3, v4);
            a52 = vfmaq_laneq_f32::<2>(a52, w2, v5);
            a53 = vfmaq_laneq_f32::<2>(a53, w3, v5);
        }
        // Ascending K preserves the bias-first FMA chain (lane 3).
        {
            // SAFETY: each full packed K row contains sixteen floats.
            let w0 = unsafe { vld1q_f32(panel.as_ptr().add((k + 3) * 16)) };
            let w1 = unsafe { vld1q_f32(panel.as_ptr().add((k + 3) * 16 + 4)) };
            a00 = vfmaq_laneq_f32::<3>(a00, w0, v0);
            a01 = vfmaq_laneq_f32::<3>(a01, w1, v0);
            a10 = vfmaq_laneq_f32::<3>(a10, w0, v1);
            a11 = vfmaq_laneq_f32::<3>(a11, w1, v1);
            a20 = vfmaq_laneq_f32::<3>(a20, w0, v2);
            a21 = vfmaq_laneq_f32::<3>(a21, w1, v2);
            a30 = vfmaq_laneq_f32::<3>(a30, w0, v3);
            a31 = vfmaq_laneq_f32::<3>(a31, w1, v3);
            a40 = vfmaq_laneq_f32::<3>(a40, w0, v4);
            a41 = vfmaq_laneq_f32::<3>(a41, w1, v4);
            a50 = vfmaq_laneq_f32::<3>(a50, w0, v5);
            a51 = vfmaq_laneq_f32::<3>(a51, w1, v5);
        }
        {
            // SAFETY: each full packed K row contains sixteen floats.
            let w2 = unsafe { vld1q_f32(panel.as_ptr().add((k + 3) * 16 + 8)) };
            let w3 = unsafe { vld1q_f32(panel.as_ptr().add((k + 3) * 16 + 12)) };
            a02 = vfmaq_laneq_f32::<3>(a02, w2, v0);
            a03 = vfmaq_laneq_f32::<3>(a03, w3, v0);
            a12 = vfmaq_laneq_f32::<3>(a12, w2, v1);
            a13 = vfmaq_laneq_f32::<3>(a13, w3, v1);
            a22 = vfmaq_laneq_f32::<3>(a22, w2, v2);
            a23 = vfmaq_laneq_f32::<3>(a23, w3, v2);
            a32 = vfmaq_laneq_f32::<3>(a32, w2, v3);
            a33 = vfmaq_laneq_f32::<3>(a33, w3, v3);
            a42 = vfmaq_laneq_f32::<3>(a42, w2, v4);
            a43 = vfmaq_laneq_f32::<3>(a43, w3, v4);
            a52 = vfmaq_laneq_f32::<3>(a52, w2, v5);
            a53 = vfmaq_laneq_f32::<3>(a53, w3, v5);
        }
    }
    for k in inner / 4 * 4..inner {
        // SAFETY: k is in the complete K interval of the packed panel.
        let w0 = unsafe { vld1q_f32(panel.as_ptr().add(k * 16)) };
        let w1 = unsafe { vld1q_f32(panel.as_ptr().add(k * 16 + 4)) };
        let w2 = unsafe { vld1q_f32(panel.as_ptr().add(k * 16 + 8)) };
        let w3 = unsafe { vld1q_f32(panel.as_ptr().add(k * 16 + 12)) };
        {
            // SAFETY: the K tail remains in this checked row.
            let value = unsafe { *rows[0].get_unchecked(k) };
            a00 = vfmaq_n_f32(a00, w0, value);
            a01 = vfmaq_n_f32(a01, w1, value);
            a02 = vfmaq_n_f32(a02, w2, value);
            a03 = vfmaq_n_f32(a03, w3, value);
        }
        {
            // SAFETY: the K tail remains in this checked row.
            let value = unsafe { *rows[1].get_unchecked(k) };
            a10 = vfmaq_n_f32(a10, w0, value);
            a11 = vfmaq_n_f32(a11, w1, value);
            a12 = vfmaq_n_f32(a12, w2, value);
            a13 = vfmaq_n_f32(a13, w3, value);
        }
        {
            // SAFETY: the K tail remains in this checked row.
            let value = unsafe { *rows[2].get_unchecked(k) };
            a20 = vfmaq_n_f32(a20, w0, value);
            a21 = vfmaq_n_f32(a21, w1, value);
            a22 = vfmaq_n_f32(a22, w2, value);
            a23 = vfmaq_n_f32(a23, w3, value);
        }
        {
            // SAFETY: the K tail remains in this checked row.
            let value = unsafe { *rows[3].get_unchecked(k) };
            a30 = vfmaq_n_f32(a30, w0, value);
            a31 = vfmaq_n_f32(a31, w1, value);
            a32 = vfmaq_n_f32(a32, w2, value);
            a33 = vfmaq_n_f32(a33, w3, value);
        }
        {
            // SAFETY: the K tail remains in this checked row.
            let value = unsafe { *rows[4].get_unchecked(k) };
            a40 = vfmaq_n_f32(a40, w0, value);
            a41 = vfmaq_n_f32(a41, w1, value);
            a42 = vfmaq_n_f32(a42, w2, value);
            a43 = vfmaq_n_f32(a43, w3, value);
        }
        {
            // SAFETY: the K tail remains in this checked row.
            let value = unsafe { *rows[5].get_unchecked(k) };
            a50 = vfmaq_n_f32(a50, w0, value);
            a51 = vfmaq_n_f32(a51, w1, value);
            a52 = vfmaq_n_f32(a52, w2, value);
            a53 = vfmaq_n_f32(a53, w3, value);
        }
    }
    [
        [portable(a00), portable(a01), portable(a02), portable(a03)],
        [portable(a10), portable(a11), portable(a12), portable(a13)],
        [portable(a20), portable(a21), portable(a22), portable(a23)],
        [portable(a30), portable(a31), portable(a32), portable(a33)],
        [portable(a40), portable(a41), portable(a42), portable(a43)],
        [portable(a50), portable(a51), portable(a52), portable(a53)],
    ]
}

/// Full-width dense tile. Reader descriptors establish every scalar bound.
/// SAFETY: the reader must bind six rows across tap_count * channels; the
/// caller entered NEON and seeds preserve the preceding ascending K chain.
#[target_feature(enable = "neon")]
#[inline(never)]
pub(in super::super) unsafe fn dense_tile(
    reader: &impl PortableReader,
    tap_count: usize,
    channels: usize,
    panel: &[f32],
    k0: usize,
    seeds: [[Simd<f32, 4>; 4]; ROWS],
) -> [[Simd<f32, 4>; 4]; ROWS] {
    assert!(tap_count > 0);
    assert_eq!(panel.len() % 16, 0);
    assert!(k0.checked_add(panel.len() / 16).is_some_and(|end| {
        channels
            .checked_mul(tap_count)
            .is_some_and(|inner| end <= inner)
    }));
    let (mut a00, mut a01, mut a02, mut a03) = (
        native(seeds[0][0]),
        native(seeds[0][1]),
        native(seeds[0][2]),
        native(seeds[0][3]),
    );
    let (mut a10, mut a11, mut a12, mut a13) = (
        native(seeds[1][0]),
        native(seeds[1][1]),
        native(seeds[1][2]),
        native(seeds[1][3]),
    );
    let (mut a20, mut a21, mut a22, mut a23) = (
        native(seeds[2][0]),
        native(seeds[2][1]),
        native(seeds[2][2]),
        native(seeds[2][3]),
    );
    let (mut a30, mut a31, mut a32, mut a33) = (
        native(seeds[3][0]),
        native(seeds[3][1]),
        native(seeds[3][2]),
        native(seeds[3][3]),
    );
    let (mut a40, mut a41, mut a42, mut a43) = (
        native(seeds[4][0]),
        native(seeds[4][1]),
        native(seeds[4][2]),
        native(seeds[4][3]),
    );
    let (mut a50, mut a51, mut a52, mut a53) = (
        native(seeds[5][0]),
        native(seeds[5][1]),
        native(seeds[5][2]),
        native(seeds[5][3]),
    );
    let (mut channel, mut tap) = (k0 / tap_count, k0 % tap_count);
    for weights in panel.as_chunks::<16>().0 {
        // SAFETY: each complete panel row contains all sixteen weights.
        let w0 = unsafe { vld1q_f32(weights.as_ptr().add(0)) };
        let w1 = unsafe { vld1q_f32(weights.as_ptr().add(4)) };
        let w2 = unsafe { vld1q_f32(weights.as_ptr().add(8)) };
        let w3 = unsafe { vld1q_f32(weights.as_ptr().add(12)) };
        {
            // SAFETY: the checked K interval and cursor retain the reader's
            // row/tap/channel invariant, including padding sentinels.
            let value = unsafe { reader.read::<0>(tap, channel) };
            a00 = vfmaq_n_f32(a00, w0, value);
            a01 = vfmaq_n_f32(a01, w1, value);
            a02 = vfmaq_n_f32(a02, w2, value);
            a03 = vfmaq_n_f32(a03, w3, value);
        }
        {
            // SAFETY: the checked K interval and cursor retain the reader's
            // row/tap/channel invariant, including padding sentinels.
            let value = unsafe { reader.read::<1>(tap, channel) };
            a10 = vfmaq_n_f32(a10, w0, value);
            a11 = vfmaq_n_f32(a11, w1, value);
            a12 = vfmaq_n_f32(a12, w2, value);
            a13 = vfmaq_n_f32(a13, w3, value);
        }
        {
            // SAFETY: the checked K interval and cursor retain the reader's
            // row/tap/channel invariant, including padding sentinels.
            let value = unsafe { reader.read::<2>(tap, channel) };
            a20 = vfmaq_n_f32(a20, w0, value);
            a21 = vfmaq_n_f32(a21, w1, value);
            a22 = vfmaq_n_f32(a22, w2, value);
            a23 = vfmaq_n_f32(a23, w3, value);
        }
        {
            // SAFETY: the checked K interval and cursor retain the reader's
            // row/tap/channel invariant, including padding sentinels.
            let value = unsafe { reader.read::<3>(tap, channel) };
            a30 = vfmaq_n_f32(a30, w0, value);
            a31 = vfmaq_n_f32(a31, w1, value);
            a32 = vfmaq_n_f32(a32, w2, value);
            a33 = vfmaq_n_f32(a33, w3, value);
        }
        {
            // SAFETY: the checked K interval and cursor retain the reader's
            // row/tap/channel invariant, including padding sentinels.
            let value = unsafe { reader.read::<4>(tap, channel) };
            a40 = vfmaq_n_f32(a40, w0, value);
            a41 = vfmaq_n_f32(a41, w1, value);
            a42 = vfmaq_n_f32(a42, w2, value);
            a43 = vfmaq_n_f32(a43, w3, value);
        }
        {
            // SAFETY: the checked K interval and cursor retain the reader's
            // row/tap/channel invariant, including padding sentinels.
            let value = unsafe { reader.read::<5>(tap, channel) };
            a50 = vfmaq_n_f32(a50, w0, value);
            a51 = vfmaq_n_f32(a51, w1, value);
            a52 = vfmaq_n_f32(a52, w2, value);
            a53 = vfmaq_n_f32(a53, w3, value);
        }
        tap += 1;
        if tap == tap_count {
            tap = 0;
            channel += 1;
        }
    }
    [
        [portable(a00), portable(a01), portable(a02), portable(a03)],
        [portable(a10), portable(a11), portable(a12), portable(a13)],
        [portable(a20), portable(a21), portable(a22), portable(a23)],
        [portable(a30), portable(a31), portable(a32), portable(a33)],
        [portable(a40), portable(a41), portable(a42), portable(a43)],
        [portable(a50), portable(a51), portable(a52), portable(a53)],
    ]
}
