//! ARM64 4 × 16 matrix tiles. Four consecutive K values are loaded once per
//! input row and selected by each FMLA instruction, avoiding scalar broadcasts.
//! Output accumulators still visit K in order; compact packed projection keeps
//! its distinct bias-after-dot-product and separate scalar-tail contracts.

use super::{IdentityStore, PackedMatrix, Simd, StoreTransform, argmax::LaneWinners};
use std::arch::aarch64::*;

const ROW_GROUP: usize = 128;
const WEIGHT_PANEL_BYTES: usize = 256 * 1024;

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn matmul(
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    // SAFETY: this entry has the same feature contract as the delegated entry.
    unsafe {
        matmul_bias_store(
            input,
            weights,
            packed,
            &[],
            output,
            rows,
            inner,
            columns,
            &IdentityStore,
            0,
        );
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn matmul_bias_store(
    input: &[f32],
    weights: &[f32],
    packed: Option<&PackedMatrix>,
    bias: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
    offset: usize,
) {
    assert!(inner > 0 && columns > 0);
    assert_eq!(
        input.len(),
        rows.checked_mul(inner).expect("matrix size overflow")
    );
    assert_eq!(
        output.len(),
        rows.checked_mul(columns).expect("matrix size overflow")
    );
    assert!(bias.is_empty() || bias.len() == columns);
    let weight_count = inner.checked_mul(columns).expect("matrix size overflow");
    // SAFETY: all complete matrix extents are validated here. PackedMatrix
    // guarantees padded 16-column panels. Inner helpers only visit these bounds.
    unsafe {
        if let Some(packed) = packed {
            assert!(
                packed.matches(inner, columns),
                "matrix panel shape mismatch"
            );
            assert!(weights.is_empty() || weights.len() == weight_count);
            dense::<true>(
                input,
                packed.data(),
                bias,
                output,
                rows,
                inner,
                columns,
                store,
                offset,
            );
        } else {
            assert_eq!(weights.len(), weight_count);
            dense::<false>(
                input, weights, bias, output, rows, inner, columns, store, offset,
            );
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn dense<const PACKED: bool>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
    offset: usize,
) {
    // SAFETY: the entry checked all slice lengths and enabled NEON. Each row
    // group is complete and the last partial column tile uses bounded loads.
    unsafe {
        let mut row = 0;
        while row + 4 <= rows {
            dense_rows::<4, PACKED>(
                input, weights, bias, output, row, inner, columns, store, offset,
            );
            row += 4;
        }
        while row < rows {
            dense_rows::<1, PACKED>(
                input, weights, bias, output, row, inner, columns, store, offset,
            );
            row += 1;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn dense_rows<const ROWS: usize, const PACKED: bool>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    row: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
    offset: usize,
) {
    // SAFETY: the caller provides ROWS valid rows and complete matrix extents.
    unsafe {
        for column in (0..columns / 16 * 16).step_by(16) {
            let sums = tile::<ROWS, PACKED, false, false>(
                input, weights, bias, row, column, inner, columns,
            );
            store_tile(sums, output, row, column, columns, store, offset);
        }
        let column = columns / 16 * 16;
        if column < columns {
            let sums = tile::<ROWS, PACKED, true, false>(
                input, weights, bias, row, column, inner, columns,
            );
            store_tile(sums, output, row, column, columns, store, offset);
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn store_tile<const ROWS: usize>(
    sums: [[float32x4_t; 4]; ROWS],
    output: &mut [f32],
    row: usize,
    column: usize,
    columns: usize,
    store: &impl StoreTransform,
    offset: usize,
) {
    for (r, sums) in sums.into_iter().enumerate() {
        for (vector, sum) in sums.into_iter().enumerate() {
            let column = column + vector * 4;
            if column < columns {
                let start = (row + r) * columns + column;
                let count = (columns - column).min(4);
                let sum = store.vector(as_simd(sum), offset + start);
                output[start..start + count].copy_from_slice(&sum.as_array()[..count]);
            }
        }
    }
}

#[inline(always)]
fn as_simd(value: float32x4_t) -> Simd<f32, 4> {
    // SAFETY: both types contain four native F32 lanes without padding.
    unsafe { std::mem::transmute(value) }
}

#[inline(always)]
unsafe fn partial_load(values: &[f32], column: usize) -> float32x4_t {
    // Each branch reads exactly its valid lanes. Explicit native loads avoid
    // a variable-length memcpy call inside the partial MatMul reduction.
    // SAFETY: count proves the complete load's bounds. No pointer arithmetic
    // occurs for an absent vector, even when column is beyond the slice.
    unsafe {
        match values.len().saturating_sub(column).min(4) {
            0 => vdupq_n_f32(0.),
            1 => vsetq_lane_f32::<0>(values[column], vdupq_n_f32(0.)),
            2 => vcombine_f32(vld1_f32(values.as_ptr().add(column)), vdup_n_f32(0.)),
            3 => vsetq_lane_f32::<2>(
                values[column + 2],
                vcombine_f32(vld1_f32(values.as_ptr().add(column)), vdup_n_f32(0.)),
            ),
            4 => vld1q_f32(values.as_ptr().add(column)),
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
// Explicit row/vector indices preserve the independently ordered FMA chains.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn tile<
    const ROWS: usize,
    const PACKED: bool,
    const PARTIAL: bool,
    const BIAS_AFTER: bool,
>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    first_row: usize,
    column: usize,
    inner: usize,
    columns: usize,
) -> [[float32x4_t; 4]; ROWS] {
    // SAFETY: callers validate whole matrices and provide complete ROWS. A
    // complete tile has 16 valid columns; partial row-major loads are padded
    // locally, and PackedMatrix already pads its final 16-column panel.
    unsafe {
        let mut sums = [[vdupq_n_f32(0.); 4]; ROWS];
        if !BIAS_AFTER && !bias.is_empty() {
            for vector in 0..4 {
                let value = if PARTIAL {
                    partial_load(bias, column + vector * 4)
                } else {
                    vld1q_f32(bias.as_ptr().add(column + vector * 4))
                };
                for sum in &mut sums {
                    sum[vector] = value;
                }
            }
        }
        let weight_stride = if PACKED { 16 } else { columns };
        let weight_start = if PACKED {
            column / 16 * inner * 16
        } else {
            column
        };
        let input_start = input.as_ptr().add(first_row * inner);
        let weight_start = weights.as_ptr().add(weight_start);
        let mut k = 0;
        while k + 4 <= inner {
            let inputs: [float32x4_t; ROWS] =
                std::array::from_fn(|row| vld1q_f32(input_start.add(row * inner + k)));
            // The lane operand must be an instruction constant. Each expansion
            // visits the next K without introducing partial-sum reductions.
            macro_rules! accumulate_lane {
                ($lane:literal) => {
                    let weight_row = weight_start.add((k + $lane) * weight_stride);
                    for vector in 0..4 {
                        let weight = if PACKED || !PARTIAL {
                            vld1q_f32(weight_row.add(vector * 4))
                        } else {
                            let row = &weights[(k + $lane) * columns..(k + $lane + 1) * columns];
                            partial_load(row, column + vector * 4)
                        };
                        for row in 0..ROWS {
                            sums[row][vector] =
                                vfmaq_laneq_f32::<$lane>(sums[row][vector], weight, inputs[row]);
                        }
                    }
                };
            }
            accumulate_lane!(0);
            accumulate_lane!(1);
            accumulate_lane!(2);
            accumulate_lane!(3);
            k += 4;
        }
        while k < inner {
            let weight_row = weight_start.add(k * weight_stride);
            for vector in 0..4 {
                let weight = if PACKED || !PARTIAL {
                    vld1q_f32(weight_row.add(vector * 4))
                } else {
                    let row = &weights[k * columns..(k + 1) * columns];
                    partial_load(row, column + vector * 4)
                };
                for row in 0..ROWS {
                    sums[row][vector] =
                        vfmaq_n_f32(sums[row][vector], weight, *input_start.add(row * inner + k));
                }
            }
            k += 1;
        }
        if BIAS_AFTER && !bias.is_empty() {
            for vector in 0..4 {
                let value = vld1q_f32(bias.as_ptr().add(column + vector * 4));
                for sum in &mut sums {
                    sum[vector] = vaddq_f32(sum[vector], value);
                }
            }
        }
        sums
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn compact_argmax(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    indices: &mut [usize],
    scores: &mut [f32],
    inner: usize,
    columns: usize,
) -> Result<(), &'static str> {
    let rows = indices.len();
    if inner == 0
        || columns == 0
        || rows.checked_mul(inner) != Some(input.len())
        || inner.checked_mul(columns) != Some(weights.len())
        || scores.len() != rows
        || (!bias.is_empty() && bias.len() != columns)
    {
        return Err("compact projection buffer shape mismatch");
    }
    let Some(packed) = packed else {
        // The reference's unpacked path retains separate multiply/adds and
        // bias-first accumulation, even when a native policy was requested.
        return super::scalar_argmax(input, weights, bias, indices, scores, inner, columns);
    };
    if !packed.matches(inner, columns) {
        return Err("compact projection panel shape mismatch");
    }
    if columns > u32::MAX as usize {
        return super::scalar_argmax(input, weights, bias, indices, scores, inner, columns);
    }
    let full_columns = columns / 16 * 16;
    let panel_tiles = (WEIGHT_PANEL_BYTES / inner.saturating_mul(16 * size_of::<f32>())).max(1);
    // SAFETY: all matrix lengths and packed panel dimensions were checked.
    // Every native tile has exactly 16 valid columns and 1 or 4 complete rows.
    unsafe {
        for first_row in (0..rows).step_by(ROW_GROUP) {
            let count = (rows - first_row).min(ROW_GROUP);
            let mut winners = [LaneWinners::<4>::new(); ROW_GROUP];
            for first_tile in (0..columns / 16).step_by(panel_tiles) {
                let last_tile = (first_tile + panel_tiles).min(columns / 16);
                let mut row = 0;
                while row + 4 <= count {
                    compact_rows::<4>(
                        input,
                        packed.data(),
                        bias,
                        &mut winners,
                        first_row,
                        row,
                        inner,
                        columns,
                        first_tile,
                        last_tile,
                    )?;
                    row += 4;
                }
                while row < count {
                    compact_rows::<1>(
                        input,
                        packed.data(),
                        bias,
                        &mut winners,
                        first_row,
                        row,
                        inner,
                        columns,
                        first_tile,
                        last_tile,
                    )?;
                    row += 1;
                }
            }
            for (row, winner) in winners[..count].iter().enumerate() {
                let (mut index, mut score) = winner.finish();
                super::compact_scalar_tail::<true>(
                    input,
                    packed.data(),
                    bias,
                    first_row + row,
                    inner,
                    columns,
                    full_columns,
                    &mut score,
                    &mut index,
                )?;
                indices[first_row + row] = index;
                scores[first_row + row] = score;
            }
        }
    }
    Ok(())
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn compact_rows<const ROWS: usize>(
    input: &[f32],
    packed: &[f32],
    bias: &[f32],
    winners: &mut [LaneWinners<4>],
    first_row: usize,
    row: usize,
    inner: usize,
    columns: usize,
    first_tile: usize,
    last_tile: usize,
) -> Result<(), &'static str> {
    // SAFETY: the entry checked matrix extents; the caller bounds row groups
    // and tiles to complete 16-column panels inside the real vocabulary.
    unsafe {
        for block in first_tile..last_tile {
            let column = block * 16;
            let sums = tile::<ROWS, true, false, true>(
                input,
                packed,
                bias,
                first_row + row,
                column,
                inner,
                columns,
            );
            for (r, sums) in sums.into_iter().enumerate() {
                for (vector, sum) in sums.into_iter().enumerate() {
                    winners[row + r].update(as_simd(sum), column + vector * 4, 4)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
