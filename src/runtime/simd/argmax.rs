//! Packed CTC projection reuses an L2-sized weight panel across bounded row
//! groups. Winners stay in SIMD lanes until the final stable reduction.
use super::{
    MATRIX_BLOCK_COLUMNS, Simd, compact_projection_tile, compact_scalar_tail,
    wide_projection_eligible,
};
use std::simd::{Mask, Select, cmp::SimdPartialOrd, num::SimdFloat};

const ROW_GROUP: usize = 128;
const WEIGHT_PANEL_BYTES: usize = 256 * 1024;

#[derive(Clone, Copy)]
pub(super) struct LaneWinners<const LANES: usize> {
    values: Simd<f32, LANES>,
    columns: Simd<u32, LANES>,
}

impl<const LANES: usize> LaneWinners<LANES> {
    pub(super) fn new() -> Self {
        Self {
            values: Simd::splat(f32::NEG_INFINITY),
            columns: Simd::splat(0),
        }
    }

    pub(super) fn from_winner(index: usize, score: f32) -> Self {
        Self {
            values: Simd::splat(score),
            columns: Simd::splat(index as u32),
        }
    }

    #[inline(always)]
    pub(super) fn update(
        &mut self,
        values: Simd<f32, LANES>,
        column: usize,
        valid: usize,
    ) -> Result<(), &'static str> {
        let active = Mask::<i32, LANES>::from_array(std::array::from_fn(|lane| lane < valid));
        if (active & !values.is_finite()).any() {
            return Err("nonfinite compact recognition logits");
        }
        let replace = active & values.simd_gt(self.values);
        let candidates = Simd::from_array(std::array::from_fn(|lane| (column + lane) as u32));
        self.values = replace.select(values, self.values);
        self.columns = replace.select(candidates, self.columns);
        Ok(())
    }

    #[inline(always)]
    pub(super) fn finish(self) -> (usize, f32) {
        let values = self.values.to_array();
        let columns = self.columns.to_array();
        let mut maximum = values[0];
        let mut best = columns[0];
        for lane in 1..LANES {
            let value = values[lane];
            let column = columns[lane];
            if value > maximum || value == maximum && column < best {
                maximum = value;
                best = column;
            }
        }
        (best as usize, maximum)
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn blocked<const LANES: usize, const VECTORS: usize, const FUSED: bool>(
    input: &[f32],
    packed: &[f32],
    bias: &[f32],
    indices: &mut [usize],
    scores: &mut [f32],
    inner: usize,
    columns: usize,
) -> Result<(), &'static str> {
    // All packed ISA paths use complete 16-column tiles and the same
    // separate bias-first scalar remainder as the pinned reference.
    let tiles = columns / MATRIX_BLOCK_COLUMNS;
    let panel_tiles = (WEIGHT_PANEL_BYTES
        / inner
            .max(1)
            .saturating_mul(MATRIX_BLOCK_COLUMNS * size_of::<f32>()))
    .max(1);
    let wide = LANES == 16 && wide_projection_eligible(indices.len(), inner, columns, true);
    for first_row in (0..indices.len()).step_by(ROW_GROUP) {
        let count = (indices.len() - first_row).min(ROW_GROUP);
        // Fixed stack state bounds memory independently of request dimensions.
        // Each row retains the exact F32 winner, including signed-zero ties.
        let mut winners = [LaneWinners::<LANES>::new(); ROW_GROUP];
        for first_tile in (0..tiles).step_by(panel_tiles) {
            let last_tile = (first_tile + panel_tiles).min(tiles);
            let mut row = 0;
            if wide {
                while row + 16 <= count {
                    rows::<LANES, VECTORS, FUSED, 16>(
                        input,
                        packed,
                        bias,
                        &mut winners,
                        first_row,
                        row,
                        inner,
                        columns,
                        first_tile,
                        last_tile,
                    )?;
                    row += 16;
                }
                if row + 8 <= count {
                    rows::<LANES, VECTORS, FUSED, 8>(
                        input,
                        packed,
                        bias,
                        &mut winners,
                        first_row,
                        row,
                        inner,
                        columns,
                        first_tile,
                        last_tile,
                    )?;
                    row += 8;
                }
            }
            // AVX uses four rows, leaving registers for weights and broadcasts.
            // The reference's wider row tiles belong to the AVX-512 path.
            while row + 4 <= count {
                rows::<LANES, VECTORS, FUSED, 4>(
                    input,
                    packed,
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
                rows::<LANES, VECTORS, FUSED, 1>(
                    input,
                    packed,
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
            compact_scalar_tail::<true>(
                input,
                packed,
                bias,
                first_row + row,
                inner,
                columns,
                tiles * MATRIX_BLOCK_COLUMNS,
                &mut score,
                &mut index,
            )?;
            indices[first_row + row] = index;
            scores[first_row + row] = score;
        }
    }
    Ok(())
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn rows<const LANES: usize, const VECTORS: usize, const FUSED: bool, const ROWS: usize>(
    input: &[f32],
    packed: &[f32],
    bias: &[f32],
    winners: &mut [LaneWinners<LANES>],
    first_row: usize,
    row: usize,
    inner: usize,
    columns: usize,
    first_tile: usize,
    last_tile: usize,
) -> Result<(), &'static str> {
    for tile in first_tile..last_tile {
        let sums = compact_projection_tile::<LANES, VECTORS, FUSED, ROWS, true>(
            input,
            packed,
            bias,
            first_row + row,
            tile,
            inner,
            columns,
        );
        for (offset, values) in sums.into_iter().enumerate() {
            for (vector, value) in values.into_iter().enumerate() {
                let column = tile * MATRIX_BLOCK_COLUMNS + vector * LANES;
                let valid = columns.saturating_sub(column).min(LANES);
                winners[row + offset].update(value, column, valid)?;
            }
        }
    }
    Ok(())
}

/// Scalar transcription of the pinned reference's backend-specific ordering,
/// intentionally independent of the production vector/packed implementations.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn reference_logits(
    backend: super::Backend,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    packed: bool,
) -> Vec<f32> {
    let packed_vectors = backend != super::Backend::Scalar && packed;
    let mut output = vec![0.; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let packed_tile =
                packed_vectors && column < columns / MATRIX_BLOCK_COLUMNS * MATRIX_BLOCK_COLUMNS;
            let mut value = if packed_tile || bias.is_empty() {
                0.
            } else {
                bias[column]
            };
            for k in 0..inner {
                let input = input[row * inner + k];
                let weight = weights[k * columns + column];
                value = if packed_tile && backend.uses_fma() {
                    input.mul_add(weight, value)
                } else {
                    value + input * weight
                };
            }
            if packed_tile && !bias.is_empty() {
                value += bias[column];
            }
            output[row * columns + column] = value;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::super::{Backend, PackedMatrix, compact_argmax};

    fn backends() -> Vec<Backend> {
        let mut result = vec![Backend::Portable4];
        if super::super::avx_available() {
            result.push(Backend::Avx);
        }
        if super::super::avx_fma_available() {
            result.push(Backend::AvxFma);
        }
        if super::super::avx2_fma_available() {
            result.push(Backend::Avx2Fma);
        }
        if super::super::avx512_available() {
            result.push(Backend::Avx512);
        }
        result
    }

    #[test]
    fn native_compact_adds_bias_after_the_complete_dot_product() {
        let (inner, columns) = (65, 1041);
        let mut weights = vec![0.; inner * columns];
        let mut bias = vec![-2.; columns];
        // The pinned native kernels produce 1 at columns 0 and 15. A
        // bias-first accumulator loses that 1 before cancellation and picks
        // column 1 instead. Column 1040 is a scalar tail and stays bias-first.
        for column in [0, 15, 1040] {
            weights[column] = 1e8;
            weights[columns + column] = -1e8;
            bias[column] = 1.;
        }
        bias[1] = 0.5;
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        for backend in backends() {
            if backend == Backend::Portable4 {
                continue;
            }
            for rows in [1, 4, 8, 17, 129] {
                let mut indices = vec![usize::MAX; rows];
                let mut scores = vec![f32::NAN; rows];
                compact_argmax(
                    backend,
                    &vec![1.; rows * inner],
                    &weights,
                    &bias,
                    Some(&packed),
                    &mut indices,
                    &mut scores,
                    inner,
                    columns,
                )
                .unwrap();
                assert_eq!(indices, vec![0; rows], "{backend:?} rows={rows}");
                assert_eq!(scores, vec![1.; rows], "{backend:?} rows={rows}");
            }
        }
    }

    #[test]
    fn native_compact_scalar_columns_do_not_fuse_multiply_and_add() {
        let (rows, inner, columns) = (8, 65, 17);
        let mut input = vec![0.; rows * inner];
        for row in 0..rows {
            input[row * inner] = 1. + f32::EPSILON;
        }
        let mut weights = vec![0.; inner * columns];
        let mut bias = vec![-2.; columns];
        bias[0] = 0.;
        weights[16] = 1. + f32::EPSILON;
        bias[16] = -(1. + 2. * f32::EPSILON);
        // Separate multiplication rounds to the magnitude of the bias. A
        // fused tail instead leaves EPSILON^2 and incorrectly wins column16.
        assert_eq!(input[0] * weights[16] + bias[16], 0.);
        assert!(input[0].mul_add(weights[16], bias[16]) > 0.);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        for backend in backends() {
            for packed in [None, Some(&packed)] {
                let mut indices = [usize::MAX; 8];
                let mut scores = [f32::NAN; 8];
                compact_argmax(
                    backend,
                    &input,
                    &weights,
                    &bias,
                    packed,
                    &mut indices,
                    &mut scores,
                    inner,
                    columns,
                )
                .unwrap();
                assert_eq!(indices, [0; 8], "{backend:?} packed={}", packed.is_some());
                assert_eq!(scores.map(f32::to_bits), [0f32.to_bits(); 8]);
            }
        }
    }

    #[test]
    fn scalar_and_unpacked_compact_preserve_bias_first_contract() {
        let (rows, inner, columns) = (5, 2, 32);
        let mut weights = vec![0.; inner * columns];
        let mut bias = vec![-2.; columns];
        weights[0] = 1e8;
        weights[columns] = -1e8;
        bias[0] = 1.;
        bias[1] = 0.5;
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        for (backend, packed) in [
            (Backend::Scalar, None),
            (Backend::Scalar, Some(&packed)),
            (Backend::Portable4, None),
        ] {
            let mut indices = [usize::MAX; 5];
            let mut scores = [f32::NAN; 5];
            compact_argmax(
                backend,
                &[1.; 10],
                &weights,
                &bias,
                packed,
                &mut indices,
                &mut scores,
                inner,
                columns,
            )
            .unwrap();
            assert_eq!(indices, [1; 5]);
            assert_eq!(scores, [0.5; 5]);
        }
        for backend in backends() {
            let mut indices = [usize::MAX; 5];
            let mut scores = [f32::NAN; 5];
            compact_argmax(
                backend,
                &[1.; 10],
                &weights,
                &bias,
                None,
                &mut indices,
                &mut scores,
                inner,
                columns,
            )
            .unwrap();
            assert_eq!(indices, [1; 5]);
            assert_eq!(scores, [0.5; 5]);
            // This public low-level operation keeps bias-first even for
            // packed native matrices; ArgMax must not redefine it.
            let mut dense = vec![f32::NAN; rows * columns];
            super::super::matmul_bias(
                backend,
                &[1.; 10],
                &weights,
                Some(&packed),
                &bias,
                &mut dense,
                rows,
                inner,
                columns,
            );
            for row in 0..rows {
                assert_eq!(dense[row * columns], 0.);
                assert_eq!(dense[row * columns + 1], 0.5);
            }
        }
    }

    #[test]
    fn packed_winners_match_reference_order_across_cache_panels_and_bounded_row_groups() {
        let (rows, inner, columns) = (129, 65, 1041);
        let input: Vec<_> = (0..rows * inner)
            .map(|i| (i % 29) as f32 * 0.031 - 0.3)
            .collect();
        let weights: Vec<_> = (0..inner * columns)
            .map(|i| (i % 47) as f32 * 0.021 - 0.2)
            .collect();
        let bias: Vec<_> = (0..columns).map(|i| (i % 7) as f32 * 0.02).collect();
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        for backend in backends() {
            let dense = super::reference_logits(
                backend, &input, &weights, &bias, rows, inner, columns, true,
            );
            let mut indices = vec![usize::MAX; rows];
            let mut scores = vec![f32::NAN; rows];
            compact_argmax(
                backend,
                &input,
                &weights,
                &bias,
                Some(&packed),
                &mut indices,
                &mut scores,
                inner,
                columns,
            )
            .unwrap();
            for row in 0..rows {
                let mut best = 0;
                for column in 1..columns {
                    if dense[row * columns + column] > dense[row * columns + best] {
                        best = column;
                    }
                }
                assert_eq!(indices[row], best, "{backend:?} row {row}");
                assert_eq!(scores[row].to_bits(), dense[row * columns + best].to_bits());
            }
        }
    }

    #[test]
    fn packed_winners_preserve_first_signed_zero_ties_and_reject_hidden_nonfinite() {
        let (inner, columns) = (65, 1041);
        let mut bias = vec![-2.; columns];
        bias[4] = -0.;
        bias[1024] = 0.;
        bias[1040] = 0.;
        let weights = vec![0.; inner * columns];
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        for backend in backends() {
            let mut indices = [usize::MAX; 5];
            let mut scores = [f32::NAN; 5];
            compact_argmax(
                backend,
                &vec![-0.; 5 * inner],
                &weights,
                &bias,
                Some(&packed),
                &mut indices,
                &mut scores,
                inner,
                columns,
            )
            .unwrap();
            assert_eq!(indices, [4; 5]);
            // Every complete packed tile starts at positive zero before
            // the final bias; the winning class 4 is inside that tile.
            assert_eq!(scores.map(f32::to_bits), [0f32.to_bits(); 5]);
            let mut invalid = bias.clone();
            invalid[1040] = f32::NEG_INFINITY;
            assert!(
                compact_argmax(
                    backend,
                    &vec![0.; 5 * inner],
                    &weights,
                    &invalid,
                    Some(&packed),
                    &mut indices,
                    &mut scores,
                    inner,
                    columns
                )
                .is_err()
            );
            invalid[1040] = f32::NAN;
            assert!(
                compact_argmax(
                    backend,
                    &vec![0.; 5 * inner],
                    &weights,
                    &invalid,
                    Some(&packed),
                    &mut indices,
                    &mut scores,
                    inner,
                    columns
                )
                .is_err()
            );
        }
    }
}

#[cfg(test)]
mod portable_tests;
