//! Deterministic column sharding of packed CTC units. Workers own contiguous
//! vocabulary panels and compact row winners; no dense vocabulary output exists.
use super::{Backend, PackedMatrix, avx512_available};
#[cfg(any(target_arch = "x86_64", test))]
use super::{compact_projection_tile, compact_scalar_tail, update_winner};
use crate::cpu::CpuContext;

#[cfg(any(target_arch = "x86_64", test))]
#[derive(Clone, Copy)]
struct Winner {
    score: f32,
    index: usize,
}
#[cfg(any(target_arch = "x86_64", test))]
impl Default for Winner {
    fn default() -> Self {
        Self {
            score: f32::NEG_INFINITY,
            index: 0,
        }
    }
}

#[derive(Default)]
pub(crate) struct ArgMaxWorkspace {
    #[cfg(any(target_arch = "x86_64", test))]
    partials: Vec<Winner>,
}
impl ArgMaxWorkspace {
    pub(crate) fn retained_bytes(&self) -> usize {
        #[cfg(any(target_arch = "x86_64", test))]
        {
            self.partials.capacity() * size_of::<Winner>()
        }
        #[cfg(not(any(target_arch = "x86_64", test)))]
        {
            0
        }
    }
    #[cfg(any(target_arch = "x86_64", test))]
    fn reserve(&mut self, rows: usize, workers: usize) -> Result<(), &'static str> {
        let count = rows
            .checked_mul(workers)
            .ok_or("CTC column state size overflow")?;
        self.partials
            .try_reserve_exact(count.saturating_sub(self.partials.len()))
            .map_err(|_| "CTC column state allocation failed")?;
        self.partials.resize(count, Winner::default());
        Ok(())
    }
}

/// A unit is one independent recognition sample's time axis. Per-unit
/// divisibility, rather than just total rows, preserves standalone eligibility.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_compact_columns(
    backend: Backend,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    indices: &mut [usize],
    scores: &mut [f32],
    unit_rows: usize,
    inner: usize,
    columns: usize,
    cpu: &CpuContext,
    threads: usize,
    workspace: &mut ArgMaxWorkspace,
) -> Result<bool, &'static str> {
    if backend != Backend::Avx512
        || threads <= 1
        || unit_rows < 4
        || !unit_rows.is_multiple_of(4)
        || inner < 64
        || columns < 1024
        || indices.is_empty()
        || !indices.len().is_multiple_of(unit_rows)
    {
        return Ok(false);
    }
    if !avx512_available() {
        return Err("AVX-512 CTC backend is unavailable on this CPU");
    }
    let Some(packed) = packed else {
        return Ok(false);
    };
    let rows = indices.len();
    if !packed.matches(inner, columns)
        || rows.checked_mul(inner) != Some(input.len())
        || inner.checked_mul(columns) != Some(weights.len())
        || scores.len() != rows
        || (!bias.is_empty() && bias.len() != columns)
    {
        return Err("CTC column projection buffer shape mismatch");
    }
    let work = rows as u128 * inner as u128 * columns as u128;
    if work < 2_000_000 {
        return Ok(false);
    }
    let workers = threads.min(cpu.capacity()).min(columns / 16).min(rows);
    if workers <= 1 {
        return Ok(false);
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: runtime detection above guards the target-feature entry.
    unsafe {
        workspace.reserve(rows, workers)?;
        execute_avx512(
            input,
            bias,
            packed,
            indices,
            scores,
            inner,
            columns,
            cpu,
            workers,
            &mut workspace.partials,
        )?;
        Ok(true)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = workspace;
        Err("AVX-512 CTC backend requires x86-64")
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn execute_avx512(
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    indices: &mut [usize],
    scores: &mut [f32],
    inner: usize,
    columns: usize,
    cpu: &CpuContext,
    workers: usize,
    state: &mut [Winner],
) -> Result<(), &'static str> {
    execute::<16, 1, true>(
        input, bias, packed, indices, scores, inner, columns, cpu, workers, state,
    )
}

#[cfg(any(target_arch = "x86_64", test))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn execute<const LANES: usize, const VECTORS: usize, const FMA: bool>(
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    indices: &mut [usize],
    scores: &mut [f32],
    inner: usize,
    columns: usize,
    cpu: &CpuContext,
    workers: usize,
    state: &mut [Winner],
) -> Result<(), &'static str> {
    let rows = indices.len();
    // Every packed tile is complete; the last owning shard handles its
    // separate scalar remainder, including an all-tail vocabulary.
    let tiles = columns / 16;
    let workers = workers.min(tiles.max(1));
    let worker = |(worker_index, winners): (usize, &mut [Winner])| {
        let begin = tiles * worker_index / workers;
        let end = tiles * (worker_index + 1) / workers;
        #[cfg(target_arch = "x86_64")]
        if LANES == 16 && FMA {
            // SAFETY: only the detected AVX-512 production entry instantiates
            // this combination. Each pool worker enters its own ISA function.
            return unsafe {
                shard_avx512(input, bias, packed, winners, inner, columns, begin, end)
            };
        }
        shard::<LANES, VECTORS, FMA>(input, bias, packed, winners, inner, columns, begin, end)
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        cpu.install(|| {
            state[..rows * workers]
                .par_chunks_mut(rows)
                .enumerate()
                .try_for_each(worker)
        })?;
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = cpu;
        for item in state[..rows * workers].chunks_mut(rows).enumerate() {
            worker(item)?;
        }
    }
    // Merge in increasing vocabulary order. Strict greater-than retains the
    // earliest class and its exact score bits, including equal signed zeros.
    for row in 0..rows {
        let mut winner = state[row];
        for shard in 1..workers {
            let next = state[shard * rows + row];
            if next.score > winner.score {
                winner = next;
            }
        }
        indices[row] = winner.index;
        scores[row] = winner.score;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn shard_avx512(
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    winners: &mut [Winner],
    inner: usize,
    columns: usize,
    begin: usize,
    end: usize,
) -> Result<(), &'static str> {
    shard::<16, 1, true>(input, bias, packed, winners, inner, columns, begin, end)
}

#[cfg(any(target_arch = "x86_64", test))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn shard<const LANES: usize, const VECTORS: usize, const FMA: bool>(
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    winners: &mut [Winner],
    inner: usize,
    columns: usize,
    begin: usize,
    end: usize,
) -> Result<(), &'static str> {
    winners.fill(Winner::default());
    let rows = winners.len();
    let block_tiles = (256 * 1024 / (inner * 16 * size_of::<f32>())).max(1);
    for first in (begin..end).step_by(block_tiles) {
        let last = (first + block_tiles).min(end);
        let mut row = 0;
        while row + 16 <= rows {
            row_block::<LANES, VECTORS, FMA, 16>(
                input, bias, packed, winners, row, inner, columns, first, last,
            )?;
            row += 16;
        }
        if row + 8 <= rows {
            row_block::<LANES, VECTORS, FMA, 8>(
                input, bias, packed, winners, row, inner, columns, first, last,
            )?;
            row += 8;
        }
        if row < rows {
            row_block::<LANES, VECTORS, FMA, 4>(
                input, bias, packed, winners, row, inner, columns, first, last,
            )?;
        }
    }
    if end == columns / 16 {
        // Only the last vocabulary shard owns the remaining scalar columns.
        // Complete every shard, including finite validation of its tail,
        // before the ordered merge can publish any caller output.
        for (row, winner) in winners.iter_mut().enumerate() {
            compact_scalar_tail::<true>(
                input,
                packed.data(),
                bias,
                row,
                inner,
                columns,
                columns / 16 * 16,
                &mut winner.score,
                &mut winner.index,
            )?;
        }
    }
    Ok(())
}

#[cfg(any(target_arch = "x86_64", test))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn row_block<const LANES: usize, const VECTORS: usize, const FMA: bool, const ROWS: usize>(
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    winners: &mut [Winner],
    row: usize,
    inner: usize,
    columns: usize,
    begin: usize,
    end: usize,
) -> Result<(), &'static str> {
    if columns <= u32::MAX as usize {
        let mut lanes: [super::argmax::LaneWinners<LANES>; ROWS] = std::array::from_fn(|offset| {
            let winner = winners[row + offset];
            super::argmax::LaneWinners::from_winner(winner.index, winner.score)
        });
        for block in begin..end {
            let sums = compact_projection_tile::<LANES, VECTORS, FMA, ROWS, true>(
                input,
                packed.data(),
                bias,
                row,
                block,
                inner,
                columns,
            );
            for (offset, vectors) in sums.into_iter().enumerate() {
                for (vector, values) in vectors.into_iter().enumerate() {
                    let column = block * 16 + vector * LANES;
                    lanes[offset].update(
                        values,
                        column,
                        columns.saturating_sub(column).min(LANES),
                    )?;
                }
            }
        }
        for (offset, state) in lanes.into_iter().enumerate() {
            let (index, score) = state.finish();
            winners[row + offset] = Winner { score, index };
        }
        return Ok(());
    }
    for block in begin..end {
        let sums = compact_projection_tile::<LANES, VECTORS, FMA, ROWS, true>(
            input,
            packed.data(),
            bias,
            row,
            block,
            inner,
            columns,
        );
        for (offset, vectors) in sums.into_iter().enumerate() {
            let winner = &mut winners[row + offset];
            for (vector, values) in vectors.into_iter().enumerate() {
                let column = block * 16 + vector * LANES;
                for (lane, value) in values
                    .to_array()
                    .into_iter()
                    .enumerate()
                    .take(columns.saturating_sub(column).min(LANES))
                {
                    update_winner(value, column + lane, &mut winner.score, &mut winner.index)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::compact_argmax;
    use super::*;

    #[test]
    fn native_column_shards_keep_reference_bias_and_scalar_tail_order() {
        let (rows, inner, columns) = (28, 65, 1041);
        #[cfg(feature = "parallel")]
        let cpu = CpuContext::new(4).unwrap();
        #[cfg(not(feature = "parallel"))]
        let cpu = CpuContext::serial();
        for tail_fma_probe in [false, true] {
            let mut input = vec![1.; rows * inner];
            let mut weights = vec![0.; inner * columns];
            let mut bias = vec![-2.; columns];
            let (expected_index, expected_score) = if tail_fma_probe {
                input.fill(0.);
                for row in 0..rows {
                    input[row * inner] = 1. + f32::EPSILON;
                }
                weights[1040] = 1. + f32::EPSILON;
                bias[1040] = -(1. + 2. * f32::EPSILON);
                bias[0] = 0.;
                (0, 0f32)
            } else {
                for column in [0, 500, 1040] {
                    weights[column] = 1e8;
                    weights[columns + column] = -1e8;
                    bias[column] = 1.;
                }
                bias[1] = 0.5;
                (0, 1f32)
            };
            let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
            let mut workspace = ArgMaxWorkspace::default();
            for workers in [1, 2, 4, 2] {
                workspace.reserve(rows, workers).unwrap();
                let mut indices = vec![usize::MAX; rows];
                let mut scores = vec![f32::NAN; rows];
                // Exercise native ordering on every host with safe portable
                // SIMD. The actual 16-lane FMA entry stays hardware-gated in
                // native_entry_checks_hardware_and_independent_unit_eligibility.
                execute::<8, 2, false>(
                    &input,
                    &bias,
                    &packed,
                    &mut indices,
                    &mut scores,
                    inner,
                    columns,
                    &cpu,
                    workers,
                    &mut workspace.partials,
                )
                .unwrap();
                assert_eq!(indices, vec![expected_index; rows]);
                assert_eq!(
                    scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    vec![expected_score.to_bits(); rows]
                );
                execute::<8, 2, true>(
                    &input,
                    &bias,
                    &packed,
                    &mut indices,
                    &mut scores,
                    inner,
                    columns,
                    &cpu,
                    workers,
                    &mut workspace.partials,
                )
                .unwrap();
                assert_eq!(indices, vec![expected_index; rows]);
                assert_eq!(
                    scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    vec![expected_score.to_bits(); rows]
                );
            }
        }
    }

    #[test]
    fn shards_preserve_bias_ties_tails_and_retained_capacity() {
        let (rows, inner, columns) = (28, 65, 1031);
        let input: Vec<f32> = (0..rows * inner)
            .map(|i| (i % 13) as f32 * 0.03 - 0.1)
            .collect();
        let mut weights: Vec<f32> = (0..inner * columns)
            .map(|i| (i % 29) as f32 * 0.01 - 0.07)
            .collect();
        let mut bias = vec![0.; columns];
        // Equal winners in different shards; the first must survive every merge.
        for k in 0..inner {
            for column in [17, 700, 1030] {
                weights[k * columns + column] = 0.;
            }
        }
        for column in [17, 700, 1030] {
            bias[column] = 100.;
        }
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let mut expected_indices = vec![0; rows];
        let mut expected_scores = vec![0.; rows];
        compact_argmax(
            Backend::Portable4,
            &input,
            &weights,
            &bias,
            Some(&packed),
            &mut expected_indices,
            &mut expected_scores,
            inner,
            columns,
        )
        .unwrap();
        #[cfg(feature = "parallel")]
        let cpu = CpuContext::new(4).unwrap();
        #[cfg(not(feature = "parallel"))]
        let cpu = CpuContext::serial();
        let mut workspace = ArgMaxWorkspace::default();
        let mut indices = vec![0; rows];
        let mut scores = vec![0.; rows];
        for workers in [2, 4, 8, 2] {
            workspace.reserve(rows, workers).unwrap();
            let capacity = workspace.retained_bytes();
            execute::<4, 4, false>(
                &input,
                &bias,
                &packed,
                &mut indices,
                &mut scores,
                inner,
                columns,
                &cpu,
                workers,
                &mut workspace.partials,
            )
            .unwrap();
            assert_eq!(indices, expected_indices);
            assert_eq!(
                scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected_scores
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            workspace.reserve(rows, workers).unwrap();
            assert_eq!(workspace.retained_bytes(), capacity);
        }
        // Exercise the full dot products as winners as well as the deliberate
        // ties above; a constant winning bias alone could hide an input bug.
        bias.fill(0.);
        compact_argmax(
            Backend::Portable4,
            &input,
            &weights,
            &bias,
            Some(&packed),
            &mut expected_indices,
            &mut expected_scores,
            inner,
            columns,
        )
        .unwrap();
        workspace.reserve(rows, 4).unwrap();
        execute::<4, 4, false>(
            &input,
            &bias,
            &packed,
            &mut indices,
            &mut scores,
            inner,
            columns,
            &cpu,
            4,
            &mut workspace.partials,
        )
        .unwrap();
        assert_eq!(indices, expected_indices);
        assert_eq!(
            scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected_scores
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn failed_shard_leaves_outputs_intact_and_recovers() {
        let (rows, inner, columns) = (4, 1, 33);
        let input = vec![1.; rows];
        let weights = vec![1.; columns];
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let mut bias = vec![0.; columns];
        bias[32] = f32::NAN;
        let mut workspace = ArgMaxWorkspace::default();
        workspace.reserve(rows, 2).unwrap();
        let mut indices = vec![91; rows];
        let mut scores = vec![42.; rows];
        assert!(
            execute::<4, 4, false>(
                &input,
                &bias,
                &packed,
                &mut indices,
                &mut scores,
                inner,
                columns,
                &CpuContext::serial(),
                2,
                &mut workspace.partials
            )
            .is_err()
        );
        assert_eq!(indices, vec![91; rows]);
        assert_eq!(scores, vec![42.; rows]);
        bias[32] = 2.;
        execute::<4, 4, false>(
            &input,
            &bias,
            &packed,
            &mut indices,
            &mut scores,
            inner,
            columns,
            &CpuContext::serial(),
            2,
            &mut workspace.partials,
        )
        .unwrap();
        assert_eq!(indices, vec![32; rows]);
        assert_eq!(scores, vec![3.; rows]);
    }

    #[test]
    fn native_entry_checks_hardware_and_independent_unit_eligibility() {
        let (rows, inner, columns) = (16, 65, 2049);
        let input = vec![0.125; rows * inner];
        let weights: Vec<f32> = (0..inner * columns)
            .map(|i| (i % 23) as f32 * 0.03125)
            .collect();
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let mut indices = vec![0; rows];
        let mut scores = vec![0.; rows];
        let mut workspace = ArgMaxWorkspace::default();
        let cpu = CpuContext::serial();
        assert!(
            !try_compact_columns(
                Backend::Avx512,
                &input,
                &weights,
                &[],
                Some(&packed),
                &mut indices,
                &mut scores,
                2,
                inner,
                columns,
                &cpu,
                4,
                &mut workspace
            )
            .unwrap()
        );
        if !avx512_available() {
            assert!(
                try_compact_columns(
                    Backend::Avx512,
                    &input,
                    &weights,
                    &[],
                    Some(&packed),
                    &mut indices,
                    &mut scores,
                    4,
                    inner,
                    columns,
                    &cpu,
                    4,
                    &mut workspace
                )
                .is_err()
            );
        }
        #[cfg(feature = "parallel")]
        if avx512_available() {
            let cpu = CpuContext::new(4).unwrap();
            let mut expected_indices = vec![0; rows];
            let mut expected_scores = vec![0.; rows];
            compact_argmax(
                Backend::Avx512,
                &input,
                &weights,
                &[],
                Some(&packed),
                &mut expected_indices,
                &mut expected_scores,
                inner,
                columns,
            )
            .unwrap();
            assert!(
                try_compact_columns(
                    Backend::Avx512,
                    &input,
                    &weights,
                    &[],
                    Some(&packed),
                    &mut indices,
                    &mut scores,
                    4,
                    inner,
                    columns,
                    &cpu,
                    4,
                    &mut workspace
                )
                .unwrap()
            );
            assert_eq!(indices, expected_indices);
            assert_eq!(
                scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected_scores
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[cfg(test)]
mod portable_tests;
