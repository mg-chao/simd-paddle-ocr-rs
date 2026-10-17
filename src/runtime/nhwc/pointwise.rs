//! Ordered pointwise matrix products with direct final stores.
//!
//! Each input row is sliced before reduction. Zipped row/panel iterators keep
//! indexing checks outside K, and paired low/high updates shorten broadcast
//! lifetimes while all twelve AVX accumulators remain live. No convolution
//! origins or reduction-block scratch is needed by the baseline kernels.
//! Large macOS ARM64 convolutions can use a guarded SME tile with request-local
//! input packing; unsupported hardware or thread state retains the NEON path.
//! Short Windows x64 AVX2
//! reductions inline a register-contained loop to avoid per-tile ABI overhead;
//! larger reductions keep the isolated AVX tile.
use super::*;
#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
mod avx2;
#[cfg(target_arch = "aarch64")]
pub(super) mod neon;

const ROWS: usize = 6;
const GROUP_ROWS: usize = ROWS * 8;

/// Short reductions amortize the Windows vector save/restore and tile-return
/// boundary poorly. Larger reductions did not benefit in local measurements.
/// Keep other ISAs/platforms and large full-K panels on their existing tiles.
pub(in crate::runtime) fn inline_avx2_eligible(
    backend: Backend,
    inner: usize,
    columns: usize,
) -> bool {
    cfg!(all(target_arch = "x86_64", target_os = "windows"))
        && backend == Backend::Avx2Fma
        && (96..=192).contains(&inner)
        && columns >= 128
}

pub(super) fn avx_group_rows(inner: usize, columns: usize) -> usize {
    // Reduce repeated weight-matrix traversals in channel-heavy recognition
    // layers by reusing each full-K panel over twice as many six-row tiles.
    // Keep the original group for smaller products.
    if inner >= 384 && columns >= 384 {
        GROUP_ROWS * 2
    } else {
        GROUP_ROWS
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute(
    backend: Backend,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
) {
    assert!(packed.matches(inner, columns));
    assert_eq!(output.len() % columns, 0);
    assert!(bias.is_empty() || bias.len() == columns);
    let rows = output.len() / columns;
    let start = first
        .checked_mul(inner)
        .expect("pointwise input offset overflow");
    let length = rows
        .checked_mul(inner)
        .expect("pointwise input size overflow");
    let input = &input[start
        ..start
            .checked_add(length)
            .expect("pointwise input end overflow")];
    match backend {
        Backend::Scalar => {
            narrow::<1, false>(input, bias, packed, output, first, inner, columns, store)
        }
        Backend::Portable4 => portable(input, bias, packed, output, first, inner, columns, store),
        Backend::NeonFma => {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: CompiledModel validates the selected NEON/FMA policy;
            // the checked input and packed spans cover every complete tile.
            unsafe {
                neon::execute(input, bias, packed, output, first, inner, columns, store)
            }
        }
        Backend::Avx => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validates the selected AVX policy.
            unsafe {
                avx(input, bias, packed, output, first, inner, columns, store)
            }
        }
        Backend::AvxFma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validates AVX and FMA.
            unsafe {
                avx_fma(input, bias, packed, output, first, inner, columns, store)
            }
        }
        Backend::Avx2Fma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validates AVX2 and FMA.
            unsafe {
                #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
                if inline_avx2_eligible(backend, inner, columns) {
                    avx2::execute(input, bias, packed, output, first, inner, columns, store);
                    return;
                }
                avx2(input, bias, packed, output, first, inner, columns, store)
            }
        }
        Backend::Avx512 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validates AVX-512F, AVX2 and FMA.
            unsafe {
                avx512(input, bias, packed, output, first, inner, columns, store)
            }
        }
    }
}

macro_rules! x86_kernel {
    ($name:ident, $features:literal, $kernel:ident, $parameter:literal, $fused:literal) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = $features)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name(
            input: &[f32],
            bias: &[f32],
            packed: &PackedMatrix,
            output: &mut [f32],
            first: usize,
            inner: usize,
            columns: usize,
            store: &impl StoreTransform,
        ) {
            $kernel::<$parameter, $fused>(
                input, bias, packed, output, first, inner, columns, store,
            );
        }
    };
}
x86_kernel!(avx, "avx", wide, 8, false);
x86_kernel!(avx_fma, "avx,fma", wide, 8, true);
x86_kernel!(avx2, "avx2,fma", wide, 8, true);
x86_kernel!(avx512, "avx512f,avx2,fma", narrow, 16, true);

#[inline(always)]
fn accumulate<const LANES: usize, const FUSED: bool>(
    value: Simd<f32, LANES>,
    weights: Simd<f32, LANES>,
    sum: Simd<f32, LANES>,
) -> Simd<f32, LANES> {
    if FUSED {
        value.mul_add(weights, sum)
    } else {
        value * weights + sum
    }
}

#[inline(always)]
fn initial<const LANES: usize>(bias: &[f32], column: usize) -> Simd<f32, LANES> {
    if column >= bias.len() {
        Simd::splat(0.)
    } else {
        Simd::load_or_default(&bias[column..(column + LANES).min(bias.len())])
    }
}

#[inline(always)]
fn row_inputs(input: &[f32], tile: usize, count: usize, inner: usize) -> [&[f32]; ROWS] {
    std::array::from_fn(|r| {
        let start = (tile + r).min(count - 1) * inner;
        &input[start..start + inner]
    })
}

#[inline(always)]
fn store_row<const LANES: usize>(
    sum: Simd<f32, LANES>,
    output: &mut [f32],
    row: usize,
    column: usize,
    columns: usize,
    first: usize,
    store: &impl StoreTransform,
) {
    if column < columns {
        let count = (columns - column).min(LANES);
        let start = row * columns + column;
        let values = store.vector_channel(sum, first * columns + start, column);
        if count == LANES {
            values.copy_to_slice(&mut output[start..start + LANES]);
        } else {
            output[start..start + count].copy_from_slice(&values.to_array()[..count]);
        }
    }
}

// A function boundary prevents LICM/register-pressure decisions from sinking
// the caller's row-tail geometry back into the reduction loop. The fixed
// return buffer is written once after K, and never read or written inside K.
macro_rules! tile_kernel {
    ($name:ident, $features:literal, $fused:literal) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = $features)]
        #[inline(never)]
        unsafe fn $name<const LANES: usize>(
            rows: [&[f32]; ROWS],
            panel: &[f32],
            low: Simd<f32, LANES>,
            high: Simd<f32, LANES>,
        ) -> [[Simd<f32, LANES>; 2]; ROWS] {
            const FUSED: bool = $fused;
            let [p0, p1, p2, p3, p4, p5] = rows;
            let (mut a0, mut b0) = (low, high);
            let (mut a1, mut b1) = (low, high);
            let (mut a2, mut b2) = (low, high);
            let (mut a3, mut b3) = (low, high);
            let (mut a4, mut b4) = (low, high);
            let (mut a5, mut b5) = (low, high);
            for ((((((x0, x1), x2), x3), x4), x5), w) in p0
                .iter()
                .zip(p1)
                .zip(p2)
                .zip(p3)
                .zip(p4)
                .zip(p5)
                .zip(panel.chunks_exact(16))
            {
                let w0 = Simd::<f32, LANES>::from_slice(&w[..LANES]);
                let w1 = Simd::<f32, LANES>::from_slice(&w[LANES..]);
                let v = Simd::splat(*x0);
                a0 = accumulate::<LANES, FUSED>(v, w0, a0);
                b0 = accumulate::<LANES, FUSED>(v, w1, b0);
                let v = Simd::splat(*x1);
                a1 = accumulate::<LANES, FUSED>(v, w0, a1);
                b1 = accumulate::<LANES, FUSED>(v, w1, b1);
                let v = Simd::splat(*x2);
                a2 = accumulate::<LANES, FUSED>(v, w0, a2);
                b2 = accumulate::<LANES, FUSED>(v, w1, b2);
                let v = Simd::splat(*x3);
                a3 = accumulate::<LANES, FUSED>(v, w0, a3);
                b3 = accumulate::<LANES, FUSED>(v, w1, b3);
                let v = Simd::splat(*x4);
                a4 = accumulate::<LANES, FUSED>(v, w0, a4);
                b4 = accumulate::<LANES, FUSED>(v, w1, b4);
                let v = Simd::splat(*x5);
                a5 = accumulate::<LANES, FUSED>(v, w0, a5);
                b5 = accumulate::<LANES, FUSED>(v, w1, b5);
            }
            [[a0, b0], [a1, b1], [a2, b2], [a3, b3], [a4, b4], [a5, b5]]
        }
    };
}
tile_kernel!(tile_avx, "avx", false);
tile_kernel!(tile_fma, "avx,fma", true);

/// The AVX policies use two eight-lane vectors per row. Keep each broadcast
/// next to both updates instead of keeping six broadcasts live across halves.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn wide<const LANES: usize, const FUSED: bool>(
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
) {
    wide_grouped::<LANES, FUSED, 0>(input, bias, packed, output, first, inner, columns, store);
}

/// A zero fixed group selects production's shape policy. Test-only controls
/// can select a fixed group; the six-row ISA reduction remains the same leaf.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn wide_grouped<const LANES: usize, const FUSED: bool, const FIXED_GROUP: usize>(
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
) {
    debug_assert_eq!(LANES, 8);
    let rows = output.len() / columns;
    let group_rows = if FIXED_GROUP == 0 {
        avx_group_rows(inner, columns)
    } else {
        FIXED_GROUP
    };
    for group in (0..rows).step_by(group_rows) {
        let count = (rows - group).min(group_rows);
        let input = &input[group * inner..(group + count) * inner];
        for (block, panel) in packed.data().chunks_exact(inner * 16).enumerate() {
            let column = block * 16;
            let low = initial::<LANES>(bias, column);
            let high = initial::<LANES>(bias, column + LANES);
            for tile in (0..count).step_by(ROWS) {
                let [p0, p1, p2, p3, p4, p5] = row_inputs(input, tile, count, inner);
                // SAFETY: execute's guarded target-feature wrapper selected
                // AVX or AVX+FMA, and row_inputs/panel slices are bounds checked.
                let sums = unsafe {
                    if FUSED {
                        tile_fma([p0, p1, p2, p3, p4, p5], panel, low, high)
                    } else {
                        tile_avx([p0, p1, p2, p3, p4, p5], panel, low, high)
                    }
                };
                for (r, &[low, high]) in sums.iter().take(count - tile).enumerate() {
                    store_row(low, output, group + tile + r, column, columns, first, store);
                    store_row(
                        high,
                        output,
                        group + tile + r,
                        column + LANES,
                        columns,
                        first,
                        store,
                    );
                }
            }
        }
    }
}

/// Both halves of a paired diagnostic enter the same checked native wrapper.
/// Zero selects production; 48 is the old policy and 192 is a diagnostic only.
#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
pub(super) fn execute_group_control<const FIXED_GROUP: usize>(
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    inner: usize,
    columns: usize,
) {
    assert!(matches!(FIXED_GROUP, 0 | 48 | 192));
    assert!(simd::avx2_fma_available());
    assert!(packed.matches(inner, columns));
    assert_eq!(output.len() % columns, 0);
    assert_eq!(input.len(), output.len() / columns * inner);
    assert!(bias.is_empty() || bias.len() == columns);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // SAFETY: the runtime guard above verifies AVX2 and FMA for both controls.
    unsafe {
        execute_group_control_avx2::<FIXED_GROUP>(input, bias, packed, output, inner, columns);
    }
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2,fma")]
unsafe fn execute_group_control_avx2<const FIXED_GROUP: usize>(
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    inner: usize,
    columns: usize,
) {
    wide_grouped::<8, true, FIXED_GROUP>(
        input,
        bias,
        packed,
        output,
        0,
        inner,
        columns,
        &IdentityStore,
    );
}

/// Private descriptors establish the scalar input bounds before entering K.
/// The caller keeps ROW < ROWS, tap < tap_count and channel < channels.
pub(super) trait PortableReader {
    unsafe fn read<const ROW: usize>(&self, tap: usize, channel: usize) -> f32;
}

struct PortableRows<'a>([&'a [f32]; ROWS]);
impl PortableReader for PortableRows<'_> {
    #[inline(always)]
    unsafe fn read<const ROW: usize>(&self, _: usize, channel: usize) -> f32 {
        // SAFETY: row_inputs binds six full reduction spans. The paired tile
        // checks its K interval before advancing the channel cursor.
        unsafe { *self.0.get_unchecked(ROW).get_unchecked(channel) }
    }
}

/// Two four-lane halves keep twelve independent SSE accumulators live. This
/// boundary prevents the caller's tail/store metadata from entering K.
///
/// SAFETY: reader must describe six valid rows for tap_count/channels; its
/// read contract must hold for every tap/channel in the checked K interval.
#[inline(never)]
pub(super) unsafe fn portable_tile<const LANES: usize, const OFFSET: usize>(
    reader: &impl PortableReader,
    tap_count: usize,
    channels: usize,
    panel: &[f32],
    k0: usize,
    seeds: [[Simd<f32, LANES>; 2]; ROWS],
) -> [[Simd<f32, LANES>; 2]; ROWS] {
    assert_eq!(LANES, 4);
    assert!(OFFSET == 0 || OFFSET == 8);
    assert!(tap_count > 0);
    assert_eq!(panel.len() % 16, 0);
    assert!(k0.checked_add(panel.len() / 16).is_some_and(|end| {
        channels
            .checked_mul(tap_count)
            .is_some_and(|inner| end <= inner)
    }));
    let [s0, s1, s2, s3, s4, s5] = seeds;
    let (mut a0, mut b0) = (s0[0], s0[1]);
    let (mut a1, mut b1) = (s1[0], s1[1]);
    let (mut a2, mut b2) = (s2[0], s2[1]);
    let (mut a3, mut b3) = (s3[0], s3[1]);
    let (mut a4, mut b4) = (s4[0], s4[1]);
    let (mut a5, mut b5) = (s5[0], s5[1]);
    let (mut channel, mut tap) = (k0 / tap_count, k0 % tap_count);
    for weights in panel.as_chunks::<16>().0 {
        let w0 = Simd::from_slice(&weights[OFFSET..OFFSET + LANES]);
        let w1 = Simd::from_slice(&weights[OFFSET + LANES..OFFSET + LANES * 2]);
        macro_rules! update {
            ($row:literal, $low:ident, $high:ident) => {
                // SAFETY: descriptors bind the rows; the checked K interval
                // and cursor keep the scalar input within the active range.
                let value = Simd::splat(unsafe { reader.read::<$row>(tap, channel) });
                $low = value * w0 + $low;
                $high = value * w1 + $high;
            };
        }
        update!(0, a0, b0);
        update!(1, a1, b1);
        update!(2, a2, b2);
        update!(3, a3, b3);
        update!(4, a4, b4);
        update!(5, a5, b5);
        tap += 1;
        if tap == tap_count {
            tap = 0;
            channel += 1;
        }
    }
    [[a0, b0], [a1, b1], [a2, b2], [a3, b3], [a4, b4], [a5, b5]]
}

#[allow(clippy::too_many_arguments)]
fn portable(
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
            for pair in 0..2 {
                let column = block * 16 + pair * 8;
                if column >= columns {
                    break;
                }
                let low = initial::<4>(bias, column);
                let high = initial::<4>(bias, column + 4);
                for tile in (0..count).step_by(ROWS) {
                    let reader = PortableRows(row_inputs(input, tile, count, inner));
                    // SAFETY: all rows cover inner coefficients, tap_count is
                    // one, and panel covers exactly that complete K interval.
                    let sums = unsafe {
                        if pair == 0 {
                            portable_tile::<4, 0>(&reader, 1, inner, panel, 0, [[low, high]; ROWS])
                        } else {
                            portable_tile::<4, 8>(&reader, 1, inner, panel, 0, [[low, high]; ROWS])
                        }
                    };
                    for (r, &[low, high]) in sums.iter().take(count - tile).enumerate() {
                        store_row(low, output, group + tile + r, column, columns, first, store);
                        store_row(
                            high,
                            output,
                            group + tile + r,
                            column + 4,
                            columns,
                            first,
                            store,
                        );
                    }
                }
            }
        }
    }
}

/// Scalar and sixteen-lane AVX-512 bodies retain six named rows too.
/// Columns are independent reductions; the same body serves row/channel tails.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn narrow<const LANES: usize, const FUSED: bool>(
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
            for tile in (0..count).step_by(ROWS) {
                let [p0, p1, p2, p3, p4, p5] = row_inputs(input, tile, count, inner);
                for v in 0..16 / LANES {
                    let column = block * 16 + v * LANES;
                    if column >= columns {
                        break;
                    }
                    let bias = initial::<LANES>(bias, column);
                    let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) =
                        (bias, bias, bias, bias, bias, bias);
                    for ((((((x0, x1), x2), x3), x4), x5), w) in p0
                        .iter()
                        .zip(p1)
                        .zip(p2)
                        .zip(p3)
                        .zip(p4)
                        .zip(p5)
                        .zip(panel.as_chunks::<16>().0)
                    {
                        let w = Simd::from_slice(&w[v * LANES..(v + 1) * LANES]);
                        a0 = accumulate::<LANES, FUSED>(Simd::splat(*x0), w, a0);
                        a1 = accumulate::<LANES, FUSED>(Simd::splat(*x1), w, a1);
                        a2 = accumulate::<LANES, FUSED>(Simd::splat(*x2), w, a2);
                        a3 = accumulate::<LANES, FUSED>(Simd::splat(*x3), w, a3);
                        a4 = accumulate::<LANES, FUSED>(Simd::splat(*x4), w, a4);
                        a5 = accumulate::<LANES, FUSED>(Simd::splat(*x5), w, a5);
                    }
                    for (r, sum) in [a0, a1, a2, a3, a4, a5]
                        .into_iter()
                        .take(count - tile)
                        .enumerate()
                    {
                        store_row(sum, output, group + tile + r, column, columns, first, store);
                    }
                }
            }
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod large_channel_tests;
#[cfg(test)]
mod tests;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod sme;

/// Reserve packing only for channel-heavy convolutions with amortized SME
/// overhead. The measured targets are 384/768 and 512/1024-channel products.
pub(in crate::runtime) fn sme_scratch_elements(
    backend: Backend,
    rows: usize,
    inner: usize,
    columns: usize,
) -> usize {
    // Packing is worthwhile for the measured 192<->384 and 256<->512
    // recognition layers too. Keep short row bands and smaller channels on
    // NEON so dispatch does not assume the same crossover for tiny products.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    if backend == Backend::NeonFma
        && rows >= 32
        && inner.min(columns) >= 192
        && inner.max(columns) >= 384
        && sme::available()
    {
        return inner.checked_mul(sme::GROUP).unwrap_or(0);
    }
    let _ = (backend, rows, inner, columns);
    0
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_with_scratch(
    backend: Backend,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    inner: usize,
    columns: usize,
    scratch: &mut [f32],
    store: &impl StoreTransform,
) {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    if sme_scratch_elements(backend, output.len() / columns, inner, columns) > 0
        && sme::try_execute(
            input, bias, packed, output, first, inner, columns, scratch, store,
        )
    {
        return;
    }
    let _ = scratch;
    execute(
        backend, input, bias, packed, output, first, inner, columns, store,
    );
}

/// Isolate SME workspace dispatch so every unaffected kernel retains its
/// original row-band closure and call path.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_sme(
    backend: Backend,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    inner: usize,
    columns: usize,
    cpu: &CpuContext,
    threads: usize,
    workspaces: &mut [Vec<f32>],
    store: &impl StoreTransform,
) -> bool {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    if sme_scratch_elements(backend, output.len() / columns, inner, columns) > 0
        && !workspaces.is_empty()
    {
        sme::execute_bands(
            backend, input, bias, packed, output, inner, columns, cpu, threads, workspaces, store,
        );
        return true;
    }
    let _ = (
        backend, input, bias, packed, output, inner, columns, cpu, threads, workspaces, store,
    );
    false
}
