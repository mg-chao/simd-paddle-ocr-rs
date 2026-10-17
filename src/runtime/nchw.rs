//! Canonical NCHW specializations. Packed kernels accumulate directly into
//! disjoint final planes; no transient layout, coefficient copy, or per-run
//! scratch allocation is needed.
use super::{
    packing::PackedMatrix,
    simd::{self, Backend, Conv2d},
};
use crate::{cpu::CpuContext, model::ConvAttributes};
use std::simd::{Simd, StdFloat};

mod native;
#[cfg(target_arch = "x86_64")]
mod pad_end2;
mod pointwise_spatial;
mod reduce3;
mod stride2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    PointwiseOcMajor,
    PointwisePacked,
    PointwiseSpatial,
    PointwiseBlocked,
    PointwiseTiled,
    Packed3,
    Dense3,
    PackedStride2,
    Stride2,
    PadEnd2,
    PackedDense,
    Dense,
    Depth3,
    Depth3Tap,
    Depth5,
    Depth5Tap,
    Depth7,
    Depth9,
    Depth1x5,
    DepthHeight3,
    DepthHeight5,
    DepthStride3,
    DepthGeneral,
    PackedDepth9,
    Packed3Pair16,
    PackedStride2Pair16,
}
impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::PointwiseOcMajor => "nchw_pointwise_oc_major",
            Self::PointwisePacked => "nchw_pointwise_packed",
            Self::PointwiseSpatial => "nchw_pointwise_spatial",
            Self::PointwiseBlocked => "nchw_pointwise_blocked_working_set",
            Self::PointwiseTiled => "nchw_pointwise_tiled_working_set",
            Self::Packed3 => "nchw_packed_3x3",
            Self::Packed3Pair16 => "nchw_packed_3x3_pair16",
            Self::Dense3 => "nchw_dense_3x3",
            Self::PackedStride2 => "nchw_packed_3x3_stride2",
            Self::PackedStride2Pair16 => "nchw_packed_3x3_stride2_pair16",
            Self::Stride2 => "nchw_dense_3x3_stride2",
            Self::PadEnd2 => "nchw_dense_2x2_pad_end",
            Self::PackedDense => "nchw_packed_dense_stride1",
            Self::Dense => "nchw_dense_stride1",
            Self::Depth3 => "nchw_depthwise_3x3_output_vector",
            Self::Depth3Tap => "nchw_depthwise_3x3_tap_major",
            Self::Depth5 => "nchw_depthwise_5x5_output_vector",
            Self::Depth5Tap => "nchw_depthwise_5x5_tap_major",
            Self::Depth7 => "nchw_depthwise_7x7_strip",
            Self::Depth9 => "nchw_depthwise_9x9_strip",
            Self::Depth1x5 => "nchw_depthwise_1x5",
            Self::DepthHeight3 => "nchw_depthwise_3x3_height2",
            Self::DepthHeight5 => "nchw_depthwise_5x5_height2",
            Self::DepthStride3 => "nchw_depthwise_3x3_stride2",
            Self::DepthGeneral => "nchw_depthwise_stride1_strip",
            Self::PackedDepth9 => "nchw_depthwise_9x9_channel_packed",
        }
    }
}

/// Static coefficient eligibility; spatial/work gates are rechecked at bind.
pub(super) fn pack_eligible(attributes: &ConvAttributes, dimensions: &[usize]) -> bool {
    if dimensions.len() != 4 || attributes.dilations != [1, 1] {
        return false;
    }
    let cout = dimensions[0];
    let kernel = &dimensions[2..];
    if attributes.group != 1 {
        return usize::try_from(attributes.group).ok() == Some(cout)
            && dimensions[1] == 1
            && kernel == [9, 9]
            && attributes.strides == [1, 1]
            && attributes.pads == [4; 4]
            && cout >= 8
            && cout.is_multiple_of(8);
    }
    (kernel == [1, 1]
        && attributes.strides == [1, 1]
        && attributes.pads == [0; 4]
        && cout >= 4
        && cout.is_multiple_of(4))
        || (kernel == [3, 3]
            && matches!(attributes.strides, [1, 1] | [2, 2])
            && attributes.pads == [1; 4]
            && cout >= 8
            && cout.is_multiple_of(8))
        || (matches!(kernel, [5, 5] | [7, 7])
            && attributes.strides == [1, 1]
            && dimensions[1] >= 16
            && cout >= 8
            && cout.is_multiple_of(8))
}
fn work(p: &Conv2d) -> u128 {
    [
        p.batch,
        p.output_channels,
        p.input_channels / p.groups,
        p.output_height,
        p.output_width,
        p.kernel_height,
        p.kernel_width,
    ]
    .into_iter()
    .map(|v| v as u128)
    .product()
}
fn interior(p: &Conv2d) -> std::ops::Range<usize> {
    let first = p.pad_left.div_ceil(p.stride_width).min(p.output_width);
    let end = (p.width + p.pad_left)
        .saturating_sub(p.kernel_width - 1)
        .div_ceil(p.stride_width)
        .min(p.output_width)
        .max(first);
    first..end
}
fn select(backend: Backend, p: &Conv2d, packed: bool, threads: usize) -> Option<Kind> {
    // Canonical specializations retain Portable4 arithmetic and fused stores;
    // the ARM64 FMA tiles specialize channels-last execution.
    if backend == Backend::NeonFma {
        return select(Backend::Portable4, p, packed, threads);
    }
    if backend == Backend::Scalar
        || p.groups == 0
        || p.dilation_height != 1
        || p.dilation_width != 1
    {
        return None;
    }
    let (kh, kw, sh, sw) = (
        p.kernel_height,
        p.kernel_width,
        p.stride_height,
        p.stride_width,
    );
    let dense = p.groups == 1;
    let same = p.output_height == p.height && p.output_width == p.width;
    let eight = p.output_channels >= 8 && p.output_channels.is_multiple_of(8);
    if dense
        && kh == 3
        && kw == 3
        && sh == 1
        && sw == 1
        && p.pad_top == 1
        && p.pad_left == 1
        && same
        && packed
        && eight
    {
        return Some(
            if backend == Backend::Portable4 && p.output_channels == 16 && threads > 2 {
                Kind::Dense3
            } else if backend != Backend::Portable4
                && threads == 1
                && p.output_channels >= 64
                && p.output_channels.is_multiple_of(16)
            {
                Kind::Packed3Pair16
            } else {
                Kind::Packed3
            },
        );
    }
    if dense
        && packed
        && sh == 1
        && sw == 1
        && matches!((kh, kw), (5, 5) | (7, 7))
        && p.input_channels >= 16
        && eight
        && matches!(backend, Backend::Avx2Fma | Backend::Avx512)
        && interior(p).len() >= 4
        && p.height + p.pad_top >= kh
        && work(p) >= 4_000_000
        && threads <= p.output_channels / 8
    {
        return Some(Kind::PackedDense);
    }
    if kh == 1 && kw == 1 && sh == 1 && sw == 1 && p.pad_top == 0 && p.pad_left == 0 && same {
        if dense
            && packed
            && p.output_channels >= 16
            && p.output_channels.is_multiple_of(16)
            && p.height * p.width < 48
        {
            return Some(Kind::PointwiseOcMajor);
        }
        if dense && packed && p.output_channels.is_multiple_of(4) {
            let plane = p.height * p.width;
            if backend == Backend::Avx2Fma
                && p.input_channels >= 32
                && p.output_channels >= 16
                && p.output_channels.is_multiple_of(16)
                && plane >= 32
                && work(p) / (p.batch as u128) >= 4_000_000
            {
                return Some(if plane > 4096 {
                    Kind::PointwiseTiled
                } else {
                    Kind::PointwiseBlocked
                });
            }
            return Some(Kind::PointwisePacked);
        }
        return Some(Kind::PointwiseSpatial);
    }
    if p.groups == p.input_channels && p.output_channels == p.input_channels {
        if sh == 1 && sw == 1 && same {
            match (kh, kw, p.pad_top, p.pad_left) {
                (3, 3, 1, 1) => {
                    return Some(
                        if backend != Backend::Portable4
                            && p.batch == 1
                            && p.height >= 50
                            && p.width >= 10
                        {
                            Kind::Depth3
                        } else {
                            Kind::Depth3Tap
                        },
                    );
                }
                (5, 5, 2, 2) => {
                    return Some(
                        if backend != Backend::Portable4
                            && p.batch == 1
                            && p.height >= 10
                            && p.width >= 10
                            && !(threads > 1 && p.output_channels >= 2 && work(p) >= 8_000_000)
                        {
                            Kind::Depth5
                        } else {
                            Kind::Depth5Tap
                        },
                    );
                }
                (7, 7, 3, 3) => return Some(Kind::Depth7),
                (9, 9, 4, 4) => {
                    return Some(
                        if packed
                            && matches!(backend, Backend::Avx2Fma | Backend::Avx512)
                            && p.input_channels >= 8
                            && p.input_channels.is_multiple_of(8)
                            && p.height >= 9
                            && p.width >= 9
                            && p.height * p.width <= 4096
                            && work(p) >= 1_000_000
                        {
                            Kind::PackedDepth9
                        } else if backend != Backend::Avx512
                            && p.batch == 1
                            && p.height >= 9
                            && p.width >= 9
                            && work(p) >= 4_000_000
                        {
                            Kind::Depth9
                        } else {
                            Kind::DepthGeneral
                        },
                    );
                }
                (1, 5, 0, 2) => return Some(Kind::Depth1x5),
                _ => {}
            }
        }
        if sh == 2 && sw == 1 && p.output_width == p.width {
            if kh == 3 && kw == 3 && p.pad_top == 1 && p.pad_left == 1 {
                return Some(Kind::DepthHeight3);
            }
            if kh == 5 && kw == 5 && p.pad_top == 2 && p.pad_left == 2 {
                return Some(Kind::DepthHeight5);
            }
        }
        if kh == 3 && kw == 3 && sh == 2 && sw == 2 && p.pad_top == 1 && p.pad_left == 1 {
            return Some(Kind::DepthStride3);
        }
        if sh == 1 && sw == 1 && !interior(p).is_empty() {
            return Some(Kind::DepthGeneral);
        }
    }
    if dense {
        if kh == 3 && kw == 3 && sh == 1 && sw == 1 && p.pad_top == 1 && p.pad_left == 1 && same {
            return Some(Kind::Dense3);
        }
        if kh == 2 && kw == 2 && sh == 1 && sw == 1 && p.pad_top == 0 && p.pad_left == 0 && same {
            return Some(Kind::PadEnd2);
        }
        if kh == 3 && kw == 3 && sh == 2 && sw == 2 && p.pad_top == 1 && p.pad_left == 1 {
            return Some(
                if packed
                    && eight
                    && !(backend == Backend::Portable4 && p.output_channels == 16 && threads > 2)
                {
                    if matches!(backend, Backend::Avx | Backend::AvxFma | Backend::Avx2Fma)
                        && (threads <= 1 || p.batch != 1 || p.output_channels < 16)
                        && p.output_channels.is_multiple_of(16)
                    {
                        Kind::PackedStride2Pair16
                    } else {
                        Kind::PackedStride2
                    }
                } else {
                    Kind::Stride2
                },
            );
        }
        let lanes = if backend == Backend::Portable4 { 4 } else { 8 };
        if sh == 1 && sw == 1 && p.input_channels * kh * kw <= 1 << 20 && interior(p).len() >= lanes
        {
            return Some(Kind::Dense);
        }
    }
    None
}
pub(super) fn family(backend: Backend, p: &Conv2d, packed: bool) -> Option<&'static str> {
    select(backend, p, packed, 1).map(Kind::name)
}
pub(super) fn family_for_threads(
    backend: Backend,
    p: &Conv2d,
    packed: bool,
    threads: usize,
) -> Option<&'static str> {
    if threads == 1 {
        family(backend, p, packed)
    } else {
        #[cfg(feature = "parallel")]
        if let Some(plan) = packed3_band_plan(backend, p, packed, threads, threads)
            .or_else(|| packed_stride2_band_plan(backend, p, packed, threads, threads))
        {
            let mut paired = false;
            let mut eight = false;
            for worker in 0..plan.workers() {
                let band = plan.band(worker).expect("planned packed3 worker");
                paired |= matches!(band.kind, Kind::Packed3Pair16 | Kind::PackedStride2Pair16);
                eight |= matches!(band.kind, Kind::Packed3 | Kind::PackedStride2);
            }
            if paired {
                return Some(match (plan.stride2, eight) {
                    (true, true) => "nchw_packed_3x3_stride2_parallel_mixed16_8",
                    (true, false) => "nchw_packed_3x3_stride2_parallel_pair16",
                    (false, true) => "nchw_packed_3x3_parallel_mixed16_8",
                    (false, false) => "nchw_packed_3x3_parallel_pair16",
                });
            }
        }
        select(backend, p, packed, threads).map(Kind::name)
    }
}
pub(super) fn selected_backend_for_threads(
    backend: Backend,
    p: &Conv2d,
    packed: bool,
    threads: usize,
) -> Backend {
    if backend == Backend::NeonFma {
        return Backend::Portable4;
    }
    if backend == Backend::Avx512 {
        let kind = select(backend, p, packed, threads);
        if matches!(
            kind,
            Some(Kind::Dense | Kind::PackedDense | Kind::PackedDepth9 | Kind::Packed3Pair16)
        ) {
            return Backend::Avx2Fma;
        }
    }
    backend
}
pub(super) fn uses_pack_for_threads(
    backend: Backend,
    p: &Conv2d,
    packed: bool,
    threads: usize,
) -> bool {
    packed
        && matches!(
            select(backend, p, packed, threads),
            Some(
                Kind::PointwiseOcMajor
                    | Kind::PointwisePacked
                    | Kind::PointwiseBlocked
                    | Kind::PointwiseTiled
                    | Kind::Packed3
                    | Kind::Packed3Pair16
                    | Kind::PackedStride2
                    | Kind::PackedStride2Pair16
                    | Kind::PackedDense
                    | Kind::PackedDepth9
            )
        )
}
fn check_backend(backend: Backend) {
    let supported = match backend {
        Backend::Scalar | Backend::Portable4 => true,
        Backend::NeonFma => simd::neon_available(),
        Backend::Avx => simd::avx_available(),
        Backend::AvxFma => simd::avx_fma_available(),
        Backend::Avx2Fma => simd::avx2_fma_available(),
        Backend::Avx512 => simd::avx512_available(),
    };
    assert!(
        supported,
        "NCHW backend {backend:?} is unavailable on this CPU"
    );
}
fn validate(p: &Conv2d, input: &[f32], weights: &[f32], bias: &[f32], output: &[f32]) {
    assert!(
        [
            p.batch,
            p.input_channels,
            p.output_channels,
            p.height,
            p.width,
            p.output_height,
            p.output_width,
            p.kernel_height,
            p.kernel_width,
            p.stride_height,
            p.stride_width,
            p.groups
        ]
        .into_iter()
        .all(|v| v > 0)
    );
    assert_eq!(p.input_channels % p.groups, 0);
    assert_eq!(p.output_channels % p.groups, 0);
    let product = |dims: &[usize]| {
        dims.iter()
            .try_fold(1usize, |n, &v| n.checked_mul(v))
            .expect("NCHW extent overflow")
    };
    assert_eq!(
        input.len(),
        product(&[p.batch, p.input_channels, p.height, p.width])
    );
    assert_eq!(
        weights.len(),
        product(&[
            p.output_channels,
            p.input_channels / p.groups,
            p.kernel_height,
            p.kernel_width
        ])
    );
    assert_eq!(
        output.len(),
        product(&[p.batch, p.output_channels, p.output_height, p.output_width])
    );
    assert!(bias.is_empty() || bias.len() == p.output_channels);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_conv(
    backend: Backend,
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
) -> bool {
    try_conv_store(
        backend,
        p,
        input,
        weights,
        bias,
        packed,
        output,
        cpu,
        threads,
        &simd::IdentityStore,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_conv_store(
    backend: Backend,
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
    store: &impl simd::StoreTransform,
) -> bool {
    check_backend(backend);
    validate(p, input, weights, bias, output);
    let inner = p.input_channels / p.groups * p.kernel_height * p.kernel_width;
    let packed = packed.filter(|panel| panel.matches(inner, p.output_channels));
    let Some(kind) = select(backend, p, packed.is_some(), threads) else {
        return false;
    };
    let actual = selected_backend_for_threads(backend, p, packed.is_some(), threads);
    let plane = p.output_height * p.output_width;
    #[cfg(feature = "parallel")]
    if p.batch > 1
        && threads > 1
        && work(p) / p.batch as u128 >= 1_000_000
        && matches!(kind, Kind::PointwisePacked | Kind::PointwiseBlocked)
    {
        use rayon::prelude::*;
        cpu.install(|| {
            output
                .par_chunks_mut(p.output_channels * plane)
                .enumerate()
                .for_each(|(batch, out)| {
                    run(
                        actual,
                        kind,
                        p,
                        input,
                        weights,
                        bias,
                        packed,
                        out,
                        batch,
                        0,
                        p.output_channels,
                        store,
                    );
                })
        });
        return true;
    }
    #[cfg(feature = "parallel")]
    if spatial_shard_eligible(actual, p, packed.is_some(), threads.min(cpu.capacity())) {
        spatial_shards(
            actual, p, input, weights, bias, packed, output, cpu, threads, store,
        );
        return true;
    }
    #[cfg(feature = "parallel")]
    if let Some(plan) = packed3_band_plan(actual, p, packed.is_some(), threads, cpu.capacity())
        .or_else(|| packed_stride2_band_plan(actual, p, packed.is_some(), threads, cpu.capacity()))
    {
        for (batch, out) in output
            .chunks_exact_mut(p.output_channels * plane)
            .enumerate()
        {
            super::parallel::aligned_bands(
                out,
                8 * plane,
                p.output_channels / 8,
                plan.workers(),
                1,
                cpu,
                |out, first_block| {
                    let worker = plan.worker_at_first_block(first_block);
                    let band = plan.band(worker).expect("aligned packed3 channel band");
                    debug_assert_eq!(band.first, first_block * 8);
                    debug_assert_eq!(band.count * plane, out.len());
                    run(
                        actual, band.kind, p, input, weights, bias, packed, out, batch, band.first,
                        band.count, store,
                    );
                },
            );
        }
        return true;
    }
    #[cfg(feature = "parallel")]
    let tile = if matches!(kind, Kind::PointwiseBlocked | Kind::PointwiseTiled) {
        16
    } else if kind == Kind::PackedDepth9 {
        8
    } else if p.groups == p.input_channels {
        1
    } else {
        8
    };
    #[cfg(feature = "parallel")]
    if channel_shard_eligible(actual, kind, p, threads) && p.output_channels > tile {
        use rayon::prelude::*;
        let channels = p
            .output_channels
            .div_ceil(threads.min(cpu.capacity()))
            .max(tile)
            .next_multiple_of(tile);
        for (batch, out) in output
            .chunks_exact_mut(p.output_channels * plane)
            .enumerate()
        {
            cpu.install(|| {
                out.par_chunks_mut(channels * plane)
                    .enumerate()
                    .for_each(|(band, out)| {
                        run(
                            actual,
                            kind,
                            p,
                            input,
                            weights,
                            bias,
                            packed,
                            out,
                            batch,
                            band * channels,
                            out.len() / plane,
                            store,
                        )
                    })
            });
        }
        return true;
    }
    #[cfg(not(feature = "parallel"))]
    let _ = cpu;
    for (batch, out) in output
        .chunks_exact_mut(p.output_channels * plane)
        .enumerate()
    {
        run(
            actual,
            kind,
            p,
            input,
            weights,
            bias,
            packed,
            out,
            batch,
            0,
            p.output_channels,
            store,
        );
    }
    true
}

/// Read-only plan for the pinned packed3/stride2 OC8 channel split. The same boundaries
/// drive execution and per-band family queries; no temporary band vector exists.
#[cfg(any(feature = "parallel", test))]
#[derive(Clone, Copy, Debug)]
pub(super) struct PackedChannelBandPlan {
    backend: Backend,
    blocks: usize,
    workers: usize,
    stride2: bool,
}

#[cfg(any(feature = "parallel", test))]
#[derive(Clone, Copy, Debug)]
pub(super) struct PackedChannelBand {
    pub(super) first: usize,
    pub(super) count: usize,
    kind: Kind,
}

#[cfg(any(feature = "parallel", test))]
impl PackedChannelBand {
    #[cfg(test)]
    pub(super) fn family(self) -> &'static str {
        self.kind.name()
    }
}

#[cfg(any(feature = "parallel", test))]
impl PackedChannelBandPlan {
    pub(super) fn workers(self) -> usize {
        self.workers
    }

    pub(super) fn band(self, worker: usize) -> Option<PackedChannelBand> {
        if worker >= self.workers {
            return None;
        }
        let first = super::parallel::band_boundary(self.blocks, self.workers, 1, worker) * 8;
        let end = super::parallel::band_boundary(self.blocks, self.workers, 1, worker + 1) * 8;
        let count = end - first;
        // AVX channel workers recursively enter Try(..., childCount, 1).
        // AVX512 children retain eight-output native bodies; Portable4
        // children enter Vector bodies, which also own OC8.
        let paired = matches!(
            self.backend,
            Backend::Avx | Backend::AvxFma | Backend::Avx2Fma
        ) && (self.stride2 || count >= 64)
            && count.is_multiple_of(16);
        let kind = match (self.stride2, paired) {
            (true, true) => Kind::PackedStride2Pair16,
            (true, false) => Kind::PackedStride2,
            (false, true) => Kind::Packed3Pair16,
            (false, false) => Kind::Packed3,
        };
        Some(PackedChannelBand { first, count, kind })
    }

    #[cfg(feature = "parallel")]
    fn worker_at_first_block(self, first: usize) -> usize {
        // There are at least as many OC8 blocks as workers, so all boundaries
        // are distinct. This is the inverse of floor(blocks * worker/workers).
        (((first as u128 + 1) * self.workers as u128 - 1) / self.blocks as u128) as usize
    }
}

#[cfg(any(feature = "parallel", test))]
pub(super) fn packed3_band_plan(
    backend: Backend,
    p: &Conv2d,
    packed: bool,
    threads: usize,
    capacity: usize,
) -> Option<PackedChannelBandPlan> {
    if capacity == 0
        || select(backend, p, packed, threads) != Some(Kind::Packed3)
        || !channel_shard_eligible(backend, Kind::Packed3, p, threads)
    {
        return None;
    }
    let blocks = p.output_channels / 8;
    let workers = threads.min(capacity).min(blocks);
    if workers <= 1 {
        return None;
    }
    Some(PackedChannelBandPlan {
        backend,
        blocks,
        workers,
        stride2: false,
    })
}

#[cfg(any(feature = "parallel", test))]
pub(super) fn packed_stride2_band_plan(
    backend: Backend,
    p: &Conv2d,
    packed: bool,
    threads: usize,
    capacity: usize,
) -> Option<PackedChannelBandPlan> {
    if capacity == 0
        || threads <= 1
        || p.batch != 1
        || select(backend, p, packed, threads) != Some(Kind::PackedStride2)
        || p.output_channels < 16
    {
        return None;
    }
    let blocks = p.output_channels / 8;
    let workers = threads.min(capacity).min(blocks);
    if workers <= 1 {
        return None;
    }
    Some(PackedChannelBandPlan {
        backend,
        blocks,
        workers,
        stride2: true,
    })
}

#[cfg(any(feature = "parallel", test))]
fn channel_shard_eligible(backend: Backend, kind: Kind, p: &Conv2d, threads: usize) -> bool {
    if threads <= 1 || p.batch != 1 || p.output_channels < 2 {
        return false;
    }
    let minimum = match kind {
        Kind::Depth3
        | Kind::Depth3Tap
        | Kind::Depth1x5
        | Kind::DepthHeight3
        | Kind::DepthHeight5
        | Kind::DepthStride3 => return false,
        Kind::Depth5 | Kind::Depth5Tap | Kind::Depth7 => 8_000_000,
        Kind::PointwiseOcMajor => {
            if backend == Backend::Avx512 {
                1_000_000
            } else {
                8_000_000
            }
        }
        Kind::Dense3
        | Kind::Packed3
        | Kind::Stride2
        | Kind::PackedStride2
        | Kind::PointwiseSpatial => 8_000_000,
        Kind::PointwisePacked => 1_000_000,
        _ => 4_000_000,
    };
    work(p) >= minimum
}

/// The source plain dense path splits rows to retain its complete eight-OC
/// accumulator set for narrow output channel counts. Pad-end 2x2 likewise
/// splits rows when its single eight-OC tile cannot be split across channels.
#[cfg(any(feature = "parallel", test))]
fn spatial_shard_eligible(backend: Backend, p: &Conv2d, packed: bool, threads: usize) -> bool {
    if threads <= 1 {
        return false;
    }
    let kind = select(backend, p, packed, threads);
    if kind == Some(Kind::PackedDepth9) {
        return true;
    }
    if p.batch != 1 || work(p) < 4_000_000 {
        return false;
    }
    match kind {
        Some(Kind::Dense) => {
            matches!(backend, Backend::Avx2Fma | Backend::Avx512)
                && p.output_channels.is_multiple_of(8)
                && p.output_channels <= 64
                && p.output_height >= threads.min(p.output_channels) * 2
        }
        Some(Kind::PadEnd2) => {
            backend != Backend::Portable4
                && p.output_channels == 8
                && p.output_height >= threads * 2
        }
        Some(Kind::PointwisePacked) => {
            matches!(backend, Backend::Avx | Backend::AvxFma | Backend::Avx2Fma)
                && p.height * p.width >= 4096
                && work(p) >= 8_000_000
        }
        _ => false,
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn spatial_shards(
    backend: Backend,
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    output: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
    store: &impl simd::StoreTransform,
) {
    let workers = threads.min(cpu.capacity()).min(p.output_channels);
    let rows = p.output_height.div_ceil(workers);
    let plane = p.output_height * p.output_width;
    let pointwise = p.kernel_height == 1;
    let chunk = if pointwise {
        plane.div_ceil(workers).next_multiple_of(16)
    } else {
        rows * p.output_width
    };
    // The stack descriptor is bounded independently of model channel count.
    // Wider packed pointwise projections traverse64-channel windows while each
    // scoped worker keeps disjoint slices from every channel in its window.
    for (batch, output) in output
        .chunks_exact_mut(p.output_channels * plane)
        .enumerate()
    {
        for (window, output) in output.chunks_mut(64 * plane).enumerate() {
            let first = window * 64;
            let mut planes = output.chunks_exact_mut(plane);
            let mut channels: [Option<std::slice::ChunksMut<'_, f32>>; 64] =
                std::array::from_fn(|_| planes.next().map(|plane| plane.chunks_mut(chunk)));
            cpu.install(|| {
                rayon::scope(|scope| {
                    for offset in (0..plane).step_by(chunk) {
                        let begin = if pointwise {
                            offset
                        } else {
                            offset / p.output_width
                        };
                        let band = std::array::from_fn(|co| {
                            channels[co].as_mut().and_then(Iterator::next)
                        });
                        scope.spawn(move |_| {
                            spatial_rows_run(
                                backend, p, input, weights, bias, packed, band, begin, batch,
                                first, store,
                            )
                        });
                    }
                })
            });
        }
    }
}
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn spatial_rows_run(
    backend: Backend,
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    band: [Option<&mut [f32]>; 64],
    begin: usize,
    batch: usize,
    first: usize,
    store: &impl simd::StoreTransform,
) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: try_conv_store checks the actual selected runtime ISA.
    unsafe {
        match backend {
            Backend::Avx => spatial_rows_avx(
                p, input, weights, bias, packed, band, begin, batch, first, store,
            ),
            Backend::AvxFma => spatial_rows_fma(
                p, input, weights, bias, packed, band, begin, batch, first, store,
            ),
            Backend::Avx2Fma => spatial_rows_avx2(
                p, input, weights, bias, packed, band, begin, batch, first, store,
            ),
            Backend::Avx512 => spatial_rows_avx512(
                p, input, weights, bias, packed, band, begin, batch, first, store,
            ),
            _ => unreachable!("spatial sharding requires x64 AVX"),
        }
    }
}
macro_rules! target_spatial_rows {
    ($name:ident,$features:literal,$lanes:literal,$fused:literal) => {
        #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
        #[target_feature(enable=$features)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name(
            p: &Conv2d,
            input: &[f32],
            weights: &[f32],
            bias: &[f32],
            packed: Option<&PackedMatrix>,
            band: [Option<&mut [f32]>; 64],
            begin: usize,
            batch: usize,
            first: usize,
            store: &impl simd::StoreTransform,
        ) {
            spatial_rows::<$lanes, $fused>(
                p, input, weights, bias, packed, band, begin, batch, first, store,
            );
        }
    };
}
target_spatial_rows!(spatial_rows_avx, "avx", 8, false);
target_spatial_rows!(spatial_rows_fma, "avx,fma", 8, true);
target_spatial_rows!(spatial_rows_avx2, "avx2,fma", 8, true);
target_spatial_rows!(spatial_rows_avx512, "avx512f,avx2,fma", 16, true);

#[cfg(feature = "parallel")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn spatial_rows_impl<const N: usize, const FMA: bool>(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    mut band: [Option<&mut [f32]>; 64],
    begin: usize,
    batch: usize,
    first: usize,
    store: &impl simd::StoreTransform,
) {
    if p.groups == p.input_channels && p.kernel_height == 9 {
        packed_depth_rows::<N, FMA>(
            p,
            input,
            bias,
            packed.expect("selected packed9 panel"),
            band,
            begin,
            batch,
            first,
            store,
        );
        return;
    }
    assert_eq!(batch, 0);
    if p.kernel_height == 1 {
        packed_rows::<N, FMA>(
            p,
            input,
            bias,
            packed.expect("selected pointwise panel"),
            band,
            begin,
            batch,
            first,
            store,
        );
        return;
    }
    let plane = p.output_height * p.output_width;
    let rows = band[0].as_ref().expect("nonempty row band").len() / p.output_width;
    let valid = interior(p);
    for local_co in (0..band.iter().take_while(|channel| channel.is_some()).count()).step_by(8) {
        let co = first + local_co;
        for local_y in 0..rows {
            let y = begin + local_y;
            let mut x = 0;
            while x < valid.start {
                for lane in 0..8 {
                    band[local_co + lane].as_mut().unwrap()[local_y * p.output_width + x] = store
                        .scalar(
                            edge(p, input, weights, bias, 0, co + lane, y, x),
                            (co + lane) * plane + y * p.output_width + x,
                        );
                }
                x += 1;
            }
            while x + N <= valid.end {
                let mut sums = std::array::from_fn::<_, 8, _>(|lane| {
                    Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + lane] })
                });
                for ci in 0..p.input_channels {
                    for ky in 0..p.kernel_height {
                        let iy = (y + ky) as isize - p.pad_top as isize;
                        if iy < 0 || iy >= p.height as isize {
                            continue;
                        }
                        for kx in 0..p.kernel_width {
                            let source =
                                (ci * p.height + iy as usize) * p.width + x + kx - p.pad_left;
                            let value = Simd::from_slice(&input[source..source + N]);
                            for (lane, sum) in sums.iter_mut().enumerate() {
                                let w = weights[(((co + lane) * p.input_channels + ci)
                                    * p.kernel_height
                                    + ky)
                                    * p.kernel_width
                                    + kx];
                                *sum = madd::<N, FMA>(value, Simd::splat(w), *sum);
                            }
                        }
                    }
                }
                for (lane, sum) in sums.into_iter().enumerate() {
                    store
                        .vector(sum, (co + lane) * plane + y * p.output_width + x)
                        .copy_to_slice(
                            &mut band[local_co + lane].as_mut().unwrap()
                                [local_y * p.output_width + x..][..N],
                        );
                }
                x += N;
            }
            for x in x..p.output_width {
                for lane in 0..8 {
                    band[local_co + lane].as_mut().unwrap()[local_y * p.output_width + x] = store
                        .scalar(
                            edge(p, input, weights, bias, 0, co + lane, y, x),
                            (co + lane) * plane + y * p.output_width + x,
                        );
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_rows_impl<const N: usize, const FMA: bool>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    mut band: [Option<&mut [f32]>; 64],
    begin: usize,
    batch: usize,
    first: usize,
    store: &impl simd::StoreTransform,
) {
    assert_eq!(batch, 0);
    let plane = p.height * p.width;
    let rows = band[0].as_ref().unwrap().len();
    let count = band.iter().take_while(|channel| channel.is_some()).count();
    for local_co in (0..count).step_by(N) {
        let co = first + local_co;
        let columns = (count - local_co).min(N);
        for spatial in (0..rows).step_by(4) {
            let active = (rows - spatial).min(4);
            let mut sums = [initial::<N>(bias, co, p.output_channels); 4];
            for ci in 0..p.input_channels {
                let w = coefficient::<N>(packed, p.input_channels, ci, co, p.output_channels);
                for (r, sum) in sums[..active].iter_mut().enumerate() {
                    *sum = madd::<N, FMA>(
                        Simd::splat(input[ci * plane + begin + spatial + r]),
                        w,
                        *sum,
                    );
                }
            }
            for (r, sum) in sums[..active].iter().enumerate() {
                for lane in 0..columns {
                    band[local_co + lane].as_mut().unwrap()[spatial + r] =
                        store.scalar(sum[lane], (co + lane) * plane + begin + spatial + r);
                }
            }
        }
    }
}

/// Four-pixel/channel-vector packed9 microkernel over scoped NCHW row slices.
/// Each coefficient vector is reused across four positions. Safe gathers avoid
/// full-layout scratch and preserve the source FMA chain, including all edges.
#[cfg(feature = "parallel")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_depth_rows_impl<const N: usize, const FMA: bool>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    mut band: [Option<&mut [f32]>; 64],
    begin: usize,
    batch: usize,
    first: usize,
    store: &impl simd::StoreTransform,
) {
    // The selected packed9 policy deliberately executes at AVX2 width.
    assert_eq!(N, 8);
    assert!(FMA);
    let plane = p.height * p.width;
    let rows = band[0].as_ref().unwrap().len() / p.width;
    let count = band.iter().take_while(|channel| channel.is_some()).count();
    for y in 0..rows {
        for x in (0..p.width).step_by(4) {
            let active = (p.width - x).min(4);
            for local_co in (0..count).step_by(8) {
                let co = first + local_co;
                let mut sums = [initial::<8>(bias, co, p.output_channels); 4];
                for ky in 0..9 {
                    let iy = (begin + y + ky) as isize - 4;
                    if iy < 0 || iy >= p.height as isize {
                        continue;
                    }
                    for kx in 0..9 {
                        let weight =
                            coefficient::<8>(packed, 81, ky * 9 + kx, co, p.output_channels);
                        for (r, sum) in sums[..active].iter_mut().enumerate() {
                            let ix = (x + r + kx) as isize - 4;
                            if ix < 0 || ix >= p.width as isize {
                                continue;
                            }
                            let values = Simd::from_array(std::array::from_fn(|lane| {
                                input[(batch * p.input_channels + co + lane) * plane
                                    + iy as usize * p.width
                                    + ix as usize]
                            }));
                            *sum = madd::<8, true>(values, weight, *sum);
                        }
                    }
                }
                for (r, sum) in sums[..active].iter().enumerate() {
                    for lane in 0..8 {
                        band[local_co + lane].as_mut().unwrap()[y * p.width + x + r] = store
                            .scalar(
                                sum[lane],
                                (batch * p.output_channels + co + lane) * plane
                                    + (begin + y) * p.width
                                    + x
                                    + r,
                            );
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    backend: Backend,
    kind: Kind,
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    batch: usize,
    first: usize,
    count: usize,
    store: &impl simd::StoreTransform,
) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: try_conv validates the requested ISA before selecting this entry.
    unsafe {
        match backend {
            Backend::Avx => {
                run_avx(
                    kind, p, input, weights, bias, packed, out, batch, first, count, store,
                );
                return;
            }
            Backend::AvxFma => {
                run_avx_fma(
                    kind, p, input, weights, bias, packed, out, batch, first, count, store,
                );
                return;
            }
            Backend::Avx2Fma => {
                run_avx2(
                    kind, p, input, weights, bias, packed, out, batch, first, count, store,
                );
                return;
            }
            Backend::Avx512 => {
                run_avx512(
                    kind, p, input, weights, bias, packed, out, batch, first, count, store,
                );
                return;
            }
            _ => {}
        }
    }
    run_core::<4, false>(
        kind, p, input, weights, bias, packed, out, batch, first, count, store,
    );
}
macro_rules! target_run {
    ($name:ident,$features:literal,$lanes:literal,$fused:literal) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable=$features)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name(
            kind: Kind,
            p: &Conv2d,
            input: &[f32],
            weights: &[f32],
            bias: &[f32],
            packed: Option<&PackedMatrix>,
            out: &mut [f32],
            batch: usize,
            first: usize,
            count: usize,
            store: &impl simd::StoreTransform,
        ) {
            run_core::<$lanes, $fused>(
                kind, p, input, weights, bias, packed, out, batch, first, count, store,
            );
        }
    };
}
target_run!(run_avx, "avx", 8, false);
target_run!(run_avx_fma, "avx,fma", 8, true);
target_run!(run_avx2, "avx2,fma", 8, true);
target_run!(run_avx512, "avx512f,avx2,fma", 16, true);

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn run_core<const N: usize, const FMA: bool>(
    kind: Kind,
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    batch: usize,
    first: usize,
    count: usize,
    store: &impl simd::StoreTransform,
) {
    match kind {
        Kind::PointwiseBlocked => pointwise_working_set::<N, FMA, 128>(
            p,
            input,
            bias,
            packed.expect("selected panel"),
            out,
            batch,
            first,
            count,
            store,
        ),
        Kind::PointwiseTiled => pointwise_working_set::<N, FMA, 64>(
            p,
            input,
            bias,
            packed.expect("selected panel"),
            out,
            batch,
            first,
            count,
            store,
        ),
        Kind::PackedDepth9 => packed_depth9(
            p,
            input,
            weights,
            bias,
            packed.expect("selected panel"),
            out,
            batch,
            first,
            count,
            store,
        ),
        Kind::PointwiseOcMajor => pointwise_oc_major::<N, FMA>(
            p,
            input,
            bias,
            packed.expect("selected panel"),
            out,
            batch,
            first,
            count,
            store,
        ),
        Kind::PointwisePacked => packed_pointwise::<N, FMA>(
            p,
            input,
            bias,
            packed.expect("selected panel"),
            out,
            batch,
            first,
            count,
            store,
        ),
        Kind::Packed3Pair16 => packed_three_pair16::<FMA>(
            p,
            input,
            weights,
            bias,
            packed.expect("selected paired 3x3 requires packed coefficients"),
            out,
            batch,
            first,
            count,
            store,
        ),
        Kind::Packed3 => packed_three_eight::<N, FMA>(
            p,
            input,
            weights,
            bias,
            packed.expect("selected panel"),
            out,
            batch,
            first,
            count,
            store,
        ),
        Kind::PackedStride2 => stride2::execute::<N, FMA, false>(
            p,
            input,
            weights,
            bias,
            packed.expect("selected panel"),
            out,
            batch,
            first,
            count,
            store,
        ),
        Kind::PackedStride2Pair16 => {
            // Public selection and recursive native children use this only
            // in guarded eight-lane AVX/FMA entries. AVX512 always owns OC8.
            if N == 8 {
                stride2::execute::<8, FMA, true>(
                    p,
                    input,
                    weights,
                    bias,
                    packed.expect("selected panel"),
                    out,
                    batch,
                    first,
                    count,
                    store,
                );
            } else {
                unreachable!("packed stride2 paired16 requires the AVX spatial width");
            }
        }
        Kind::PackedDense => packed_dense::<N, FMA, 0, 0, 1>(
            p,
            input,
            weights,
            bias,
            packed.expect("selected panel"),
            out,
            batch,
            first,
            count,
            store,
        ),
        Kind::Depth3Tap => depthwise_tap_major::<N, FMA, 3, 3, 1, 1>(
            p, input, weights, bias, out, batch, first, count, store,
        ),
        Kind::Depth5Tap => depthwise_tap_major::<N, FMA, 5, 5, 1, 1>(
            p, input, weights, bias, out, batch, first, count, store,
        ),
        Kind::Depth3 => depthwise_strip::<N, FMA, 3, 3, 1, 1, 2>(
            p, input, weights, bias, out, batch, first, count, store,
        ),
        Kind::Depth5 => depthwise_strip::<N, FMA, 5, 5, 1, 1, 2>(
            p, input, weights, bias, out, batch, first, count, store,
        ),
        Kind::Depth7 => depthwise_strip::<N, FMA, 7, 7, 1, 1, 4>(
            p, input, weights, bias, out, batch, first, count, store,
        ),
        Kind::Depth9 => depthwise_strip::<N, FMA, 9, 9, 1, 1, 4>(
            p, input, weights, bias, out, batch, first, count, store,
        ),
        Kind::Depth1x5 => depthwise_tap_major::<N, FMA, 1, 5, 1, 1>(
            p, input, weights, bias, out, batch, first, count, store,
        ),
        Kind::DepthHeight3 => depthwise_tap_major::<N, FMA, 3, 3, 2, 1>(
            p, input, weights, bias, out, batch, first, count, store,
        ),
        Kind::DepthHeight5 => depthwise_tap_major::<N, FMA, 5, 5, 2, 1>(
            p, input, weights, bias, out, batch, first, count, store,
        ),
        Kind::DepthStride3 => depthwise_tap_major::<N, FMA, 3, 3, 2, 2>(
            p, input, weights, bias, out, batch, first, count, store,
        ),
        Kind::DepthGeneral => depthwise_strip::<N, FMA, 0, 0, 1, 1, 4>(
            p, input, weights, bias, out, batch, first, count, store,
        ),
        _ => {
            let mut channel = first;
            while channel < first + count {
                let group_end = ((channel / (p.output_channels / p.groups) + 1)
                    * (p.output_channels / p.groups))
                    .min(first + count);
                if channel + 16 <= group_end
                    && N == 8
                    && matches!(kind, Kind::Dense3 | Kind::PointwiseSpatial)
                {
                    spatial_kind::<N, FMA, 16>(
                        kind, p, input, weights, bias, out, batch, channel, first, store,
                    );
                    channel += 16;
                } else if channel + 8 <= group_end {
                    spatial_kind::<N, FMA, 8>(
                        kind, p, input, weights, bias, out, batch, channel, first, store,
                    );
                    channel += 8;
                } else if channel + 4 <= group_end {
                    spatial_kind::<N, FMA, 4>(
                        kind, p, input, weights, bias, out, batch, channel, first, store,
                    );
                    channel += 4;
                } else {
                    spatial_kind::<N, FMA, 1>(
                        kind, p, input, weights, bias, out, batch, channel, first, store,
                    );
                    channel += 1;
                }
            }
        }
    }
}
#[inline(always)]
fn madd<const N: usize, const FMA: bool>(
    a: Simd<f32, N>,
    b: Simd<f32, N>,
    sum: Simd<f32, N>,
) -> Simd<f32, N> {
    if FMA { a.mul_add(b, sum) } else { a * b + sum }
}
fn initial<const N: usize>(bias: &[f32], first: usize, cout: usize) -> Simd<f32, N> {
    Simd::from_array(std::array::from_fn(|lane| {
        if first + lane < cout && !bias.is_empty() {
            bias[first + lane]
        } else {
            0.
        }
    }))
}
#[inline(always)]
fn coefficient<const N: usize>(
    packed: &PackedMatrix,
    inner: usize,
    k: usize,
    first: usize,
    columns: usize,
) -> Simd<f32, N> {
    if first % 16 + N <= 16 && first + N <= columns {
        return Simd::from_slice(
            &packed.data()[((first / 16) * inner + k) * 16 + first % 16..][..N],
        );
    }
    Simd::from_array(std::array::from_fn(|lane| {
        let col = first + lane;
        if col < columns {
            packed.data()[((col / 16) * inner + k) * 16 + col % 16]
        } else {
            0.
        }
    }))
}
/// Small-plane OC-major microkernel. Register tiles reuse each coefficient
/// vector across eight/four positions, then transpose the bounded [plane][16]
/// temporary directly into the final planar output. The shared 16-column
/// coefficient panel is layout-equivalent to the reference OC-major format.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn pointwise_oc_major_impl<const N: usize, const FMA: bool>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    out: &mut [f32],
    batch: usize,
    first: usize,
    count: usize,
    store: &impl simd::StoreTransform,
) {
    let plane = p.height * p.width;
    assert!(plane < 48);
    // With one spatial value, channel-vector lanes are contiguous NCHW values.
    if plane == 1 {
        packed_pointwise_single::<N, FMA>(p, input, bias, packed, out, batch, first, count, store);
        return;
    }
    let mut tile = [0f32; 48 * 16];
    for co in (first..first + count).step_by(16) {
        let columns = (first + count - co).min(16);
        let mut spatial = 0;
        while spatial < plane {
            let rows = if spatial + 8 <= plane {
                8
            } else if spatial + 4 <= plane {
                4
            } else {
                1
            };
            for channel in (0..columns).step_by(N) {
                let active = (columns - channel).min(N);
                let mut sums = [initial::<N>(bias, co + channel, p.output_channels); 8];
                for ci in 0..p.input_channels {
                    let weights = coefficient::<N>(
                        packed,
                        p.input_channels,
                        ci,
                        co + channel,
                        p.output_channels,
                    );
                    let source = (batch * p.input_channels + ci) * plane + spatial;
                    for (row, sum) in sums[..rows].iter_mut().enumerate() {
                        *sum = madd::<N, FMA>(Simd::splat(input[source + row]), weights, *sum);
                    }
                }
                for (row, sum) in sums[..rows].iter().enumerate() {
                    tile[(spatial + row) * 16 + channel..][..active]
                        .copy_from_slice(&sum.as_array()[..active]);
                }
            }
            spatial += rows;
        }
        for channel in 0..columns {
            let local = (co + channel - first) * plane;
            let global = (batch * p.output_channels + co + channel) * plane;
            let mut spatial = 0;
            while spatial + N <= plane {
                let value = Simd::<f32, N>::from_array(std::array::from_fn(|lane| {
                    tile[(spatial + lane) * 16 + channel]
                }));
                store
                    .vector(value, global + spatial)
                    .copy_to_slice(&mut out[local + spatial..][..N]);
                spatial += N;
            }
            while spatial + 4 <= plane {
                let value = Simd::<f32, 4>::from_array(std::array::from_fn(|lane| {
                    tile[(spatial + lane) * 16 + channel]
                }));
                store
                    .vector(value, global + spatial)
                    .copy_to_slice(&mut out[local + spatial..][..4]);
                spatial += 4;
            }
            for spatial in spatial..plane {
                out[local + spatial] = store.scalar(tile[spatial * 16 + channel], global + spatial);
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_pointwise_impl<const N: usize, const FMA: bool>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    out: &mut [f32],
    batch: usize,
    first: usize,
    count: usize,
    store: &impl simd::StoreTransform,
) {
    let plane = p.height * p.width;
    if plane == 1 {
        packed_pointwise_single::<N, FMA>(p, input, bias, packed, out, batch, first, count, store);
        return;
    }
    if N == 8 && plane >= 16 && first.is_multiple_of(4) && count.is_multiple_of(4) {
        packed_pointwise_spatial::<FMA>(p, input, bias, packed, out, batch, first, count, store);
        return;
    }
    let inner = p.input_channels;
    for co in (first..first + count).step_by(N) {
        for x in (0..plane).step_by(4) {
            let rows = (plane - x).min(4);
            let bias = initial::<N>(bias, co, p.output_channels);
            if rows == 4 {
                let mut s0 = bias;
                let mut s1 = bias;
                let mut s2 = bias;
                let mut s3 = bias;
                for ci in 0..inner {
                    let w = coefficient::<N>(packed, inner, ci, co, p.output_channels);
                    let source = (batch * inner + ci) * plane + x;
                    s0 = madd::<N, FMA>(Simd::splat(input[source]), w, s0);
                    s1 = madd::<N, FMA>(Simd::splat(input[source + 1]), w, s1);
                    s2 = madd::<N, FMA>(Simd::splat(input[source + 2]), w, s2);
                    s3 = madd::<N, FMA>(Simd::splat(input[source + 3]), w, s3);
                }
                store_four_spatial(
                    s0,
                    s1,
                    s2,
                    s3,
                    out,
                    N.min(first + count - co),
                    plane,
                    (co - first) * plane + x,
                    (batch * p.output_channels + co) * plane + x,
                    store,
                );
                continue;
            }
            let mut sums = [bias; 4];
            for ci in 0..inner {
                let w = coefficient::<N>(packed, inner, ci, co, p.output_channels);
                for (row, sum) in sums[..rows].iter_mut().enumerate() {
                    let value = Simd::splat(input[(batch * inner + ci) * plane + x + row]);
                    *sum = madd::<N, FMA>(value, w, *sum);
                }
            }
            for (row, sum) in sums[..rows].iter().enumerate() {
                for lane in 0..N.min(first + count - co) {
                    out[(co + lane - first) * plane + x + row] = store.scalar(
                        sum[lane],
                        (batch * p.output_channels + co + lane) * plane + x + row,
                    );
                }
            }
        }
    }
}

/// OC4 x 16 spatial values keep NCHW loads/stores contiguous. Large input
/// planes traverse a bounded spatial tile before moving to the next OC4 block,
/// so all output blocks can reuse that input tile in cache. The existing OC16
/// panels are read directly; every output retains its ascending-ci FMA chain.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_pointwise_spatial<const FMA: bool>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    out: &mut [f32],
    batch: usize,
    first: usize,
    count: usize,
    store: &impl simd::StoreTransform,
) {
    let plane = p.height * p.width;
    let inner = p.input_channels;
    let tile_spatial = if count > 4 && inner.saturating_mul(plane) > 262_144 {
        ((49_152 / inner) & !15).max(64)
    } else {
        plane
    };
    for tile in (0..plane).step_by(tile_spatial) {
        let end = tile.saturating_add(tile_spatial).min(plane);
        for co in (first..first + count).step_by(4) {
            let mut x = tile;
            while x + 16 <= end {
                packed_pointwise_spatial_packet::<8, FMA, true>(
                    p, input, bias, packed, out, batch, first, co, x, store,
                );
                x += 16;
            }
            if x + 8 <= end {
                packed_pointwise_spatial_packet::<8, FMA, false>(
                    p, input, bias, packed, out, batch, first, co, x, store,
                );
                x += 8;
            }
            if x + 4 <= end {
                packed_pointwise_spatial_packet::<4, FMA, false>(
                    p, input, bias, packed, out, batch, first, co, x, store,
                );
                x += 4;
            }
            // Keep the pre-existing fused arithmetic even on scalar spatial
            // tails. Only completed contiguous packets use vector epilogues.
            for x in x..end {
                let mut sum = initial::<4>(bias, co, p.output_channels);
                for ci in 0..inner {
                    let weights = coefficient::<4>(packed, inner, ci, co, p.output_channels);
                    let value = Simd::splat(input[(batch * inner + ci) * plane + x]);
                    sum = madd::<4, FMA>(value, weights, sum);
                }
                for lane in 0..4 {
                    out[(co + lane - first) * plane + x] = store.scalar(
                        sum[lane],
                        (batch * p.output_channels + co + lane) * plane + x,
                    );
                }
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_pointwise_spatial_packet<const W: usize, const FMA: bool, const PAIR: bool>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    out: &mut [f32],
    batch: usize,
    first: usize,
    co: usize,
    x: usize,
    store: &impl simd::StoreTransform,
) {
    let plane = p.height * p.width;
    let sums = pointwise_spatial::packet::<W, FMA, PAIR>(p, input, bias, packed, batch, co, x);
    let local = (co - first) * plane + x;
    let global = (batch * p.output_channels + co) * plane + x;
    for (lane, [sum, next]) in sums.into_iter().enumerate() {
        store
            .vector(sum, global + lane * plane)
            .copy_to_slice(&mut out[local + lane * plane..][..W]);
        if PAIR {
            store
                .vector(next, global + lane * plane + W)
                .copy_to_slice(&mut out[local + lane * plane + W..][..W]);
        }
    }
}

/// Plane-one NCHW has contiguous output channels, including narrow panels.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_pointwise_single<const N: usize, const FMA: bool>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    out: &mut [f32],
    batch: usize,
    first: usize,
    count: usize,
    store: &impl simd::StoreTransform,
) {
    for co in (first..first + count).step_by(N) {
        let active = (first + count - co).min(N);
        let mut sum = initial::<N>(bias, co, p.output_channels);
        for ci in 0..p.input_channels {
            let weights = coefficient::<N>(packed, p.input_channels, ci, co, p.output_channels);
            sum = madd::<N, FMA>(
                Simd::splat(input[batch * p.input_channels + ci]),
                weights,
                sum,
            );
        }
        let global = batch * p.output_channels + co;
        if active == N {
            store
                .vector(sum, global)
                .copy_to_slice(&mut out[co - first..][..N]);
        } else {
            for lane in 0..active {
                out[co - first + lane] = store.scalar(sum[lane], global + lane);
            }
        }
    }
}

/// Transpose four channel-vector sums into contiguous per-channel spatial values.
/// Existing ordered store operands retain their ordinary flat NCHW indexing.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn store_four_spatial<const N: usize>(
    s0: Simd<f32, N>,
    s1: Simd<f32, N>,
    s2: Simd<f32, N>,
    s3: Simd<f32, N>,
    out: &mut [f32],
    columns: usize,
    plane: usize,
    local: usize,
    global: usize,
    store: &impl simd::StoreTransform,
) {
    for lane in 0..columns {
        let value = Simd::<f32, 4>::from_array([s0[lane], s1[lane], s2[lane], s3[lane]]);
        store
            .vector(value, global + lane * plane)
            .copy_to_slice(&mut out[local + lane * plane..][..4]);
    }
}

#[allow(clippy::too_many_arguments)]
fn edge(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    batch: usize,
    co: usize,
    y: usize,
    x: usize,
) -> f32 {
    let cpg = p.input_channels / p.groups;
    let group = co / (p.output_channels / p.groups);
    let mut sum = if bias.is_empty() { 0. } else { bias[co] };
    for ci in 0..cpg {
        for ky in 0..p.kernel_height {
            let iy = (y * p.stride_height + ky) as isize - p.pad_top as isize;
            if iy < 0 || iy >= p.height as isize {
                continue;
            }
            for kx in 0..p.kernel_width {
                let ix = (x * p.stride_width + kx) as isize - p.pad_left as isize;
                if ix < 0 || ix >= p.width as isize {
                    continue;
                }
                sum += input[((batch * p.input_channels + group * cpg + ci) * p.height
                    + iy as usize)
                    * p.width
                    + ix as usize]
                    * weights[((co * cpg + ci) * p.kernel_height + ky) * p.kernel_width + kx];
            }
        }
    }
    sum
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_dense_impl<
    const N: usize,
    const FMA: bool,
    const KH: usize,
    const KW: usize,
    const STRIDE: usize,
>(
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
    let kh = if KH == 0 { p.kernel_height } else { KH };
    let kw = if KW == 0 { p.kernel_width } else { KW };
    let inner = p.input_channels * kh * kw;
    let plane = p.output_height * p.output_width;
    let valid = interior(p);
    for co in (first..first + count).step_by(N) {
        let columns = N.min(first + count - co);
        for y in 0..p.output_height {
            let ky_begin = p.pad_top.saturating_sub(y * STRIDE).min(kh);
            let ky_end = (p.height + p.pad_top).saturating_sub(y * STRIDE).min(kh);
            let mut x = 0;
            while x < valid.start {
                for lane in 0..columns {
                    out[(co + lane - first) * plane + y * p.output_width + x] = store.scalar(
                        edge(p, input, weights, bias, batch, co + lane, y, x),
                        (batch * p.output_channels + co + lane) * plane + y * p.output_width + x,
                    );
                }
                x += 1;
            }
            while x + 4 <= valid.end {
                let seed = initial::<N>(bias, co, p.output_channels);
                let mut s0 = seed;
                let mut s1 = seed;
                let mut s2 = seed;
                let mut s3 = seed;
                for ci in 0..p.input_channels {
                    for ky in ky_begin..ky_end {
                        for kx in 0..kw {
                            let w = coefficient::<N>(
                                packed,
                                inner,
                                (ci * kh + ky) * kw + kx,
                                co,
                                p.output_channels,
                            );
                            let source =
                                ((batch * p.input_channels + ci) * p.height + y * STRIDE + ky
                                    - p.pad_top)
                                    * p.width
                                    + x * STRIDE
                                    + kx
                                    - p.pad_left;
                            s0 = madd::<N, FMA>(Simd::splat(input[source]), w, s0);
                            s1 = madd::<N, FMA>(Simd::splat(input[source + STRIDE]), w, s1);
                            s2 = madd::<N, FMA>(Simd::splat(input[source + 2 * STRIDE]), w, s2);
                            s3 = madd::<N, FMA>(Simd::splat(input[source + 3 * STRIDE]), w, s3);
                        }
                    }
                }
                store_four_spatial(
                    s0,
                    s1,
                    s2,
                    s3,
                    out,
                    columns,
                    plane,
                    (co - first) * plane + y * p.output_width + x,
                    (batch * p.output_channels + co) * plane + y * p.output_width + x,
                    store,
                );
                x += 4;
            }
            for x in x..p.output_width {
                for lane in 0..columns {
                    out[(co + lane - first) * plane + y * p.output_width + x] = store.scalar(
                        edge(p, input, weights, bias, batch, co + lane, y, x),
                        (batch * p.output_channels + co + lane) * plane + y * p.output_width + x,
                    );
                }
            }
        }
    }
}

/// Pinned eight-output packed 3x3 family with full spatial-vector packets.
/// Packet width follows the guarded ISA; all image borders and packet tails
/// retain separate multiply/add, including FMA-capable policies.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_three_eight_impl<const N: usize, const FMA: bool>(
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
    // Public binding validated full lengths and group-one 3x3 same geometry.
    // Every exclusive channel band contains complete source OC8 panels.
    assert!(first.is_multiple_of(8) && count.is_multiple_of(8));
    let plane = p.height * p.width;
    let inner = p.input_channels * 9;
    for co in (first..first + count).step_by(8) {
        for y in 0..p.height {
            let mut x = 0;
            if y > 0 && y + 1 < p.height {
                let border = if N == 8 {
                    Some(reduce3::border::<1>(
                        p, input, bias, packed, batch, co, y, 0,
                    ))
                } else {
                    None
                };
                for lane in 0..8 {
                    out[(co + lane - first) * plane + y * p.width] = store.scalar(
                        border.map_or_else(
                            || edge(p, input, weights, bias, batch, co + lane, y, 0),
                            |values| values[lane],
                        ),
                        (batch * p.output_channels + co + lane) * plane + y * p.width,
                    );
                }
                x = 1;
                while x + N < p.width {
                    if N == 8 {
                        let sums =
                            reduce3::eight::<FMA, 1>(p, input, bias, packed, batch, co, y, x);
                        let local = (co - first) * plane + y * p.width + x;
                        let global = (batch * p.output_channels + co) * plane + y * p.width + x;
                        for (lane, value) in sums.into_iter().enumerate() {
                            store
                                .vector(value, global + lane * plane)
                                .copy_to_slice(&mut out[local + lane * plane..][..8]);
                        }
                        x += 8;
                        continue;
                    }
                    let mut s0 = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co] });
                    let mut s1 =
                        Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 1] });
                    let mut s2 =
                        Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 2] });
                    let mut s3 =
                        Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 3] });
                    let mut s4 =
                        Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 4] });
                    let mut s5 =
                        Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 5] });
                    let mut s6 =
                        Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 6] });
                    let mut s7 =
                        Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + 7] });
                    for ci in 0..p.input_channels {
                        for ky in 0..3 {
                            let row = ((batch * p.input_channels + ci) * p.height + y + ky - 1)
                                * p.width
                                + x
                                - 1;
                            for kx in 0..3 {
                                // x>0 && x+N<width proves complete source rows
                                // for all N input lanes and each kx0..2.
                                let value = Simd::<f32, N>::from_slice(&input[row + kx..][..N]);
                                // One immutable16 panel holds adjacent source OC8
                                // groups; select its lower or upper eight channels.
                                let k = (ci * 3 + ky) * 3 + kx;
                                let at = ((co / 16) * inner + k) * 16 + co % 16;
                                let w = &packed.data()[at..at + 8];
                                s0 = madd::<N, FMA>(value, Simd::splat(w[0]), s0);
                                s1 = madd::<N, FMA>(value, Simd::splat(w[1]), s1);
                                s2 = madd::<N, FMA>(value, Simd::splat(w[2]), s2);
                                s3 = madd::<N, FMA>(value, Simd::splat(w[3]), s3);
                                s4 = madd::<N, FMA>(value, Simd::splat(w[4]), s4);
                                s5 = madd::<N, FMA>(value, Simd::splat(w[5]), s5);
                                s6 = madd::<N, FMA>(value, Simd::splat(w[6]), s6);
                                s7 = madd::<N, FMA>(value, Simd::splat(w[7]), s7);
                            }
                        }
                    }
                    let local = (co - first) * plane + y * p.width + x;
                    let global = (batch * p.output_channels + co) * plane + y * p.width + x;
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
                    x += N;
                }
            }
            // Top/bottom rows and the right scalar tail deliberately use
            // separate multiply/add through the independent source-order edge.
            for x in x..p.width {
                let border = if N == 8 {
                    Some(reduce3::border::<1>(
                        p, input, bias, packed, batch, co, y, x,
                    ))
                } else {
                    None
                };
                for lane in 0..8 {
                    out[(co + lane - first) * plane + y * p.width + x] = store.scalar(
                        border.map_or_else(
                            || edge(p, input, weights, bias, batch, co + lane, y, x),
                            |values| values[lane],
                        ),
                        (batch * p.output_channels + co + lane) * plane + y * p.width + x,
                    );
                }
            }
        }
    }
}

/// Reference-eligible single-thread paired eight-output packed 3x3 panels.
/// A single bounded spatial-vector load feeds all sixteen output chains.
/// Interior geometry and scalar edges retain the pinned AVX family's arithmetic.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_three_pair16<const FMA: bool>(
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
    // Public binding validated full lengths and group-one 3x3 same geometry.
    // Each reference child owns complete OC8 bands; an eligible paired band
    // can start at the upper half of our shared immutable16 coefficient panel.
    assert!(first.is_multiple_of(8) && count.is_multiple_of(16));
    let plane = p.height * p.width;
    let inner = p.input_channels * 9;
    for co in (first..first + count).step_by(16) {
        let low = (co / 16) * inner * 16 + co % 16;
        let high = ((co + 8) / 16) * inner * 16 + (co + 8) % 16;
        for y in 0..p.height {
            let mut x = 0;
            if y > 0 && y + 1 < p.height {
                for lane in 0..16 {
                    out[(co + lane - first) * plane + y * p.width] = store.scalar(
                        edge(p, input, weights, bias, batch, co + lane, y, 0),
                        (batch * p.output_channels + co + lane) * plane + y * p.width,
                    );
                }
                x = 1;
                while x + 8 < p.width {
                    let mut s0 = Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co] });
                    let mut s1 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 1] });
                    let mut s2 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 2] });
                    let mut s3 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 3] });
                    let mut s4 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 4] });
                    let mut s5 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 5] });
                    let mut s6 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 6] });
                    let mut s7 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 7] });
                    let mut s8 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 8] });
                    let mut s9 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 9] });
                    let mut s10 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 10] });
                    let mut s11 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 11] });
                    let mut s12 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 12] });
                    let mut s13 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 13] });
                    let mut s14 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 14] });
                    let mut s15 =
                        Simd::<f32, 8>::splat(if bias.is_empty() { 0. } else { bias[co + 15] });
                    for ci in 0..p.input_channels {
                        for ky in 0..3 {
                            let row = ((batch * p.input_channels + ci) * p.height + y + ky - 1)
                                * p.width
                                + x
                                - 1;
                            for kx in 0..3 {
                                // x>0 && x+8<width proves all eight input lanes
                                // stay inside this complete source row for kx0..2.
                                let value = Simd::<f32, 8>::from_slice(&input[row + kx..][..8]);
                                // Borrow both source OC8 panels directly, including
                                // the cross-format boundary when first%16 is 8.
                                let k = (ci * 3 + ky) * 3 + kx;
                                let w0 = &packed.data()[low + k * 16..low + k * 16 + 8];
                                let w8 = &packed.data()[high + k * 16..high + k * 16 + 8];
                                s0 = madd::<8, FMA>(value, Simd::splat(w0[0]), s0);
                                s1 = madd::<8, FMA>(value, Simd::splat(w0[1]), s1);
                                s2 = madd::<8, FMA>(value, Simd::splat(w0[2]), s2);
                                s3 = madd::<8, FMA>(value, Simd::splat(w0[3]), s3);
                                s4 = madd::<8, FMA>(value, Simd::splat(w0[4]), s4);
                                s5 = madd::<8, FMA>(value, Simd::splat(w0[5]), s5);
                                s6 = madd::<8, FMA>(value, Simd::splat(w0[6]), s6);
                                s7 = madd::<8, FMA>(value, Simd::splat(w0[7]), s7);
                                s8 = madd::<8, FMA>(value, Simd::splat(w8[0]), s8);
                                s9 = madd::<8, FMA>(value, Simd::splat(w8[1]), s9);
                                s10 = madd::<8, FMA>(value, Simd::splat(w8[2]), s10);
                                s11 = madd::<8, FMA>(value, Simd::splat(w8[3]), s11);
                                s12 = madd::<8, FMA>(value, Simd::splat(w8[4]), s12);
                                s13 = madd::<8, FMA>(value, Simd::splat(w8[5]), s13);
                                s14 = madd::<8, FMA>(value, Simd::splat(w8[6]), s14);
                                s15 = madd::<8, FMA>(value, Simd::splat(w8[7]), s15);
                            }
                        }
                    }
                    let local = (co - first) * plane + y * p.width + x;
                    let global = (batch * p.output_channels + co) * plane + y * p.width + x;
                    store
                        .vector(s0, global)
                        .copy_to_slice(&mut out[local..][..8]);
                    store
                        .vector(s1, global + plane)
                        .copy_to_slice(&mut out[local + plane..][..8]);
                    store
                        .vector(s2, global + 2 * plane)
                        .copy_to_slice(&mut out[local + 2 * plane..][..8]);
                    store
                        .vector(s3, global + 3 * plane)
                        .copy_to_slice(&mut out[local + 3 * plane..][..8]);
                    store
                        .vector(s4, global + 4 * plane)
                        .copy_to_slice(&mut out[local + 4 * plane..][..8]);
                    store
                        .vector(s5, global + 5 * plane)
                        .copy_to_slice(&mut out[local + 5 * plane..][..8]);
                    store
                        .vector(s6, global + 6 * plane)
                        .copy_to_slice(&mut out[local + 6 * plane..][..8]);
                    store
                        .vector(s7, global + 7 * plane)
                        .copy_to_slice(&mut out[local + 7 * plane..][..8]);
                    store
                        .vector(s8, global + 8 * plane)
                        .copy_to_slice(&mut out[local + 8 * plane..][..8]);
                    store
                        .vector(s9, global + 9 * plane)
                        .copy_to_slice(&mut out[local + 9 * plane..][..8]);
                    store
                        .vector(s10, global + 10 * plane)
                        .copy_to_slice(&mut out[local + 10 * plane..][..8]);
                    store
                        .vector(s11, global + 11 * plane)
                        .copy_to_slice(&mut out[local + 11 * plane..][..8]);
                    store
                        .vector(s12, global + 12 * plane)
                        .copy_to_slice(&mut out[local + 12 * plane..][..8]);
                    store
                        .vector(s13, global + 13 * plane)
                        .copy_to_slice(&mut out[local + 13 * plane..][..8]);
                    store
                        .vector(s14, global + 14 * plane)
                        .copy_to_slice(&mut out[local + 14 * plane..][..8]);
                    store
                        .vector(s15, global + 15 * plane)
                        .copy_to_slice(&mut out[local + 15 * plane..][..8]);
                    x += 8;
                }
            }
            // Top/bottom rows and the right scalar tail deliberately use
            // separate multiply/add through the independent source-order edge.
            for x in x..p.width {
                for lane in 0..16 {
                    out[(co + lane - first) * plane + y * p.width + x] = store.scalar(
                        edge(p, input, weights, bias, batch, co + lane, y, x),
                        (batch * p.output_channels + co + lane) * plane + y * p.width + x,
                    );
                }
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn spatial_kind<const N: usize, const FMA: bool, const OUT: usize>(
    kind: Kind,
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
    match kind {
        Kind::PointwiseSpatial => spatial::<N, FMA, OUT, 1, 1, 1, 1>(
            p, input, weights, bias, out, batch, co, first, store,
        ),
        Kind::Dense3 => spatial::<N, FMA, OUT, 3, 3, 1, 1>(
            p, input, weights, bias, out, batch, co, first, store,
        ),
        #[cfg(target_arch = "x86_64")]
        Kind::PadEnd2 if N == 8 && OUT == 8 => {
            pad_end2::execute::<FMA>(p, input, weights, bias, out, batch, co, first, store);
        }
        Kind::PadEnd2 => spatial::<N, FMA, OUT, 2, 2, 1, 1>(
            p, input, weights, bias, out, batch, co, first, store,
        ),
        Kind::Stride2 => spatial::<N, FMA, OUT, 3, 3, 2, 2>(
            p, input, weights, bias, out, batch, co, first, store,
        ),
        _ => spatial::<N, FMA, OUT, 0, 0, 1, 1>(
            p, input, weights, bias, out, batch, co, first, store,
        ),
    }
}
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn spatial_impl<
    const N: usize,
    const FMA: bool,
    const OUT: usize,
    const KH: usize,
    const KW: usize,
    const SH: usize,
    const SW: usize,
>(
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
    let kh = if KH == 0 { p.kernel_height } else { KH };
    let kw = if KW == 0 { p.kernel_width } else { KW };
    let plane = p.output_height * p.output_width;
    let valid = interior(p);
    let cpg = p.input_channels / p.groups;
    let group = co / (p.output_channels / p.groups);
    for y in 0..p.output_height {
        let mut x = 0;
        while x < valid.start {
            for lane in 0..OUT {
                out[(co + lane - first) * plane + y * p.output_width + x] = store.scalar(
                    edge(p, input, weights, bias, batch, co + lane, y, x),
                    (batch * p.output_channels + co + lane) * plane + y * p.output_width + x,
                );
            }
            x += 1;
        }
        while x + N <= valid.end {
            let mut sums = std::array::from_fn::<_, OUT, _>(|lane| {
                Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + lane] })
            });
            for ci in 0..cpg {
                for ky in 0..kh {
                    let iy = (y * SH + ky) as isize - p.pad_top as isize;
                    if iy < 0 || iy >= p.height as isize {
                        continue;
                    }
                    for kx in 0..kw {
                        let source = ((batch * p.input_channels + group * cpg + ci) * p.height
                            + iy as usize)
                            * p.width
                            + x * SW
                            + kx
                            - p.pad_left;
                        let value = if SW == 1 {
                            Simd::from_slice(&input[source..source + N])
                        } else {
                            Simd::from_array(std::array::from_fn(|lane| input[source + lane * SW]))
                        };
                        for (lane, sum) in sums.iter_mut().enumerate() {
                            let w = weights[(((co + lane) * cpg + ci) * kh + ky) * kw + kx];
                            *sum = madd::<N, FMA>(value, Simd::splat(w), *sum);
                        }
                    }
                }
            }
            for (lane, sum) in sums.into_iter().enumerate() {
                store
                    .vector(
                        sum,
                        (batch * p.output_channels + co + lane) * plane + y * p.output_width + x,
                    )
                    .copy_to_slice(
                        &mut out[(co + lane - first) * plane + y * p.output_width + x..][..N],
                    );
            }
            x += N;
        }
        for x in x..p.output_width {
            for lane in 0..OUT {
                out[(co + lane - first) * plane + y * p.output_width + x] = store.scalar(
                    edge(p, input, weights, bias, batch, co + lane, y, x),
                    (batch * p.output_channels + co + lane) * plane + y * p.output_width + x,
                );
            }
        }
    }
}

/// Tap-major fallback streams one coefficient over each valid output row.
/// Partial accumulators live in the final plane. A position is transformed
/// only at its final valid tap, so epilogues need no additional output pass.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn depthwise_tap_major_impl<
    const N: usize,
    const FMA: bool,
    const KH: usize,
    const KW: usize,
    const SH: usize,
    const SW: usize,
>(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    out: &mut [f32],
    batch: usize,
    first: usize,
    count: usize,
    store: &impl simd::StoreTransform,
) {
    let plane = p.output_height * p.output_width;
    for co in first..first + count {
        let channel = &mut out[(co - first) * plane..][..plane];
        channel.fill(if bias.is_empty() { 0. } else { bias[co] });
        for ky in 0..KH {
            let y_begin = p
                .pad_top
                .saturating_sub(ky)
                .div_ceil(SH)
                .min(p.output_height);
            let y_end = (p.height + p.pad_top)
                .saturating_sub(ky)
                .div_ceil(SH)
                .min(p.output_height);
            for kx in 0..KW {
                let x_begin = p
                    .pad_left
                    .saturating_sub(kx)
                    .div_ceil(SW)
                    .min(p.output_width);
                let x_end = (p.width + p.pad_left)
                    .saturating_sub(kx)
                    .div_ceil(SW)
                    .min(p.output_width);
                let weight = weights[(co * KH + ky) * KW + kx];
                for y in y_begin..y_end {
                    let source = ((batch * p.input_channels + co) * p.height + y * SH + ky
                        - p.pad_top)
                        * p.width;
                    let last_y = (p.height + p.pad_top - y * SH).min(KH) - 1;
                    let mut x = x_begin;
                    let mut vector_end = x_end;
                    if SW == 2 {
                        let check_lanes = N.min(8);
                        while vector_end.saturating_sub(x) >= check_lanes
                            && 2 * (vector_end - 1) + kx + 2 * check_lanes - 2
                                >= p.width + p.pad_left
                        {
                            vector_end -= 1;
                        }
                    }
                    while x + N <= vector_end {
                        let offset = y * p.output_width + x;
                        let value = if SW == 1 {
                            Simd::from_slice(&input[source + x + kx - p.pad_left..][..N])
                        } else {
                            Simd::from_array(std::array::from_fn(|lane| {
                                input[source + (x + lane) * SW + kx - p.pad_left]
                            }))
                        };
                        let sum = madd::<N, FMA>(
                            value,
                            Simd::splat(weight),
                            Simd::from_slice(&channel[offset..][..N]),
                        );
                        let final_tap = std::array::from_fn::<_, N, _>(|lane| {
                            ky == last_y
                                && kx == (p.width + p.pad_left - (x + lane) * SW).min(KW) - 1
                        });
                        if final_tap.iter().all(|&last| last) {
                            store
                                .vector(sum, (batch * p.output_channels + co) * plane + offset)
                                .copy_to_slice(&mut channel[offset..][..N]);
                        } else if final_tap.iter().any(|&last| last) {
                            for lane in 0..N {
                                channel[offset + lane] = if final_tap[lane] {
                                    store.scalar(
                                        sum[lane],
                                        (batch * p.output_channels + co) * plane + offset + lane,
                                    )
                                } else {
                                    sum[lane]
                                };
                            }
                        } else {
                            sum.copy_to_slice(&mut channel[offset..][..N]);
                        }
                        x += N;
                    }
                    for x in x..x_end {
                        let offset = y * p.output_width + x;
                        let sum =
                            channel[offset] + input[source + x * SW + kx - p.pad_left] * weight;
                        channel[offset] =
                            if ky == last_y && kx == (p.width + p.pad_left - x * SW).min(KW) - 1 {
                                store.scalar(sum, (batch * p.output_channels + co) * plane + offset)
                            } else {
                                sum
                            };
                    }
                }
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn depthwise_strip_impl<
    const N: usize,
    const FMA: bool,
    const KH: usize,
    const KW: usize,
    const SH: usize,
    const SW: usize,
    const ROWS: usize,
>(
    p: &Conv2d,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    out: &mut [f32],
    batch: usize,
    first: usize,
    count: usize,
    store: &impl simd::StoreTransform,
) {
    let kh = if KH == 0 { p.kernel_height } else { KH };
    let kw = if KW == 0 { p.kernel_width } else { KW };
    let plane = p.output_height * p.output_width;
    let valid = interior(p);
    let complete_rows = SH == 1 && SW == 1 && matches!((KH, KW), (3, 3) | (5, 5) | (7, 7));
    for channel in first..first + count {
        for y0 in (0..p.output_height).step_by(ROWS) {
            let rows = (p.output_height - y0).min(ROWS);
            let initial = Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[channel] });
            let mut x = valid.start;
            for row in 0..rows {
                for x in 0..valid.start {
                    out[(channel - first) * plane + (y0 + row) * p.output_width + x] = store
                        .scalar(
                            edge(p, input, weights, bias, batch, channel, y0 + row, x),
                            (batch * p.output_channels + channel) * plane
                                + (y0 + row) * p.output_width
                                + x,
                        );
                }
            }
            while x + N <= valid.end {
                let mut sums = [initial; ROWS];
                for ky in 0..kh {
                    for kx in 0..kw {
                        let w = Simd::splat(weights[(channel * kh + ky) * kw + kx]);
                        for (row, sum) in sums[..rows].iter_mut().enumerate() {
                            if complete_rows
                                && ((y0 + row) * SH < p.pad_top
                                    || (y0 + row) * SH + kh > p.height + p.pad_top)
                            {
                                continue;
                            }
                            let iy = ((y0 + row) * SH + ky) as isize - p.pad_top as isize;
                            if iy < 0 || iy >= p.height as isize {
                                continue;
                            }
                            let source = ((batch * p.input_channels + channel) * p.height
                                + iy as usize)
                                * p.width
                                + x * SW
                                + kx
                                - p.pad_left;
                            let value = if SW == 1 {
                                Simd::from_slice(&input[source..source + N])
                            } else {
                                Simd::from_array(std::array::from_fn(|lane| {
                                    input[source + lane * SW]
                                }))
                            };
                            *sum = madd::<N, FMA>(value, w, *sum);
                        }
                    }
                }
                for (row, sum) in sums[..rows].iter().enumerate() {
                    if complete_rows
                        && ((y0 + row) * SH < p.pad_top
                            || (y0 + row) * SH + kh > p.height + p.pad_top)
                    {
                        for lane in 0..N {
                            let x = x + lane;
                            out[(channel - first) * plane + (y0 + row) * p.output_width + x] =
                                store.scalar(
                                    edge(p, input, weights, bias, batch, channel, y0 + row, x),
                                    (batch * p.output_channels + channel) * plane
                                        + (y0 + row) * p.output_width
                                        + x,
                                );
                        }
                        continue;
                    }
                    store
                        .vector(
                            *sum,
                            (batch * p.output_channels + channel) * plane
                                + (y0 + row) * p.output_width
                                + x,
                        )
                        .copy_to_slice(
                            &mut out[(channel - first) * plane + (y0 + row) * p.output_width + x..]
                                [..N],
                        );
                }
                x += N;
            }
            for row in 0..rows {
                for x in x..p.output_width {
                    out[(channel - first) * plane + (y0 + row) * p.output_width + x] = store
                        .scalar(
                            edge(p, input, weights, bias, batch, channel, y0 + row, x),
                            (batch * p.output_channels + channel) * plane
                                + (y0 + row) * p.output_width
                                + x,
                        );
                }
            }
        }
    }
}

/// Bounded K panels retain a spatial-major working set on the stack. Each
/// input panel is converted once per spatial tile and reused by every OC
/// block. Intermediate F32 stores preserve the original ascending-K order.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn pointwise_working_set_impl<const N: usize, const FMA: bool, const SPATIAL: usize>(
    p: &Conv2d,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    out: &mut [f32],
    batch: usize,
    first: usize,
    count: usize,
    store: &impl simd::StoreTransform,
) {
    const K: usize = 32;
    let plane = p.height * p.width;
    let mut input_tile = [0f32; 128 * K];
    for co in first..first + count {
        out[(co - first) * plane..(co - first + 1) * plane].fill(if bias.is_empty() {
            0.
        } else {
            bias[co]
        });
    }
    for spatial in (0..plane).step_by(SPATIAL) {
        let rows = (plane - spatial).min(SPATIAL);
        for reduction in (0..p.input_channels).step_by(K) {
            let inner = (p.input_channels - reduction).min(K);
            for ci in 0..inner {
                let source =
                    &input[(batch * p.input_channels + reduction + ci) * plane + spatial..][..rows];
                for (row, &value) in source.iter().enumerate() {
                    input_tile[row * K + ci] = value;
                }
            }
            for co in (first..first + count).step_by(N) {
                let columns = (first + count - co).min(N);
                for row in (0..rows).step_by(4) {
                    let active = (rows - row).min(4);
                    let mut sums = std::array::from_fn::<_, 4, _>(|r| {
                        Simd::<f32, N>::from_array(std::array::from_fn(|lane| {
                            if r < active && lane < columns {
                                out[(co + lane - first) * plane + spatial + row + r]
                            } else {
                                0.
                            }
                        }))
                    });
                    for ci in 0..inner {
                        let w = coefficient::<N>(
                            packed,
                            p.input_channels,
                            reduction + ci,
                            co,
                            p.output_channels,
                        );
                        for (r, sum) in sums[..active].iter_mut().enumerate() {
                            *sum = madd::<N, FMA>(
                                Simd::splat(input_tile[(row + r) * K + ci]),
                                w,
                                *sum,
                            );
                        }
                    }
                    for (r, sum) in sums[..active].iter().enumerate() {
                        for lane in 0..columns {
                            out[(co + lane - first) * plane + spatial + row + r] =
                                if reduction + inner == p.input_channels {
                                    store.scalar(
                                        sum[lane],
                                        (batch * p.output_channels + co + lane) * plane
                                            + spatial
                                            + row
                                            + r,
                                    )
                                } else {
                                    sum[lane]
                                };
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn packed_depth9(
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
    assert!(
        simd::avx2_fma_available(),
        "packed depthwise9 requires AVX2 and FMA"
    );
    #[cfg(target_arch = "x86_64")]
    // SAFETY: explicit runtime check guards AVX2 and FMA, including forged calls.
    unsafe {
        packed_depth9_avx2(
            p, input, weights, bias, packed, out, batch, first, count, store,
        );
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn packed_depth9_avx2(
    p: &Conv2d,
    input: &[f32],
    _weights: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    out: &mut [f32],
    batch: usize,
    first: usize,
    count: usize,
    store: &impl simd::StoreTransform,
) {
    use std::arch::x86_64::*;
    let plane = p.height * p.width;
    assert!(plane <= 4096 && count.is_multiple_of(8));
    for co in (first..first + count).step_by(8) {
        let source = &input[(batch * p.input_channels + co) * plane..][..8 * plane];
        for y in 0..p.height {
            for x in 0..p.width {
                let mut sum = initial::<8>(bias, co, p.output_channels);
                for ky in 0..9 {
                    let iy = (y + ky) as isize - 4;
                    if iy < 0 || iy >= p.height as isize {
                        continue;
                    }
                    for kx in 0..9 {
                        let ix = (x + kx) as isize - 4;
                        if ix < 0 || ix >= p.width as isize {
                            continue;
                        }
                        let spatial = iy as usize * p.width + ix as usize;
                        let offsets =
                            std::array::from_fn::<_, 8, _>(|lane| (lane * plane + spatial) as i32);
                        let mut values = [0f32; 8];
                        // SAFETY: every lane references one of eight complete,
                        // validated channel planes; offsets fit I32 (plane<=4096).
                        unsafe {
                            let indices = _mm256_loadu_si256(offsets.as_ptr().cast());
                            let values_v = _mm256_i32gather_ps::<4>(source.as_ptr(), indices);
                            _mm256_storeu_ps(values.as_mut_ptr(), values_v);
                        }
                        sum = madd::<8, true>(
                            Simd::from_array(values),
                            coefficient::<8>(packed, 81, ky * 9 + kx, co, p.output_channels),
                            sum,
                        );
                    }
                }
                for lane in 0..8 {
                    out[(co + lane - first) * plane + y * p.width + x] = store.scalar(
                        sum[lane],
                        (batch * p.output_channels + co + lane) * plane + y * p.width + x,
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_transpose(
    backend: Backend,
    p: &super::kernels::ConvTranspose,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    out: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
) -> bool {
    try_transpose_store(
        backend,
        p,
        input,
        weights,
        bias,
        out,
        cpu,
        threads,
        &simd::IdentityStore,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_transpose_store(
    backend: Backend,
    p: &super::kernels::ConvTranspose,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    out: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
    store: &impl simd::StoreTransform,
) -> bool {
    check_backend(backend);
    if backend == Backend::Scalar {
        return false;
    }
    let plane = p.height * p.width * 4;
    assert_eq!(input.len(), p.batch * p.cin * p.height * p.width);
    assert_eq!(weights.len(), p.cin * p.cout * 4);
    assert_eq!(out.len(), p.batch * p.cout * plane);
    assert!(bias.is_empty() || bias.len() == p.cout);
    #[cfg(feature = "parallel")]
    if threads > 1
        && p.batch == 1
        && p.cout >= 2
        && (p.cout * p.cin * p.height * p.width * 4) >= 8_000_000
    {
        use rayon::prelude::*;
        let channels = p.cout.div_ceil(threads.min(cpu.capacity()));
        cpu.install(|| {
            out.par_chunks_mut(channels * plane)
                .enumerate()
                .for_each(|(band, out)| {
                    transpose_run(
                        backend,
                        p,
                        input,
                        weights,
                        bias,
                        out,
                        0,
                        band * channels,
                        out.len() / plane,
                        store,
                    )
                })
        });
        return true;
    }
    #[cfg(not(feature = "parallel"))]
    let _ = (cpu, threads);
    for (batch, out) in out.chunks_exact_mut(p.cout * plane).enumerate() {
        transpose_run(
            backend, p, input, weights, bias, out, batch, 0, p.cout, store,
        );
    }
    true
}
#[allow(clippy::too_many_arguments)]
fn transpose_run(
    backend: Backend,
    p: &super::kernels::ConvTranspose,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    out: &mut [f32],
    batch: usize,
    first: usize,
    count: usize,
    store: &impl simd::StoreTransform,
) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: try_transpose validates the policy's runtime CPU features.
    unsafe {
        match backend {
            Backend::Avx => {
                transpose_avx(p, input, weights, bias, out, batch, first, count, store);
                return;
            }
            Backend::AvxFma => {
                transpose_avx_fma(p, input, weights, bias, out, batch, first, count, store);
                return;
            }
            Backend::Avx2Fma => {
                transpose_avx2(p, input, weights, bias, out, batch, first, count, store);
                return;
            }
            Backend::Avx512 => {
                transpose_avx512(p, input, weights, bias, out, batch, first, count, store);
                return;
            }
            _ => {}
        }
    }
    transpose_core::<4, false>(p, input, weights, bias, out, batch, first, count, store);
}
macro_rules! target_transpose {
    ($name:ident,$features:literal,$lanes:literal,$fused:literal) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable=$features)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name(
            p: &super::kernels::ConvTranspose,
            input: &[f32],
            weights: &[f32],
            bias: &[f32],
            out: &mut [f32],
            batch: usize,
            first: usize,
            count: usize,
            store: &impl simd::StoreTransform,
        ) {
            transpose_core::<$lanes, $fused>(
                p, input, weights, bias, out, batch, first, count, store,
            );
        }
    };
}
target_transpose!(transpose_avx, "avx", 8, false);
target_transpose!(transpose_avx_fma, "avx,fma", 8, true);
target_transpose!(transpose_avx2, "avx2,fma", 8, true);
target_transpose!(transpose_avx512, "avx512f,avx2,fma", 16, true);
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn transpose_core<const N: usize, const FMA: bool>(
    p: &super::kernels::ConvTranspose,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    out: &mut [f32],
    batch: usize,
    first: usize,
    count: usize,
    store: &impl simd::StoreTransform,
) {
    let mut co = first;
    while co < first + count {
        if co + 8 <= first + count {
            transpose_tile::<N, FMA, 8>(p, input, weights, bias, out, batch, co, first, store);
            co += 8;
        } else if co + 4 <= first + count {
            transpose_tile::<N, FMA, 4>(p, input, weights, bias, out, batch, co, first, store);
            co += 4;
        } else {
            transpose_tile::<N, FMA, 1>(p, input, weights, bias, out, batch, co, first, store);
            co += 1;
        }
    }
}
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn transpose_tile_impl<const N: usize, const FMA: bool, const OC: usize>(
    p: &super::kernels::ConvTranspose,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    out: &mut [f32],
    batch: usize,
    co: usize,
    first: usize,
    store: &impl simd::StoreTransform,
) {
    let in_plane = p.height * p.width;
    let out_width = p.width * 2;
    let plane = in_plane * 4;
    for y in 0..p.height {
        let mut x = 0;
        while x + N <= p.width {
            let mut sums = std::array::from_fn::<_, OC, _>(|lane| {
                [Simd::<f32, N>::splat(if bias.is_empty() { 0. } else { bias[co + lane] }); 4]
            });
            for ci in 0..p.cin {
                let source = (batch * p.cin + ci) * in_plane + y * p.width + x;
                let value = Simd::from_slice(&input[source..source + N]);
                for (lane, taps) in sums.iter_mut().enumerate() {
                    for (tap, sum) in taps.iter_mut().enumerate() {
                        *sum = madd::<N, FMA>(
                            value,
                            Simd::splat(weights[(ci * p.cout + co + lane) * 4 + tap]),
                            *sum,
                        );
                    }
                }
            }
            for (lane, taps) in sums.iter().enumerate() {
                for tap in 0..4 {
                    for i in 0..N {
                        out[(co + lane - first) * plane
                            + (y * 2 + tap / 2) * out_width
                            + (x + i) * 2
                            + tap % 2] = store.scalar(
                            taps[tap][i],
                            (batch * p.cout + co + lane) * plane
                                + (y * 2 + tap / 2) * out_width
                                + (x + i) * 2
                                + tap % 2,
                        );
                    }
                }
            }
            x += N;
        }
        for x in x..p.width {
            for lane in 0..OC {
                for tap in 0..4 {
                    let mut sum = if bias.is_empty() { 0. } else { bias[co + lane] };
                    for ci in 0..p.cin {
                        sum += input[(batch * p.cin + ci) * in_plane + y * p.width + x]
                            * weights[(ci * p.cout + co + lane) * 4 + tap];
                    }
                    out[(co + lane - first) * plane
                        + (y * 2 + tap / 2) * out_width
                        + x * 2
                        + tap % 2] = store.scalar(
                        sum,
                        (batch * p.cout + co + lane) * plane
                            + (y * 2 + tap / 2) * out_width
                            + x * 2
                            + tap % 2,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;

// Whole-family boundaries keep shape dispatch small while retaining native reductions.
native::outline! {
    fn spatial => spatial_impl <
        const N: usize,
        const FMA: bool,
        const OUT: usize,
        const KH: usize,
        const KW: usize,
        const SH: usize,
        const SW: usize,
    > (
        p: &Conv2d,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        out: &mut [f32],
        batch: usize,
        co: usize,
        first: usize,
    )
    store: Store
}

native::outline! {
    fn depthwise_tap_major => depthwise_tap_major_impl <
        const N: usize,
        const FMA: bool,
        const KH: usize,
        const KW: usize,
        const SH: usize,
        const SW: usize,
    > (
        p: &Conv2d,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        out: &mut [f32],
        batch: usize,
        first: usize,
        count: usize,
    )
    store: Store
}

native::outline! {
    fn depthwise_strip => depthwise_strip_impl <
        const N: usize,
        const FMA: bool,
        const KH: usize,
        const KW: usize,
        const SH: usize,
        const SW: usize,
        const ROWS: usize,
    > (
        p: &Conv2d,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        out: &mut [f32],
        batch: usize,
        first: usize,
        count: usize,
    )
    store: Store
}

native::outline! {
    #[cfg(feature = "parallel")]
    fn spatial_rows => spatial_rows_impl <
        const N: usize,
        const FMA: bool,
    > (
        p: &Conv2d,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        packed: Option<&PackedMatrix>,
        band: [Option<&mut [f32]>; 64],
        begin: usize,
        batch: usize,
        first: usize,
    )
    store: Store
}

native::outline! {
    #[cfg(feature = "parallel")]
    fn packed_rows => packed_rows_impl <
        const N: usize,
        const FMA: bool,
    > (
        p: &Conv2d,
        input: &[f32],
        bias: &[f32],
        packed: &PackedMatrix,
        band: [Option<&mut [f32]>; 64],
        begin: usize,
        batch: usize,
        first: usize,
    )
    store: Store
}

native::outline! {
    #[cfg(feature = "parallel")]
    fn packed_depth_rows => packed_depth_rows_impl <
        const N: usize,
        const FMA: bool,
    > (
        p: &Conv2d,
        input: &[f32],
        bias: &[f32],
        packed: &PackedMatrix,
        band: [Option<&mut [f32]>; 64],
        begin: usize,
        batch: usize,
        first: usize,
    )
    store: Store
}

native::outline! {
    fn pointwise_oc_major => pointwise_oc_major_impl <
        const N: usize,
        const FMA: bool,
    > (
        p: &Conv2d,
        input: &[f32],
        bias: &[f32],
        packed: &PackedMatrix,
        out: &mut [f32],
        batch: usize,
        first: usize,
        count: usize,
    )
    store: Store
}

native::outline! {
    fn packed_pointwise => packed_pointwise_impl <
        const N: usize,
        const FMA: bool,
    > (
        p: &Conv2d,
        input: &[f32],
        bias: &[f32],
        packed: &PackedMatrix,
        out: &mut [f32],
        batch: usize,
        first: usize,
        count: usize,
    )
    store: Store
}

native::outline! {
    fn packed_dense => packed_dense_impl <
        const N: usize,
        const FMA: bool,
        const KH: usize,
        const KW: usize,
        const STRIDE: usize,
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

native::outline! {
    fn packed_three_eight => packed_three_eight_impl <
        const N: usize,
        const FMA: bool,
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

native::outline! {
    fn pointwise_working_set => pointwise_working_set_impl <
        const N: usize,
        const FMA: bool,
        const SPATIAL: usize,
    > (
        p: &Conv2d,
        input: &[f32],
        bias: &[f32],
        packed: &PackedMatrix,
        out: &mut [f32],
        batch: usize,
        first: usize,
        count: usize,
    )
    store: Store
}

native::outline! {
    fn transpose_tile => transpose_tile_impl <
        const N: usize,
        const FMA: bool,
        const OC: usize,
    > (
        p: &crate::runtime::kernels::ConvTranspose,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        out: &mut [f32],
        batch: usize,
        co: usize,
        first: usize,
    )
    store: Store
}
