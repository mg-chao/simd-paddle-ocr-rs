use super::*;

#[path = "audit_pair16.rs"]
mod audit_pair16;

#[path = "audit_packed3_envelope.rs"]
mod audit_packed3_envelope;

#[path = "audit_packed3_packets.rs"]
mod audit_packed3_packets;

/// Separate callback counters expose loss of vector epilogues even when both
/// callback implementations produce numerically identical results.
struct PacketVisits {
    scalar: Vec<std::sync::atomic::AtomicUsize>,
    vector: Vec<std::sync::atomic::AtomicUsize>,
    owned: std::ops::Range<usize>,
    plane: usize,
}
impl PacketVisits {
    fn new(length: usize, owned: std::ops::Range<usize>, plane: usize) -> Self {
        let counters = || {
            (0..length)
                .map(|_| std::sync::atomic::AtomicUsize::new(0))
                .collect()
        };
        Self {
            scalar: counters(),
            vector: counters(),
            owned,
            plane,
        }
    }
    fn assert_coverage(&self, expected_vector: impl Fn(usize) -> bool) {
        use std::sync::atomic::Ordering::Relaxed;
        for index in 0..self.scalar.len() {
            let scalar = self.scalar[index].load(Relaxed);
            let vector = self.vector[index].load(Relaxed);
            let owned = self.owned.contains(&index);
            assert_eq!(
                scalar + vector,
                usize::from(owned),
                "final store visits at {index}"
            );
            if owned {
                assert_eq!(
                    vector,
                    usize::from(expected_vector(index)),
                    "contiguous packet coverage at {index}"
                );
            }
        }
    }
}
impl simd::StoreTransform for PacketVisits {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        assert!(
            self.owned.contains(&index),
            "scalar store outside owned output: {index}"
        );
        self.scalar[index].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        value
    }
    fn vector<const N: usize>(&self, value: Simd<f32, N>, index: usize) -> Simd<f32, N> {
        assert!(
            N >= 4,
            "a completed packet must contain at least four final values"
        );
        let end = index.checked_add(N).unwrap();
        assert!(
            index >= self.owned.start && end <= self.owned.end,
            "vector store outside owned output: {index}..{end}"
        );
        if self.plane > 1 {
            assert_eq!(
                index / self.plane,
                (end - 1) / self.plane,
                "spatial packet crossed a channel plane"
            );
        }
        for item in &self.vector[index..end] {
            item.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        value
    }
}

/// Delta weights make every expected output a direct input lookup, independent
/// of both convolution reduction and StoreTransform implementations.
fn packed_packet_case(p: Conv2d, kind: Kind, vector_spatial: impl Fn(usize, usize) -> bool + Copy) {
    let plane = p.output_height * p.output_width;
    let input: Vec<_> = (0..p.batch * p.input_channels * p.height * p.width)
        .map(|index| (index % 127 + 1) as f32 / 64.)
        .collect();
    let mut weights =
        vec![0.; p.output_channels * p.input_channels * p.kernel_height * p.kernel_width];
    for channel in 0..p.output_channels {
        weights[((channel * p.input_channels + channel % p.input_channels) * p.kernel_height
            + p.kernel_height / 2)
            * p.kernel_width
            + p.kernel_width / 2] = 1.;
    }
    let packed = panels(&weights, &p);
    let length = p.batch * p.output_channels * plane;
    let expected = |index: usize| {
        let batch = index / (p.output_channels * plane);
        let channel = index / plane % p.output_channels;
        let spatial = index % plane;
        let y = spatial / p.output_width * p.stride_height;
        let x = spatial % p.output_width * p.stride_width;
        input
            [((batch * p.input_channels + channel % p.input_channels) * p.height + y) * p.width + x]
    };
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 4 } else { 1 }).unwrap();
    for backend in backends() {
        for threads in [1, cpu.capacity()] {
            assert_eq!(
                select(backend, &p, true, threads),
                Some(kind),
                "{backend:?}"
            );
            let store = PacketVisits::new(length, 0..length, plane);
            let guard = f32::from_bits(0x7fc0_55aa);
            let mut output = vec![guard; length + 8];
            assert!(try_conv_store(
                backend,
                &p,
                &input,
                &weights,
                &[],
                Some(&packed),
                &mut output[4..4 + length],
                &cpu,
                threads,
                &store
            ));
            assert!(
                output[..4]
                    .iter()
                    .chain(&output[4 + length..])
                    .all(|v| v.to_bits() == guard.to_bits())
            );
            for (index, value) in output[4..4 + length].iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    expected(index).to_bits(),
                    "{backend:?} output {index}"
                );
            }
            // A direct production band covers nonzero batch/channel offsets
            // without inflating this fixture to a parallel-work threshold.
            let batch = 1;
            let first = if p.output_channels >= 20 {
                16
            } else if p.output_channels > 8 {
                8
            } else {
                4
            };
            let count = (p.output_channels - first).min(8);
            let begin = (batch * p.output_channels + first) * plane;
            let end = begin + count * plane;
            let band_store = PacketVisits::new(length, begin..end, plane);
            let mut band = vec![guard; count * plane + 8];
            run(
                selected_backend_for_threads(backend, &p, true, threads),
                kind,
                &p,
                &input,
                &weights,
                &[],
                Some(&packed),
                &mut band[4..4 + count * plane],
                batch,
                first,
                count,
                &band_store,
            );
            assert!(
                band[..4]
                    .iter()
                    .chain(&band[4 + count * plane..])
                    .all(|v| v.to_bits() == guard.to_bits())
            );
            for (offset, value) in band[4..4 + count * plane].iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    expected(begin + offset).to_bits(),
                    "{backend:?} band {offset}"
                );
            }
            let actual = selected_backend_for_threads(backend, &p, true, threads);
            let lanes = match actual {
                Backend::Portable4 => 4,
                Backend::Avx512 => 16,
                _ => 8,
            };
            store.assert_coverage(|index| {
                if plane == 1 {
                    index % p.output_channels < p.output_channels / lanes * lanes
                } else {
                    vector_spatial(lanes, index % plane)
                }
            });
            band_store.assert_coverage(|index| {
                if plane == 1 {
                    index - begin < count / lanes * lanes
                } else {
                    vector_spatial(lanes, index % plane)
                }
            });
        }
    }
}

#[test]
fn packed_epilogues_deliver_spatial_packets_for_three_by_three() {
    for width in [1, 5, 6, 8, 9, 10, 17, 18, 33, 34] {
        let p = parameters(4, 24, 3, width, [3, 3], [1, 1], [1, 1], 1);
        // Pinned Vector/AVX/AVX512 loops start at x0, use one scalar
        // left border, then complete spatial packets only on interior rows.
        packed_packet_case(p, Kind::Packed3, |lanes, spatial| {
            let y = spatial / width;
            let x = spatial % width;
            let end = 1 + width.saturating_sub(2) / lanes * lanes;
            y > 0 && y + 1 < 3 && x > 0 && x < end
        });
    }
}
#[test]
fn packed_epilogues_deliver_spatial_packets_for_stride_two() {
    let p = parameters(4, 24, 5, 17, [3, 3], [2, 2], [1, 1], 1);
    packed_packet_case(p, Kind::PackedStride2, |_, spatial| {
        (1..5).contains(&(spatial % 9))
    });
}
#[test]
fn packed_epilogues_deliver_spatial_packets_for_pointwise() {
    for width in [49, 51, 53] {
        let p = parameters(4, 24, 1, width, [1, 1], [1, 1], [0, 0], 1);
        packed_packet_case(p, Kind::PointwisePacked, |_, spatial| {
            spatial < width / 4 * 4
        });
    }
}
#[test]
fn packed_epilogues_deliver_oc_major_packets_and_single_pixel_channels() {
    for width in [1, 7, 47] {
        let p = parameters(4, 32, 1, width, [1, 1], [1, 1], [0, 0], 1);
        packed_packet_case(p, Kind::PointwiseOcMajor, |_, spatial| {
            width == 1 || spatial < width / 4 * 4
        });
    }
}

#[test]
fn packed_epilogues_deliver_single_pixel_pointwise_panels_and_channel_tails() {
    for channels in [8, 12, 20] {
        let p = parameters(4, channels, 1, 1, [1, 1], [1, 1], [0, 0], 1);
        packed_packet_case(p, Kind::PointwisePacked, |_, _| true);
    }
}

#[allow(clippy::too_many_arguments)]
fn parameters(
    channels: usize,
    outputs: usize,
    height: usize,
    width: usize,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    groups: usize,
) -> Conv2d {
    Conv2d {
        batch: 2,
        input_channels: channels,
        output_channels: outputs,
        height,
        width,
        output_height: (height + 2 * padding[0] - kernel[0]) / stride[0] + 1,
        output_width: (width + 2 * padding[1] - kernel[1]) / stride[1] + 1,
        kernel_height: kernel[0],
        kernel_width: kernel[1],
        stride_height: stride[0],
        stride_width: stride[1],
        dilation_height: 1,
        dilation_width: 1,
        pad_top: padding[0],
        pad_left: padding[1],
        groups,
    }
}
fn panels(weights: &[f32], p: &Conv2d) -> PackedMatrix {
    let inner = weights.len() / p.output_channels;
    let mut transposed = vec![0.; weights.len()];
    for co in 0..p.output_channels {
        for k in 0..inner {
            transposed[k * p.output_channels + co] = weights[co * inner + k];
        }
    }
    PackedMatrix::new(&transposed, inner, p.output_channels).unwrap()
}
fn backends() -> impl Iterator<Item = Backend> {
    [
        Backend::Portable4,
        Backend::Avx,
        Backend::AvxFma,
        Backend::Avx2Fma,
        Backend::Avx512,
    ]
    .into_iter()
    .filter(|backend| match backend {
        Backend::Avx => simd::avx_available(),
        Backend::AvxFma => simd::avx_fma_available(),
        Backend::Avx2Fma => simd::avx2_fma_available(),
        Backend::Avx512 => simd::avx512_available(),
        _ => true,
    })
}
fn reference(
    p: &Conv2d,
    backend: Backend,
    kind: Kind,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let lanes = match backend {
        Backend::Portable4 => 4,
        Backend::Avx512 => 16,
        _ => 8,
    };
    let fused = matches!(
        backend,
        Backend::AvxFma | Backend::Avx2Fma | Backend::Avx512
    );
    let valid = interior(p);
    let vector_width = if matches!(kind, Kind::PackedStride2 | Kind::PackedDense) {
        4
    } else if kind == Kind::PackedDepth9 {
        1
    } else {
        lanes
    };
    let vector_end = valid.start + valid.len() / vector_width * vector_width;
    let mut output = Vec::new();
    let cpg = p.input_channels / p.groups;
    let opg = p.output_channels / p.groups;
    for batch in 0..p.batch {
        for co in 0..p.output_channels {
            for y in 0..p.output_height {
                for x in 0..p.output_width {
                    let do_fma = fused
                        && (!matches!(kind, Kind::Packed3 | Kind::Packed3Pair16)
                            || y > 0 && y + 1 < p.height)
                        && (matches!(
                            kind,
                            Kind::PointwiseOcMajor
                                | Kind::PackedDepth9
                                | Kind::PointwisePacked
                                | Kind::PointwiseBlocked
                                | Kind::PointwiseTiled
                        ) || (x >= valid.start && x < vector_end));
                    let mut sum = if bias.is_empty() { 0. } else { bias[co] };
                    for ci in 0..cpg {
                        for ky in 0..p.kernel_height {
                            for kx in 0..p.kernel_width {
                                let iy = (y * p.stride_height + ky) as isize - p.pad_top as isize;
                                let ix = (x * p.stride_width + kx) as isize - p.pad_left as isize;
                                if iy >= 0
                                    && iy < p.height as isize
                                    && ix >= 0
                                    && ix < p.width as isize
                                {
                                    let value =
                                        input[((batch * p.input_channels + (co / opg) * cpg + ci)
                                            * p.height
                                            + iy as usize)
                                            * p.width
                                            + ix as usize];
                                    let weight = weights[((co * cpg + ci) * p.kernel_height + ky)
                                        * p.kernel_width
                                        + kx];
                                    let tap_fma = if matches!(
                                        kind,
                                        Kind::Depth3Tap
                                            | Kind::Depth5Tap
                                            | Kind::Depth1x5
                                            | Kind::DepthHeight3
                                            | Kind::DepthHeight5
                                            | Kind::DepthStride3
                                    ) {
                                        let begin = p
                                            .pad_left
                                            .saturating_sub(kx)
                                            .div_ceil(p.stride_width)
                                            .min(p.output_width);
                                        let mut end = (p.width + p.pad_left)
                                            .saturating_sub(kx)
                                            .div_ceil(p.stride_width)
                                            .min(p.output_width);
                                        if p.stride_width == 2 {
                                            let check = lanes.min(8);
                                            while end.saturating_sub(begin) >= check
                                                && 2 * (end - 1) + kx + 2 * check - 2
                                                    >= p.width + p.pad_left
                                            {
                                                end -= 1;
                                            }
                                        }
                                        fused
                                            && x >= begin
                                            && x < begin + (end - begin) / lanes * lanes
                                    } else if matches!(
                                        kind,
                                        Kind::Depth3 | Kind::Depth5 | Kind::Depth7
                                    ) {
                                        do_fma
                                            && y >= p.pad_top
                                            && y + p.kernel_height <= p.height + p.pad_top
                                    } else {
                                        do_fma
                                    };
                                    sum = if tap_fma {
                                        value.mul_add(weight, sum)
                                    } else {
                                        value * weight + sum
                                    };
                                }
                            }
                        }
                    }
                    output.push(sum);
                }
            }
        }
    }
    output
}
fn compare(p: Conv2d, use_packed: bool) {
    let input: Vec<_> = (0..p.batch * p.input_channels * p.height * p.width)
        .map(|i| ((i * 37 % 43) as f32 - 21.) / 17.)
        .collect();
    let weights: Vec<_> =
        (0..p.output_channels * (p.input_channels / p.groups) * p.kernel_height * p.kernel_width)
            .map(|i| ((i * 19 % 29) as f32 - 14.) / 13.)
            .collect();
    let panel = use_packed.then(|| panels(&weights, &p));
    for backend in backends() {
        for with_bias in [false, true] {
            let bias: Vec<_> = if with_bias {
                (0..p.output_channels)
                    .map(|i| i as f32 / 19. - 0.3)
                    .collect()
            } else {
                vec![]
            };
            let Some(kind) = select(backend, &p, panel.is_some(), 1) else {
                continue;
            };
            let actual_backend = selected_backend_for_threads(backend, &p, panel.is_some(), 1);
            let expected = reference(&p, actual_backend, kind, &input, &weights, &bias);
            let mut actual = vec![0.; expected.len()];
            assert!(try_conv(
                backend,
                &p,
                &input,
                &weights,
                &bias,
                panel.as_ref(),
                &mut actual,
                &CpuContext::new(1).unwrap(),
                1
            ));
            for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{backend:?}/{kind:?}/{p:?}/index{i}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn pointwise_panels_spatial_groups_and_partial_channel_tiles_preserve_bias_order() {
    for outputs in [4, 8, 16, 24, 32] {
        for width in [3, 7, 15, 17, 49] {
            compare(
                parameters(5, outputs, 1, width, [1, 1], [1, 1], [0, 0], 1),
                true,
            );
        }
    }
    compare(parameters(6, 10, 3, 17, [1, 1], [1, 1], [0, 0], 2), false);
    compare(parameters(3, 7, 3, 17, [1, 1], [1, 1], [0, 0], 1), false);
}
#[test]
fn packed_and_unpacked_dense_kernels_preserve_edges_odd_widths_and_batches() {
    for width in [7, 8, 15, 17, 23] {
        for packed in [false, true] {
            compare(
                parameters(3, 8, 7, width, [3, 3], [1, 1], [1, 1], 1),
                packed,
            );
            compare(
                parameters(3, 8, 7, width, [3, 3], [2, 2], [1, 1], 1),
                packed,
            );
        }
    }
    let mut pad_end = parameters(3, 9, 7, 17, [2, 2], [1, 1], [0, 0], 1);
    pad_end.output_height = 7;
    pad_end.output_width = 17;
    compare(pad_end, false);
    compare(parameters(3, 9, 7, 23, [1, 7], [1, 1], [0, 3], 1), false);
}
#[test]
fn specialized_depthwise_strips_cover_constant_kernels_strides_and_tails() {
    for width in [9, 17, 33] {
        for kernel in [3, 5, 7, 9] {
            compare(
                parameters(
                    3,
                    3,
                    11,
                    width,
                    [kernel, kernel],
                    [1, 1],
                    [kernel / 2, kernel / 2],
                    3,
                ),
                false,
            );
        }
    }
    compare(parameters(3, 3, 7, 33, [1, 5], [1, 1], [0, 2], 3), false);
    compare(parameters(3, 3, 11, 33, [3, 3], [2, 1], [1, 1], 3), false);
    compare(parameters(3, 3, 11, 33, [5, 5], [2, 1], [2, 2], 3), false);
    compare(parameters(3, 3, 11, 33, [3, 3], [2, 2], [1, 1], 3), false);
    compare(parameters(3, 3, 11, 33, [1, 7], [1, 1], [0, 3], 3), false);
}

#[test]
fn selectors_reject_unreachable_shapes_and_keep_scalar_oracle() {
    let mut p = parameters(16, 8, 16, 16, [5, 5], [1, 1], [2, 2], 1);
    assert!(family(Backend::Scalar, &p, true).is_none());
    assert_eq!(
        family(Backend::Avx2Fma, &p, true),
        Some("nchw_dense_stride1")
    );
    p.batch = 8;
    assert_eq!(
        family_for_threads(Backend::Avx2Fma, &p, true, 1),
        Some("nchw_packed_dense_stride1")
    );
    assert_eq!(
        family_for_threads(Backend::Avx2Fma, &p, true, 2),
        Some("nchw_dense_stride1")
    );
    assert_eq!(
        selected_backend_for_threads(Backend::Avx512, &p, true, 1),
        Backend::Avx2Fma
    );
    p.dilation_width = 2;
    assert!(family(Backend::Avx2Fma, &p, true).is_none());
}

#[test]
fn transpose_eight_output_reuse_preserves_every_tap_batch_and_tail() {
    for width in [3, 4, 7, 8, 9, 17] {
        for cout in [1, 4, 8, 9, 16] {
            let p = super::super::kernels::ConvTranspose {
                batch: 2,
                cin: 3,
                cout,
                height: 3,
                width,
            };
            let input: Vec<_> = (0..p.batch * p.cin * p.height * p.width)
                .map(|i| ((i * 31 % 37) as f32 - 18.) / 19.)
                .collect();
            let weights: Vec<_> = (0..p.cin * p.cout * 4)
                .map(|i| ((i * 13 % 29) as f32 - 14.) / 17.)
                .collect();
            let bias: Vec<_> = (0..cout).map(|i| i as f32 / 23. - 0.7).collect();
            for backend in backends() {
                let lanes = match backend {
                    Backend::Portable4 => 4,
                    Backend::Avx512 => 16,
                    _ => 8,
                };
                let fused = matches!(
                    backend,
                    Backend::AvxFma | Backend::Avx2Fma | Backend::Avx512
                );
                let mut expected = vec![0.; p.batch * p.cout * p.height * p.width * 4];
                for batch in 0..p.batch {
                    for (co, &initial) in bias.iter().enumerate() {
                        for y in 0..p.height {
                            for x in 0..p.width {
                                for tap in 0..4 {
                                    let mut sum = initial;
                                    for ci in 0..p.cin {
                                        let value = input
                                            [((batch * p.cin + ci) * p.height + y) * p.width + x];
                                        let w = weights[(ci * p.cout + co) * 4 + tap];
                                        sum = if fused && x < p.width / lanes * lanes {
                                            value.mul_add(w, sum)
                                        } else {
                                            value * w + sum
                                        };
                                    }
                                    expected[((batch * p.cout + co) * p.height * 2
                                        + y * 2
                                        + tap / 2)
                                        * (p.width * 2)
                                        + x * 2
                                        + tap % 2] = sum;
                                }
                            }
                        }
                    }
                }
                let mut actual = vec![0.; expected.len()];
                assert!(try_transpose(
                    backend,
                    &p,
                    &input,
                    &weights,
                    &bias,
                    &mut actual,
                    &CpuContext::new(1).unwrap(),
                    1
                ));
                for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(a.to_bits(), e.to_bits(), "{backend:?}/{width}/{cout}/{i}");
                }
            }
        }
    }
}

#[test]
fn source_selected_packed9_and_pointwise_working_sets_keep_full_scalar_accumulation_contract() {
    if !simd::avx2_fma_available() {
        return;
    }
    for (p, kind) in [
        (
            parameters(24, 24, 17, 17, [9, 9], [1, 1], [4, 4], 24),
            Kind::PackedDepth9,
        ),
        (
            parameters(64, 64, 1, 1024, [1, 1], [1, 1], [0, 0], 1),
            Kind::PointwiseBlocked,
        ),
        (
            parameters(32, 16, 1, 8193, [1, 1], [1, 1], [0, 0], 1),
            Kind::PointwiseTiled,
        ),
    ] {
        let input: Vec<_> = (0..p.batch * p.input_channels * p.height * p.width)
            .map(|i| ((i * 37 % 43) as f32 - 21.) / 17.)
            .collect();
        let weights: Vec<_> = (0..p.output_channels
            * (p.input_channels / p.groups)
            * p.kernel_height
            * p.kernel_width)
            .map(|i| ((i * 19 % 29) as f32 - 14.) / 13.)
            .collect();
        let panel = panels(&weights, &p);
        let bias: Vec<_> = (0..p.output_channels)
            .map(|i| i as f32 / 19. - 0.3)
            .collect();
        assert_eq!(select(Backend::Avx2Fma, &p, true, 1), Some(kind));
        let expected = reference(&p, Backend::Avx2Fma, kind, &input, &weights, &bias);
        let mut actual = vec![0.; expected.len()];
        assert!(try_conv(
            Backend::Avx2Fma,
            &p,
            &input,
            &weights,
            &bias,
            Some(&panel),
            &mut actual,
            &CpuContext::new(1).unwrap(),
            1
        ));
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "{kind:?}/{i}: {a}!={e}");
        }
    }
}

#[test]
fn output_bands_are_bitwise_stable_across_cpu_budgets_and_reject_unavailable_policies() {
    if !simd::avx2_fma_available() {
        return;
    }
    let mut p = parameters(64, 64, 33, 33, [9, 9], [1, 1], [4, 4], 64);
    p.batch = 1;
    let input: Vec<_> = (0..p.input_channels * p.height * p.width)
        .map(|i| ((i * 37 % 43) as f32 - 21.) / 17.)
        .collect();
    let weights: Vec<_> = (0..p.output_channels * 81)
        .map(|i| ((i * 19 % 29) as f32 - 14.) / 13.)
        .collect();
    let panel = panels(&weights, &p);
    let bias = vec![0.3; p.output_channels];
    assert!(uses_pack_for_threads(Backend::Avx2Fma, &p, true, 1));
    let mut expected = vec![0.; input.len()];
    try_conv(
        Backend::Avx2Fma,
        &p,
        &input,
        &weights,
        &bias,
        Some(&panel),
        &mut expected,
        &CpuContext::new(1).unwrap(),
        1,
    );
    #[cfg(feature = "parallel")]
    let capacities = [1, 2, 4];
    #[cfg(not(feature = "parallel"))]
    let capacities = [1];
    for capacity in capacities {
        let mut actual = vec![0.; expected.len()];
        assert!(try_conv(
            Backend::Avx2Fma,
            &p,
            &input,
            &weights,
            &bias,
            Some(&panel),
            &mut actual,
            &CpuContext::new(capacity).unwrap(),
            capacity
        ));
        assert_eq!(actual, expected);
    }
    if !simd::avx512_available() {
        assert!(
            std::panic::catch_unwind(|| {
                let mut actual = vec![0.; expected.len()];
                try_conv(
                    Backend::Avx512,
                    &p,
                    &input,
                    &weights,
                    &bias,
                    Some(&panel),
                    &mut actual,
                    &CpuContext::new(1).unwrap(),
                    1,
                );
            })
            .is_err()
        );
    }
}

struct IndexedStore {
    visits: Vec<std::sync::atomic::AtomicUsize>,
}
impl IndexedStore {
    fn new(length: usize) -> Self {
        Self {
            visits: (0..length)
                .map(|_| std::sync::atomic::AtomicUsize::new(0))
                .collect(),
        }
    }
    fn expected(value: f32, index: usize) -> f32 {
        (value + ((index % 17) as f32 - 8.) / 31.).max(0.)
    }
    fn assert_once(&self) {
        for (index, count) in self.visits.iter().enumerate() {
            assert_eq!(
                count.load(std::sync::atomic::Ordering::Relaxed),
                1,
                "store visits at {index}"
            );
        }
    }
}
impl simd::StoreTransform for IndexedStore {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        self.visits[index].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self::expected(value, index)
    }
    fn vector<const N: usize>(&self, value: Simd<f32, N>, index: usize) -> Simd<f32, N> {
        Simd::from_array(std::array::from_fn(|lane| {
            self.scalar(value[lane], index + lane)
        }))
    }
}

#[test]
fn canonical_epilogues_transform_every_final_output_once_with_global_indices() {
    let mut pad_end = parameters(3, 9, 7, 17, [2, 2], [1, 1], [0, 0], 1);
    pad_end.output_height = pad_end.height;
    pad_end.output_width = pad_end.width;
    let mut cases = vec![
        (parameters(5, 32, 1, 47, [1, 1], [1, 1], [0, 0], 1), true),
        (parameters(5, 24, 3, 17, [1, 1], [1, 1], [0, 0], 1), true),
        (parameters(6, 10, 5, 17, [1, 1], [1, 1], [0, 0], 2), false),
        (parameters(3, 16, 5, 17, [3, 3], [1, 1], [1, 1], 1), true),
        (parameters(3, 9, 5, 17, [3, 3], [1, 1], [1, 1], 1), false),
        (parameters(3, 16, 7, 19, [3, 3], [2, 2], [1, 1], 1), true),
        (pad_end, false),
        (parameters(3, 3, 1, 1, [5, 5], [1, 1], [2, 2], 3), false),
        (parameters(3, 3, 7, 17, [1, 5], [1, 1], [0, 2], 3), false),
        (parameters(3, 3, 7, 19, [3, 3], [2, 1], [1, 1], 3), false),
        (parameters(3, 3, 9, 19, [5, 5], [2, 1], [2, 2], 3), false),
        (parameters(3, 3, 9, 33, [3, 3], [2, 2], [1, 1], 3), false),
        (parameters(5, 5, 9, 19, [5, 5], [1, 1], [2, 2], 5), false),
        (parameters(24, 24, 17, 17, [9, 9], [1, 1], [4, 4], 24), true),
        (parameters(64, 64, 32, 32, [1, 1], [1, 1], [0, 0], 1), true),
        (parameters(32, 16, 1, 8193, [1, 1], [1, 1], [0, 0], 1), true),
    ];
    let mut sharded = parameters(64, 64, 17, 33, [3, 3], [1, 1], [1, 1], 1);
    sharded.batch = 1;
    cases.push((sharded, true));
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 4 } else { 1 }).unwrap();
    for (p, packed) in cases {
        let input: Vec<_> = (0..p.batch * p.input_channels * p.height * p.width)
            .map(|i| (i % 37) as f32 / 31. - 0.5)
            .collect();
        let weights: Vec<_> = (0..p.output_channels
            * (p.input_channels / p.groups)
            * p.kernel_height
            * p.kernel_width)
            .map(|i| (i % 19) as f32 / 17. - 0.5)
            .collect();
        let bias: Vec<_> = (0..p.output_channels).map(|i| i as f32 / 31.).collect();
        let panel = packed.then(|| panels(&weights, &p));
        for backend in backends() {
            for threads in [1, 4] {
                let length = p.batch * p.output_channels * p.output_height * p.output_width;
                let mut plain = vec![0.; length];
                let mut fused = vec![0.; length];
                assert!(try_conv(
                    backend,
                    &p,
                    &input,
                    &weights,
                    &bias,
                    panel.as_ref(),
                    &mut plain,
                    &cpu,
                    threads
                ));
                let store = IndexedStore::new(length);
                assert!(try_conv_store(
                    backend,
                    &p,
                    &input,
                    &weights,
                    &bias,
                    panel.as_ref(),
                    &mut fused,
                    &cpu,
                    threads,
                    &store
                ));
                store.assert_once();
                for (index, (&actual, &raw)) in fused.iter().zip(&plain).enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        IndexedStore::expected(raw, index).to_bits(),
                        "{backend:?} {index} {}x{}",
                        p.kernel_height,
                        p.kernel_width
                    );
                }
            }
        }
    }
}

#[test]
fn transpose_epilogue_indices_cover_interleaved_taps_batches_and_channel_bands() {
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 4 } else { 1 }).unwrap();
    for batch in [1, 2] {
        let p = super::super::kernels::ConvTranspose {
            batch,
            cin: 5,
            cout: 17,
            height: 3,
            width: 19,
        };
        let input: Vec<_> = (0..batch * p.cin * p.height * p.width)
            .map(|i| (i % 23) as f32 / 19. - 0.5)
            .collect();
        let weights: Vec<_> = (0..p.cin * p.cout * 4)
            .map(|i| (i % 17) as f32 / 13. - 0.5)
            .collect();
        let bias: Vec<_> = (0..p.cout).map(|i| i as f32 / 41.).collect();
        for backend in backends() {
            let length = batch * p.cout * p.height * p.width * 4;
            let mut plain = vec![0.; length];
            let mut fused = vec![0.; length];
            assert!(try_transpose(
                backend, &p, &input, &weights, &bias, &mut plain, &cpu, 4
            ));
            let store = IndexedStore::new(length);
            assert!(try_transpose_store(
                backend, &p, &input, &weights, &bias, &mut fused, &cpu, 4, &store
            ));
            store.assert_once();
            for (index, (&actual, &raw)) in fused.iter().zip(&plain).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    IndexedStore::expected(raw, index).to_bits()
                );
            }
        }
    }
}

#[test]
fn row_shard_guards_and_safe_eight_channel_bands_preserve_exact_stores() {
    let mut dense = parameters(16, 8, 40, 32, [5, 5], [1, 1], [2, 2], 1);
    dense.batch = 1;
    let mut pad_end = parameters(64, 8, 32, 64, [2, 2], [1, 1], [0, 0], 1);
    pad_end.batch = 1;
    pad_end.output_height = pad_end.height;
    pad_end.output_width = pad_end.width;
    assert!(spatial_shard_eligible(Backend::Avx2Fma, &dense, false, 4));
    assert!(spatial_shard_eligible(Backend::Avx, &pad_end, false, 4));
    assert!(!spatial_shard_eligible(Backend::Avx, &dense, false, 4));
    assert!(!spatial_shard_eligible(Backend::Avx2Fma, &dense, false, 1));
    let mut below = dense;
    below.height = 39;
    below.output_height = 39;
    assert!(!spatial_shard_eligible(Backend::Avx2Fma, &below, false, 4));
    let mut batch = dense;
    batch.batch = 2;
    assert!(!spatial_shard_eligible(Backend::Avx2Fma, &batch, false, 4));
    let mut narrow = dense;
    narrow.output_channels = 16;
    narrow.output_height = 7;
    assert!(!spatial_shard_eligible(Backend::Avx2Fma, &narrow, false, 4));
    for plane in [47, 48] {
        let p = parameters(5, 16, 1, plane, [1, 1], [1, 1], [0, 0], 1);
        assert_eq!(
            select(Backend::Avx2Fma, &p, true, 1),
            Some(if plane == 47 {
                Kind::PointwiseOcMajor
            } else {
                Kind::PointwisePacked
            })
        );
    }
    if !simd::avx2_fma_available() {
        return;
    }
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 4 } else { 1 }).unwrap();
    let mut pointwise = parameters(32, 68, 1, 4097, [1, 1], [1, 1], [0, 0], 1);
    pointwise.batch = 1;
    assert!(spatial_shard_eligible(
        Backend::Avx2Fma,
        &pointwise,
        true,
        4
    ));
    for p in [dense, pad_end, pointwise] {
        let input: Vec<_> = (0..p.input_channels * p.height * p.width)
            .map(|i| (i % 37) as f32 / 31. - 0.5)
            .collect();
        let weights: Vec<_> =
            (0..p.output_channels * p.input_channels * p.kernel_height * p.kernel_width)
                .map(|i| (i % 19) as f32 / 17. - 0.5)
                .collect();
        let bias: Vec<_> = (0..p.output_channels).map(|i| i as f32 / 31.).collect();
        let panel = (p.kernel_height == 1).then(|| panels(&weights, &p));
        let length = p.output_channels * p.output_height * p.output_width;
        let mut plain = vec![0.; length];
        assert!(try_conv(
            Backend::Avx2Fma,
            &p,
            &input,
            &weights,
            &bias,
            panel.as_ref(),
            &mut plain,
            &cpu,
            1
        ));
        for threads in [2, 4] {
            let store = IndexedStore::new(length);
            let mut fused = vec![0.; length];
            assert!(try_conv_store(
                Backend::Avx2Fma,
                &p,
                &input,
                &weights,
                &bias,
                panel.as_ref(),
                &mut fused,
                &cpu,
                threads,
                &store
            ));
            store.assert_once();
            for (index, (&actual, &raw)) in fused.iter().zip(&plain).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    IndexedStore::expected(raw, index).to_bits(),
                    "threads{threads} index{index}"
                );
            }
        }
    }
}

#[test]
fn depthwise_source_selectors_and_tap_major_final_tap_contract_cover_thresholds() {
    let mut p = parameters(3, 3, 49, 10, [3, 3], [1, 1], [1, 1], 3);
    p.batch = 1;
    assert_eq!(
        select(Backend::Avx2Fma, &p, false, 1),
        Some(Kind::Depth3Tap)
    );
    p.height = 50;
    p.output_height = 50;
    assert_eq!(select(Backend::Avx2Fma, &p, false, 1), Some(Kind::Depth3));
    assert_eq!(
        select(Backend::Portable4, &p, false, 1),
        Some(Kind::Depth3Tap)
    );
    p.width = 9;
    p.output_width = 9;
    assert_eq!(
        select(Backend::Avx2Fma, &p, false, 1),
        Some(Kind::Depth3Tap)
    );
    let mut five = parameters(32, 32, 100, 100, [5, 5], [1, 1], [2, 2], 32);
    five.batch = 1;
    assert_eq!(work(&five), 8_000_000);
    assert_eq!(
        select(Backend::Avx2Fma, &five, false, 1),
        Some(Kind::Depth5)
    );
    assert_eq!(
        select(Backend::Avx2Fma, &five, false, 4),
        Some(Kind::Depth5Tap)
    );
    assert!(channel_shard_eligible(
        Backend::Avx2Fma,
        Kind::Depth5Tap,
        &five,
        4
    ));
    five.height = 99;
    five.output_height = 99;
    assert_eq!(
        select(Backend::Avx2Fma, &five, false, 4),
        Some(Kind::Depth5)
    );
    assert!(!channel_shard_eligible(
        Backend::Avx2Fma,
        Kind::Depth5,
        &five,
        4
    ));
    assert!(!channel_shard_eligible(
        Backend::Avx2Fma,
        Kind::Depth3,
        &p,
        4
    ));
    for kernel in [3, 5] {
        for height in [1, 2, 5] {
            for width in [1, 2, 7, 8, 9, 15, 16, 17] {
                compare(
                    parameters(
                        3,
                        3,
                        height,
                        width,
                        [kernel, kernel],
                        [1, 1],
                        [kernel / 2, kernel / 2],
                        3,
                    ),
                    false,
                );
            }
        }
    }
    for (kernel, height) in [(3, 50), (5, 10), (7, 9)] {
        let mut p = parameters(
            2,
            2,
            height,
            17,
            [kernel, kernel],
            [1, 1],
            [kernel / 2, kernel / 2],
            2,
        );
        p.batch = 1;
        compare(p, false);
    }
}

#[test]
fn sharded_depthwise_five_tap_major_matches_independent_per_tap_fma_oracle() {
    if !simd::avx2_fma_available() {
        return;
    }
    let mut p = parameters(32, 32, 100, 100, [5, 5], [1, 1], [2, 2], 32);
    p.batch = 1;
    let input: Vec<_> = (0..p.input_channels * p.height * p.width)
        .map(|i| (i % 29) as f32 / 23. - 0.5)
        .collect();
    let weights: Vec<_> = (0..p.output_channels * 25)
        .map(|i| (i % 17) as f32 / 13. - 0.5)
        .collect();
    let bias: Vec<_> = (0..p.output_channels).map(|i| i as f32 / 41.).collect();
    let expected = reference(
        &p,
        Backend::Avx2Fma,
        Kind::Depth5Tap,
        &input,
        &weights,
        &bias,
    );
    let mut actual = vec![0.; expected.len()];
    let store = IndexedStore::new(actual.len());
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 4 } else { 1 }).unwrap();
    assert!(try_conv_store(
        Backend::Avx2Fma,
        &p,
        &input,
        &weights,
        &bias,
        None,
        &mut actual,
        &cpu,
        4,
        &store
    ));
    store.assert_once();
    for (index, (&value, &raw)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            value.to_bits(),
            IndexedStore::expected(raw, index).to_bits(),
            "tap-major at {index}"
        );
    }
}

#[test]
fn pad_end_two_forced_dispatch_keeps_ordered_reduction_borders_and_indexed_stores() {
    // The previous pad-end fixture used9 output channels. Complete8-channel
    // blocks at24 outputs independently cover the production native OUT8 path.
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 8 } else { 1 }).unwrap();
    for height in [1, 2, 3] {
        for width in [7, 8, 9, 15, 16, 17] {
            for channels in [1, 3, 33] {
                // Padding exists at the bottom/right ends. The valid-conv
                // fixture helper subtracts2 before this geometry can be set,
                // so construct the supported H=1 case directly.
                let p = Conv2d {
                    batch: 2,
                    input_channels: channels,
                    output_channels: 24,
                    height,
                    width,
                    output_height: height,
                    output_width: width,
                    kernel_height: 2,
                    kernel_width: 2,
                    stride_height: 1,
                    stride_width: 1,
                    dilation_height: 1,
                    dilation_width: 1,
                    pad_top: 0,
                    pad_left: 0,
                    groups: 1,
                };
                let cancellation_case = height == 2 && width == 9 && channels == 33;
                for cancellation in [false, true]
                    .into_iter()
                    .filter(|&value| !value || cancellation_case)
                {
                    let mut input: Vec<_> = (0..p.batch * channels * height * width)
                        .map(|index| ((index % 23) as f32 - 11.) / 17.)
                        .collect();
                    let mut weights: Vec<_> = (0..24 * channels * 4)
                        .map(|index| ((index % 19) as f32 - 9.) / 13.)
                        .collect();
                    let mut bias: Vec<_> = (0..24)
                        .map(|channel| (channel as f32 - 11.) / 19.)
                        .collect();
                    if cancellation {
                        input.fill(1.);
                        weights.fill(0.);
                        bias.fill(16_777_216.);
                        for channel in 0..24 {
                            let base = channel * channels * 4;
                            weights[base] = 1.;
                            weights[base + 4] = -16_777_216.;
                            weights[base + 8] = 1.;
                        }
                    }
                    let plane = height * width;
                    let length = p.batch * 24 * plane;
                    for backend in backends() {
                        let lanes = match backend {
                            Backend::Portable4 => 4,
                            Backend::Avx512 => 16,
                            _ => 8,
                        };
                        // Source contract: maximal full spatial vectors end
                        // before the final input column. Scalar borders/tails
                        // skip invalid taps and retain separate multiply/add.
                        let vector_end = (width - 1) / lanes * lanes;
                        let mut expected = vec![0u32; length];
                        for batch in 0..p.batch {
                            for channel in 0..24 {
                                for y in 0..height {
                                    for x in 0..width {
                                        let fused = backend.uses_fma() && x < vector_end;
                                        let mut sum = bias[channel];
                                        for ci in 0..channels {
                                            for ky in 0..2 {
                                                for kx in 0..2 {
                                                    if y + ky < height && x + kx < width {
                                                        let value = input[((batch * channels
                                                            + ci)
                                                            * height
                                                            + y
                                                            + ky)
                                                            * width
                                                            + x
                                                            + kx];
                                                        let weight =
                                                            weights[((channel * channels + ci)
                                                                * 2
                                                                + ky)
                                                                * 2
                                                                + kx];
                                                        sum = if fused {
                                                            value.mul_add(weight, sum)
                                                        } else {
                                                            value * weight + sum
                                                        };
                                                    }
                                                }
                                            }
                                        }
                                        if cancellation {
                                            assert_eq!(sum.to_bits(), 1f32.to_bits());
                                        }
                                        let index = (batch * 24 + channel) * plane + y * width + x;
                                        expected[index] =
                                            IndexedStore::expected(sum, index).to_bits();
                                    }
                                }
                            }
                        }
                        let mut first_bits = None;
                        for budget in [1, 2, 4, 8] {
                            let threads = budget.min(cpu.capacity());
                            assert_eq!(select(backend, &p, false, threads), Some(Kind::PadEnd2));
                            let store = IndexedStore::new(length);
                            let mut actual = vec![f32::NAN; length];
                            assert!(try_conv_store(
                                backend,
                                &p,
                                &input,
                                &weights,
                                &bias,
                                None,
                                &mut actual,
                                &cpu,
                                threads,
                                &store
                            ));
                            store.assert_once();
                            let bits: Vec<_> = actual.iter().map(|value| value.to_bits()).collect();
                            assert_eq!(
                                bits, expected,
                                "{backend:?} budget{budget} cin{channels} {height}x{width} cancellation{cancellation}"
                            );
                            if let Some(first) = &first_bits {
                                assert_eq!(
                                    &bits, first,
                                    "thread configuration changed result bits"
                                );
                            } else {
                                first_bits = Some(bits);
                            }
                        }
                    }
                }
            }
        }
    }
}
