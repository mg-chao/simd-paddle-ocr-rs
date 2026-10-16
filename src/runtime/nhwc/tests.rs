use super::*;
mod performance;
fn backends() -> Vec<Backend> {
    let mut result = vec![Backend::Scalar, Backend::Portable4];
    for (available, backend) in [
        (simd::avx_available(), Backend::Avx),
        (simd::avx_fma_available(), Backend::AvxFma),
        (simd::avx2_fma_available(), Backend::Avx2Fma),
        (simd::avx512_available(), Backend::Avx512),
    ] {
        if available {
            result.push(backend);
        }
    }
    result
}

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i.wrapping_mul(37).wrapping_add(seed * 17)) % 101) as f32 / 101. - 0.5)
        .collect()
}

fn plan(
    dimensions: [usize; 5],
    kernel: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    pad: [usize; 4],
    groups: usize,
) -> Conv {
    let [batch, cin, cout, height, width] = dimensions;
    let [kh, kw] = kernel;
    let [sh, sw] = stride;
    let [dh, dw] = dilation;
    Conv {
        batch,
        cin,
        cout,
        height,
        width,
        oh: (height + pad[0] + pad[2] - (kh - 1) * dh - 1) / sh + 1,
        ow: (width + pad[1] + pad[3] - (kw - 1) * dw - 1) / sw + 1,
        cpg: cin / groups,
        opg: cout / groups,
        kh,
        kw,
        sh,
        sw,
        dh,
        dw,
        pt: pad[0],
        pl: pad[1],
    }
}

fn reference(p: &Conv, input: &[f32], weights: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut output = vec![0.; p.batch * p.oh * p.ow * p.cout];
    for (pixel, row) in output.chunks_exact_mut(p.cout).enumerate() {
        let n = pixel / (p.oh * p.ow);
        let y = pixel / p.ow % p.oh;
        let x = pixel % p.ow;
        for (co, value) in row.iter_mut().enumerate() {
            let mut sum = bias.get(co).copied().unwrap_or(0.);
            for ci in 0..p.cpg {
                for ky in 0..p.kh {
                    let sy = y * p.sh + ky * p.dh;
                    if sy < p.pt || sy - p.pt >= p.height {
                        continue;
                    }
                    for kx in 0..p.kw {
                        let sx = x * p.sw + kx * p.dw;
                        if sx < p.pl || sx - p.pl >= p.width {
                            continue;
                        }
                        let source = ((n * p.height + sy - p.pt) * p.width + sx - p.pl) * p.cin
                            + co / p.opg * p.cpg
                            + ci;
                        let weight = ((co * p.cpg + ci) * p.kh + ky) * p.kw + kx;
                        sum += input[source] * weights[weight];
                    }
                }
            }
            *value = sum;
        }
    }
    output
}

fn pack(p: &Conv, weights: &[f32]) -> Option<PackedMatrix> {
    let groups = p.cin / p.cpg;
    let depthwise = groups == p.cin && p.opg == 1;
    if groups != 1 && !depthwise {
        return None;
    }
    let inner = if depthwise {
        p.kh * p.kw
    } else {
        p.cin * p.kh * p.kw
    };
    let mut transposed = vec![0.; inner * p.cout];
    for co in 0..p.cout {
        for k in 0..inner {
            transposed[k * p.cout + co] = weights[co * inner + k];
        }
    }
    Some(PackedMatrix::new(&transposed, inner, p.cout).unwrap())
}

fn check(p: Conv) {
    let input = values(p.batch * p.cin * p.height * p.width, 3);
    let weights = values(p.cout * p.cpg * p.kh * p.kw, 5);
    let bias = values(p.cout, 7);
    let expected = reference(&p, &input, &weights, &bias);
    let packed = pack(&p, &weights);
    let backends = backends();
    for backend in backends {
        if backend == Backend::Avx2Fma && !simd::avx2_fma_available() {
            continue;
        }
        let mut scratch = vec![0.; 32 * p.cin * p.kh * p.kw];
        let mut output = vec![0.; expected.len()];
        conv_dispatch(
            backend,
            &p,
            &input,
            &weights,
            &bias,
            packed.as_ref(),
            &mut output,
            0,
            &mut scratch,
        );
        for (index, (&actual, &expected)) in output.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 5e-5 * (1. + expected.abs()),
                "{backend:?} pixel/channel {index}: {actual} != {expected}"
            );
        }
        // Model output bands may begin anywhere, including within a row and
        // near a batch boundary. Their arithmetic must match a whole call.
        for band in [1, 3, 7, 9, 31, 32, 33] {
            let mut sharded = vec![f32::NAN; expected.len()];
            for (chunk, destination) in sharded.chunks_mut(band * p.cout).enumerate() {
                conv_dispatch(
                    backend,
                    &p,
                    &input,
                    &weights,
                    &bias,
                    packed.as_ref(),
                    destination,
                    chunk * band,
                    &mut scratch,
                );
            }
            assert!(
                output
                    .iter()
                    .zip(&sharded)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{backend:?} band {band} differs"
            );
        }
    }
}

#[test]
fn pointwise_direct_matrix_and_row_tails() {
    for width in [7, 31, 32, 33] {
        check(plan(
            [2, 3, 17, 2, width],
            [1, 1],
            [1, 1],
            [1, 1],
            [0; 4],
            1,
        ));
        check(plan(
            [1, 17, 5, 1, width],
            [1, 1],
            [1, 1],
            [1, 1],
            [0; 4],
            1,
        ));
    }
}

#[test]
fn im2row_dense_handles_padding_strides_dilation_batches_and_tails() {
    for cout in [3, 16, 17, 32] {
        check(plan([2, 3, cout, 5, 17], [3, 3], [1, 1], [1, 1], [1; 4], 1));
        check(plan(
            [2, 5, cout, 9, 15],
            [2, 3],
            [2, 2],
            [2, 1],
            [1, 2, 0, 1],
            1,
        ));
    }
    check(plan([2, 6, 10, 7, 13], [3, 3], [1, 2], [1, 1], [1; 4], 2));
}

#[test]
fn depthwise_tiles_share_packed_weights_without_crossing_rows() {
    for channels in [3, 8, 17, 32] {
        for width in [3, 9, 17, 33] {
            check(plan(
                [2, channels, channels, 4, width],
                [3, 3],
                [1, 1],
                [1, 1],
                [1; 4],
                channels,
            ));
        }
        check(plan(
            [2, channels, channels, 7, 19],
            [5, 5],
            [2, 2],
            [1, 1],
            [2; 4],
            channels,
        ));
        check(plan(
            [1, channels, channels, 5, 17],
            [3, 3],
            [1, 1],
            [2, 2],
            [2; 4],
            channels,
        ));
        check(plan(
            [1, channels, channels, 3, 9],
            [9, 9],
            [1, 1],
            [1, 1],
            [4; 4],
            channels,
        ));
    }
}

#[test]
fn absent_bias_and_unpacked_weights_keep_scalar_fallback_valid() {
    for p in [
        plan([2, 3, 17, 3, 9], [1, 1], [1, 1], [1, 1], [0; 4], 1),
        plan([2, 3, 17, 3, 9], [3, 3], [1, 1], [1, 1], [1; 4], 1),
        plan([2, 17, 17, 3, 9], [5, 5], [1, 1], [1, 1], [2; 4], 17),
    ] {
        let input = values(p.batch * p.cin * p.height * p.width, 11);
        let weights = values(p.cout * p.cpg * p.kh * p.kw, 13);
        let expected = reference(&p, &input, &weights, &[]);
        let mut actual = vec![0.; expected.len()];
        let mut workspace = vec![0.; 32 * p.cin * p.kh * p.kw];
        for backend in backends() {
            if backend == Backend::Avx2Fma && !simd::avx2_fma_available() {
                continue;
            }
            conv_dispatch(
                backend,
                &p,
                &input,
                &weights,
                &[],
                None,
                &mut actual,
                0,
                &mut workspace,
            );
            for (&actual, &expected) in actual.iter().zip(&expected) {
                assert!((actual - expected).abs() <= 5e-5 * (1. + expected.abs()));
            }
        }
    }
}

#[test]
fn channel_classifier_preserves_batched_left_gate_and_rank_one_meaning() {
    let full = Shape::new(&[2, 17, 3, 5]).unwrap();
    for n in [1, 2] {
        let gate = Shape::new(&[n, 17, 1, 1]).unwrap();
        for left in [false, true] {
            let (a, b) = if left { (gate, full) } else { (full, gate) };
            let channel = kernels::Broadcast::new(a, b, full).channel().unwrap();
            assert_eq!(channel.channels, 17);
            assert_eq!(channel.plane, 15);
            assert_eq!(channel.batched, n == 2);
            assert_eq!(channel.gate_left, left);
        }
    }
    assert!(
        kernels::Broadcast::new(full, Shape::new(&[5]).unwrap(), full)
            .channel()
            .is_none()
    );
}

#[test]
fn packed_transpose_preserves_bias_first_tails_batches_and_exclusive_row_bands() {
    for (cin, cout, height, width) in [
        (3, 16, 1, 1),
        (17, 17, 3, 7),
        (64, 24, 3, 9),
        (256, 64, 2, 33),
    ] {
        let p = kernels::ConvTranspose {
            batch: 2,
            cin,
            cout,
            height,
            width,
        };
        let input = values(2 * cin * height * width, 23);
        let weights = values(cin * cout * 4, 29);
        let bias = values(cout, 31);
        let panels = std::array::from_fn(|tap| {
            let matrix = (0..cin * cout)
                .map(|i| weights[i * 4 + tap])
                .collect::<Vec<_>>();
            PackedMatrix::new(&matrix, cin, cout).unwrap()
        });
        for bias in [bias.as_slice(), &[]] {
            let mut expected = vec![0.0; 2 * height * width * cout * 4];
            transpose_scalar(&p, &input, &weights, bias, &mut expected);
            for backend in backends() {
                if backend == Backend::Avx2Fma && !simd::avx2_fma_available() {
                    continue;
                }
                let mut actual = vec![f32::NAN; expected.len()];
                let mut workspace = vec![0.0; 32 * cout];
                transpose_packed_rows(
                    backend,
                    &p,
                    &input,
                    bias,
                    &panels,
                    &mut actual,
                    0,
                    &mut workspace,
                );
                for (&a, &b) in actual.iter().zip(&expected) {
                    if backend.uses_fma() {
                        assert!((a - b).abs() <= 5e-5 * (1.0 + b.abs()));
                    } else {
                        assert_eq!(a.to_bits(), b.to_bits());
                    }
                }
                for rows in [1, 2, 3] {
                    let mut split = vec![f32::NAN; expected.len()];
                    for (band, out) in split.chunks_mut(rows * 4 * width * cout).enumerate() {
                        transpose_packed_rows(
                            backend,
                            &p,
                            &input,
                            bias,
                            &panels,
                            out,
                            band * rows,
                            &mut workspace,
                        );
                    }
                    assert!(
                        actual
                            .iter()
                            .zip(&split)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                }
            }
        }
    }
}

#[test]
fn dense_interior_preserves_order_across_borders_dilation_and_worker_bands() {
    for (dimensions, kernel, stride, dilation, padding) in [
        ([2, 3, 17, 11, 21], [3, 3], [1, 1], [2, 2], [3, 3, 3, 3]),
        ([2, 57, 17, 5, 8], [3, 3], [1, 1], [1, 1], [1, 1, 1, 1]),
        ([2, 57, 19, 5, 8], [3, 3], [1, 1], [2, 2], [2, 2, 2, 2]),
        ([1, 3, 17, 1, 8], [3, 3], [1, 1], [1, 1], [4, 4, 4, 4]),
        ([1, 3, 19, 1, 1], [3, 3], [1, 1], [1, 1], [1, 1, 1, 1]),
        ([1, 21, 17, 3, 8], [5, 5], [1, 1], [1, 1], [2, 2, 2, 2]),
        ([2, 7, 17, 7, 8], [3, 3], [2, 2], [1, 1], [1, 1, 1, 1]),
        ([2, 65, 17, 9, 23], [3, 3], [1, 1], [1, 1], [1, 1, 1, 1]),
        ([1, 5, 33, 17, 29], [5, 5], [2, 1], [1, 1], [2, 1, 0, 3]),
        ([1, 1, 9, 7, 13], [7, 7], [1, 1], [1, 1], [3, 3, 3, 3]),
        ([1, 32, 32, 9, 13], [7, 7], [1, 1], [1, 1], [3, 3, 3, 3]),
        ([2, 11, 17, 11, 17], [7, 7], [2, 1], [2, 2], [6, 6, 6, 6]),
    ] {
        let p = plan(dimensions, kernel, stride, dilation, padding, 1);
        let mut input = values(p.batch * p.height * p.width * p.cin, 101);
        let mut weights = values(p.cout * p.cin * p.kh * p.kw, 103);
        if p.cin == 57 && p.width == 8 {
            // Channel-major reduction must preserve cancellation through the
            // 512-K split, including border +0 products and dilated taps.
            for (index, value) in input.iter_mut().enumerate() {
                *value = [16_777_216.0, 1.0, -16_777_216.0][index % p.cin % 3];
            }
            for (index, value) in weights.iter_mut().enumerate() {
                *value = if (index / (p.cin * p.kh * p.kw)).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
            }
        }
        let bias = values(p.cout, 107);
        let packed = pack(&p, &weights).unwrap();
        for backend in backends() {
            let mut expected = vec![0.; p.batch * p.oh * p.ow * p.cout];
            for (pixel, dst) in expected.chunks_exact_mut(p.cout).enumerate() {
                let (n, y, x) = (pixel / (p.oh * p.ow), pixel / p.ow % p.oh, pixel % p.ow);
                for (co, dst) in dst.iter_mut().enumerate() {
                    let mut sum = bias[co];
                    for ci in 0..p.cin {
                        for ky in 0..p.kh {
                            for kx in 0..p.kw {
                                let sy = y * p.sh + ky * p.dh;
                                let sx = x * p.sw + kx * p.dw;
                                let value = if sy >= p.pt
                                    && sy - p.pt < p.height
                                    && sx >= p.pl
                                    && sx - p.pl < p.width
                                {
                                    input[((n * p.height + sy - p.pt) * p.width + sx - p.pl)
                                        * p.cin
                                        + ci]
                                } else {
                                    0.
                                };
                                let w = weights[((co * p.cin + ci) * p.kh + ky) * p.kw + kx];
                                sum = if backend.uses_fma() {
                                    value.mul_add(w, sum)
                                } else {
                                    value * w + sum
                                };
                            }
                        }
                    }
                    *dst = sum;
                }
            }
            let pixels = p.batch * p.oh * p.ow;
            for band in [1, 2, 3, 4, 5, 7, pixels] {
                let mut actual = vec![f32::NAN; expected.len()];
                for (chunk, dst) in actual.chunks_mut(band * p.cout).enumerate() {
                    dense::execute(
                        backend,
                        &p,
                        &input,
                        &bias,
                        &packed,
                        dst,
                        chunk * band,
                        &IdentityStore,
                    );
                }
                for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{backend:?} {dimensions:?} band={band} index={index}"
                    );
                }
            }
        }
    }
}

#[test]
fn six_row_panel_groups_preserve_im2row_bits_and_512_reduction_boundaries() {
    for (cin, kernel) in [(56, 3), (57, 3), (513, 1)] {
        let p = plan(
            [2, cin, 17, 5, 13],
            [kernel, kernel],
            [1, 1],
            [1, 1],
            [kernel / 2; 4],
            1,
        );
        let input = values(p.batch * p.cin * p.height * p.width, 41);
        let weights = values(p.cout * p.cin * p.kh * p.kw, 43);
        let bias = values(p.cout, 47);
        let packed = pack(&p, &weights).unwrap();
        for backend in backends() {
            let mut expected = vec![0.; p.batch * p.oh * p.ow * p.cout];
            let mut scratch = vec![0.; 32 * p.cin * p.kh * p.kw];
            dense_im2row(
                backend,
                &p,
                &input,
                &bias,
                &packed,
                &mut expected,
                0,
                &mut scratch,
                &IdentityStore,
            );
            let mut actual = vec![0.; expected.len()];
            dense::execute(
                backend,
                &p,
                &input,
                &bias,
                &packed,
                &mut actual,
                0,
                &IdentityStore,
            );
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{backend:?} cin={cin} kernel={kernel}"
            );
            // A short band starts between two six-row tiles and crosses a
            // batch boundary; grouping must not alter any reduction.
            let mut split = vec![0.; expected.len()];
            for (band, out) in split.chunks_mut(7 * p.cout).enumerate() {
                dense::execute(
                    backend,
                    &p,
                    &input,
                    &bias,
                    &packed,
                    out,
                    band * 7,
                    &IdentityStore,
                );
            }
            assert!(
                actual
                    .iter()
                    .zip(&split)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
}

#[test]
fn large_depthwise_strips_preserve_each_pixels_tap_order_and_channel_tails() {
    fn check_strip<const LANES: usize, const FUSED: bool>(
        p: &Conv,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        packed: &PackedMatrix,
    ) {
        let pixels = p.batch * p.oh * p.ow;
        let mut expected = vec![0.; pixels * p.cout];
        let vector_channels = p.cout / LANES * LANES;
        for pixel in 0..pixels {
            let (n, y, x) = (pixel / (p.oh * p.ow), pixel / p.ow % p.oh, pixel % p.ow);
            for channel in (0..vector_channels).step_by(LANES) {
                depthwise_pixels::<LANES, FUSED, 1>(
                    p,
                    input,
                    weights,
                    bias,
                    Some(packed),
                    &mut expected,
                    pixel,
                    n,
                    y,
                    x,
                    channel,
                    0,
                    &IdentityStore,
                );
            }
            for channel in vector_channels..p.cout {
                depthwise_pixels::<1, false, 1>(
                    p,
                    input,
                    weights,
                    bias,
                    Some(packed),
                    &mut expected,
                    pixel,
                    n,
                    y,
                    x,
                    channel,
                    0,
                    &IdentityStore,
                );
            }
        }
        for band in [3, 17, pixels] {
            let mut actual = vec![0.; expected.len()];
            for (chunk, out) in actual.chunks_mut(band * p.cout).enumerate() {
                strip::execute::<LANES, FUSED>(
                    p,
                    input,
                    weights,
                    bias,
                    Some(packed),
                    out,
                    chunk * band,
                    &IdentityStore,
                );
            }
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "LANES={LANES} FUSED={FUSED} band={band}"
            );
        }
    }
    for (kernel, stride, dilation) in [(6, 1, 1), (7, 2, 1), (9, 1, 2)] {
        let p = plan(
            [2, 35, 35, 19, 23],
            [kernel, kernel],
            [stride, 1],
            [dilation, dilation],
            [kernel * dilation / 2; 4],
            35,
        );
        let input = values(p.batch * p.cin * p.height * p.width, 53);
        let weights = values(p.cout * p.kh * p.kw, 59);
        let bias = values(p.cout, 61);
        let packed = pack(&p, &weights).unwrap();
        check_strip::<4, false>(&p, &input, &weights, &bias, &packed);
        check_strip::<8, true>(&p, &input, &weights, &bias, &packed);
    }
}

#[test]
fn channel_pool_and_mean_vectors_match_logical_nchw_oracles() {
    for channels in [1, 3, 4, 7, 8, 9, 17, 33] {
        let shape = Shape::new(&[2, channels, 6, 10]).unwrap();
        let input = values(shape.element_count(), 67);
        let mut packed = vec![0.; input.len()];
        for n in 0..2 {
            simd::transpose_matrix(
                &input[n * channels * 60..(n + 1) * channels * 60],
                &mut packed[n * channels * 60..(n + 1) * channels * 60],
                channels,
                60,
            );
        }
        for (output_shape, kind) in [
            kernels::bind_max_pool(shape).unwrap(),
            kernels::bind_average_pool(shape).unwrap(),
            kernels::bind_reduce(shape, &[2, 3], true, false).unwrap(),
        ] {
            let mut canonical = vec![0.; output_shape.element_count()];
            kernels::execute(&kind, &[TensorId(0)], |_| input.as_slice(), &mut canonical);
            let mut expected = vec![0.; canonical.len()];
            let plane = canonical.len() / (2 * channels);
            for n in 0..2 {
                simd::transpose_matrix(
                    &canonical[n * channels * plane..(n + 1) * channels * plane],
                    &mut expected[n * channels * plane..(n + 1) * channels * plane],
                    channels,
                    plane,
                );
            }
            for backend in backends() {
                let mut actual = vec![f32::NAN; expected.len()];
                misc::execute(backend, &kind, &packed, shape, &mut actual);
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{backend:?} channels={channels}"
                );
            }
        }
    }
}

#[test]
fn single_output_transpose_uses_bias_first_pixel_vectors_and_disjoint_rows() {
    for (cin, width) in [(1, 1), (3, 7), (17, 8), (17, 9), (17, 17)] {
        let p = kernels::ConvTranspose {
            batch: 2,
            cin,
            cout: 1,
            height: 3,
            width,
        };
        let input = values(2 * 3 * width * cin, 71);
        let weights = values(cin * 4, 73);
        let bias = [-0.125];
        for backend in backends() {
            let mut expected = vec![0.; 2 * 3 * width * 4];
            for row in 0..6 {
                for x in 0..width {
                    for tap in 0..4 {
                        let mut sum = bias[0];
                        for ci in 0..cin {
                            let value = input[(row * width + x) * cin + ci];
                            let weight = weights[ci * 4 + tap];
                            sum = if backend.uses_fma() {
                                value.mul_add(weight, sum)
                            } else {
                                value * weight + sum
                            };
                        }
                        expected[(row * 2 + tap / 2) * width * 2 + x * 2 + tap % 2] = sum;
                    }
                }
            }
            for rows in [1, 2, 6] {
                let mut actual = vec![f32::NAN; expected.len()];
                for (band, out) in actual.chunks_mut(rows * 4 * width).enumerate() {
                    transpose::single(
                        backend,
                        &p,
                        &input,
                        &weights,
                        &bias,
                        out,
                        band * rows,
                        &IdentityStore,
                    );
                }
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
    }
}

#[test]
fn batch_norm_affine_uses_reserved_multipliers_and_preserves_channel_tails() {
    for channels in [1, 3, 4, 7, 8, 9, 17, 33] {
        let pixels = 13;
        let input = values(pixels * channels, 79);
        let scale = values(channels, 83);
        let bias = values(channels, 89);
        let mean = values(channels, 97);
        let variance = values(channels, 101)
            .into_iter()
            .map(|x| x.abs() + 0.25)
            .collect::<Vec<_>>();
        let mut expected = vec![0.; input.len()];
        for pixel in 0..pixels {
            for c in 0..channels {
                let multiplier = scale[c] / (variance[c] + 0.001).sqrt();
                expected[pixel * channels + c] =
                    (input[pixel * channels + c] - mean[c]) * multiplier + bias[c];
            }
        }
        for backend in backends() {
            let mut scratch = vec![f32::NAN; channels];
            let mut actual = vec![0.; expected.len()];
            misc::batch_norm(
                backend,
                &input,
                &mut actual,
                channels,
                &scale,
                &bias,
                &mean,
                &variance,
                0.001,
                &mut scratch,
            );
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
}

#[test]
fn dense_border_keeps_padding_products_and_signed_zero() {
    // 7x7 exercises the largest checked descriptor; 9x9 must retain the
    // general gather. Dilation must preserve padding products in both paths.
    for (kernel, dilation) in [(3, 1), (7, 1), (7, 2), (9, 1), (9, 2)] {
        let p = plan(
            [1, 1, 17, 1, 1],
            [kernel, kernel],
            [1, 1],
            [dilation, dilation],
            [(kernel - 1) * dilation / 2; 4],
            1,
        );
        let taps = p.kh * p.kw;
        for backend in backends() {
            let input = [-0.0];
            let weights = vec![1.0; p.cout * taps];
            let bias = vec![-0.0; p.cout];
            let packed = pack(&p, &weights).unwrap();
            let mut output = vec![f32::NAN; p.cout];
            dense::execute(
                backend,
                &p,
                &input,
                &bias,
                &packed,
                &mut output,
                0,
                &IdentityStore,
            );
            // Padding products are +0, so they change a negative-zero seed to +0.
            // Skipping all padded K entries would leave -0 for the lone valid tap.
            assert!(
                output.iter().all(|x| x.to_bits() == 0),
                "{backend:?} kernel={kernel} dilation={dilation}"
            );
            let input = [1.0];
            let mut weights = vec![1.0; p.cout * taps];
            for row in weights.chunks_exact_mut(taps) {
                row[0] = f32::INFINITY;
            }
            let packed = pack(&p, &weights).unwrap();
            dense::execute(
                backend,
                &p,
                &input,
                &bias,
                &packed,
                &mut output,
                0,
                &IdentityStore,
            );
            // The first padded +0*Inf is still evaluated; skipping it gives finite1.
            assert!(
                output.iter().all(|x| x.is_nan()),
                "{backend:?} kernel={kernel} dilation={dilation}"
            );
        }
    }
}
