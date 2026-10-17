use super::*;
use std::simd::num::SimdFloat;

fn values(length: usize, seed: usize) -> Vec<f32> {
    (0..length)
        .map(|i| ((i * 31 + seed * 17) % 103) as f32 / 79. - 0.625)
        .collect()
}

fn plan(cin: usize, cout: usize, kernel: usize, dilation: usize) -> Conv {
    let padding = (kernel - 1) * dilation / 2;
    Conv {
        batch: 2,
        cin,
        cout,
        height: 9,
        width: 17,
        oh: 9,
        ow: 17,
        cpg: cin,
        opg: cout,
        kh: kernel,
        kw: kernel,
        sh: 1,
        sw: 1,
        dh: dilation,
        dw: dilation,
        pt: padding,
        pl: padding,
    }
}

fn pack(p: &Conv, weights: &[f32]) -> PackedMatrix {
    let inner = p.cin * p.kh * p.kw;
    let matrix = (0..inner * p.cout)
        .map(|index| weights[index % p.cout * inner + index / p.cout])
        .collect::<Vec<_>>();
    PackedMatrix::new(&matrix, inner, p.cout).unwrap()
}

// Independent logical convolution, with bias first and an ascending
// IC/KY/KX FMA chain. Production readers and packed panels are not involved.
fn oracle(p: &Conv, input: &[f32], weights: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut result = vec![0.; p.batch * p.oh * p.ow * p.cout];
    for n in 0..p.batch {
        for y in 0..p.oh {
            for x in 0..p.ow {
                for co in 0..p.cout {
                    let mut sum = bias.get(co).copied().unwrap_or(0.);
                    for ci in 0..p.cin {
                        for ky in 0..p.kh {
                            for kx in 0..p.kw {
                                let sy = (y * p.sh + ky * p.dh) as isize - p.pt as isize;
                                let sx = (x * p.sw + kx * p.dw) as isize - p.pl as isize;
                                let value = if sy < 0
                                    || sy >= p.height as isize
                                    || sx < 0
                                    || sx >= p.width as isize
                                {
                                    0.
                                } else {
                                    input[((n * p.height + sy as usize) * p.width + sx as usize)
                                        * p.cin
                                        + ci]
                                };
                                let weight = weights[((co * p.cin + ci) * p.kh + ky) * p.kw + kx];
                                sum = value.mul_add(weight, sum);
                            }
                        }
                    }
                    result[((n * p.oh + y) * p.ow + x) * p.cout + co] = sum;
                }
            }
        }
    }
    result
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "index={index}");
    }
}

#[test]
fn neon_dense_preserves_fma_order_across_panels_k_blocks_and_receptive_fields() {
    for p in [
        plan(3, 1, 3, 1),
        plan(7, 5, 3, 2),
        plan(65, 17, 3, 1),
        plan(11, 33, 7, 1),
        plan(7, 9, 9, 1),
    ] {
        let input = values(p.batch * p.height * p.width * p.cin, 7);
        let weights = values(p.cout * p.cin * p.kh * p.kw, 11);
        let bias = values(p.cout, 13);
        let packed = pack(&p, &weights);
        for bias in [bias.as_slice(), &[]] {
            let expected = oracle(&p, &input, &weights, bias);
            let mut actual = vec![f32::NAN; expected.len()];
            execute(
                Backend::NeonFma,
                &p,
                &input,
                bias,
                &packed,
                &mut actual,
                0,
                &IdentityStore,
            );
            assert_bits(&actual, &expected);
        }
    }
}

#[test]
fn neon_narrow_pad_end_preserves_ordered_fma_padding_and_channel_tails() {
    // Pair each shipped channel family with a distinct short spatial shape;
    // these include a single row, full/partial six-row tiles and batch edges.
    for (cin, cout, height, width) in [
        (16, 8, 1, 5),
        (24, 12, 3, 6),
        (12, 24, 2, 5),
        (48, 24, 3, 7),
        (96, 24, 4, 17),
    ] {
        let mut p = plan(cin, cout, 2, 1);
        p.height = height;
        p.oh = height;
        p.width = width;
        p.ow = width;
        let input = (0..p.batch * height * width * cin)
            .map(|i| [1.0000001, 1., -1., -0.0, 0.375][i % 5])
            .collect::<Vec<_>>();
        let weights = (0..cout * cin * 4)
            .map(|i| [1.0000001, -1., 1., 0.5, -0.25, 0.][i % 6])
            .collect::<Vec<_>>();
        let bias = (0..cout)
            .map(|i| [-1., 1., -0.0][i % 3])
            .collect::<Vec<_>>();
        let packed = pack(&p, &weights);
        for bias in [bias.as_slice(), &[]] {
            // The independent oracle explicitly executes zero-valued padding
            // taps; skipping them would change signed-zero/FMA semantics.
            let expected = oracle(&p, &input, &weights, bias);
            let mut actual = vec![f32::NAN; expected.len()];
            execute(
                Backend::NeonFma,
                &p,
                &input,
                bias,
                &packed,
                &mut actual,
                0,
                &IdentityStore,
            );
            assert_bits(&actual, &expected);
        }
    }
}

struct ResidualRelu<'a> {
    residual: &'a [f32],
    columns: usize,
}

#[test]
fn neon_narrow_three_preserves_ordered_fma_for_both_strides_and_worker_borders() {
    for stride in [1usize, 2] {
        let mut p = plan(32, 24, 3, 1);
        p.sh = stride;
        p.sw = stride;
        p.oh = p.height.div_ceil(stride);
        p.ow = p.width.div_ceil(stride);
        let input = values(p.batch * p.height * p.width * p.cin, 7);
        let weights = values(p.cout * p.cin * 9, 11);
        let bias = values(p.cout, 13);
        let packed = pack(&p, &weights);
        for bias in [bias.as_slice(), &[]] {
            let expected = oracle(&p, &input, &weights, bias);
            let mut actual = vec![f32::NAN; expected.len()];
            execute(
                Backend::NeonFma,
                &p,
                &input,
                bias,
                &packed,
                &mut actual,
                0,
                &IdentityStore,
            );
            assert_bits(&actual, &expected);
        }
        let residual = values(p.batch * p.oh * p.ow * p.cout, 17);
        let store = ResidualRelu {
            residual: &residual,
            columns: p.cout,
        };
        let mut expected = oracle(&p, &input, &weights, &bias);
        for (index, value) in expected.iter_mut().enumerate() {
            *value = store.scalar(*value, index);
        }
        for (first, count) in [(1, 5), (p.ow - 1, 6), (p.oh * p.ow - 3, 7)] {
            let sentinel = f32::from_bits(0x7fc0_1234);
            let mut guarded = vec![sentinel; count * p.cout + 6];
            execute(
                Backend::NeonFma,
                &p,
                &input,
                &bias,
                &packed,
                &mut guarded[3..3 + count * p.cout],
                first,
                &store,
            );
            assert!(
                guarded[..3]
                    .iter()
                    .chain(&guarded[3 + count * p.cout..])
                    .all(|value| value.to_bits() == sentinel.to_bits())
            );
            assert_bits(
                &guarded[3..3 + count * p.cout],
                &expected[first * p.cout..(first + count) * p.cout],
            );
        }
    }
}

struct PadEndEpilogue<'a> {
    residual: &'a [f32],
    channel_bias: &'a [f32],
}
impl StoreTransform for PadEndEpilogue<'_> {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        (((value + self.channel_bias[index % self.channel_bias.len()]) + 0.125)
            + self.residual[index])
            .max(0.)
    }
    fn vector<const LANES: usize>(
        &self,
        value: Simd<f32, LANES>,
        index: usize,
    ) -> Simd<f32, LANES> {
        self.vector_channel(value, index, index % self.channel_bias.len())
    }
    fn vector_channel<const LANES: usize>(
        &self,
        value: Simd<f32, LANES>,
        index: usize,
        channel: usize,
    ) -> Simd<f32, LANES> {
        assert_eq!(channel, index % self.channel_bias.len());
        let bias = Simd::load_or_default(
            &self.channel_bias[channel..(channel + LANES).min(self.channel_bias.len())],
        );
        let residual =
            Simd::load_or_default(&self.residual[index..(index + LANES).min(self.residual.len())]);
        (((value + bias) + Simd::splat(0.125)) + residual).simd_max(Simd::splat(0.))
    }
}

#[test]
fn neon_narrow_pad_end_preserves_fused_stores_and_guarded_worker_bands() {
    let p = plan(24, 12, 2, 1);
    let input = values(p.batch * p.height * p.width * p.cin, 7);
    let weights = values(p.cout * p.cin * 4, 11);
    let bias = values(p.cout, 13);
    let packed = pack(&p, &weights);
    let residual = values(p.batch * p.oh * p.ow * p.cout, 17);
    let channel_bias = values(p.cout, 19);
    let store = PadEndEpilogue {
        residual: &residual,
        channel_bias: &channel_bias,
    };
    let mut expected = oracle(&p, &input, &weights, &bias);
    for (index, value) in expected.iter_mut().enumerate() {
        *value = store.scalar(*value, index);
    }
    // A band crosses a batch, one spans a 48-row group, and the remaining
    // bands exercise all six short-tile lengths and both padding edges.
    for (first, count) in [
        (16, 1),
        (32, 2),
        (47, 3),
        (73, 4),
        (148, 5),
        (197, 6),
        (149, 49),
    ] {
        let sentinel = f32::from_bits(0x7fc0_1234);
        let mut guarded = vec![sentinel; count * p.cout + 6];
        execute(
            Backend::NeonFma,
            &p,
            &input,
            &bias,
            &packed,
            &mut guarded[3..3 + count * p.cout],
            first,
            &store,
        );
        assert!(
            guarded[..3]
                .iter()
                .chain(&guarded[3 + count * p.cout..])
                .all(|value| value.to_bits() == sentinel.to_bits())
        );
        assert_bits(
            &guarded[3..3 + count * p.cout],
            &expected[first * p.cout..(first + count) * p.cout],
        );
    }
}
impl StoreTransform for ResidualRelu<'_> {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        (value + self.residual[index]).max(0.)
    }
    fn vector<const LANES: usize>(
        &self,
        value: Simd<f32, LANES>,
        index: usize,
    ) -> Simd<f32, LANES> {
        let residual =
            Simd::load_or_default(&self.residual[index..(index + LANES).min(self.residual.len())]);
        (value + residual).simd_max(Simd::splat(0.))
    }
    fn vector_channel<const LANES: usize>(
        &self,
        value: Simd<f32, LANES>,
        index: usize,
        channel: usize,
    ) -> Simd<f32, LANES> {
        assert_eq!(channel, index % self.columns);
        self.vector(value, index)
    }
}

#[test]
fn neon_dense_keeps_bias_first_residual_activation_and_guarded_worker_tails() {
    let p = plan(65, 17, 3, 2);
    let input = (0..p.batch * p.height * p.width * p.cin)
        .map(|i| [1.0000001, 1., -1., -0.0, 0.375][i % 5])
        .collect::<Vec<_>>();
    let weights = (0..p.cout * p.cin * p.kh * p.kw)
        .map(|i| [1.0000001, -1., 1., 0.5, -0.25, 0.][i % 6])
        .collect::<Vec<_>>();
    let bias = (0..p.cout)
        .map(|i| [-1., 1., -0.0][i % 3])
        .collect::<Vec<_>>();
    let packed = pack(&p, &weights);
    let residual = values(p.batch * p.oh * p.ow * p.cout, 17);
    let store = ResidualRelu {
        residual: &residual,
        columns: p.cout,
    };
    let mut expected = oracle(&p, &input, &weights, &bias);
    for (index, value) in expected.iter_mut().enumerate() {
        *value = store.scalar(*value, index);
    }
    // Starts within rows, crosses a batch, spans a 48-row group, and covers
    // all six possible short tile lengths without writing past the band.
    for (first, count) in [
        (3, 1),
        (9, 2),
        (47, 3),
        (73, 4),
        (97, 5),
        (149, 49),
        (197, 6),
    ] {
        let sentinel = f32::from_bits(0x7fc0_1234);
        let mut guarded = vec![sentinel; count * p.cout + 6];
        execute(
            Backend::NeonFma,
            &p,
            &input,
            &bias,
            &packed,
            &mut guarded[3..3 + count * p.cout],
            first,
            &store,
        );
        assert!(
            guarded[..3]
                .iter()
                .chain(&guarded[3 + count * p.cout..])
                .all(|value| value.to_bits() == sentinel.to_bits())
        );
        assert_bits(
            &guarded[3..3 + count * p.cout],
            &expected[first * p.cout..(first + count) * p.cout],
        );
    }
}
