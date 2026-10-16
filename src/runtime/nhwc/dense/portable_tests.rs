use super::*;

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
        height: 7,
        width: 11,
        oh: 7,
        ow: 11,
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
    let mut matrix = vec![0.; inner * p.cout];
    for co in 0..p.cout {
        for k in 0..inner {
            matrix[k * p.cout + co] = weights[co * inner + k];
        }
    }
    PackedMatrix::new(&matrix, inner, p.cout).unwrap()
}

// Independent logical N/OH/OW/OC/IC/KY/KX loops; no production descriptors,
// paired tile, flattened K cursor, or panel traversal participates in the oracle.
// Keep the reference's product + accumulator operand order explicit.
#[allow(clippy::assign_op_pattern)]
fn oracle(p: &Conv, input: &[f32], weights: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut output = vec![0.; p.batch * p.oh * p.ow * p.cout];
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
                                sum = value * weight + sum;
                            }
                        }
                    }
                    output[((n * p.oh + y) * p.ow + x) * p.cout + co] = sum;
                }
            }
        }
    }
    output
}

fn assert_bits(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{context}: index={index}"
        );
    }
}

struct Indexed {
    columns: usize,
}
impl StoreTransform for Indexed {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        value + (index % 97) as f32 / 127.
    }
    fn vector<const LANES: usize>(
        &self,
        value: Simd<f32, LANES>,
        index: usize,
    ) -> Simd<f32, LANES> {
        value
            + Simd::from_array(std::array::from_fn(|lane| {
                ((index + lane) % 97) as f32 / 127.
            }))
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
fn portable_dense_pairs_cover_panels_blocks_borders_dilation_and_general_taps() {
    for p in [
        plan(3, 1, 3, 1),
        plan(3, 5, 3, 2),
        plan(7, 9, 3, 1),
        plan(65, 17, 3, 1),
        plan(23, 33, 5, 1),
        plan(17, 15, 6, 1),
        plan(11, 17, 7, 1),
        plan(11, 17, 7, 2),
        plan(7, 17, 9, 1),
        plan(7, 17, 9, 2),
    ] {
        let input = values(p.batch * p.height * p.width * p.cin, 7);
        let weights = values(p.cout * p.cin * p.kh * p.kw, 11);
        let bias = values(p.cout, 13);
        let packed = pack(&p, &weights);
        for bias in [bias.as_slice(), &[]] {
            let expected = oracle(&p, &input, &weights, bias);
            let mut actual = vec![f32::NAN; expected.len()];
            execute(
                Backend::Portable4,
                &p,
                &input,
                bias,
                &packed,
                &mut actual,
                0,
                &IdentityStore,
            );
            assert_bits(
                &actual,
                &expected,
                &format!(
                    "cin={} cout={} kh={} dilation={}",
                    p.cin, p.cout, p.kh, p.dh
                ),
            );
        }
    }
}

#[test]
fn portable_dense_pairs_preserve_nonfused_bias_order_and_global_worker_stores() {
    let p = plan(65, 33, 3, 2);
    let input: Vec<_> = (0..p.batch * p.height * p.width * p.cin)
        .map(|i| [1.0000001, 1., -1., -0.0, 0.375][i % 5])
        .collect();
    let weights: Vec<_> = (0..p.cout * p.cin * p.kh * p.kw)
        .map(|i| [1.0000001, -1., 1., 0.5, -0.25, 0.][i % 6])
        .collect();
    let bias: Vec<_> = (0..p.cout).map(|i| [-1., 1., -0.0][i % 3]).collect();
    let packed = pack(&p, &weights);
    let store = Indexed { columns: p.cout };
    let mut expected = oracle(&p, &input, &weights, &bias);
    for (index, value) in expected.iter_mut().enumerate() {
        *value = store.scalar(*value, index);
    }
    let rows = p.batch * p.oh * p.ow;
    for threads in [1usize, 2, 4, 8] {
        let band_rows = rows.div_ceil(threads);
        let mut actual = vec![f32::NAN; expected.len()];
        std::thread::scope(|scope| {
            for (band, output) in actual.chunks_mut(band_rows * p.cout).enumerate() {
                let (p, input, bias, packed, store) = (&p, &input, &bias, &packed, &store);
                scope.spawn(move || {
                    execute(
                        Backend::Portable4,
                        p,
                        input,
                        bias,
                        packed,
                        output,
                        band * band_rows,
                        store,
                    )
                });
            }
        });
        assert_bits(&actual, &expected, &format!("dense paired bands={threads}"));
    }
    // Worker slices start within a row, cross a batch, and end in a six-row tail.
    for (first, count) in [(3, 1), (9, 5), (47, 7), (73, 13), (97, 49)] {
        let sentinel = f32::from_bits(0x7fc0_1234);
        let mut guarded = vec![sentinel; count * p.cout + 6];
        execute(
            Backend::Portable4,
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
                .all(|v| v.to_bits() == sentinel.to_bits())
        );
        assert!(
            guarded[3 + count * p.cout..]
                .iter()
                .all(|v| v.to_bits() == sentinel.to_bits())
        );
        assert_bits(
            &guarded[3..3 + count * p.cout],
            &expected[first * p.cout..(first + count) * p.cout],
            "guarded dense worker band",
        );
    }
}
