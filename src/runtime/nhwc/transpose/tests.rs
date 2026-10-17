use super::*;

struct ResidualRelu<'a>(&'a [f32]);
impl StoreTransform for ResidualRelu<'_> {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        (value + self.0[index]).max(0.)
    }
    fn vector<const N: usize>(&self, value: Simd<f32, N>, index: usize) -> Simd<f32, N> {
        // Deliberately read the complete vector: a partial channel block must
        // use the scalar epilogue rather than read across its output extent.
        let residual = Simd::from_slice(&self.0[index..index + N]);
        std::simd::num::SimdFloat::simd_max(value + residual, Simd::splat(0.))
    }
}

#[test]
fn direct_transpose_matches_buffered_bits_with_fused_stores_and_row_bands() {
    if !simd::avx2_fma_available() {
        return;
    }
    for (cin, cout, height, width) in [
        (3, 16, 1, 1),
        (17, 17, 3, 7),
        (24, 24, 3, 9),
        (64, 64, 2, 33),
        (65, 31, 3, 5),
    ] {
        let p = kernels::ConvTranspose {
            batch: 2,
            cin,
            cout,
            height,
            width,
        };
        let values = |count, seed| {
            (0..count)
                .map(|i| {
                    // Non-dyadic values expose any change in FMA or bias placement.
                    ((i * 17 + seed) % 37) as f32 / 19. - 0.8
                })
                .collect::<Vec<_>>()
        };
        let input = values(2 * cin * height * width, 11);
        let weights = values(cin * cout * 4, 13);
        let bias = values(cout, 17);
        let residual = values(2 * height * width * cout * 4, 19);
        let store = ResidualRelu(&residual);
        let panels = std::array::from_fn(|tap| {
            let matrix = (0..cin * cout)
                .map(|i| weights[i * 4 + tap])
                .collect::<Vec<_>>();
            PackedMatrix::new(&matrix, cin, cout).unwrap()
        });
        for bias in [bias.as_slice(), &[]] {
            // Original production path: packed GEMM then scalar scatter.
            let mut expected = vec![0.; residual.len()];
            let mut temporary = vec![0.; 2 * height * width * cout];
            for (tap, panel) in panels.iter().enumerate() {
                simd::matmul_bias(
                    Backend::Avx2Fma,
                    &input,
                    &[],
                    Some(panel),
                    bias,
                    &mut temporary,
                    2 * height * width,
                    cin,
                    cout,
                );
                for (pixel, source) in temporary.chunks_exact(cout).enumerate() {
                    let at = (pixel / width * 4 * width
                        + pixel % width * 2
                        + tap / 2 * 2 * width
                        + tap % 2)
                        * cout;
                    for (channel, &value) in source.iter().enumerate() {
                        expected[at + channel] = store.scalar(value, at + channel);
                    }
                }
            }
            for band_rows in [1, 2, 3, 2 * height] {
                let sentinel = f32::from_bits(0x7fc0_1234);
                let mut output = vec![sentinel; expected.len() + 6];
                for (band, destination) in output[3..3 + expected.len()]
                    .chunks_mut(band_rows * 4 * width * cout)
                    .enumerate()
                {
                    // SAFETY: runtime guard above and checked owned extents.
                    unsafe {
                        packed_direct(
                            &p,
                            &input,
                            bias,
                            &panels,
                            destination,
                            band * band_rows,
                            &store,
                        )
                    };
                }
                assert!(
                    output[3..3 + expected.len()]
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "cin={cin} cout={cout} width={width} band_rows={band_rows}"
                );
                assert!(
                    output[..3]
                        .iter()
                        .chain(&output[3 + expected.len()..])
                        .all(|v| v.to_bits() == sentinel.to_bits())
                );
            }
        }
    }
}
