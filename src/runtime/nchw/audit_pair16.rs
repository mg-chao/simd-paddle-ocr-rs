//! Independent selector, arithmetic and store contracts transcribed from pinned
//! Conv3x3Packed.cs and Conv3x3Packed.Avx.cs, including its eight-pixel boundary.
use super::*;

#[test]
fn audit_pair16_selector_matches_reference_eligibility() {
    for backend in [
        Backend::Scalar,
        Backend::Portable4,
        Backend::Avx,
        Backend::AvxFma,
        Backend::Avx2Fma,
        Backend::Avx512,
    ] {
        for channels in [56, 64, 72, 80] {
            let p = parameters(3, channels, 3, 18, [3, 3], [1, 1], [1, 1], 1);
            for threads in [1, 2, 4] {
                let paired = !matches!(backend, Backend::Scalar | Backend::Portable4)
                    && threads == 1
                    && channels >= 64
                    && channels % 16 == 0;
                let expected = if backend == Backend::Scalar {
                    None
                } else if paired {
                    Some("nchw_packed_3x3_pair16")
                } else {
                    Some("nchw_packed_3x3")
                };
                assert_eq!(
                    family_for_threads(backend, &p, true, threads),
                    expected,
                    "paired16 selector {backend:?}, cout={channels}, threads={threads}"
                );
                assert_eq!(
                    uses_pack_for_threads(backend, &p, true, threads),
                    backend != Backend::Scalar
                );
                assert_ne!(
                    family_for_threads(backend, &p, false, threads),
                    Some("nchw_packed_3x3_pair16")
                );
                assert_eq!(
                    selected_backend_for_threads(backend, &p, true, threads),
                    if paired && backend == Backend::Avx512 {
                        Backend::Avx2Fma
                    } else {
                        backend
                    }
                );
            }
        }
    }
}

// Literal geometry from the reference's while loop: x starts at zero, one
// scalar pixel precedes each row's first eligible packet, then packets have 8
// pixels and require a valid neighbour on both sides and above/below.
fn packet(p: &Conv2d, y: usize, x: usize) -> bool {
    y > 0 && y + 1 < p.height && x >= 1 && x < 1 + p.width.saturating_sub(2) / 8 * 8
}

fn oracle(p: &Conv2d, backend: Backend, input: &[f32], weights: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut result = Vec::new();
    for batch in 0..p.batch {
        for co in 0..p.output_channels {
            for y in 0..p.height {
                for x in 0..p.width {
                    let mut sum = if bias.is_empty() { 0.0 } else { bias[co] };
                    for ci in 0..p.input_channels {
                        for ky in 0..3 {
                            if y + ky == 0 || y + ky > p.height {
                                continue;
                            }
                            for kx in 0..3 {
                                if x + kx == 0 || x + kx > p.width {
                                    continue;
                                }
                                let value =
                                    input[((batch * p.input_channels + ci) * p.height + y + ky
                                        - 1)
                                        * p.width
                                        + x
                                        + kx
                                        - 1];
                                let weight =
                                    weights[(co * p.input_channels + ci) * 9 + ky * 3 + kx];
                                sum = if backend != Backend::Avx && packet(p, y, x) {
                                    value.mul_add(weight, sum)
                                } else {
                                    sum + value * weight
                                };
                            }
                        }
                    }
                    result.push(sum);
                }
            }
        }
    }
    result
}

#[test]
fn audit_pair16_matches_pinned_arithmetic_at_packets_and_edges() {
    for (height, width) in [(1, 7), (2, 9), (3, 9), (3, 10), (3, 17), (5, 18)] {
        for outputs in [64, 80] {
            let p = parameters(3, outputs, height, width, [3, 3], [1, 1], [1, 1], 1);
            let input: Vec<_> = (0..p.batch * 3 * height * width)
                .map(|i| ((i * 37 % 43) as f32 - 21.) / 17.)
                .collect();
            let weights: Vec<_> = (0..outputs * 3 * 9)
                .map(|i| ((i * 19 % 29) as f32 - 14.) / 13.)
                .collect();
            let packed = panels(&weights, &p);
            for backend in backends().filter(|b| *b != Backend::Portable4) {
                for bias in [vec![], (0..outputs).map(|i| i as f32 / 19. - 0.3).collect()] {
                    let expected = oracle(&p, backend, &input, &weights, &bias);
                    let mut actual = vec![f32::NAN; expected.len()];
                    assert!(try_conv(
                        backend,
                        &p,
                        &input,
                        &weights,
                        &bias,
                        Some(&packed),
                        &mut actual,
                        &CpuContext::new(1).unwrap(),
                        1
                    ));
                    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            e.to_bits(),
                            "reference paired16 arithmetic {backend:?} {height}x{width} cout={outputs} index={index}"
                        );
                    }
                }
            }
        }
    }
}

struct IndexedStore(PacketVisits);
impl simd::StoreTransform for IndexedStore {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        self.0.scalar(value, index) + index as f32 / 1024.
    }
    fn vector<const N: usize>(&self, value: Simd<f32, N>, index: usize) -> Simd<f32, N> {
        assert_eq!(
            N, 8,
            "the paired kernel must deliver contiguous spatial octets"
        );
        self.0.vector(value, index)
            + Simd::from_array(std::array::from_fn(|lane| (index + lane) as f32 / 1024.))
    }
}

#[test]
fn audit_pair16_stores_spatial_octets_with_global_band_indices() {
    let p = parameters(3, 80, 3, 18, [3, 3], [1, 1], [1, 1], 1);
    let plane = p.height * p.width;
    let length = p.batch * p.output_channels * plane;
    let input: Vec<_> = (0..p.batch * p.input_channels * plane)
        .map(|index| (index % 127 + 1) as f32 / 32.)
        .collect();
    let mut weights = vec![0.; p.output_channels * p.input_channels * 9];
    for co in 0..p.output_channels {
        weights[(co * p.input_channels + co % p.input_channels) * 9 + 4] = 1.;
    }
    let bias: Vec<_> = (0..p.output_channels).map(|co| co as f32 / 8.).collect();
    let packed = panels(&weights, &p);
    for backend in backends().filter(|b| *b != Backend::Portable4) {
        let guard = f32::from_bits(0x7fc0_55aa);
        for band in [None, Some((1, 16, 32))] {
            let (begin, end) = band.map_or((0, length), |(batch, first, count)| {
                let begin = (batch * p.output_channels + first) * plane;
                (begin, begin + count * plane)
            });
            let store = IndexedStore(PacketVisits::new(length, begin..end, plane));
            let mut output = vec![guard; end - begin + 8];
            if let Some((batch, first, count)) = band {
                let kind = select(backend, &p, true, 1).unwrap();
                run(
                    selected_backend_for_threads(backend, &p, true, 1),
                    kind,
                    &p,
                    &input,
                    &weights,
                    &bias,
                    Some(&packed),
                    &mut output[4..4 + end - begin],
                    batch,
                    first,
                    count,
                    &store,
                );
            } else {
                assert!(try_conv_store(
                    backend,
                    &p,
                    &input,
                    &weights,
                    &bias,
                    Some(&packed),
                    &mut output[4..4 + length],
                    &CpuContext::new(1).unwrap(),
                    1,
                    &store
                ));
            }
            assert!(
                output[..4]
                    .iter()
                    .chain(&output[4 + end - begin..])
                    .all(|value| value.to_bits() == guard.to_bits())
            );
            for (offset, value) in output[4..4 + end - begin].iter().enumerate() {
                let index = begin + offset;
                let batch = index / (p.output_channels * plane);
                let co = index / plane % p.output_channels;
                let expected = input
                    [(batch * p.input_channels + co % p.input_channels) * plane + index % plane]
                    + bias[co]
                    + index as f32 / 1024.;
                assert_eq!(
                    value.to_bits(),
                    expected.to_bits(),
                    "global final store index {index}"
                );
            }
            store
                .0
                .assert_coverage(|index| packet(&p, index % plane / p.width, index % p.width));
        }
    }
}
