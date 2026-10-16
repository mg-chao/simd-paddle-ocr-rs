use super::*;

fn oracle(
    input: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    factor: usize,
) -> Vec<f32> {
    let (oh, ow) = (height * factor, width * factor);
    let mut output = vec![0.; batch * channels * oh * ow];
    for n in 0..batch {
        for y in 0..oh {
            for x in 0..ow {
                for c in 0..channels {
                    output[((n * oh + y) * ow + x) * channels + c] =
                        input[((n * height + y / factor) * width + x / factor) * channels + c];
                }
            }
        }
    }
    output
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "resize payload at {i}");
    }
}

fn input(length: usize) -> Vec<f32> {
    let patterns = [
        0x8000_0000,
        0,
        0x7fc0_1234,
        0xffc0_5678,
        0x7f80_0000,
        0xff80_0000,
        0x3f80_0000,
        0x0000_0001,
        0x8080_0001,
    ];
    (0..length)
        .map(|i| f32::from_bits(patterns[i % patterns.len()]))
        .collect()
}

#[test]
fn nearest_row_reuse_preserves_all_payloads_width_tails_and_vertical_factors() {
    for channels in [1, 3, 4, 7, 8, 9, 17] {
        for (height, width, factor) in [(1, 1, 1), (1, 3, 4), (3, 5, 2), (4, 7, 3)] {
            let batch = 2;
            let source = input(batch * height * width * channels);
            let expected = oracle(&source, batch, channels, height, width, factor);
            let shape = Shape::new(&[batch, channels, height, width]).unwrap();
            let kind = BoundKind::Resize {
                height,
                width,
                factor,
            };
            for backend in [Backend::Scalar, Backend::Portable4] {
                let mut actual = vec![0.; expected.len()];
                execute(backend, &kind, &source, shape, &mut actual);
                assert_bits(&actual, &expected);
            }
        }
    }
}

#[test]
fn nearest_row_reuse_keeps_worker_bands_disjoint_and_source_identity_exact() {
    let (batch, channels, height, width, factor) = (3usize, 9usize, 3usize, 5usize, 3usize);
    let source = input(batch * height * width * channels);
    let expected = oracle(&source, batch, channels, height, width, factor);
    let pixels = expected.len() / channels;
    let shape = Shape::new(&[batch, channels, height, width]).unwrap();
    let kind = BoundKind::Resize {
        height,
        width,
        factor,
    };
    for workers in [1usize, 2, 4, 8] {
        let band_pixels = pixels.div_ceil(workers);
        let mut actual = vec![0.; expected.len()];
        std::thread::scope(|scope| {
            for (band, output) in actual.chunks_mut(band_pixels * channels).enumerate() {
                let (source, kind) = (&source, &kind);
                scope.spawn(move || {
                    execute_range(
                        Backend::Portable4,
                        kind,
                        source,
                        shape,
                        output,
                        band * band_pixels,
                    )
                });
            }
        });
        assert_bits(&actual, &expected);
    }
    for first in [0usize, 1, 14, 15, 16, 44, 45, 46, 133, 134, 135, 136] {
        for count in [1usize, 7, 15, 16, 31, 46] {
            let count = count.min(pixels - first);
            let sentinel = f32::from_bits(0x7fc0_aaaa);
            let mut guarded = vec![sentinel; count * channels + 6];
            execute_range(
                Backend::Portable4,
                &kind,
                &source,
                shape,
                &mut guarded[3..3 + count * channels],
                first,
            );
            assert_bits(
                &guarded[3..3 + count * channels],
                &expected[first * channels..(first + count) * channels],
            );
            assert!(
                guarded[..3]
                    .iter()
                    .all(|v| v.to_bits() == sentinel.to_bits())
            );
            assert!(
                guarded[3 + count * channels..]
                    .iter()
                    .all(|v| v.to_bits() == sentinel.to_bits())
            );
        }
    }
}
