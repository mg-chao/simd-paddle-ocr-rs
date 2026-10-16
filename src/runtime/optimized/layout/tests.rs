use super::*;

#[test]
fn layout_worker_selection_matches_reference_thresholds_and_direction_caps() {
    assert_eq!(workers(1, 64, 4095, Layout::Canonical, 8), 1);
    assert_eq!(workers(1, 64, 4096, Layout::Canonical, 8), 8);
    assert_eq!(workers(2, 64, 2048, Layout::Canonical, 8), 8);
    assert_eq!(workers(1, 4096, 64, Layout::Canonical, 8), 1);
    assert_eq!(workers(1, 4096, 127, Layout::Canonical, 8), 1);
    assert_eq!(workers(1, 4096, 128, Layout::Canonical, 8), 2);
    assert_eq!(workers(1, 7, 40_000, Layout::Nhwc, 8), 1);
    assert_eq!(workers(1, 15, 20_000, Layout::Nhwc, 8), 1);
    assert_eq!(workers(1, 16, 20_000, Layout::Nhwc, 8), 2);
    assert_eq!(workers(1, 63, 5_000, Layout::Nhwc, 8), 7);
    assert_eq!(workers(1, 64, 4096, Layout::Nhwc, 8), 8);
    assert_eq!(workers(1, 64, 4096, Layout::Nhwc, 1), 1);
}

#[test]
fn parallel_layout_bands_preserve_all_bits_batches_and_both_direction_tails() {
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 8 } else { 1 }).unwrap();
    for (batch, channels, plane) in [
        (1, 7, 41),
        (2, 17, 9),
        (1, 64, 4096),
        (2, 65, 2051),
        (2, 4097, 65),
    ] {
        let shape = Shape::new(&[batch, channels, 1, plane]).unwrap();
        let input = (0..shape.element_count())
            .map(|i| {
                f32::from_bits(match i % 11 {
                    0 => 0x8000_0000,
                    1 => 0x7fc0_0001 + i as u32,
                    2 => 0x7f80_0000,
                    3 => 0xff80_0000,
                    _ => 0x3e00_0000 + i as u32,
                })
            })
            .collect::<Vec<_>>();
        let mut expected = vec![0.; input.len()];
        for b in 0..batch {
            for c in 0..channels {
                for p in 0..plane {
                    expected[(b * plane + p) * channels + c] =
                        input[(b * channels + c) * plane + p];
                }
            }
        }
        for budget in [1, 2, 4, 8, 32] {
            let mut output = vec![f32::NAN; input.len()];
            convert(
                &input,
                &mut output,
                shape,
                Layout::Canonical,
                simd::Backend::Portable4,
                &cpu,
                budget,
            );
            assert!(
                output
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            let mut restored = vec![f32::NAN; input.len()];
            convert(
                &output,
                &mut restored,
                shape,
                Layout::Nhwc,
                simd::Backend::Portable4,
                &cpu,
                budget,
            );
            assert!(
                restored
                    .iter()
                    .zip(&input)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
}

#[test]
fn native_transpose_column_bands_keep_guards_and_raw_bit_payloads() {
    let mut backends = vec![simd::Backend::Portable4];
    if simd::avx2_fma_available() {
        backends.push(simd::Backend::Avx2Fma);
    }
    if simd::avx512_available() {
        backends.push(simd::Backend::Avx512);
    }
    for backend in backends {
        for rows in [1, 7, 8, 9, 17, 31] {
            for columns in [1, 7, 8, 9, 17, 33] {
                let input = (0..rows * columns)
                    .map(|i| f32::from_bits(0x7fc0_0001 + i as u32))
                    .collect::<Vec<_>>();
                for first in [0, columns / 2, columns] {
                    let count = columns - first;
                    let mut guard = vec![f32::from_bits(0xffc0_002a); count * rows + 2];
                    simd::transpose_matrix_columns(
                        backend,
                        &input,
                        &mut guard[1..count * rows + 1],
                        rows,
                        columns,
                        first,
                    );
                    assert_eq!(guard[0].to_bits(), 0xffc0_002a);
                    assert_eq!(guard[count * rows + 1].to_bits(), 0xffc0_002a);
                    for c in 0..count {
                        for r in 0..rows {
                            assert_eq!(
                                guard[1 + c * rows + r].to_bits(),
                                input[r * columns + first + c].to_bits()
                            );
                        }
                    }
                }
            }
        }
    }
}
