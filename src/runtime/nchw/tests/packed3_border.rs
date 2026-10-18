//! Grouping independent border channels must retain the scalar spatial contract.
use super::*;

fn vector_position(p: &Conv2d, stride: usize, spatial: usize) -> bool {
    let y = spatial / p.output_width;
    let x = spatial % p.output_width;
    if stride == 1 {
        y > 0 && y + 1 < p.height && x >= 1 && x < 1 + p.width.saturating_sub(2) / 8 * 8
    } else if y > 0 && y * 2 + 1 < p.height {
        let mut begin = 1;
        while begin + 8 <= p.output_width && (begin + 7) * 2 + 1 < p.width {
            if (begin..begin + 8).contains(&x) {
                return true;
            }
            begin += 8;
        }
        false
    } else {
        false
    }
}

#[test]
fn packed3_grouped_borders_keep_separate_arithmetic_padding_and_scalar_stores() {
    for stride in [1, 2] {
        for (height, width) in [(1, 1), (2, 9), (3, 10), (5, 17), (5, 18)] {
            for case in ["rounding", "signed_zero", "cancellation", "nonfinite"] {
                let channels = if matches!(case, "rounding" | "signed_zero") {
                    1
                } else {
                    3
                };
                let p = parameters(channels, 24, height, width, [3, 3], [stride; 2], [1; 2], 1);
                let plane = p.output_height * p.output_width;
                let input: Vec<_> = (0..p.batch * channels * height * width)
                    .map(|i| match case {
                        "rounding" => f32::from_bits(0x3f80_0001),
                        "signed_zero" => 1.,
                        "cancellation" => {
                            [16_777_216., 1., -16_777_216., -0., 0., 0.5, -0.5][i % 7]
                        }
                        _ if i < channels * height * width => 1.,
                        _ => [
                            f32::INFINITY,
                            f32::NEG_INFINITY,
                            -0.,
                            0.,
                            f32::from_bits(0x7fc0_1234),
                            1.,
                            -1.,
                        ][i % 7],
                    })
                    .collect();
                let weights: Vec<_> = (0..24 * channels * 9)
                    .map(|i| match case {
                        "rounding" if i % 9 == 4 => f32::from_bits(0x3f80_0001),
                        "rounding" => 0.,
                        "signed_zero" => -0.,
                        "cancellation" => [1., -1., 0.5, -0.5][i % 4],
                        // Invalid taps must be skipped, including poisoned
                        // padding around the only pixel of a 1x1 input.
                        _ if i % 9 != 4 => f32::from_bits(0x7fc0_5678),
                        _ => [1., -1., 0., f32::INFINITY][i / (channels * 9) % 4],
                    })
                    .collect();
                let bias: Vec<_> = (0..24)
                    .map(|_| match case {
                        "rounding" => f32::from_bits(0xbf80_0002),
                        "signed_zero" => -0.,
                        _ => 0.,
                    })
                    .collect();
                let packed = panels(&weights, &p);
                let kind = if stride == 1 {
                    Kind::Packed3
                } else {
                    Kind::PackedStride2
                };
                for backend in backends()
                    .filter(|b| matches!(b, Backend::Avx | Backend::AvxFma | Backend::Avx2Fma))
                {
                    assert_eq!(select(backend, &p, true, 1), Some(kind));
                    let expected = reference(&p, backend, kind, &input, &weights, &bias);
                    let length = expected.len();
                    let store = PacketVisits::new(length, 0..length, plane);
                    let guard = f32::from_bits(0x7fc0_55aa);
                    let mut output = vec![guard; length + 8];
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
                    assert!(
                        output[..4]
                            .iter()
                            .chain(&output[4 + length..])
                            .all(|v| v.to_bits() == guard.to_bits())
                    );
                    for (index, (&actual, &expected)) in
                        output[4..4 + length].iter().zip(&expected).enumerate()
                    {
                        // Rust leaves arithmetic NaN payload selection
                        // unspecified; every non-NaN result is bit-exact.
                        if expected.is_nan() {
                            assert!(
                                actual.is_nan(),
                                "{backend:?} {case} stride={stride} {height}x{width} index={index}"
                            );
                        } else {
                            assert_eq!(
                                actual.to_bits(),
                                expected.to_bits(),
                                "{backend:?} {case} stride={stride} {height}x{width} index={index}"
                            );
                        }
                        if !vector_position(&p, stride, index % plane) && case == "rounding" {
                            // FMA would leave 2^-46 here; the scalar border
                            // rounds the multiplication before its addition.
                            assert_eq!(actual.to_bits(), 0);
                        }
                        if case == "signed_zero" {
                            assert_eq!(actual.to_bits(), (-0.0f32).to_bits());
                        }
                    }
                    store.assert_coverage(|index| vector_position(&p, stride, index % plane));
                }
            }
        }
    }
}
