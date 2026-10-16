//! Independent raw-bit expectations from exact-rational binary32 arithmetic.
//! Pinned source: Tensor.Avx.cs:24,91,154-163, revision
//! 33ff8a81672452041d110e8ba52dcd2639e54228. This is not a C# execution capture.
use super::*;

const ERF_INPUTS: [u32; 8] = [
    0x3eb1af60, 0xbeb1af60, 0x80000000, 0x3e800000, 0xbf000000, 0x3f800001, 0xbfc00000, 0x3fffffff,
];
const ERF_SEPARATE: [u32; 8] = [
    0x3ec0bace, 0xbec0bace, 0x80000000, 0x3e8d7aa8, 0xbf053f7b, 0x3f57bb3f, 0xbf7752ab, 0x3f7ecd70,
];
const ERF_FMA: [u32; 8] = [
    0x3ec0bacf, 0xbec0bacf, 0x80000000, 0x3e8d7aa8, 0xbf053f7b, 0x3f57bb3f, 0xbf7752ab, 0x3f7ecd70,
];
const GELU_INPUTS: [u32; 8] = [
    0x3f230c95, 0xbf230c95, 0x3e027077, 0xbe027077, 0xc0300000, 0xbfe00000, 0xbf90a3d7, 0xbe9eb852,
];
const GELU_NATIVE_SEPARATE: [u32; 8] = [
    0x3ef0a177, 0xbe2aef66, 0x3d8fa931, 0xbd6a6f79, 0xbc064180, 0xbd8f926e, 0xbe158b43, 0xbdf029a7,
];
const GELU_NATIVE_FMA: [u32; 8] = [
    0x3ef0a176, 0xbe2aef68, 0x3d8fa931, 0xbd6a6f79, 0xbc064180, 0xbd8f926e, 0xbe158b43, 0xbdf029a7,
];
const GELU_DEFAULT_DIVIDE: [u32; 8] = [
    0x3ef0a177, 0xbe2aef66, 0x3d8fa932, 0xbd6a6f79, 0xbc064180, 0xbd8f926e, 0xbe158b43, 0xbdf029a7,
];
const SIGMOID_INPUTS: [u32; 8] = [
    0xc1880000, 0xc1800000, 0xc0533333, 0xbdcccccd, 0x3dcccccd, 0x40200000, 0x41000000, 0x41a00000,
];
const SIGMOID_NATIVE: [u32; 8] = [
    0x00000000, 0x34000000, 0x3d11b310, 0x3ef335ee, 0x3f066509, 0x3f6c948f, 0x3f7fea06, 0x3f800000,
];

fn optimized_backends() -> Vec<Backend> {
    let mut result = vec![Backend::Portable4];
    for (available, backend) in [
        (avx_available(), Backend::Avx),
        (avx_fma_available(), Backend::AvxFma),
        (avx2_fma_available(), Backend::Avx2Fma),
        (avx512_available(), Backend::Avx512),
    ] {
        if available {
            result.push(backend);
        }
    }
    result
}

fn check(backend: Backend, operation: UnaryOp, input: &[u32; 8], expected: &[u32; 8]) {
    let input = input.map(f32::from_bits);
    let mut output = [0.0; 8];
    unary(backend, operation, &input, &mut output);
    assert_eq!(
        output.map(f32::to_bits),
        *expected,
        "{backend:?} {operation:?}"
    );
    let mut inplace = input;
    unary_inplace(backend, operation, &mut inplace);
    assert_eq!(
        inplace.map(f32::to_bits),
        *expected,
        "inplace {backend:?} {operation:?}"
    );
}

#[test]
fn optimized_erf_uses_the_reference_fma_eligibility() {
    for backend in optimized_backends() {
        let expected = if backend.uses_fma() {
            &ERF_FMA
        } else {
            &ERF_SEPARATE
        };
        check(backend, UnaryOp::Erf, &ERF_INPUTS, expected);
    }
}

#[test]
fn optimized_gelu_multiplies_the_reference_inverse_sqrt_two() {
    for backend in optimized_backends() {
        let expected = if backend.uses_fma() {
            &GELU_NATIVE_FMA
        } else {
            &GELU_NATIVE_SEPARATE
        };
        check(backend, UnaryOp::Gelu, &GELU_INPUTS, expected);
    }
}

#[test]
fn optimized_sigmoid_uses_vector_exp_and_reference_negative_subtraction() {
    for backend in optimized_backends() {
        check(backend, UnaryOp::Sigmoid, &SIGMOID_INPUTS, &SIGMOID_NATIVE);
    }
}

#[test]
fn optimized_policy_goldens_preserve_all_tail_and_partition_bits() {
    for backend in optimized_backends() {
        let fused = backend.uses_fma();
        let cases = [
            (
                UnaryOp::Erf,
                &ERF_INPUTS,
                if fused { &ERF_FMA } else { &ERF_SEPARATE },
            ),
            (
                UnaryOp::Gelu,
                &GELU_INPUTS,
                if fused {
                    &GELU_NATIVE_FMA
                } else {
                    &GELU_NATIVE_SEPARATE
                },
            ),
            (UnaryOp::Sigmoid, &SIGMOID_INPUTS, &SIGMOID_NATIVE),
        ];
        for (operation, inputs, goldens) in cases {
            for length in 0..=33 {
                let input: Vec<_> = (0..length).map(|i| f32::from_bits(inputs[i % 8])).collect();
                let expected: Vec<_> = (0..length).map(|i| goldens[i % 8]).collect();
                for partition in [1, 3, 7, 16, 31] {
                    let mut actual = vec![0.0; length];
                    for (src, dst) in input.chunks(partition).zip(actual.chunks_mut(partition)) {
                        unary(backend, operation, src, dst);
                    }
                    assert_eq!(
                        actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                        expected,
                        "{backend:?} {operation:?} length={length} partition={partition}"
                    );
                }
            }
        }
    }
}

#[test]
fn selected_math_preserves_nan_rejection_and_special_values() {
    for backend in optimized_backends() {
        let math = UnaryMath::for_backend(backend);
        for operation in [UnaryOp::Erf, UnaryOp::Gelu, UnaryOp::Sigmoid] {
            assert!(unary_value_with_math(operation, f32::NAN, math).is_nan());
        }
        assert_eq!(
            unary_value_with_math(UnaryOp::Erf, -0.0, math).to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(
            unary_value_with_math(UnaryOp::Gelu, -0.0, math).to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(
            unary_value_with_math(UnaryOp::Gelu, f32::INFINITY, math),
            f32::INFINITY
        );
        assert!(unary_value_with_math(UnaryOp::Gelu, f32::NEG_INFINITY, math).is_nan());
        assert_eq!(unary_value_with_math(UnaryOp::Sigmoid, -0.0, math), 0.5);
        assert_eq!(
            unary_value_with_math(UnaryOp::Sigmoid, f32::INFINITY, math),
            1.0
        );
        assert_eq!(
            unary_value_with_math(UnaryOp::Sigmoid, f32::NEG_INFINITY, math).to_bits(),
            0
        );
    }
}

#[test]
fn forced_policy_and_default_helper_math_are_explicit() {
    for backend in [
        Backend::Scalar,
        Backend::Portable4,
        Backend::Avx,
        Backend::AvxFma,
        Backend::Avx2Fma,
        Backend::Avx512,
    ] {
        let math = UnaryMath::for_backend(backend);
        assert_eq!(math.erf_fma, backend.uses_fma());
        assert_eq!(math.inverse_gelu_scale, backend != Backend::Scalar);
        assert_eq!(math.sigmoid_vector_exp, backend != Backend::Scalar);
    }
    let input = GELU_INPUTS.map(f32::from_bits);
    assert_eq!(
        unary_vector(UnaryOp::Gelu, Simd::from_array(input))
            .to_array()
            .map(f32::to_bits),
        GELU_DEFAULT_DIVIDE
    );
    assert_eq!(
        input.map(|value| unary_value(UnaryOp::Gelu, value).to_bits()),
        GELU_DEFAULT_DIVIDE
    );
    assert_eq!(
        ERF_INPUTS.map(|value| unary_value(UnaryOp::Erf, f32::from_bits(value)).to_bits()),
        ERF_SEPARATE
    );
}
