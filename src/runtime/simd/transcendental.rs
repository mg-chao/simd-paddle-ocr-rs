//! Float32 vector approximations from the pinned reference's Tensor.Avx.cs.
//! Erf's interval fast paths evaluate the same polynomial as mixed vectors;
//! changing the tile width or worker partition cannot change a lane's result.
//!
//! Polynomial FMA follows the selected unary policy. Forced AVX and portable
//! kernels retain separate operations. Exp remains nonfused for every policy.

use std::simd::{
    Select, Simd, StdFloat,
    cmp::SimdPartialOrd,
    num::{SimdFloat, SimdInt},
};

// Keep the reference's coefficient literals for auditing against revision
// 33ff8a81672452041d110e8ba52dcd2639e54228, rather than rounding their text.
#[allow(clippy::excessive_precision)]
const SMALL: [f32; 9] = [
    1.0590875083315439e-6,
    -1.3906452410711274e-5,
    1.1955437252428243e-4,
    -8.542475960079766e-4,
    5.223771899427153e-3,
    -2.686612888828867e-2,
    1.128379122662546e-1,
    -3.761263888304388e-1,
    1.1283791670929921,
];
#[allow(clippy::excessive_precision)]
const MIDDLE: [f32; 9] = [
    -2.400667527836574e-3,
    -3.8855162788028288e-3,
    1.9332860298601401e-2,
    -1.501360723487143e-2,
    -4.4599369472787913e-2,
    1.3876136033191724e-1,
    -1.7839541988688759e-1,
    1.1893013063163335e-1,
    9.661051464140682e-1,
];
#[allow(clippy::excessive_precision)]
const LARGE: [f32; 9] = [
    -8.875076532056503e-5,
    3.880528563007222e-4,
    -7.781201071142843e-4,
    1.0255980996254569e-3,
    -1.0307062838910627e-3,
    7.858608012972956e-4,
    -4.193605385522123e-4,
    1.3951109720745927e-4,
    9.999779388683872e-1,
];

#[inline(always)]
fn polynomial<const LANES: usize, const FUSED: bool>(
    x: Simd<f32, LANES>,
    coefficients: &[f32; 9],
) -> Simd<f32, LANES> {
    let mut result = Simd::splat(coefficients[0]);
    for &coefficient in &coefficients[1..] {
        result = if FUSED {
            x.mul_add(result, Simd::splat(coefficient))
        } else {
            x * result + Simd::splat(coefficient)
        };
    }
    result
}

#[inline(always)]
pub(super) fn erf<const LANES: usize>(value: Simd<f32, LANES>) -> Simd<f32, LANES> {
    erf_impl::<LANES, false>(value)
}

#[inline(always)]
pub(super) fn erf_with_fma<const LANES: usize>(
    value: Simd<f32, LANES>,
    fused: bool,
) -> Simd<f32, LANES> {
    if fused {
        erf_impl::<LANES, true>(value)
    } else {
        erf_impl::<LANES, false>(value)
    }
}

#[inline(always)]
fn erf_impl<const LANES: usize, const FUSED: bool>(value: Simd<f32, LANES>) -> Simd<f32, LANES> {
    let magnitude = value.abs();
    let below_one = magnitude.simd_lt(Simd::splat(1.0));
    let below_two = magnitude.simd_lt(Simd::splat(2.0));
    let saturated = magnitude.simd_ge(Simd::splat(4.0));
    let one = Simd::splat(1.0);
    let result = if below_one.all() {
        magnitude * polynomial::<LANES, FUSED>(magnitude * magnitude, &SMALL)
    } else if saturated.all() {
        one
    } else if ((!below_one) & below_two).all() {
        polynomial::<LANES, FUSED>(magnitude - Simd::splat(1.5), &MIDDLE)
    } else if ((!below_two) & (!saturated)).all() {
        polynomial::<LANES, FUSED>(magnitude - Simd::splat(3.0), &LARGE)
    } else {
        let small = magnitude * polynomial::<LANES, FUSED>(magnitude * magnitude, &SMALL);
        let middle = polynomial::<LANES, FUSED>(magnitude - Simd::splat(1.5), &MIDDLE);
        let large = polynomial::<LANES, FUSED>(magnitude - Simd::splat(3.0), &LARGE);
        saturated.select(
            one,
            below_one.select(small, below_two.select(middle, large)),
        )
    };
    Simd::from_bits(result.to_bits() | (value.to_bits() & Simd::splat(0x8000_0000)))
}

/// The reference's non-fused float32 range reduction and degree-eight exp.
/// Used for optimized Sigmoid, including scalar tails, and complete softmax
/// vectors. Softmax scalar tails retain the scalar exp.
#[inline(always)]
#[allow(clippy::excessive_precision, clippy::approx_constant)]
pub(super) fn exp<const LANES: usize>(value: Simd<f32, LANES>) -> Simd<f32, LANES> {
    // Remove NaNs before unchecked conversion; restore the original payload
    // below. Infinities are safely clamped to the finite endpoint constants.
    let nan = value.is_nan();
    let bounded = nan.select(Simd::splat(0.0), value).simd_clamp(
        Simd::splat(-88.3762626647949),
        Simd::splat(88.3762626647949),
    );
    let scaled = bounded * Simd::splat(1.4426950408889634) + Simd::splat(0.5);
    // Match both pinned native and generic Vector range reduction. Unlike
    // floor(), this also remains vector arithmetic on SSE2 Portable4 paths.
    // SAFETY: NaNs were removed, |bounded| < 89 and log2e < 1.5. Product
    // rounding and +0.5 keep |scaled| < 135, finite and representable in i32.
    let integer = unsafe { scaled.to_int_unchecked::<i32>() }.cast::<f32>();
    let integer = integer
        - integer
            .simd_gt(scaled)
            .select(Simd::splat(1.0), Simd::splat(0.0));
    let reduced =
        (bounded - integer * Simd::splat(0.693359375)) - integer * Simd::splat(-2.12194440e-4);
    let square = reduced * reduced;
    let mut p = Simd::splat(2.4801587302e-5);
    for coefficient in [
        1.9841269841e-4,
        1.3888888889e-3,
        8.3333333333e-3,
        4.1666666667e-2,
        1.6666666667e-1,
        0.5,
    ] {
        p = Simd::splat(coefficient) + reduced * p;
    }
    p = Simd::splat(1.0) + (reduced + square * p);
    // SAFETY: the correction is at most one, yielding an exact finite
    // integer with magnitude < 136, still representable in i32.
    let exponent = unsafe { integer.to_int_unchecked::<i32>() };
    let bits = (exponent + Simd::splat(127)).cast::<u32>() << 23;
    let result = p * Simd::from_bits(bits);
    // Preserve invalid-input propagation instead of allowing clamping to turn
    // NaN into a finite probability. Runtime output validation rejects it.
    nan.select(value, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_independent_analytic_values_at_intervals_and_saturation() {
        // Mathematical erf values, rounded from high precision; independent of
        // either implementation's approximation and polynomial evaluation.
        for (input, expected) in [
            (0.0, 0.0),
            (0.25, 0.2763263901682369),
            (0.5, 0.5204998778130465),
            (1.0, 0.8427007929497149),
            (1.5, 0.9661051464753108),
            (2.0, 0.9953222650189527),
            (3.0, 0.9999779095030014),
            (4.0, 0.9999999845827421),
        ] {
            for sign in [-1.0, 1.0] {
                let actual = erf(Simd::<f32, 1>::splat(input * sign))[0];
                assert!(
                    (f64::from(actual) - expected * f64::from(sign)).abs() < 3e-7,
                    "erf({input}*{sign})={actual}"
                );
            }
        }
        assert_eq!(
            erf(Simd::<f32, 1>::splat(-0.0))[0].to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(erf(Simd::<f32, 1>::splat(f32::INFINITY))[0], 1.0);
        assert_eq!(erf(Simd::<f32, 1>::splat(f32::NEG_INFINITY))[0], -1.0);
        assert!(erf(Simd::<f32, 1>::splat(f32::NAN))[0].is_nan());
    }

    #[test]
    fn erf_homogeneous_and_mixed_tiles_have_identical_lane_bits() {
        let boundary = [
            0.0, -0.0, 1e-20, -1e-20, 0.99999994, 1.0, 1.0000001, 1.9999999, 2.0, 2.0000002,
            3.9999998, 4.0, 4.0000005, -5.0,
        ];
        for input in boundary {
            let solo = erf(Simd::<f32, 1>::splat(input))[0];
            let homogeneous = erf(Simd::<f32, 8>::splat(input));
            let mixed = erf(Simd::from_array([
                input, 0.25, -1.25, 2.5, 4.0, -0.0, -3.5, input,
            ]));
            assert_eq!(solo.to_bits(), homogeneous[0].to_bits());
            assert_eq!(solo.to_bits(), mixed[0].to_bits());
            assert_eq!(solo.to_bits(), mixed[7].to_bits());
        }
    }

    #[test]
    fn vector_exp_matches_f64_over_attention_range_with_safe_extremes() {
        for step in -640..=640 {
            let input = step as f32 * 0.125;
            let expected = f64::from(input).exp();
            let actual = exp(Simd::<f32, 8>::splat(input))[0];
            assert!(
                (f64::from(actual) - expected).abs() <= expected * 2e-6,
                "exp({input})={actual} expected {expected}"
            );
        }
        assert_eq!(exp(Simd::<f32, 8>::splat(-1000.0))[0], 0.0);
        assert!(exp(Simd::<f32, 8>::splat(f32::NAN))[0].is_nan());
    }
    #[test]
    fn exp_unchecked_conversion_is_guarded_for_nonfinite_and_clamp_edges() {
        // Finite expectations come from the maintained exact-rational pinned
        // coefficient/order oracle, not this Rust exp implementation. NaNs
        // retain the original payload; infinities use finite clamp endpoints.
        const CASES: [(u32, u32); 14] = [
            (0x7fc0_1234, 0x7fc0_1234),
            (0xffc0_4321, 0xffc0_4321),
            (0x7f80_0000, 0x7f35_04a4),
            (0xff80_0000, 0x0000_0000),
            (0x7f7f_ffff, 0x7f35_04a4),
            (0xff7f_ffff, 0x0000_0000),
            (0x42b0_c0a4, 0x7f35_044a),
            (0x42b0_c0a5, 0x7f35_04a4),
            (0x42b0_c0a6, 0x7f35_04a4),
            (0xc2b0_c0a4, 0x0000_0000),
            (0xc2b0_c0a5, 0x0000_0000),
            (0xc2b0_c0a6, 0x0000_0000),
            (0x0000_0000, 0x3f80_0000),
            (0x8000_0000, 0x3f80_0000),
        ];
        fn check<const LANES: usize>() {
            for (bits, expected) in CASES {
                let actual = exp(Simd::<f32, LANES>::splat(f32::from_bits(bits)));
                assert!(
                    actual
                        .to_array()
                        .iter()
                        .all(|value| value.to_bits() == expected)
                );
            }
            for offset in 0..CASES.len() {
                let input = Simd::<f32, LANES>::from_array(std::array::from_fn(|lane| {
                    f32::from_bits(CASES[(offset + lane) % CASES.len()].0)
                }));
                for (lane, actual) in exp(input).to_array().into_iter().enumerate() {
                    assert_eq!(actual.to_bits(), CASES[(offset + lane) % CASES.len()].1);
                }
            }
        }
        check::<1>();
        check::<4>();
        check::<8>();
        check::<16>();
    }
}
