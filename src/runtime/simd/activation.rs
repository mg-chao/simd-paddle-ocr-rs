//! Shared transcendental programs. Each native boundary owns its ISA so a
//! convolution dispatcher need not inline this code to retain vector codegen.
use super::*;

#[inline(always)]
pub(super) fn complex<const LANES: usize>(
    operation: UnaryOp,
    value: Simd<f32, LANES>,
    math: UnaryMath,
) -> Simd<f32, LANES> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    match math.backend {
        Some(Backend::Avx) => {
            checked_avx();
            // SAFETY: the guard validates the entry's target features.
            return unsafe { avx(operation, value, math) };
        }
        Some(Backend::AvxFma) => {
            checked_avx_fma();
            // SAFETY: the guard validates AVX and FMA.
            return unsafe { avx_fma(operation, value, math) };
        }
        Some(Backend::Avx2Fma) => {
            checked_avx2();
            // SAFETY: the guard validates AVX2 and FMA.
            return unsafe { avx2(operation, value, math) };
        }
        Some(Backend::Avx512) => {
            checked_avx512();
            // SAFETY: the guard validates AVX-512F, AVX2 and FMA.
            return unsafe { avx512(operation, value, math) };
        }
        _ => {}
    }
    portable(operation, value, math)
}

#[inline(never)]
fn portable<const LANES: usize>(
    operation: UnaryOp,
    value: Simd<f32, LANES>,
    math: UnaryMath,
) -> Simd<f32, LANES> {
    program(operation, value, math)
}

macro_rules! native {
    ($name:ident, $features:literal) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = $features)]
        #[inline(never)]
        unsafe fn $name<const LANES: usize>(
            operation: UnaryOp,
            value: Simd<f32, LANES>,
            math: UnaryMath,
        ) -> Simd<f32, LANES> {
            program(operation, value, math)
        }
    };
}

native!(avx, "avx");
native!(avx_fma, "avx,fma");
native!(avx2, "avx2,fma");
native!(avx512, "avx512f,avx2,fma");

// Inline only into the small, explicitly attributed entries above. There is
// one program per lane width/ISA, independent of convolution geometry/stores.
#[inline(always)]
fn program<const LANES: usize>(
    operation: UnaryOp,
    value: Simd<f32, LANES>,
    math: UnaryMath,
) -> Simd<f32, LANES> {
    match operation {
        UnaryOp::Erf => transcendental::erf_with_fma(value, math.erf_fma),
        UnaryOp::Sigmoid => {
            let one = Simd::splat(1.0);
            let zero = Simd::splat(0.0);
            if math.sigmoid_vector_exp {
                let exponent = transcendental::exp(-value.abs());
                let positive = one / (one + exponent);
                value.simd_lt(zero).select(one - positive, positive)
            } else {
                let exponent = Simd::from_array((-value.abs()).to_array().map(exp_f32));
                let denominator = one + exponent;
                value
                    .simd_ge(zero)
                    .select(one / denominator, exponent / denominator)
            }
        }
        UnaryOp::Gelu => {
            #[allow(clippy::excessive_precision, clippy::approx_constant)]
            const REFERENCE_INVERSE_SQRT_TWO: f32 = 0.70710678118654752;
            let scaled = if math.inverse_gelu_scale {
                value * Simd::splat(REFERENCE_INVERSE_SQRT_TWO)
            } else {
                value / Simd::splat(std::f32::consts::SQRT_2)
            };
            let erf = transcendental::erf_with_fma(scaled, math.erf_fma);
            ((erf + Simd::splat(1.0)) * value) * Simd::splat(0.5)
        }
        _ => unreachable!("only transcendental activations cross this boundary"),
    }
}
