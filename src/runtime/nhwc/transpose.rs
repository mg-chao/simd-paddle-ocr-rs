//! Single-channel 2x2/stride-two transpose convolution. Vector lanes own
//! independent source pixels, preserving bias-first and input-channel order.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn single(
    backend: Backend,
    p: &kernels::ConvTranspose,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    first_row: usize,
    store: &impl StoreTransform,
) {
    assert_eq!(p.cout, 1);
    assert!(output.len().is_multiple_of(4 * p.width));
    match backend {
        Backend::Scalar => kernel::<1, false>(p, input, weights, bias, output, first_row, store),
        Backend::Portable4 => kernel::<4, false>(p, input, weights, bias, output, first_row, store),
        Backend::NeonFma => kernel::<4, true>(p, input, weights, bias, output, first_row, store),
        Backend::Avx => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: the compiled model validated AVX.
            unsafe {
                avx(p, input, weights, bias, output, first_row, store);
            }
        }
        Backend::AvxFma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: the compiled model validated AVX/FMA.
            unsafe {
                avx_fma(p, input, weights, bias, output, first_row, store);
            }
        }
        Backend::Avx2Fma | Backend::Avx512 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: both compiled policies require AVX2 and FMA.
            unsafe {
                avx2(p, input, weights, bias, output, first_row, store);
            }
        }
    }
}
macro_rules! x86_kernel {
    ($name:ident,$features:literal,$fused:literal) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable=$features)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name(
            p: &kernels::ConvTranspose,
            input: &[f32],
            weights: &[f32],
            bias: &[f32],
            output: &mut [f32],
            first_row: usize,
            store: &impl StoreTransform,
        ) {
            kernel::<8, $fused>(p, input, weights, bias, output, first_row, store);
        }
    };
}
x86_kernel!(avx, "avx", false);
x86_kernel!(avx_fma, "avx,fma", true);
x86_kernel!(avx2, "avx2,fma", true);

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn kernel<const LANES: usize, const FUSED: bool>(
    p: &kernels::ConvTranspose,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    first_row: usize,
    store: &impl StoreTransform,
) {
    let rows = output.len() / (4 * p.width);
    assert!(first_row + rows <= p.batch * p.height);
    let initial = Simd::<f32, LANES>::splat(bias.first().copied().unwrap_or(0.));
    for row in 0..rows {
        for x in (0..p.width).step_by(LANES) {
            let count = (p.width - x).min(LANES);
            let mut sums = [initial; 4];
            for ci in 0..p.cin {
                let values = Simd::from_array(std::array::from_fn(|lane| {
                    input[((first_row + row) * p.width + x + lane.min(count - 1)) * p.cin + ci]
                }));
                for (tap, sum) in sums.iter_mut().enumerate() {
                    let weight = Simd::splat(weights[ci * 4 + tap]);
                    *sum = if FUSED {
                        values.mul_add(weight, *sum)
                    } else {
                        values * weight + *sum
                    };
                }
            }
            for (tap, sum) in sums.into_iter().enumerate() {
                for (lane, value) in sum.to_array().into_iter().enumerate().take(count) {
                    let at = (row * 2 + tap / 2) * p.width * 2 + (x + lane) * 2 + tap % 2;
                    output[at] = store.scalar(value, first_row * 4 * p.width + at);
                }
            }
        }
    }
}
