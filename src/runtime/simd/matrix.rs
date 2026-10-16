//! Direct row-major dynamic attention GEMM. Four rows share a 32-column
//! coefficient walk; constant RHS panels retain their own packed consumers.
use super::*;

pub(crate) fn eligible(backend: Backend, rows: usize, columns: usize, packed: bool) -> bool {
    !packed
        && rows >= 4
        && columns >= 32
        && matches!(
            backend,
            Backend::Avx | Backend::AvxFma | Backend::Avx2Fma | Backend::Avx512
        )
}
pub(crate) fn actual_backend(backend: Backend) -> Backend {
    if backend == Backend::Avx512 {
        Backend::Avx2Fma
    } else {
        backend
    }
}
#[allow(clippy::too_many_arguments)]
pub(super) fn try_dynamic(
    backend: Backend,
    input: &[f32],
    weights: &[f32],
    packed: bool,
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) -> bool {
    if !eligible(backend, rows, columns, packed) {
        return false;
    }
    matrix_lengths(input, weights, output.len(), rows, inner, columns);
    let supported = match backend {
        Backend::Avx => avx_available(),
        Backend::AvxFma => avx_fma_available(),
        Backend::Avx2Fma => avx2_fma_available(),
        Backend::Avx512 => avx512_available(),
        _ => false,
    };
    assert!(
        supported,
        "dynamic MatMul backend {backend:?} is unavailable"
    );
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the complete matrix extents and requested native features were
    // checked above. Every load/store below uses bounded slices and safe tails.
    unsafe {
        match backend {
            Backend::Avx => avx(input, weights, output, rows, inner, columns),
            Backend::AvxFma => avx_fma(input, weights, output, rows, inner, columns),
            Backend::Avx2Fma | Backend::Avx512 => {
                avx2(input, weights, output, rows, inner, columns)
            }
            _ => unreachable!(),
        }
    }
    true
}
macro_rules! native {
    ($name:ident,$features:literal,$fused:literal) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable=$features)]
        unsafe fn $name(
            input: &[f32],
            weights: &[f32],
            output: &mut [f32],
            rows: usize,
            inner: usize,
            columns: usize,
        ) {
            run::<$fused>(input, weights, output, rows, inner, columns);
        }
    };
}
native!(avx, "avx", false);
native!(avx_fma, "avx,fma", true);
native!(avx2, "avx2,fma", true);
#[inline(always)]
fn madd<const FMA: bool>(
    value: Simd<f32, 8>,
    weight: Simd<f32, 8>,
    sum: Simd<f32, 8>,
) -> Simd<f32, 8> {
    if FMA {
        value.mul_add(weight, sum)
    } else {
        value * weight + sum
    }
}
#[inline(always)]
fn run<const FMA: bool>(
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    let mut row = 0;
    while row + 4 <= rows {
        let mut column = 0;
        while column + 32 <= columns {
            let mut a00 = Simd::<f32, 8>::splat(0.);
            let mut a01 = Simd::<f32, 8>::splat(0.);
            let mut a02 = Simd::<f32, 8>::splat(0.);
            let mut a03 = Simd::<f32, 8>::splat(0.);
            let mut a10 = Simd::<f32, 8>::splat(0.);
            let mut a11 = Simd::<f32, 8>::splat(0.);
            let mut a12 = Simd::<f32, 8>::splat(0.);
            let mut a13 = Simd::<f32, 8>::splat(0.);
            let mut a20 = Simd::<f32, 8>::splat(0.);
            let mut a21 = Simd::<f32, 8>::splat(0.);
            let mut a22 = Simd::<f32, 8>::splat(0.);
            let mut a23 = Simd::<f32, 8>::splat(0.);
            let mut a30 = Simd::<f32, 8>::splat(0.);
            let mut a31 = Simd::<f32, 8>::splat(0.);
            let mut a32 = Simd::<f32, 8>::splat(0.);
            let mut a33 = Simd::<f32, 8>::splat(0.);
            for k in 0..inner {
                let w0 = Simd::from_slice(&weights[k * columns + column..][..8]);
                let w1 = Simd::from_slice(&weights[k * columns + column + 8..][..8]);
                let w2 = Simd::from_slice(&weights[k * columns + column + 16..][..8]);
                let w3 = Simd::from_slice(&weights[k * columns + column + 24..][..8]);
                let v0 = Simd::splat(input[row * inner + k]);
                let v1 = Simd::splat(input[(row + 1) * inner + k]);
                let v2 = Simd::splat(input[(row + 2) * inner + k]);
                let v3 = Simd::splat(input[(row + 3) * inner + k]);
                a00 = madd::<FMA>(v0, w0, a00);
                a01 = madd::<FMA>(v0, w1, a01);
                a02 = madd::<FMA>(v0, w2, a02);
                a03 = madd::<FMA>(v0, w3, a03);
                a10 = madd::<FMA>(v1, w0, a10);
                a11 = madd::<FMA>(v1, w1, a11);
                a12 = madd::<FMA>(v1, w2, a12);
                a13 = madd::<FMA>(v1, w3, a13);
                a20 = madd::<FMA>(v2, w0, a20);
                a21 = madd::<FMA>(v2, w1, a21);
                a22 = madd::<FMA>(v2, w2, a22);
                a23 = madd::<FMA>(v2, w3, a23);
                a30 = madd::<FMA>(v3, w0, a30);
                a31 = madd::<FMA>(v3, w1, a31);
                a32 = madd::<FMA>(v3, w2, a32);
                a33 = madd::<FMA>(v3, w3, a33);
            }
            a00.copy_to_slice(&mut output[row * columns + column..][..8]);
            a01.copy_to_slice(&mut output[row * columns + column + 8..][..8]);
            a02.copy_to_slice(&mut output[row * columns + column + 16..][..8]);
            a03.copy_to_slice(&mut output[row * columns + column + 24..][..8]);
            a10.copy_to_slice(&mut output[(row + 1) * columns + column..][..8]);
            a11.copy_to_slice(&mut output[(row + 1) * columns + column + 8..][..8]);
            a12.copy_to_slice(&mut output[(row + 1) * columns + column + 16..][..8]);
            a13.copy_to_slice(&mut output[(row + 1) * columns + column + 24..][..8]);
            a20.copy_to_slice(&mut output[(row + 2) * columns + column..][..8]);
            a21.copy_to_slice(&mut output[(row + 2) * columns + column + 8..][..8]);
            a22.copy_to_slice(&mut output[(row + 2) * columns + column + 16..][..8]);
            a23.copy_to_slice(&mut output[(row + 2) * columns + column + 24..][..8]);
            a30.copy_to_slice(&mut output[(row + 3) * columns + column..][..8]);
            a31.copy_to_slice(&mut output[(row + 3) * columns + column + 8..][..8]);
            a32.copy_to_slice(&mut output[(row + 3) * columns + column + 16..][..8]);
            a33.copy_to_slice(&mut output[(row + 3) * columns + column + 24..][..8]);
            column += 32;
        }
        for column in (column..columns).step_by(8) {
            let count = (columns - column).min(8);
            let mut sums = [Simd::<f32, 8>::splat(0.); 4];
            for k in 0..inner {
                let weight = Simd::load_or_default(&weights[k * columns + column..][..count]);
                for (r, sum) in sums.iter_mut().enumerate() {
                    *sum = madd::<FMA>(Simd::splat(input[(row + r) * inner + k]), weight, *sum);
                }
            }
            for (r, sum) in sums.into_iter().enumerate() {
                output[(row + r) * columns + column..][..count]
                    .copy_from_slice(&sum.as_array()[..count]);
            }
        }
        row += 4;
    }
    for row in row..rows {
        for column in (0..columns).step_by(8) {
            let count = (columns - column).min(8);
            let mut sum = Simd::<f32, 8>::splat(0.);
            for k in 0..inner {
                let weight = Simd::load_or_default(&weights[k * columns + column..][..count]);
                sum = madd::<FMA>(Simd::splat(input[row * inner + k]), weight, sum);
            }
            output[row * columns + column..][..count].copy_from_slice(&sum.as_array()[..count]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dynamic_four_by_thirty_two_preserves_order_tails_and_no_packing() {
        for backend in [
            Backend::Avx,
            Backend::AvxFma,
            Backend::Avx2Fma,
            Backend::Avx512,
        ] {
            if !match backend {
                Backend::Avx => avx_available(),
                Backend::AvxFma => avx_fma_available(),
                Backend::Avx2Fma => avx2_fma_available(),
                Backend::Avx512 => avx512_available(),
                _ => false,
            } {
                continue;
            }
            for (rows, inner, columns) in [
                (4, 3, 32),
                (5, 65, 33),
                (7, 127, 47),
                (8, 17, 64),
                (9, 129, 67),
                (12, 128, 128),
            ] {
                let input: Vec<_> = (0..rows * inner)
                    .map(|i| [1e20, -1e20, 3., -0., 0.25][i % 5])
                    .collect();
                let weights: Vec<_> = (0..inner * columns)
                    .map(|i| ((i * 17 % 29) as f32 - 14.) / 16.)
                    .collect();
                let mut actual = vec![f32::NAN; rows * columns];
                assert!(try_dynamic(
                    backend,
                    &input,
                    &weights,
                    false,
                    &mut actual,
                    rows,
                    inner,
                    columns
                ));
                for row in 0..rows {
                    for column in 0..columns {
                        let mut sum = 0.;
                        for k in 0..inner {
                            sum = if backend == Backend::Avx {
                                input[row * inner + k] * weights[k * columns + column] + sum
                            } else {
                                input[row * inner + k].mul_add(weights[k * columns + column], sum)
                            };
                        }
                        assert_eq!(
                            actual[row * columns + column].to_bits(),
                            sum.to_bits(),
                            "{backend:?}/{rows}/{inner}/{columns} at {row},{column}"
                        );
                    }
                }
            }
        }
        assert!(!eligible(Backend::Portable4, 8, 64, false));
        assert!(!eligible(Backend::Avx2Fma, 3, 64, false));
        assert!(!eligible(Backend::Avx2Fma, 4, 31, false));
        assert!(!eligible(Backend::Avx2Fma, 4, 32, true));
        assert_eq!(actual_backend(Backend::Avx512), Backend::Avx2Fma);
    }
    #[test]
    fn forced_dynamic_native_policy_rejects_missing_features() {
        if !avx512_available() {
            assert!(
                std::panic::catch_unwind(|| {
                    try_dynamic(
                        Backend::Avx512,
                        &[1.; 16],
                        &[1.; 128],
                        false,
                        &mut [0.; 128],
                        4,
                        4,
                        32,
                    )
                })
                .is_err()
            );
        }
    }
}

#[cfg(test)]
mod broadcast_tests {
    use super::*;
    #[test]
    fn dynamic_four_by_thirty_two_retains_broadcast_batch_offsets() {
        if !avx2_fma_available() {
            return;
        }
        let (_, kind) = crate::runtime::kernels::bind_matmul(
            crate::tensor::Shape::new(&[2, 1, 5, 7]).unwrap(),
            crate::tensor::Shape::new(&[1, 3, 7, 33]).unwrap(),
        )
        .unwrap();
        let input: Vec<_> = (0..2 * 5 * 7)
            .map(|i| (i % 17) as f32 / 16. - 0.5)
            .collect();
        let weights: Vec<_> = (0..3 * 7 * 33)
            .map(|i| (i % 23) as f32 / 16. - 0.5)
            .collect();
        let mut output = vec![0.; 2 * 3 * 5 * 33];
        crate::runtime::kernels::execute_optimized(
            &kind,
            &[crate::model::TensorId(0), crate::model::TensorId(1)],
            |id| if id.0 == 0 { &input } else { &weights },
            &mut output,
            Backend::Avx2Fma,
            None,
            &crate::cpu::CpuContext::serial(),
            1,
        );
        for a in 0..2 {
            for b in 0..3 {
                for row in 0..5 {
                    for column in 0..33 {
                        let mut expected = 0.;
                        for k in 0..7 {
                            expected = input[(a * 5 + row) * 7 + k]
                                .mul_add(weights[(b * 7 + k) * 33 + column], expected);
                        }
                        assert_eq!(
                            output[((a * 3 + b) * 5 + row) * 33 + column].to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        }
    }
}
