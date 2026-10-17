//! NCHW pooling and spatial reduction families. Trailing-axis LayerNorm
//! statistics deliberately remain in the sequential reference executor.
use super::*;

#[derive(Clone, Copy)]
enum Task {
    MaxPool {
        height: usize,
        width: usize,
    },
    AveragePool {
        height: usize,
        width: usize,
        output_height: usize,
        output_width: usize,
    },
    SpatialMean {
        spatial: usize,
    },
}

pub(crate) fn max_pool(
    backend: Backend,
    input: &[f32],
    output: &mut [f32],
    height: usize,
    width: usize,
) {
    assert!(height > 0 && width > 0 && input.len().is_multiple_of(height * width));
    assert_eq!(input.len(), output.len());
    execute(backend, Task::MaxPool { height, width }, input, output);
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn average_pool(
    backend: Backend,
    input: &[f32],
    output: &mut [f32],
    height: usize,
    width: usize,
    output_height: usize,
    output_width: usize,
) {
    assert!(height > 0 && width > 0 && input.len().is_multiple_of(height * width));
    assert_eq!(
        input.len() / (height * width) * output_height * output_width,
        output.len()
    );
    execute(
        backend,
        Task::AveragePool {
            height,
            width,
            output_height,
            output_width,
        },
        input,
        output,
    );
}
pub(crate) fn spatial_mean(backend: Backend, input: &[f32], output: &mut [f32], spatial: usize) {
    assert!(spatial > 0);
    assert_eq!(
        input.len(),
        output
            .len()
            .checked_mul(spatial)
            .expect("mean size overflow")
    );
    execute(backend, Task::SpatialMean { spatial }, input, output);
}
fn execute(backend: Backend, task: Task, input: &[f32], output: &mut [f32]) {
    match backend {
        Backend::Scalar => kernel::<1, false>(task, input, output),
        Backend::Portable4 | Backend::NeonFma => kernel::<4, false>(task, input, output),
        Backend::Avx => {
            checked_avx();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: AVX was checked immediately above.
            unsafe {
                avx(task, input, output);
            }
        }
        Backend::AvxFma => {
            checked_avx_fma();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: AVX and FMA were checked immediately above.
            unsafe {
                avx_fma(task, input, output);
            }
        }
        Backend::Avx2Fma => {
            checked_avx2();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: AVX2 and FMA were checked immediately above.
            unsafe {
                avx2(task, input, output);
            }
        }
        Backend::Avx512 => {
            checked_avx512();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: AVX-512F and fallback features were checked above.
            unsafe {
                avx512(task, input, output);
            }
        }
    }
}
macro_rules! x86_kernel {
    ($name:ident,$features:literal,$lanes:literal) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable=$features)]
        unsafe fn $name(task: Task, input: &[f32], output: &mut [f32]) {
            kernel::<$lanes, true>(task, input, output);
        }
    };
}
x86_kernel!(avx, "avx", 8);
x86_kernel!(avx_fma, "avx,fma", 8);
x86_kernel!(avx2, "avx2,fma", 8);
x86_kernel!(avx512, "avx512f,avx2,fma", 16);

#[inline(always)]
fn kernel<const LANES: usize, const PAIRS: bool>(task: Task, input: &[f32], output: &mut [f32]) {
    match task {
        Task::MaxPool { height, width } => {
            let plane = height * width;
            for (source, dst) in input
                .chunks_exact(plane)
                .zip(output.chunks_exact_mut(plane))
            {
                for y in 0..height {
                    let row = y * width;
                    let bottom = (y + 1).min(height - 1) * width;
                    let end = (width - 1) / LANES * LANES;
                    for x in (0..end).step_by(LANES) {
                        let a = Simd::<f32, LANES>::from_slice(&source[row + x..row + x + LANES]);
                        let b = Simd::from_slice(&source[row + x + 1..row + x + 1 + LANES]);
                        let c = Simd::from_slice(&source[bottom + x..bottom + x + LANES]);
                        let d = Simd::from_slice(&source[bottom + x + 1..bottom + x + 1 + LANES]);
                        a.simd_max(b)
                            .simd_max(c.simd_max(d))
                            .copy_to_slice(&mut dst[row + x..row + x + LANES]);
                    }
                    for x in end..width - 1 {
                        dst[row + x] = source[row + x]
                            .max(source[row + x + 1])
                            .max(source[bottom + x].max(source[bottom + x + 1]));
                    }
                    dst[row + width - 1] = source[row + width - 1].max(source[bottom + width - 1]);
                }
            }
        }
        Task::AveragePool {
            height,
            width,
            output_height,
            output_width,
        } => {
            for (source, dst) in input
                .chunks_exact(height * width)
                .zip(output.chunks_exact_mut(output_height * output_width))
            {
                for y in 0..output_height {
                    let end = output_width / LANES * LANES;
                    for x in (0..end).step_by(LANES) {
                        let mut sum = Simd::<f32, LANES>::splat(0.);
                        for ky in 0..3 {
                            for kx in 0..2 {
                                sum += Simd::from_array(std::array::from_fn(|lane| {
                                    source[(y * 3 + ky) * width + (x + lane) * 2 + kx]
                                }));
                            }
                        }
                        (sum / Simd::splat(6.)).copy_to_slice(
                            &mut dst[y * output_width + x..y * output_width + x + LANES],
                        );
                    }
                    for x in end..output_width {
                        let mut sum = 0.;
                        for ky in 0..3 {
                            for kx in 0..2 {
                                sum += source[(y * 3 + ky) * width + x * 2 + kx];
                            }
                        }
                        dst[y * output_width + x] = sum / 6.;
                    }
                }
            }
        }
        Task::SpatialMean { spatial } => {
            for (source, dst) in input.chunks_exact(spatial).zip(output) {
                let mut a = Simd::<f32, LANES>::splat(0.);
                let mut b = Simd::<f32, LANES>::splat(0.);
                let mut i = 0;
                let stride = if PAIRS { 2 * LANES } else { LANES };
                while i + stride <= spatial {
                    a += Simd::from_slice(&source[i..i + LANES]);
                    if PAIRS {
                        b += Simd::from_slice(&source[i + LANES..i + 2 * LANES]);
                    }
                    i += stride;
                }
                let mut sum = if PAIRS { a + b } else { a }
                    .to_array()
                    .into_iter()
                    .fold(0., |a, b| a + b);
                // The reference's AVX-512 remainder loads one additional
                // vector but folds its lanes sequentially into the scalar.
                if PAIRS && LANES == 16 && i + LANES <= spatial {
                    for &value in &source[i..i + LANES] {
                        sum += value;
                    }
                    i += LANES;
                }
                for &value in &source[i..] {
                    sum += value;
                }
                *dst = sum / spatial as f32;
            }
        }
    }
}
