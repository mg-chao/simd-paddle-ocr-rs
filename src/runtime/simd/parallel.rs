//! Reference-aligned bounded workers for large independent tensor elements.
//! No allocation or retained state: each scoped band borrows disjoint output.
use super::{Backend, BinaryOp, UnaryOp};
use crate::cpu::CpuContext;

pub(crate) const THRESHOLD: usize = 1 << 18;

pub(crate) fn elementwise_workers(
    backend: Backend,
    length: usize,
    capacity: usize,
    threads: usize,
) -> usize {
    if backend == Backend::Scalar || length < THRESHOLD {
        1
    } else {
        threads.min(capacity).clamp(1, 16)
    }
}

pub(crate) fn unary_workers(
    backend: Backend,
    operation: UnaryOp,
    length: usize,
    capacity: usize,
    threads: usize,
) -> usize {
    if matches!(
        operation,
        UnaryOp::Relu | UnaryOp::Sigmoid | UnaryOp::Erf | UnaryOp::Sqrt | UnaryOp::Gelu
    ) {
        elementwise_workers(backend, length, capacity, threads)
    } else {
        1
    }
}

pub(crate) fn unary(
    backend: Backend,
    operation: UnaryOp,
    input: &[f32],
    output: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
) {
    assert_eq!(input.len(), output.len());
    let length = output.len();
    let workers = unary_workers(backend, operation, length, cpu.capacity(), threads);
    if workers == 1 {
        super::unary(backend, operation, input, output);
        return;
    }
    crate::runtime::parallel::aligned_bands(output, 1, length, workers, 16, cpu, |band, first| {
        super::unary(backend, operation, &input[first..first + band.len()], band);
    });
}

pub(crate) fn unary_inplace(
    backend: Backend,
    operation: UnaryOp,
    output: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
) {
    let length = output.len();
    let workers = unary_workers(backend, operation, length, cpu.capacity(), threads);
    if workers == 1 {
        super::unary_inplace(backend, operation, output);
        return;
    }
    crate::runtime::parallel::aligned_bands(output, 1, length, workers, 16, cpu, |band, _| {
        super::unary_inplace(backend, operation, band);
    });
}

pub(crate) fn binary(
    backend: Backend,
    operation: BinaryOp,
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
) {
    assert_eq!(left.len(), output.len());
    assert_eq!(right.len(), output.len());
    let length = output.len();
    let workers = elementwise_workers(backend, length, cpu.capacity(), threads);
    if workers == 1 {
        super::binary(backend, operation, left, right, output);
        return;
    }
    crate::runtime::parallel::aligned_bands(output, 1, length, workers, 16, cpu, |band, first| {
        let end = first + band.len();
        super::binary(
            backend,
            operation,
            &left[first..end],
            &right[first..end],
            band,
        );
    });
}

pub(crate) fn binary_inplace(
    backend: Backend,
    operation: BinaryOp,
    other: Option<&[f32]>,
    other_left: bool,
    output: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
) {
    if let Some(values) = other {
        assert!(values.len() == 1 || values.len() == output.len());
    }
    let length = output.len();
    // Only equal-shaped operands have the reference's parallel selector.
    // None reads each output value as both operands, preserving alias support.
    let workers = if other.is_none_or(|values| values.len() == length) {
        elementwise_workers(backend, length, cpu.capacity(), threads)
    } else {
        1
    };
    if workers == 1 {
        super::binary_inplace(backend, operation, other, other_left, output);
        return;
    }
    crate::runtime::parallel::aligned_bands(output, 1, length, workers, 16, cpu, |band, first| {
        let operand = other.map(|values| &values[first..first + band.len()]);
        super::binary_inplace(backend, operation, operand, other_left, band);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backends() -> Vec<Backend> {
        let mut result = vec![Backend::Scalar, Backend::Portable4];
        for (available, backend) in [
            (super::super::avx_available(), Backend::Avx),
            (super::super::avx_fma_available(), Backend::AvxFma),
            (super::super::avx2_fma_available(), Backend::Avx2Fma),
            (super::super::avx512_available(), Backend::Avx512),
        ] {
            if available {
                result.push(backend);
            }
        }
        result
    }
    fn cpu(threads: usize) -> CpuContext {
        CpuContext::new(if cfg!(feature = "parallel") {
            threads
        } else {
            1
        })
        .unwrap()
    }
    fn assert_bits(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (index, (left, right)) in left.iter().zip(right).enumerate() {
            assert_eq!(left.to_bits(), right.to_bits(), "index {index}");
        }
    }

    #[test]
    fn selectors_cover_reference_threshold_and_budget_boundaries() {
        for length in [0, 15, 16, 17, THRESHOLD - 1] {
            assert_eq!(elementwise_workers(Backend::Portable4, length, 8, 8), 1);
        }
        for length in [THRESHOLD, THRESHOLD + 1, THRESHOLD + 15, THRESHOLD + 16] {
            assert_eq!(elementwise_workers(Backend::Portable4, length, 8, 4), 4);
            assert_eq!(elementwise_workers(Backend::Portable4, length, 3, 8), 3);
            assert_eq!(elementwise_workers(Backend::Portable4, length, 32, 64), 16);
            assert_eq!(elementwise_workers(Backend::Scalar, length, 8, 8), 1);
            for operation in [
                UnaryOp::Relu,
                UnaryOp::Sigmoid,
                UnaryOp::Erf,
                UnaryOp::Sqrt,
                UnaryOp::Gelu,
            ] {
                assert_eq!(
                    unary_workers(Backend::Portable4, operation, length, 8, 8),
                    8
                );
            }
            for operation in [
                UnaryOp::Square,
                UnaryOp::HardSwish,
                UnaryOp::HardSigmoid {
                    alpha: 0.2,
                    beta: 0.5,
                },
            ] {
                assert_eq!(
                    unary_workers(Backend::Portable4, operation, length, 8, 8),
                    1
                );
            }
        }
    }

    #[test]
    fn unary_worker_bands_preserve_policy_bits_and_inplace_tails() {
        let length = THRESHOLD + 33;
        let contexts: Vec<_> = [1, 2, 4, 8]
            .into_iter()
            .map(|threads| (threads, cpu(threads)))
            .collect();
        let input: Vec<_> = (0..length)
            .map(|i| match i % 9 {
                0 => -0.0,
                1 => f32::from_bits(0x3f23_0c95),
                2 => f32::from_bits(0x3eb1_af60),
                _ => (i % 127) as f32 / 17.,
            })
            .collect();
        for backend in backends() {
            for operation in [
                UnaryOp::Relu,
                UnaryOp::Sigmoid,
                UnaryOp::Erf,
                UnaryOp::Sqrt,
                UnaryOp::Gelu,
            ] {
                let mut expected = vec![0.; length];
                super::super::unary(backend, operation, &input, &mut expected);
                for (threads, cpu) in &contexts {
                    let mut output = vec![f32::NAN; length];
                    unary(backend, operation, &input, &mut output, cpu, *threads);
                    assert_bits(&output, &expected);
                    output.copy_from_slice(&input);
                    unary_inplace(backend, operation, &mut output, cpu, *threads);
                    assert_bits(&output, &expected);
                }
            }
        }
    }

    #[test]
    fn full_binary_worker_bands_preserve_operand_order_aliases_and_scalar_fallback() {
        let length = THRESHOLD + 17;
        let contexts: Vec<_> = [1, 2, 4, 8]
            .into_iter()
            .map(|threads| (threads, cpu(threads)))
            .collect();
        let left: Vec<_> = (0..length).map(|i| (i % 131 + 1) as f32 / 11.).collect();
        let right: Vec<_> = (0..length).map(|i| (i % 137 + 2) as f32 / 13.).collect();
        for backend in backends() {
            for operation in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
                for (threads, cpu) in &contexts {
                    let mut expected = vec![0.; length];
                    let mut actual = vec![0.; length];
                    super::super::binary(backend, operation, &left, &right, &mut expected);
                    binary(
                        backend,
                        operation,
                        &left,
                        &right,
                        &mut actual,
                        cpu,
                        *threads,
                    );
                    assert_bits(&actual, &expected);
                    for other_left in [false, true] {
                        for operand in [None, Some(&right[..]), Some(&[1.25][..])] {
                            expected.copy_from_slice(&left);
                            actual.copy_from_slice(&left);
                            super::super::binary_inplace(
                                backend,
                                operation,
                                operand,
                                other_left,
                                &mut expected,
                            );
                            binary_inplace(
                                backend,
                                operation,
                                operand,
                                other_left,
                                &mut actual,
                                cpu,
                                *threads,
                            );
                            assert_bits(&actual, &expected);
                        }
                    }
                }
            }
        }
    }
}
