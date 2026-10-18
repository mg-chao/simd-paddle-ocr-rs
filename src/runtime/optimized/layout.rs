//! Reference-eligible plane/channel bands for bit-preserving layout conversion.
use super::{Layout, Shape, simd};
use crate::cpu::CpuContext;

pub(super) fn workers(
    batch: usize,
    channels: usize,
    plane: usize,
    source: Layout,
    budget: usize,
) -> usize {
    let volume = batch.saturating_mul(channels).saturating_mul(plane);
    if volume < 262_144 || budget <= 1 {
        return 1;
    }
    let bands = match source {
        Layout::Canonical => plane / 64,
        Layout::Nhwc => channels / 8,
    };
    budget.min(bands.max(1))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn convert(
    input: &[f32],
    output: &mut [f32],
    shape: Shape,
    source: Layout,
    backend: simd::Backend,
    cpu: &CpuContext,
    threads: usize,
) {
    assert_eq!(input.len(), shape.element_count());
    assert_eq!(input.len(), output.len());
    if input.is_empty() {
        return;
    }
    let d = shape.dimensions();
    let plane = d[2] * d[3];
    let batch_elements = d[1] * plane;
    let workers = workers(d[0], d[1], plane, source, threads.min(cpu.capacity()));
    let (rows, columns) = if source == Layout::Canonical {
        (d[1], plane)
    } else {
        (plane, d[1])
    };
    cpu.install(|| {
        for (input, output) in input
            .chunks_exact(batch_elements)
            .zip(output.chunks_exact_mut(batch_elements))
        {
            super::super::parallel::aligned_bands(
                output,
                rows,
                columns,
                workers,
                8,
                cpu,
                |band, first| {
                    simd::transpose_matrix_columns(backend, input, band, rows, columns, first);
                },
            );
        }
    });
}

#[cfg(test)]
mod tests;
