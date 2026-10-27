//! Exact output aliasing for request-owned elementwise operands.
use super::*;
use crate::tensor::MAX_RANK;

impl Plan {
    pub(super) fn alias_elementwise(&mut self, end: &[usize]) {
        for (index, step) in self.steps.iter().enumerate() {
            if self.values[step.output].alias.is_some() {
                continue;
            }
            let unary = matches!(
                &step.op,
                Operation::Activation(_) | Operation::LayerNorm { .. }
            ) || matches!(&step.op,
                Operation::Native { kind: BoundKind::Square | BoundKind::Sqrt |
                    BoundKind::Relu | BoundKind::Sigmoid | BoundKind::Erf |
                    BoundKind::HardSigmoid(..), epilogues, .. } if epilogues.is_empty());
            let binary = matches!(&step.op,
                Operation::Native { kind: BoundKind::Binary(..), epilogues, .. }
                    if epilogues.is_empty());
            if !unary && !binary {
                continue;
            }
            let candidates = if binary { 2 } else { 1 };
            for &source in step.inputs.iter().take(candidates) {
                let input = &self.values[source];
                let output = &self.values[step.output];
                if input.constant.is_some()
                    || input.shape != output.shape
                    || input.layout != output.layout
                    || end[source] != index + 1
                {
                    continue;
                }
                let (root, delta) = self.root(source);
                let length = input.shape.element_count();
                // Reuse the entire group. A small Concat slice cannot acquire
                // ownership of its larger destination or a neighboring slice.
                if delta != 0
                    || self.values[root].constant.is_some()
                    || self.values.iter().enumerate().any(|(id, value)| {
                        let (parent, offset) = self.root(id);
                        parent == root
                            && (end[id] > index + 1
                                || offset
                                    .checked_add(value.shape.element_count())
                                    .is_none_or(|extent| extent > length))
                    })
                {
                    continue;
                }
                // The other operand may broadcast, but may not be a partial
                // view of the destination: earlier stores could destroy values
                // that a later broadcast read still needs. Exact self operands
                // are safe because both values are loaded before each store.
                if step.inputs.iter().any(|&id| {
                    let value = &self.values[id];
                    if value.constant.is_some() {
                        return false;
                    }
                    let (parent, offset) = self.root(id);
                    parent == root
                        && (offset != 0
                            || value.shape != output.shape
                            || value.layout != output.layout)
                }) {
                    continue;
                }
                // Input activations are request-owned copies, so a dying input
                // is eligible too. Constants and live graph outputs are not.
                self.values[step.output].alias = Some((source, 0));
                break;
            }
        }
    }
}

/// Execute a Binary whose full-shaped operand already occupies `out`.
/// `other == None` denotes an exact self operand; no shared borrow of `out`
/// is constructed. The caller has rejected every partial overlap beforehand.
#[allow(clippy::too_many_arguments)]
pub(super) fn binary_inplace(
    backend: Backend,
    op: BinaryOp,
    plan: &bound::Broadcast,
    info: impl Fn(usize) -> (Shape, Layout),
    alias_slot: usize,
    other: Option<&[f32]>,
    out: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
) {
    if out.is_empty() {
        return;
    }
    let operation = binary_op(&op);
    let other_left = alias_slot == 1;
    let (shape, layout) = info(alias_slot);
    let (other_shape, other_layout) = info(1 - alias_slot);
    if other
        .is_none_or(|values| values.len() == 1 || (other_shape == shape && other_layout == layout))
    {
        simd::parallel::binary_inplace(backend, operation, other, other_left, out, cpu, threads);
        return;
    }
    let other = other.expect("non-self broadcast operand");
    if let Some(channel) = plan.channel() {
        simd::binary_channel_inplace(
            backend,
            operation,
            other,
            channel,
            layout == Layout::Nhwc,
            out,
        );
        return;
    }
    // Canonical row scalars are contiguous in the descriptor's row order.
    // Keep that order when splitting bands, and broadcast once per SIMD row
    // instead of reconstructing logical coordinates for every element.
    if layout == Layout::Canonical
        && other_layout == Layout::Canonical
        && let Some(columns) = plan.row_scalar_columns()
    {
        let rows = out.len() / columns;
        assert_eq!(other.len(), rows);
        let workers = if out.len() >= 262_144 { threads } else { 1 };
        super::super::parallel::aligned_bands(
            out,
            columns,
            rows,
            workers,
            1,
            cpu,
            |band, first| {
                let scalars = &other[first..first + band.len() / columns];
                for (row, scalar) in band.chunks_exact_mut(columns).zip(scalars) {
                    simd::binary_inplace(
                        backend,
                        operation,
                        Some(std::slice::from_ref(scalar)),
                        other_left,
                        row,
                    );
                }
            },
        );
        return;
    }
    // MatMul vector biases repeat over contiguous canonical rows. Keep the
    // unbiased accumulation followed by Add, with SIMD only across elements.
    if layout == Layout::Canonical
        && other_shape.rank() == 1
        && shape.rank() > 1
        && other.len() == shape.dimensions()[shape.rank() - 1]
    {
        let columns = other.len();
        let rows = out.len() / columns;
        let workers = if out.len() >= 262_144 { threads } else { 1 };
        super::super::parallel::aligned_bands(out, columns, rows, workers, 1, cpu, |band, _| {
            for row in band.chunks_exact_mut(columns) {
                simd::binary_inplace(backend, operation, Some(other), other_left, row);
            }
        });
        return;
    }
    let strides = if alias_slot == 0 {
        plan.b_strides
    } else {
        plan.a_strides
    };
    let mut coordinates = [0usize; MAX_RANK];
    for (physical, value) in out.iter_mut().enumerate() {
        let mut logical = if layout == Layout::Nhwc {
            let d = shape.dimensions();
            let plane = d[2] * d[3];
            let pixel = physical / d[1];
            (pixel / plane * d[1] + physical % d[1]) * plane + pixel % plane
        } else {
            physical
        };
        for axis in (0..plan.rank).rev() {
            coordinates[axis] = logical % plan.dimensions[axis];
            logical /= plan.dimensions[axis];
        }
        let logical_other: usize = coordinates[..plan.rank]
            .iter()
            .zip(strides)
            .map(|(&coordinate, stride)| coordinate * stride)
            .sum();
        let other_index = if other_layout == Layout::Nhwc && other_shape.rank() == 4 {
            let d = other_shape.dimensions();
            let plane = d[2] * d[3];
            (logical_other / (d[1] * plane) * plane + logical_other % plane) * d[1]
                + logical_other / plane % d[1]
        } else {
            logical_other
        };
        let (a, b) = if other_left {
            (other[other_index], *value)
        } else {
            (*value, other[other_index])
        };
        *value = match op {
            BinaryOp::Add => a + b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Sub => a - b,
        };
    }
}
