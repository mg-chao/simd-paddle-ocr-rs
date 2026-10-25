//! Constant token projections: canonical rank-N × rank-2, without layout packs.
//!
//! Only explicit F16 matrix policy admits this family. Products and reductions
//! remain FP32. Half activation storage preserves every removed ONNX store
//! boundary, including the separate Sigmoid store inside Swish.
use super::*;
use crate::runtime::{Op, graph_patterns as patterns};
use operators::{Recipe, RecipeKind, ShaderKind};

#[cfg(test)]
#[path = "linear_native_tests.rs"]
mod tests;

pub(super) const WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

#[allow(clippy::too_many_arguments)]
pub(super) fn replacement(
    program: &Program,
    plan: &ShapePlan,
    start: usize,
    selected: TensorId,
    limits: &crate::runtime::gpu::GpuLimits,
    descriptors: &[GpuTensorDescriptor],
    matrix_precision: GpuMatrixPrecision,
) -> RuntimeResult<Option<Recipe>> {
    if matrix_precision != GpuMatrixPrecision::F16
        || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE[0]
        || limits.max_compute_workgroup_size_x < WORKGROUP_SIZE[0]
        || limits.max_compute_workgroup_storage_size < 256 * 17 * 4
        || !matches!(
            plan.steps[start].kind,
            BoundKind::MatMul { .. } | BoundKind::BatchedMatMul(_)
        )
    {
        return Ok(None);
    }
    let n = &program.instructions;
    let count = plan.steps.len();
    let inputs = &n[start].inputs;
    if inputs.len() != 2 || !constant(program, inputs[1]) {
        return Ok(None);
    }
    let a = plan.shapes[inputs[0].0];
    let b = plan.shapes[inputs[1].0];
    let output_shape = plan.shapes[n[start].output.0];
    if a.rank() < 2 || b.rank() != 2 {
        return Ok(None);
    }
    let inner = b.dimensions()[0];
    let columns = b.dimensions()[1];
    if inner < 16 || columns < 4 || a.dimensions().last() != Some(&inner) {
        return Ok(None);
    }
    let rows = a.element_count() / inner;
    if rows == 0 || rows > 16384 || output_shape.element_count() != rows * columns {
        return Ok(None);
    }
    let mut members = vec![start];
    let mut output = n[start].output;
    let mut bias = None;
    let mut residual = None;
    let mut swish = false;
    if output != selected
        && let Some(next) = program.graph_uses.unique_consumer(output, count)
        && matches!(n[next].kind, Op::Add)
        && let Some(other) = patterns::other_binary(&n[next], output)
        && constant(program, other)
        && trailing_bias(plan.shapes[other.0], columns)
        && plan.shapes[n[next].output.0] == output_shape
    {
        bias = Some(other);
        output = n[next].output;
        members.push(next);
    }
    if output != selected {
        // Swish has two uses of its input, so it cannot use unique_consumer.
        let sigmoid = n[..count].iter().enumerate().find_map(|(i, instruction)| {
            (i > *members.last().unwrap()
                && matches!(instruction.kind, Op::Sigmoid)
                && instruction.inputs[0] == output)
                .then_some(i)
        });
        if let Some(sigmoid) = sigmoid
            && let Some(end) = patterns::swish(program, sigmoid, output, count)
        {
            let mut candidate = members.clone();
            candidate.extend(sigmoid..=end);
            if patterns::closed_group(program, &candidate, n[end].output, selected, count) {
                members = candidate;
                output = n[end].output;
                swish = true;
            }
        }
        if !swish
            && let Some(next) = program.graph_uses.unique_consumer(output, count)
            && matches!(n[next].kind, Op::Add)
            && let Some(other) = patterns::other_binary(&n[next], output)
            && other != output
            && plan.shapes[other.0] == output_shape
            && plan.shapes[n[next].output.0] == output_shape
        {
            residual = Some(other);
            output = n[next].output;
            members.push(next);
        }
    }
    if !patterns::closed_group(program, &members, output, selected, count) {
        return Ok(None);
    }
    // Fill dummy read-only slots with A so their storage specialization is
    // explicit and stable; the unused pointers are never dereferenced.
    let bindings = vec![
        inputs[0],
        inputs[1],
        bias.unwrap_or(inputs[0]),
        residual.unwrap_or(inputs[0]),
        inputs[0],
    ];
    let precision = descriptors[inputs[0].0].precision;
    if !bindings.iter().chain(std::iter::once(&output)).all(|id| {
        descriptors.get(id.0).is_some_and(|descriptor| {
            descriptor.precision == precision && descriptor.layout == GpuLayout::Canonical
        })
    }) {
        return Ok(None);
    }
    let tiles = rows
        .div_ceil(4)
        .checked_mul(columns.div_ceil(4))
        .ok_or_else(|| resource("native token projection tile count overflow"))?;
    let split = inner >= 64 && tiles <= 16384;
    let groups = if split {
        tiles.div_ceil(16)
    } else {
        tiles.div_ceil(256)
    };
    // dispatch_grid accepts a count of 64-thread groups, while this native
    // kernel has 256 threads. Supply one logical group for each physical group.
    let (x, y) = dispatch_grid(
        groups
            .checked_mul(64)
            .ok_or_else(|| resource("native token projection dispatch overflow"))?,
        limits,
    )?;
    let mut words = [0; 128];
    words[0] =
        u32::try_from(tiles).map_err(|_| resource("native token projection tiles exceed U32"))?;
    words[1] = x;
    for (word, value) in [(3, rows), (4, inner), (5, columns)] {
        words[word] = u32::try_from(value)
            .map_err(|_| resource("native token projection geometry exceeds U32"))?;
    }
    for value in [
        a.element_count(),
        b.element_count(),
        output_shape.element_count(),
    ] {
        u32::try_from(value)
            .map_err(|_| resource("native token projection storage exceeds U32"))?;
    }
    words[19] = u32::from(bias.is_some());
    words[20] = u32::from(swish);
    words[21] = u32::from(residual.is_some());
    words[23] = u32::from(split);
    words[38] = u32::from(precision == GpuPrecision::MixedF16);
    Ok(Some(Recipe {
        node: *members.last().unwrap(),
        nodes: members,
        kind: RecipeKind::Compute(ShaderKind::LinearNative),
        inputs: bindings,
        output,
        words,
        x,
        y,
    }))
}

fn constant(program: &Program, id: TensorId) -> bool {
    matches!(
        program.model.graph().tensors[id.0].constant,
        Some(TensorData::F32(_))
    )
}

fn trailing_bias(shape: Shape, columns: usize) -> bool {
    shape.dimensions().last() == Some(&columns)
        && shape.dimensions()[..shape.rank() - 1]
            .iter()
            .all(|&d| d == 1)
}
