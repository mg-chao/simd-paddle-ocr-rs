//! Standalone activation chains retain each removed storage-rounding boundary.
use super::*;
use crate::runtime::{Op, graph_patterns as patterns};
use operators::{Recipe, RecipeKind, ShaderKind};

pub(super) fn replacement(
    program: &Program,
    plan: &ShapePlan,
    start: usize,
    selected: TensorId,
    limits: &crate::runtime::gpu::GpuLimits,
) -> RuntimeResult<Option<Recipe>> {
    let count = plan.steps.len();
    let instruction = &program.instructions[start];
    if !matches!(
        instruction.kind,
        Op::Div | Op::HardSigmoid { .. } | Op::Sigmoid
    ) {
        return Ok(None);
    }
    let input = instruction.inputs[0];
    let (mode, end) = if let Some(end) = patterns::swish(program, start, input, count) {
        (3, end)
    } else if let Some((kind, end)) = patterns::activation(program, start, input, count) {
        match kind {
            patterns::Activation::Gelu => (1, end),
            patterns::Activation::HardSwish => (2, end),
            _ => return Ok(None),
        }
    } else {
        return Ok(None);
    };
    let output = program.instructions[end].output;
    let members: Vec<_> = (start..=end).collect();
    if plan.shapes[input.0] != plan.shapes[output.0]
        || !patterns::closed_group(program, &members, output, selected, count)
    {
        return Ok(None);
    }
    let elements = plan.shapes[output.0].element_count();
    let (x, y) = dispatch_grid(elements.div_ceil(4), limits)?;
    let mut words = [0; 128];
    words[0] = u32::try_from(elements)
        .map_err(|_| resource("fused activation element count exceeds U32 indexing"))?;
    words[1] = x
        .checked_mul(64)
        .ok_or_else(|| resource("fused activation dispatch stride overflow"))?;
    words[2] = 25;
    words[3] = mode;
    Ok(Some(Recipe {
        node: end,
        nodes: members,
        kind: RecipeKind::Compute(ShaderKind::General),
        inputs: if mode == 1 {
            vec![input, instruction.inputs[1]]
        } else {
            vec![input]
        },
        output,
        words,
        x,
        y,
    }))
}

#[cfg(test)]
#[path = "fused_activation_tests.rs"]
mod tests;
