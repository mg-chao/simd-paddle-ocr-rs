//! Excitation gate application with separate ONNX channel-bias Add nodes.
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
    let n = &program.instructions;
    let BoundKind::Conv(g) = &plan.steps[start].kind else {
        return Ok(None);
    };
    if g.kh != 1
        || g.kw != 1
        || g.cpg != g.cin
        || g.height != 1
        || g.width != 1
        || g.oh != 1
        || g.ow != 1
        || g.sh != 1
        || g.sw != 1
        || g.pt != 0
        || g.pl != 0
        || n[start].inputs.len() != 2
    {
        return Ok(None);
    }
    let next = |id| program.graph_uses.unique_consumer(id, count);
    let Some(add) = next(n[start].output) else {
        return Ok(None);
    };
    if !matches!(n[add].kind, Op::Add) {
        return Ok(None);
    }
    let Some(bias) = patterns::other_binary(&n[add], n[start].output) else {
        return Ok(None);
    };
    let shape = plan.shapes[n[start].output.0];
    let bias_shape = plan.shapes[bias.0];
    let scalar = bias_shape.element_count() == 1;
    if !matches!(
        program.model.graph().tensors[bias.0].constant,
        Some(TensorData::F32(_))
    ) || !(scalar
        || bias_shape.dimensions() == [g.cout, 1, 1]
        || bias_shape.dimensions() == [1, g.cout, 1, 1])
        || plan.shapes[n[add].output.0] != shape
    {
        return Ok(None);
    }
    let Some(activation) = next(n[add].output) else {
        return Ok(None);
    };
    let (mode, alpha, beta) = match n[activation].kind {
        Op::Sigmoid => (0, 0.0f32, 0.0f32),
        Op::HardSigmoid { alpha, beta } => (1, alpha, beta),
        _ => return Ok(None),
    };
    let Some(apply) = next(n[activation].output) else {
        return Ok(None);
    };
    if !matches!(n[apply].kind, Op::Mul) {
        return Ok(None);
    }
    let Some(feature) = patterns::other_binary(&n[apply], n[activation].output) else {
        return Ok(None);
    };
    let feature_shape = plan.shapes[feature.0];
    if feature_shape.rank() != 4
        || feature_shape.dimensions()[..2] != [g.batch, g.cout]
        || plan.shapes[n[apply].output.0] != feature_shape
    {
        return Ok(None);
    }
    let members = vec![start, add, activation, apply];
    let output = n[apply].output;
    if !patterns::closed_group(program, &members, output, selected, count) {
        return Ok(None);
    }
    let groups = g
        .batch
        .checked_mul(g.cout)
        .ok_or_else(|| resource("excitation gate group count overflow"))?;
    let (x, y) = dispatch_grid(
        groups
            .checked_mul(64)
            .ok_or_else(|| resource("excitation gate dispatch size overflow"))?,
        limits,
    )?;
    let mut words = [0; 128];
    for (index, value) in [
        (0, groups),
        (3, g.cin),
        (4, g.cout),
        (5, feature_shape.element_count() / groups),
    ] {
        words[index] = u32::try_from(value)
            .map_err(|_| resource("excitation gate geometry exceeds U32 indexing"))?;
    }
    words[1] = x;
    words[7] = 1;
    words[8] = mode;
    words[9] = alpha.to_bits();
    words[10] = beta.to_bits();
    words[11] = u32::from(scalar);
    // The separate gate tensor is rounded to output precision before Mul.
    // Keep the existing physical Conv epilogue's F32 dot/bias/activation;
    // its only stored boundary is the gate consumed by the separate Mul.
    words[12] = 1;
    Ok(Some(Recipe {
        node: apply,
        nodes: members,
        kind: RecipeKind::Compute(ShaderKind::SqueezeExcite),
        inputs: vec![n[start].inputs[0], n[start].inputs[1], bias, feature],
        output,
        words,
        x,
        y,
    }))
}

#[cfg(test)]
#[path = "se_fusions_tests.rs"]
mod tests;
