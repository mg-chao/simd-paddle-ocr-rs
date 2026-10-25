//! Packed SVTR attention with an exact proof of every removed address transform.
use super::*;
use crate::runtime::{Op, graph_patterns as patterns};
use operators::{Recipe, RecipeKind, ShaderKind};

#[path = "attention_native.rs"]
pub(super) mod native;

pub(super) const Q_SCALE: u32 = 1;
pub(super) const PACKED: u32 = 2;
pub(super) const TRANSPOSE_OUTPUT: u32 = 4;

struct Operand {
    source: TensorId,
    nodes: Vec<usize>,
    views: Vec<usize>,
    scale: Option<TensorId>,
}

fn walk(
    program: &Program,
    plan: &ShapePlan,
    start: usize,
    mut source: TensorId,
    query: bool,
) -> Option<Operand> {
    let mut nodes = Vec::new();
    let mut views = Vec::new();
    let mut scale = None;
    for _ in 0..16 {
        let Some(producer) = program.graph_uses.producer(source) else {
            break;
        };
        if producer >= start {
            return None;
        }
        let instruction = &program.instructions[producer];
        match &plan.steps[producer].kind {
            BoundKind::Copy => {
                if plan.shapes[source.0].element_count()
                    != plan.shapes[instruction.inputs[0].0].element_count()
                {
                    return None;
                }
                source = instruction.inputs[0];
            }
            BoundKind::Transpose(_) if views.len() < 3 => {
                views.push(producer);
                source = instruction.inputs[0];
            }
            _ if query
                && views.is_empty()
                && scale.is_none()
                && matches!(instruction.kind, Op::Mul) =>
            {
                let mut pair = instruction.inputs.iter().copied();
                let (a, b) = (pair.next()?, pair.next()?);
                let (input, scalar) = if patterns::scalar(program, a).is_some() {
                    (b, a)
                } else if patterns::scalar(program, b).is_some() {
                    (a, b)
                } else {
                    break;
                };
                if plan.shapes[input.0] != plan.shapes[source.0] {
                    return None;
                }
                scale = Some(scalar);
                source = input;
            }
            _ => break,
        }
        nodes.push(producer);
    }
    Some(Operand {
        source,
        nodes,
        views,
        scale,
    })
}

fn view_matches(
    plan: &ShapePlan,
    node: usize,
    dimensions: &[usize],
    strides: &[usize],
    base: usize,
) -> bool {
    let BoundKind::Transpose(view) = &plan.steps[node].kind else {
        return false;
    };
    view.rank == dimensions.len()
        && view.dimensions[..view.rank] == *dimensions
        && view.strides[..view.rank] == *strides
        && view.base == base
}

/// Extend only the existing three-operation online-attention core. Its scale,
/// dot, and online-softmax arithmetic stay intact; only movement is removed.
pub(super) fn replacement(
    program: &Program,
    plan: &ShapePlan,
    start: usize,
    selected: TensorId,
    base: &Recipe,
) -> Option<Recipe> {
    if !matches!(base.kind, RecipeKind::Compute(ShaderKind::Attention)) || base.inputs.len() != 3 {
        return None;
    }
    let [n, heads, tokens, channels] = *plan.shapes[base.inputs[0].0].dimensions() else {
        return None;
    };
    if [n, heads, tokens, channels].contains(&0)
        || channels > 32
        || plan.shapes[base.inputs[1].0].dimensions() != [n, heads, channels, tokens]
        || plan.shapes[base.inputs[2].0].dimensions() != [n, heads, tokens, channels]
        || plan.shapes[base.output.0].dimensions() != [n, heads, tokens, channels]
    {
        return None;
    }
    let hd = heads.checked_mul(channels)?;
    let packed_stride = hd.checked_mul(3)?;
    let batch_stride = tokens.checked_mul(packed_stride)?;
    let count = n.checked_mul(batch_stride)?;
    if count > u32::MAX as usize {
        return None;
    }
    let q = walk(program, plan, start, base.inputs[0], true)?;
    let k = walk(program, plan, start, base.inputs[1], false)?;
    let v = walk(program, plan, start, base.inputs[2], false)?;
    if q.source != k.source
        || q.source != v.source
        || q.views.len() != 2
        || k.views.len() != 3
        || v.views.len() != 2
        || !(plan.shapes[q.source.0].dimensions() == [n, tokens, packed_stride]
            || plan.shapes[q.source.0].dimensions() == [n, tokens, 3, heads, channels])
    {
        return None;
    }
    let slice_stride = n
        .checked_mul(heads)?
        .checked_mul(tokens)?
        .checked_mul(channels)?;
    let slice_dimensions = [1, n, heads, tokens, channels];
    let slice_strides = [
        slice_stride,
        heads * tokens * channels,
        tokens * channels,
        channels,
        1,
    ];
    let shared_dimensions = [3, n, heads, tokens, channels];
    let shared_strides = [hd, batch_stride, channels, packed_stride, 1];
    for (operand, role) in [(&q, 0), (&k, 1), (&v, 2)] {
        let slice = operand.views[usize::from(role == 1)];
        let shared = *operand.views.last()?;
        if !view_matches(
            plan,
            slice,
            &slice_dimensions,
            &slice_strides,
            role * slice_stride,
        ) || !view_matches(plan, shared, &shared_dimensions, &shared_strides, 0)
            || plan.shapes[program.instructions[shared].inputs[0].0].dimensions()
                != [n, tokens, 3, heads, channels]
        {
            return None;
        }
    }
    if !view_matches(
        plan,
        k.views[0],
        &[n, heads, channels, tokens],
        &[heads * tokens * channels, tokens * channels, 1, channels],
        0,
    ) {
        return None;
    }
    // Identical packed sources alone are insufficient: every private value
    // must be one of the proven Slice/Transpose/linear Copy chains above.
    let mut recipe = base.clone();
    recipe.kind = RecipeKind::Compute(ShaderKind::AttentionViews);
    recipe.inputs = vec![q.source; 3];
    recipe.words[2] = PACKED;
    recipe.words[9..15].fill(0);
    recipe.words[15] = heads as u32;
    if let Some(scale) = q.scale {
        recipe.inputs.push(scale);
        recipe.words[2] |= Q_SCALE;
    }
    for operand in [&q, &k, &v] {
        recipe.nodes.extend(&operand.nodes);
    }
    // Fold precisely [N,H,T,D] -> [N,T,H,D]. Other output views retain their
    // own materialization; a selected intermediate always remains available.
    if base.output != selected
        && let Some(next) = program
            .graph_uses
            .unique_consumer(base.output, plan.steps.len())
        && view_matches(
            plan,
            next,
            &[n, tokens, heads, channels],
            &[heads * tokens * channels, channels, tokens * channels, 1],
            0,
        )
    {
        recipe.nodes.push(next);
        recipe.node = next;
        recipe.output = program.instructions[next].output;
        recipe.words[2] |= TRANSPOSE_OUTPUT;
    }
    recipe.nodes.sort_unstable();
    recipe.nodes.dedup();
    patterns::closed_group(
        program,
        &recipe.nodes,
        recipe.output,
        selected,
        plan.steps.len(),
    )
    .then_some(recipe)
}

#[cfg(test)]
#[path = "attention_packed_tests.rs"]
mod tests;
