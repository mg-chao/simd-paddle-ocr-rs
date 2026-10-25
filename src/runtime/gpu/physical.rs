//! A GPU schedule with explicit dependencies and source lineage.
use super::*;
use crate::runtime::{Op, graph_patterns as patterns};
use operators::{Recipe, RecipeKind, ShaderKind};

#[allow(clippy::too_many_arguments)]
pub(super) fn lower(
    program: &Program,
    plan: &mut ShapePlan,
    selected: TensorId,
    limits: &crate::runtime::gpu::GpuLimits,
    precision: GpuPrecision,
    descriptors: &mut Vec<GpuTensorDescriptor>,
    matrix_precision: GpuMatrixPrecision,
    admit_matrix: &mut dyn FnMut(&Recipe) -> RuntimeResult<bool>,
) -> RuntimeResult<Vec<Recipe>> {
    let mut recipes = operators::recipes(program, plan, limits)?;
    let count = plan.steps.len();
    let mut removed = vec![false; count];
    let mut replacements = vec![None; count];
    for start in 0..count {
        if removed[start] {
            continue;
        }
        if let Some(recipe) = super::linear_native::replacement(
            program,
            plan,
            start,
            selected,
            limits,
            descriptors,
            matrix_precision,
        )? && recipe.nodes.iter().all(|&i| !removed[i])
            && let Some(recipe) = super::linear_mpp::admitted(
                recipe,
                descriptors,
                limits,
                matrix_precision,
                admit_matrix,
            )?
        {
            for &i in &recipe.nodes {
                removed[i] = true;
            }
            let end = recipe.node;
            replacements[end] = Some(recipe);
            continue;
        }
        if let Some(recipe) =
            super::fusions::replacement(program, plan, start, selected, limits, matrix_precision)?
            && recipe.nodes.iter().all(|&i| !removed[i])
        {
            for &i in &recipe.nodes {
                removed[i] = true;
            }
            let end = recipe.node;
            replacements[end] = Some(recipe);
            continue;
        }
        if let Some((inner, epsilon, gamma, beta)) =
            patterns::layer_norm(program, plan, start, count)
        {
            let members: Vec<_> = (start..start + 9).collect();
            let output = plan.steps[start + 8].output;
            if patterns::closed_group(program, &members, output, selected, count) {
                let mut r = recipes.iter().find(|r| r.node == start).unwrap().clone();
                r.inputs = vec![program.instructions[start].inputs[0], gamma, beta];
                r.output = output;
                r.nodes = members.clone();
                r.node = start + 8;
                r.kind = RecipeKind::Compute(ShaderKind::LayerNorm);
                r.words = [0; 128];
                r.words[0] = (plan.shapes[output.0].element_count() / inner) as u32;
                r.words[3] = inner as u32;
                r.words[4] = epsilon.to_bits();
                (r.x, r.y) = dispatch_grid(r.words[0] as usize * 64, limits)?;
                r.words[1] = r.x;
                for member in members {
                    removed[member] = true;
                }
                replacements[start + 8] = Some(r);
                continue;
            }
        }
        if !matches!(
            plan.steps[start].kind,
            BoundKind::Conv(_) | BoundKind::ConvTranspose(_)
        ) {
            continue;
        }
        let mut r = recipes.iter().find(|r| r.node == start).unwrap().clone();
        let shape = plan.shapes[r.output.0];
        let channels = shape.dimensions()[1];
        let plane = shape.element_count() / (shape.dimensions()[0] * channels);
        r.words[65] = channels as u32;
        r.words[66] = plane as u32;
        while r.inputs.len() < 3 {
            r.inputs.push(r.inputs[0]);
        }
        let mut members = vec![start];
        let mut value = r.output;
        loop {
            if value == selected || r.words[64] >= 20 {
                break;
            }
            let Some(next) =
                program.instructions[..count]
                    .iter()
                    .enumerate()
                    .find_map(|(i, op)| {
                        (i > *members.last().unwrap() && op.inputs.contains(&value)).then_some(i)
                    })
            else {
                break;
            };
            if removed[next] {
                break;
            }
            let mut candidate = members.clone();
            let mut packet = [0; 3];
            let mut operand = None;
            let end;
            if matrix_precision == GpuMatrixPrecision::F16
                && let Some(last) = patterns::swish(program, next, value, count)
            {
                end = last;
                packet[0] = 8;
                candidate.extend(next..=end);
            } else if let Some((act, last)) = patterns::activation(program, next, value, count) {
                end = last;
                packet[0] = match act {
                    patterns::Activation::Relu => 3,
                    patterns::Activation::Sigmoid => 4,
                    patterns::Activation::HardSwish => 6,
                    patterns::Activation::Gelu => 7,
                };
                candidate.extend(next..=end);
            } else if let Op::HardSigmoid { alpha, beta } = program.instructions[next].kind {
                end = next;
                packet = [5, alpha.to_bits(), beta.to_bits()];
                candidate.push(next);
            } else if matches!(program.instructions[next].kind, Op::Add) && r.inputs.len() < 5 {
                let Some(other) = patterns::other_binary(&program.instructions[next], value) else {
                    break;
                };
                let operand_shape = plan.shapes[other.0];
                let mode = if operand_shape.element_count() == 1 {
                    0
                } else if operand_shape == shape {
                    2
                } else if operand_shape.dimensions() == [channels, 1, 1]
                    || operand_shape.dimensions() == [1, channels, 1, 1]
                {
                    1
                } else {
                    break;
                };
                end = next;
                operand = Some(other);
                packet = [(r.inputs.len() - 2) as u32, mode, 0];
                candidate.push(next);
            } else {
                break;
            }
            let output = program.instructions[end].output;
            if plan.shapes[output.0] != shape
                || !patterns::closed_group(program, &candidate, output, selected, count)
            {
                break;
            }
            let base = 68 + r.words[64] as usize * 3;
            r.words[base..base + 3].copy_from_slice(&packet);
            r.words[64] += 1;
            if let Some(other) = operand {
                r.inputs.push(other);
            }
            value = output;
            members = candidate;
        }
        if members.len() > 1 {
            let end = *members.last().unwrap();
            for &member in &members {
                removed[member] = true;
            }
            r.node = end;
            r.nodes = members;
            r.output = value;
            replacements[end] = Some(r);
        }
    }
    let mut physical = Vec::new();
    for r in recipes.drain(..) {
        if let Some(replacement) = replacements[r.node].take() {
            physical.push(replacement);
        }
        if !removed[r.node] {
            physical.push(r);
        }
    }
    operators::specialize(&mut physical, plan, limits, precision)?;
    super::max_pool::select(&mut physical, plan, descriptors, limits, matrix_precision)?;
    let mut staged = Vec::new();
    for mut recipe in physical {
        if matches!(
            recipe.kind,
            RecipeKind::Compute(ShaderKind::Reduce | ShaderKind::Softmax)
        ) && recipe.words[3] > 4096
            && precision == GpuPrecision::F32
        {
            let softmax = matches!(recipe.kind, RecipeKind::Compute(ShaderKind::Softmax));
            let rows = recipe.words[0] as usize;
            let chunks = (recipe.words[3] as usize).div_ceil(1024);
            let scratch = TensorId(plan.shapes.len());
            plan.shapes.push(Shape::new(&[rows, chunks, 2])?);
            descriptors.push(GpuTensorDescriptor::canonical(
                *plan.shapes.last().unwrap(),
                precision,
            ));
            plan.slots.push(usize::MAX);
            let mut partial = recipe.clone();
            partial.output = scratch;
            partial.kind = RecipeKind::Compute(if softmax {
                ShaderKind::SoftmaxPartial
            } else {
                ShaderKind::ReducePartial
            });
            partial.words[4] = chunks as u32;
            (partial.x, partial.y) = dispatch_grid(rows * chunks * 64, limits)?;
            partial.words[1] = partial.x;
            recipe.kind = RecipeKind::Compute(if softmax {
                ShaderKind::SoftmaxFinish
            } else {
                ShaderKind::ReduceFinish
            });
            recipe.inputs.push(scratch);
            recipe.words[4] = chunks as u32;
            staged.push(partial);
        }
        staged.push(recipe);
    }
    let mut physical = super::conv_mpp_lower::lower(
        program,
        plan,
        staged,
        descriptors,
        limits,
        matrix_precision,
        admit_matrix,
    )?;
    let mut aliases: Vec<_> = (0..plan.shapes.len()).collect();
    if matrix_precision == GpuMatrixPrecision::F16 {
        super::tail_projection::select(
            program,
            plan,
            selected,
            &mut physical,
            descriptors,
            limits,
            admit_matrix,
        )?;
    }
    // Canonical movement operations preserve physical order. Extend the
    // source lifetime through every alias instead of copying its elements.
    physical.retain(|r| {
        if r.nodes.len() == 1
            && matches!(plan.steps[r.node].kind, BoundKind::Copy)
            && program.model.graph().tensors[r.inputs[0].0]
                .constant
                .is_none()
        {
            aliases[r.output.0] = aliases[r.inputs[0].0];
            false
        } else {
            true
        }
    });
    // Contiguous concatenations can place unique producer results directly.
    // No operand may read another member of the shared destination allocation.
    for (node, step) in plan.steps.iter().enumerate() {
        let BoundKind::Concat(concat) = &step.kind else {
            continue;
        };
        if concat.outer != 1 {
            continue;
        }
        let inputs = &program.instructions[node].inputs;
        if inputs.iter().any(|&id| {
            id == selected
                || aliases[id.0] != id.0
                || program.graph_uses.unique_consumer(id, count) != Some(node)
        }) {
            continue;
        }
        let producers: Option<Vec<_>> = inputs
            .iter()
            .map(|id| {
                let indices: Vec<_> = physical
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.output == *id)
                    .map(|(i, _)| i)
                    .collect();
                if indices.len() != 1 {
                    return None;
                }
                let i = indices[0];
                let r = &physical[i];
                (matches!(r.kind, RecipeKind::Compute(_))
                    && !r.inputs.iter().any(|id| inputs.contains(id)))
                .then_some(i)
            })
            .collect();
        let Some(producers) = producers else { continue };
        let mut offset = 0;
        for (&id, i) in inputs.iter().zip(producers) {
            physical[i].words[60] = offset as u32;
            aliases[id.0] = step.output.0;
            descriptors[id.0].storage_offset = offset;
            offset += plan.shapes[id.0].element_count();
        }
        physical.retain(|r| r.node != node);
    }
    // Resolve chains after all placements, and account for the complete union
    // lifetime. A storage slot is never reused while any alias is live.
    for id in 0..aliases.len() {
        let mut root = aliases[id];
        while root != aliases[root] {
            root = aliases[root];
        }
        aliases[id] = root;
    }
    allocate_slots(program, plan, &physical, selected, &aliases)?;
    Ok(physical)
}

fn allocate_slots(
    program: &Program,
    plan: &mut ShapePlan,
    recipes: &[Recipe],
    selected: TensorId,
    aliases: &[usize],
) -> RuntimeResult<()> {
    let mut first = vec![usize::MAX; plan.shapes.len()];
    let mut last = vec![0; first.len()];
    first[aliases[program.input.0]] = 0;
    for (i, r) in recipes.iter().enumerate() {
        first[aliases[r.output.0]] = first[aliases[r.output.0]].min(i + 1);
        last[aliases[r.output.0]] = last[aliases[r.output.0]].max(i + 1);
        for input in &r.inputs {
            last[aliases[input.0]] = last[aliases[input.0]].max(i + 1);
        }
    }
    last[aliases[selected.0]] = recipes.len() + 1;
    let mut values: Vec<_> = (0..first.len())
        .filter(|&i| {
            first[i] != usize::MAX
                && program
                    .model
                    .graph()
                    .tensors
                    .get(i)
                    .is_none_or(|t| t.constant.is_none())
        })
        .collect();
    values.sort_by_key(|&i| (first[i], i));
    plan.slots.fill(usize::MAX);
    plan.lengths.clear();
    let mut ends = Vec::new();
    for id in values {
        let length = plan.shapes[id].element_count();
        let slot = (0..ends.len())
            .filter(|&s| ends[s] < first[id] && plan.lengths[s] >= length)
            .min_by_key(|&s| plan.lengths[s])
            .or_else(|| {
                (0..ends.len())
                    .filter(|&s| ends[s] < first[id])
                    .max_by_key(|&s| plan.lengths[s])
            });
        let slot = slot.unwrap_or_else(|| {
            ends.push(0);
            plan.lengths.push(0);
            ends.len() - 1
        });
        plan.lengths[slot] = plan.lengths[slot].max(length);
        ends[slot] = last[id];
        plan.slots[id] = slot;
    }
    for (id, &root) in aliases.iter().enumerate() {
        plan.slots[id] = plan.slots[root];
    }
    plan.activation_bytes = plan
        .lengths
        .iter()
        .try_fold(0usize, |sum, n| sum.checked_add(n * 4))
        .ok_or_else(|| resource("GPU physical arena overflow"))?;
    Ok(())
}
