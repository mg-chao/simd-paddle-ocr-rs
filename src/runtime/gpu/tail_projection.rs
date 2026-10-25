//! Canonical token output for a closed NTC residual/pointwise projection chain.
use super::*;
use crate::runtime::graph_patterns as patterns;
use operators::{Recipe, RecipeKind, ShaderKind};

pub(super) const ENTRY: &str = "pointwise_mpp64";
pub(super) const WORKGROUP: [u32; 3] = [128, 1, 1];

pub(super) fn source(storage: [GpuPrecision; 6]) -> RuntimeResult<String> {
    if storage != [GpuPrecision::MixedF16; 6] {
        return Err(RuntimeError::with_kind(
            ErrorKind::Unsupported,
            "token tail requires all-Half storage",
        ));
    }
    let source = half_native::pointwise_epilogue_source();
    let before = "const uint index = (batch * rows + row) * columns + column;";
    if source.matches(before).count() != 2 {
        return Err(RuntimeError::with_kind(
            ErrorKind::Unsupported,
            "token tail source no longer matches its qualified index proof",
        ));
    }
    // Both the full residual read and the final store use this index. Matrix
    // operands, Float bias seeding, Swish and accumulation remain unchanged.
    Ok(source.replace(
        before,
        "const uint index = (batch * columns + column) * rows + row;",
    ))
}

fn view(plan: &ShapePlan, node: usize, dimensions: &[usize], strides: &[usize]) -> bool {
    matches!(&plan.steps[node].kind, BoundKind::Transpose(v)
        if v.rank == dimensions.len() && v.dimensions[..v.rank] == *dimensions
            && v.strides[..v.rank] == *strides && v.base == 0)
}

fn candidate(
    program: &Program,
    plan: &ShapePlan,
    selected: TensorId,
    base: &Recipe,
    descriptors: &[GpuTensorDescriptor],
    limits: &wgpu::Limits,
) -> RuntimeResult<Option<Recipe>> {
    let mut recipe = base.clone();
    half_native::select(
        std::slice::from_mut(&mut recipe),
        descriptors,
        limits,
        GpuMatrixPrecision::F16,
    )?;
    let p = &recipe.words;
    if !matches!(
        recipe.kind,
        RecipeKind::Compute(ShaderKind::PointwiseNative64)
    ) || recipe.inputs.len() != 4
        || p[5] != 1
        || p[7] != 1
        || p[19] != 1
        || p[60] != 0
        || p[64] != 2
        || p[68..74] != [8, 0, 0, 1, 2, 0]
        || !half_native::known_pointwise_epilogue(
            ShaderKind::PointwiseNative64,
            [GpuPrecision::MixedF16; 6],
            p,
        )
    {
        return Ok(None);
    }
    let (n, c, t) = (p[2] as usize, p[4] as usize, p[8] as usize);
    let count = plan.steps.len();
    let Some(pre) = program
        .graph_uses
        .producer(recipe.inputs[3])
        .filter(|&i| i < count)
    else {
        return Ok(None);
    };
    if !view(plan, pre, &[n, c, 1, t], &[t * c, 1, t * c, c]) {
        return Ok(None);
    }
    let pre_input = program.instructions[pre].inputs[0];
    let Some(reshape) = program
        .graph_uses
        .producer(pre_input)
        .filter(|&i| i < count)
    else {
        return Ok(None);
    };
    let residual = program.instructions[reshape].inputs[0];
    if !matches!(plan.steps[reshape].kind, BoundKind::Copy)
        || plan.shapes[pre_input.0].dimensions() != [n, 1, t, c]
        || plan.shapes[residual.0].dimensions() != [n, t, c]
    {
        return Ok(None);
    }
    let Some(squeeze) = program.graph_uses.unique_consumer(recipe.output, count) else {
        return Ok(None);
    };
    let squeezed = plan.steps[squeeze].output;
    if !matches!(plan.steps[squeeze].kind, BoundKind::Copy)
        || plan.shapes[squeezed.0].dimensions() != [n, c, t]
    {
        return Ok(None);
    }
    let Some(post) = program.graph_uses.unique_consumer(squeezed, count) else {
        return Ok(None);
    };
    if !view(plan, post, &[n, t, c], &[c * t, 1, t]) {
        return Ok(None);
    }
    recipe.nodes.extend([reshape, pre, squeeze, post]);
    recipe.nodes.sort_unstable();
    recipe.nodes.dedup();
    recipe.node = *recipe.nodes.last().unwrap();
    recipe.output = plan.steps[post].output;
    recipe.inputs[3] = residual;
    if !patterns::closed_group(program, &recipe.nodes, recipe.output, selected, count)
        || !recipe
            .inputs
            .iter()
            .chain(std::iter::once(&recipe.output))
            .all(|id| {
                descriptors.get(id.0).is_some_and(|d| {
                    d.precision == GpuPrecision::MixedF16
                        && d.layout == GpuLayout::Canonical
                        && d.storage_offset == 0
                })
            })
    {
        return Ok(None);
    }
    recipe.kind = RecipeKind::Compute(ShaderKind::TailProjection);
    recipe.words[37] = 1;
    recipe.words[38] = 1;
    Ok(Some(recipe))
}

/// Compile before publishing any removed movement. Slot/alias allocation runs
/// afterwards over the final list, so source and residual lifetimes extend to
/// this dispatch and the two movement allocations can disappear entirely.
pub(super) fn select(
    program: &Program,
    plan: &ShapePlan,
    selected: TensorId,
    recipes: &mut Vec<Recipe>,
    descriptors: &[GpuTensorDescriptor],
    limits: &wgpu::Limits,
    admit: &mut dyn FnMut(&Recipe) -> RuntimeResult<bool>,
) -> RuntimeResult<()> {
    let mut i = 0;
    while i < recipes.len() {
        let Some(candidate) = candidate(program, plan, selected, &recipes[i], descriptors, limits)?
        else {
            i += 1;
            continue;
        };
        let members = &candidate.nodes;
        let overlaps = recipes
            .iter()
            .enumerate()
            .filter(|(_, r)| r.nodes.iter().any(|n| members.contains(n)))
            .map(|(j, _)| j)
            .collect::<Vec<_>>();
        if overlaps
            .iter()
            .any(|&j| recipes[j].nodes.iter().any(|n| !members.contains(n)))
            || !admit(&candidate)?
        {
            i += 1;
            continue;
        }
        // Place at the last original physical member. Every external operand
        // is available there; closed_group proves all removed outputs private.
        let end = *overlaps.last().unwrap();
        let mut result = Vec::with_capacity(recipes.len() - overlaps.len() + 1);
        for (j, r) in recipes.drain(..).enumerate() {
            if j == end {
                result.push(candidate.clone());
            }
            if !overlaps.contains(&j) {
                result.push(r);
            }
        }
        *recipes = result;
        i = 0;
    }
    Ok(())
}

#[cfg(test)]
#[path = "tail_projection_tests.rs"]
mod tests;
