//! Four adjacent canonical spatial outputs share MaxPool address arithmetic.
use super::*;
use operators::{Recipe, RecipeKind, ShaderKind};

pub(super) const OPCODE: u32 = 26;

pub(super) fn select(
    recipes: &mut [Recipe],
    plan: &ShapePlan,
    descriptors: &[GpuTensorDescriptor],
    limits: &crate::runtime::gpu::GpuLimits,
    matrix_precision: GpuMatrixPrecision,
) -> RuntimeResult<usize> {
    if matrix_precision != GpuMatrixPrecision::F16 {
        return Ok(0);
    }
    let mut selected = 0;
    for recipe in recipes {
        if !matches!(recipe.kind, RecipeKind::Compute(ShaderKind::General))
            || recipe.words[2] != 12
            || recipe.nodes.as_slice() != [recipe.node]
            || recipe.inputs.len() != 1
        {
            continue;
        }
        // The bound family is produced only for validated 2x2/stride-one,
        // trailing-only padding with identical input and output geometry.
        let Some(step) = plan.steps.get(recipe.node) else {
            continue;
        };
        let BoundKind::MaxPool { height, width } = &step.kind else {
            continue;
        };
        let Some(source) = descriptors.get(recipe.inputs[0].0) else {
            continue;
        };
        let Some(output) = descriptors.get(recipe.output.0) else {
            continue;
        };
        let [n, channels, h, w] = *source.shape.dimensions() else {
            continue;
        };
        if source.layout != GpuLayout::Canonical
            || output.layout != GpuLayout::Canonical
            || source.shape != output.shape
            || [n, channels, h, w].contains(&0)
            || (h, w) != (*height, *width)
            || recipe.words[3] as usize != h
            || recipe.words[4] as usize != w
            || recipe.words[0] as usize != output.shape.element_count()
        {
            continue;
        }
        let row_tiles = w.div_ceil(4);
        let tiles = n
            .checked_mul(channels)
            .and_then(|count| count.checked_mul(h))
            .and_then(|count| count.checked_mul(row_tiles))
            .ok_or_else(|| resource("vector MaxPool tile count overflow"))?;
        let tiles_u32 = u32::try_from(tiles)
            .map_err(|_| resource("vector MaxPool tile count exceeds U32 indexing"))?;
        let (x, y) = dispatch_grid(tiles, limits)?;
        let row_tiles_u32 = u32::try_from(row_tiles)
            .map_err(|_| resource("vector MaxPool row tile count exceeds U32 indexing"))?;
        let stride = x
            .checked_mul(64)
            .ok_or_else(|| resource("vector MaxPool grid stride overflow"))?;
        recipe.words[1] = stride;
        recipe.words[2] = OPCODE;
        recipe.words[5] = row_tiles_u32;
        recipe.words[6] = tiles_u32;
        recipe.x = x;
        recipe.y = y;
        selected += 1;
    }
    Ok(selected)
}

#[cfg(test)]
#[path = "max_pool_tests.rs"]
mod tests;
