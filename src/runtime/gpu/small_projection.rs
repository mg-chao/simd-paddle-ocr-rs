//! Parallel channel projection after spatial pooling, with canonical storage.
use super::*;
use operators::{Recipe, RecipeKind, ShaderKind};

pub(super) fn select(
    program: &Program,
    recipes: &mut [Recipe],
    descriptors: &[GpuTensorDescriptor],
    limits: &crate::runtime::gpu::GpuLimits,
    matrix_precision: GpuMatrixPrecision,
) -> RuntimeResult<usize> {
    if matrix_precision != GpuMatrixPrecision::F16
        || limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_storage_size < 64 * 4
    {
        return Ok(0);
    }
    let mut selected = 0;
    for recipe in recipes {
        let p = &recipe.words;
        // Generic selection may already have tiled a batched pooled Conv.
        // Its 1x1 spatial plane leaves fifteen Conv16 columns idle; the shape
        // guards below prove the same canonical dot and epilogue contract.
        if !matches!(
            recipe.kind,
            RecipeKind::Compute(ShaderKind::Convolution | ShaderKind::Conv16)
        ) || !(2..=5).contains(&recipe.inputs.len())
            || p[2] == 0
            || p[3] < 96
            || p[4] == 0
            || p[5..9] != [1; 4]
            || p[9] != p[3]
            || p[10] != p[4]
            || p[11..17] != [1; 6]
            || p[17..19] != [0; 2]
            || p[52] != 0
            || p[64] > 20
            || (p[64] != 0 && (p[65] != p[4] || p[66] != 1))
            || p[2].checked_mul(p[4]) != Some(p[0])
            || p[2].checked_mul(p[3]).is_none()
            || p[3].checked_mul(p[4]).is_none()
            || p[60].checked_add(p[0]).is_none()
            || !recipe
                .inputs
                .iter()
                .chain(std::iter::once(&recipe.output))
                .all(|id| {
                    descriptors.get(id.0).is_some_and(|d| {
                        d.precision == GpuPrecision::MixedF16 && d.layout == GpuLayout::Canonical
                    })
                })
        {
            continue;
        }
        let shape_matches = [
            (recipe.inputs[0], [p[2], p[3], 1, 1]),
            (recipe.inputs[1], [p[4], p[3], 1, 1]),
            (recipe.output, [p[2], p[4], 1, 1]),
        ]
        .into_iter()
        .all(|(id, shape)| descriptors[id.0].shape.dimensions() == shape.map(|n| n as usize));
        if !shape_matches
            || !matches!(
                program.model.graph().tensors[recipe.inputs[1].0].constant,
                Some(TensorData::F32(_))
            )
        {
            continue;
        }
        let Some(threads) = (p[0] as usize).checked_mul(64) else {
            continue;
        };
        let Ok((x, y)) = dispatch_grid(threads, limits) else {
            continue;
        };
        recipe.kind = RecipeKind::Compute(ShaderKind::SmallProjection);
        recipe.words[1] = x;
        recipe.x = x;
        recipe.y = y;
        selected += 1;
    }
    Ok(selected)
}

#[cfg(test)]
#[path = "small_projection_tests.rs"]
mod tests;
