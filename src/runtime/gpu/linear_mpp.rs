//! Bounded canonical Half token GEMM; other geometries retain LinearNative.
use super::*;
use operators::{Recipe, RecipeKind, ShaderKind};

pub(super) const WORKGROUP_SIZE: [u32; 3] = [128, 1, 1];

/// The caller supplies an already closed and validated constant LinearNative
/// recipe. A candidate clone changes only its shader and launch geometry.
pub(super) fn candidate(
    manual: &Recipe,
    descriptors: &[GpuTensorDescriptor],
    limits: &crate::runtime::gpu::GpuLimits,
    matrix_precision: GpuMatrixPrecision,
) -> Option<Recipe> {
    let p = &manual.words;
    if matrix_precision != GpuMatrixPrecision::F16
        || !matches!(manual.kind, RecipeKind::Compute(ShaderKind::LinearNative))
        || manual.inputs.len() != 5
        || !(32..=16384).contains(&p[3])
        || p[4] != 120
        || ![240, 360].contains(&p[5])
        || p[19] > 1
        || p[20] > 1
        || p[21] > 1
        || p[24] != 0
        || p[38] != 1
        || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE[0]
        || limits.max_compute_workgroup_size_x < WORKGROUP_SIZE[0]
        || !manual
            .inputs
            .iter()
            .chain(std::iter::once(&manual.output))
            .all(|id| {
                descriptors.get(id.0).is_some_and(|d| {
                    d.precision == GpuPrecision::MixedF16 && d.layout == GpuLayout::Canonical
                })
            })
    {
        return None;
    }
    let input = &descriptors[manual.inputs[0].0].shape;
    let weight = &descriptors[manual.inputs[1].0].shape;
    let output = &descriptors[manual.output.0].shape;
    let input_count = p[3].checked_mul(p[4])?;
    let output_count = p[3].checked_mul(p[5])?;
    p[60].checked_add(output_count)?;
    if input.rank() < 2
        || input.dimensions().last() != Some(&(p[4] as usize))
        || input.element_count() != input_count as usize
        || weight.dimensions() != [p[4] as usize, p[5] as usize]
        || output.element_count() != output_count as usize
    {
        return None;
    }
    let groups = (p[3] as usize)
        .div_ceil(32)
        .checked_mul((p[5] as usize).div_ceil(64))?;
    let (x, y) = dispatch_grid(groups.checked_mul(64)?, limits).ok()?;
    let mut candidate = manual.clone();
    candidate.kind = RecipeKind::Compute(ShaderKind::LinearMpp);
    candidate.words[1] = x;
    candidate.x = x;
    candidate.y = y;
    Some(candidate)
}

/// Both admissions precede graph publication. A scoped MPP rejection retains
/// the original manual packet/grid; device and allocation errors propagate.
pub(super) fn admitted(
    manual: Recipe,
    descriptors: &[GpuTensorDescriptor],
    limits: &crate::runtime::gpu::GpuLimits,
    matrix_precision: GpuMatrixPrecision,
    admit: &mut dyn FnMut(&Recipe) -> RuntimeResult<bool>,
) -> RuntimeResult<Option<Recipe>> {
    if let Some(candidate) = candidate(&manual, descriptors, limits, matrix_precision)
        && admit(&candidate)?
    {
        return Ok(Some(candidate));
    }
    Ok(admit(&manual)?.then_some(manual))
}

#[cfg(test)]
#[cfg(target_os = "macos")]
#[path = "linear_mpp_tests.rs"]
mod tests;
