//! Native matrix-F16 convolution lowering with request-owned layout buffers.
//! Admission precedes all temporary tensor publication. Strict F32 keeps its
//! original arithmetic on required native generic kernels.
use super::*;
use operators::{Recipe, RecipeKind, ShaderKind};
use std::collections::HashMap;

#[cfg(test)]
#[path = "conv_mpp_lower_tests.rs"]
mod tests;

#[allow(clippy::too_many_arguments)]
pub(super) fn lower(
    program: &Program,
    plan: &mut ShapePlan,
    recipes: Vec<Recipe>,
    descriptors: &mut Vec<GpuTensorDescriptor>,
    limits: &crate::runtime::gpu::GpuLimits,
    matrix_precision: GpuMatrixPrecision,
    admit: &mut dyn FnMut(&Recipe) -> RuntimeResult<bool>,
) -> RuntimeResult<Vec<Recipe>> {
    let operand_precision = GpuPrecision::MixedF16;
    if matrix_precision != GpuMatrixPrecision::F16
        || limits.max_compute_invocations_per_workgroup < 128
        || limits.max_compute_workgroup_size_x < 128
    {
        return Ok(recipes);
    }
    if descriptors.len() != plan.shapes.len() || plan.slots.len() != plan.shapes.len() {
        return Err(resource(
            "MPP convolution lowering has inconsistent tensor metadata",
        ));
    }
    let mut lowered = Vec::with_capacity(recipes.len());
    let mut packed = HashMap::<(TensorId, Option<TensorId>, GpuPrecision), TensorId>::new();
    for recipe in recipes {
        // The central owner caches immutable FP32 weight permutations. Dynamic
        // weights and already transformed activations keep their existing path.
        let constant_weights = recipe.inputs.get(1).is_some_and(|id| {
            matches!(
                program
                    .model
                    .graph()
                    .tensors
                    .get(id.0)
                    .and_then(|tensor| tensor.constant.as_ref()),
                Some(TensorData::F32(_))
            )
        });
        if !constant_weights || !canonical_shapes_match(plan, &recipe, descriptors) {
            lowered.push(recipe);
            continue;
        }
        let source_precision = descriptors[recipe.inputs[0].0].precision;
        // Keep the resident detector's RGB stem in F32. Introducing operand
        // rounding at this previously strict boundary can shift box edges and
        // the resulting recognition crop widths. Half-storage recognition
        // inputs are already rounded and may use the native RGB convolution.
        if source_precision == GpuPrecision::F32 && recipe.words[3] < 8 {
            lowered.push(recipe);
            continue;
        }
        // Admission must precede the input layout rewrite and scale folding.
        let mut candidate = recipe.clone();
        let scale = if prepare_pointwise(&mut candidate, descriptors, limits)? {
            let scale = match recipe.words[52] {
                0 => None,
                1 => recipe.inputs.get(3).copied(),
                2 => recipe.inputs.get(4).copied(),
                _ => unreachable!("prepare_pointwise validates prescale mode"),
            };
            candidate.words[52] = 0;
            scale
        } else {
            let Some(kernel) = conv_mpp::prepare_recipe(&mut candidate, descriptors, limits)?
            else {
                lowered.push(recipe);
                continue;
            };
            candidate.kind = RecipeKind::Compute(kernel.shader());
            None
        };
        // Binding specialization depends on the physical matrix operand type.
        candidate.words[37] = 1;
        candidate.words[38] = u32::from(source_precision == GpuPrecision::MixedF16);
        // Stage a bounded all-Half K2 or RGB K3s2 patch from canonical NCHW. Try
        // this immutable Cin-only specialization before compiling full-image
        // geometry. Rejection leaves the original full-view candidate intact.
        if source_precision == GpuPrecision::MixedF16 {
            let mut staged = candidate.clone();
            if conv_mpp_staged::prepare_recipe(&mut staged, descriptors, limits) && admit(&staged)?
            {
                lowered.push(staged);
                continue;
            }
        }
        // Capability checks may compile a shape specialization. Keep their
        // rejected/error paths free of converters, descriptor changes and
        // mutations to the original recipe's grid or parameter packet.
        if !admit(&candidate)? {
            lowered.push(recipe);
            continue;
        }
        let mut recipe = candidate;
        let source = recipe.inputs[0];
        let key = (source, scale, operand_precision);
        let storage = if let Some(&storage) = packed.get(&key) {
            storage
        } else {
            let batch = recipe.words[2] as usize;
            let channels = recipe.words[3] as usize;
            let height = recipe.words[5] as usize;
            let width = recipe.words[6] as usize;
            let plane = height
                .checked_mul(width)
                .ok_or_else(|| resource("MPP convolution input plane overflow"))?;
            let pixel_stride = width
                .checked_mul(channels)
                .ok_or_else(|| resource("MPP convolution NHWC row stride overflow"))?;
            let batch_stride = plane
                .checked_mul(channels)
                .ok_or_else(|| resource("MPP convolution NHWC batch stride overflow"))?;
            let count = batch
                .checked_mul(batch_stride)
                .ok_or_else(|| resource("MPP convolution input pack size overflow"))?;
            checked_pack_bytes(count, operand_precision, limits)?;
            let count_u32 = u32::try_from(count)
                .map_err(|_| resource("MPP convolution input pack exceeds U32 indexing"))?;
            let plane_u32 = u32::try_from(plane)
                .map_err(|_| resource("MPP convolution input plane exceeds U32 indexing"))?;
            let (x, y) = dispatch_grid(count, limits)?;
            let row_stride = x
                .checked_mul(64)
                .ok_or_else(|| resource("MPP convolution input pack dispatch stride overflow"))?;
            // Compute every fallible value before publishing the new tensor.
            let storage_shape = Shape::new(&[count])?;
            let logical_shape = descriptors[source.0].shape;
            let storage = TensorId(plan.shapes.len());
            plan.shapes.push(storage_shape);
            plan.slots.push(usize::MAX);
            descriptors.push(GpuTensorDescriptor {
                shape: logical_shape,
                layout: GpuLayout::PaddedNhwc { channels },
                precision: operand_precision,
                strides: vec![batch_stride, 1, pixel_stride, channels],
                storage_offset: 0,
            });
            let mut words = [0; 128];
            words[0] = count_u32;
            words[1] = row_stride;
            words[3] = recipe.words[3];
            words[4] = plane_u32;
            words[5] = recipe.words[3]; // No physical channel padding.
            words[6] = u32::from(scale.is_some());
            lowered.push(Recipe {
                node: recipe.node,
                nodes: vec![recipe.nodes.first().copied().unwrap_or(recipe.node)],
                kind: RecipeKind::Compute(ShaderKind::PackNhwc),
                inputs: vec![source, scale.unwrap_or(source)],
                output: storage,
                words,
                x,
                y,
            });
            packed.insert(key, storage);
            storage
        };
        recipe.inputs[0] = storage;
        lowered.push(recipe);
    }
    Ok(lowered)
}

/// Prepare a plain or already prescaled K1 pointwise operator for explicit
/// FP16 operands. All existing bias/residual bindings retain their indices.
fn prepare_pointwise(
    recipe: &mut Recipe,
    descriptors: &[GpuTensorDescriptor],
    limits: &crate::runtime::gpu::GpuLimits,
) -> RuntimeResult<bool> {
    if !matches!(
        recipe.kind,
        RecipeKind::Compute(
            ShaderKind::Convolution
                | ShaderKind::Conv16
                | ShaderKind::Conv32
                | ShaderKind::PointwiseNative64
        )
    ) {
        return Ok(false);
    }
    let words = &recipe.words;
    let Some(columns) = words[7].checked_mul(words[8]) else {
        return Ok(false);
    };
    if words[2] == 0
        || words[3] < 16
        || words[4] < 16
        || columns < 16
        || words[9] != words[3]
        || words[10] != words[4]
        || words[11..=16] != [1; 6]
        || words[17] != 0
        || words[18] != 0
        || words[5] != words[7]
        || words[6] != words[8]
        || !recipe
            .inputs
            .iter()
            .chain(std::iter::once(&recipe.output))
            .all(|id| {
                descriptors.get(id.0).is_some_and(|descriptor| {
                    descriptor.precision == GpuPrecision::F32
                        && descriptor.layout == GpuLayout::Canonical
                })
            })
    {
        return Ok(false);
    }
    let scale = match words[52] {
        0 => None,
        1 => recipe.inputs.get(3).copied(),
        2 => recipe.inputs.get(4).copied(),
        _ => return Ok(false),
    };
    if words[52] != 0 {
        let Some(scale) = scale else {
            return Ok(false);
        };
        let Some(scale_count) = words[2].checked_mul(words[3]) else {
            return Ok(false);
        };
        if descriptors[scale.0].shape.element_count() != scale_count as usize {
            return Ok(false);
        }
    }
    let Some(padded_rows) = words[4].checked_add(63).map(|n| n / 64 * 64) else {
        return Ok(false);
    };
    // Tensor extents, outer strides, padded weights and all batch address
    // products must remain representable by signed Metal tensor coordinates.
    for factors in [
        [words[2], words[3], words[5], words[6]],
        [words[2], words[4], words[7], words[8]],
        [1, padded_rows, words[3], 1],
    ] {
        if factors
            .into_iter()
            .try_fold(1u32, u32::checked_mul)
            .is_none_or(|count| count == 0 || count > i32::MAX as u32)
        {
            return Ok(false);
        }
    }
    let row_tiles = words[4].div_ceil(64);
    let column_tiles = columns.div_ceil(64);
    let groups = (row_tiles as usize)
        .checked_mul(column_tiles as usize)
        .and_then(|n| n.checked_mul(words[2] as usize))
        .and_then(|n| n.checked_mul(64))
        .ok_or_else(|| resource("FP16 pointwise matrix dispatch overflow"))?;
    let (x, y) = dispatch_grid(groups, limits)?;
    recipe.kind = RecipeKind::Compute(ShaderKind::PointwiseF16);
    recipe.words[32] = row_tiles;
    recipe.words[33] = column_tiles;
    recipe.words[34] = x;
    recipe.words[35] = recipe.words[2];
    recipe.words[36] = padded_rows;
    recipe.x = x;
    recipe.y = y;
    Ok(true)
}

fn checked_pack_bytes(
    count: usize,
    precision: GpuPrecision,
    limits: &crate::runtime::gpu::GpuLimits,
) -> RuntimeResult<()> {
    if count == 0 || u32::try_from(count).is_err() {
        return Err(resource(
            "matrix operand pack count must be positive and fit U32",
        ));
    }
    let bytes_per_element = if precision == GpuPrecision::F32 { 4 } else { 2 };
    let bytes = count
        .checked_mul(bytes_per_element)
        .and_then(|n| n.checked_add(3))
        .map(|n| n / 4 * 4)
        .ok_or_else(|| resource("matrix operand pack byte size overflow"))?;
    if bytes as u64 > limits.max_storage_buffer_binding_size
        || bytes as u64 > limits.max_buffer_size
    {
        return Err(resource(
            "matrix operand pack exceeds device storage limits",
        ));
    }
    Ok(())
}

fn canonical_shapes_match(
    plan: &ShapePlan,
    recipe: &Recipe,
    descriptors: &[GpuTensorDescriptor],
) -> bool {
    let Some(&source) = recipe.inputs.first() else {
        return false;
    };
    let Some(storage) = descriptors
        .get(source.0)
        .map(|descriptor| descriptor.precision)
    else {
        return false;
    };
    let words = &recipe.words;
    for (id, expected) in [
        (source, [words[2], words[3], words[5], words[6]]),
        (recipe.output, [words[2], words[4], words[7], words[8]]),
    ] {
        let Some(descriptor) = descriptors.get(id.0) else {
            return false;
        };
        if descriptor.precision != storage
            || descriptor.layout != GpuLayout::Canonical
            || descriptor.shape.dimensions() != expected.map(|dimension| dimension as usize)
            || plan.shapes.get(id.0) != Some(&descriptor.shape)
        {
            return false;
        }
    }
    words[2]
        .checked_mul(words[4])
        .and_then(|n| n.checked_mul(words[7]))
        .and_then(|n| n.checked_mul(words[8]))
        == Some(words[0])
}
