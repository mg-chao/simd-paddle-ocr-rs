//! Native canonical half storage with FP32 accumulation and fused arithmetic.
//!
//! This explicit F16 matrix mode does not change the default F32 policies.
//! The caller must qualify each selected native pipeline before committing its
//! recipe; compiler rejection preserves that recipe's original generic native kernel.
use super::*;
use operators::{Recipe, RecipeKind, ShaderKind};

pub(super) fn supports_kind(kind: ShaderKind) -> bool {
    matches!(
        kind,
        ShaderKind::PointwiseNative64
            | ShaderKind::DepthwiseNative3
            | ShaderKind::DepthwiseNative5
            | ShaderKind::DepthwiseNative7
            | ShaderKind::DepthwiseNative9
            | ShaderKind::DepthwiseNative1x7
    )
}

fn half_storage(source: String) -> String {
    source
        .replace("device const float*", "device const half*")
        .replace("device float*", "device half*")
        .replace("tensor<device float,", "tensor<device half,")
}

pub(super) fn pointwise_source() -> String {
    // Scalar fallback products need explicit promotion after the storage
    // conversion. The collective MPP destination remains float in the source.
    half_storage(pointwise_mpp64::source()).replace(
        "sum = sum + raw_b[row * inner + k]\n                    * raw_a[(batch * inner + k) * columns + column];",
        "sum = sum + float(raw_b[row * inner + k])\n                    * float(raw_a[(batch * inner + k) * columns + column]);",
    )
}

/// Shared channel/index proof, independent of the qualified reduction sizes.
/// All-half Native64 is emitted only by the explicit F16 matrix policy.
fn pointwise_epilogue_coordinates(
    kind: ShaderKind,
    storage: [GpuPrecision; 6],
    words: &[u32; 128],
) -> bool {
    kind == ShaderKind::PointwiseNative64
        && storage == [GpuPrecision::MixedF16; 6]
        && words[65] == words[4]
        && words[7]
            .checked_mul(words[8])
            .is_some_and(|plane| plane != 0 && words[66] == plane)
}

/// Keep the general Known packet specialization at its original K interval.
pub(super) fn known_pointwise_epilogue(
    kind: ShaderKind,
    storage: [GpuPrecision; 6],
    words: &[u32; 128],
) -> bool {
    pointwise_epilogue_coordinates(kind, storage, words)
        && (96..=768).contains(&words[3])
        && (1..=3).contains(&words[64])
}

pub(super) fn pointwise_epilogue_source() -> String {
    pointwise_source()
        .replace(
            "kernel void pointwise_mpp64(",
            &format!(
                "{}\nkernel void pointwise_mpp64(",
                include_str!("pointwise_epilogue.metal")
            ),
        )
        .replace(
            "epilogue(sum, index, raw_d, raw_e, params)",
            "epilogue_known(sum, index, row, raw_d, raw_e, params)",
        )
        .replace(
            "epilogue(accum[i], index, raw_d, raw_e, params)",
            "epilogue_known(accum[i], index, row, raw_d, raw_e, params)",
        )
}

pub(super) fn fixed_pointwise_epilogue(
    kind: ShaderKind,
    storage: [GpuPrecision; 6],
    words: &[u32; 128],
) -> bool {
    known_pointwise_epilogue(kind, storage, words)
        && words[64] == 3
        && words[68] == 1
        && words[69] == 1
        && words[71] == 2
        && words[72] == 2
        && words[74] == 7
}

pub(super) fn fixed_pointwise_bias_gelu(
    kind: ShaderKind,
    storage: [GpuPrecision; 6],
    words: &[u32; 128],
) -> bool {
    pointwise_epilogue_coordinates(kind, storage, words)
        && (words[3] == 48 || (96..=768).contains(&words[3]))
        && words[64] == 2
        && words[68] == 1
        && words[69] == 1
        && words[71] == 7
}

pub(super) fn fixed_pointwise_adds(
    kind: ShaderKind,
    storage: [GpuPrecision; 6],
    words: &[u32; 128],
) -> bool {
    known_pointwise_epilogue(kind, storage, words)
        && words[64] == 2
        && words[68] == 1
        && words[69] == 1
        && words[71] == 2
        && words[72] == 2
}

pub(super) fn pointwise_adds_source() -> String {
    pointwise_source()
        .replace(
            "kernel void pointwise_mpp64(",
            &format!(
                "{}\nkernel void pointwise_mpp64(",
                include_str!("pointwise_adds.metal")
            ),
        )
        .replace(
            "epilogue(sum, index, raw_d, raw_e, params)",
            "epilogue_fixed_adds(sum, index, row, raw_d, raw_e, params)",
        )
        .replace(
            "epilogue(accum[i], index, raw_d, raw_e, params)",
            "epilogue_fixed_adds(accum[i], index, row, raw_d, raw_e, params)",
        )
}

pub(super) fn pointwise_bias_gelu_source() -> String {
    pointwise_source()
        .replace(
            "kernel void pointwise_mpp64(",
            &format!(
                "{}\nkernel void pointwise_mpp64(",
                include_str!("pointwise_bias_gelu.metal")
            ),
        )
        .replace(
            "epilogue(sum, index, raw_d, raw_e, params)",
            "epilogue_fixed_bg(sum, index, row, raw_d, raw_e, params)",
        )
        .replace(
            "epilogue(accum[i], index, raw_d, raw_e, params)",
            "epilogue_fixed_bg(accum[i], index, row, raw_d, raw_e, params)",
        )
}

pub(super) fn pointwise_fixed_epilogue_source() -> String {
    pointwise_source()
        .replace(
            "kernel void pointwise_mpp64(",
            &format!(
                "{}\nkernel void pointwise_mpp64(",
                include_str!("pointwise_fixed_epilogue.metal")
            ),
        )
        .replace(
            "epilogue(sum, index, raw_d, raw_e, params)",
            "epilogue_fixed_brg(sum, index, row, raw_d, raw_e, params)",
        )
        .replace(
            "epilogue(accum[i], index, raw_d, raw_e, params)",
            "epilogue_fixed_brg(accum[i], index, row, raw_d, raw_e, params)",
        )
}

pub(super) fn depthwise_source(kernel: u32) -> String {
    half_storage(depthwise_native::source(kernel)).replace("packed_float4", "packed_half4")
}

/// Native Half DW3 already knows the output channel. Only the complete
/// bias-only signature can bypass the shared dynamic epilogue.
pub(super) fn fixed_depthwise_bias(
    kind: ShaderKind,
    storage: [GpuPrecision; 6],
    words: &[u32; 128],
) -> bool {
    kind == ShaderKind::DepthwiseNative3
        && storage == [GpuPrecision::MixedF16; 6]
        && depthwise_geometry(words)
        && words[11] == 3
        && words[19] == 0
        && words[64] == 1
        && words[68] == 1
        && words[69] == 1
        && words[65] == words[4]
        && words[7]
            .checked_mul(words[8])
            .is_some_and(|plane| plane != 0 && words[66] == plane)
}

pub(super) fn depthwise_bias_source() -> String {
    depthwise_source(3)
        .replace(
            "    const uint index = row * ow + ox;",
            "    const uint index = row * ow + ox;\n    const float channel_bias = float(raw_d[channel]);",
        )
        .replace(
            "epilogue(sum[pixel], index + pixel, raw_d, raw_e, params)",
            "sum[pixel] + channel_bias",
        )
}

fn depthwise_1x7_source() -> String {
    half_storage(format!(
        "{}\n{}",
        include_str!("common_native.metal"),
        include_str!("depthwise_1x7.metal")
    ))
}

pub(super) fn source(kind: ShaderKind) -> RuntimeResult<(String, &'static str, [u32; 3])> {
    match kind {
        ShaderKind::PointwiseNative64 => Ok((pointwise_source(), "pointwise_mpp64", [128, 1, 1])),
        ShaderKind::DepthwiseNative3 => Ok((depthwise_source(3), "depthwise_native", [64, 1, 1])),
        ShaderKind::DepthwiseNative5 => Ok((depthwise_source(5), "depthwise_native", [64, 1, 1])),
        ShaderKind::DepthwiseNative7 => Ok((depthwise_source(7), "depthwise_native", [64, 1, 1])),
        ShaderKind::DepthwiseNative9 => Ok((depthwise_source(9), "depthwise_native", [64, 1, 1])),
        ShaderKind::DepthwiseNative1x7 => {
            Ok((depthwise_1x7_source(), "depthwise_1x7_half", [64, 1, 1]))
        }
        _ => Err(RuntimeError::with_kind(
            ErrorKind::Unsupported,
            "native half storage is unavailable for this shader family",
        )),
    }
}

#[cfg(test)]
#[path = "pointwise_epilogue_tests.rs"]
mod epilogue_tests;

#[cfg(test)]
#[path = "depthwise_bias_tests.rs"]
mod depthwise_bias_tests;

/// Propose native families only for explicit F16 matrix mode and canonical
/// half bindings. The caller supplies Metal/capability and pipeline admission.
pub(super) fn select(
    recipes: &mut [Recipe],
    descriptors: &[GpuTensorDescriptor],
    limits: &wgpu::Limits,
    matrix_precision: GpuMatrixPrecision,
) -> RuntimeResult<usize> {
    if matrix_precision != GpuMatrixPrecision::F16 {
        return Ok(0);
    }
    let mut selected = 0;
    for recipe in recipes {
        if !recipe
            .inputs
            .iter()
            .chain(std::iter::once(&recipe.output))
            .all(|id| {
                descriptors.get(id.0).is_some_and(|descriptor| {
                    descriptor.precision == GpuPrecision::MixedF16
                        && descriptor.layout == GpuLayout::Canonical
                })
            })
            || !shapes_match(recipe, descriptors)
        {
            continue;
        }
        if pointwise_kind(&recipe.kind)
            && pointwise_geometry(&recipe.words)
            && limits.max_compute_invocations_per_workgroup >= 128
            && limits.max_compute_workgroup_size_x >= 128
        {
            let rows = recipe.words[4].div_ceil(64);
            let columns = (recipe.words[7] * recipe.words[8]).div_ceil(64);
            let groups = (rows as usize)
                .checked_mul(columns as usize)
                .and_then(|n| n.checked_mul(recipe.words[2] as usize))
                .and_then(|n| n.checked_mul(64))
                .ok_or_else(|| resource("native half pointwise dispatch overflow"))?;
            let (x, y) = dispatch_grid(groups, limits)?;
            recipe.kind = RecipeKind::Compute(ShaderKind::PointwiseNative64);
            recipe.words[32] = rows;
            recipe.words[33] = columns;
            recipe.words[34] = x;
            recipe.words[35] = recipe.words[2];
            recipe.x = x;
            recipe.y = y;
            selected += 1;
        } else if matches!(
            recipe.kind,
            RecipeKind::Compute(ShaderKind::Convolution | ShaderKind::DepthwiseNative1x7)
        ) && depthwise_1x7_geometry(&recipe.words)
            && recipe.inputs.get(1).is_some_and(|id| {
                descriptors[id.0].shape.dimensions() == [recipe.words[3] as usize, 1, 1, 7]
            })
            && limits.max_compute_invocations_per_workgroup >= 64
            && limits.max_compute_workgroup_size_x >= 64
        {
            let tiles = (recipe.words[2] as usize)
                .checked_mul(recipe.words[4] as usize)
                .and_then(|rows| rows.checked_mul(recipe.words[8].div_ceil(4) as usize))
                .ok_or_else(|| resource("native half 1x7 depthwise dispatch overflow"))?;
            let (x, y) = dispatch_grid(tiles, limits)?;
            let stride = x
                .checked_mul(64)
                .ok_or_else(|| resource("native half 1x7 depthwise grid stride overflow"))?;
            recipe.kind = RecipeKind::Compute(ShaderKind::DepthwiseNative1x7);
            recipe.words[1] = stride;
            recipe.x = x;
            recipe.y = y;
            selected += 1;
        } else if matches!(
            recipe.kind,
            RecipeKind::Compute(
                ShaderKind::Convolution
                    | ShaderKind::Depthwise3
                    | ShaderKind::Depthwise5
                    | ShaderKind::DepthwiseNative3
                    | ShaderKind::DepthwiseNative5
                    | ShaderKind::DepthwiseNative7
                    | ShaderKind::DepthwiseNative9
            )
        ) && depthwise_geometry(&recipe.words)
            && limits.max_compute_invocations_per_workgroup >= 64
            && limits.max_compute_workgroup_size_x >= 64
        {
            let rows = recipe.words[0] / recipe.words[8];
            let tiles = (rows as usize)
                .checked_mul(recipe.words[8].div_ceil(4) as usize)
                .ok_or_else(|| resource("native half depthwise dispatch overflow"))?;
            let (x, y) = dispatch_grid(tiles, limits)?;
            recipe.words[1] = x
                .checked_mul(64)
                .ok_or_else(|| resource("native half depthwise dispatch stride overflow"))?;
            recipe.kind = RecipeKind::Compute(match recipe.words[11] {
                3 => ShaderKind::DepthwiseNative3,
                5 => ShaderKind::DepthwiseNative5,
                7 => ShaderKind::DepthwiseNative7,
                9 => ShaderKind::DepthwiseNative9,
                _ => unreachable!(),
            });
            recipe.x = x;
            recipe.y = y;
            selected += 1;
        }
    }
    Ok(selected)
}

fn pointwise_kind(kind: &RecipeKind) -> bool {
    matches!(
        kind,
        RecipeKind::Compute(
            ShaderKind::Convolution
                | ShaderKind::Conv16
                | ShaderKind::Conv32
                | ShaderKind::PointwiseNative64
        )
    )
}

fn shapes_match(recipe: &Recipe, descriptors: &[GpuTensorDescriptor]) -> bool {
    let Some(&source) = recipe.inputs.first() else {
        return false;
    };
    let words = &recipe.words;
    [
        (source, [words[2], words[3], words[5], words[6]]),
        (recipe.output, [words[2], words[4], words[7], words[8]]),
    ]
    .into_iter()
    .all(|(id, dimensions)| {
        descriptors.get(id.0).is_some_and(|descriptor| {
            descriptor.shape.dimensions() == dimensions.map(|n| n as usize)
        })
    })
}

fn pointwise_geometry(words: &[u32; 128]) -> bool {
    let Some(columns) = words[7].checked_mul(words[8]) else {
        return false;
    };
    words[2] != 0
        && words[3] >= 16
        && words[4] >= 16
        && columns >= 16
        && words[9] == words[3]
        && words[10] == words[4]
        && words[11..=16] == [1; 6]
        && words[17] == 0
        && words[18] == 0
        && words[5] == words[7]
        && words[6] == words[8]
        && words[52] == 0
        && [
            [words[2], words[3], columns],
            [words[2], words[4], columns],
            [1, words[3], words[4]],
        ]
        .into_iter()
        .all(|factors| {
            factors
                .into_iter()
                .try_fold(1u32, u32::checked_mul)
                .is_some_and(|n| n > 0 && n <= i32::MAX as u32)
        })
        && words[2]
            .checked_mul(words[4])
            .and_then(|n| n.checked_mul(columns))
            == Some(words[0])
}

fn depthwise_geometry(words: &[u32; 128]) -> bool {
    let [batch, cin, cout, height, width, oh, ow] = words[2..9] else {
        unreachable!()
    };
    batch != 0
        && cin != 0
        && cin == cout
        && height != 0
        && width != 0
        && oh != 0
        && ow != 0
        && words[9] == 1
        && words[10] == 1
        && words[11] == words[12]
        && matches!(words[11], 3 | 5 | 7 | 9)
        && matches!(words[13], 1 | 2)
        && matches!(words[14], 1 | 2)
        && words[15] == 1
        && words[16] == 1
        && words[52] == 0
        && [height, width, words[17], words[18]]
            .into_iter()
            .all(|n| n <= i32::MAX as u32)
        && oh
            .checked_mul(words[13])
            .and_then(|n| n.checked_add(words[11]))
            .is_some_and(|n| n <= i32::MAX as u32)
        && ow
            .checked_mul(words[14])
            .and_then(|n| n.checked_add(words[12]))
            .is_some_and(|n| n <= i32::MAX as u32)
        // K7's new admission must also bound coordinate arithmetic for all
        // four lanes of the last tile, including masked output-tail lanes.
        && (words[11] != 7
            || ow
                .checked_add(3)
                .and_then(|n| n.checked_mul(words[14]))
                .and_then(|n| n.checked_add(7))
                .is_some_and(|n| n <= i32::MAX as u32))
        && batch
            .checked_mul(cin)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .is_some()
        && batch
            .checked_mul(cout)
            .and_then(|n| n.checked_mul(oh))
            .and_then(|n| n.checked_mul(ow))
            == Some(words[0])
}

#[cfg(test)]
#[path = "depthwise7_tests.rs"]
mod depthwise7_tests;

fn depthwise_1x7_geometry(words: &[u32; 128]) -> bool {
    words[2] != 0
        && words[3] != 0
        && words[3] == words[4]
        && words[5] == 1
        && words[7] == 1
        && words[6] > 0
        && words[6] <= i32::MAX as u32
        && words[8] > 0
        && words[9] == 1
        && words[10] == 1
        && words[11..=16] == [1, 7, 1, 1, 1, 1]
        && words[17] == 0
        && words[18] <= i32::MAX as u32
        && words[52] == 0
        && words[8]
            .checked_add(10)
            .is_some_and(|n| n <= i32::MAX as u32)
        && words[2]
            .checked_mul(words[3])
            .and_then(|n| n.checked_mul(words[6]))
            .is_some()
        && words[3].checked_mul(7).is_some()
        && words[2]
            .checked_mul(words[4])
            .and_then(|n| n.checked_mul(words[8]))
            == Some(words[0])
        && words[60].checked_add(words[0]).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(kernel: u32) -> (Recipe, Vec<GpuTensorDescriptor>) {
        let (batch, cin, cout, height, width) = if kernel == 1 {
            (2, 17, 73, 3, 23)
        } else {
            (2, 3, 3, 3, 17)
        };
        let mut words = [0; 128];
        words[..20].copy_from_slice(&[
            batch * cout * height * width,
            64,
            batch,
            cin,
            cout,
            height,
            width,
            height,
            width,
            if kernel == 1 { cin } else { 1 },
            if kernel == 1 { cout } else { 1 },
            kernel,
            kernel,
            1,
            1,
            1,
            1,
            kernel / 2,
            kernel / 2,
            1,
        ]);
        let descriptors = [
            [batch, cin, height, width],
            [cout, if kernel == 1 { cin } else { 1 }, kernel, kernel],
            [cout, 1, 1, 1],
            [batch, cout, height, width],
        ]
        .map(|shape| {
            GpuTensorDescriptor::canonical(
                Shape::new(&shape.map(|n| n as usize)).unwrap(),
                GpuPrecision::MixedF16,
            )
        })
        .to_vec();
        (
            Recipe {
                node: 0,
                nodes: vec![0],
                kind: RecipeKind::Compute(ShaderKind::Convolution),
                inputs: vec![TensorId(0), TensorId(1), TensorId(2)],
                output: TensorId(3),
                words,
                x: 1,
                y: 1,
            },
            descriptors,
        )
    }

    #[test]
    fn half_native_pointwise_accepts_small_reduction_and_complete_binding_contract() {
        let (recipe, descriptors) = fixture(1);
        let mut selected = [recipe.clone()];
        assert_eq!(
            select(
                &mut selected,
                &descriptors,
                &wgpu::Limits::default(),
                GpuMatrixPrecision::F16
            )
            .unwrap(),
            1
        );
        assert!(matches!(
            selected[0].kind,
            RecipeKind::Compute(ShaderKind::PointwiseNative64)
        ));
        assert_eq!(selected[0].words[32..36], [2, 2, 8, 2]);
        for mixed in 0..descriptors.len() {
            let mut changed = descriptors.clone();
            changed[mixed].precision = GpuPrecision::F32;
            assert_eq!(
                select(
                    &mut [recipe.clone()],
                    &changed,
                    &wgpu::Limits::default(),
                    GpuMatrixPrecision::F16
                )
                .unwrap(),
                0
            );
        }
        assert_eq!(
            select(
                &mut [recipe],
                &descriptors,
                &wgpu::Limits::default(),
                GpuMatrixPrecision::F32
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn half_native_depthwise_retains_offsets_and_scalar_fallback_uses_float_products() {
        for kernel in [3, 5, 9] {
            let (mut recipe, descriptors) = fixture(kernel);
            recipe.words[60] = 13;
            let mut selected = [recipe];
            assert_eq!(
                select(
                    &mut selected,
                    &descriptors,
                    &wgpu::Limits::default(),
                    GpuMatrixPrecision::F16
                )
                .unwrap(),
                1
            );
            assert_eq!(
                (selected[0].x, selected[0].y, selected[0].words[1]),
                (2, 1, 128)
            );
            assert_eq!(selected[0].words[60], 13);
            let source = depthwise_source(kernel);
            assert!(source.contains("packed_half4"));
            assert!(source.contains("float4 sum"));
            assert!(!source.contains("device const float*"));
        }
        let source = pointwise_source();
        assert!(source.contains("tensor<device half,"));
        assert!(source.contains("decltype(a), decltype(b), float>"));
        assert!(source.contains("float(raw_b[row * inner + k])"));
    }
}
