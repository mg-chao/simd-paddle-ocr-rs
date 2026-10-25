//! CPU-only structural checks. Install beneath gpu::tests, alongside its ONNX
//! fixtures, when promoting conv_mpp_lower.rs into the production source tree.
use super::super::tests::{bind_program, integers, model, node, tensor, values};
use super::*;
use operators::{RecipeKind, ShaderKind};

fn fixture() -> (
    Program,
    ShapePlan,
    Vec<GpuTensorDescriptor>,
    Vec<operators::Recipe>,
) {
    let shape = [2, 8, 9, 13];
    let graph = model(
        vec![
            tensor("w1", &[8, 8, 3, 3], &values(8 * 8 * 3 * 3, 7)),
            tensor("w2", &[8, 8, 7, 7], &values(8 * 8 * 7 * 7, 11)),
            node(
                "Conv",
                &["x", "w1"],
                "a",
                &[integers("kernel_shape", &[3; 2]), integers("pads", &[1; 4])],
            ),
            node(
                "Conv",
                &["x", "w2"],
                "b",
                &[integers("kernel_shape", &[7; 2]), integers("pads", &[3; 4])],
            ),
            node("Add", &["a", "b"], "y", &[]),
        ],
        &shape,
        &shape,
    );
    let program = Program::compile(graph).unwrap();
    let plan = bind_program(&program, &shape, ExecutionLimits::default()).unwrap();
    let descriptors = plan
        .shapes
        .iter()
        .map(|&shape| GpuTensorDescriptor::canonical(shape, GpuPrecision::F32))
        .collect();
    let recipes = operators::recipes(&program, &plan, &wgpu::Limits::default()).unwrap();
    (program, plan, descriptors, recipes)
}

#[test]
fn mpp_convolution_lowering_reuses_one_half_operand_pack_for_siblings() {
    let (program, mut plan, mut descriptors, recipes) = fixture();
    let original_shapes = plan.shapes.clone();
    let original_descriptors = descriptors.clone();
    let lowered = super::lower(
        &program,
        &mut plan,
        recipes,
        &mut descriptors,
        &wgpu::Limits::default(),
        GpuMatrixPrecision::F16,
        &mut |_| Ok(true),
    )
    .unwrap();
    assert_eq!(lowered.len(), 4);
    assert_eq!(plan.shapes.len(), original_shapes.len() + 1);
    assert_eq!(&plan.shapes[..original_shapes.len()], &original_shapes);
    let packed = TensorId(original_shapes.len());
    assert_eq!(plan.shapes[packed.0].dimensions(), &[2 * 8 * 9 * 13]);
    assert_eq!(plan.slots[packed.0], usize::MAX);
    assert!(matches!(
        lowered[0].kind,
        RecipeKind::Compute(ShaderKind::PackNhwc)
    ));
    assert_eq!(lowered[0].inputs, [program.input, program.input]);
    assert_eq!(lowered[0].output, packed);
    assert_eq!(lowered[0].words[0], 2 * 8 * 9 * 13);
    assert_eq!(lowered[0].words[1], lowered[0].x * 64);
    assert_eq!(lowered[0].words[3..7], [8, 9 * 13, 8, 0]);
    assert_eq!(lowered[0].words[60], 0);
    assert!(matches!(
        lowered[1].kind,
        RecipeKind::Compute(ShaderKind::ConvMpp3s1)
    ));
    assert!(matches!(
        lowered[2].kind,
        RecipeKind::Compute(ShaderKind::ConvMpp7s1)
    ));
    assert_eq!(lowered[1].inputs[0], packed);
    assert_eq!(lowered[2].inputs[0], packed);
    for recipe in [&lowered[1], &lowered[2]] {
        assert_eq!(recipe.words[36], 32);
        assert_eq!(recipe.words[37], 1);
        assert_eq!(descriptors[recipe.output.0].layout, GpuLayout::Canonical);
        assert_eq!(
            descriptors[recipe.output.0].shape,
            original_descriptors[recipe.output.0].shape
        );
    }
    let descriptor = &descriptors[packed.0];
    assert_eq!(descriptor.shape.dimensions(), &[2, 8, 9, 13]);
    assert_eq!(descriptor.precision, GpuPrecision::MixedF16);
    assert_eq!(descriptor.layout, GpuLayout::PaddedNhwc { channels: 8 });
    assert_eq!(descriptor.strides, [8 * 9 * 13, 1, 8 * 13, 8]);
    assert_eq!(descriptor.storage_offset, 0);
}

#[test]
fn mpp_convolution_lowering_strict_matrix_policy_preserves_f32_and_half_storage() {
    for mixed_input in [false, true] {
        let (program, mut plan, mut descriptors, recipes) = fixture();
        let original_count = plan.shapes.len();
        if mixed_input {
            descriptors[program.input.0].precision = GpuPrecision::MixedF16;
        }
        let lowered = super::lower(
            &program,
            &mut plan,
            recipes,
            &mut descriptors,
            &wgpu::Limits::default(),
            GpuMatrixPrecision::F32,
            &mut |_| Ok(true),
        )
        .unwrap();
        assert_eq!(lowered.len(), 3);
        assert_eq!(plan.shapes.len(), original_count);
        assert_eq!(descriptors.len(), original_count);
        assert!(matches!(
            lowered[0].kind,
            RecipeKind::Compute(ShaderKind::Convolution)
        ));
        assert!(matches!(
            lowered[1].kind,
            RecipeKind::Compute(ShaderKind::Convolution)
        ));
    }
}

#[test]
fn mpp_convolution_lowering_checks_pack_binding_limit_before_publishing_temp() {
    let (program, mut plan, mut descriptors, recipes) = fixture();
    let original_count = plan.shapes.len();
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: 64,
        max_buffer_size: 64,
        ..Default::default()
    };
    assert!(
        super::lower(
            &program,
            &mut plan,
            recipes,
            &mut descriptors,
            &limits,
            GpuMatrixPrecision::F16,
            &mut |_| Ok(true)
        )
        .is_err()
    );
    assert_eq!(plan.shapes.len(), original_count);
    assert_eq!(plan.slots.len(), original_count);
    assert_eq!(descriptors.len(), original_count);
}

fn assert_recipes_unchanged(actual: &[operators::Recipe], expected: &[operators::Recipe]) {
    assert_eq!(actual.len(), expected.len(), "unexpected layout converter");
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.node, expected.node);
        assert_eq!(actual.nodes, expected.nodes);
        assert_eq!(actual.inputs, expected.inputs);
        assert_eq!(actual.output, expected.output);
        assert_eq!(actual.words, expected.words);
        assert_eq!((actual.x, actual.y), (expected.x, expected.y));
        match (&actual.kind, &expected.kind) {
            (RecipeKind::Compute(actual), RecipeKind::Compute(expected)) => {
                assert_eq!(actual, expected);
            }
            (
                RecipeKind::Copy {
                    source_offset: actual_source,
                    destination_offset: actual_destination,
                    bytes: actual_bytes,
                },
                RecipeKind::Copy {
                    source_offset: expected_source,
                    destination_offset: expected_destination,
                    bytes: expected_bytes,
                },
            ) => {
                assert_eq!(actual_source, expected_source);
                assert_eq!(actual_destination, expected_destination);
                assert_eq!(actual_bytes, expected_bytes);
            }
            _ => panic!("rejected admission changed the recipe kind"),
        }
    }
}

fn assert_descriptors_unchanged(actual: &[GpuTensorDescriptor], expected: &[GpuTensorDescriptor]) {
    assert_eq!(actual.len(), expected.len(), "unexpected temporary tensor");
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.shape, expected.shape);
        assert_eq!(actual.layout, expected.layout);
        assert_eq!(actual.precision, expected.precision);
        assert_eq!(actual.strides, expected.strides);
        assert_eq!(actual.storage_offset, expected.storage_offset);
    }
}

#[test]
fn mpp_convolution_rejected_admission_preserves_original_recipe_and_tensor_metadata() {
    let (program, mut plan, mut descriptors, mut recipes) = fixture();
    // These dispatch words differ from the prepared native schedule. Rejection
    // must retain the complete original packet rather than a partially prepared
    // convolution whose original kind interprets those words differently.
    recipes[0].x = 13;
    recipes[0].y = 2;
    recipes[0].words[32..37].copy_from_slice(&[123, 456, 789, 234, 567]);
    let original_shapes = plan.shapes.clone();
    let original_slots = plan.slots.clone();
    let original_descriptors = descriptors.clone();
    let original_recipes = recipes.clone();
    let mut calls = 0;
    let lowered = super::lower(
        &program,
        &mut plan,
        recipes,
        &mut descriptors,
        &wgpu::Limits::default(),
        GpuMatrixPrecision::F16,
        &mut |candidate| {
            assert!(matches!(
                candidate.kind,
                RecipeKind::Compute(ShaderKind::ConvMpp3s1 | ShaderKind::ConvMpp7s1)
            ));
            assert_eq!(candidate.inputs[0], program.input);
            assert_eq!(candidate.words[32], 1);
            assert_eq!(candidate.words[33], 3 * 2);
            assert_eq!(candidate.words[35], 2);
            calls += 1;
            Ok(false)
        },
    )
    .unwrap();
    assert_eq!(calls, 2);
    assert_eq!(plan.shapes, original_shapes);
    assert_eq!(plan.slots, original_slots);
    assert_descriptors_unchanged(&descriptors, &original_descriptors);
    assert_recipes_unchanged(&lowered, &original_recipes);
}

#[test]
fn mpp_convolution_admission_error_propagates_before_temporary_publication() {
    let (program, mut plan, mut descriptors, recipes) = fixture();
    let original_shapes = plan.shapes.clone();
    let original_slots = plan.slots.clone();
    let original_descriptors = descriptors.clone();
    let mut calls = 0;
    let error = super::lower(
        &program,
        &mut plan,
        recipes,
        &mut descriptors,
        &wgpu::Limits::default(),
        GpuMatrixPrecision::F16,
        &mut |_| {
            calls += 1;
            if calls == 1 {
                Ok(false)
            } else {
                Err(RuntimeError::with_kind(
                    ErrorKind::DeviceLost,
                    "specialization admission failure",
                ))
            }
        },
    )
    .err()
    .expect("admission failure must propagate");
    assert_eq!(calls, 2);
    assert_eq!(error.kind(), ErrorKind::DeviceLost);
    assert!(
        error
            .to_string()
            .contains("specialization admission failure")
    );
    assert_eq!(plan.shapes, original_shapes);
    assert_eq!(plan.slots, original_slots);
    assert_descriptors_unchanged(&descriptors, &original_descriptors);
}

fn pointwise_fixture() -> (
    Program,
    ShapePlan,
    Vec<GpuTensorDescriptor>,
    Vec<operators::Recipe>,
    [TensorId; 2],
) {
    let input_shape = [2, 32, 3, 19];
    let output_shape = [2, 48, 3, 19];
    let graph = model(
        vec![
            tensor("w1", &[48, 32, 1, 1], &values(48 * 32, 7)),
            tensor("w2", &[48, 32, 1, 1], &values(48 * 32, 11)),
            tensor("scale_a", &[2, 32, 1, 1], &values(2 * 32, 13)),
            tensor("scale_b", &[2, 32, 1, 1], &values(2 * 32, 17)),
            node("Mul", &["x", "scale_a"], "scaled_a", &[]),
            node("Mul", &["x", "scale_b"], "scaled_b", &[]),
            node(
                "Conv",
                &["x", "w1"],
                "a",
                &[integers("kernel_shape", &[1; 2])],
            ),
            node(
                "Conv",
                &["scaled_a", "w1"],
                "b",
                &[integers("kernel_shape", &[1; 2])],
            ),
            node(
                "Conv",
                &["scaled_b", "w2"],
                "c",
                &[integers("kernel_shape", &[1; 2])],
            ),
            node("Add", &["a", "b"], "sum", &[]),
            node("Add", &["sum", "c"], "y", &[]),
        ],
        &input_shape,
        &output_shape,
    );
    let scales = ["scale_a", "scale_b"].map(|name| {
        TensorId(
            graph
                .graph()
                .tensors
                .iter()
                .position(|tensor| tensor.name == name)
                .unwrap(),
        )
    });
    let program = Program::compile(graph).unwrap();
    let plan = bind_program(&program, &input_shape, ExecutionLimits::default()).unwrap();
    let descriptors = plan
        .shapes
        .iter()
        .map(|&shape| GpuTensorDescriptor::canonical(shape, GpuPrecision::F32))
        .collect();
    let mut recipes: Vec<_> = operators::recipes(&program, &plan, &wgpu::Limits::default())
        .unwrap()
        .into_iter()
        .filter(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::Convolution)))
        .collect();
    assert_eq!(recipes.len(), 3);
    // Represent the physical prescale fold, which removes the two Mul kernels
    // and binds their channel scales at either available residual operand slot.
    for (index, operand) in [(1, 3), (2, 4)] {
        let recipe = &mut recipes[index];
        recipe.inputs[0] = program.input;
        while recipe.inputs.len() < operand {
            recipe.inputs.push(program.input);
        }
        recipe.inputs.push(scales[index - 1]);
        recipe.words[52] = (operand - 2) as u32;
        recipe.words[53] = 32;
        recipe.kind = RecipeKind::Compute(if operand == 3 {
            ShaderKind::Conv16
        } else {
            ShaderKind::Conv32
        });
    }
    (program, plan, descriptors, recipes, scales)
}

#[test]
fn matrix_f16_spatial_policy_packs_half_inputs_without_enabling_f32_spatial() {
    let (program, mut plan, mut descriptors, recipes) = fixture();
    let original_count = descriptors.len();
    let original_descriptors = descriptors.clone();
    let lowered = super::lower(
        &program,
        &mut plan,
        recipes,
        &mut descriptors,
        &wgpu::Limits::default(),
        GpuMatrixPrecision::F16,
        &mut |candidate| {
            assert_eq!(candidate.words[36], 32);
            assert_eq!(candidate.words[37], 1);
            Ok(true)
        },
    )
    .unwrap();
    let converters: Vec<_> = lowered
        .iter()
        .filter(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::PackNhwc)))
        .collect();
    assert_eq!(converters.len(), 1);
    assert_eq!(lowered.len(), 4);
    assert_eq!(descriptors.len(), original_count + 1);
    assert_eq!(
        descriptors[original_count].precision,
        GpuPrecision::MixedF16
    );
    assert_eq!(
        descriptors[original_count].layout,
        GpuLayout::PaddedNhwc { channels: 8 }
    );
    assert_eq!(converters[0].inputs, [program.input, program.input]);
    assert_eq!(converters[0].words[3..7], [8, 9 * 13, 8, 0]);
    assert_descriptors_unchanged(&descriptors[..original_count], &original_descriptors);
    for recipe in lowered.iter().filter(|recipe| {
        matches!(
            recipe.kind,
            RecipeKind::Compute(ShaderKind::ConvMpp3s1 | ShaderKind::ConvMpp7s1)
        )
    }) {
        assert_eq!(recipe.inputs[0], TensorId(original_count));
        assert_eq!(descriptors[recipe.output.0].precision, GpuPrecision::F32);
        assert_eq!(descriptors[recipe.output.0].layout, GpuLayout::Canonical);
    }
}

#[test]
fn matrix_f16_pointwise_folds_both_prescale_slots_and_reuses_identical_packs() {
    let (program, mut plan, mut descriptors, mut recipes, scales) = pointwise_fixture();
    let original_count = descriptors.len();
    let original_descriptors = descriptors.clone();
    // The same logical scale at binding 4 must reuse binding 3's pack. A second
    // scale and the unscaled input require distinct temporary tensors.
    let mut same_scale = recipes[1].clone();
    same_scale.inputs[3] = program.input;
    same_scale.inputs.push(scales[0]);
    same_scale.words[52] = 2;
    same_scale.kind = RecipeKind::Compute(ShaderKind::Conv32);
    recipes.push(same_scale);
    let mut admitted = 0;
    let lowered = super::lower(
        &program,
        &mut plan,
        recipes,
        &mut descriptors,
        &wgpu::Limits::default(),
        GpuMatrixPrecision::F16,
        &mut |candidate| {
            assert!(matches!(
                candidate.kind,
                RecipeKind::Compute(ShaderKind::PointwiseF16)
            ));
            assert_eq!(candidate.words[32..38], [1, 1, candidate.x, 2, 64, 1]);
            assert_eq!(candidate.words[52], 0, "prescale belongs in the input pack");
            admitted += 1;
            Ok(true)
        },
    )
    .unwrap();
    assert_eq!(admitted, 4);
    let converters: Vec<_> = lowered
        .iter()
        .filter(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::PackNhwc)))
        .collect();
    let matrix: Vec<_> = lowered
        .iter()
        .filter(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::PointwiseF16)))
        .collect();
    assert_eq!(lowered.len(), 7);
    assert_eq!(converters.len(), 3);
    assert_eq!(matrix.len(), 4);
    assert_eq!(plan.shapes.len(), original_count + 3);
    assert_eq!(plan.slots.len(), original_count + 3);
    assert_eq!(descriptors.len(), original_count + 3);
    assert_eq!(converters[0].inputs, [program.input, program.input]);
    assert_eq!(converters[1].inputs, [program.input, scales[0]]);
    assert_eq!(converters[2].inputs, [program.input, scales[1]]);
    assert_eq!(converters[0].words[6], 0);
    assert_eq!(converters[1].words[6], 1);
    assert_eq!(converters[2].words[6], 1);
    assert_eq!(matrix[0].inputs[0], converters[0].output);
    assert_eq!(matrix[1].inputs[0], converters[1].output);
    assert_eq!(matrix[2].inputs[0], converters[2].output);
    assert_eq!(matrix[3].inputs[0], converters[1].output);
    for converter in converters {
        assert_eq!(converter.words[0], 2 * 32 * 3 * 19);
        assert_eq!(converter.words[1], converter.x * 64);
        assert_eq!(converter.words[3..6], [32, 3 * 19, 32]);
        let packed = converter.output;
        assert_eq!(plan.shapes[packed.0].dimensions(), &[2 * 32 * 3 * 19]);
        assert_eq!(plan.slots[packed.0], usize::MAX);
        assert_eq!(descriptors[packed.0].shape.dimensions(), &[2, 32, 3, 19]);
        assert_eq!(descriptors[packed.0].precision, GpuPrecision::MixedF16);
        assert_eq!(
            descriptors[packed.0].layout,
            GpuLayout::PaddedNhwc { channels: 32 }
        );
        assert_eq!(descriptors[packed.0].strides, [32 * 3 * 19, 1, 32 * 19, 32]);
        assert_eq!(descriptors[packed.0].storage_offset, 0);
    }
    for recipe in matrix {
        assert_eq!(recipe.words[52], 0);
        assert_eq!(descriptors[recipe.output.0].precision, GpuPrecision::F32);
        assert_eq!(descriptors[recipe.output.0].layout, GpuLayout::Canonical);
    }
    assert_descriptors_unchanged(&descriptors[..original_count], &original_descriptors);
}

#[test]
fn matrix_pointwise_f32_and_noncanonical_source_retain_original_schedule() {
    for matrix_precision in [GpuMatrixPrecision::F32, GpuMatrixPrecision::F16] {
        let (program, mut plan, mut descriptors, recipes, _) = pointwise_fixture();
        if matches!(matrix_precision, GpuMatrixPrecision::F16) {
            descriptors[program.input.0].precision = GpuPrecision::MixedF16;
        }
        let original_shapes = plan.shapes.clone();
        let original_slots = plan.slots.clone();
        let original_descriptors = descriptors.clone();
        let original_recipes = recipes.clone();
        let mut calls = 0;
        let lowered = super::lower(
            &program,
            &mut plan,
            recipes,
            &mut descriptors,
            &wgpu::Limits::default(),
            matrix_precision,
            &mut |_| {
                calls += 1;
                Ok(true)
            },
        )
        .unwrap();
        assert_eq!(calls, 0);
        assert_eq!(plan.shapes, original_shapes);
        assert_eq!(plan.slots, original_slots);
        assert_descriptors_unchanged(&descriptors, &original_descriptors);
        assert_recipes_unchanged(&lowered, &original_recipes);
    }
}

#[test]
fn matrix_f16_rejected_pointwise_admission_preserves_prescale_and_grid_metadata() {
    let (program, mut plan, mut descriptors, mut recipes, _) = pointwise_fixture();
    for recipe in &mut recipes {
        recipe.words[32..38].copy_from_slice(&[123, 456, 789, 234, 567, 890]);
        recipe.x = 13;
        recipe.y = 2;
    }
    let original_shapes = plan.shapes.clone();
    let original_slots = plan.slots.clone();
    let original_descriptors = descriptors.clone();
    let original_recipes = recipes.clone();
    let mut calls = 0;
    let lowered = super::lower(
        &program,
        &mut plan,
        recipes,
        &mut descriptors,
        &wgpu::Limits::default(),
        GpuMatrixPrecision::F16,
        &mut |candidate| {
            assert!(matches!(
                candidate.kind,
                RecipeKind::Compute(ShaderKind::PointwiseF16)
            ));
            assert_eq!(candidate.words[36..38], [64, 1]);
            assert_eq!(candidate.words[52], 0);
            calls += 1;
            Ok(false)
        },
    )
    .unwrap();
    assert_eq!(calls, 3);
    assert_eq!(plan.shapes, original_shapes);
    assert_eq!(plan.slots, original_slots);
    assert_descriptors_unchanged(&descriptors, &original_descriptors);
    assert_recipes_unchanged(&lowered, &original_recipes);
}
