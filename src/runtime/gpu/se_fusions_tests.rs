use super::super::tests::{bind_program, integers, model, node, tensor};
use super::*;
use crate::runtime::CompileOptions;

#[path = "se_parallel_tests.rs"]
mod parallel;

fn fixture(scalar_bias: bool, escape: bool) -> Arc<Model> {
    let mut nodes = vec![
        tensor(
            "weights",
            &[5, 5, 1, 1],
            &(0..25)
                .map(|i| (i as f32 - 12.0) / 64.0)
                .collect::<Vec<_>>(),
        ),
        tensor(
            "bias",
            if scalar_bias { &[1, 1, 1] } else { &[5, 1, 1] },
            &vec![0.03125; if scalar_bias { 1 } else { 5 }],
        ),
        node("ReduceMean", &["x"], "pooled", &[integers("axes", &[2, 3])]),
        node(
            "Conv",
            &["pooled", "weights"],
            "projection",
            &[integers("kernel_shape", &[1, 1])],
        ),
        node("Add", &["projection", "bias"], "affine", &[]),
        node("HardSigmoid", &["affine"], "gate", &[]),
        node("Mul", &["x", "gate"], "applied", &[]),
    ];
    if escape {
        nodes.extend([
            tensor("zero", &[1], &[0.0]),
            node("Mul", &["gate", "zero"], "escape", &[]),
            node("Add", &["applied", "escape"], "y", &[]),
        ]);
    } else {
        nodes.push(node("Identity", &["applied"], "y", &[]));
    }
    model(nodes, &[2, 5, 3, 17], &[2, 5, 3, 17])
}

fn assert_original_gate_recipe(recipes: &[Recipe]) {
    let gate = recipes
        .iter()
        .find(|recipe| recipe.nodes == [1, 2, 3])
        .expect("materialized control must retain Conv+bias+HardSigmoid physical fusion");
    assert!(matches!(
        gate.kind,
        RecipeKind::Compute(ShaderKind::Convolution)
    ));
    assert_eq!(
        gate.words[64], 2,
        "bias and activation are F32 Conv epilogues"
    );
    assert!(
        recipes.iter().any(|recipe| recipe.nodes == [4]
            && matches!(recipe.kind, RecipeKind::Compute(ShaderKind::General))
            && recipe.words[2] == 1),
        "materialized control must keep the separate feature Mul"
    );
}

#[test]
fn excitation_bias_add_fusion_preserves_gate_rounding_and_rejects_escaping_gate() {
    for scalar in [false, true] {
        let program = Program::compile(fixture(scalar, false)).unwrap();
        let plan = bind_program(&program, &[2, 5, 3, 17], ExecutionLimits::default()).unwrap();
        let recipe = replacement(
            &program,
            &plan,
            1,
            program.output,
            &crate::runtime::gpu::GpuLimits::default(),
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            recipe.kind,
            RecipeKind::Compute(ShaderKind::SqueezeExcite)
        ));
        assert_eq!(recipe.nodes, [1, 2, 3, 4]);
        assert_eq!(recipe.inputs.len(), 4);
        assert_eq!(recipe.inputs[3], program.input);
        assert_eq!(recipe.output, program.output);
        assert_eq!(recipe.words[0], 10);
        assert_eq!(recipe.words[3..6], [5, 5, 51]);
        assert_eq!(recipe.words[7..9], [1, 1]);
        assert_eq!(recipe.words[11..13], [u32::from(scalar), 1]);
        assert!(
            replacement(
                &program,
                &plan,
                1,
                program.instructions[3].output,
                &crate::runtime::gpu::GpuLimits::default()
            )
            .unwrap()
            .is_none()
        );

        let program = Program::compile(fixture(scalar, true)).unwrap();
        let mut plan = bind_program(&program, &[2, 5, 3, 17], ExecutionLimits::default()).unwrap();
        assert!(
            replacement(
                &program,
                &plan,
                1,
                program.output,
                &crate::runtime::gpu::GpuLimits::default()
            )
            .unwrap()
            .is_none()
        );
        let mut descriptors = plan
            .shapes
            .iter()
            .map(|&shape| GpuTensorDescriptor::canonical(shape, GpuPrecision::MixedF16))
            .collect();
        let recipes = physical::lower(
            &program,
            &mut plan,
            program.output,
            &crate::runtime::gpu::GpuLimits::default(),
            GpuPrecision::MixedF16,
            &mut descriptors,
            GpuMatrixPrecision::F16,
        )
        .unwrap();
        assert_original_gate_recipe(&recipes);
    }
}

#[test]
#[ignore = "requires physical Vulkan GPU; separate-bias excitation fusion with half gate rounding"]
fn gpu_hardware_excitation_bias_add_fusion_matches_materialized_half_gate() {
    let context = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
    let shape = [2, 5, 3, 17];
    let input: Vec<_> = (0..shape.iter().product())
        .map(|i| ((i * 17 % 97) as f32 - 48.0) / 32.0)
        .collect();
    for scalar in [false, true] {
        let session = |escape| {
            let compiled = Arc::new(
                GpuCompiledModel::compile_with_options(
                    fixture(scalar, escape),
                    context.clone(),
                    CompileOptions {
                        gpu: GpuCompileOptions {
                            precision: GpuPrecision::MixedF16,
                            matrix_precision: GpuMatrixPrecision::F16,
                        },
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
            GpuSession::new(compiled, &shape, ExecutionLimits::default()).unwrap()
        };
        let mut fused = session(false);
        let mut materialized = session(true);
        assert!(fused.recipes.iter().any(|r| matches!(
            r.kind,
            RecipeKind::Compute(ShaderKind::SqueezeExcite)
        ) && r.words[12] == 1));
        assert!(
            !materialized
                .recipes
                .iter()
                .any(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::SqueezeExcite)))
        );
        assert_original_gate_recipe(&materialized.recipes);
        let actual = fused.run(&input).unwrap().data().to_vec();
        let expected = materialized.run(&input).unwrap().data().to_vec();
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-4 + 1e-4 * expected.abs(),
                "scalar{scalar} element{index}: {actual} != {expected}"
            );
        }
    }
}
