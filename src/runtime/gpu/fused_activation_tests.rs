use super::super::tests::{bind_program, model, node, tensor};
use super::*;
use crate::runtime::CompileOptions;

fn float_attribute(name: &str, value: f32) -> Vec<u8> {
    let mut attribute = vec![10, name.len() as u8];
    attribute.extend(name.as_bytes());
    attribute.push(21);
    attribute.extend(value.to_le_bytes());
    attribute.extend([160, 1, 1]);
    let mut field = vec![42, attribute.len() as u8];
    field.extend(attribute);
    field
}

fn fixture(mode: u32, escape: bool) -> Arc<Model> {
    let mut nodes = match mode {
        1 => vec![
            tensor("sqrt_two", &[1], &[std::f32::consts::SQRT_2]),
            tensor("one", &[1], &[1.0]),
            tensor("half", &[1], &[0.5]),
            node("Div", &["x", "sqrt_two"], "scaled", &[]),
            node("Erf", &["scaled"], "error", &[]),
            node("Add", &["error", "one"], "shifted", &[]),
            node("Mul", &["shifted", "x"], "product", &[]),
            node("Mul", &["product", "half"], "activated", &[]),
        ],
        2 => vec![
            node("HardSigmoid", &["x"], "gate", &[]),
            node("Mul", &["gate", "x"], "activated", &[]),
        ],
        3 => vec![
            node("Sigmoid", &["x"], "gate", &[]),
            node("Mul", &["x", "gate"], "activated", &[]),
        ],
        _ => unreachable!(),
    };
    if mode == 2 {
        // Use the existing exact HardSwish pattern's ONNX coefficient.
        nodes[0] = node(
            "HardSigmoid",
            &["x"],
            "gate",
            &[
                float_attribute("alpha", 0.16666667),
                float_attribute("beta", 0.5),
            ],
        );
    }
    if escape {
        nodes.extend([
            tensor("zero", &[1], &[0.0]),
            node(
                "Mul",
                &[if mode == 1 { "error" } else { "gate" }, "zero"],
                "escape",
                &[],
            ),
            node("Add", &["activated", "escape"], "y", &[]),
        ]);
    } else {
        nodes.push(node("Identity", &["activated"], "y", &[]));
    }
    model(nodes, &[1, 1, 1, 67], &[1, 1, 1, 67])
}

#[test]
fn standalone_activation_matches_private_chains_and_four_element_tails() {
    for mode in [1, 2, 3] {
        let program = Program::compile(fixture(mode, false)).unwrap();
        let plan = bind_program(&program, &[1, 1, 1, 67], ExecutionLimits::default()).unwrap();
        let recipe = replacement(&program, &plan, 0, program.output, &wgpu::Limits::default())
            .unwrap()
            .unwrap();
        assert_eq!(recipe.words[0..4], [67, 64, 25, mode]);
        assert_eq!((recipe.x, recipe.y), (1, 1));
        assert_eq!(recipe.nodes.len(), if mode == 1 { 5 } else { 2 });
        assert_eq!(recipe.inputs[0], program.input);
        assert_eq!(recipe.inputs.len(), if mode == 1 { 2 } else { 1 });
        assert_eq!(recipe.output, program.output);
        assert!(
            replacement(
                &program,
                &plan,
                0,
                program.instructions[0].output,
                &wgpu::Limits::default()
            )
            .unwrap()
            .is_none()
        );

        let program = Program::compile(fixture(mode, true)).unwrap();
        let plan = bind_program(&program, &[1, 1, 1, 67], ExecutionLimits::default()).unwrap();
        assert!(
            replacement(&program, &plan, 0, program.output, &wgpu::Limits::default())
                .unwrap()
                .is_none()
        );
    }
}

#[test]
#[ignore = "requires physical Metal GPU; compares fused and materialized activation boundaries"]
fn gpu_hardware_standalone_activation_retains_half_intermediate_rounding() {
    let context = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
    let shape = [1, 1, 1, 67];
    let input: Vec<_> = (0..67)
        .map(|i| ((i * 19 % 113) as f32 - 56.0) / 16.0)
        .collect();
    for precision in [GpuPrecision::F32, GpuPrecision::MixedF16] {
        for mode in [1, 2, 3] {
            let session = |escape| {
                let compiled = Arc::new(
                    GpuCompiledModel::compile_with_options(
                        fixture(mode, escape),
                        context.clone(),
                        CompileOptions {
                            gpu: GpuCompileOptions {
                                precision,
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
            assert!(fused.recipes.iter().any(|r| r.words[2] == 25));
            assert!(!materialized.recipes.iter().any(|r| r.words[2] == 25));
            let actual = fused.run(&input).unwrap().data().to_vec();
            let expected = materialized.run(&input).unwrap().data().to_vec();
            if precision == GpuPrecision::MixedF16 && mode == 1 {
                // erf(-1.6357421875) rounds to -0.9794921875 in Half. Losing
                // that store changes the cancellation in 1+erf and yields
                // -0.02394104 instead of this materialized result.
                assert_eq!(input[1], -2.3125);
                assert_eq!(expected[1], -0.023_712_158);
            }
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 1e-4 + 1e-4 * expected.abs(),
                    "{precision:?} mode{mode} element{index}: {actual} != {expected}"
                );
            }
        }
    }
}
