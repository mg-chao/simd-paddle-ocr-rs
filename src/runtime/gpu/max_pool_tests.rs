use super::super::tests::{bind_program, integer, integers, model, node};
use super::*;
use crate::runtime::CompileOptions;

#[derive(Clone, Copy)]
enum Use {
    Direct,
    Offset,
    Escape,
}

fn fixture(shape: [usize; 4], usage: Use) -> Arc<Model> {
    let mut nodes = vec![node(
        "MaxPool",
        &["x"],
        "pooled",
        &[
            integers("kernel_shape", &[2, 2]),
            integers("strides", &[1, 1]),
            integers("pads", &[0, 0, 1, 1]),
        ],
    )];
    let mut output = shape;
    match usage {
        Use::Direct => nodes.push(node("Identity", &["pooled"], "y", &[])),
        Use::Offset => {
            nodes.extend([
                node("Relu", &["x"], "left", &[]),
                node("Concat", &["left", "pooled"], "y", &[integer("axis", 0)]),
            ]);
            output[0] *= 2;
        }
        Use::Escape => nodes.extend([
            node("Relu", &["pooled"], "positive", &[]),
            node("Add", &["pooled", "positive"], "y", &[]),
        ]),
    }
    model(nodes, &shape, &output)
}

fn tensor_descriptors(plan: &ShapePlan, precision: GpuPrecision) -> Vec<GpuTensorDescriptor> {
    plan.shapes
        .iter()
        .map(|&shape| GpuTensorDescriptor::canonical(shape, precision))
        .collect()
}

fn assert_unchanged(actual: &Recipe, expected: &Recipe) {
    assert!(matches!(
        actual.kind,
        RecipeKind::Compute(ShaderKind::General)
    ));
    assert_eq!(actual.words, expected.words);
    assert_eq!((actual.x, actual.y), (expected.x, expected.y));
    assert_eq!(actual.inputs, expected.inputs);
    assert_eq!(
        (actual.node, &actual.nodes, actual.output),
        (expected.node, &expected.nodes, expected.output)
    );
}

#[test]
fn max_pool_vector_admission_checks_geometry_layout_and_explicit_fast_policy() {
    let shape = [2, 3, 3, 17];
    let program = Program::compile(fixture(shape, Use::Direct)).unwrap();
    let plan = bind_program(&program, &shape, ExecutionLimits::default()).unwrap();
    let descriptors = tensor_descriptors(&plan, GpuPrecision::MixedF16);
    let original = operators::recipes(&program, &plan, &crate::runtime::gpu::GpuLimits::default())
        .unwrap()
        .remove(0);
    {
        let matrix = GpuMatrixPrecision::F32;
        let mut recipes = [original.clone()];
        assert_eq!(
            select(
                &mut recipes,
                &plan,
                &descriptors,
                &crate::runtime::gpu::GpuLimits::default(),
                matrix
            )
            .unwrap(),
            0
        );
        assert_unchanged(&recipes[0], &original);
    }
    for invalid in 0..3 {
        let mut descriptors = descriptors.clone();
        let mut original = original.clone();
        match invalid {
            0 => descriptors[original.inputs[0].0].layout = GpuLayout::PaddedNhwc { channels: 3 },
            1 => original.words[4] -= 1,
            2 => original.nodes.push(original.node),
            _ => unreachable!(),
        }
        let mut recipes = [original.clone()];
        assert_eq!(
            select(
                &mut recipes,
                &plan,
                &descriptors,
                &crate::runtime::gpu::GpuLimits::default(),
                GpuMatrixPrecision::F16
            )
            .unwrap(),
            0
        );
        assert_unchanged(&recipes[0], &original);
    }
    let mut recipes = [original.clone()];
    recipes[0].words[60] = 19;
    assert_eq!(
        select(
            &mut recipes,
            &plan,
            &descriptors,
            &crate::runtime::gpu::GpuLimits::default(),
            GpuMatrixPrecision::F16
        )
        .unwrap(),
        1
    );
    assert_eq!(recipes[0].words[0], 306);
    assert_eq!(recipes[0].words[2], OPCODE);
    assert_eq!(recipes[0].words[3..7], [3, 17, 5, 90]);
    assert_eq!(
        (recipes[0].x, recipes[0].y, recipes[0].words[1]),
        (2, 1, 128)
    );
    assert_eq!(recipes[0].words[60], 19);
    assert_eq!(recipes[0].nodes, original.nodes);
    assert_eq!(recipes[0].inputs, original.inputs);
    assert_eq!(recipes[0].output, original.output);
}

#[test]
fn max_pool_vector_keeps_prefix_escaping_values_and_concat_offsets() {
    let shape = [2, 3, 3, 17];
    for usage in [Use::Escape, Use::Offset] {
        let program = Program::compile(fixture(shape, usage)).unwrap();
        let mut plan = bind_program(&program, &shape, ExecutionLimits::default()).unwrap();
        let pooled = program.instructions[0].output;
        let mut descriptors = tensor_descriptors(&plan, GpuPrecision::MixedF16);
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
        let recipe = recipes.iter().find(|recipe| recipe.nodes == [0]).unwrap();
        assert_eq!(recipe.words[2], OPCODE);
        match usage {
            Use::Escape => {
                assert_eq!(recipe.output, pooled);
                assert_eq!(recipe.words[60], 0);
                assert!(
                    recipes
                        .iter()
                        .filter(|r| r.inputs.contains(&pooled))
                        .count()
                        >= 2
                );
            }
            Use::Offset => assert_eq!(recipe.words[60], shape.iter().product::<usize>() as u32),
            Use::Direct => unreachable!(),
        }
        let mut prefix = ShapePlan::empty(
            program.model.graph().tensors.len(),
            program.instructions.len(),
        );
        let mut scratch = BindScratch::new(&program);
        bind_into(
            &program,
            &shape,
            ExecutionLimits::default(),
            1,
            pooled,
            &mut prefix,
            &mut scratch,
        )
        .unwrap();
        let descriptors = tensor_descriptors(&prefix, GpuPrecision::MixedF16);
        let mut recipes = operators::recipes(
            &program,
            &prefix,
            &crate::runtime::gpu::GpuLimits::default(),
        )
        .unwrap();
        assert_eq!(
            select(
                &mut recipes,
                &prefix,
                &descriptors,
                &crate::runtime::gpu::GpuLimits::default(),
                GpuMatrixPrecision::F16
            )
            .unwrap(),
            1
        );
        assert_eq!(recipes.len(), 1);
        assert_eq!(recipes[0].output, pooled);
        assert_eq!(recipes[0].nodes, [0]);
    }
}

// Public inference rejects nonfinite host tensors. Exercise the same compiled
// dispatches directly so their NaN/infinity/zero behavior can be compared too.
fn raw_run(session: &mut GpuSession, input: &[f32]) -> RuntimeResult<Vec<f32>> {
    let context = session.compiled.context.clone();
    let gate = context.acquire_gate()?;
    let precision = session.output_precision();
    let mut output = vec![0.0; session.transfers.output.len()];
    let bytes = precision.bytes(output.len());
    let readback = session.transfers.readback.clone().unwrap();
    context.upload_resident(
        &session.resident_input().buffer,
        &session.input_precision().encode(input),
    )?;
    let mut batch = resident::Batch::new(context.clone())?;
    let tensor = batch.graph(session, "MaxPool exact kernel control")?;
    batch.read_into(&tensor.buffer, &readback, bytes, &gate, |view| {
        precision.decode(view, &mut output)
    })?;
    Ok(output)
}

#[test]
#[ignore = "requires physical Vulkan GPU; exact vector/scalar MaxPool including nonfinite values and offsets"]
fn gpu_hardware_max_pool_vector_matches_scalar_bits_edges_and_offsets() {
    let context = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
    for (shape, usage) in [
        ([2, 3, 1, 1], Use::Direct),
        ([2, 3, 3, 3], Use::Direct),
        ([1, 1, 2, 4], Use::Direct),
        ([2, 3, 3, 5], Use::Escape),
        ([2, 3, 5, 17], Use::Offset),
        ([1, 5, 1, 33], Use::Direct),
    ] {
        let input: Vec<_> = (0..shape.iter().product())
            .map(|i| match i % 19 {
                0 => f32::NAN,
                1 => -0.0,
                2 => 0.0,
                3 => f32::NEG_INFINITY,
                4 => f32::INFINITY,
                _ => ((i * 17 % 97) as f32 - 48.0) / 32.0,
            })
            .collect();
        let graph = fixture(shape, usage);
        for precision in [GpuPrecision::F32, GpuPrecision::MixedF16] {
            let session = |matrix_precision| {
                let compiled = Arc::new(
                    GpuCompiledModel::compile_with_options(
                        graph.clone(),
                        context.clone(),
                        CompileOptions {
                            gpu: GpuCompileOptions {
                                precision,
                                matrix_precision,
                            },
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                );
                GpuSession::new(compiled, &shape, ExecutionLimits::default()).unwrap()
            };
            let mut vector = session(GpuMatrixPrecision::F16);
            let mut scalar = session(GpuMatrixPrecision::F32);
            let recipe = vector.recipes.iter().find(|r| r.nodes == [0]).unwrap();
            assert_eq!(recipe.words[2], OPCODE);
            assert!(
                scalar
                    .recipes
                    .iter()
                    .any(|r| r.nodes == [0] && r.words[2] == 12)
            );
            if matches!(usage, Use::Offset) {
                assert_eq!(recipe.words[60], shape.iter().product::<usize>() as u32);
            }
            let actual = raw_run(&mut vector, &input).unwrap();
            let expected = raw_run(&mut scalar, &input).unwrap();
            assert_eq!(actual.len(), expected.len());
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                if expected.is_nan() {
                    assert!(
                        actual.is_nan(),
                        "{precision:?} {shape:?} element{index}: expected NaN, got{actual}"
                    );
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{precision:?} {shape:?} element{index}: {actual} != {expected}"
                    );
                }
            }
            let finite: Vec<_> = input
                .iter()
                .map(|&value| if value.is_finite() { value } else { 0.25 })
                .collect();
            let actual = vector.run(&finite).unwrap().data().to_vec();
            let expected = scalar.run(&finite).unwrap().data().to_vec();
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
}
