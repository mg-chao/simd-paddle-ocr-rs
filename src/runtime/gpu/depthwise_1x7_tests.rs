//! Canonical Half1x7 depthwise selection and independent physical execution.
use super::*;
use backend::test_support::MetalTestBatch;
use operators::Recipe;

#[derive(Clone, Copy)]
enum Use {
    Plain,
    ResidualSwish,
    Escape,
    Concat,
}

fn rect_graph(shape: [usize; 4], pads: [usize; 4], usage: Use) -> Arc<Model> {
    let channels = shape[1];
    let case = ConvCase {
        name: "Half1x7",
        input: shape,
        channels,
        kernel: [1, 7],
        stride: [1; 2],
        dilation: [1; 2],
        pads,
        group: channels,
        bias: true,
    };
    let output = case.output();
    let mut parts = vec![tensor("bias", &[channels], &dyadic(channels, 11))];
    for branch in 0..if matches!(usage, Use::Concat) { 2 } else { 1 } {
        let weight = format!("w{branch}");
        let conv = format!("conv{branch}");
        let sum = format!("sum{branch}");
        let gate = format!("gate{branch}");
        let result = if matches!(usage, Use::Concat) {
            format!("branch{branch}")
        } else {
            "y".into()
        };
        parts.push(tensor(
            &weight,
            &[channels, 1, 1, 7],
            &dyadic(channels * 7, 7 + branch),
        ));
        parts.push(node(
            "Conv",
            &["x", &weight, "bias"],
            if matches!(usage, Use::Plain) {
                &result
            } else {
                &conv
            },
            &[
                integers("kernel_shape", &[1, 7]),
                integers("strides", &[1; 2]),
                integers("dilations", &[1; 2]),
                integers("pads", &pads),
                integer("group", channels),
            ],
        ));
        match usage {
            Use::Plain => {}
            Use::ResidualSwish | Use::Concat => {
                let residual = format!("residual{branch}");
                parts.push(tensor(
                    &residual,
                    &output,
                    &dyadic(output.iter().product(), 13 + branch),
                ));
                parts.push(node("Add", &[&conv, &residual], &sum, &[]));
                parts.push(node("Sigmoid", &[&sum], &gate, &[]));
                parts.push(node("Mul", &[&sum, &gate], &result, &[]));
            }
            Use::Escape => {
                parts.push(node("Relu", &[&conv], "positive", &[]));
                parts.push(node("Add", &[&conv, "positive"], &result, &[]));
            }
        }
    }
    let mut final_shape = output;
    if matches!(usage, Use::Concat) {
        parts.push(node(
            "Concat",
            &["branch0", "branch1"],
            "y",
            &[integer("axis", 1)],
        ));
        final_shape[1] *= 2;
    }
    model(parts, &shape, &final_shape)
}

fn rect_fixture() -> (Recipe, Vec<GpuTensorDescriptor>) {
    let shape = [2, 7, 1, 17];
    let program = Program::compile(rect_graph(shape, [0, 3, 0, 3], Use::Plain)).unwrap();
    let plan = bind_program(&program, &shape, ExecutionLimits::default()).unwrap();
    let descriptors = plan
        .shapes
        .iter()
        .map(|&shape| GpuTensorDescriptor::canonical(shape, GpuPrecision::MixedF16))
        .collect();
    let recipe = operators::recipes(&program, &plan, &crate::runtime::gpu::GpuLimits::default())
        .unwrap()
        .remove(0);
    (recipe, descriptors)
}

fn assert_unchanged(actual: &Recipe, original: &Recipe) {
    assert_eq!(actual.words, original.words);
    assert_eq!(actual.inputs, original.inputs);
    assert_eq!(
        (
            actual.node,
            &actual.nodes,
            actual.output,
            actual.x,
            actual.y
        ),
        (
            original.node,
            &original.nodes,
            original.output,
            original.x,
            original.y
        )
    );
    assert!(
        matches!((&actual.kind, &original.kind), (RecipeKind::Compute(a), RecipeKind::Compute(b)) if a == b)
    );
}

#[test]
fn half_native_depthwise_1x7_admission_is_explicit_half_and_preserves_metadata() {
    let (mut recipe, descriptors) = rect_fixture();
    recipe.words[60] = 13;
    let mut selected = [recipe.clone()];
    assert_eq!(
        half_native::select(
            &mut selected,
            &descriptors,
            &crate::runtime::gpu::GpuLimits::default(),
            GpuMatrixPrecision::F16
        )
        .unwrap(),
        1
    );
    assert!(matches!(
        selected[0].kind,
        RecipeKind::Compute(ShaderKind::DepthwiseNative1x7)
    ));
    assert_eq!(
        (selected[0].x, selected[0].y, selected[0].words[1]),
        (2, 1, 128)
    );
    assert_eq!(selected[0].words[60], 13);
    assert_eq!(selected[0].inputs, recipe.inputs);
    assert_eq!(
        (selected[0].node, &selected[0].nodes, selected[0].output),
        (recipe.node, &recipe.nodes, recipe.output)
    );
    let (source, entry, threads) = half_native::source(ShaderKind::DepthwiseNative1x7).unwrap();
    assert_eq!(entry, "depthwise_1x7_half");
    assert_eq!(threads, [64, 1, 1]);
    assert!(source.contains("packed_half4") && source.contains("float4 sum"));
    assert!(!source.contains("threads_per_simdgroup") && !source.contains("threadgroup_barrier"));
    assert!(!source.contains("device const float*"));
    let mut strict = [recipe.clone()];
    assert_eq!(
        half_native::select(
            &mut strict,
            &descriptors,
            &crate::runtime::gpu::GpuLimits::default(),
            GpuMatrixPrecision::F32
        )
        .unwrap(),
        0
    );
    assert_unchanged(&strict[0], &recipe);
    for index in 0..descriptors.len() {
        let mut changed = descriptors.clone();
        changed[index].precision = GpuPrecision::F32;
        let mut selected = [recipe.clone()];
        assert_eq!(
            half_native::select(
                &mut selected,
                &changed,
                &crate::runtime::gpu::GpuLimits::default(),
                GpuMatrixPrecision::F16
            )
            .unwrap(),
            0
        );
        assert_unchanged(&selected[0], &recipe);
    }
}

#[test]
fn half_native_depthwise_1x7_rejects_other_geometry_layouts_and_capacity() {
    let (recipe, descriptors) = rect_fixture();
    for (index, value) in [
        (0, 1),
        (2, 0),
        (3, 8),
        (4, 8),
        (5, 2),
        (6, 0),
        (7, 2),
        (8, 0),
        (9, 2),
        (10, 2),
        (11, 2),
        (12, 5),
        (13, 2),
        (14, 2),
        (15, 2),
        (16, 2),
        (17, 1),
        (18, u32::MAX),
        (52, 1),
        (60, u32::MAX),
    ] {
        let mut original = recipe.clone();
        original.words[index] = value;
        let mut selected = [original.clone()];
        assert_eq!(
            half_native::select(
                &mut selected,
                &descriptors,
                &crate::runtime::gpu::GpuLimits::default(),
                GpuMatrixPrecision::F16
            )
            .unwrap(),
            0,
            "accepted word{index}={value}"
        );
        assert_unchanged(&selected[0], &original);
    }
    for case in 0..3 {
        let mut changed = descriptors.clone();
        let mut limits = crate::runtime::gpu::GpuLimits::default();
        match case {
            0 => changed[recipe.inputs[0].0].layout = GpuLayout::PaddedNhwc { channels: 7 },
            1 => changed[recipe.inputs[1].0].shape = Shape::new(&[7, 1, 7, 1]).unwrap(),
            2 => limits.max_compute_workgroup_size_x = 63,
            _ => unreachable!(),
        }
        let mut selected = [recipe.clone()];
        assert_eq!(
            half_native::select(&mut selected, &changed, &limits, GpuMatrixPrecision::F16).unwrap(),
            0
        );
        assert_unchanged(&selected[0], &recipe);
    }
}

#[test]
fn half_native_depthwise_1x7_prefix_and_escaping_values_remain_available() {
    let shape = [2, 7, 1, 17];
    for usage in [Use::ResidualSwish, Use::Escape] {
        let program = Program::compile(rect_graph(shape, [0, 3, 0, 3], usage)).unwrap();
        let conv = program.instructions[0].output;
        let mut plan = bind_program(&program, &shape, ExecutionLimits::default()).unwrap();
        let mut descriptors = plan
            .shapes
            .iter()
            .map(|&shape| GpuTensorDescriptor::canonical(shape, GpuPrecision::MixedF16))
            .collect();
        let mut recipes = physical::lower(
            &program,
            &mut plan,
            program.output,
            &crate::runtime::gpu::GpuLimits::default(),
            GpuPrecision::MixedF16,
            &mut descriptors,
            GpuMatrixPrecision::F16,
            &mut |_| Ok(false),
        )
        .unwrap();
        assert_eq!(
            half_native::select(
                &mut recipes,
                &descriptors,
                &crate::runtime::gpu::GpuLimits::default(),
                GpuMatrixPrecision::F16
            )
            .unwrap(),
            1
        );
        let dw = recipes
            .iter()
            .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::DepthwiseNative1x7)))
            .unwrap();
        if matches!(usage, Use::Escape) {
            assert_eq!(dw.output, conv);
            assert_eq!(dw.nodes, [0]);
            assert_eq!(dw.words[64], 0);
            assert!(recipes.iter().filter(|r| r.inputs.contains(&conv)).count() >= 2);
        } else {
            assert_eq!(dw.nodes, [0, 1, 2, 3]);
            assert_eq!(dw.words[64], 2);
            assert_eq!(dw.words[71], 8);
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
            conv,
            &mut prefix,
            &mut scratch,
        )
        .unwrap();
        let mut descriptors = prefix
            .shapes
            .iter()
            .map(|&shape| GpuTensorDescriptor::canonical(shape, GpuPrecision::MixedF16))
            .collect();
        let mut recipes = physical::lower(
            &program,
            &mut prefix,
            conv,
            &crate::runtime::gpu::GpuLimits::default(),
            GpuPrecision::MixedF16,
            &mut descriptors,
            GpuMatrixPrecision::F16,
            &mut |_| Ok(false),
        )
        .unwrap();
        assert_eq!(
            half_native::select(
                &mut recipes,
                &descriptors,
                &crate::runtime::gpu::GpuLimits::default(),
                GpuMatrixPrecision::F16
            )
            .unwrap(),
            1
        );
        assert_eq!(recipes.len(), 1);
        assert_eq!(recipes[0].nodes, [0]);
        assert_eq!(recipes[0].output, conv);
        assert_eq!(recipes[0].words[64], 0);
    }
}

#[test]
#[ignore = "requires physical Metal GPU; Half1x7 N2/odd/narrow/asymmetric tails against independent CPU plus strict matrix policy"]
fn gpu_hardware_half_native_depthwise_1x7_borders_tails_and_strict_matrices() {
    let disabled = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
    for (shape, pads) in [
        ([2, 3, 1, 1], [0, 3, 0, 3]),
        ([2, 7, 1, 3], [0, 0, 0, 6]),
        ([2, 7, 1, 17], [0, 1, 0, 7]),
        ([1, 120, 1, 33], [0, 3, 0, 3]),
    ] {
        let graph = rect_graph(shape, pads, Use::Plain);
        let input = dyadic(shape.iter().product(), 3);
        let oracle = expected_half(graph.clone(), &shape, &input);
        let mut session =
            GpuSession::new(compile(graph.clone()), &shape, ExecutionLimits::default()).unwrap();
        assert_selected(&session, ShaderKind::DepthwiseNative1x7);
        assert_close(
            "Half1x7 bounded windows",
            session.run(&input).unwrap().data(),
            &oracle,
        );
        let reference_expected = reference(graph.clone(), &shape, &input).1;
        let compiled = Arc::new(
            GpuCompiledModel::compile_with_options(
                graph,
                disabled.clone(),
                CompileOptions {
                    gpu: GpuCompileOptions {
                        precision: GpuPrecision::MixedF16,
                        matrix_precision: GpuMatrixPrecision::F32,
                    },
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let mut session = GpuSession::new(compiled, &shape, ExecutionLimits::default()).unwrap();
        assert!(
            !session
                .recipes
                .iter()
                .any(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::DepthwiseNative1x7)))
        );
        let reference_expected = match session.descriptors[session.selected_output.0].precision {
            GpuPrecision::F32 => reference_expected,
            GpuPrecision::MixedF16 => reference_expected
                .into_iter()
                .map(|value| half::f16::from_f32(value).to_f32())
                .collect(),
        };
        assert_close(
            "Half1x7 strict matrix storage",
            session.run(&input).unwrap().data(),
            &reference_expected,
        );
    }
}

#[test]
#[ignore = "requires physical Metal GPU; Half1x7 existing bias/residual/Swish fusion, escaping output, prefix and Concat offsets"]
fn gpu_hardware_half_native_depthwise_1x7_fusions_prefix_and_offsets() {
    for (shape, usage) in [
        ([2, 7, 1, 17], Use::ResidualSwish),
        ([1, 7, 1, 17], Use::Concat),
        ([2, 7, 1, 17], Use::Escape),
    ] {
        let graph = rect_graph(shape, [0, 3, 0, 3], usage);
        let input = dyadic(shape.iter().product(), 3);
        let oracle = expected_half(graph.clone(), &shape, &input);
        let compiled = compile(graph);
        let mut session =
            GpuSession::new(compiled.clone(), &shape, ExecutionLimits::default()).unwrap();
        assert_selected(&session, ShaderKind::DepthwiseNative1x7);
        if matches!(usage, Use::Concat) {
            assert!(session.recipes.iter().any(|r| matches!(
                r.kind,
                RecipeKind::Compute(ShaderKind::DepthwiseNative1x7)
            ) && r.words[60] > 0));
        }
        if matches!(usage, Use::ResidualSwish | Use::Concat) {
            assert!(session.recipes.iter().any(|r| matches!(
                r.kind,
                RecipeKind::Compute(ShaderKind::DepthwiseNative1x7)
            ) && r.words[64] == 2
                && r.words[71] == 8));
        }
        assert_close(
            "Half1x7 fused physical graph",
            session.run(&input).unwrap().data(),
            &oracle,
        );
        let mut prefix =
            GpuSession::new_prefix(compiled, &shape, ExecutionLimits::default(), 0).unwrap();
        assert_selected(&prefix, ShaderKind::DepthwiseNative1x7);
        assert_eq!(prefix.recipes[0].words[64], 0);
        let oracle = expected_half(rect_graph(shape, [0, 3, 0, 3], Use::Plain), &shape, &input);
        assert_close(
            "Half1x7 selected Conv prefix",
            prefix.run(&input).unwrap().data(),
            &oracle,
        );
    }
}

#[test]
#[ignore = "requires physical Metal GPU; native Half1x7 direct seven-buffer ABI with odd output offset and sentinels"]
fn gpu_hardware_half_native_depthwise_1x7_native_offset_sentinels() {
    use backend::test_support::MetalTestShader;
    use resident::Batch;
    let shape = [2, 7, 1, 17];
    let graph = rect_graph(shape, [0, 3, 0, 3], Use::ResidualSwish);
    let context = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
    let compiled = Arc::new(
        GpuCompiledModel::compile_with_options(
            graph.clone(),
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
    let session = GpuSession::new(compiled, &shape, ExecutionLimits::default()).unwrap();
    assert_selected(&session, ShaderKind::DepthwiseNative1x7);
    let recipe = session
        .recipes
        .iter()
        .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::DepthwiseNative1x7)))
        .unwrap();
    let input = dyadic(shape.iter().product(), 3);
    let oracle = expected_half(graph, &shape, &input);
    let precision = GpuPrecision::MixedF16;
    let resolve = |id: TensorId| {
        session
            .compiled
            .weights
            .constants
            .get(id.0)
            .and_then(Option::as_ref)
            .unwrap_or_else(|| &session.buffers[session.plan.slots[id.0]])
            .clone()
    };
    let a = resolve(recipe.inputs[0]);
    context
        .upload_resident(&a, &precision.encode(&input))
        .unwrap();
    let b = resolve(recipe.inputs[1]);
    let c = resolve(recipe.inputs[2]);
    let d = resolve(recipe.inputs[3]);
    let dummy = session.compiled.weights.dummy.clone();
    let prefix = 5;
    let suffix = 7;
    let initial = vec![-777.; prefix + oracle.len() + suffix];
    let output = context
        .create_buffer_init(
            "Half1x7 sentinel output",
            &precision.encode(&initial),
            BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        )
        .unwrap();
    let mut words = recipe.words;
    words[60] = prefix as u32;
    let (source, entry, workgroup_size) =
        half_native::source(ShaderKind::DepthwiseNative1x7).unwrap();
    let mut batch = Batch::new(context).unwrap();
    batch
        .dispatch_msl(
            "half_depthwise1x7_sentinels",
            MetalTestShader {
                source: &source,
                entry_point: entry,
                workgroup_size,
            },
            &[
                (&a, true),
                (&b, true),
                (&c, true),
                (&d, true),
                (&dummy, true),
                (&output, false),
            ],
            &words,
            [recipe.x, recipe.y, 1],
        )
        .unwrap();
    assert_eq!(batch.statistics.native_resident_dispatches, 1);
    let (bytes, _) = batch
        .read(&output, precision.bytes(initial.len()), false)
        .unwrap();
    let mut actual = vec![0.; initial.len()];
    precision.decode(&bytes, &mut actual).unwrap();
    assert_eq!(&actual[..prefix], &initial[..prefix]);
    assert_eq!(
        &actual[prefix + oracle.len()..],
        &initial[prefix + oracle.len()..]
    );
    assert_close(
        "Half1x7 native odd offset",
        &actual[prefix..prefix + oracle.len()],
        &oracle,
    );
}
