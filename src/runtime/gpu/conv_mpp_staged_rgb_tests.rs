//! Measured stride-two staging preserves the original full-view fallback.
use super::*;

fn rgb_graph(
    input: [usize; 4],
    channels: usize,
    pads: [usize; 4],
    fused: bool,
    concat: bool,
) -> Arc<Model> {
    let output = ConvCase {
        name: "staged RGB K3s2",
        input,
        channels,
        kernel: [3; 2],
        stride: [2; 2],
        dilation: [1; 2],
        pads,
        group: 1,
        bias: true,
    }
    .output();
    let mut parts = vec![tensor("bias", &[channels], &dyadic(channels, 2, 16.))];
    for branch in 0..if concat { 2 } else { 1 } {
        let weight = format!("w{branch}");
        let convolved = format!("conv{branch}");
        let added = format!("added{branch}");
        let activated = if concat {
            format!("branch{branch}")
        } else {
            "y".into()
        };
        parts.push(tensor(
            &weight,
            &[channels, input[1], 3, 3],
            &dyadic(channels * input[1] * 9, branch, 64.),
        ));
        parts.push(node(
            "Conv",
            &["x", &weight, "bias"],
            if fused { &convolved } else { &activated },
            &[
                integers("kernel_shape", &[3; 2]),
                integers("strides", &[2; 2]),
                integers("pads", &pads),
            ],
        ));
        if fused {
            let residual = format!("residual{branch}");
            parts.push(tensor(
                &residual,
                &output,
                &dyadic(output.iter().product(), branch + 1, 32.),
            ));
            parts.push(node("Add", &[&convolved, &residual], &added, &[]));
            parts.push(node("Relu", &[&added], &activated, &[]));
        }
    }
    let mut result = output;
    if concat {
        parts.push(node(
            "Concat",
            &["branch0", "branch1"],
            "y",
            &[integer("axis", 1)],
        ));
        result[1] *= 2;
    }
    model(parts, &input, &result)
}

fn rgb_fixture(
    channels: usize,
    output: usize,
) -> (Program, ShapePlan, Vec<GpuTensorDescriptor>, Vec<Recipe>) {
    let shape = [2, channels, 9, 17];
    let program = Program::compile(rgb_graph(shape, output, [1; 4], false, false)).unwrap();
    let plan = bind_program(&program, &shape, ExecutionLimits::default()).unwrap();
    let descriptors = plan
        .shapes
        .iter()
        .map(|&shape| GpuTensorDescriptor::canonical(shape, GpuPrecision::MixedF16))
        .collect();
    let recipes = operators::recipes(&program, &plan, &wgpu::Limits::default()).unwrap();
    (program, plan, descriptors, recipes)
}

#[test]
fn staged_rgb_k3s2_admission_preserves_canonical_operands_and_checked_grid() {
    for (input_channels, channels) in [(3, 48), (3, 64), (24, 48)] {
        let (program, mut plan, mut descriptors, recipes) = rgb_fixture(input_channels, channels);
        let shapes = plan.shapes.clone();
        let slots = plan.slots.clone();
        let inputs = recipes[0].inputs.clone();
        let mut probed = Vec::new();
        let lowered = conv_mpp_lower::lower(
            &program,
            &mut plan,
            recipes,
            &mut descriptors,
            &wgpu::Limits::default(),
            GpuMatrixPrecision::F16,
            &mut |recipe| {
                probed.push(recipe.kind.clone());
                Ok(true)
            },
        )
        .unwrap();
        assert!(matches!(
            probed.as_slice(),
            [RecipeKind::Compute(ShaderKind::ConvMpp3s2Staged)]
        ));
        assert_eq!(lowered.len(), 1);
        assert!(matches!(
            lowered[0].kind,
            RecipeKind::Compute(ShaderKind::ConvMpp3s2Staged)
        ));
        assert_eq!(lowered[0].inputs, inputs);
        assert_eq!(lowered[0].words[32..39], [2, 4, 16, 2, 64, 1, 1]);
        assert_eq!((lowered[0].x, lowered[0].y), (16, 1));
        assert_eq!(plan.shapes, shapes);
        assert_eq!(plan.slots, slots);
        assert!(descriptors.iter().all(|d| d.layout == GpuLayout::Canonical));
    }
}

#[test]
fn staged_rgb_k3s2_rejected_compilation_retains_fullview_pack_or_original() {
    for (input_channels, all_rejected) in [(3, false), (3, true), (24, false), (24, true)] {
        let (program, mut plan, mut descriptors, recipes) = rgb_fixture(input_channels, 48);
        let original = recipes.clone();
        let shapes = plan.shapes.clone();
        let slots = plan.slots.clone();
        let mut probed = Vec::new();
        let lowered = conv_mpp_lower::lower(
            &program,
            &mut plan,
            recipes,
            &mut descriptors,
            &wgpu::Limits::default(),
            GpuMatrixPrecision::F16,
            &mut |recipe| {
                let RecipeKind::Compute(kind) = recipe.kind else {
                    panic!("compute recipe")
                };
                probed.push(kind);
                Ok(!all_rejected && kind == ShaderKind::ConvMpp3s2)
            },
        )
        .unwrap();
        assert_eq!(
            probed,
            [ShaderKind::ConvMpp3s2Staged, ShaderKind::ConvMpp3s2]
        );
        if all_rejected {
            assert_eq!(plan.shapes, shapes);
            assert_eq!(plan.slots, slots);
            assert_eq!(descriptors.len(), shapes.len());
            assert_eq!(lowered.len(), original.len());
            for (actual, original) in lowered.iter().zip(&original) {
                assert_eq!(actual.words, original.words);
                assert_eq!(actual.inputs, original.inputs);
                assert_eq!((actual.x, actual.y), (original.x, original.y));
                assert!(
                    matches!((&actual.kind, &original.kind), (RecipeKind::Compute(a), RecipeKind::Compute(b)) if a == b)
                );
            }
        } else {
            assert_eq!(plan.shapes.len(), shapes.len() + 1);
            assert!(matches!(
                lowered[0].kind,
                RecipeKind::Compute(ShaderKind::PackNhwc)
            ));
            assert!(matches!(
                lowered[1].kind,
                RecipeKind::Compute(ShaderKind::ConvMpp3s2)
            ));
            assert_eq!(lowered[1].inputs[0], lowered[0].output);
            assert_eq!(
                descriptors[lowered[0].output.0].layout,
                GpuLayout::PaddedNhwc {
                    channels: input_channels,
                }
            );
        }
    }
    for (input_channels, kind) in [
        (3, ErrorKind::DeviceLost),
        (3, ErrorKind::Allocation),
        (24, ErrorKind::DeviceLost),
        (24, ErrorKind::Allocation),
    ] {
        let (program, mut plan, mut descriptors, recipes) = rgb_fixture(input_channels, 48);
        let shapes = plan.shapes.clone();
        let slots = plan.slots.clone();
        let count = descriptors.len();
        let error = conv_mpp_lower::lower(
            &program,
            &mut plan,
            recipes,
            &mut descriptors,
            &wgpu::Limits::default(),
            GpuMatrixPrecision::F16,
            &mut |_| {
                Err(RuntimeError::with_kind(
                    kind,
                    "staged RGB admission failure",
                ))
            },
        )
        .err()
        .unwrap();
        assert_eq!(error.kind(), kind);
        assert_eq!(plan.shapes, shapes);
        assert_eq!(plan.slots, slots);
        assert_eq!(descriptors.len(), count);
    }
}

#[test]
fn staged_rgb_k3s2_guards_preserve_default_precision_capacity_and_wide_fallbacks() {
    // The noisy RGB ->24 comparison is not an admission. Cin24's one measured
    // output family and the original RGB families cannot admit nearby cases.
    for (cin, cout) in [(128, 256), (96, 48), (3, 24), (3, 32), (24, 24), (24, 64)] {
        let (program, mut plan, mut descriptors, recipes) = rgb_fixture(cin, cout);
        let mut probed = Vec::new();
        let lowered = conv_mpp_lower::lower(
            &program,
            &mut plan,
            recipes,
            &mut descriptors,
            &wgpu::Limits::default(),
            GpuMatrixPrecision::F16,
            &mut |recipe| {
                probed.push(recipe.kind.clone());
                Ok(true)
            },
        )
        .unwrap();
        assert!(matches!(
            probed.as_slice(),
            [RecipeKind::Compute(ShaderKind::ConvMpp3s2)]
        ));
        assert!(matches!(
            lowered[0].kind,
            RecipeKind::Compute(ShaderKind::PackNhwc)
        ));
    }
    for (input_channels, case) in (0..7).flat_map(|case| [(3, case), (24, case)]) {
        let (program, mut plan, mut descriptors, recipes) = rgb_fixture(input_channels, 48);
        let mut limits = wgpu::Limits::default();
        let matrix = if case == 0 {
            GpuMatrixPrecision::F32
        } else {
            GpuMatrixPrecision::F16
        };
        match case {
            1 => {
                limits.max_compute_workgroup_storage_size =
                    if input_channels == 3 { 927 } else { 7343 };
            }
            2 => limits.max_compute_invocations_per_workgroup = 127,
            3 => {
                for descriptor in &mut descriptors {
                    descriptor.precision = GpuPrecision::F32;
                }
            }
            4 => {
                descriptors[program.input.0].layout = GpuLayout::PaddedNhwc {
                    channels: input_channels,
                };
            }
            5 => {
                let bias = recipes[0].inputs[2];
                descriptors[bias.0].precision = GpuPrecision::F32;
            }
            6 => limits.max_compute_workgroup_size_x = 127,
            _ => {}
        }
        let mut staged = false;
        let lowered = conv_mpp_lower::lower(
            &program,
            &mut plan,
            recipes,
            &mut descriptors,
            &limits,
            matrix,
            &mut |recipe| {
                staged |= matches!(
                    recipe.kind,
                    RecipeKind::Compute(ShaderKind::ConvMpp3s2Staged)
                );
                Ok(true)
            },
        )
        .unwrap();
        assert!(!staged);
        assert!(
            !lowered
                .iter()
                .any(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::ConvMpp3s2Staged)))
        );
        if case == 1 {
            assert!(
                lowered
                    .iter()
                    .any(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::ConvMpp3s2)))
            );
        }
    }
}

#[test]
fn staged_c24_k3s2_shared_capacity_boundary_is_exact() {
    for shared_bytes in [7343, 7344] {
        let (program, mut plan, mut descriptors, recipes) = rgb_fixture(24, 48);
        let limits = wgpu::Limits {
            max_compute_workgroup_storage_size: shared_bytes,
            ..Default::default()
        };
        let mut probed = Vec::new();
        let lowered = conv_mpp_lower::lower(
            &program,
            &mut plan,
            recipes,
            &mut descriptors,
            &limits,
            GpuMatrixPrecision::F16,
            &mut |recipe| {
                probed.push(recipe.kind.clone());
                Ok(true)
            },
        )
        .unwrap();
        if shared_bytes == 7344 {
            assert!(matches!(
                probed.as_slice(),
                [RecipeKind::Compute(ShaderKind::ConvMpp3s2Staged)]
            ));
            assert_eq!(lowered.len(), 1);
            assert_eq!(lowered[0].inputs[0], program.input);
        } else {
            assert!(matches!(
                probed.as_slice(),
                [RecipeKind::Compute(ShaderKind::ConvMpp3s2)]
            ));
            assert!(matches!(
                lowered[0].kind,
                RecipeKind::Compute(ShaderKind::PackNhwc)
            ));
        }
    }
}

#[test]
fn staged_rgb_k3s2_specialization_excludes_runtime_shape_and_rejects_other_families() {
    let (_, _, descriptors, recipes) = rgb_fixture(3, 48);
    let mut candidate = recipes[0].clone();
    let kernel = conv_mpp::prepare_recipe(&mut candidate, &descriptors, &wgpu::Limits::default())
        .unwrap()
        .unwrap();
    candidate.kind = RecipeKind::Compute(kernel.shader());
    candidate.words[37..39].copy_from_slice(&[1, 1]);
    assert!(conv_mpp_staged::prepare_recipe(
        &mut candidate,
        &descriptors,
        &wgpu::Limits::default()
    ));
    let mut words = candidate.words;
    let specialization =
        conv_mpp_staged::Specialization::for_shader(ShaderKind::ConvMpp3s2Staged, &words).unwrap();
    let source = conv_mpp_staged::source(specialization, [GpuPrecision::MixedF16; 6]).unwrap();
    words[2] = 7;
    words[4] = 64;
    words[5] = 48;
    words[6] = 992;
    words[17] = 2;
    let changed =
        conv_mpp_staged::Specialization::for_shader(ShaderKind::ConvMpp3s2Staged, &words).unwrap();
    assert_eq!(specialization, changed);
    assert_eq!(
        source,
        conv_mpp_staged::source(changed, [GpuPrecision::MixedF16; 6]).unwrap()
    );
    assert!(source.contains("threadgroup half staging[3 * 17 * 9]"));
    assert_eq!(specialization.entry(), "conv_mpp3s2_staged");
    assert!(
        conv_mpp_staged::Specialization::for_shader(ShaderKind::ConvMpp2Staged, &words).is_none()
    );
    words[3] = 128;
    assert!(conv_mpp_staged::Specialization::from_words(&words).is_none());
    assert!(conv_mpp_staged::source(specialization, [GpuPrecision::F32; 6]).is_err());
}

#[test]
fn staged_c24_k3s2_specialization_is_shared_across_runtime_widths() {
    let (_, _, descriptors, recipes) = rgb_fixture(24, 48);
    let mut candidate = recipes[0].clone();
    let kernel = conv_mpp::prepare_recipe(&mut candidate, &descriptors, &wgpu::Limits::default())
        .unwrap()
        .unwrap();
    candidate.kind = RecipeKind::Compute(kernel.shader());
    candidate.words[37..39].copy_from_slice(&[1, 1]);
    assert!(conv_mpp_staged::prepare_recipe(
        &mut candidate,
        &descriptors,
        &wgpu::Limits::default(),
    ));
    let specialization =
        conv_mpp_staged::Specialization::for_shader(ShaderKind::ConvMpp3s2Staged, &candidate.words)
            .unwrap();
    let source = conv_mpp_staged::source(specialization, [GpuPrecision::MixedF16; 6]).unwrap();
    candidate.words[2] = 7;
    candidate.words[5] = 24;
    candidate.words[6] = 992;
    candidate.words[17] = 2;
    let changed =
        conv_mpp_staged::Specialization::for_shader(ShaderKind::ConvMpp3s2Staged, &candidate.words)
            .unwrap();
    assert_eq!(specialization, changed);
    assert_eq!(
        source,
        conv_mpp_staged::source(changed, [GpuPrecision::MixedF16; 6]).unwrap()
    );
    assert_eq!(specialization.entry(), "conv_mpp3s2_staged");
    assert!(conv_mpp_staged::source(specialization, [GpuPrecision::F32; 6]).is_err());
}

fn selected_rgb(session: &GpuSession) -> Arc<executor::NativePipeline> {
    assert_eq!(session.input_precision(), GpuPrecision::MixedF16);
    assert!(
        session
            .descriptors
            .iter()
            .all(|d| d.layout == GpuLayout::Canonical)
    );
    assert!(
        !session
            .recipes
            .iter()
            .any(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::PackNhwc)))
    );
    session
        .dispatches
        .iter()
        .find_map(|dispatch| match &dispatch.command {
            Command::Compute {
                shader: ShaderKind::ConvMpp3s2Staged,
                native,
            } => Some(native.pipeline_for_test()),
            _ => None,
        })
        .expect("actual production staged RGB pipeline was not selected")
}

#[test]
#[ignore = "requires physical Metal GPU; production staged RGB stride-two batch/tails/borders and bias/residual/ReLU/Concat offsets against independent dyadic CPU oracle"]
fn gpu_hardware_staged_rgb_k3s2_half_borders_tails_fusion_and_offsets() {
    let context = context();
    for (shape, channels, pads, concat) in [
        ([2, 3, 9, 17], 48, [1; 4], false),
        ([1, 3, 9, 21], 64, [0, 0, 2, 2], true),
    ] {
        let graph = rgb_graph(shape, channels, pads, true, concat);
        let input = dyadic(shape.iter().product(), 1, 32.);
        let oracle = expected(graph.clone(), &shape, &input);
        let mut session = GpuSession::new(
            compile(graph, context.clone()),
            &shape,
            ExecutionLimits::default(),
        )
        .unwrap();
        selected_rgb(&session);
        assert!(session.recipes.iter().any(|r| matches!(
            r.kind,
            RecipeKind::Compute(ShaderKind::ConvMpp3s2Staged)
        ) && r.words[64] > 0));
        if concat {
            assert!(session.recipes.iter().any(|r| matches!(
                r.kind,
                RecipeKind::Compute(ShaderKind::ConvMpp3s2Staged)
            ) && r.words[60] > 0));
        }
        assert_close(
            "production staged RGB K3s2 Half",
            session.run(&input).unwrap().data(),
            &oracle,
        );
    }
}

#[test]
#[ignore = "requires physical Metal GPU; Cin24 ->48 Half production native tails, borders, residual fusion and Concat output offsets"]
fn gpu_hardware_staged_c24_k3s2_half_native_tails_offsets() {
    {
        let context = context();
        for (shape, pads, fused, concat) in [
            ([2, 24, 9, 19], [1; 4], false, false),
            ([1, 24, 7, 21], [0, 2, 2, 0], true, true),
        ] {
            let graph = rgb_graph(shape, 48, pads, fused, concat);
            let input = dyadic(shape.iter().product(), 1, 32.);
            let oracle = expected(graph.clone(), &shape, &input);
            let compiled = compile(graph, context.clone());
            let mut session =
                GpuSession::new(compiled.clone(), &shape, ExecutionLimits::default()).unwrap();
            let pipeline = selected_rgb(&session);
            assert!(session.recipes.iter().any(|recipe| matches!(
                recipe.kind,
                RecipeKind::Compute(ShaderKind::ConvMpp3s2Staged)
            ) && recipe.words[3..5] == [24, 48]));
            let _ = pipeline;
            if concat {
                assert!(session.recipes.iter().any(|recipe| matches!(
                    recipe.kind,
                    RecipeKind::Compute(ShaderKind::ConvMpp3s2Staged)
                ) && recipe.words[60] > 0));
            }
            assert_close(
                "production staged Cin24 K3s2 Half",
                session.run(&input).unwrap().data(),
                &oracle,
            );
        }
    }
}

fn symbolic_rgb_graph(input_channels: usize) -> Arc<Model> {
    let symbolic_info = |name: &str, channels, height, symbol: &[u8]| {
        let dimensions = [
            bytes(1, &number(1, 1)),
            bytes(1, &number(1, channels)),
            bytes(1, &number(1, height)),
            bytes(1, &bytes(2, symbol)),
        ]
        .concat();
        [
            bytes(1, name.as_bytes()),
            bytes(
                2,
                &bytes(1, &[number(1, 1), bytes(2, &dimensions)].concat()),
            ),
        ]
        .concat()
    };
    let graph = [
        tensor(
            "w",
            &[48, input_channels, 3, 3],
            &dyadic(48 * input_channels * 9, 0, 64.),
        ),
        tensor("bias", &[48], &dyadic(48, 2, 16.)),
        node(
            "Conv",
            &["x", "w", "bias"],
            "y",
            &[
                integers("kernel_shape", &[3; 2]),
                integers("strides", &[2; 2]),
                integers("pads", &[1; 4]),
            ],
        ),
        bytes(11, &symbolic_info("x", input_channels as u64, 9, b"width")),
        bytes(12, &symbolic_info("y", 48, 5, b"output_width")),
    ]
    .concat();
    let source = [number(1, 8), bytes(8, &number(2, 13)), bytes(7, &graph)].concat();
    Arc::new(
        Model::load_bytes_with_options(
            &source,
            &LoadOptions {
                optimization: OptimizationPolicy::PreserveArithmetic,
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

#[test]
#[ignore = "requires physical Metal GPU; actual RGB/Cin24 staged pipelines reused across symbolic runtime widths with native graph registration"]
fn gpu_hardware_staged_rgb_k3s2_width_cache_reuses_pipeline() {
    for input_channels in [3, 24] {
        let graph = symbolic_rgb_graph(input_channels);
        let compiled = compile(graph.clone(), context());
        let mut session = GpuSession::new(
            compiled.clone(),
            &[1, input_channels, 9, 17],
            ExecutionLimits::default(),
        )
        .unwrap();
        let first = selected_rgb(&session);
        for width in [17, 21] {
            let shape = [1, input_channels, 9, width];
            session.reshape(&shape).unwrap();
            assert_eq!(
                first,
                selected_rgb(&session),
                "runtime width unexpectedly changed the staged RGB pipeline"
            );
            let input = dyadic(shape.iter().product(), 1, 32.);
            let oracle = expected(graph.clone(), &shape, &input);
            assert_close(
                "staged RGB width reuse",
                session.run(&input).unwrap().data(),
                &oracle,
            );
        }
    }
}

#[test]
#[ignore = "requires physical Metal GPU; force production RGB/Cin24 staged alternate-SIMD scalar branches and validate output-offset sentinels"]
fn gpu_hardware_staged_rgb_k3s2_alternate_simd_fallback_and_sentinels() {
    use resident::{Batch, NativeShader};
    let context = context();
    for input_channels in [3, 24] {
        let shape = [2, input_channels, 9, 17];
        let graph = rgb_graph(shape, 48, [1; 4], false, false);
        let input = dyadic(shape.iter().product(), 1, 32.);
        let oracle = expected(graph.clone(), &shape, &input);
        let session = GpuSession::new(
            compile(graph, context.clone()),
            &shape,
            ExecutionLimits::default(),
        )
        .unwrap();
        selected_rgb(&session);
        let recipe = session
            .recipes
            .iter()
            .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::ConvMpp3s2Staged)))
            .unwrap();
        let mut words = recipe.words;
        let prefix = 5;
        let suffix = 7;
        words[60] = prefix as u32;
        let specialization =
            conv_mpp_staged::Specialization::for_shader(ShaderKind::ConvMpp3s2Staged, &words)
                .unwrap();
        let source = conv_mpp_staged::source(specialization, [GpuPrecision::MixedF16; 6]).unwrap();
        let marker = "group, tid, width, staging);";
        assert!(source.contains(marker));
        let source = source.replace(marker, "group, tid, 16u, staging);");
        let precision = GpuPrecision::MixedF16;
        let make = |label, values: &[f32], usage| {
            context
                .create_buffer_init(label, &precision.encode(values), usage)
                .unwrap()
        };
        let a = make(
            "canonical Half RGB NCHW",
            &input,
            wgpu::BufferUsages::STORAGE,
        );
        let weights = conv_weight_pack::pack_hwio(
            &dyadic(48 * input_channels * 9, 0, 64.),
            48,
            input_channels,
            3,
            3,
        )
        .unwrap();
        let b = make("Half RGB HWIO", &weights, wgpu::BufferUsages::STORAGE);
        let c = make(
            "Half RGB bias",
            &dyadic(48, 2, 16.),
            wgpu::BufferUsages::STORAGE,
        );
        let dummy = make("Half dummy", &[0.], wgpu::BufferUsages::STORAGE);
        let initial = vec![-777.; prefix + oracle.len() + suffix];
        let output = make(
            "Half RGB sentinel output",
            &initial,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let mut batch = Batch::new(context.clone()).unwrap();
        let label = format!("staged_rgb_alternate_simd_c{input_channels}");
        batch
            .dispatch_native(
                &label,
                NativeShader {
                    source: &source,
                    entry_point: specialization.entry(),
                    workgroup_size: conv_mpp_staged::WORKGROUP,
                },
                &[
                    (&a, true),
                    (&b, true),
                    (&c, true),
                    (&dummy, true),
                    (&dummy, true),
                    (&output, false),
                ],
                &words,
                [recipe.x, recipe.y, 1],
            )
            .unwrap();
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
            "staged RGB alternate-width GPU fallback",
            &actual[prefix..prefix + oracle.len()],
            &oracle,
        );
    }
}
