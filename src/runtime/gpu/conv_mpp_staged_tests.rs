//! Cin-only staged K2 admission and physical Metal numerical qualification.
use super::*;
use operators::{Recipe, RecipeKind, ShaderKind};

fn dyadic(count: usize, seed: usize, denominator: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 7 + i / 5 + seed * 11) % 3) as i32 - 1) as f32 / denominator)
        .collect()
}

fn graph(
    input: [usize; 4],
    channels: usize,
    pads: [usize; 4],
    fused: bool,
    concat: bool,
) -> Arc<Model> {
    let output = ConvCase {
        name: "staged K2",
        input,
        channels,
        kernel: [2; 2],
        stride: [1; 2],
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
            &[channels, input[1], 2, 2],
            &dyadic(channels * input[1] * 4, branch, 64.),
        ));
        parts.push(node(
            "Conv",
            &["x", &weight, "bias"],
            if fused { &convolved } else { &activated },
            &[integers("kernel_shape", &[2; 2]), integers("pads", &pads)],
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

fn fixture() -> (Program, ShapePlan, Vec<GpuTensorDescriptor>, Vec<Recipe>) {
    let input = [2, 9, 9, 13];
    let program = Program::compile(graph(input, 33, [0, 0, 1, 1], false, false)).unwrap();
    let plan = bind_program(&program, &input, ExecutionLimits::default()).unwrap();
    let descriptors = plan
        .shapes
        .iter()
        .map(|&shape| GpuTensorDescriptor::canonical(shape, GpuPrecision::MixedF16))
        .collect();
    let recipes = operators::recipes(&program, &plan, &wgpu::Limits::default()).unwrap();
    (program, plan, descriptors, recipes)
}

#[test]
fn staged_k2_admission_uses_canonical_input_without_pack() {
    let (program, mut plan, mut descriptors, recipes) = fixture();
    let shapes = plan.shapes.clone();
    let mut probed = Vec::new();
    let lowered = conv_mpp_lower::lower(
        &program,
        &mut plan,
        recipes,
        &mut descriptors,
        &wgpu::Limits::default(),
        GpuMatrixPrecision::F16,
        &mut |candidate| {
            probed.push(candidate.kind.clone());
            Ok(true)
        },
    )
    .unwrap();
    assert_eq!(lowered.len(), 1);
    assert!(matches!(
        lowered[0].kind,
        RecipeKind::Compute(ShaderKind::ConvMpp2Staged)
    ));
    assert!(matches!(
        probed.as_slice(),
        [RecipeKind::Compute(ShaderKind::ConvMpp2Staged)]
    ));
    assert_eq!(lowered[0].inputs[0], program.input);
    assert_eq!(lowered[0].words[37..39], [1, 1]);
    assert_eq!(plan.shapes, shapes);
    assert!(descriptors.iter().all(|d| d.layout == GpuLayout::Canonical));
}

#[test]
fn staged_k2_unsupported_compilation_keeps_fullview_and_its_pack() {
    let (program, mut plan, mut descriptors, recipes) = fixture();
    let count = plan.shapes.len();
    let mut probed = Vec::new();
    let lowered = conv_mpp_lower::lower(
        &program,
        &mut plan,
        recipes,
        &mut descriptors,
        &wgpu::Limits::default(),
        GpuMatrixPrecision::F16,
        &mut |candidate| {
            let RecipeKind::Compute(kind) = candidate.kind else {
                panic!("compute candidate")
            };
            probed.push(kind);
            Ok(kind != ShaderKind::ConvMpp2Staged)
        },
    )
    .unwrap();
    assert_eq!(probed, [ShaderKind::ConvMpp2Staged, ShaderKind::ConvMpp2s1]);
    assert_eq!(plan.shapes.len(), count + 1);
    assert!(matches!(
        lowered[0].kind,
        RecipeKind::Compute(ShaderKind::PackNhwc)
    ));
    assert!(matches!(
        lowered[1].kind,
        RecipeKind::Compute(ShaderKind::ConvMpp2s1)
    ));
    assert_eq!(lowered[1].inputs[0], lowered[0].output);
}

#[test]
fn staged_k2_rejection_and_resource_failures_publish_no_temporary() {
    for failure in [
        None,
        Some(ErrorKind::DeviceLost),
        Some(ErrorKind::Allocation),
        Some(ErrorKind::ResourceLimit),
    ] {
        let (program, mut plan, mut descriptors, recipes) = fixture();
        let original = recipes.clone();
        let shapes = plan.shapes.clone();
        let slots = plan.slots.clone();
        let count = descriptors.len();
        let result = conv_mpp_lower::lower(
            &program,
            &mut plan,
            recipes,
            &mut descriptors,
            &wgpu::Limits::default(),
            GpuMatrixPrecision::F16,
            &mut |_| {
                failure.map_or(Ok(false), |kind| {
                    Err(RuntimeError::with_kind(kind, "staged admission failure"))
                })
            },
        );
        assert_eq!(plan.shapes, shapes);
        assert_eq!(plan.slots, slots);
        assert_eq!(descriptors.len(), count);
        if let Some(kind) = failure {
            assert_eq!(result.err().unwrap().kind(), kind);
        } else {
            let actual = result.unwrap();
            assert_eq!(actual.len(), original.len());
            for (actual, original) in actual.iter().zip(&original) {
                assert_eq!(actual.words, original.words);
                assert_eq!(actual.inputs, original.inputs);
                assert_eq!((actual.x, actual.y), (original.x, original.y));
                assert!(
                    matches!((&actual.kind, &original.kind), (RecipeKind::Compute(a), RecipeKind::Compute(b)) if a == b)
                );
            }
        }
    }
}

#[test]
fn staged_k2_capacity_precision_and_matrix_policy_retain_existing_paths() {
    for case in 0..4 {
        let (program, mut plan, mut descriptors, recipes) = fixture();
        let mut limits = wgpu::Limits::default();
        let matrix = if case == 2 {
            GpuMatrixPrecision::F32
        } else {
            GpuMatrixPrecision::F16
        };
        if case == 0 {
            limits.max_compute_workgroup_storage_size = 815;
        }
        if case == 1 {
            limits.max_compute_invocations_per_workgroup = 127;
        }
        if case == 3 {
            for descriptor in &mut descriptors {
                descriptor.precision = GpuPrecision::F32;
            }
        }
        let mut staged = false;
        let lowered = conv_mpp_lower::lower(
            &program,
            &mut plan,
            recipes,
            &mut descriptors,
            &limits,
            matrix,
            &mut |candidate| {
                staged |= matches!(
                    candidate.kind,
                    RecipeKind::Compute(ShaderKind::ConvMpp2Staged)
                );
                Ok(true)
            },
        )
        .unwrap();
        assert!(!staged);
        assert!(
            !lowered
                .iter()
                .any(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::ConvMpp2Staged)))
        );
        if matches!(case, 0 | 3) {
            assert!(
                lowered
                    .iter()
                    .any(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::ConvMpp2s1)))
            );
        }
    }
}

#[test]
fn staged_k2_specialization_depends_only_on_channels_and_rejects_nonhalf() {
    let mut words = [0; 128];
    words[3] = 32;
    words[11..=16].copy_from_slice(&[2, 2, 1, 1, 1, 1]);
    words[5] = 9;
    words[6] = 13;
    let first = conv_mpp_staged::Specialization::from_words(&words).unwrap();
    words[2] = 2;
    words[4] = 64;
    words[5] = 24;
    words[6] = 992;
    assert_eq!(
        Some(first),
        conv_mpp_staged::Specialization::from_words(&words)
    );
    assert_eq!(
        conv_mpp_staged::source(first, [GpuPrecision::F32; 6])
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    let mut mixed = [GpuPrecision::MixedF16; 6];
    mixed[5] = GpuPrecision::F32;
    assert_eq!(
        conv_mpp_staged::source(first, mixed).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    words[3] = 65;
    assert!(conv_mpp_staged::Specialization::from_words(&words).is_none());
}

fn context() -> Arc<GpuContext> {
    GpuContext::new_with_options(GpuContextOptions::default()).unwrap()
}
fn compile(graph: Arc<Model>, context: Arc<GpuContext>) -> Arc<GpuCompiledModel> {
    Arc::new(
        GpuCompiledModel::compile_with_options(
            graph,
            context,
            CompileOptions {
                gpu: GpuCompileOptions {
                    precision: GpuPrecision::MixedF16,
                    matrix_precision: GpuMatrixPrecision::F16,
                },
                ..Default::default()
            },
        )
        .unwrap(),
    )
}
fn expected(graph: Arc<Model>, shape: &[usize], input: &[f32]) -> Vec<f32> {
    // Fixtures stay below1/4, with products on a1/2048 lattice; every removed
    // Conv/Add storage boundary is Half-exact. Only final output rounding is
    // needed to turn the independent CPU F32 graph into the exact Half oracle.
    reference(graph, shape, input)
        .1
        .into_iter()
        .map(|x| half::f16::from_f32(x).to_f32())
        .collect()
}
fn selected(session: &GpuSession) -> Arc<executor::NativePipeline> {
    assert_eq!(session.input_precision(), GpuPrecision::MixedF16);
    assert_eq!(
        session.descriptors[session.selected_output.0].precision,
        GpuPrecision::MixedF16
    );
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
        .find_map(|d| match &d.command {
            Command::Compute {
                shader: ShaderKind::ConvMpp2Staged,
                native,
            } => Some(native.pipeline_for_test()),
            _ => None,
        })
        .expect("actual production staged K2 pipeline was not selected")
}

#[test]
#[ignore = "requires physical Metal tensor GPU; production staged K2 Half borders, batch/channel/spatial tails, fused residuals and Concat offsets"]
fn gpu_hardware_staged_k2_half_borders_tails_fusion_and_offsets() {
    let context = context();
    for (shape, channels, pads, concat) in [
        ([2, 9, 9, 13], 33, [0, 0, 1, 1], false),
        ([1, 32, 9, 17], 33, [1, 1, 0, 0], true),
    ] {
        let graph = graph(shape, channels, pads, true, concat);
        let input = dyadic(shape.iter().product(), 1, 32.);
        let oracle = expected(graph.clone(), &shape, &input);
        let mut session = GpuSession::new(
            compile(graph, context.clone()),
            &shape,
            ExecutionLimits::default(),
        )
        .unwrap();
        selected(&session);
        assert!(session.recipes.iter().any(|r| matches!(
            r.kind,
            RecipeKind::Compute(ShaderKind::ConvMpp2Staged)
        ) && r.words[64] > 0));
        if concat {
            assert!(session.recipes.iter().any(|r| matches!(
                r.kind,
                RecipeKind::Compute(ShaderKind::ConvMpp2Staged)
            ) && r.words[60] > 0));
        }
        assert_close(
            "production staged K2 Half",
            session.run(&input).unwrap().data(),
            &oracle,
        );
    }
}

fn symbolic_width_graph() -> Arc<Model> {
    let symbolic_info = |name: &str, channels| {
        let dimensions = [
            bytes(1, &number(1, 1)),
            bytes(1, &number(1, channels)),
            bytes(1, &number(1, 9)),
            bytes(1, &bytes(2, b"width")),
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
        tensor("w", &[33, 32, 2, 2], &dyadic(33 * 32 * 4, 0, 64.)),
        tensor("bias", &[33], &dyadic(33, 2, 16.)),
        node(
            "Conv",
            &["x", "w", "bias"],
            "y",
            &[
                integers("kernel_shape", &[2; 2]),
                integers("pads", &[0, 0, 1, 1]),
            ],
        ),
        bytes(11, &symbolic_info("x", 32)),
        bytes(12, &symbolic_info("y", 33)),
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
#[ignore = "requires physical Metal tensor GPU; actual production Cin-only pipeline reuse across widths and native concurrent registration"]
fn gpu_hardware_staged_k2_width_cache_reuses_pipeline() {
    let shape = [1, 32, 9, 13];
    let graph = symbolic_width_graph();
    let compiled = compile(graph.clone(), context());
    let mut session =
        GpuSession::new(compiled.clone(), &shape, ExecutionLimits::default()).unwrap();
    let first = selected(&session);
    for width in [13, 17] {
        let shape = [1, 32, 9, width];
        session.reshape(&shape).unwrap();
        assert_eq!(
            first,
            selected(&session),
            "image width unexpectedly compiled another pipeline"
        );
        let input = dyadic(shape.iter().product(), 1, 32.);
        let oracle = expected(graph.clone(), &shape, &input);
        assert_close(
            "staged K2 width reuse",
            session.run(&input).unwrap().data(),
            &oracle,
        );
    }
}

#[test]
#[ignore = "requires physical Metal GPU; force the production alternate-width scalar branch over canonical NCHW with Half HWIO weights and sentinels"]
fn gpu_hardware_staged_k2_alternate_simd_fallback_and_sentinels() {
    use resident::{Batch, NativeShader};
    let context = context();
    let shape = [2, 9, 9, 17];
    let graph = graph(shape, 33, [1, 1, 0, 0], false, false);
    let input = dyadic(shape.iter().product(), 1, 32.);
    let oracle = expected(graph.clone(), &shape, &input);
    let session = GpuSession::new(
        compile(graph, context.clone()),
        &shape,
        ExecutionLimits::default(),
    )
    .unwrap();
    selected(&session);
    let recipe = session
        .recipes
        .iter()
        .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::ConvMpp2Staged)))
        .unwrap();
    let mut words = recipe.words;
    let prefix = 5;
    let suffix = 7;
    words[60] = prefix as u32;
    let specialization = conv_mpp_staged::Specialization::from_words(&words).unwrap();
    let source = conv_mpp_staged::source(specialization, [GpuPrecision::MixedF16; 6]).unwrap();
    let marker = "group, tid, width, staging);";
    assert!(source.contains(marker));
    // Exercise this exact production fallback on a width32 machine by changing
    // only the uniform width argument supplied to its body. No CPU fallback.
    let source = source.replace(marker, "group, tid, 16u, staging);");
    let precision = GpuPrecision::MixedF16;
    let make = |label, values: &[f32], usage| {
        context
            .create_buffer_init(label, &precision.encode(values), usage)
            .unwrap()
    };
    let a = make("canonical Half NCHW", &input, wgpu::BufferUsages::STORAGE);
    let weights = dyadic(33 * 9 * 4, 0, 64.);
    let weights = conv_weight_pack::pack_hwio(&weights, 33, 9, 2, 2).unwrap();
    let b = make("Half HWIO", &weights, wgpu::BufferUsages::STORAGE);
    let c = make(
        "Half bias",
        &dyadic(33, 2, 16.),
        wgpu::BufferUsages::STORAGE,
    );
    let dummy = make("Half dummy", &[0.], wgpu::BufferUsages::STORAGE);
    let initial = vec![-777.; prefix + oracle.len() + suffix];
    let output = make(
        "Half sentinel output",
        &initial,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let mut batch = Batch::new(context).unwrap();
    batch
        .dispatch_native(
            "staged_k2_alternate_simd",
            NativeShader {
                source: &source,
                entry_point: conv_mpp_staged::ENTRY,
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
        "staged K2 alternate-width GPU fallback",
        &actual[prefix..prefix + oracle.len()],
        &oracle,
    );
}

#[path = "conv_mpp_staged_rgb_tests.rs"]
mod rgb;
