//! Square7 coverage is confined to canonical Half storage and explicit F16 math.
use super::super::tests::{bind_program, integer, integers, model, node, tensor};
use super::*;
use crate::runtime::CompileOptions;
use resident::{Batch, NativeShader};

#[derive(Clone, Copy)]
struct Case {
    shape: [usize; 4],
    stride: [usize; 2],
    pads: [usize; 4],
}
impl Case {
    fn output(self) -> [usize; 4] {
        let [n, c, h, w] = self.shape;
        [
            n,
            c,
            (h + self.pads[0] + self.pads[2] - 7) / self.stride[0] + 1,
            (w + self.pads[1] + self.pads[3] - 7) / self.stride[1] + 1,
        ]
    }

    fn operands(self, branch: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let [n, c, h, w] = self.shape;
        let mut input: Vec<_> = (0..n * c * h * w)
            .map(|i| ((i * 7 + 3) % 31) as f32 / 32. - 15. / 32.)
            .collect();
        let mut weights: Vec<_> = (0..c * 49)
            .map(|i| ((i * 5 + 1 + branch) % 17) as f32 / 64. - 8. / 64.)
            .collect();
        let bias = (0..c)
            .map(|i| ((i * 3 + 2) % 9) as f32 / 128. - 4. / 128.)
            .collect();
        for batch in 0..n {
            input[batch * c * h * w..batch * c * h * w + h * w].fill(1.);
        }
        // An ascending Float dot retains1 here; a Half dot loses it.
        weights[..49].fill(0.);
        weights[3 * 7 + 2] = 2048.;
        weights[3 * 7 + 3] = 1.;
        weights[3 * 7 + 4] = -2048.;
        (input, weights, bias)
    }
}

#[derive(Clone, Copy)]
enum Usage {
    Plain,
    ResidualRelu,
    Concat,
}

fn graph(case: Case, usage: Usage) -> Arc<Model> {
    let channels = case.shape[1];
    let (_, _, bias) = case.operands(0);
    let mut parts = vec![tensor("bias", &[channels], &bias)];
    let branches = if matches!(usage, Usage::Concat) { 2 } else { 1 };
    for branch in 0..branches {
        let weights = format!("weights{branch}");
        let conv = format!("conv{branch}");
        let result = if branches == 2 {
            format!("branch{branch}")
        } else {
            "y".to_owned()
        };
        parts.push(tensor(
            &weights,
            &[channels, 1, 7, 7],
            &case.operands(branch).1,
        ));
        parts.push(node(
            "Conv",
            &["x", &weights, "bias"],
            if matches!(usage, Usage::ResidualRelu) {
                &conv
            } else {
                &result
            },
            &[
                integers("kernel_shape", &[7; 2]),
                integers("strides", &case.stride),
                integers("dilations", &[1; 2]),
                integers("pads", &case.pads),
                integer("group", channels),
            ],
        ));
        if matches!(usage, Usage::ResidualRelu) {
            let residual: Vec<_> = (0..case.output().iter().product::<usize>())
                .map(|i| ((i * 11 + 4) % 17) as f32 / 128. - 8. / 128.)
                .collect();
            parts.push(tensor("residual", &case.output(), &residual));
            parts.push(node("Add", &[&conv, "residual"], "sum", &[]));
            parts.push(node("Relu", &["sum"], &result, &[]));
        }
    }
    let mut output = case.output();
    if branches == 2 {
        parts.push(node(
            "Concat",
            &["branch0", "branch1"],
            "y",
            &[integer("axis", 1)],
        ));
        output[1] *= 2;
    }
    model(parts, &case.shape, &output)
}

fn fixture(case: Case) -> (Recipe, Vec<GpuTensorDescriptor>) {
    let program = Program::compile(graph(case, Usage::Plain)).unwrap();
    let plan = bind_program(&program, &case.shape, ExecutionLimits::default()).unwrap();
    let descriptors = plan
        .shapes
        .iter()
        .map(|&shape| GpuTensorDescriptor::canonical(shape, GpuPrecision::MixedF16))
        .collect();
    let recipe = operators::recipes(&program, &plan, &wgpu::Limits::default())
        .unwrap()
        .remove(0);
    (recipe, descriptors)
}

fn unchanged(actual: &Recipe, original: &Recipe) {
    assert!(matches!((&actual.kind, &original.kind),
        (RecipeKind::Compute(a), RecipeKind::Compute(b)) if a == b));
    assert_eq!(actual.words, original.words);
    assert_eq!(actual.inputs, original.inputs);
    assert_eq!(actual.output, original.output);
    assert_eq!(actual.nodes, original.nodes);
    assert_eq!(
        (actual.node, actual.x, actual.y),
        (original.node, original.x, original.y)
    );
}

#[test]
fn half_depthwise7_actual_detector_recipes_preserve_storage_and_strict_policy() {
    let limits = wgpu::Limits::default();
    for side in [16, 32, 64, 128] {
        let case = Case {
            shape: [1, 96, side, side],
            stride: [1; 2],
            pads: [3; 4],
        };
        let (mut original, descriptors) = fixture(case);
        assert!(matches!(
            original.kind,
            RecipeKind::Compute(ShaderKind::Convolution)
        ));
        assert_eq!(original.words[19], 1);
        original.words[60] = 7;
        let mut recipes = [original.clone()];
        assert_eq!(
            select(&mut recipes, &descriptors, &limits, GpuMatrixPrecision::F16).unwrap(),
            1
        );
        let selected = &recipes[0];
        assert!(matches!(
            selected.kind,
            RecipeKind::Compute(ShaderKind::DepthwiseNative7)
        ));
        let mut packet = selected.words;
        packet[1] = original.words[1];
        assert_eq!(packet, original.words);
        assert_eq!(selected.inputs, original.inputs);
        assert_eq!(selected.output, original.output);
        assert_eq!(selected.nodes, original.nodes);
        let expected = dispatch_grid(96 * side * side.div_ceil(4), &limits).unwrap();
        assert_eq!((selected.x, selected.y), expected);
        assert_eq!(selected.words[1], selected.x * 64);

        let mut strict = [original.clone()];
        assert_eq!(
            select(&mut strict, &descriptors, &limits, GpuMatrixPrecision::F32).unwrap(),
            0
        );
        unchanged(&strict[0], &original);
        let fp32 = descriptors
            .iter()
            .cloned()
            .map(|mut d| {
                d.precision = GpuPrecision::F32;
                d
            })
            .collect::<Vec<_>>();
        let mut strict = [original.clone()];
        assert_eq!(
            select(&mut strict, &fp32, &limits, GpuMatrixPrecision::F16).unwrap(),
            0,
        );
        unchanged(&strict[0], &original);
    }
    // Bindings rather than a model label determine admission; every used
    // source/output must remain canonical Half, and a64-thread group must fit.
    let case = Case {
        shape: [2, 7, 9, 19],
        stride: [1; 2],
        pads: [3; 4],
    };
    let (original, descriptors) = fixture(case);
    for &id in original
        .inputs
        .iter()
        .chain(std::iter::once(&original.output))
    {
        let mut bad = descriptors.clone();
        bad[id.0].precision = GpuPrecision::F32;
        let mut recipes = [original.clone()];
        assert_eq!(
            select(&mut recipes, &bad, &limits, GpuMatrixPrecision::F16).unwrap(),
            0
        );
        unchanged(&recipes[0], &original);
    }
    for (word, value) in [(9, 2), (12, 5), (13, 3), (15, 2), (52, 1)] {
        let mut recipe = original.clone();
        recipe.words[word] = value;
        let mut recipes = [recipe.clone()];
        assert_eq!(
            select(&mut recipes, &descriptors, &limits, GpuMatrixPrecision::F16).unwrap(),
            0
        );
        unchanged(&recipes[0], &recipe);
    }
    let limited = wgpu::Limits {
        max_compute_invocations_per_workgroup: 63,
        ..limits
    };
    let mut recipes = [original.clone()];
    assert_eq!(
        select(
            &mut recipes,
            &descriptors,
            &limited,
            GpuMatrixPrecision::F16
        )
        .unwrap(),
        0
    );
    unchanged(&recipes[0], &original);
    let mut boundary = original.words;
    boundary[2..9].copy_from_slice(&[1, 1, 1, 1, i32::MAX as u32 - 10, 1, i32::MAX as u32 - 10]);
    boundary[0] = boundary[8];
    assert!(depthwise_geometry(&boundary));
    boundary[6] += 1;
    boundary[8] += 1;
    boundary[0] += 1;
    assert!(
        !depthwise_geometry(&boundary),
        "masked tail coordinates must not overflow signed indexing"
    );
}

#[test]
fn half_depthwise7_source_matches_qualified_native_window() {
    let (source, entry, group) = source(ShaderKind::DepthwiseNative7).unwrap();
    // Fingerprint the complete root-qualified MSL, rather than maintaining
    // a second implementation or depending on an untracked reports directory.
    assert_eq!(source.len(), 17_541);
    let fingerprint = source.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    });
    assert_eq!(fingerprint, 0x370a_b714_2bc7_3ec4);
    assert_eq!((entry, group), ("depthwise_native", [64, 1, 1]));
    assert!(!ShaderKind::ALL.contains(&ShaderKind::DepthwiseNative7));
}

fn context() -> Arc<GpuContext> {
    GpuContext::new_with_options(GpuContextOptions::default()).unwrap()
}

fn compile(context: &Arc<GpuContext>, graph: Arc<Model>) -> Arc<GpuCompiledModel> {
    Arc::new(
        GpuCompiledModel::compile_with_options(
            graph,
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
    )
}

fn oracle(case: Case, usage: Usage) -> Vec<f32> {
    let input = case.operands(0).0;
    let (_, values) = super::super::tests::reference(graph(case, usage), &case.shape, &input);
    values
        .into_iter()
        .map(|x| half::f16::from_f32(x).to_f32())
        .collect()
}

#[test]
#[ignore = "requires physical Metal; Half7x7 actual production selection, N2/borders/stride/tails, fused residual and Concat offsets"]
fn gpu_hardware_half_depthwise7_production_borders_fusions_and_offsets() {
    {
        let context = context();
        for (case, usage) in [
            (
                Case {
                    shape: [1, 96, 32, 32],
                    stride: [1; 2],
                    pads: [3; 4],
                },
                Usage::Plain,
            ),
            (
                Case {
                    shape: [2, 17, 9, 19],
                    stride: [1; 2],
                    pads: [3; 4],
                },
                Usage::Plain,
            ),
            (
                Case {
                    shape: [2, 5, 11, 31],
                    stride: [2; 2],
                    pads: [2, 3, 4, 2],
                },
                Usage::ResidualRelu,
            ),
            (
                Case {
                    shape: [1, 7, 8, 35],
                    stride: [1, 2],
                    pads: [2, 3, 4, 2],
                },
                Usage::Concat,
            ),
        ] {
            let expected = oracle(case, usage);
            let compiled = compile(&context, graph(case, usage));
            let mut session =
                GpuSession::new_resident(compiled, &case.shape, ExecutionLimits::default(), None)
                    .unwrap();
            assert!(
                session
                    .recipes
                    .iter()
                    .any(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::DepthwiseNative7)))
            );
            if matches!(usage, Usage::Concat) {
                assert!(session.recipes.iter().any(|r| matches!(
                    r.kind,
                    RecipeKind::Compute(ShaderKind::DepthwiseNative7)
                ) && r.words[60] > 0));
            }
            if matches!(usage, Usage::ResidualRelu) {
                assert!(session.recipes.iter().any(|r| matches!(
                    r.kind,
                    RecipeKind::Compute(ShaderKind::DepthwiseNative7)
                ) && r.nodes.len() == 3));
            }
            context
                .upload_resident(
                    &session.resident_input().buffer,
                    &GpuPrecision::MixedF16.encode(&case.operands(0).0),
                )
                .unwrap();
            let mut batch = Batch::new(context.clone()).unwrap();
            let output = batch.graph(&mut session, "Half7x7 production").unwrap();
            let (bytes, stats) = batch
                .read(
                    &output.buffer,
                    GpuPrecision::MixedF16.bytes(expected.len()),
                    false,
                )
                .unwrap();
            assert!(stats.native_graph_dispatches > 0);
            let mut actual = vec![0.; expected.len()];
            GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
            assert_eq!(
                actual, expected,
                "Half7x7 Float accumulation and final storage"
            );
        }
    }
}

#[test]
#[ignore = "requires physical Metal; unsupported Half7x7 PSO retains original generic recipe and valid output"]
fn gpu_hardware_half_depthwise7_compile_fallback() {
    let case = Case {
        shape: [2, 7, 9, 19],
        stride: [1; 2],
        pads: [3; 4],
    };
    for rejection in [ErrorKind::Unsupported, ErrorKind::GpuValidation] {
        let context = context();
        let compiled = compile(&context, graph(case, Usage::Plain));
        compiled
            .pipelines
            .reject_depthwise7_for_test(RuntimeError::with_kind(
                rejection,
                "fixture rejected native7 source",
            ));
        let mut session =
            GpuSession::new(compiled, &case.shape, ExecutionLimits::default()).unwrap();
        assert_eq!(session.recipes.len(), 1);
        assert!(matches!(
            session.recipes[0].kind,
            RecipeKind::Compute(ShaderKind::Convolution)
        ));
        let expected = oracle(case, Usage::Plain);
        assert_eq!(session.run(&case.operands(0).0).unwrap().data(), &expected);
    }
}

#[test]
#[ignore = "requires physical Metal; Half7x7 invalid vertical NaN taps, forced scalar path, odd p60 and sentinels"]
fn gpu_hardware_half_depthwise7_raw_nan_padding_scalar_and_guards() {
    let context = context();
    let case = Case {
        shape: [2, 5, 1, 19],
        stride: [1; 2],
        pads: [3; 4],
    };
    let (input, mut weights, bias) = case.operands(0);
    for channel in 0..case.shape[1] {
        for ky in 0..7 {
            if ky != 3 {
                weights[channel * 49 + ky * 7..channel * 49 + ky * 7 + 7].fill(f32::NAN);
            }
        }
    }
    let expected = oracle(case, Usage::Plain);
    let (mut recipe, descriptors) = fixture(case);
    select(
        std::slice::from_mut(&mut recipe),
        &descriptors,
        &context.limits,
        GpuMatrixPrecision::F16,
    )
    .unwrap();
    assert!(matches!(
        recipe.kind,
        RecipeKind::Compute(ShaderKind::DepthwiseNative7)
    ));
    let upload = |label, values: &[f32], usage| {
        context
            .create_buffer_init(label, &GpuPrecision::MixedF16.encode(values), usage)
            .unwrap()
    };
    let a = upload("DW7 Half input", &input, wgpu::BufferUsages::STORAGE);
    let b = upload(
        "DW7 invalid-tap NaN weights",
        &weights,
        wgpu::BufferUsages::STORAGE,
    );
    let c = upload("DW7 initial bias", &bias, wgpu::BufferUsages::STORAGE);
    let unused = upload(
        "DW7 unused NaN",
        &[f32::NAN; 8],
        wgpu::BufferUsages::STORAGE,
    );
    let prefix = 7;
    let guarded = vec![-777.; prefix + expected.len() + 11];
    for forced in [false, true] {
        let output = upload(
            "DW7 guarded output",
            &guarded,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let (mut source, entry, workgroup_size) = source(ShaderKind::DepthwiseNative7).unwrap();
        if forced {
            source = source.replace(
                "if (ox + 4u <= ow && ix0 >= 0",
                "if (false && ox + 4u <= ow && ix0 >= 0",
            );
        }
        let mut words = recipe.words;
        words[60] = prefix as u32;
        let mut batch = Batch::new(context.clone()).unwrap();
        batch
            .dispatch_native(
                if forced {
                    "DW7 forced scalar"
                } else {
                    "DW7 complete windows"
                },
                NativeShader {
                    source: &source,
                    entry_point: entry,
                    workgroup_size,
                },
                &[
                    (&a, true),
                    (&b, true),
                    (&c, true),
                    (&unused, true),
                    (&unused, true),
                    (&output, false),
                ],
                &words,
                [recipe.x + 3, recipe.y, 1],
            )
            .unwrap();
        let (bytes, _) = batch
            .read(&output, GpuPrecision::MixedF16.bytes(guarded.len()), false)
            .unwrap();
        let mut actual = vec![0.; guarded.len()];
        GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
        assert_eq!(&actual[..prefix], &guarded[..prefix]);
        assert_eq!(
            &actual[prefix + expected.len()..],
            &guarded[prefix + expected.len()..]
        );
        assert_eq!(&actual[prefix..prefix + expected.len()], expected);
    }
}
