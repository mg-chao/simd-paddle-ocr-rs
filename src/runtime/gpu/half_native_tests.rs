//! Production native families with canonical half storage and F32 accumulation.
use super::*;
use operators::{RecipeKind, ShaderKind};

fn context() -> Arc<GpuContext> {
    static CONTEXT: OnceLock<Arc<GpuContext>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| GpuContext::new_with_options(GpuContextOptions::default()).unwrap())
        .clone()
}

fn compile(graph: Arc<Model>) -> Arc<GpuCompiledModel> {
    Arc::new(
        GpuCompiledModel::compile_with_options(
            graph,
            context(),
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

fn dyadic(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| (((index * 7 + seed * 11) % 15) as i32 - 7) as f32 / 32.0)
        .collect()
}

fn convolution(case: ConvCase) -> Arc<Model> {
    let weights_shape = [
        case.channels,
        case.input[1] / case.group,
        case.kernel[0],
        case.kernel[1],
    ];
    model(
        vec![
            tensor(
                "w",
                &weights_shape,
                &dyadic(weights_shape.iter().product(), 7),
            ),
            tensor("bias", &[case.channels], &dyadic(case.channels, 11)),
            node(
                "Conv",
                &["x", "w", "bias"],
                "y",
                &[
                    integers("kernel_shape", &case.kernel),
                    integers("strides", &case.stride),
                    integers("dilations", &case.dilation),
                    integers("pads", &case.pads),
                    integer("group", case.group),
                ],
            ),
        ],
        &case.input,
        &case.output(),
    )
}

fn expected_half(graph: Arc<Model>, shape: &[usize], input: &[f32]) -> Vec<f32> {
    // Input, weights and bias are exact half dyadics. Their short products and
    // sums are exact in F32, so only the final storage boundary needs rounding.
    let (_, expected) = reference(graph, shape, input);
    expected
        .iter()
        .map(|&value| half::f16::from_f32(value).to_f32())
        .collect()
}

fn assert_selected(session: &GpuSession, expected: ShaderKind) {
    assert_eq!(session.input_precision(), GpuPrecision::MixedF16);
    assert_eq!(
        session.descriptors[session.selected_output.0].precision,
        GpuPrecision::MixedF16
    );
    assert!(session.recipes.iter().any(|recipe| {
        matches!(recipe.kind, RecipeKind::Compute(kind) if kind == expected)
            && recipe
                .inputs
                .iter()
                .chain(std::iter::once(&recipe.output))
                .all(|id| session.descriptors[id.0].precision == GpuPrecision::MixedF16)
    }));
    assert!(session.dispatches.iter().any(|dispatch| {
        matches!(dispatch.command, Command::Compute { shader, .. } if shader == expected)
    }));
}

#[test]
#[ignore = "requires physical Metal tensor GPU; canonical half pointwise tails and strict-model isolation"]
fn gpu_hardware_half_native_pointwise_tails_and_strict_model() {
    let case = ConvCase {
        name: "canonical half native pointwise",
        input: [2, 17, 1, 67],
        channels: 73,
        kernel: [1; 2],
        stride: [1; 2],
        dilation: [1; 2],
        pads: [0; 4],
        group: 1,
        bias: true,
    };
    let graph = convolution(case);
    let input = dyadic(case.input.iter().product(), 3);
    let expected = expected_half(graph.clone(), &case.input, &input);
    let compiled = compile(graph.clone());
    let strict = Arc::new(
        GpuCompiledModel::compile_with_options(
            graph.clone(),
            compiled.context.clone(),
            CompileOptions::default(),
        )
        .unwrap(),
    );
    assert_eq!(compiled.precision, GpuPrecision::MixedF16);
    assert_eq!(compiled.matrix_precision, GpuMatrixPrecision::F16);
    assert_eq!(strict.precision, GpuPrecision::F32);
    assert_eq!(strict.matrix_precision, GpuMatrixPrecision::F32);

    assert!(Arc::ptr_eq(&compiled.model_arc(), &strict.model_arc()));
    assert!(Arc::ptr_eq(&compiled.context, &strict.context));
    assert!(
        !Arc::ptr_eq(&compiled.weights, &strict.weights),
        "strict F32 and half storage require distinct immutable weight owners"
    );
    let mut selected = GpuSession::new(compiled, &case.input, ExecutionLimits::default()).unwrap();
    assert_selected(&selected, ShaderKind::PointwiseNative64);
    assert!(
        !selected
            .recipes
            .iter()
            .any(|recipe| { matches!(recipe.kind, RecipeKind::Compute(ShaderKind::PackNhwc)) })
    );
    assert_close(case.name, selected.run(&input).unwrap().data(), &expected);
}

#[test]
#[ignore = "requires physical Metal GPU; half depthwise 3/5/9 windows with both stride and spatial tails"]
fn gpu_hardware_half_native_depthwise_windows_strides_and_tails() {
    for (kernel, shader) in [
        (3, ShaderKind::DepthwiseNative3),
        (5, ShaderKind::DepthwiseNative5),
        (9, ShaderKind::DepthwiseNative9),
    ] {
        for stride in [1, 2] {
            let case = ConvCase {
                name: "canonical half native depthwise",
                input: [2, 7, 13, 17],
                channels: 7,
                kernel: [kernel; 2],
                stride: [stride; 2],
                dilation: [1; 2],
                pads: [kernel / 2; 4],
                group: 7,
                bias: true,
            };
            let graph = convolution(case);
            let input = dyadic(case.input.iter().product(), kernel + stride);
            let expected = expected_half(graph.clone(), &case.input, &input);
            let mut selected =
                GpuSession::new(compile(graph), &case.input, ExecutionLimits::default()).unwrap();
            assert_selected(&selected, shader);
            assert_close(case.name, selected.run(&input).unwrap().data(), &expected);
        }
    }
}

#[test]
#[ignore = "requires physical Metal tensor GPU; all-half spatial vertical-stride lowering and packed ABI"]
fn gpu_hardware_half_native_spatial_vertical_stride_and_pack() {
    let case = ConvCase {
        name: "all-half spatial vertical stride",
        input: [2, 9, 11, 17],
        channels: 33,
        kernel: [3; 2],
        stride: [2, 1],
        dilation: [1; 2],
        pads: [1, 2, 3, 0],
        group: 1,
        bias: true,
    };
    let graph = convolution(case);
    let input = dyadic(case.input.iter().product(), 5);
    let expected = expected_half(graph.clone(), &case.input, &input);
    let mut selected =
        GpuSession::new(compile(graph), &case.input, ExecutionLimits::default()).unwrap();
    assert_selected(&selected, ShaderKind::ConvMpp3s21);
    assert!(
        selected
            .descriptors
            .iter()
            .all(|descriptor| descriptor.precision == GpuPrecision::MixedF16)
    );
    let matrix = selected
        .recipes
        .iter()
        .find(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::ConvMpp3s21)))
        .unwrap();
    assert_eq!(matrix.words[37], 1);
    assert_eq!(matrix.words[38], 1);
    let converter = selected
        .recipes
        .iter()
        .find(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::PackNhwc)))
        .unwrap();
    assert_eq!(converter.inputs[0], selected.compiled.program.input);
    assert_eq!(converter.output, matrix.inputs[0]);
    assert_eq!(converter.words[6], 0);
    assert!(selected.dispatches.iter().any(|dispatch| {
        matches!(
            dispatch.command,
            Command::Compute {
                shader: ShaderKind::PackNhwc,
                ..
            }
        )
    }));
    assert_close(case.name, selected.run(&input).unwrap().data(), &expected);
}

#[test]
#[ignore = "requires physical Metal tensor GPU; RGB and even-kernel recognition stems"]
fn gpu_hardware_half_native_recognition_stems() {
    for (channels, kernel, stride, pads, shader) in [
        (3, 3, 2, [1; 4], ShaderKind::ConvMpp3s2),
        (9, 2, 1, [0, 0, 1, 1], ShaderKind::ConvMpp2Staged),
    ] {
        let case = ConvCase {
            name: "all-half recognition stem",
            input: [2, channels, 11, 17],
            channels: 33,
            kernel: [kernel; 2],
            stride: [stride; 2],
            dilation: [1; 2],
            pads,
            group: 1,
            bias: true,
        };
        let graph = convolution(case);
        let input = dyadic(case.input.iter().product(), 5);
        let expected = expected_half(graph.clone(), &case.input, &input);
        let mut session =
            GpuSession::new(compile(graph), &case.input, ExecutionLimits::default()).unwrap();
        assert_selected(&session, shader);
        if shader == ShaderKind::ConvMpp2Staged {
            assert!(
                !session
                    .recipes
                    .iter()
                    .any(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::PackNhwc)))
            );
        }
        assert_close(case.name, session.run(&input).unwrap().data(), &expected);
    }
}

#[cfg(target_os = "macos")]
#[path = "depthwise_1x7_tests.rs"]
mod depthwise_1x7;
