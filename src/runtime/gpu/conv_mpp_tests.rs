//! Production NHWC/HWIO lowering, matrix precision, tails, and shared packs.
use super::*;
use crate::runtime::GpuMatrixPrecision;
use operators::{RecipeKind, ShaderKind};

fn context() -> Arc<GpuContext> {
    GpuContext::new_with_options(GpuContextOptions::default()).unwrap()
}

fn compile(
    context: &Arc<GpuContext>,
    graph: Arc<Model>,
    matrix_precision: GpuMatrixPrecision,
) -> Arc<GpuCompiledModel> {
    Arc::new(
        GpuCompiledModel::compile_with_options(
            graph,
            context.clone(),
            CompileOptions {
                gpu: GpuCompileOptions {
                    matrix_precision,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

fn rounded(values: &[f32], matrix_precision: GpuMatrixPrecision) -> Vec<f32> {
    values
        .iter()
        .map(|&value| match matrix_precision {
            GpuMatrixPrecision::F32 => value,
            GpuMatrixPrecision::F16 => half::f16::from_f32(value).to_f32(),
        })
        .collect()
}

fn convolution(case: ConvCase, weight_precision: GpuMatrixPrecision) -> Arc<Model> {
    let weights_shape = [
        case.channels,
        case.input[1] / case.group,
        case.kernel[0],
        case.kernel[1],
    ];
    let mut parts = vec![tensor(
        "weights",
        &weights_shape,
        &rounded(&values(weights_shape.iter().product(), 7), weight_precision),
    )];
    let mut inputs = vec!["x", "weights"];
    if case.bias {
        // Bias is never quantized by matrix precision.
        parts.push(tensor("bias", &[case.channels], &values(case.channels, 11)));
        inputs.push("bias");
    }
    parts.push(node(
        "Conv",
        &inputs,
        "y",
        &[
            integers("kernel_shape", &case.kernel),
            integers("strides", &case.stride),
            integers("dilations", &case.dilation),
            integers("pads", &case.pads),
            integer("group", case.group),
        ],
    ));
    model(parts, &case.input, &case.output())
}

fn cases() -> [ConvCase; 8] {
    let base = ConvCase {
        name: "MPP 3x3 stride one asymmetric borders and tails",
        input: [2, 8, 9, 13],
        channels: 33,
        kernel: [3; 2],
        stride: [1; 2],
        dilation: [1; 2],
        pads: [2, 0, 1, 3],
        group: 1,
        bias: true,
    };
    [
        base,
        ConvCase {
            name: "MPP 3x3 RGB stem reduction tails",
            input: [2, 3, 11, 17],
            stride: [2; 2],
            pads: [1; 4],
            ..base
        },
        ConvCase {
            name: "MPP 2x2 recognition stem asymmetric SAME upper borders",
            input: [2, 9, 11, 17],
            kernel: [2; 2],
            pads: [0, 0, 1, 1],
            ..base
        },
        ConvCase {
            name: "MPP 3x3 stride two reduction tails",
            input: [2, 9, 11, 17],
            channels: 48,
            stride: [2; 2],
            pads: [0, 2, 3, 1],
            bias: false,
            ..base
        },
        ConvCase {
            name: "MPP 3x3 recognition vertical stride",
            input: [2, 9, 11, 17],
            channels: 33,
            stride: [2, 1],
            pads: [1, 2, 3, 0],
            ..base
        },
        ConvCase {
            name: "MPP 3x3 horizontal stride",
            input: [2, 9, 11, 17],
            channels: 33,
            stride: [1, 2],
            pads: [2, 1, 0, 3],
            ..base
        },
        ConvCase {
            name: "MPP 7x7 stride one oversized asymmetric padding",
            kernel: [7; 2],
            pads: [4, 1, 3, 6],
            ..base
        },
        ConvCase {
            name: "MPP 7x7 stride two batched channel tails",
            input: [2, 9, 11, 17],
            kernel: [7; 2],
            stride: [2; 2],
            pads: [1, 5, 4, 2],
            ..base
        },
    ]
}

fn assert_packed_execution(
    session: &GpuSession,
    matrix_precision: GpuMatrixPrecision,
    expected_convolutions: usize,
) {
    assert_eq!(session.input_precision(), GpuPrecision::F32);
    let operand = match matrix_precision {
        GpuMatrixPrecision::F32 => GpuPrecision::F32,
        GpuMatrixPrecision::F16 => GpuPrecision::MixedF16,
    };
    let mut convolutions = 0;
    for (index, recipe) in session.recipes.iter().enumerate() {
        let RecipeKind::Compute(shader) = recipe.kind else {
            continue;
        };
        let Some(kernel) = conv_mpp::Kernel::from_shader(shader) else {
            continue;
        };
        convolutions += 1;
        assert_eq!(conv_mpp::Kernel::from_words(&recipe.words), Some(kernel));
        assert!(
            session.recipes[..index].iter().any(|pack| {
                matches!(pack.kind, RecipeKind::Compute(ShaderKind::PackNhwc))
                    && pack.output == recipe.inputs[0]
            }),
            "the production schedule must pack the native activation before use"
        );
        let input = &session.descriptors[recipe.inputs[0].0];
        assert_eq!(input.precision, operand);
        assert_eq!(input.strides[1], 1, "physical channels must be contiguous");
        assert_eq!(input.strides[3], recipe.words[3] as usize);
        assert_eq!(
            session.descriptors[recipe.output.0].precision,
            GpuPrecision::F32
        );
        assert_eq!(recipe.words[36], recipe.words[4].div_ceil(32) * 32);
        assert!(session.dispatches.iter().any(|dispatch| {
            matches!(dispatch.command, Command::Compute { shader: selected, .. } if selected == shader)
        }), "the admitted native shader must reach actual dispatch");
    }
    assert_eq!(convolutions, expected_convolutions);
}

#[test]
#[ignore = "requires a physical Metal tensor GPU; four modest packed shapes in two explicit matrix modes"]
fn gpu_hardware_mpp_convolution_packed_geometry_and_precision() {
    let context = context();
    for case in cases() {
        let graph = convolution(case, GpuMatrixPrecision::F32);
        let input = values(case.input.iter().product(), 1);
        for matrix_precision in [GpuMatrixPrecision::F32, GpuMatrixPrecision::F16] {
            let effective_precision = if case.input[1] < 8 {
                GpuMatrixPrecision::F32
            } else {
                matrix_precision
            };
            // The FP16 oracle rounds only matrix operands. It keeps bias and
            // accumulation in FP32 and uses the existing strict F32 error gate.
            let oracle = convolution(case, effective_precision);
            let (_, expected) =
                reference(oracle, &case.input, &rounded(&input, effective_precision));
            let compiled = compile(&context, graph.clone(), matrix_precision);
            let mut session =
                GpuSession::new(compiled, &case.input, ExecutionLimits::default()).unwrap();
            assert_packed_execution(
                &session,
                matrix_precision,
                usize::from(case.input[1] >= 8 && matrix_precision == GpuMatrixPrecision::F16),
            );
            assert_close(case.name, session.run(&input).unwrap().data(), &expected);
            eprintln!(
                "MPP_PACKED_QUALIFICATION {}",
                serde_json::json!({
                    "case":case.name, "shape":case.input, "channels":case.channels,
                    "kernel":case.kernel, "stride":case.stride, "pads":case.pads,
                    "matrix_precision":matrix_precision, "checked_elements":expected.len(),
                    "oracle":"CPU F32 accumulation after requested operand rounding"
                })
            );
        }
    }
}

fn fused_graph(weight_precision: GpuMatrixPrecision) -> Arc<Model> {
    let shape = [1, 8, 7, 9];
    let channels = 33;
    model(
        vec![
            tensor(
                "w1",
                &[channels, 8, 3, 3],
                &rounded(&values(channels * 8 * 9, 7), weight_precision),
            ),
            tensor(
                "w2",
                &[channels, 8, 3, 3],
                &rounded(&values(channels * 8 * 9, 17), weight_precision),
            ),
            tensor("bias", &[channels], &values(channels, 11)),
            tensor("residual", &[1, channels, 1, 1], &values(channels, 13)),
            node(
                "Conv",
                &["x", "w1", "bias"],
                "a0",
                &[integers("kernel_shape", &[3; 2]), integers("pads", &[1; 4])],
            ),
            node("Add", &["a0", "residual"], "a1", &[]),
            node("Relu", &["a1"], "a", &[]),
            node(
                "Conv",
                &["x", "w2", "bias"],
                "b0",
                &[integers("kernel_shape", &[3; 2]), integers("pads", &[1; 4])],
            ),
            node("HardSigmoid", &["b0"], "b", &[]),
            node("Concat", &["a", "b"], "y", &[integer("axis", 1)]),
        ],
        &shape,
        &[1, channels * 2, 7, 9],
    )
}

#[test]
#[ignore = "requires a physical Metal tensor GPU; packed convolution residual/activation and Concat output offsets"]
fn gpu_hardware_mpp_convolution_fusions_and_offsets() {
    let context = context();
    let shape = [1, 8, 7, 9];
    let input = values(shape.iter().product(), 5);
    let graph = fused_graph(GpuMatrixPrecision::F32);
    for matrix_precision in [GpuMatrixPrecision::F32, GpuMatrixPrecision::F16] {
        let (_, expected) = reference(
            fused_graph(matrix_precision),
            &shape,
            &rounded(&input, matrix_precision),
        );
        let compiled = compile(&context, graph.clone(), matrix_precision);
        let mut session = GpuSession::new(compiled, &shape, ExecutionLimits::default()).unwrap();
        assert_packed_execution(
            &session,
            matrix_precision,
            if matrix_precision == GpuMatrixPrecision::F16 {
                2
            } else {
                0
            },
        );
        assert!(
            session
                .recipes
                .iter()
                .any(|recipe| recipe.words[60] != 0 && recipe.words[64] != 0)
        );

        assert_close(
            "MPP packed residual/activation and output offsets",
            session.run(&input).unwrap().data(),
            &expected,
        );
    }
}

#[test]
#[ignore = "requires physical Metal; shared Half HWIO packs and strict F32 arithmetic isolation"]
fn gpu_hardware_mpp_convolution_shared_packs() {
    let context = context();
    let case = cases()[0];
    let graph = convolution(case, GpuMatrixPrecision::F32);
    let input = values(case.input.iter().product(), 3);
    let (_, expected) = reference(
        convolution(case, GpuMatrixPrecision::F16),
        &case.input,
        &rounded(&input, GpuMatrixPrecision::F16),
    );
    let first = compile(&context, graph.clone(), GpuMatrixPrecision::F16);
    let second = compile(&context, graph.clone(), GpuMatrixPrecision::F16);
    assert!(Arc::ptr_eq(&first.weights, &second.weights));
    let mut a = GpuSession::new(first.clone(), &case.input, ExecutionLimits::default()).unwrap();
    assert_packed_execution(&a, GpuMatrixPrecision::F16, 1);
    let pack = first
        .weights
        .packed
        .lock()
        .unwrap()
        .values()
        .next()
        .unwrap()
        .clone();
    let packed_elements = case.channels.div_ceil(32) * 32 * case.input[1] * 9;
    assert_eq!(first.packed_weight_bytes(), packed_elements * 2);
    let mut b = GpuSession::new(second, &case.input, ExecutionLimits::default()).unwrap();
    assert_packed_execution(&b, GpuMatrixPrecision::F16, 1);
    {
        let cache = first.weights.packed.lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert!(Arc::ptr_eq(
            &pack.reservation,
            &cache.values().next().unwrap().reservation
        ));
    }
    assert_close(
        "MPP shared first request",
        a.run(&input).unwrap().data(),
        &expected,
    );
    assert_close(
        "MPP shared second request",
        b.run(&input).unwrap().data(),
        &expected,
    );
    let strict = compile(&context, graph.clone(), GpuMatrixPrecision::F32);
    let mut strict = GpuSession::new(strict, &case.input, ExecutionLimits::default()).unwrap();
    assert_packed_execution(&strict, GpuMatrixPrecision::F32, 0);
    let (_, original) = reference(graph, &case.input, &input);
    assert_close(
        "strict F32 original operands",
        strict.run(&input).unwrap().data(),
        &original,
    );
}
