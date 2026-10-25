//! Physical FP16 matrix operands with canonical F32 external tensor storage.
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
                    precision: GpuPrecision::F32,
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

fn assert_fast(session: &GpuSession) {
    assert!(
        session
            .recipes
            .iter()
            .any(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::PointwiseF16)))
    );
    assert!(session.dispatches.iter().any(|dispatch| matches!(
        dispatch.command,
        Command::Compute {
            shader: ShaderKind::PointwiseF16,
            ..
        }
    )));
    assert_eq!(session.input_precision(), GpuPrecision::F32);
    assert_eq!(
        session.descriptors[session.selected_output.0].precision,
        GpuPrecision::F32
    );
    for recipe in &session.recipes {
        if matches!(recipe.kind, RecipeKind::Compute(ShaderKind::PackNhwc)) {
            assert_eq!(
                session.descriptors[recipe.output.0].precision,
                GpuPrecision::MixedF16
            );
        }
    }
}

fn dynamic_pointwise() -> Arc<Model> {
    let information = |name: &str, channels: usize| {
        let dimensions = [
            bytes(1, &[]),
            bytes(1, &number(1, channels as u64)),
            bytes(1, &[]),
            bytes(1, &[]),
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
        tensor("w", &[73, 17, 1, 1], &dyadic(73 * 17, 7)),
        node(
            "Conv",
            &["x", "w"],
            "y",
            &[integers("kernel_shape", &[1; 2])],
        ),
        bytes(11, &information("x", 17)),
        bytes(12, &information("y", 73)),
    ]
    .concat();
    Arc::new(
        Model::load_bytes_with_options(
            &[number(1, 8), bytes(8, &number(2, 13)), bytes(7, &graph)].concat(),
            &LoadOptions {
                optimization: OptimizationPolicy::PreserveArithmetic,
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

#[test]
#[ignore = "requires physical Metal tensor GPU; FP16 pointwise production path and shape reuse"]
fn gpu_hardware_pointwise_f16_tails_shape_reuse_and_strict_weight_owner() {
    let graph = dynamic_pointwise();
    let fast = compile(graph.clone());
    let strict = Arc::new(
        GpuCompiledModel::compile_with_options(
            graph.clone(),
            fast.context.clone(),
            CompileOptions::default(),
        )
        .unwrap(),
    );
    assert_eq!(fast.matrix_precision, GpuMatrixPrecision::F16);
    assert_eq!(strict.matrix_precision, GpuMatrixPrecision::F32);
    assert!(Arc::ptr_eq(&fast.weights, &strict.weights));

    assert!(Arc::ptr_eq(&fast.context, &strict.context));
    let first_shape = [2, 17, 1, 67];
    let mut selected =
        GpuSession::new(fast.clone(), &first_shape, ExecutionLimits::default()).unwrap();
    let mut strict_session =
        GpuSession::new(strict, &first_shape, ExecutionLimits::default()).unwrap();
    assert!(
        !strict_session
            .recipes
            .iter()
            .any(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::PointwiseF16)))
    );
    for (seed, shape) in [[2, 17, 1, 67], [1, 17, 3, 23], [2, 17, 1, 67]]
        .into_iter()
        .enumerate()
    {
        selected.reshape(&shape).unwrap();
        assert_fast(&selected);
        let input = dyadic(shape.iter().product(), seed + 1);
        // These inputs and constants are exact FP16 dyadics. Their short dot
        // products are exact FP32 too, so the existing CPU comparison gate is
        // applicable without changing its tolerance for fast mode.
        let (_, expected) = reference(graph.clone(), &shape, &input);
        assert_close(
            "FP16 pointwise shape/tails",
            selected.run(&input).unwrap().data(),
            &expected,
        );
        if seed == 0 {
            assert_close(
                "strict shared weights",
                strict_session.run(&input).unwrap().data(),
                &expected,
            );
        }
    }
    assert!(fast.packed_weight_bytes() > 0);
}

#[test]
#[ignore = "requires physical Metal tensor GPU; nonexact FP16 matrix operand rounding with FP32 bias"]
fn gpu_hardware_pointwise_f16_nondyadic_operands_and_f32_bias() {
    let shape = [2, 17, 1, 67];
    let channels = 73;
    let weights = values(channels * 17, 7);
    let bias = values(channels, 11);
    let input = values(shape.iter().product(), 3);
    // This oracle uses the public half conversion independently of the GPU
    // pack/weight encoder. Bias retains its original non-half-exact F32 values.
    let rounded_weights: Vec<_> = weights
        .iter()
        .map(|&value| half::f16::from_f32(value).to_f32())
        .collect();
    let rounded_input: Vec<_> = input
        .iter()
        .map(|&value| half::f16::from_f32(value).to_f32())
        .collect();
    for (original, rounded) in [(&weights, &rounded_weights), (&input, &rounded_input)] {
        assert!(
            original
                .iter()
                .zip(rounded)
                .any(|(&a, &b)| a.to_bits() != b.to_bits()),
            "the fixture must exercise operand rounding"
        );
    }
    assert!(
        bias.iter()
            .any(|&value| value.to_bits() != half::f16::from_f32(value).to_f32().to_bits())
    );
    let graph_with_weights = |weight_values: &[f32]| {
        model(
            vec![
                tensor("w", &[channels, 17, 1, 1], weight_values),
                tensor("bias", &[channels], &bias),
                node(
                    "Conv",
                    &["x", "w", "bias"],
                    "y",
                    &[integers("kernel_shape", &[1; 2])],
                ),
            ],
            &shape,
            &[2, channels, 1, 67],
        )
    };
    let (_, expected) = reference(graph_with_weights(&rounded_weights), &shape, &rounded_input);
    let mut selected = GpuSession::new(
        compile(graph_with_weights(&weights)),
        &shape,
        ExecutionLimits::default(),
    )
    .unwrap();
    assert_fast(&selected);
    assert_close(
        "FP16 pointwise rounded operands/F32 bias",
        selected.run(&input).unwrap().data(),
        &expected,
    );
}

#[test]
#[ignore = "requires physical Metal tensor GPU; finite operand half overflow and session recovery"]
fn gpu_hardware_pointwise_f16_operand_overflow_and_recovery() {
    let shape = [2, 17, 1, 67];
    let channels = 73;
    let graph = model(
        vec![
            tensor("w", &[channels, 17, 1, 1], &vec![1.0; channels * 17]),
            node(
                "Conv",
                &["x", "w"],
                "y",
                &[integers("kernel_shape", &[1; 2])],
            ),
        ],
        &shape,
        &[2, channels, 1, 67],
    );
    let mut selected =
        GpuSession::new(compile(graph.clone()), &shape, ExecutionLimits::default()).unwrap();
    assert_fast(&selected);
    let overflowing = 65_536.0_f32;
    assert!(overflowing.is_finite());
    assert!(half::f16::from_f32(overflowing).is_infinite());
    let error = selected
        .run(&vec![overflowing; shape.iter().product()])
        .expect_err("half operand overflow must fail the existing finite-output check");
    assert_eq!(error.kind(), ErrorKind::Execution);
    assert!(error.to_string().contains("nonfinite output F32"));
    let input = vec![0.25; shape.iter().product()];
    let (_, expected) = reference(graph, &shape, &input);
    assert_close(
        "FP16 pointwise recovery after operand overflow",
        selected.run(&input).unwrap().data(),
        &expected,
    );
}

#[test]
#[ignore = "requires physical Metal tensor GPU; FP32 prescale folded before FP16 operand conversion"]
fn gpu_hardware_pointwise_f16_prescale_fused_residual_and_bias() {
    // The existing minimum group policy must first expose the prescaled
    // pointwise recipe, so use 156 logical 16x16 groups before fast lowering.
    let shape = [2, 32, 3, 137];
    let channels = 48;
    let graph = model(
        vec![
            tensor("w", &[channels, 32, 1, 1], &dyadic(channels * 32, 7)),
            tensor("bias", &[channels], &dyadic(channels, 11)),
            tensor("scale", &[2, 32, 1, 1], &dyadic(2 * 32, 13)),
            tensor("residual", &[1, channels, 1, 1], &dyadic(channels, 17)),
            node("Mul", &["x", "scale"], "scaled", &[]),
            node(
                "Conv",
                &["scaled", "w", "bias"],
                "conv",
                &[integers("kernel_shape", &[1; 2])],
            ),
            node("Add", &["conv", "residual"], "sum", &[]),
            node("Relu", &["sum"], "y", &[]),
        ],
        &shape,
        &[2, channels, 3, 137],
    );
    let input = dyadic(shape.iter().product(), 3);
    let (_, expected) = reference(graph.clone(), &shape, &input);
    let mut selected = GpuSession::new(compile(graph), &shape, ExecutionLimits::default()).unwrap();
    assert_fast(&selected);
    assert_eq!(selected.recipes.len(), 2);
    let converter = selected
        .recipes
        .iter()
        .find(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::PackNhwc)))
        .unwrap();
    assert_eq!(converter.words[6], 1);
    assert_ne!(converter.inputs[0], converter.inputs[1]);
    let matrix = selected
        .recipes
        .iter()
        .find(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::PointwiseF16)))
        .unwrap();
    assert_eq!(matrix.words[52], 0);
    assert_eq!(matrix.words[64], 2);
    assert_eq!(
        matrix.inputs.len(),
        4,
        "prescale must preserve residual binding3"
    );
    assert_close(
        "FP16 pointwise prescale/bias/residual",
        selected.run(&input).unwrap().data(),
        &expected,
    );
}

#[test]
#[ignore = "requires physical Metal tensor GPU; FP16 pointwise shared pack and native output offsets"]
fn gpu_hardware_pointwise_f16_fused_concat_and_dynamic_uniform_offsets() {
    let shape = [1, 17, 1, 67];
    let channels = 73;
    let graph = model(
        vec![
            tensor("w1", &[channels, 17, 1, 1], &dyadic(channels * 17, 7)),
            tensor("w2", &[channels, 17, 1, 1], &dyadic(channels * 17, 17)),
            tensor("bias", &[channels], &dyadic(channels, 11)),
            tensor("residual", &[1, channels, 1, 1], &dyadic(channels, 13)),
            node(
                "Conv",
                &["x", "w1", "bias"],
                "a0",
                &[integers("kernel_shape", &[1; 2])],
            ),
            node("Add", &["a0", "residual"], "a1", &[]),
            node("Relu", &["a1"], "a", &[]),
            node(
                "Conv",
                &["x", "w2", "bias"],
                "b0",
                &[integers("kernel_shape", &[1; 2])],
            ),
            node("Relu", &["b0"], "b", &[]),
            node("Concat", &["a", "b"], "y", &[integer("axis", 1)]),
        ],
        &shape,
        &[1, channels * 2, 1, 67],
    );
    let input = dyadic(shape.iter().product(), 5);
    let (_, expected) = reference(graph.clone(), &shape, &input);
    let mut selected = GpuSession::new(compile(graph), &shape, ExecutionLimits::default()).unwrap();
    assert_fast(&selected);
    assert_eq!(
        selected
            .recipes
            .iter()
            .filter(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::PackNhwc)))
            .count(),
        1
    );
    assert_eq!(
        selected
            .recipes
            .iter()
            .filter(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::PointwiseF16)))
            .count(),
        2
    );
    assert!(
        selected
            .recipes
            .iter()
            .any(|recipe| recipe.words[60] == (channels * 67) as u32)
    );

    assert_close(
        "FP16 pointwise fused concat offsets",
        selected.run(&input).unwrap().data(),
        &expected,
    );
}
