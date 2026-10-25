use super::super::tests::{bind_program, integers, model, node, tensor};
use super::*;
use crate::runtime::CompileOptions;
use resident::Batch;

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: output length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label}[{index}]: nonfinite output"
        );
        let tolerance = 1e-4 + 1e-4 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}[{index}]: GPU={actual}, scalar={expected}, tolerance={tolerance}"
        );
    }
}

fn dyadic(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 17 + seed * 7) % 31) as f32 / 32.0 - 15.0 / 32.0)
        .collect()
}

struct Case {
    n: usize,
    inner: usize,
    channels: usize,
    input: Vec<f32>,
    weights: Vec<f32>,
    bias: Vec<f32>,
    residual: Vec<f32>,
}
impl Case {
    fn new(n: usize, inner: usize, channels: usize) -> Self {
        let mut case = Self {
            n,
            inner,
            channels,
            input: dyadic(n * inner, 3),
            weights: dyadic(channels * inner, 11),
            bias: dyadic(channels, 7),
            residual: dyadic(n * channels, 13),
        };
        for sample in case.input.chunks_exact_mut(inner) {
            sample[..2].fill(1.0);
        }
        case.weights[..inner].fill(0.0);
        case.weights[0] = 2048.0;
        case.weights[1] = -2048.0;
        case.bias[0] = 1.0;
        case
    }
    fn shape(&self) -> [usize; 4] {
        [self.n, self.inner, 1, 1]
    }
    fn graph(&self, fused: bool, bias: bool) -> Arc<Model> {
        let output = [self.n, self.channels, 1, 1];
        let mut parts = vec![
            tensor("w", &[self.channels, self.inner, 1, 1], &self.weights),
            tensor("bias", &[self.channels], &self.bias),
        ];
        parts.push(node(
            "Conv",
            if bias {
                &["x", "w", "bias"]
            } else {
                &["x", "w"]
            },
            if fused { "projected" } else { "y" },
            &[integers("kernel_shape", &[1, 1])],
        ));
        if fused {
            parts.extend([
                tensor("residual", &output, &self.residual),
                node("Add", &["projected", "residual"], "sum", &[]),
                node("Relu", &["sum"], "y", &[]),
            ]);
        }
        model(parts, &self.shape(), &output)
    }
    // Ascending-K scalar oracle, independent of the shader's lane partition
    // and reduction tree. Dyadic products/sums are exact in F32 for these
    // fixtures; verify that fact separately before using strict hardware gates.
    fn expected(&self, fused: bool, bias: bool) -> Vec<f32> {
        (0..self.n * self.channels)
            .map(|row| {
                let channel = row % self.channels;
                let mut value = if bias { self.bias[channel] } else { 0.0 };
                for k in 0..self.inner {
                    value += self.input[row / self.channels * self.inner + k]
                        * self.weights[channel * self.inner + k];
                }
                if fused {
                    value = (value + self.residual[row]).max(0.0);
                }
                half::f16::from_f32(value).to_f32()
            })
            .collect()
    }
    fn hard_swish_graph(&self) -> Arc<Model> {
        model(
            vec![
                tensor("w", &[self.channels, self.inner, 1, 1], &self.weights),
                node(
                    "Conv",
                    &["x", "w"],
                    "projected",
                    &[integers("kernel_shape", &[1, 1])],
                ),
                node(
                    "HardSigmoid",
                    &["projected"],
                    "gate",
                    &[
                        float_attribute("alpha", 0.16666667),
                        float_attribute("beta", 0.5),
                    ],
                ),
                node("Mul", &["projected", "gate"], "y", &[]),
            ],
            &self.shape(),
            &[self.n, self.channels, 1, 1],
        )
    }
    fn expected_hard_swish(&self) -> Vec<f32> {
        (0..self.n * self.channels)
            .map(|row| {
                let mut value = 0.0f32;
                for k in 0..self.inner {
                    value += self.input[row / self.channels * self.inner + k]
                        * self.weights[row % self.channels * self.inner + k];
                }
                // The existing physical Conv epilogue keeps all arithmetic
                // in Float and stores Half only after the complete HardSwish.
                half::f16::from_f32(value * (value * 0.16666667 + 0.5).clamp(0.0, 1.0)).to_f32()
            })
            .collect()
    }
}

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

fn classifier_case(n: usize) -> Case {
    let mut case = Case::new(n, 128, 1280);
    for sample in case.input.chunks_exact_mut(case.inner) {
        sample[0] = 1.0;
        sample[64] = 1.0;
        sample[127] = 1.0;
    }
    case.weights[..128].fill(0.0);
    case.weights[0] = 2048.0;
    case.weights[64] = 1.0;
    case.weights[127] = -2048.0;
    case
}

fn classifier_recipes(case: &Case) -> (Program, Vec<Recipe>, Vec<GpuTensorDescriptor>) {
    let program = Program::compile(case.hard_swish_graph()).unwrap();
    let mut plan = bind_program(&program, &case.shape(), ExecutionLimits::default()).unwrap();
    let mut descriptors = plan
        .shapes
        .iter()
        .map(|&shape| GpuTensorDescriptor::canonical(shape, GpuPrecision::MixedF16))
        .collect();
    let recipes = physical::lower(
        &program,
        &mut plan,
        program.output,
        &wgpu::Limits::default(),
        GpuPrecision::MixedF16,
        &mut descriptors,
        GpuMatrixPrecision::F16,
        &mut |_| Ok(false),
    )
    .unwrap();
    (program, recipes, descriptors)
}

#[test]
fn small_projection_admits_actual_classifier_conv16_and_retains_scoped_fallback() {
    let case = classifier_case(16);
    let (program, recipes, descriptors) = classifier_recipes(&case);
    assert_eq!(recipes.len(), 1);
    assert!(matches!(
        recipes[0].kind,
        RecipeKind::Compute(ShaderKind::Conv16)
    ));
    assert_eq!(recipes[0].nodes, [0, 1, 2]);
    assert_eq!(recipes[0].words[19], 0);
    assert_eq!(recipes[0].words[64..70], [1, 1280, 1, 0, 6, 0]);
    assert_eq!(recipes[0].words[32..36], [80, 1, recipes[0].x, 16]);
    for kind in [
        ShaderKind::Convolution,
        ShaderKind::Conv16,
        ShaderKind::Conv32,
        ShaderKind::MatMul16,
    ] {
        let mut original = recipes[0].clone();
        original.kind = RecipeKind::Compute(kind);
        original.words[60] = 7;
        let mut candidate = vec![original.clone()];
        let admitted = matches!(kind, ShaderKind::Convolution | ShaderKind::Conv16);
        assert_eq!(
            select(
                &program,
                &mut candidate,
                &descriptors,
                &wgpu::Limits::default(),
                GpuMatrixPrecision::F16
            )
            .unwrap(),
            usize::from(admitted)
        );
        assert!(matches!(candidate[0].kind, RecipeKind::Compute(actual)
            if actual == if admitted { ShaderKind::SmallProjection } else { kind }));
        let mut words = candidate[0].words;
        words[1] = original.words[1];
        assert_eq!(words, original.words);
        assert_eq!(candidate[0].inputs, original.inputs);
        assert_eq!(candidate[0].output, original.output);
        assert_eq!(candidate[0].nodes, original.nodes);
    }
    for (matrix, precision, too_small, prescale) in [
        (
            GpuMatrixPrecision::F32,
            GpuPrecision::MixedF16,
            false,
            false,
        ),
        (GpuMatrixPrecision::F16, GpuPrecision::F32, false, false),
        (GpuMatrixPrecision::F16, GpuPrecision::MixedF16, true, false),
        (GpuMatrixPrecision::F16, GpuPrecision::MixedF16, false, true),
    ] {
        let mut candidate = recipes.clone();
        candidate[0].words[52] = u32::from(prescale);
        let original = candidate[0].clone();
        let mut storage = descriptors.clone();
        for descriptor in &mut storage {
            descriptor.precision = precision;
        }
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: if too_small { 1 } else { 65535 },
            ..Default::default()
        };
        assert_eq!(
            select(&program, &mut candidate, &storage, &limits, matrix).unwrap(),
            0
        );
        assert!(matches!(
            candidate[0].kind,
            RecipeKind::Compute(ShaderKind::Conv16)
        ));
        assert_eq!((candidate[0].x, candidate[0].y), (original.x, original.y));
        assert_eq!(candidate[0].words, original.words);
        assert_eq!(candidate[0].inputs, original.inputs);
        assert_eq!(candidate[0].output, original.output);
        assert_eq!(candidate[0].nodes, original.nodes);
    }
    let expected = case.expected_hard_swish();
    for sample in expected.as_chunks::<1280>().0 {
        assert_eq!(sample[0], half::f16::from_f32(2.0 / 3.0).to_f32());
    }
}

fn fixture_recipes(case: &Case) -> (Program, Vec<Recipe>, Vec<GpuTensorDescriptor>) {
    let program = Program::compile(case.graph(false, true)).unwrap();
    let plan = bind_program(&program, &case.shape(), ExecutionLimits::default()).unwrap();
    let recipes = operators::recipes(&program, &plan, &wgpu::Limits::default()).unwrap();
    let descriptors = plan
        .shapes
        .iter()
        .map(|&s| GpuTensorDescriptor::canonical(s, GpuPrecision::MixedF16))
        .collect();
    (program, recipes, descriptors)
}

#[test]
fn small_projection_selection_is_explicit_canonical_half_and_preserves_packets() {
    for inner in [24, 95, 96, 129, 768] {
        let case = Case::new(2, inner, 17);
        let (program, recipes, descriptors) = fixture_recipes(&case);
        for precision in [GpuPrecision::F32, GpuPrecision::MixedF16] {
            for matrix in [GpuMatrixPrecision::F32, GpuMatrixPrecision::F16] {
                let mut candidate = recipes.clone();
                let mut storage = descriptors.clone();
                for d in &mut storage {
                    d.precision = precision;
                }
                let expected = usize::from(
                    inner >= 96
                        && matrix == GpuMatrixPrecision::F16
                        && precision == GpuPrecision::MixedF16,
                );
                assert_eq!(
                    select(
                        &program,
                        &mut candidate,
                        &storage,
                        &wgpu::Limits::default(),
                        matrix
                    )
                    .unwrap(),
                    expected
                );
                if expected != 0 {
                    assert!(matches!(
                        candidate[0].kind,
                        RecipeKind::Compute(ShaderKind::SmallProjection)
                    ));
                    let mut actual_words = candidate[0].words;
                    actual_words[1] = recipes[0].words[1];
                    assert_eq!(actual_words, recipes[0].words);
                    assert_eq!(candidate[0].inputs, recipes[0].inputs);
                    assert_eq!(candidate[0].output, recipes[0].output);
                    assert_eq!(candidate[0].nodes, recipes[0].nodes);
                }
            }
        }
    }
    let case = Case::new(2, 129, 17);
    let (program, recipes, descriptors) = fixture_recipes(&case);
    for word in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 52] {
        let mut candidate = recipes.clone();
        candidate[0].words[word] = candidate[0].words[word].wrapping_add(1);
        assert_eq!(
            select(
                &program,
                &mut candidate,
                &descriptors,
                &wgpu::Limits::default(),
                GpuMatrixPrecision::F16
            )
            .unwrap(),
            0,
            "badword{word}"
        );
    }
    let mut candidate = recipes.clone();
    let limits = wgpu::Limits {
        max_compute_workgroups_per_dimension: 1,
        ..Default::default()
    };
    assert_eq!(
        select(
            &program,
            &mut candidate,
            &descriptors,
            &limits,
            GpuMatrixPrecision::F16
        )
        .unwrap(),
        0
    );
    assert_eq!(candidate[0].words, recipes[0].words);
    assert!(matches!(
        candidate[0].kind,
        RecipeKind::Compute(ShaderKind::Convolution)
    ));
    let mut noncanonical = descriptors;
    noncanonical[recipes[0].inputs[0].0].layout = GpuLayout::PaddedNhwc { channels: 129 };
    assert_eq!(
        select(
            &program,
            &mut candidate,
            &noncanonical,
            &wgpu::Limits::default(),
            GpuMatrixPrecision::F16
        )
        .unwrap(),
        0
    );
}

#[test]
fn small_projection_oracle_preserves_float_accumulation_and_bias_initialization() {
    for (n, inner, channels) in [(2, 129, 17), (1, 768, 192), (2, 1536, 384)] {
        let case = Case::new(n, inner, channels);
        for row in 0..n * channels {
            let co = row % channels;
            let mut partial = [0.0f32; 64];
            partial[0] = case.bias[co];
            let mut serial = case.bias[co];
            for k in 0..inner {
                let product = case.input[row / channels * inner + k] * case.weights[co * inner + k];
                serial += product;
                partial[k % 64] += product;
            }
            for stride in [32, 16, 8, 4, 2, 1] {
                for lane in 0..stride {
                    partial[lane] += partial[lane + stride];
                }
            }
            assert_eq!(partial[0].to_bits(), serial.to_bits());
        }
    }
    // A nonzero initial bias followed by cancelling large terms must retain
    // Float precision; per-step Half accumulation loses the unit contribution.
    let mut float = 1.0f32;
    let mut narrowed = half::f16::from_f32(float).to_f32();
    for term in [2048.0f32, -2048.0] {
        float += term;
        narrowed = half::f16::from_f32(narrowed + term).to_f32();
    }
    assert_eq!(float, 1.0);
    assert_eq!(narrowed, 0.0);
}

fn hardware() {
    let context = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
    for (n, inner, channels) in [(2, 129, 17), (1, 768, 192), (2, 1536, 384)] {
        let case = Case::new(n, inner, channels);
        let compiled = Arc::new(
            GpuCompiledModel::compile_with_options(
                case.graph(true, true),
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
        let mut session =
            GpuSession::new_resident(compiled, &case.shape(), ExecutionLimits::default(), None)
                .unwrap();
        let recipe = session
            .recipes
            .iter()
            .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::SmallProjection)))
            .unwrap()
            .clone();
        assert_eq!(recipe.words[19], 1);
        assert_eq!(recipe.words[64], 2);
        assert_eq!(recipe.words[68..74], [1, 2, 0, 3, 0, 0]);
        assert_eq!(session.input_precision(), GpuPrecision::MixedF16);
        context
            .upload_resident(
                &session.resident_input().buffer,
                &GpuPrecision::MixedF16.encode(&case.input),
            )
            .unwrap();
        let mut batch = Batch::new(context.clone()).unwrap();
        let output = batch.graph(&mut session, "small-projection").unwrap();
        let (bytes, statistics) = batch
            .read(
                &output.buffer,
                GpuPrecision::MixedF16.bytes(n * channels),
                false,
            )
            .unwrap();
        assert_eq!(statistics.native_graph_dispatches, 1);
        let mut actual = vec![0.0; n * channels];
        GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
        assert_close(
            "actual small projection",
            &actual,
            &case.expected(true, true),
        );

        // Execute the same production WGSL with poisoned guards and p60=7.
        let kind = ShaderKind::SmallProjection;
        let mut source = format!("{}\n{}", kind.common_source(), kind.source())
            .replace("$ENABLE", "enable f16;")
            .replace("$HALF_OUTPUT", "true");
        for name in ["$A", "$B", "$C", "$D", "$E", "$O"] {
            source = source.replace(name, "f16");
        }
        let mut batch = Batch::new(context.clone()).unwrap();
        let dummy = vec![0.0; 2];
        let mut operands = Vec::new();
        for data in [
            &case.input,
            &case.weights,
            &case.bias,
            &case.residual,
            &dummy,
        ] {
            let bytes = GpuPrecision::MixedF16.encode(data);
            let buffer = batch
                .storage("small projection operand", bytes.len())
                .unwrap();
            context.upload_resident(&buffer, &bytes).unwrap();
            operands.push(buffer);
        }
        let prefix = 7;
        let suffix = 11;
        let sentinel = -777.0;
        let values = vec![sentinel; prefix + n * channels + suffix];
        let output = batch
            .storage(
                "small projection guarded output",
                GpuPrecision::MixedF16.bytes(values.len()),
            )
            .unwrap();
        context
            .upload_resident(&output, &GpuPrecision::MixedF16.encode(&values))
            .unwrap();
        let mut words = recipe.words;
        words[60] = prefix as u32;
        batch
            .dispatch(
                "small_projection_production_wgsl",
                &source,
                &[
                    (&operands[0], true),
                    (&operands[1], true),
                    (&operands[2], true),
                    (&operands[3], true),
                    (&operands[4], true),
                    (&output, false),
                ],
                &words,
                [recipe.x, recipe.y, 1],
            )
            .unwrap();
        let (bytes, statistics) = batch.read(&output, output.bytes(), false).unwrap();
        assert_eq!(statistics.native_resident_dispatches, 1);
        let mut actual = values;
        GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
        assert!(actual[..prefix].iter().all(|&v| v == sentinel));
        assert!(
            actual[prefix + n * channels..]
                .iter()
                .all(|&v| v == sentinel)
        );
        assert_close(
            "guarded small projection",
            &actual[prefix..prefix + n * channels],
            &case.expected(true, true),
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires physical Metal GPU; native translated small projection, fused epilogue and offset guards"]
fn gpu_hardware_small_projection_native_metal() {
    hardware();
}

fn classifier_hardware() {
    let context = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
    let case = classifier_case(16);
    let expected = case.expected_hard_swish();
    let compiled = Arc::new(
        GpuCompiledModel::compile_with_options(
            case.hard_swish_graph(),
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
    let mut session =
        GpuSession::new_resident(compiled, &case.shape(), ExecutionLimits::default(), None)
            .unwrap();
    assert_eq!(session.recipes.len(), 1);
    let recipe = session.recipes[0].clone();
    assert!(matches!(
        recipe.kind,
        RecipeKind::Compute(ShaderKind::SmallProjection)
    ));
    assert_eq!(recipe.words[19], 0);
    assert_eq!(recipe.words[64..70], [1, 1280, 1, 0, 6, 0]);
    assert_eq!(recipe.nodes, [0, 1, 2]);
    context
        .upload_resident(
            &session.resident_input().buffer,
            &GpuPrecision::MixedF16.encode(&case.input),
        )
        .unwrap();
    let mut batch = Batch::new(context.clone()).unwrap();
    let output = batch
        .graph(&mut session, "classifier-pooled-projection")
        .unwrap();
    let (bytes, statistics) = batch
        .read(
            &output.buffer,
            GpuPrecision::MixedF16.bytes(expected.len()),
            false,
        )
        .unwrap();
    assert_eq!(statistics.native_graph_dispatches, 1);
    let mut actual = vec![0.0; expected.len()];
    GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
    assert_close("classifier pooled projection", &actual, &expected);
    for row in 0..case.n {
        assert_eq!(actual[row * case.channels], expected[row * case.channels]);
    }

    // Run the same production source with an offset, stale output guards,
    // unused NaN C/D/E operands, and three extra workgroups. This checks the
    // all-Half common ABI independently of the graph's slot allocation.
    let kind = ShaderKind::SmallProjection;
    let mut source = format!("{}\n{}", kind.common_source(), kind.source())
        .replace("$ENABLE", "enable f16;")
        .replace("$HALF_OUTPUT", "true");
    for token in ["$A", "$B", "$C", "$D", "$E", "$O"] {
        source = source.replace(token, "f16");
    }
    let mut batch = Batch::new(context.clone()).unwrap();
    let mut operands = Vec::new();
    let poison = vec![f32::NAN; 2];
    for values in [&case.input, &case.weights, &poison] {
        let bytes = GpuPrecision::MixedF16.encode(values);
        let operand = batch
            .storage("classifier projection operand", bytes.len())
            .unwrap();
        context.upload_resident(&operand, &bytes).unwrap();
        operands.push(operand);
    }
    let (prefix, suffix, sentinel) = (7, 11, -777.0);
    let values = vec![sentinel; prefix + expected.len() + suffix];
    let output = batch
        .storage(
            "classifier projection guarded output",
            GpuPrecision::MixedF16.bytes(values.len()),
        )
        .unwrap();
    context
        .upload_resident(&output, &GpuPrecision::MixedF16.encode(&values))
        .unwrap();
    let mut words = recipe.words;
    words[60] = prefix as u32;
    batch
        .dispatch(
            "classifier_pooled_projection_production_wgsl_half",
            &source,
            &[
                (&operands[0], true),
                (&operands[1], true),
                (&operands[2], true),
                (&operands[2], true),
                (&operands[2], true),
                (&output, false),
            ],
            &words,
            [recipe.x + 3, recipe.y, 1],
        )
        .unwrap();
    let (bytes, statistics) = batch.read(&output, output.bytes(), false).unwrap();
    assert_eq!(statistics.native_resident_dispatches, 1);
    let mut actual = values;
    GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
    assert!(actual[..prefix].iter().all(|&value| value == sentinel));
    assert!(
        actual[prefix + expected.len()..]
            .iter()
            .all(|&value| value == sentinel)
    );
    assert_close(
        "guarded classifier projection",
        &actual[prefix..prefix + expected.len()],
        &expected,
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires physical Metal GPU; native batched classifier Conv16 admission and guarded production WGSL"]
fn gpu_hardware_small_projection_classifier_conv16_native() {
    classifier_hardware();
}
