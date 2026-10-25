//! Actual ONNX lowering plus independent Float-dot/Half-gate contract checks.
use super::*;
use resident::Batch;

const BATCH: usize = 2;
const CHANNELS: usize = 17;

#[derive(Clone, Copy)]
enum Witness {
    HardBoundary,
    SigmoidBoundary,
    DenseCancellation,
}

struct Case {
    inner: usize,
    plane: usize,
    hard: bool,
    scalar_bias: bool,
    input: Vec<f32>,
    weights: Vec<f32>,
    bias: Vec<f32>,
    feature: Vec<f32>,
}

fn half(value: f32) -> f32 {
    half::f16::from_f32(value).to_f32()
}

impl Case {
    fn new(inner: usize, witness: Witness) -> Self {
        let hard = matches!(witness, Witness::HardBoundary);
        let scalar_bias = matches!(witness, Witness::DenseCancellation);
        let plane = match witness {
            Witness::HardBoundary => 67,
            Witness::SigmoidBoundary => 128,
            Witness::DenseCancellation => 65,
        };
        let mut input = (0..BATCH * inner)
            .map(|i| ((i * 37 + 19) % 63) as f32 / 64.0 - 31.0 / 64.0)
            .collect::<Vec<_>>();
        let mut weights = (0..CHANNELS * inner)
            .map(|i| ((i * 13 + 7) % 63) as f32 / 64.0 - 31.0 / 64.0)
            .collect::<Vec<_>>();
        let mut bias = (0..if scalar_bias { 1 } else { CHANNELS })
            .map(|i| ((i * 11 + 17) % 31) as f32 / 256.0 - 15.0 / 256.0)
            .collect::<Vec<_>>();
        if scalar_bias {
            assert_eq!(inner, 129);
            input.fill(1.0);
            bias.fill(0.0);
            // Every coefficient is nonzero. The exact Float lane-zero sum is
            // 2048+1-2048=1; Half accumulation loses its middle term.
            for row in weights.chunks_exact_mut(inner) {
                row.fill(1.0 / 32.0);
                row[0] = 2048.0;
                row[64] = 1.0;
                row[128] = -2048.0;
                row[65..128].fill(-1.0 / 32.0);
            }
        } else {
            for batch in 0..BATCH {
                input[batch * inner] = 1.0 / 1024.0;
                input[(batch + 1) * inner - 1] = 1.0 / 1024.0;
            }
            for channel in 0..3 {
                weights[channel * inner..(channel + 1) * inner].fill(0.0);
                weights[(channel + 1) * inner - 1] = if hard {
                    1.25
                } else {
                    // Keep exp-based sigmoid cases away from Half midpoints;
                    // HardSigmoid below owns the exact rounding witnesses.
                    [128.0, -128.0, 64.0][channel]
                };
                bias[channel] = 0.0;
            }
            // Rows 1 and 2 lie just above row 0's Half-gate midpoint.
            // Row 1 rejects rounding after bias; row 2 rejects rounding the
            // dot before bias. Only the gate tensor itself is rounded.
            bias[1] = 1.0 / 2_097_152.0;
            weights[2 * inner] = 1.0 / 2048.0;
        }
        let mut feature = (0..BATCH * CHANNELS * plane)
            .map(|i| ((i * 17 + 31) % 255) as f32 / 32.0 - 127.0 / 32.0)
            .collect::<Vec<_>>();
        for row in 0..BATCH * CHANNELS {
            feature[row * plane] = 1.0;
            feature[row * plane + 1] = 3.0;
        }
        Self {
            inner,
            plane,
            hard,
            scalar_bias,
            input,
            weights,
            bias,
            feature,
        }
    }

    fn shape(&self) -> [usize; 4] {
        [BATCH, self.inner, 1, 1]
    }

    fn graph(&self, escape: bool) -> Arc<Model> {
        let mut parts = vec![
            tensor("weights", &[CHANNELS, self.inner, 1, 1], &self.weights),
            tensor(
                "bias",
                if self.scalar_bias {
                    &[1, 1, 1]
                } else {
                    &[CHANNELS, 1, 1]
                },
                &self.bias,
            ),
            tensor("feature", &[BATCH, CHANNELS, 1, self.plane], &self.feature),
            node(
                "Conv",
                &["x", "weights"],
                "projection",
                &[integers("kernel_shape", &[1, 1])],
            ),
            node("Add", &["projection", "bias"], "biased", &[]),
            node(
                if self.hard { "HardSigmoid" } else { "Sigmoid" },
                &["biased"],
                "gate",
                &[],
            ),
            node("Mul", &["feature", "gate"], "applied", &[]),
        ];
        if escape {
            parts.extend([
                tensor("zero", &[1], &[0.0]),
                node("Mul", &["gate", "zero"], "escaped", &[]),
                node("Add", &["applied", "escaped"], "y", &[]),
            ]);
        } else {
            parts.push(node("Identity", &["applied"], "y", &[]));
        }
        model(parts, &self.shape(), &[BATCH, CHANNELS, 1, self.plane])
    }

    fn activate(&self, value: f32) -> f32 {
        if self.hard {
            (value * 0.2 + 0.5).clamp(0.0, 1.0)
        } else if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exponent = value.exp();
            exponent / (1.0 + exponent)
        }
    }

    fn dot(&self, batch: usize, channel: usize) -> f32 {
        let mut dot = 0.0;
        for k in 0..self.inner {
            dot += half(self.input[batch * self.inner + k])
                * half(self.weights[channel * self.inner + k]);
        }
        dot
    }

    fn expected(&self) -> Vec<f32> {
        let mut result = vec![0.0; self.feature.len()];
        for row in 0..BATCH * CHANNELS {
            let channel = row % CHANNELS;
            let value = self.dot(row / CHANNELS, channel)
                + half(self.bias[if self.scalar_bias { 0 } else { channel }]);
            let gate = half(self.activate(value));
            for i in 0..self.plane {
                let index = row * self.plane + i;
                result[index] = half(half(self.feature[index]) * gate);
            }
        }
        result
    }
}

fn lower_case(
    case: &Case,
    precision: GpuPrecision,
    matrix: GpuMatrixPrecision,
    escape: bool,
) -> Vec<Recipe> {
    let program = Program::compile(case.graph(escape)).unwrap();
    let mut plan = bind_program(&program, &case.shape(), ExecutionLimits::default()).unwrap();
    let mut descriptors = plan
        .shapes
        .iter()
        .map(|&s| GpuTensorDescriptor::canonical(s, precision))
        .collect();
    physical::lower(
        &program,
        &mut plan,
        program.output,
        &crate::runtime::gpu::GpuLimits::default(),
        precision,
        &mut descriptors,
        matrix,
        &mut |_| Ok(false),
    )
    .unwrap()
}

#[test]
fn excitation_parallel_admission_requires_explicit_fast_policy_and_large_private_projection() {
    for inner in [24, 95, 96, 129] {
        let case = Case::new(inner, Witness::HardBoundary);
        for precision in [GpuPrecision::F32, GpuPrecision::MixedF16] {
            for matrix in [GpuMatrixPrecision::F32, GpuMatrixPrecision::F16] {
                let recipes = lower_case(&case, precision, matrix, false);
                let parallel = recipes
                    .iter()
                    .filter(|r| {
                        matches!(r.kind, RecipeKind::Compute(ShaderKind::SqueezeExcite))
                            && r.words[13] != 0
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    parallel.len(),
                    usize::from(matrix == GpuMatrixPrecision::F16 && inner >= 96)
                );
                if matrix == GpuMatrixPrecision::F16 {
                    let recipe = recipes
                        .iter()
                        .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::SqueezeExcite)))
                        .expect("small and large private projections retain the SE fusion");
                    assert_eq!(recipe.words[12], 1);
                    assert_eq!(recipe.words[13], u32::from(inner >= 96));
                }
                for recipe in parallel {
                    assert!(matches!(
                        recipe.kind,
                        RecipeKind::Compute(ShaderKind::SqueezeExcite)
                    ));
                    assert_eq!(recipe.words[12..14], [1, 1]);
                    assert_eq!(recipe.nodes, [0, 1, 2, 3]);
                }
            }
        }
    }
    let case = Case::new(129, Witness::HardBoundary);
    let program = Program::compile(case.graph(false)).unwrap();
    let plan = bind_program(&program, &case.shape(), ExecutionLimits::default()).unwrap();
    for selected in program.instructions[..3].iter().map(|op| op.output) {
        assert!(
            fusions::replacement(
                &program,
                &plan,
                0,
                selected,
                &crate::runtime::gpu::GpuLimits::default(),
                GpuMatrixPrecision::F16
            )
            .unwrap()
            .is_none()
        );
    }
    let escaped = lower_case(&case, GpuPrecision::MixedF16, GpuMatrixPrecision::F16, true);
    assert!(escaped.iter().all(|r| !matches!(
        r.kind,
        RecipeKind::Compute(ShaderKind::SqueezeExcite)
    ) || r.words[13] == 0));
    assert!(
        escaped
            .iter()
            .any(|r| r.nodes == [0, 1, 2] && r.words[64] == 2)
    );
    assert!(escaped.iter().any(|r| r.nodes == [3]));
}

#[test]
fn excitation_parallel_oracle_rejects_half_dot_and_missing_or_premature_gate_rounding() {
    for witness in [
        Witness::HardBoundary,
        Witness::SigmoidBoundary,
        Witness::DenseCancellation,
    ] {
        let case = Case::new(129, witness);
        for batch in 0..BATCH {
            for channel in 0..CHANNELS {
                let mut partial = [0.0f32; 64];
                for k in 0..case.inner {
                    partial[k % 64] += half(case.input[batch * case.inner + k])
                        * half(case.weights[channel * case.inner + k]);
                }
                for stride in [32, 16, 8, 4, 2, 1] {
                    for lane in 0..stride {
                        partial[lane] += partial[lane + stride];
                    }
                }
                assert_eq!(
                    partial[0],
                    case.dot(batch, channel),
                    "independent ascending dot and lane tree must be exact on these dyadics"
                );
            }
        }
        let expected = case.expected();
        if case.hard {
            assert_eq!(expected[1], 1.5);
            assert_eq!(expected[case.plane + 1], 1.501_953_1);
            assert_eq!(expected[case.plane * 2 + 1], 1.501_953_1);
            let dot = case.dot(0, 0);
            assert_eq!(dot, 5.0 / 4096.0);
            assert_eq!(half(3.0 * case.activate(dot)), 1.500_976_6);
            assert!(
                (half(3.0 * case.activate(dot)) - expected[1]).abs()
                    > 1e-4 + 1e-4 * expected[1].abs()
            );
            for channel in [1, 2] {
                let early_value = half(half(case.dot(0, channel)) + half(case.bias[channel]));
                assert_eq!(half(3.0 * half(case.activate(early_value))), 1.5);
            }
        } else if case.scalar_bias {
            assert_eq!(case.dot(0, 0), 1.0);
            let mut bad_half_sum = 0.0;
            for k in 0..case.inner {
                bad_half_sum = half(bad_half_sum + case.input[k] * case.weights[k]);
            }
            assert_eq!(bad_half_sum, 0.0);
            assert_ne!(half(3.0 * half(case.activate(bad_half_sum))), expected[1]);
        }
    }
}

fn context() -> Arc<GpuContext> {
    GpuContext::new_with_options(GpuContextOptions::default()).unwrap()
}

fn compare(case: &Case, actual: &[f32]) {
    let expected = case.expected();
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-4 + 1e-4 * e.abs(),
            "SE element{i}: {a} != {e}"
        );
    }
    if case.hard {
        for batch in 0..BATCH {
            for channel in 0..3 {
                let i = (batch * CHANNELS + channel) * case.plane + 1;
                assert_eq!(
                    actual[i].to_bits(),
                    expected[i].to_bits(),
                    "Half midpoint witness row{channel}"
                );
            }
        }
    }
}

fn hardware() {
    let context = context();
    for witness in [
        Witness::HardBoundary,
        Witness::SigmoidBoundary,
        Witness::DenseCancellation,
    ] {
        let case = Case::new(129, witness);
        let compiled = Arc::new(
            GpuCompiledModel::compile_with_options(
                case.graph(false),
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
            .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::SqueezeExcite)))
            .unwrap()
            .clone();
        assert_eq!(recipe.words[3], 129);
        assert_eq!(recipe.words[12..14], [1, 1]);
        assert_eq!(recipe.nodes, [0, 1, 2, 3]);
        assert_eq!(session.input_precision(), GpuPrecision::MixedF16);
        context
            .upload_resident(
                &session.resident_input().buffer,
                &GpuPrecision::MixedF16.encode(&case.input),
            )
            .unwrap();
        let mut batch = Batch::new(context.clone()).unwrap();
        let output = batch
            .graph(&mut session, "parallel-se-actual-onnx")
            .unwrap();
        let (bytes, stats) = batch
            .read(
                &output.buffer,
                GpuPrecision::MixedF16.bytes(case.feature.len()),
                false,
            )
            .unwrap();
        assert_eq!(stats.native_graph_dispatches, 1);
        let mut actual = vec![0.0; case.feature.len()];
        GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
        compare(&case, &actual);

        // The same production MSL and admitted packet must honor nonzero p60
        // without touching guards; this also verifies the native buffer ABI.
        let source = KernelKey::storage(KernelId::SqueezeExcite, GpuPrecision::MixedF16);
        let mut batch = Batch::new(context.clone()).unwrap();
        let mut operands = Vec::new();
        for values in [
            &case.input,
            &case.weights,
            &case.bias,
            &case.feature,
            &vec![0.0; 2],
        ] {
            let bytes = GpuPrecision::MixedF16.encode(values);
            let buffer = batch.storage("parallel SE operand", bytes.len()).unwrap();
            context.upload_resident(&buffer, &bytes).unwrap();
            operands.push(buffer);
        }
        let prefix = 7;
        let suffix = 11;
        let sentinel = -777.0;
        let output_values = vec![sentinel; prefix + case.feature.len() + suffix];
        let output = batch
            .storage(
                "parallel SE guarded output",
                GpuPrecision::MixedF16.bytes(output_values.len()),
            )
            .unwrap();
        context
            .upload_resident(&output, &GpuPrecision::MixedF16.encode(&output_values))
            .unwrap();
        let mut words = recipe.words;
        words[60] = prefix as u32;
        batch
            .dispatch(
                "parallel_se_production_kernel",
                source,
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
        let (bytes, stats) = batch.read(&output, output.bytes(), false).unwrap();
        assert_eq!(stats.native_resident_dispatches, 1);
        let mut actual = output_values.clone();
        GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
        assert!(actual[..prefix].iter().all(|&v| v == sentinel));
        assert!(
            actual[prefix + case.feature.len()..]
                .iter()
                .all(|&v| v == sentinel)
        );
        compare(&case, &actual[prefix..prefix + case.feature.len()]);
    }
}

#[test]
#[ignore = "requires a physical GPU; native SE tree, gate boundaries and offset guards"]
fn gpu_hardware_parallel_excitation_native_metal() {
    hardware();
}
