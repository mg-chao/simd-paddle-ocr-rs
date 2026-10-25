use super::super::tests::{integers, model, node, tensor};
use super::*;
use crate::runtime::CompileOptions;
use backend::test_support::MetalTestBatch;
use backend::test_support::MetalTestShader;
use operators::RecipeKind;
use resident::Batch;

const STORAGE: [GpuPrecision; 6] = [GpuPrecision::MixedF16; 6];
const ORIGINAL_SPECIALIZATIONS: [Specialization; 3] = [
    Specialization::BiasGelu192,
    Specialization::BiasGelu384,
    Specialization::Adds768,
];
const TINY_SPECIALIZATIONS: [Specialization; 2] =
    [Specialization::BiasGelu160, Specialization::Adds320];

fn packet(inner: u32, gelu: bool) -> [u32; 128] {
    let mut w = [0; 128];
    for (i, value) in [
        (0, 2 * 73 * 67),
        (2, 2),
        (3, inner),
        (4, 73),
        (5, 1),
        (6, 67),
        (7, 1),
        (8, 67),
        (9, inner),
        (10, 73),
        (11, 1),
        (12, 1),
        (13, 1),
        (14, 1),
        (15, 1),
        (16, 1),
        (32, 2),
        (33, 2),
        (34, 8),
        (35, 2),
        (38, 1),
        (60, 7),
        (64, 2),
        (65, 73),
        (66, 67),
        (68, 1),
        (69, 1),
        (71, if gelu { 7 } else { 2 }),
        (72, if gelu { 0 } else { 2 }),
    ] {
        w[i] = value;
    }
    w
}

#[test]
fn static_pointwise_admission_is_finite_and_preserves_existing_signature_guards() {
    for inner in [
        48, 96, 159, 160, 161, 191, 192, 193, 319, 320, 321, 383, 384, 385, 767, 768, 769,
    ] {
        for gelu in [false, true] {
            let words = packet(inner, gelu);
            let expected = match (inner, gelu) {
                (160, true) => Some(Specialization::BiasGelu160),
                (192, true) => Some(Specialization::BiasGelu192),
                (384, true) => Some(Specialization::BiasGelu384),
                (320, false) => Some(Specialization::Adds320),
                (768, false) => Some(Specialization::Adds768),
                _ => None,
            };
            assert_eq!(
                Specialization::for_recipe(ShaderKind::PointwiseNative64, STORAGE, &words),
                expected
            );
            assert!(Specialization::for_recipe(ShaderKind::MatMul16, STORAGE, &words).is_none());
            for i in 0..6 {
                let mut storage = STORAGE;
                storage[i] = GpuPrecision::F32;
                assert!(
                    Specialization::for_recipe(ShaderKind::PointwiseNative64, storage, &words)
                        .is_none()
                );
            }
            if expected.is_some() {
                for (i, value) in [
                    (64, 1),
                    (64, 3),
                    (68, 2),
                    (69, 0),
                    (71, 8),
                    (65, 72),
                    (66, 68),
                ] {
                    let mut invalid = words;
                    invalid[i] = value;
                    assert!(
                        Specialization::for_recipe(
                            ShaderKind::PointwiseNative64,
                            STORAGE,
                            &invalid
                        )
                        .is_none()
                    );
                }
                if !gelu {
                    let mut channel_residual = words;
                    channel_residual[72] = 1;
                    assert!(
                        Specialization::for_recipe(
                            ShaderKind::PointwiseNative64,
                            STORAGE,
                            &channel_residual
                        )
                        .is_none()
                    );
                }
                // Conv bias and output placement remain runtime fields. The
                // descriptor reduction changes neither arithmetic nor addresses.
                for (i, value) in [(19, 1), (60, 8193)] {
                    let mut runtime = words;
                    runtime[i] = value;
                    assert_eq!(
                        Specialization::for_recipe(
                            ShaderKind::PointwiseNative64,
                            STORAGE,
                            &runtime
                        ),
                        expected
                    );
                }
            }
        }
    }
    for specialization in Specialization::ALL {
        assert_eq!(
            source(specialization, [GpuPrecision::F32; 6])
                .unwrap_err()
                .kind(),
            ErrorKind::Unsupported
        );
    }
}

#[test]
fn static_pointwise_sources_match_all_five_qualified_shader_bodies() {
    // FNV-1a fingerprints of the frozen qualified MSL artifacts. This checks
    // the complete generated source without adding a hashing dependency or
    // making the installed crate's tests depend on reports being present.
    for (specialization, length, fingerprint) in [
        (Specialization::BiasGelu160, 7965, 0x3699_167e_0fa7_6a87),
        (Specialization::BiasGelu192, 7965, 0xb0b0_74fb_c8f1_cf07),
        (Specialization::BiasGelu384, 7965, 0x2d67_6448_f17a_b47b),
        (Specialization::Adds320, 8008, 0x5fab_d344_3dd6_60d4),
        (Specialization::Adds768, 8008, 0x64bb_4e67_b4ce_ffb4),
    ] {
        let generated = source(specialization, STORAGE).unwrap();
        assert_eq!(generated.len(), length, "{}", specialization.name());
        let actual = generated.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
        assert_eq!(actual, fingerprint, "{}", specialization.name());
    }
}

fn half(value: f32) -> f32 {
    half::f16::from_f32(value).to_f32()
}

fn gelu(value: f32) -> f32 {
    let x = value / std::f32::consts::SQRT_2;
    let error = if x.abs() >= 6.0 {
        if x < 0.0 { -1.0 } else { 1.0 }
    } else {
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let mut polynomial = 1.0614054 * t - 1.4531521;
        polynomial = polynomial * t + 1.4214138;
        polynomial = polynomial * t - 0.28449672;
        polynomial = polynomial * t + 0.2548296;
        let magnitude = 1.0 - polynomial * t * (-x * x).exp();
        if x < 0.0 { -magnitude } else { magnitude }
    };
    ((error + 1.0) * value) * 0.5
}

struct Values {
    input: Vec<f32>,
    weight: Vec<f32>,
    bias: Vec<f32>,
    residual: Vec<f32>,
    expected: Vec<f32>,
}
fn values(inner: usize, gelu_activation: bool) -> Values {
    let input = (0..2 * inner * 67)
        .map(|i| ((i * 7 + 3) % 15) as f32 / 32.0 - 7.0 / 32.0)
        .collect::<Vec<_>>();
    let weight = (0..73 * inner)
        .map(|i| {
            let co = i / inner;
            let k = i % inner;
            ((co * 3 + k * 5 + 1) % 17) as f32 / 16384.0 - 8.0 / 16384.0
        })
        .collect::<Vec<_>>();
    let bias = (0..73)
        .map(|i| ((i * 3 + 2) % 9) as f32 / 1024.0 - 4.0 / 1024.0)
        .collect::<Vec<_>>();
    let residual = (0..2 * 73 * 67)
        .map(|i| ((i * 11 + 1) % 13) as f32 / 1024.0 - 6.0 / 1024.0)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(2 * 73 * 67);
    for n in 0..2 {
        for co in 0..73 {
            for px in 0..67 {
                let mut dot = 0.0;
                for k in 0..inner {
                    dot += half(input[(n * inner + k) * 67 + px]) * half(weight[co * inner + k]);
                }
                let affine = dot + half(bias[co]);
                expected.push(half(if gelu_activation {
                    gelu(affine)
                } else {
                    affine + half(residual[(n * 73 + co) * 67 + px])
                }));
            }
        }
    }
    Values {
        input,
        weight,
        bias,
        residual,
        expected,
    }
}

fn fixture(inner: usize, gelu_activation: bool, values: &Values) -> Arc<Model> {
    let mut parts = vec![
        tensor("w", &[73, inner, 1, 1], &values.weight),
        tensor("bias", &[73, 1, 1], &values.bias),
        node(
            "Conv",
            &["x", "w"],
            "conv",
            &[integers("kernel_shape", &[1, 1])],
        ),
        node("Add", &["conv", "bias"], "affine", &[]),
    ];
    if gelu_activation {
        parts.extend([
            tensor("sqrt_two", &[1], &[std::f32::consts::SQRT_2]),
            tensor("one", &[1], &[1.0]),
            tensor("half", &[1], &[0.5]),
            node("Div", &["affine", "sqrt_two"], "scaled", &[]),
            node("Erf", &["scaled"], "error", &[]),
            node("Add", &["error", "one"], "shifted", &[]),
            node("Mul", &["shifted", "affine"], "product", &[]),
            node("Mul", &["product", "half"], "y", &[]),
        ]);
    } else {
        parts.extend([
            tensor("residual", &[2, 73, 1, 67], &values.residual),
            node("Add", &["affine", "residual"], "y", &[]),
        ]);
    }
    model(parts, &[2, inner, 1, 67], &[2, 73, 1, 67])
}

fn context() -> Arc<GpuContext> {
    GpuContext::new_with_options(GpuContextOptions::default()).unwrap()
}

fn compile(
    context: &Arc<GpuContext>,
    inner: usize,
    gelu: bool,
    values: &Values,
) -> Arc<GpuCompiledModel> {
    Arc::new(
        GpuCompiledModel::compile_with_options(
            fixture(inner, gelu, values),
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

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= 1e-4 + 1e-4 * e.abs(),
            "element{i}: {a} != {e}"
        );
    }
}

fn raw_guards(
    context: &Arc<GpuContext>,
    specialization: Specialization,
    inner: usize,
    force_scalar: bool,
) {
    let v = values(inner, specialization.is_bias_gelu());
    let mut batch = Batch::new(context.clone()).unwrap();
    let unused = [f32::NAN; 2];
    let mut operands = Vec::new();
    for data in [
        &v.input[..],
        &v.weight,
        &unused,
        &v.bias,
        if specialization.is_bias_gelu() {
            &unused[..]
        } else {
            &v.residual[..]
        },
    ] {
        let bytes = GpuPrecision::MixedF16.encode(data);
        let buffer = batch
            .storage("static pointwise operand", bytes.len())
            .unwrap();
        context.upload_resident(&buffer, &bytes).unwrap();
        operands.push(buffer);
    }
    let prefix = 7;
    let suffix = 11;
    let sentinel = -777.0;
    let mut guarded = vec![sentinel; prefix + v.expected.len() + suffix];
    let output = batch
        .storage(
            "static pointwise guards",
            GpuPrecision::MixedF16.bytes(guarded.len()),
        )
        .unwrap();
    context
        .upload_resident(&output, &GpuPrecision::MixedF16.encode(&guarded))
        .unwrap();
    let mut source = source(specialization, STORAGE).unwrap();
    if force_scalar {
        source = source.replacen("if (width != 32u ||", "if (true || width != 32u ||", 1);
    }
    let mut buffers = operands.iter().map(|b| (b, true)).collect::<Vec<_>>();
    buffers.push((&output, false));
    batch
        .dispatch_msl(
            "static_pointwise_guards",
            MetalTestShader {
                source: &source,
                entry_point: ENTRY,
                workgroup_size: WORKGROUP_SIZE,
            },
            &buffers,
            &packet(inner as u32, specialization.is_bias_gelu()),
            [8, 1, 1],
        )
        .unwrap();
    let (bytes, _) = batch.read(&output, output.bytes(), false).unwrap();
    GpuPrecision::MixedF16.decode(&bytes, &mut guarded).unwrap();
    assert!(guarded[..prefix].iter().all(|&v| v == sentinel));
    assert!(
        guarded[prefix + v.expected.len()..]
            .iter()
            .all(|&v| v == sentinel)
    );
    close(&guarded[prefix..prefix + v.expected.len()], &v.expected);
}

fn hardware(specializations: &[Specialization], extra_guards: &[(Specialization, usize, bool)]) {
    let context = context();
    for &specialization in specializations {
        let inner = specialization.inner() as usize;
        let v = values(inner, specialization.is_bias_gelu());
        let compiled = compile(&context, inner, specialization.is_bias_gelu(), &v);
        let mut session = GpuSession::new_resident(
            compiled.clone(),
            &[2, inner, 1, 67],
            ExecutionLimits::default(),
            None,
        )
        .unwrap();
        let recipe = session
            .recipes
            .iter()
            .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::PointwiseNative64)))
            .unwrap();
        assert_eq!(
            Specialization::for_recipe(ShaderKind::PointwiseNative64, STORAGE, &recipe.words),
            Some(specialization)
        );
        let selected = compiled
            .pipelines
            .get_for_recipe(
                &context,
                ShaderKind::PointwiseNative64,
                STORAGE,
                &recipe.words,
            )
            .unwrap();
        let mut dynamic_words = recipe.words;
        dynamic_words[3] = 96;
        let dynamic = compiled
            .pipelines
            .get_for_recipe(
                &context,
                ShaderKind::PointwiseNative64,
                STORAGE,
                &dynamic_words,
            )
            .unwrap();
        assert_ne!(
            selected, dynamic,
            "actual graph must use the static reduction PSO"
        );
        context
            .upload_resident(
                &session.resident_input().buffer,
                &GpuPrecision::MixedF16.encode(&v.input),
            )
            .unwrap();
        let mut batch = Batch::new(context.clone()).unwrap();
        let output = batch.graph(&mut session, "static-pointwise").unwrap();
        let (bytes, stats) = batch
            .read(
                &output.buffer,
                GpuPrecision::MixedF16.bytes(v.expected.len()),
                false,
            )
            .unwrap();
        assert_eq!(stats.native_graph_dispatches, 1);
        let mut actual = vec![0.0; v.expected.len()];
        GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
        close(&actual, &v.expected);
        raw_guards(
            &context,
            specialization,
            inner,
            specialization == Specialization::BiasGelu384,
        );
    }
    for &(specialization, inner, force_scalar) in extra_guards {
        raw_guards(&context, specialization, inner, force_scalar);
    }
}

#[test]
#[ignore = "requires physical Metal tensor GPU; native actual static PSOs, N2/tails/offset and scalar mismatch"]
fn gpu_hardware_static_pointwise_half_native_graph() {
    hardware(
        &ORIGINAL_SPECIALIZATIONS,
        &[(Specialization::BiasGelu192, 193, false)],
    );
}

#[cfg(target_os = "macos")]
fn rejection_preserves_fixed_pipeline(specializations: &[Specialization]) {
    let context = context();
    for &specialization in specializations {
        let inner = specialization.inner() as usize;
        let v = values(inner, specialization.is_bias_gelu());
        let compiled = compile(&context, inner, specialization.is_bias_gelu(), &v);
        let mut words = packet(inner as u32, specialization.is_bias_gelu());
        words[3] = 96;
        let dynamic = compiled
            .pipelines
            .get_for_recipe(&context, ShaderKind::PointwiseNative64, STORAGE, &words)
            .unwrap();
        let before = dynamic.clone();
        compiled.pipelines.reject_pointwise_static_for_test(
            specialization,
            RuntimeError::with_kind(ErrorKind::Unsupported, "fixture static source rejected"),
        );
        let mut session = GpuSession::new_resident(
            compiled.clone(),
            &[2, inner, 1, 67],
            ExecutionLimits::default(),
            None,
        )
        .unwrap();
        let recipe = session
            .recipes
            .iter()
            .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::PointwiseNative64)))
            .unwrap();
        let selected = compiled
            .pipelines
            .get_for_recipe(
                &context,
                ShaderKind::PointwiseNative64,
                STORAGE,
                &recipe.words,
            )
            .unwrap();
        assert_eq!(selected, dynamic);
        assert!(Arc::ptr_eq(&before, &selected.clone()));
        context
            .upload_resident(
                &session.resident_input().buffer,
                &GpuPrecision::MixedF16.encode(&v.input),
            )
            .unwrap();
        let mut batch = Batch::new(context.clone()).unwrap();
        let output = batch.graph(&mut session, "static-fallback").unwrap();
        let (bytes, stats) = batch
            .read(
                &output.buffer,
                GpuPrecision::MixedF16.bytes(v.expected.len()),
                false,
            )
            .unwrap();
        assert_eq!(stats.native_graph_dispatches, 1);
        let mut actual = vec![0.0; v.expected.len()];
        GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
        close(&actual, &v.expected);
    }
}

#[test]
#[ignore = "requires physical Metal tensor GPU; BG160/Adds320 native graph PSOs, N2/tails/p60, forced scalar and K mismatch"]
fn gpu_hardware_tiny_static_pointwise_half_native_graph() {
    hardware(
        &TINY_SPECIALIZATIONS,
        &[
            (Specialization::BiasGelu160, 160, true),
            (Specialization::Adds320, 320, true),
            (Specialization::BiasGelu160, 161, false),
            (Specialization::Adds320, 321, false),
        ],
    );
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires physical Metal tensor GPU; rejected static reduction retains existing Fixed/native PSO"]
fn gpu_hardware_static_pointwise_rejection_preserves_fixed_pipeline() {
    rejection_preserves_fixed_pipeline(&ORIGINAL_SPECIALIZATIONS);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires physical Metal tensor GPU; rejected BG160/Adds320 retains exact Fixed/native PSO identity"]
fn gpu_hardware_tiny_static_pointwise_rejection_preserves_fixed_pipeline() {
    rejection_preserves_fixed_pipeline(&TINY_SPECIALIZATIONS);
}
