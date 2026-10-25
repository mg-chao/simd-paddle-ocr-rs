use super::super::tests::{integers, model, node, tensor};
use super::*;
use crate::runtime::CompileOptions;
use resident::Batch;

const STORAGE: [GpuPrecision; 6] = [GpuPrecision::MixedF16; 6];

fn words(inner: u32, packets: &[[u32; 3]]) -> [u32; 128] {
    let mut w = [0; 128];
    // N2, Cout73 and HW67 exercise batch/channel/spatial tails together.
    for (i, v) in [
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
        (60, 7),
        (65, 73),
        (66, 67),
    ] {
        w[i] = v;
    }
    w[64] = packets.len() as u32;
    for (i, packet) in packets.iter().enumerate() {
        w[68 + i * 3..71 + i * 3].copy_from_slice(packet);
    }
    w
}

#[test]
fn known_pointwise_epilogue_admission_is_bounded_and_proves_channel_address() {
    for inner in [
        17, 47, 48, 49, 95, 96, 128, 160, 192, 193, 320, 384, 512, 768, 769, 1536,
    ] {
        for count in 0..=4 {
            let w = words(inner, &vec![[1, 1, 0]; count]);
            let eligible = (96..=768).contains(&inner) && (1..=3).contains(&count);
            assert_eq!(
                known_pointwise_epilogue(ShaderKind::PointwiseNative64, STORAGE, &w),
                eligible
            );
            assert!(!known_pointwise_epilogue(ShaderKind::MatMul16, STORAGE, &w));
            for binding in 0..6 {
                let mut mixed = STORAGE;
                mixed[binding] = GpuPrecision::F32;
                assert!(!known_pointwise_epilogue(
                    ShaderKind::PointwiseNative64,
                    mixed,
                    &w
                ));
            }
        }
    }
    let valid = words(128, &[[1, 1, 0], [2, 2, 0], [7, 0, 0]]);
    for (i, v) in [(65, 72), (66, 66), (7, 0), (7, u32::MAX)] {
        let mut invalid = valid;
        invalid[i] = v;
        assert!(!known_pointwise_epilogue(
            ShaderKind::PointwiseNative64,
            STORAGE,
            &invalid
        ));
    }
    // The kernel's known row equals the old mode1 address for every valid
    // logical element; p60 is physical output placement, not a channel input.
    for offset in [0, 7, 8193] {
        let mut w = valid;
        w[60] = offset;
        assert!(known_pointwise_epilogue(
            ShaderKind::PointwiseNative64,
            STORAGE,
            &w
        ));
        for batch in 0..2 {
            for channel in 0..73 {
                for pixel in 0..67 {
                    let index = (batch * 73 + channel) * 67 + pixel;
                    assert_eq!((index / w[66]) % w[65], channel);
                }
            }
        }
    }
}

#[test]
fn low_k_pointwise_admits_only_exact_48_bias_gelu_and_preserves_other_guards() {
    for inner in [16, 47, 48, 49, 95, 96, 768, 769, 1536] {
        let bg = words(inner, &[[1, 1, 0], [7, 0, 0]]);
        let known = (96..=768).contains(&inner);
        assert_eq!(
            fixed_pointwise_bias_gelu(ShaderKind::PointwiseNative64, STORAGE, &bg),
            inner == 48 || known
        );
        assert_eq!(
            known_pointwise_epilogue(ShaderKind::PointwiseNative64, STORAGE, &bg),
            known
        );
        let adds = words(inner, &[[1, 1, 0], [2, 2, 0]]);
        let brg = words(inner, &[[1, 1, 0], [2, 2, 0], [7, 0, 0]]);
        assert_eq!(
            fixed_pointwise_adds(ShaderKind::PointwiseNative64, STORAGE, &adds),
            known
        );
        assert_eq!(
            fixed_pointwise_epilogue(ShaderKind::PointwiseNative64, STORAGE, &brg),
            known
        );
    }
    let valid = words(48, &[[1, 1, 0], [7, 0, 0]]);
    for (index, value) in [
        (64, 1),
        (64, 3),
        (68, 2),
        (69, 0),
        (69, 2),
        (71, 3),
        (65, 72),
        (66, 66),
        (7, 0),
        (7, u32::MAX),
    ] {
        let mut mismatch = valid;
        mismatch[index] = value;
        assert!(!fixed_pointwise_bias_gelu(
            ShaderKind::PointwiseNative64,
            STORAGE,
            &mismatch
        ));
    }
    for binding in 0..6 {
        let mut storage = STORAGE;
        storage[binding] = GpuPrecision::F32;
        assert!(!fixed_pointwise_bias_gelu(
            ShaderKind::PointwiseNative64,
            storage,
            &valid
        ));
    }
    assert!(!fixed_pointwise_bias_gelu(
        ShaderKind::MatMul16,
        STORAGE,
        &valid
    ));
    for (index, value) in [(19, 1), (60, 8193), (70, 123), (72, 456), (73, 789)] {
        let mut unused = valid;
        unused[index] = value;
        assert!(fixed_pointwise_bias_gelu(
            ShaderKind::PointwiseNative64,
            STORAGE,
            &unused
        ));
    }
}

#[test]
fn known_pointwise_epilogue_keeps_generic_source_and_only_changes_epilogue_calls() {
    let generic = pointwise_source();
    let candidate = pointwise_epilogue_source();
    assert!(!generic.contains("epilogue_known"));
    assert!(candidate.contains("epilogue_known(sum, index, row, raw_d, raw_e, params)"));
    assert!(candidate.contains("epilogue_known(accum[i], index, row, raw_d, raw_e, params)"));
    let restored = candidate
        .replace(
            &format!("{}\n", include_str!("pointwise_epilogue.metal")),
            "",
        )
        .replace(
            "epilogue_known(sum, index, row, raw_d, raw_e, params)",
            "epilogue(sum, index, raw_d, raw_e, params)",
        )
        .replace(
            "epilogue_known(accum[i], index, row, raw_d, raw_e, params)",
            "epilogue(accum[i], index, raw_d, raw_e, params)",
        );
    assert_eq!(
        restored, generic,
        "matrix/tail/scalar code must stay byte-identical"
    );
    let helper = include_str!("pointwise_epilogue.metal");
    assert!(
        helper.contains("if (count > 3u) return epilogue(initial, index, raw_d, raw_e, params)")
    );
}

#[test]
fn fixed_pointwise_epilogues_require_the_exact_packet_chain() {
    for (packets, expected, meaningful) in [
        (vec![[1, 1, 0], [7, 0, 0]], 0, vec![64, 68, 69, 71]),
        (
            vec![[1, 1, 0], [2, 2, 0], [7, 0, 0]],
            1,
            vec![64, 68, 69, 71, 72, 74],
        ),
        (vec![[1, 1, 0], [2, 2, 0]], 2, vec![64, 68, 69, 71, 72]),
    ] {
        let valid = words(128, &packets);
        let classify = |w: &[u32; 128], storage| {
            [
                fixed_pointwise_bias_gelu(ShaderKind::PointwiseNative64, storage, w),
                fixed_pointwise_epilogue(ShaderKind::PointwiseNative64, storage, w),
                fixed_pointwise_adds(ShaderKind::PointwiseNative64, storage, w),
            ]
        };
        let mut admitted = [false; 3];
        admitted[expected] = true;
        assert_eq!(classify(&valid, STORAGE), admitted);
        for i in meaningful {
            for value in [0, valid[i] + 1] {
                let mut bad = valid;
                bad[i] = value;
                assert_eq!(classify(&bad, STORAGE), [false; 3], "word{i}={value}");
            }
        }
        for i in [3, 65, 66] {
            let mut bad = valid;
            bad[i] = 0;
            assert_eq!(classify(&bad, STORAGE), [false; 3]);
        }
        for binding in 0..6 {
            let mut mixed = STORAGE;
            mixed[binding] = GpuPrecision::F32;
            assert_eq!(classify(&valid, mixed), [false; 3]);
        }
        // Unused packet immediates and the Conv bias flag are not epilogue
        // signature fields. Initial accumulator/bias handling stays unchanged.
        for i in [19, 60, 70, 73] {
            let mut unchanged = valid;
            unchanged[i] = 8193;
            assert_eq!(classify(&unchanged, STORAGE), admitted);
        }
    }
    for (source, helper, name) in [
        (
            pointwise_bias_gelu_source(),
            include_str!("pointwise_bias_gelu.metal"),
            "epilogue_fixed_bg",
        ),
        (
            pointwise_fixed_epilogue_source(),
            include_str!("pointwise_fixed_epilogue.metal"),
            "epilogue_fixed_brg",
        ),
        (
            pointwise_adds_source(),
            include_str!("pointwise_adds.metal"),
            "epilogue_fixed_adds",
        ),
    ] {
        let restored = source
            .replace(&format!("{helper}\n"), "")
            .replace(
                &format!("{name}(sum, index, row, raw_d, raw_e, params)"),
                "epilogue(sum, index, raw_d, raw_e, params)",
            )
            .replace(
                &format!("{name}(accum[i], index, row, raw_d, raw_e, params)"),
                "epilogue(accum[i], index, raw_d, raw_e, params)",
            );
        assert_eq!(
            restored,
            pointwise_source(),
            "fixed epilogue must not change the matrix/scalar/tail body"
        );
    }
    assert!(!include_str!("pointwise_bias_gelu.metal").contains("raw_e["));
}

fn half(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

fn erf(x: f32) -> f32 {
    if x.abs() >= 6.0 {
        return if x < 0.0 { -1.0 } else { 1.0 };
    }
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly =
        (((((1.0614054 * t - 1.4531521) * t) + 1.4214138) * t - 0.28449672) * t + 0.2548296) * t;
    (if x < 0.0 { -1.0 } else { 1.0 }) * (1.0 - poly * (-x * x).exp())
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn activate(value: f32, packet: [u32; 3]) -> f32 {
    match packet[0] {
        3 => {
            if value > 0.0 {
                value
            } else {
                0.0
            }
        }
        4 => sigmoid(value),
        5 => (value * f32::from_bits(packet[1]) + f32::from_bits(packet[2])).clamp(0.0, 1.0),
        6 => value * (value * 0.16666667 + 0.5).clamp(0.0, 1.0),
        7 => ((erf(value / std::f32::consts::SQRT_2) + 1.0) * value) * 0.5,
        8 => value * sigmoid(value),
        _ => unreachable!(),
    }
}

fn context() -> Arc<GpuContext> {
    GpuContext::new_with_options(GpuContextOptions::default()).unwrap()
}

fn close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-4 + 1e-4 * e.abs(),
            "{label} element{i}: {a} != {e}"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires physical Metal tensor GPU; known-channel Half tails, packets, all opcode and output-offset contracts"]
fn gpu_hardware_known_pointwise_epilogue_packets_tails_and_offsets() {
    let context = context();
    let inner = 128usize;
    let count = 2 * 73 * 67;
    let a = (0..2 * inner * 67)
        .map(|i| ((i * 7 + 3) % 15) as f32 / 32.0 - 7.0 / 32.0)
        .collect::<Vec<_>>();
    let b = (0..73 * inner)
        .map(|i| ((i * 5 + 1) % 17) as f32 / 16384.0 - 8.0 / 16384.0)
        .collect::<Vec<_>>();
    let c = (0..73)
        .map(|i| (i % 7 + 1) as f32 / 1024.0)
        .collect::<Vec<_>>();
    let d = (0..count)
        .map(|i| 0.0625 + ((i * 3 + 2) % 9) as f32 / 1024.0)
        .collect::<Vec<_>>();
    let e = (0..count)
        .map(|i| ((i * 11 + 1) % 13) as f32 / 1024.0 - 6.0 / 1024.0)
        .collect::<Vec<_>>();
    let mut dots = vec![0.0; count];
    let mut biased_dots = vec![0.0; count];
    for n in 0..2 {
        for co in 0..73 {
            for px in 0..67 {
                let index = (n * 73 + co) * 67 + px;
                biased_dots[index] = half(c[co]);
                for k in 0..inner {
                    let product = half(a[(n * inner + k) * 67 + px]) * half(b[co * inner + k]);
                    dots[index] += product;
                    biased_dots[index] += product;
                }
            }
        }
    }
    let mut cases = vec![vec![[1, 1, 0]], vec![[1, 0, 0], [2, 2, 0]]];
    for op in 3..=8 {
        cases.push(vec![
            [1, 1, 0],
            [2, 2, 0],
            [op, 0.2f32.to_bits(), 0.5f32.to_bits()],
        ]);
    }
    cases.push(vec![[1, 2, 0], [2, 1, 0], [7, 0, 0]]);
    cases.push(vec![[1, 1, 0], [2, 0, 0]]);
    cases.push(vec![[1, 2, 0], [2, 1, 0], [7, 0, 0], [3, 0, 0]]);
    let mut scalar = cases[6].clone();
    scalar[2][0] = 7;
    let scalar3 = cases.len();
    cases.push(scalar);
    cases.push(vec![[1, 1, 0], [7, 0, 0]]);
    let scalar2 = cases.len();
    cases.push(vec![[1, 1, 0], [7, 0, 0]]);
    let adds = cases.len();
    cases.push(vec![[1, 1, 0], [2, 2, 0]]);
    let scalar_adds = cases.len();
    cases.push(vec![[1, 1, 0], [2, 2, 0]]);
    for (case_index, packets) in cases.iter().enumerate() {
        let with_conv_bias = case_index >= adds;
        let mut expected = if with_conv_bias {
            biased_dots.clone()
        } else {
            dots.clone()
        };
        for (index, value) in expected.iter_mut().enumerate() {
            for &packet in packets {
                if packet[0] <= 2 {
                    let address = match packet[1] {
                        0 => 0,
                        1 => (index / 67) % 73,
                        _ => index,
                    };
                    *value += half(if packet[0] == 1 {
                        d[address]
                    } else {
                        e[address]
                    });
                } else {
                    *value = activate(*value, packet);
                }
            }
            *value = half(*value);
        }
        let mut w = words(inner as u32, packets);
        w[19] = u32::from(with_conv_bias);
        let mut outputs = Vec::new();
        let fixed3 = fixed_pointwise_epilogue(ShaderKind::PointwiseNative64, STORAGE, &w);
        let fixed2 = fixed_pointwise_bias_gelu(ShaderKind::PointwiseNative64, STORAGE, &w);
        let fixed_adds = fixed_pointwise_adds(ShaderKind::PointwiseNative64, STORAGE, &w);
        let variants = if fixed2 || fixed3 || fixed_adds { 3 } else { 2 };
        for variant in 0..variants {
            let mut source = match variant {
                0 => pointwise_source(),
                1 => pointwise_epilogue_source(),
                _ if fixed2 => pointwise_bias_gelu_source(),
                _ if fixed_adds => pointwise_adds_source(),
                _ => pointwise_fixed_epilogue_source(),
            };
            if [scalar3, scalar2, scalar_adds].contains(&case_index) {
                source = source.replace("if (width != 32u)", "if (true)");
            }
            let mut batch = Batch::new(context.clone()).unwrap();
            let mut operands = Vec::new();
            // A fixed bias+GELU source cannot depend on the unused E binding.
            let unused_e = if fixed2 {
                vec![f32::NAN; count]
            } else {
                e.clone()
            };
            for values in [&a, &b, &c, &d, &unused_e] {
                let bytes = GpuPrecision::MixedF16.encode(values);
                let buffer = batch.storage("known PW operand", bytes.len()).unwrap();
                context.upload_resident(&buffer, &bytes).unwrap();
                operands.push(buffer);
            }
            let prefix = 7;
            let suffix = 11;
            let sentinel = -777.0;
            let values = vec![sentinel; prefix + count + suffix];
            let output = batch
                .storage(
                    "known PW guarded output",
                    GpuPrecision::MixedF16.bytes(values.len()),
                )
                .unwrap();
            context
                .upload_resident(&output, &GpuPrecision::MixedF16.encode(&values))
                .unwrap();
            batch
                .dispatch_native(
                    "known_pointwise_contract",
                    resident::NativeShader {
                        source: &source,
                        entry_point: "pointwise_mpp64",
                        workgroup_size: [128, 1, 1],
                    },
                    &[
                        (&operands[0], true),
                        (&operands[1], true),
                        (&operands[2], true),
                        (&operands[3], true),
                        (&operands[4], true),
                        (&output, false),
                    ],
                    &w,
                    [8, 1, 1],
                )
                .unwrap();
            let (bytes, stats) = batch.read(&output, output.bytes(), false).unwrap();
            assert_eq!(stats.native_resident_dispatches, 1);
            let mut actual = values;
            GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
            assert!(actual[..prefix].iter().all(|&v| v == sentinel));
            assert!(actual[prefix + count..].iter().all(|&v| v == sentinel));
            close(
                "known PW independent oracle",
                &actual[prefix..prefix + count],
                &expected,
            );
            outputs.push(actual);
        }
        for output in &outputs[1..] {
            assert_eq!(
                outputs[0], *output,
                "fixed/known and generic Half results must match for case{case_index}"
            );
        }
    }
}

#[derive(Clone, Copy)]
enum ProductionActivation {
    Relu,
    Gelu,
    None,
}

fn production_case(inner: usize, activation: ProductionActivation, residual: bool) {
    let context = context();
    let gelu = matches!(activation, ProductionActivation::Gelu);
    let no_activation = matches!(activation, ProductionActivation::None);
    let shape = [2, inner, 1, 67];
    let bias = if gelu { 0.03125 } else { 0.25 };
    let residual_value = if gelu { 0.0625 } else { 0.125 };
    let weights = (0..73 * inner)
        .map(|i| ((i * 5 + 1) % 17) as f32 / 16384.0 - 8.0 / 16384.0)
        .collect::<Vec<_>>();
    let mut parts = vec![
        tensor("w", &[73, inner, 1, 1], &weights),
        tensor("bias", &[73, 1, 1], &vec![bias; 73]),
        node(
            "Conv",
            &["x", "w"],
            "conv",
            &[integers("kernel_shape", &[1, 1])],
        ),
        node(
            "Add",
            &["conv", "bias"],
            if no_activation && !residual {
                "y"
            } else {
                "affine"
            },
            &[],
        ),
    ];
    let value = if residual {
        parts.extend([
            tensor(
                "residual",
                &[2, 73, 1, 67],
                &vec![residual_value; 2 * 73 * 67],
            ),
            node(
                "Add",
                &["affine", "residual"],
                if no_activation { "y" } else { "sum" },
                &[],
            ),
        ]);
        if no_activation { "y" } else { "sum" }
    } else if no_activation {
        "y"
    } else {
        "affine"
    };
    if gelu {
        parts.extend([
            tensor("sqrt_two", &[1], &[std::f32::consts::SQRT_2]),
            tensor("one", &[1], &[1.0]),
            tensor("half", &[1], &[0.5]),
            node("Div", &[value, "sqrt_two"], "scaled", &[]),
            node("Erf", &["scaled"], "error", &[]),
            node("Add", &["error", "one"], "shifted", &[]),
            node("Mul", &["shifted", value], "product", &[]),
            node("Mul", &["product", "half"], "y", &[]),
        ]);
    } else if !no_activation {
        parts.push(node("Relu", &[value], "y", &[]));
    }
    let graph = model(parts, &shape, &[2, 73, 1, 67]);
    let input = (0..shape.iter().product())
        .map(|i| ((i * 7 + 3) % 15) as f32 / 32.0 - 7.0 / 32.0)
        .collect::<Vec<_>>();
    let mut expected = vec![0.0; 2 * 73 * 67];
    for n in 0..2 {
        for co in 0..73 {
            for px in 0..67 {
                let mut sum = 0.0;
                for k in 0..inner {
                    sum += half(input[(n * inner + k) * 67 + px]) * half(weights[co * inner + k]);
                }
                sum += bias;
                if residual {
                    sum += residual_value;
                }
                expected[(n * 73 + co) * 67 + px] = half(if gelu {
                    activate(sum, [7, 0, 0])
                } else if !no_activation {
                    sum.max(0.0)
                } else {
                    sum
                });
            }
        }
    }
    let compiled = Arc::new(
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
    );
    let mut session =
        GpuSession::new_resident(compiled.clone(), &shape, ExecutionLimits::default(), None)
            .unwrap();
    let recipe = session
        .recipes
        .iter()
        .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::PointwiseNative64)))
        .unwrap();
    assert_eq!(
        recipe.words[64],
        1 + u32::from(residual) + u32::from(!no_activation)
    );
    let generic = compiled
        .pipelines
        .get(&context, ShaderKind::PointwiseNative64, STORAGE)
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
    let known_eligible = (96..=768).contains(&inner);
    let low_k_bg = inner == 48 && gelu && !residual;
    assert_eq!(selected == generic, !(known_eligible || low_k_bg));
    if ((gelu || no_activation) && known_eligible) || low_k_bg {
        assert_eq!(
            fixed_pointwise_epilogue(ShaderKind::PointwiseNative64, STORAGE, &recipe.words),
            known_eligible && gelu && residual
        );
        assert_eq!(
            fixed_pointwise_bias_gelu(ShaderKind::PointwiseNative64, STORAGE, &recipe.words),
            (known_eligible || inner == 48) && gelu && !residual
        );
        assert_eq!(
            fixed_pointwise_adds(ShaderKind::PointwiseNative64, STORAGE, &recipe.words),
            known_eligible && no_activation && residual
        );
        let mut mismatch = recipe.words;
        mismatch[69] = 0; // A scalar bias rejects Fixed; K48 must stay Generic.
        let fallback = compiled
            .pipelines
            .get_for_recipe(&context, ShaderKind::PointwiseNative64, STORAGE, &mismatch)
            .unwrap();
        assert_ne!(
            selected, fallback,
            "actual production recipe must use its fixed pipeline"
        );
        assert_eq!(fallback == generic, !known_eligible);
    }
    context
        .upload_resident(
            &session.resident_input().buffer,
            &GpuPrecision::MixedF16.encode(&input),
        )
        .unwrap();
    let mut batch = Batch::new(context.clone()).unwrap();
    let output = batch
        .graph(&mut session, "fixed/known PW production")
        .unwrap();
    let (bytes, stats) = batch
        .read(
            &output.buffer,
            GpuPrecision::MixedF16.bytes(expected.len()),
            false,
        )
        .unwrap();
    assert_eq!(stats.native_graph_dispatches, 1);
    let mut actual = vec![0.0; expected.len()];
    GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
    close("fixed/known PW actual production", &actual, &expected);
}

#[test]
#[ignore = "requires physical Metal tensor GPU; production known-channel and unchanged generic pipeline selection"]
fn gpu_hardware_known_pointwise_epilogue_production_and_large_k_isolation() {
    {
        for inner in [128, 320, 768, 769] {
            production_case(inner, ProductionActivation::Relu, true);
        }
    }
}

#[test]
#[ignore = "requires physical Metal tensor GPU; actual fixed two/three-packet GELU and unchanged generic execution"]
fn gpu_hardware_fixed_pointwise_epilogue_production_gelu() {
    {
        for inner in [128, 769] {
            for residual in [false, true] {
                production_case(inner, ProductionActivation::Gelu, residual);
            }
        }
    }
}

#[test]
#[ignore = "requires physical Metal tensor GPU; exact K48 bias/GELU native execution, neighboring K/residual opt-outs"]
fn gpu_hardware_low_k_fixed_pointwise_bias_gelu_production() {
    {
        for (inner, residual) in [(48, false), (48, true), (49, false)] {
            production_case(inner, ProductionActivation::Gelu, residual);
        }
    }
}

#[test]
#[ignore = "requires physical Metal tensor GPU; exact fixed channel-bias/full-residual production signature and generic isolation"]
fn gpu_hardware_fixed_pointwise_adds_production() {
    {
        for inner in [128, 769] {
            production_case(inner, ProductionActivation::None, true);
        }
    }
}
