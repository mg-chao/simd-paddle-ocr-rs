//! Exact canonical Half DW3 channel-bias specialization and physical boundaries.
use super::super::tests::{bind_program, integer, integers, model, node, tensor};
use super::*;
use crate::runtime::CompileOptions;
use backend::test_support::MetalTestBatch;
use resident::Batch;

const STORAGE: [GpuPrecision; 6] = [GpuPrecision::MixedF16; 6];

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
            (h + self.pads[0] + self.pads[2] - 3) / self.stride[0] + 1,
            (w + self.pads[1] + self.pads[3] - 3) / self.stride[1] + 1,
        ]
    }
    fn words(self) -> [u32; 128] {
        let [n, c, h, w] = self.shape;
        let [_, _, oh, ow] = self.output();
        let mut words = [0; 128];
        for (i, value) in [
            (0, n * c * oh * ow),
            (2, n),
            (3, c),
            (4, c),
            (5, h),
            (6, w),
            (7, oh),
            (8, ow),
            (9, 1),
            (10, 1),
            (11, 3),
            (12, 3),
            (13, self.stride[0]),
            (14, self.stride[1]),
            (15, 1),
            (16, 1),
            (17, self.pads[0]),
            (18, self.pads[1]),
            (60, 7),
            (64, 1),
            (65, c),
            (66, oh * ow),
            (68, 1),
            (69, 1),
        ] {
            words[i] = value as u32;
        }
        words
    }
    fn values(self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let [n, c, h, w] = self.shape;
        let mut input = (0..n * c * h * w)
            .map(|i| ((i * 7 + 3) % 31) as f32 / 32.0 - 15.0 / 32.0)
            .collect::<Vec<_>>();
        let mut weights = (0..c * 9)
            .map(|i| ((i * 5 + 1) % 17) as f32 / 64.0 - 8.0 / 64.0)
            .collect::<Vec<_>>();
        let bias = (0..c)
            .map(|i| ((i * 3 + 2) % 9) as f32 / 128.0 - 4.0 / 128.0)
            .collect();
        // Interior channel0 includes2048+1-2048. Float retains1; Half loses it.
        for batch in 0..n {
            input[batch * c * h * w..batch * c * h * w + h * w].fill(1.0);
        }
        weights[..9].fill(0.0);
        weights[3..6].copy_from_slice(&[2048.0, 1.0, -2048.0]);
        (input, weights, bias)
    }
}

#[derive(Clone, Copy)]
enum Usage {
    Bias,
    Relu,
    Escape,
}

fn graph(case: Case, usage: Usage) -> Arc<Model> {
    let (_, weights, bias) = case.values();
    let mut parts = vec![
        tensor("w", &[case.shape[1], 1, 3, 3], &weights),
        tensor("bias", &[1, case.shape[1], 1, 1], &bias),
        node(
            "Conv",
            &["x", "w"],
            "conv",
            &[
                integers("kernel_shape", &[3; 2]),
                integers("strides", &case.stride),
                integers("pads", &case.pads),
                integer("group", case.shape[1]),
            ],
        ),
        node(
            "Add",
            &["conv", "bias"],
            if matches!(usage, Usage::Bias) {
                "y"
            } else {
                "affine"
            },
            &[],
        ),
    ];
    match usage {
        Usage::Bias => {}
        Usage::Relu => parts.push(node("Relu", &["affine"], "y", &[])),
        Usage::Escape => parts.extend([
            node("Relu", &["conv"], "escaped", &[]),
            node("Add", &["affine", "escaped"], "y", &[]),
        ]),
    }
    model(parts, &case.shape, &case.output())
}

fn oracle(case: Case) -> Vec<f32> {
    let (input, weights, bias) = case.values();
    let [n, channels, height, width] = case.shape;
    let [_, _, oh, ow] = case.output();
    let mut expected = vec![0.0; n * channels * oh * ow];
    for batch in 0..n {
        for channel in 0..channels {
            for y in 0..oh {
                for x in 0..ow {
                    let mut sum = 0.0f32;
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let iy = (y * case.stride[0] + ky) as isize - case.pads[0] as isize;
                            let ix = (x * case.stride[1] + kx) as isize - case.pads[1] as isize;
                            if iy >= 0 && iy < height as isize && ix >= 0 && ix < width as isize {
                                let a = input[((batch * channels + channel) * height
                                    + iy as usize)
                                    * width
                                    + ix as usize];
                                sum += a * weights[channel * 9 + ky * 3 + kx];
                            }
                        }
                    }
                    expected[((batch * channels + channel) * oh + y) * ow + x] =
                        half::f16::from_f32(sum + bias[channel]).to_f32();
                }
            }
        }
    }
    expected
}

#[test]
fn half_depthwise_bias_admission_and_source_preserve_float_boundaries() {
    let case = Case {
        shape: [2, 7, 7, 27],
        stride: [1, 2],
        pads: [0, 1, 2, 0],
    };
    let valid = case.words();
    assert!(fixed_depthwise_bias(
        ShaderKind::DepthwiseNative3,
        STORAGE,
        &valid
    ));
    for kind in [
        ShaderKind::Depthwise3,
        ShaderKind::DepthwiseNative5,
        ShaderKind::DepthwiseNative9,
        ShaderKind::DepthwiseNative1x7,
        ShaderKind::PointwiseNative64,
    ] {
        assert!(!fixed_depthwise_bias(kind, STORAGE, &valid));
    }
    for binding in 0..6 {
        let mut storage = STORAGE;
        storage[binding] = GpuPrecision::F32;
        assert!(!fixed_depthwise_bias(
            ShaderKind::DepthwiseNative3,
            storage,
            &valid
        ));
    }
    for (index, value) in [
        (0, 0),
        (2, 0),
        (3, 8),
        (9, 7),
        (11, 5),
        (12, 5),
        (13, 3),
        (14, 3),
        (15, 2),
        (16, 2),
        (19, 1),
        (52, 1),
        (64, 0),
        (64, 2),
        (65, 6),
        (66, 0),
        (68, 0),
        (68, 2),
        (69, 0),
        (69, 2),
        (7, u32::MAX),
    ] {
        let mut invalid = valid;
        invalid[index] = value;
        assert!(
            !fixed_depthwise_bias(ShaderKind::DepthwiseNative3, STORAGE, &invalid),
            "word{index}={value}"
        );
    }
    for index in [60, 70, 71, 72] {
        let mut unchanged = valid;
        unchanged[index] = 8193;
        assert!(fixed_depthwise_bias(
            ShaderKind::DepthwiseNative3,
            STORAGE,
            &unchanged
        ));
    }
    let candidate = depthwise_bias_source();
    assert_eq!(
        candidate
            .matches("const float channel_bias = float(raw_d[channel]);")
            .count(),
        1
    );
    let restored = candidate
        .replace(
            "\n    const float channel_bias = float(raw_d[channel]);",
            "",
        )
        .replace(
            "sum[pixel] + channel_bias;",
            "epilogue(sum[pixel], index + pixel, raw_d, raw_e, params);",
        );
    assert_eq!(
        restored,
        depthwise_source(3),
        "all Float tap operations/windows/tails must remain identical"
    );
    assert!(!depthwise_source(5).contains("channel_bias"));
    assert!(!depthwise_source(9).contains("channel_bias"));
    let [n, c, oh, ow] = case.output();
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    let logical = ((batch * c + channel) * oh + y) * ow + x;
                    assert_eq!((logical / (oh * ow)) % c, channel);
                }
            }
        }
    }
}

#[test]
fn half_depthwise_bias_physical_prefix_escape_and_f32_matrix_remain_generic() {
    let case = Case {
        shape: [2, 7, 3, 17],
        stride: [1; 2],
        pads: [1; 4],
    };
    for usage in [Usage::Bias, Usage::Relu, Usage::Escape] {
        let program = Program::compile(graph(case, usage)).unwrap();
        let conv = program.instructions[0].output;
        for matrix in [GpuMatrixPrecision::F16, GpuMatrixPrecision::F32] {
            let mut plan = bind_program(&program, &case.shape, ExecutionLimits::default()).unwrap();
            let mut descriptors = plan
                .shapes
                .iter()
                .map(|&s| GpuTensorDescriptor::canonical(s, GpuPrecision::MixedF16))
                .collect();
            let mut recipes = physical::lower(
                &program,
                &mut plan,
                program.output,
                &crate::runtime::gpu::GpuLimits::default(),
                GpuPrecision::MixedF16,
                &mut descriptors,
                matrix,
                &mut |_| Ok(false),
            )
            .unwrap();
            select(
                &mut recipes,
                &descriptors,
                &crate::runtime::gpu::GpuLimits::default(),
                matrix,
            )
            .unwrap();
            let dw = recipes.iter().find(|r| r.nodes.contains(&0)).unwrap();
            let admitted = matches!(usage, Usage::Bias) && matrix == GpuMatrixPrecision::F16;
            let RecipeKind::Compute(kind) = dw.kind else {
                panic!("compute expected")
            };
            assert_eq!(fixed_depthwise_bias(kind, STORAGE, &dw.words), admitted);
            if matches!(usage, Usage::Escape) {
                assert_eq!(dw.output, conv);
                assert_eq!(dw.nodes, [0]);
                assert!(recipes.iter().filter(|r| r.inputs.contains(&conv)).count() >= 2);
            }
        }
        let mut prefix = ShapePlan::empty(
            program.model.graph().tensors.len(),
            program.instructions.len(),
        );
        bind_into(
            &program,
            &case.shape,
            ExecutionLimits::default(),
            1,
            conv,
            &mut prefix,
            &mut BindScratch::new(&program),
        )
        .unwrap();
        let mut descriptors = prefix
            .shapes
            .iter()
            .map(|&s| GpuTensorDescriptor::canonical(s, GpuPrecision::MixedF16))
            .collect();
        let mut recipes = physical::lower(
            &program,
            &mut prefix,
            conv,
            &crate::runtime::gpu::GpuLimits::default(),
            GpuPrecision::MixedF16,
            &mut descriptors,
            GpuMatrixPrecision::F16,
            &mut |_| Ok(false),
        )
        .unwrap();
        select(
            &mut recipes,
            &descriptors,
            &crate::runtime::gpu::GpuLimits::default(),
            GpuMatrixPrecision::F16,
        )
        .unwrap();
        assert_eq!(recipes[0].output, conv);
        assert_eq!(recipes[0].words[64], 0);
        assert!(!fixed_depthwise_bias(
            ShaderKind::DepthwiseNative3,
            STORAGE,
            &recipes[0].words
        ));
    }
}

fn context() -> Arc<GpuContext> {
    GpuContext::new_with_options(GpuContextOptions::default()).unwrap()
}

#[test]
#[ignore = "requires physical Metal GPU; actual Half DW3 bias-only pipeline in native graph"]
fn gpu_hardware_half_depthwise_bias_production() {
    {
        for case in [
            Case {
                shape: [2, 17, 3, 17],
                stride: [1; 2],
                pads: [1; 4],
            },
            Case {
                shape: [2, 7, 1, 27],
                stride: [1, 2],
                pads: [1, 1, 1, 0],
            },
        ] {
            let context = context();
            let compiled = Arc::new(
                GpuCompiledModel::compile_with_options(
                    graph(case, Usage::Bias),
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
            let mut session = GpuSession::new_resident(
                compiled.clone(),
                &case.shape,
                ExecutionLimits::default(),
                None,
            )
            .unwrap();
            assert_eq!(session.recipes.len(), 1);
            let recipe = &session.recipes[0];
            assert!(matches!(
                recipe.kind,
                RecipeKind::Compute(ShaderKind::DepthwiseNative3)
            ));
            assert!(fixed_depthwise_bias(
                ShaderKind::DepthwiseNative3,
                STORAGE,
                &recipe.words
            ));
            let generic = compiled
                .pipelines
                .get(&context, ShaderKind::DepthwiseNative3, STORAGE)
                .unwrap();
            let actual = compiled
                .pipelines
                .get_for_recipe(
                    &context,
                    ShaderKind::DepthwiseNative3,
                    STORAGE,
                    &recipe.words,
                )
                .unwrap();
            assert_ne!(
                actual, generic,
                "actual production recipe must use its bias-only PSO"
            );
            let (input, _, _) = case.values();
            context
                .upload_resident(
                    &session.resident_input().buffer,
                    &GpuPrecision::MixedF16.encode(&input),
                )
                .unwrap();
            let mut batch = Batch::new(context.clone()).unwrap();
            let output = batch
                .graph(&mut session, "Half DW3 bias-only production")
                .unwrap();
            let expected = oracle(case);
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
            assert_eq!(
                actual, expected,
                "dense dyadics and Float accumulator witness must be Half-exact"
            );
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires physical Metal GPU; Half DW3 vector/scalar channel bias and guarded offset stores"]
fn gpu_hardware_half_depthwise_bias_raw_offsets_and_scalar_windows() {
    let context = context();
    let case = Case {
        shape: [2, 7, 7, 27],
        stride: [1, 2],
        pads: [0, 1, 2, 0],
    };
    let expected = oracle(case);
    let (input, weights, bias) = case.values();
    let mut words = case.words();
    let (x, y) = dispatch_grid(
        case.shape[0] * case.shape[1] * case.output()[2] * case.output()[3].div_ceil(4),
        &context.limits,
    )
    .unwrap();
    words[1] = x * 64;
    for variant in 0..4 {
        let mut source = if variant % 2 == 0 {
            depthwise_source(3)
        } else {
            depthwise_bias_source()
        };
        if variant >= 2 {
            source = source.replace(
                "if (ox + 4u <= ow && ix0 >= 0",
                "if (false && ox + 4u <= ow && ix0 >= 0",
            );
        }
        let mut batch = Batch::new(context.clone()).unwrap();
        let unused_c = vec![f32::NAN; case.shape[1]];
        let unused_e = vec![f32::NAN; expected.len()];
        let mut operands = Vec::new();
        for values in [&input, &weights, &unused_c, &bias, &unused_e] {
            let bytes = GpuPrecision::MixedF16.encode(values);
            let buffer = batch.storage("Half DW3 operand", bytes.len()).unwrap();
            context.upload_resident(&buffer, &bytes).unwrap();
            operands.push(buffer);
        }
        let prefix = 7;
        let suffix = 11;
        let initial = vec![-777.0; prefix + expected.len() + suffix];
        let output = batch
            .storage(
                "Half DW3 guarded output",
                GpuPrecision::MixedF16.bytes(initial.len()),
            )
            .unwrap();
        context
            .upload_resident(&output, &GpuPrecision::MixedF16.encode(&initial))
            .unwrap();
        batch
            .dispatch_msl(
                "half_dw3_bias_contract",
                backend::test_support::MetalTestShader {
                    source: &source,
                    entry_point: "depthwise_native",
                    workgroup_size: [64, 1, 1],
                },
                &[
                    (&operands[0], true),
                    (&operands[1], true),
                    (&operands[2], true),
                    (&operands[3], true),
                    (&operands[4], true),
                    (&output, false),
                ],
                &words,
                [x, y, 1],
            )
            .unwrap();
        let (bytes, stats) = batch.read(&output, output.bytes(), false).unwrap();
        assert_eq!(stats.native_resident_dispatches, 1);
        let mut actual = initial.clone();
        GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
        assert_eq!(&actual[..prefix], &initial[..prefix]);
        assert_eq!(
            &actual[prefix + expected.len()..],
            &initial[prefix + expected.len()..]
        );
        assert_eq!(&actual[prefix..prefix + expected.len()], expected);
    }
}
