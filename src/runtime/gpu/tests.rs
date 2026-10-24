use super::*;
use crate::model::{LoadOptions, OptimizationPolicy};
use crate::runtime::{
    BindScratch, CompileOptions, CpuCompiledModel, CpuSession, Program, ShapePlan, bind_into,
};
use std::sync::OnceLock;

// Small ONNX protobuf fixtures keep these tests independent of downloads and of
// third-party model generators. Preserve arithmetic so every requested operator
// actually reaches GPU compilation.
fn varint(mut value: u64) -> Vec<u8> {
    let mut result = Vec::new();
    loop {
        let byte = (value & 127) as u8;
        value >>= 7;
        result.push(byte | if value == 0 { 0 } else { 128 });
        if value == 0 {
            return result;
        }
    }
}

fn number(field: u64, value: u64) -> Vec<u8> {
    [varint(field << 3), varint(value)].concat()
}

fn bytes(field: u64, value: &[u8]) -> Vec<u8> {
    [
        varint(field << 3 | 2),
        varint(value.len() as u64),
        value.to_vec(),
    ]
    .concat()
}

fn info(name: &str, shape: &[usize]) -> Vec<u8> {
    let dimensions = shape
        .iter()
        .flat_map(|&dimension| bytes(1, &number(1, dimension as u64)))
        .collect::<Vec<_>>();
    [
        bytes(1, name.as_bytes()),
        bytes(
            2,
            &bytes(1, &[number(1, 1), bytes(2, &dimensions)].concat()),
        ),
    ]
    .concat()
}

fn tensor(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
    assert_eq!(shape.iter().product::<usize>(), values.len());
    bytes(
        5,
        &[
            shape
                .iter()
                .flat_map(|&dimension| number(1, dimension as u64))
                .collect(),
            number(2, 1),
            bytes(8, name.as_bytes()),
            bytes(
                9,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        ]
        .concat(),
    )
}

fn integers(name: &str, values: &[usize]) -> Vec<u8> {
    bytes(
        5,
        &[
            bytes(1, name.as_bytes()),
            number(20, 7),
            values
                .iter()
                .flat_map(|&value| number(8, value as u64))
                .collect(),
        ]
        .concat(),
    )
}

fn integer(name: &str, value: usize) -> Vec<u8> {
    bytes(
        5,
        &[
            bytes(1, name.as_bytes()),
            number(20, 2),
            number(3, value as u64),
        ]
        .concat(),
    )
}

fn node(op: &str, inputs: &[&str], output: &str, attributes: &[Vec<u8>]) -> Vec<u8> {
    bytes(
        1,
        &[
            inputs
                .iter()
                .flat_map(|name| bytes(1, name.as_bytes()))
                .collect(),
            bytes(2, output.as_bytes()),
            bytes(3, format!("{op}_{output}").as_bytes()),
            bytes(4, op.as_bytes()),
            attributes.concat(),
        ]
        .concat(),
    )
}

fn model(parts: Vec<Vec<u8>>, input: &[usize], output: &[usize]) -> Arc<Model> {
    let graph = [
        parts.concat(),
        bytes(11, &info("x", input)),
        bytes(12, &info("y", output)),
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
        .expect("valid synthetic ONNX fixture"),
    )
}

fn values(length: usize, seed: usize) -> Vec<f32> {
    (0..length)
        .map(|index| ((index * 37 + seed * 19) % 127) as f32 / 63.0 - 1.0)
        .collect()
}

fn hardware() -> Arc<GpuContext> {
    static CONTEXT: OnceLock<Arc<GpuContext>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            let backend = match std::env::var("SIMD_OCR_GPU_BACKEND").as_deref() {
                Ok("auto") => GpuBackendPreference::Auto,
                Ok("dx12") => GpuBackendPreference::Dx12,
                Ok("metal") => GpuBackendPreference::Metal,
                Ok("vulkan") | Err(_) => GpuBackendPreference::Vulkan,
                Ok(other) => panic!("unsupported SIMD_OCR_GPU_BACKEND={other}"),
            };
            let context = GpuContext::new_with_options(GpuContextOptions {
                backend,
                ..Default::default()
            })
            .expect("explicit hardware tests require a working GPU");
            let adapter = context.adapter_info();
            eprintln!(
                "GPU correctness adapter: {} ({:?}, {:?})",
                adapter.name, adapter.backend, adapter.device_type
            );
            assert!(
                matches!(
                    adapter.device_type,
                    wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
                ),
                "hardware tests require a physical GPU, got {adapter:?}",
            );
            context
        })
        .clone()
}

fn reference(model: Arc<Model>, shape: &[usize], input: &[f32]) -> (Shape, Vec<f32>) {
    let compiled = Arc::new(CpuCompiledModel::compile(model, CompileOptions::reference()).unwrap());
    let mut session = CpuSession::new_serial(compiled, shape, ExecutionLimits::default()).unwrap();
    let output = session.run(input).unwrap();
    (output.shape(), output.data().to_vec())
}

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

fn compare(model: Arc<Model>, shape: &[usize], label: &str) {
    let compiler = crate::runtime::Compiler::new(crate::runtime::CompilerOptions {
        gpu: Some(hardware()),
        ..Default::default()
    });
    assert_eq!(compiler.cached_packed_weight_bytes(), 0);
    let compiled = compiler
        .compile(
            model.clone(),
            CompileOptions::default(),
            crate::ExecutionTarget::Gpu,
        )
        .unwrap()
        .as_gpu()
        .unwrap()
        .clone();
    assert_eq!(
        compiler.cached_packed_weight_bytes(),
        0,
        "GPU compilation must not prepare CPU packs"
    );
    let mut session = GpuSession::new(compiled, shape, ExecutionLimits::default()).unwrap();
    assert_eq!(session.input_shape().dimensions(), shape);
    for seed in 1..=2 {
        let input = values(shape.iter().product(), seed);
        let (output_shape, expected) = reference(model.clone(), shape, &input);
        assert_eq!(session.output_shape(), output_shape, "{label}");
        let output = session.run(&input).unwrap();
        assert_eq!(output.shape(), output_shape, "{label}");
        assert_close(label, output.data(), &expected);
    }
}

#[derive(Clone, Copy)]
struct ConvCase {
    name: &'static str,
    input: [usize; 4],
    channels: usize,
    kernel: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    pads: [usize; 4],
    group: usize,
    bias: bool,
}

impl ConvCase {
    fn output(self) -> [usize; 4] {
        let extent = |axis: usize| {
            (self.input[axis + 2] + self.pads[axis] + self.pads[axis + 2]
                - (self.kernel[axis] - 1) * self.dilation[axis]
                - 1)
                / self.stride[axis]
                + 1
        };
        [self.input[0], self.channels, extent(0), extent(1)]
    }

    fn model(self) -> Arc<Model> {
        let weights_shape = [
            self.channels,
            self.input[1] / self.group,
            self.kernel[0],
            self.kernel[1],
        ];
        let mut parts = vec![tensor(
            "weights",
            &weights_shape,
            &values(weights_shape.iter().product(), 7),
        )];
        let mut inputs = vec!["x", "weights"];
        if self.bias {
            parts.push(tensor("bias", &[self.channels], &values(self.channels, 11)));
            inputs.push("bias");
        }
        parts.push(node(
            "Conv",
            &inputs,
            "y",
            &[
                integers("kernel_shape", &self.kernel),
                integers("strides", &self.stride),
                integers("dilations", &self.dilation),
                integers("pads", &self.pads),
                integer("group", self.group),
            ],
        ));
        model(parts, &self.input, &self.output())
    }
}

#[test]
#[ignore = "requires a physical GPU; run explicitly with --ignored --nocapture"]
fn gpu_hardware_convolution_matches_scalar() {
    let base = ConvCase {
        name: "pointwise tail",
        input: [1, 2, 5, 7],
        channels: 3,
        kernel: [1, 1],
        stride: [1, 1],
        dilation: [1, 1],
        pads: [0; 4],
        group: 1,
        bias: true,
    };
    let cases = [
        base,
        ConvCase {
            name: "3x3 borders no bias",
            kernel: [3, 3],
            pads: [1; 4],
            bias: false,
            ..base
        },
        ConvCase {
            name: "batch two stride two",
            input: [2, 2, 7, 9],
            kernel: [3, 3],
            stride: [2, 2],
            pads: [1; 4],
            ..base
        },
        ConvCase {
            name: "dilation two asymmetric padding",
            input: [1, 2, 7, 8],
            kernel: [3, 3],
            dilation: [2, 2],
            pads: [1, 2, 3, 0],
            ..base
        },
        ConvCase {
            name: "grouped",
            input: [2, 4, 5, 7],
            channels: 6,
            kernel: [3, 3],
            pads: [1; 4],
            group: 2,
            ..base
        },
        ConvCase {
            name: "depthwise",
            input: [2, 3, 5, 7],
            channels: 3,
            kernel: [3, 3],
            pads: [1; 4],
            group: 3,
            ..base
        },
        ConvCase {
            name: "depthwise multiplier rectangular kernel",
            input: [1, 3, 5, 7],
            channels: 6,
            kernel: [2, 3],
            stride: [2, 1],
            pads: [0, 2, 1, 0],
            group: 3,
            bias: false,
            ..base
        },
    ];
    for case in cases {
        compare(case.model(), &case.input, case.name);
    }
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_specialized_tiles_tails_and_long_reductions() {
    for (rows, inner, columns) in [(7, 19, 23), (35, 33, 67), (35, 33, 129), (65, 33, 129)] {
        let shape = [2, 1, rows, inner];
        let model = model(
            vec![
                tensor(
                    "rhs",
                    &[1, 3, inner, columns],
                    &values(3 * inner * columns, 5),
                ),
                node("MatMul", &["x", "rhs"], "y", &[]),
            ],
            &shape,
            &[2, 3, rows, columns],
        );
        compare_paths(model, &shape, "tiled broadcast matmul");
    }
    for (kernel, channels, group, stride) in [
        (1, 17, 1, 1),
        (3, 35, 1, 2),
        (3, 65, 1, 1),
        (3, 9, 9, 1),
        (5, 9, 9, 2),
    ] {
        let case = ConvCase {
            name: "specialized conv",
            input: [3, 9, 27, 35],
            channels,
            kernel: [kernel; 2],
            stride: [stride; 2],
            dilation: [1; 2],
            pads: [kernel / 2; 4],
            group,
            bias: true,
        };
        compare_paths(case.model(), &case.input, case.name);
    }
    for columns in [63, 64, 65, 4095, 4096, 4097, 18710] {
        for softmax in [false, true] {
            let shape = [3, columns];
            let attrs = if softmax {
                vec![integer("axis", 1)]
            } else {
                vec![integers("axes", &[1]), integer("keepdims", 1)]
            };
            let graph = model(
                vec![node(
                    if softmax { "Softmax" } else { "ReduceMean" },
                    &["x"],
                    "y",
                    &attrs,
                )],
                &shape,
                &[3, if softmax { columns } else { 1 }],
            );
            compare_paths(graph, &shape, "parallel reduction boundary");
        }
    }
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_dense_two_by_two_padding_and_tails() {
    let mixed_context = GpuContext::new_with_options(GpuContextOptions {
        backend: hardware().options().backend,
        shader_f16: true,
        ..Default::default()
    })
    .unwrap();
    for (input, channels, stride, pads) in [
        ([2, 17, 27, 35], 35, [1, 1], [0, 0, 1, 1]),
        ([3, 33, 27, 67], 65, [2, 1], [0, 0, 1, 1]),
        ([1, 65, 31, 69], 129, [1, 2], [1, 0, 0, 1]),
    ] {
        let case = ConvCase {
            name: "dense 2x2 asymmetric padding",
            input,
            channels,
            kernel: [2; 2],
            stride,
            dilation: [1; 2],
            pads,
            group: 1,
            bias: true,
        };
        let compiled = Arc::new(GpuCompiledModel::compile(case.model(), hardware()).unwrap());
        let session = GpuSession::new(compiled, &input, ExecutionLimits::default()).unwrap();
        assert!(session.recipes.iter().any(|r| matches!(
            r.kind,
            operators::RecipeKind::Compute(
                operators::ShaderKind::Conv16 | operators::ShaderKind::Conv32
            )
        )));
        compare_paths(case.model(), &input, case.name);
        let data = values(input.iter().product(), 3);
        let (_, expected) = reference(case.model(), &input, &data);
        let compiled = Arc::new(
            GpuCompiledModel::compile_with_options(
                case.model(),
                mixed_context.clone(),
                super::super::CompileOptions {
                    gpu: GpuCompileOptions {
                        precision: GpuPrecision::MixedF16,
                    },
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let mut mixed = GpuSession::new(compiled, &input, ExecutionLimits::default()).unwrap();
        let actual = mixed.run(&data).unwrap();
        assert_eq!(actual.data().len(), expected.len());
        for (&a, &b) in actual.data().iter().zip(&expected) {
            assert!(a.is_finite() && (a - b).abs() <= 0.05 + 0.01 * b.abs());
        }
    }
}

fn compare_paths(model: Arc<Model>, shape: &[usize], label: &str) {
    let input = values(shape.iter().product(), 3);
    let (_, expected) = reference(model.clone(), shape, &input);
    let compiled = Arc::new(GpuCompiledModel::compile(model, hardware()).unwrap());
    let mut fast = GpuSession::new(compiled.clone(), shape, ExecutionLimits::default()).unwrap();
    let mut generic =
        GpuSession::new_unoptimized(compiled, shape, ExecutionLimits::default()).unwrap();
    assert_close(label, fast.run(&input).unwrap().data(), &expected);
    assert_close(label, generic.run(&input).unwrap().data(), &expected);
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_layer_norm_closed_group_and_selected_output() {
    for escape in [false, true] {
        let mut parts = vec![
            tensor("two", &[], &[2.]),
            tensor("eps", &[], &[1e-5]),
            tensor("gamma", &[65], &vec![0.8; 65]),
            tensor("beta", &[65], &vec![0.2; 65]),
            node(
                "ReduceMean",
                &["x"],
                "mean",
                &[integers("axes", &[1]), integer("keepdims", 1)],
            ),
            node("Sub", &["x", "mean"], "centered", &[]),
            node("Pow", &["centered", "two"], "square", &[]),
            node(
                "ReduceMean",
                &["square"],
                "variance",
                &[integers("axes", &[1]), integer("keepdims", 1)],
            ),
            node("Add", &["variance", "eps"], "stable", &[]),
            node("Sqrt", &["stable"], "root", &[]),
            node("Div", &["centered", "root"], "normalized", &[]),
            node("Mul", &["normalized", "gamma"], "scaled", &[]),
            node(
                "Add",
                &["scaled", "beta"],
                if escape { "ln" } else { "y" },
                &[],
            ),
        ];
        if escape {
            parts.push(node("Add", &["ln", "centered"], "y", &[]));
        }
        let graph = model(parts, &[3, 65], &[3, 65]);
        compare_paths(graph.clone(), &[3, 65], "centered LayerNorm");
        let compiled = Arc::new(GpuCompiledModel::compile(graph, hardware()).unwrap());
        let session =
            GpuSession::new(compiled.clone(), &[3, 65], ExecutionLimits::default()).unwrap();
        assert_eq!(
            session
                .statistics()
                .specialized_kernels
                .iter()
                .any(|s| s.family == "layer_norm"),
            !escape
        );
        let prefix =
            GpuSession::new_prefix(compiled, &[3, 65], ExecutionLimits::default(), 4).unwrap();
        assert_eq!(prefix.statistics().fused_nodes, 0);
    }
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_physical_fusion_preserves_order_escapes_and_trace() {
    // The residual producer is deliberately between Conv and its Add consumer.
    for escape in [false, true] {
        let graph = model(
            vec![
                tensor("w", &[2, 2, 1, 1], &[1., 0., 0., 1.]),
                tensor("s", &[], &[0.5]),
                node(
                    "Conv",
                    &["x", "w"],
                    "conv",
                    &[integers("kernel_shape", &[1, 1])],
                ),
                node("Mul", &["x", "s"], "residual", &[]),
                node("Relu", &["conv"], "active", &[]),
                node("Add", &["active", "residual"], "sum", &[]),
                node(
                    "Add",
                    &["sum", if escape { "conv" } else { "residual" }],
                    "y",
                    &[],
                ),
            ],
            &[1, 2, 3, 5],
            &[1, 2, 3, 5],
        );
        compare_paths(graph.clone(), &[1, 2, 3, 5], "ordered residual epilogue");
        let compiled = Arc::new(GpuCompiledModel::compile(graph, hardware()).unwrap());
        let mut session =
            GpuSession::new(compiled.clone(), &[1, 2, 3, 5], ExecutionLimits::default()).unwrap();
        assert_eq!(session.statistics().fused_nodes > 0, !escape);
        let mut seen = Vec::new();
        session
            .run_with_trace(&values(30, 3), |node| {
                seen.push(node.index);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            seen,
            (0..compiled.program.instructions.len()).collect::<Vec<_>>()
        );
        assert!(
            session
                .run_with_trace(&values(30, 3), |_| Err(RuntimeError::new("cancel")))
                .is_err()
        );
        session.run(&values(30, 3)).unwrap();
    }
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_profile_and_warmed_shape_bindings_are_bounded() {
    let compiled = Arc::new(GpuCompiledModel::compile(dynamic_relu_model(), hardware()).unwrap());
    let mut session = GpuSession::new(compiled, &[1, 1024], ExecutionLimits::default()).unwrap();
    for width in 1..=12 {
        session.reshape(&[1, width * 32]).unwrap();
        session.run(&values(width * 32, 1)).unwrap();
    }
    assert_eq!(session.cache.len(), 8);
    for width in [128, 256, 128, 256] {
        session.reshape(&[1, width]).unwrap();
        session.run(&values(width, 2)).unwrap();
    }
    let arena = session.parameters.buffer.clone();
    let buffers: Vec<_> = session.buffers.iter().map(|b| b.buffer.clone()).collect();
    let bindings: Vec<_> = session
        .dispatches
        .iter()
        .filter_map(|d| {
            if let Command::Compute { bind_group, .. } = &d.command {
                Some(bind_group.clone())
            } else {
                None
            }
        })
        .collect();
    session.reshape(&[1, 128]).unwrap();
    session.reshape(&[1, 256]).unwrap();
    assert_eq!(arena, session.parameters.buffer);
    assert_eq!(
        buffers,
        session
            .buffers
            .iter()
            .map(|b| b.buffer.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        bindings,
        session
            .dispatches
            .iter()
            .filter_map(
                |d| if let Command::Compute { bind_group, .. } = &d.command {
                    Some(bind_group.clone())
                } else {
                    None
                }
            )
            .collect::<Vec<_>>()
    );
    assert_eq!(session.memory.profiling_bytes, 0);
    #[cfg(feature = "profiling")]
    {
        session.run_profiled(&values(256, 2)).unwrap();
        let profile = session.last_gpu_profile().unwrap();
        assert!(!profile.kernels.is_empty());
        if hardware()
            .timestamp_features
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            assert!(profile.compute_ns().is_some_and(|ns| ns > 0));
            assert!(session.memory.profiling_bytes > 0);
        }
        let retained = session.memory.profiling_bytes;
        session.run(&values(256, 2)).unwrap();
        assert!(session.last_gpu_profile().is_none());
        assert_eq!(session.memory.profiling_bytes, retained);
    }
}

fn binary_model(op: &str, input: &[usize], constant: &[usize], constant_first: bool) -> Arc<Model> {
    let output = Shape::new(input)
        .unwrap()
        .broadcast(Shape::new(constant).unwrap())
        .unwrap();
    let inputs = if constant_first {
        ["constant", "x"]
    } else {
        ["x", "constant"]
    };
    model(
        vec![
            tensor("constant", constant, &values(constant.iter().product(), 5)),
            node(op, &inputs, "y", &[]),
        ],
        input,
        output.dimensions(),
    )
}

#[test]
#[ignore = "requires a physical GPU; run explicitly with --ignored --nocapture"]
fn gpu_hardware_add_and_mul_broadcast_match_scalar() {
    let cases: &[(&str, &[usize], &[usize])] = &[
        ("same shape", &[1, 3, 5, 7], &[1, 3, 5, 7]),
        ("scalar", &[1, 3, 5, 7], &[]),
        ("channel", &[2, 3, 5, 7], &[1, 3, 1, 1]),
        ("trailing axis", &[2, 3, 5, 7], &[7]),
        ("input scalar", &[], &[3, 5, 7]),
        (
            "rank eight",
            &[2, 1, 2, 1, 2, 1, 2, 3],
            &[1, 2, 1, 2, 1, 2, 1, 3],
        ),
    ];
    for op in ["Add", "Mul"] {
        for &(name, input, constant) in cases {
            for first in [false, true] {
                compare(
                    binary_model(op, input, constant, first),
                    input,
                    &format!("{op} {name} constant_first={first}"),
                );
            }
        }
    }
}

fn branch_model() -> Arc<Model> {
    model(
        vec![
            tensor("weights", &[3, 2, 1, 1], &values(6, 3)),
            tensor("bias", &[3], &values(3, 4)),
            tensor("scale", &[1, 3, 1, 1], &[0.5, -0.25, 0.75]),
            node(
                "Conv",
                &["x", "weights", "bias"],
                "feature",
                &[integers("kernel_shape", &[1, 1])],
            ),
            node("Relu", &["feature"], "positive", &[]),
            node("Mul", &["feature", "scale"], "scaled", &[]),
            node("Add", &["positive", "scaled"], "joined", &[]),
            node("Relu", &["joined"], "y", &[]),
        ],
        &[1, 2, 5, 7],
        &[1, 3, 5, 7],
    )
}

#[test]
#[ignore = "requires a physical GPU; run explicitly with --ignored --nocapture"]
fn gpu_hardware_branch_rejoin_repeated_runs_and_shared_sessions() {
    let model = branch_model();
    let compiler = crate::runtime::Compiler::new(crate::runtime::CompilerOptions {
        gpu: Some(hardware()),
        ..Default::default()
    });
    let compiled = compiler
        .compile(
            model.clone(),
            CompileOptions::default(),
            crate::ExecutionTarget::Gpu,
        )
        .unwrap()
        .as_gpu()
        .unwrap()
        .clone();
    assert_eq!(compiler.cached_packed_weight_bytes(), 0);
    let shape = [1, 2, 5, 7];
    let mut first = GpuSession::new(compiled.clone(), &shape, ExecutionLimits::default()).unwrap();
    let mut second = GpuSession::new(compiled, &shape, ExecutionLimits::default()).unwrap();
    for seed in 0..3 {
        let input = values(70, seed);
        let (_, expected) = reference(model.clone(), &shape, &input);
        assert_close(
            "first shared session",
            first.run(&input).unwrap().data(),
            &expected,
        );
        let other = values(70, seed + 20);
        let (_, expected_other) = reference(model.clone(), &shape, &other);
        assert_close(
            "second shared session",
            second.run(&other).unwrap().data(),
            &expected_other,
        );
        assert_close(
            "first session after sibling",
            first.run(&input).unwrap().data(),
            &expected,
        );
    }
    assert!(first.run(&[0.0; 69]).is_err());
    assert!(first.run(&[0.0; 71]).is_err());
    let input = values(70, 33);
    let (_, expected) = reference(model, &shape, &input);
    assert_close(
        "valid after rejected input",
        first.run(&input).unwrap().data(),
        &expected,
    );
}

#[test]
#[ignore = "requires a physical GPU; run explicitly with --ignored --nocapture"]
fn gpu_hardware_relu_comparison_semantics() {
    let model = model(vec![node("Relu", &["x"], "y", &[])], &[7], &[7]);
    let compiled = Arc::new(GpuCompiledModel::compile(model.clone(), hardware()).unwrap());
    let mut session = GpuSession::new(compiled, &[7], ExecutionLimits::default()).unwrap();
    let input = [-1.0, -0.0, 0.0, 1.0, -1e-20, 1e-20, 0.25];
    let output = session.run(&input).unwrap();
    assert_eq!(output.data(), &[0.0, 0.0, 0.0, 1.0, 0.0, 1e-20, 0.25]);
    assert_eq!(output.data()[1].to_bits(), 0.0f32.to_bits());
    for invalid in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
        let mut invalid_input = input;
        invalid_input[4] = invalid;
        assert!(session.run(&invalid_input).is_err());
    }
    assert_eq!(
        session.run(&input).unwrap().data(),
        &[0.0, 0.0, 0.0, 1.0, 0.0, 1e-20, 0.25]
    );
}

fn bind_program(
    program: &Program,
    shape: &[usize],
    limits: ExecutionLimits,
) -> RuntimeResult<ShapePlan> {
    let mut plan = ShapePlan::empty(
        program.model.graph().tensors.len(),
        program.instructions.len(),
    );
    let mut scratch = BindScratch::new(program);
    bind_into(
        program,
        shape,
        limits,
        program.instructions.len(),
        program.output,
        &mut plan,
        &mut scratch,
    )?;
    Ok(plan)
}

#[test]
fn supported_operators_are_validated_without_gpu_or_cpu_packing() {
    let compiler = crate::runtime::Compiler::default();
    for model in [
        branch_model(),
        binary_model("Add", &[2, 3], &[3], true),
        binary_model("Mul", &[2, 3], &[], false),
    ] {
        let program = compiler.resources(model.clone()).unwrap().graph.clone();
        validate_supported(&program).unwrap();
        assert_eq!(compiler.cached_packed_weight_bytes(), 0);
    }
    for op in ["Sigmoid", "Sqrt", "Erf"] {
        let model = model(vec![node(op, &["x"], "y", &[])], &[2, 3], &[2, 3]);
        let program = compiler.resources(model.clone()).unwrap().graph.clone();
        validate_supported(&program).unwrap();
        assert_eq!(compiler.cached_packed_weight_bytes(), 0);
    }
    for op in ["Sub", "Div"] {
        let model = binary_model(op, &[2, 3], &[3], false);
        let program = compiler.resources(model.clone()).unwrap().graph.clone();
        validate_supported(&program).unwrap();
        assert_eq!(compiler.cached_packed_weight_bytes(), 0);
    }
}

#[test]
fn gpu_binding_rejects_incompatible_broadcast_and_convolution_geometry() {
    let bad_broadcast = model(
        vec![
            tensor("constant", &[3], &[1.0; 3]),
            node("Add", &["x", "constant"], "y", &[]),
        ],
        &[1, 3, 5, 7],
        &[1, 3, 5, 7],
    );
    let program = Program::compile(bad_broadcast).unwrap();
    validate_supported(&program).unwrap();
    let error = bind_program(&program, &[1, 3, 5, 7], ExecutionLimits::default())
        .err()
        .unwrap();
    assert!(error.to_string().contains("broadcast"), "{error}");

    // Deliberately provide three input channels of coefficients for a two-channel input.
    let bad_channels = model(
        vec![
            tensor("weights", &[2, 3, 3, 3], &[0.25; 54]),
            node(
                "Conv",
                &["x", "weights"],
                "y",
                &[integers("kernel_shape", &[3, 3])],
            ),
        ],
        &[1, 2, 5, 5],
        &[1, 2, 3, 3],
    );
    let program = Program::compile(bad_channels).unwrap();
    let error = bind_program(&program, &[1, 2, 5, 5], ExecutionLimits::default())
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("channels/groups/kernel"),
        "{error}"
    );

    let bad_bias = model(
        vec![
            tensor("weights", &[2, 2, 1, 1], &[0.25; 4]),
            tensor("bias", &[1], &[0.1]),
            node(
                "Conv",
                &["x", "weights", "bias"],
                "y",
                &[integers("kernel_shape", &[1, 1])],
            ),
        ],
        &[1, 2, 5, 5],
        &[1, 2, 5, 5],
    );
    let program = Program::compile(bad_bias).unwrap();
    let error = bind_program(&program, &[1, 2, 5, 5], ExecutionLimits::default())
        .err()
        .unwrap();
    assert!(error.to_string().contains("bias"), "{error}");

    let oversized_kernel = model(
        vec![
            tensor("weights", &[1, 1, 3, 3], &[0.25; 9]),
            node(
                "Conv",
                &["x", "weights"],
                "y",
                &[integers("kernel_shape", &[3, 3])],
            ),
        ],
        &[1, 1, 2, 2],
        &[1, 1, 1, 1],
    );
    let program = Program::compile(oversized_kernel).unwrap();
    assert!(bind_program(&program, &[1, 1, 2, 2], ExecutionLimits::default()).is_err());
}

#[test]
fn gpu_plan_validates_device_storage_and_activation_limits_without_allocating() {
    let program = Program::compile(branch_model()).unwrap();
    let plan = bind_program(&program, &[1, 2, 5, 7], ExecutionLimits::default()).unwrap();
    validate_plan(&program, &plan, &wgpu::Limits::default()).unwrap();
    assert!(
        bind_program(
            &program,
            &[1, 2, 5, 7],
            ExecutionLimits {
                max_activation_bytes: 1
            }
        )
        .is_err()
    );
    assert!(
        bind_program(
            &program,
            &[1, 2, 5, 7],
            ExecutionLimits {
                max_activation_bytes: 0
            }
        )
        .is_err()
    );
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: 16,
        max_buffer_size: 16,
        ..Default::default()
    };
    assert!(validate_plan(&program, &plan, &limits).is_err());
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: 64,
        max_buffer_size: 64,
        ..Default::default()
    };
    assert_eq!(checked_storage_bytes(16, &limits).unwrap(), 64);
    assert!(checked_storage_bytes(17, &limits).is_err());
    assert!(checked_storage_bytes(usize::MAX, &wgpu::Limits::default()).is_err());
    assert!(checked_storage_bytes(u32::MAX as usize + 1, &wgpu::Limits::default()).is_err());
}

#[test]
fn gpu_large_shape_index_range_is_rejected_without_allocating() {
    let shape = [1usize, u32::MAX as usize + 1];
    let model = model(vec![node("Relu", &["x"], "y", &[])], &shape, &shape);
    let program = Program::compile(model).unwrap();
    let plan = bind_program(
        &program,
        &shape,
        ExecutionLimits {
            max_activation_bytes: usize::MAX,
        },
    )
    .unwrap();
    assert!(validate_plan(&program, &plan, &wgpu::Limits::default()).is_err());
}

#[test]
#[ignore = "requires a physical GPU; run explicitly with --ignored --nocapture"]
fn gpu_hardware_memory_accounting_and_budget_boundaries() {
    let compiled = Arc::new(GpuCompiledModel::compile(branch_model(), hardware()).unwrap());
    assert!(compiled.weight_bytes() > 0);
    let shape = [1, 2, 5, 7];
    let session = GpuSession::new(compiled.clone(), &shape, ExecutionLimits::default()).unwrap();
    let memory = session.memory_statistics();
    assert!(memory.device_activation_bytes > 0);
    assert!(
        memory.device_activation_bytes < (70 + 5 * 105) * 4,
        "branch lifetimes should reuse device activation storage"
    );
    assert!(memory.readback_bytes >= 105 * 4);
    assert!(memory.host_output_bytes >= 105 * 4);
    assert_eq!(
        memory.total_bytes(),
        memory.device_activation_bytes
            + memory.readback_bytes
            + memory.host_output_bytes
            + memory.host_input_bytes
            + memory.diagnostic_bytes
            + memory.parameter_bytes
            + memory.profiling_bytes
            + memory.host_parameter_bytes
            + memory.upload_bytes
    );
    assert_eq!(session.activation_bytes(), memory.total_bytes());
    assert!(
        GpuSession::new(
            compiled.clone(),
            &shape,
            ExecutionLimits {
                max_activation_bytes: memory.total_bytes() - 1
            }
        )
        .is_err()
    );
    let mut exact = GpuSession::new(
        compiled.clone(),
        &shape,
        ExecutionLimits {
            max_activation_bytes: memory.total_bytes(),
        },
    )
    .unwrap();
    assert_eq!(
        exact.memory_statistics().total_bytes(),
        memory.total_bytes()
    );
    exact.run(&values(70, 8)).unwrap();
    assert!(
        GpuSession::new(
            compiled.clone(),
            &shape,
            ExecutionLimits {
                max_activation_bytes: 0
            }
        )
        .is_err()
    );
    assert!(GpuSession::new(compiled.clone(), &[1, 2, 5], ExecutionLimits::default()).is_err());
    assert!(GpuSession::new(compiled, &[1, 2, 5, 8], ExecutionLimits::default()).is_err());
}

#[test]
#[ignore = "requires a physical GPU; run explicitly with --ignored --nocapture"]
fn gpu_hardware_nonfinite_output_is_an_error_and_session_remains_usable() {
    let model = model(
        vec![
            tensor("large", &[], &[f32::MAX]),
            node("Mul", &["x", "large"], "y", &[]),
        ],
        &[1],
        &[1],
    );
    let compiled = Arc::new(GpuCompiledModel::compile(model, hardware()).unwrap());
    let mut session = GpuSession::new(compiled, &[1], ExecutionLimits::default()).unwrap();
    assert!(session.run(&[2.0]).is_err());
    assert_eq!(session.run(&[0.0]).unwrap().data(), &[0.0]);
}

#[test]
fn gpu_dispatch_grid_covers_tails_and_rejects_device_dimension_overflow() {
    let limits = wgpu::Limits {
        max_compute_workgroups_per_dimension: 2,
        ..Default::default()
    };
    assert_eq!(dispatch_grid(1, &limits).unwrap(), (1, 1));
    assert_eq!(dispatch_grid(WORKGROUP_SIZE + 1, &limits).unwrap(), (2, 1));
    assert_eq!(dispatch_grid(4 * WORKGROUP_SIZE, &limits).unwrap(), (2, 2));
    assert!(dispatch_grid(4 * WORKGROUP_SIZE + 1, &limits).is_err());
    assert!(dispatch_grid(usize::MAX, &limits).is_err());
}

fn int_tensor(name: &str, shape: &[usize], values: &[i64]) -> Vec<u8> {
    bytes(
        5,
        &[
            shape.iter().flat_map(|&n| number(1, n as u64)).collect(),
            number(2, 7),
            bytes(8, name.as_bytes()),
            bytes(
                9,
                &values
                    .iter()
                    .flat_map(|n| n.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        ]
        .concat(),
    )
}
fn float_list(name: &str, values: &[f32]) -> Vec<u8> {
    bytes(
        5,
        &[
            bytes(1, name.as_bytes()),
            number(20, 6),
            values
                .iter()
                .flat_map(|v| [varint(7 << 3 | 5), v.to_le_bytes().to_vec()].concat())
                .collect(),
        ]
        .concat(),
    )
}
fn string_attribute(name: &str, value: &str) -> Vec<u8> {
    bytes(
        5,
        &[
            bytes(1, name.as_bytes()),
            number(20, 3),
            bytes(4, value.as_bytes()),
        ]
        .concat(),
    )
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_extended_arithmetic_and_batched_matmul_match_scalar() {
    let arithmetic = model(
        vec![
            tensor("two", &[], &[2.]),
            tensor("one", &[], &[1.]),
            node("Pow", &["x", "two"], "squared", &[]),
            node("Add", &["squared", "one"], "positive", &[]),
            node("Sqrt", &["positive"], "root", &[]),
            node("Div", &["root", "two"], "divided", &[]),
            node("Sub", &["divided", "one"], "shifted", &[]),
            node("Erf", &["shifted"], "erf", &[]),
            node("Sigmoid", &["erf"], "sigmoid", &[]),
            node("HardSigmoid", &["sigmoid"], "y", &[]),
        ],
        &[2, 3, 5],
        &[2, 3, 5],
    );
    compare(arithmetic, &[2, 3, 5], "extended arithmetic");
    let matrix = model(
        vec![
            node(
                "Transpose",
                &["x"],
                "rhs",
                &[integers("perm", &[0, 1, 3, 2])],
            ),
            node("MatMul", &["x", "rhs"], "product", &[]),
            node("Softmax", &["product"], "y", &[integer("axis", 3)]),
        ],
        &[2, 3, 4, 5],
        &[2, 3, 4, 4],
    );
    compare(
        matrix,
        &[2, 3, 4, 5],
        "activation batched matmul and softmax",
    );
    let shared = model(
        vec![
            tensor("weights", &[5, 7], &values(35, 8)),
            node("MatMul", &["x", "weights"], "y", &[]),
        ],
        &[2, 3, 5],
        &[2, 3, 7],
    );
    compare(shared, &[2, 3, 5], "shared RHS matmul");
    let broadcast = model(
        vec![
            tensor("rhs", &[1, 5, 4, 2], &values(40, 4)),
            node("MatMul", &["x", "rhs"], "y", &[]),
        ],
        &[2, 1, 3, 4],
        &[2, 5, 3, 2],
    );
    compare(
        broadcast,
        &[2, 1, 3, 4],
        "unequal broadcast batches on both operands",
    );
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_extended_spatial_and_reduction_match_scalar() {
    let spatial = model(
        vec![
            tensor("scale", &[2], &[1.25, 0.75]),
            tensor("bias", &[2], &[0.1, -0.2]),
            tensor("mean", &[2], &[0.2, 0.3]),
            tensor("variance", &[2], &[0.5, 1.5]),
            tensor("weights", &[2, 3, 2, 2], &values(24, 9)),
            node(
                "BatchNormalization",
                &["x", "scale", "bias", "mean", "variance"],
                "normalized",
                &[],
            ),
            node(
                "AveragePool",
                &["normalized"],
                "average",
                &[
                    integers("kernel_shape", &[3, 2]),
                    integers("strides", &[3, 2]),
                ],
            ),
            node(
                "MaxPool",
                &["average"],
                "pooled",
                &[
                    integers("kernel_shape", &[2, 2]),
                    integers("strides", &[1, 1]),
                    integers("pads", &[0, 0, 1, 1]),
                ],
            ),
            node(
                "Resize",
                &["pooled"],
                "resized",
                &[
                    float_list("scales", &[1., 1., 2., 2.]),
                    string_attribute("coordinate_transformation_mode", "asymmetric"),
                    string_attribute("nearest_mode", "floor"),
                ],
            ),
            node(
                "ConvTranspose",
                &["resized", "weights"],
                "transposed",
                &[
                    integers("kernel_shape", &[2, 2]),
                    integers("strides", &[2, 2]),
                ],
            ),
            node(
                "ReduceMean",
                &["transposed"],
                "reduced",
                &[integers("axes", &[3]), integer("keepdims", 1)],
            ),
            node("GlobalAveragePool", &["reduced"], "y", &[]),
        ],
        &[1, 2, 6, 8],
        &[1, 3, 1, 1],
    );
    compare(spatial, &[1, 2, 6, 8], "spatial and reduction chain");
    let generic = model(
        vec![node(
            "ReduceMean",
            &["x"],
            "y",
            &[integers("axes", &[0, 2]), integer("keepdims", 0)],
        )],
        &[2, 3, 5],
        &[3],
    );
    compare(generic, &[2, 3, 5], "noncontiguous reduction axes");
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_extended_shapes_slice_and_concat_match_scalar() {
    let model = model(
        vec![
            int_tensor("shape", &[3], &[1, 3, 8]),
            int_tensor("starts", &[1], &[0]),
            int_tensor("ends", &[1], &[2]),
            int_tensor("axes", &[1], &[1]),
            node("Squeeze", &["x"], "squeezed", &[integers("axes", &[0])]),
            node(
                "Unsqueeze",
                &["squeezed"],
                "unsqueezed",
                &[integers("axes", &[0])],
            ),
            node(
                "Transpose",
                &["unsqueezed"],
                "permuted",
                &[integers("perm", &[0, 2, 3, 1])],
            ),
            node("Reshape", &["permuted", "shape"], "reshaped", &[]),
            node(
                "Slice",
                &["reshaped", "starts", "ends", "axes"],
                "sliced",
                &[],
            ),
            node("Concat", &["sliced", "sliced"], "y", &[integer("axis", 1)]),
        ],
        &[1, 2, 3, 4],
        &[1, 4, 8],
    );
    let compiled = Arc::new(GpuCompiledModel::compile(model.clone(), hardware()).unwrap());
    let mut session = GpuSession::new(compiled, &[1, 2, 3, 4], ExecutionLimits::default()).unwrap();
    for seed in [1, 2] {
        let mut input = values(24, seed);
        input[0] = -0.0;
        input[1] = 0.0;
        let (_, expected) = reference(model.clone(), &[1, 2, 3, 4], &input);
        let actual = session.run(&input).unwrap();
        assert_eq!(
            actual
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "movement kernels must preserve every value bit"
        );
    }
}

fn dynamic_relu_model() -> Arc<Model> {
    let dynamic_info = |name: &str| {
        let dimensions = [bytes(1, &[]), bytes(1, &[])].concat();
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
        node("Relu", &["x"], "y", &[]),
        bytes(11, &dynamic_info("x")),
        bytes(12, &dynamic_info("y")),
    ]
    .concat();
    Arc::new(
        Model::load_bytes(&[number(1, 8), bytes(8, &number(2, 13)), bytes(7, &graph)].concat())
            .unwrap(),
    )
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_reshape_cache_compaction_rollback_and_trace_reentry() {
    let context = hardware();
    let model = dynamic_relu_model();
    let compiled = Arc::new(GpuCompiledModel::compile(model, context.clone()).unwrap());
    let mut session =
        GpuSession::new(compiled.clone(), &[1, 64], ExecutionLimits::default()).unwrap();
    let minimum = session.activation_bytes();
    eprintln!("reshape memory: narrow={:?}", session.memory_statistics());
    session.reshape(&[1, 512]).unwrap();
    eprintln!("reshape memory: wide={:?}", session.memory_statistics());
    let expected = vec![0.25; 512];
    assert_eq!(session.run(&expected).unwrap().data(), &expected);
    session.reshape(&[1, 64]).unwrap();
    eprintln!(
        "reshape memory: reused narrow={:?}",
        session.memory_statistics()
    );
    assert!(session.activation_bytes() > minimum);
    session
        .reshape_with_limits(
            &[1, 64],
            ExecutionLimits {
                max_activation_bytes: minimum,
            },
        )
        .unwrap();
    assert_eq!(session.activation_bytes(), minimum);
    eprintln!(
        "reshape memory: compacted narrow={:?}",
        session.memory_statistics()
    );
    let previous_shape = session.input_shape();
    let retained = context.memory_statistics().used_bytes;
    assert!(
        session
            .reshape_with_limits(
                &[1, 64],
                ExecutionLimits {
                    max_activation_bytes: minimum - 1
                }
            )
            .is_err()
    );
    assert_eq!(session.input_shape(), previous_shape);
    assert_eq!(session.execution_limits().max_activation_bytes, minimum);
    assert_eq!(context.memory_statistics().used_bytes, retained);
    assert!(session.reshape(&[1, 0]).is_err());
    assert_eq!(session.run(&[-0.5; 64]).unwrap().data(), &[0.; 64]);
    session
        .reshape_with_limits(&[1, 64], ExecutionLimits::default())
        .unwrap();
    for width in 1..=10 {
        session.reshape(&[1, width]).unwrap();
    }
    assert_eq!(session.cache.len(), PLAN_CACHE_CAPACITY);
    let mut sibling = GpuSession::new(compiled, &[1, 10], ExecutionLimits::default()).unwrap();
    let mut observed = 0;
    session
        .run_with_trace(&[0.5; 10], |_| {
            observed += 1;
            assert_eq!(sibling.run(&[0.75; 10])?.data(), &[0.75; 10]);
            Ok(())
        })
        .unwrap();
    assert_eq!(observed, 1);
    assert!(
        session
            .run_with_trace(&[0.5; 10], |_| Err(RuntimeError::new("observer stopped")))
            .is_err()
    );
    assert_eq!(session.run(&[0.5; 10]).unwrap().data(), &[0.5; 10]);
}

#[test]
fn support_report_aggregates_node_variant_failures_with_lineage() {
    let fixture = model(
        vec![
            node(
                "MaxPool",
                &["x"],
                "pooled",
                &[integers("kernel_shape", &[3, 3])],
            ),
            node(
                "AveragePool",
                &["pooled"],
                "y",
                &[integers("kernel_shape", &[2, 2])],
            ),
        ],
        &[1, 1, 8, 8],
        &[1, 1, 3, 3],
    );
    let report = GpuCompiledModel::support_report(&fixture);
    assert!(!report.is_supported());
    assert_eq!(report.issues.len(), 2);
    assert_eq!(report.issues[0].operator, "MaxPool");
    assert_eq!(report.issues[1].operator, "AveragePool");
    assert!(!report.issues[0].source_indices.is_empty());
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_gate_failure_wakes_all_waiters() {
    // This context is deliberately poisoned; never use the cached test context.
    let context = GpuContext::new_with_options(GpuContextOptions {
        backend: hardware().options().backend,
        ..Default::default()
    })
    .unwrap();
    let holder = context.acquire_gate().unwrap();
    let (send, receive) = mpsc::channel();
    let mut workers = Vec::new();
    for _ in 0..2 {
        let context = context.clone();
        let send = send.clone();
        workers.push(std::thread::spawn(move || {
            let failed = context.acquire_gate().is_err();
            send.send(failed).unwrap();
        }));
    }
    drop(send);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while context.waiting.load(Ordering::Acquire) != 2 && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    let both_waiting = context.waiting.load(Ordering::Acquire) == 2;
    // Match device callbacks: recording the error itself does not notify the gate.
    *context.failure.lock().unwrap() = Some("injected callback device failure".into());
    drop(holder);
    let first = receive.recv_timeout(std::time::Duration::from_secs(2));
    let second = receive.recv_timeout(std::time::Duration::from_secs(2));
    // Ensure a regression fails with a bounded result instead of stranding a test thread.
    context.available.notify_all();
    for worker in workers {
        worker.join().unwrap();
    }
    assert!(
        both_waiting,
        "both contenders must reach the condition-variable wait"
    );
    assert!(
        matches!(first, Ok(true)),
        "first waiter did not observe context failure: {first:?}"
    );
    assert!(
        matches!(second, Ok(true)),
        "second waiter remained asleep after fatal failure: {second:?}"
    );
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_attention_se_resize_placement_and_rank_five() {
    for keys in [17, 65] {
        let shape = [2, 3, 19, 24];
        let graph = model(
            vec![
                tensor("k", &[2, 3, 24, keys], &values(6 * 24 * keys, 4)),
                tensor("v", &[2, 3, keys, 29], &values(6 * keys * 29, 5)),
                tensor("scale", &[], &[0.2]),
                node("MatMul", &["x", "k"], "qk", &[]),
                node("Mul", &["qk", "scale"], "scaled", &[]),
                node("Softmax", &["scaled"], "prob", &[integer("axis", 3)]),
                node("MatMul", &["prob", "v"], "y", &[]),
            ],
            &shape,
            &[2, 3, 19, 29],
        );
        compare_paths(graph.clone(), &shape, "online attention");
        let compiled = Arc::new(GpuCompiledModel::compile(graph, hardware()).unwrap());
        let session = GpuSession::new(compiled, &shape, ExecutionLimits::default()).unwrap();
        assert!(
            session
                .statistics()
                .specialized_kernels
                .iter()
                .any(|k| k.family == "attention_online")
        );
    }
    let shape = [2, 5, 7, 9];
    let graph = model(
        vec![
            tensor("w", &[5, 5, 1, 1], &values(25, 4)),
            node("GlobalAveragePool", &["x"], "pool", &[]),
            node(
                "Conv",
                &["pool", "w"],
                "excite",
                &[integers("kernel_shape", &[1, 1])],
            ),
            node("HardSigmoid", &["excite"], "gate", &[]),
            node("Mul", &["x", "gate"], "y", &[]),
        ],
        &shape,
        &shape,
    );
    compare_paths(graph.clone(), &shape, "squeeze excitation");
    let s = GpuSession::new(
        Arc::new(GpuCompiledModel::compile(graph, hardware()).unwrap()),
        &shape,
        ExecutionLimits::default(),
    )
    .unwrap();
    assert!(
        s.statistics()
            .specialized_kernels
            .iter()
            .any(|k| k.family == "squeeze_excitation")
    );
    let graph = model(
        vec![
            node("Relu", &["x"], "a", &[]),
            node("Sigmoid", &["x"], "b", &[]),
            node("Concat", &["a", "b"], "y", &[integer("axis", 1)]),
        ],
        &[1, 3, 5, 7],
        &[1, 6, 5, 7],
    );
    compare_paths(graph.clone(), &[1, 3, 5, 7], "concat placement");
    let s = GpuSession::new(
        Arc::new(GpuCompiledModel::compile(graph, hardware()).unwrap()),
        &[1, 3, 5, 7],
        ExecutionLimits::default(),
    )
    .unwrap();
    assert_eq!(s.dispatches.len(), 2);
    // Generic softmax reads its own output twice. Both reads and writes must
    // include the destination offset after concat places it behind another value.
    compare_paths(
        model(
            vec![
                node("Relu", &["x"], "a", &[]),
                node("Softmax", &["x"], "b", &[integer("axis", 1)]),
                node("Concat", &["a", "b"], "y", &[integer("axis", 0)]),
            ],
            &[3, 17],
            &[6, 17],
        ),
        &[3, 17],
        "softmax concat offset",
    );
    compare_paths(
        model(
            vec![node(
                "Transpose",
                &["x"],
                "y",
                &[integers("perm", &[2, 0, 3, 1, 4])],
            )],
            &[2, 3, 4, 5, 7],
            &[4, 2, 5, 3, 7],
        ),
        &[2, 3, 4, 5, 7],
        "rank-five transpose",
    );
    compare_paths(
        model(
            vec![
                tensor("residual", &[2, 3, 10, 14], &values(840, 5)),
                node(
                    "Resize",
                    &["x"],
                    "resize",
                    &[
                        float_list("scales", &[1., 1., 2., 2.]),
                        string_attribute("coordinate_transformation_mode", "asymmetric"),
                        string_attribute("nearest_mode", "floor"),
                    ],
                ),
                node("Add", &["resize", "residual"], "y", &[]),
            ],
            &[2, 3, 5, 7],
            &[2, 3, 10, 14],
        ),
        &[2, 3, 5, 7],
        "resize add",
    );
    compare_paths(
        model(
            vec![
                tensor("w", &[9, 11, 2, 2], &values(396, 4)),
                node(
                    "ConvTranspose",
                    &["x", "w"],
                    "y",
                    &[
                        integers("kernel_shape", &[2, 2]),
                        integers("strides", &[2, 2]),
                    ],
                ),
            ],
            &[2, 9, 7, 9],
            &[2, 11, 14, 18],
        ),
        &[2, 9, 7, 9],
        "transposed tile tails",
    );
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_resident_fusions_and_experimental_selection() {
    let shape = [2, 3, 19, 24];
    let graph = model(
        vec![
            tensor("k", &[2, 3, 19, 24], &values(2 * 3 * 19 * 24, 4)),
            tensor("v", &[2, 3, 24, 19], &values(2 * 3 * 24 * 19, 5)),
            node(
                "Transpose",
                &["k"],
                "kt",
                &[integers("perm", &[0, 1, 3, 2])],
            ),
            node(
                "Transpose",
                &["v"],
                "vt",
                &[integers("perm", &[0, 1, 3, 2])],
            ),
            node("MatMul", &["x", "kt"], "scores", &[]),
            node("Softmax", &["scores"], "prob", &[integer("axis", 3)]),
            node("MatMul", &["prob", "vt"], "y", &[]),
        ],
        &shape,
        &shape,
    );
    compare_paths(graph.clone(), &shape, "folded QKV address transforms");
    let session = GpuSession::new(
        Arc::new(GpuCompiledModel::compile(graph, hardware()).unwrap()),
        &shape,
        ExecutionLimits::default(),
    )
    .unwrap();
    assert_eq!(session.dispatches.len(), 1);
    let shape = [2, 17, 32, 33];
    let graph = model(
        vec![
            tensor("w1", &[5, 17, 1, 1], &values(85, 2)),
            tensor("b1", &[5], &values(5, 3)),
            tensor("w2", &[17, 5, 1, 1], &values(85, 4)),
            tensor("b2", &[17], &values(17, 5)),
            tensor("w3", &[19, 17, 1, 1], &values(323, 6)),
            node("GlobalAveragePool", &["x"], "pool", &[]),
            node(
                "Conv",
                &["pool", "w1", "b1"],
                "fc1",
                &[integers("kernel_shape", &[1, 1])],
            ),
            node("Relu", &["fc1"], "relu", &[]),
            node(
                "Conv",
                &["relu", "w2", "b2"],
                "fc2",
                &[integers("kernel_shape", &[1, 1])],
            ),
            node("HardSigmoid", &["fc2"], "gate", &[]),
            node("Mul", &["x", "gate"], "scaled", &[]),
            node(
                "Conv",
                &["scaled", "w3"],
                "y",
                &[integers("kernel_shape", &[1, 1])],
            ),
        ],
        &shape,
        &[2, 19, 32, 33],
    );
    compare_paths(graph.clone(), &shape, "SE chain and prescaling");
    let session = GpuSession::new(
        Arc::new(GpuCompiledModel::compile(graph, hardware()).unwrap()),
        &shape,
        ExecutionLimits::default(),
    )
    .unwrap();
    assert!(
        session
            .statistics()
            .specialized_kernels
            .iter()
            .any(|k| k.family == "se_projection_chain")
    );
    assert_eq!(session.dispatches.len(), 3);
    let context = GpuContext::new_with_options(GpuContextOptions {
        backend: hardware().options().backend,
        experimental_cooperative_matrix: true,
        ..Default::default()
    })
    .unwrap();
    let shape = [256, 17];
    let graph = model(
        vec![
            tensor("w", &[17, 129], &values(2193, 3)),
            node("MatMul", &["x", "w"], "y", &[]),
        ],
        &shape,
        &[256, 129],
    );
    let input = values(256 * 17, 1);
    let (_, expected) = reference(graph.clone(), &shape, &input);
    let mut session = GpuSession::new(
        Arc::new(GpuCompiledModel::compile(graph, context.clone()).unwrap()),
        &shape,
        ExecutionLimits::default(),
    )
    .unwrap();
    assert_close(
        "experimental or portable fallback",
        session.run(&input).unwrap().data(),
        &expected,
    );
    assert_eq!(
        session
            .statistics()
            .specialized_kernels
            .iter()
            .any(|k| k.family == "experimental_cooperative_matmul8"),
        context
            .device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    );
    eprintln!(
        "Cooperative physical support: {}",
        context
            .device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    );
}

#[test]
#[ignore = "requires FP16 capable physical GPU"]
fn gpu_hardware_nhwc_pointwise_region_preserves_boundaries() {
    let context = GpuContext::new_with_options(GpuContextOptions {
        backend: hardware().options().backend,
        shader_f16: true,
        ..Default::default()
    })
    .unwrap();
    let shape = [1, 65, 32, 33];
    let graph = model(
        vec![
            tensor("w1", &[67, 65, 1, 1], &values(4355, 3)),
            tensor("w2", &[69, 67, 1, 1], &values(4623, 4)),
            node(
                "Conv",
                &["x", "w1"],
                "first",
                &[integers("kernel_shape", &[1, 1])],
            ),
            node("Relu", &["first"], "activation", &[]),
            node(
                "Conv",
                &["activation", "w2"],
                "y",
                &[integers("kernel_shape", &[1, 1])],
            ),
        ],
        &shape,
        &[1, 69, 32, 33],
    );
    let input = values(shape.iter().product(), 1);
    let (_, expected) = reference(graph.clone(), &shape, &input);
    let compiled = Arc::new(
        GpuCompiledModel::compile_with_options(
            graph,
            context,
            super::super::CompileOptions {
                gpu: GpuCompileOptions {
                    precision: GpuPrecision::MixedF16,
                },
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let mut session =
        GpuSession::new(compiled.clone(), &shape, ExecutionLimits::default()).unwrap();
    let actual = session.run(&input).unwrap();
    for (&a, &b) in actual.data().iter().zip(&expected) {
        assert!((a - b).abs() <= 0.05 + 0.01 * b.abs(), "{a} vs {b}");
    }
    assert_eq!(session.statistics().layout_conversions, 1);
    let mut prefix =
        GpuSession::new_prefix(compiled, &shape, ExecutionLimits::default(), 1).unwrap();
    prefix.run(&input).unwrap();
    assert_eq!(prefix.output_shape().dimensions(), &[1, 67, 32, 33]);
}

#[test]
#[ignore = "requires FP16 and subgroup capable physical GPU"]
fn gpu_hardware_mixed_storage_and_optional_subgroups() {
    for precision in [GpuPrecision::F32, GpuPrecision::MixedF16] {
        let context = GpuContext::new_with_options(GpuContextOptions {
            shader_f16: precision == GpuPrecision::MixedF16,
            subgroups: true,
            backend: hardware().options().backend,
            ..Default::default()
        })
        .unwrap();
        // Repeated half stores destroy small residuals even with F32 arithmetic.
        // The unfused path must retain F32 activations while reading shared half
        // constants; fusion is not required for its numerical correctness.
        let shape = [3, 65];
        let graph = model(
            vec![
                tensor("offset", &[], &[1000.]),
                node("Add", &["x", "offset"], "large", &[]),
                node("Sub", &["large", "offset"], "y", &[]),
            ],
            &shape,
            &shape,
        );
        let compiled = Arc::new(
            GpuCompiledModel::compile_with_options(
                graph.clone(),
                context.clone(),
                super::super::CompileOptions {
                    gpu: GpuCompileOptions { precision },
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let mut unfused =
            GpuSession::new_unoptimized(compiled, &shape, ExecutionLimits::default()).unwrap();
        assert_eq!(unfused.input_precision(), GpuPrecision::F32);
        let input = vec![0.123; 195];
        assert_close(
            "unfused residual precision",
            unfused.run(&input).unwrap().data(),
            &reference(graph, &shape, &input).1,
        );
        for columns in [65, 4097] {
            let shape = [3, columns];
            let graph = model(
                vec![node("Softmax", &["x"], "y", &[integer("axis", 1)])],
                &shape,
                &shape,
            );
            let input = values(3 * columns, 1);
            let (_, expected) = reference(graph.clone(), &shape, &input);
            let compiled = Arc::new(
                GpuCompiledModel::compile_with_options(
                    graph,
                    context.clone(),
                    super::super::CompileOptions {
                        gpu: GpuCompileOptions { precision },
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
            for fusion in [false, true] {
                let mut session = if fusion {
                    GpuSession::new(compiled.clone(), &shape, ExecutionLimits::default())
                } else {
                    GpuSession::new_unoptimized(
                        compiled.clone(),
                        &shape,
                        ExecutionLimits::default(),
                    )
                }
                .unwrap();
                let actual = session.run(&input).unwrap();
                for (&a, &b) in actual.data().iter().zip(&expected) {
                    assert!(
                        a.is_finite() && (a - b).abs() <= 5e-4 + 1e-2 * b.abs(),
                        "{a} != {b}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a physical GPU; injects device failure into a private context"]
fn gpu_hardware_strict_auto_callbacks_and_concurrent_sessions() {
    use crate::runtime::{ExecutionBackend, ExecutionTarget, Session};
    let context = GpuContext::new_with_options(GpuContextOptions {
        backend: hardware().options().backend,
        max_in_flight: 2,
        ..Default::default()
    })
    .unwrap();
    let model = branch_model();
    let shape = [1, 2, 5, 7];
    let compiler = crate::runtime::Compiler::new(crate::runtime::CompilerOptions {
        gpu: Some(context.clone()),
        ..Default::default()
    });
    let compile = |target| {
        compiler
            .compile(model.clone(), CompileOptions::default(), target)
            .unwrap()
    };
    let mut strict = Session::new_serial(
        compile(ExecutionTarget::Gpu),
        &shape,
        ExecutionLimits::default(),
    )
    .unwrap();
    let auto_model = compile(ExecutionTarget::Auto);
    let weight_bytes = auto_model.as_gpu().unwrap().weight_bytes();
    let before = context.memory_statistics().used_bytes;
    let duplicate = compile(ExecutionTarget::Gpu);
    assert_eq!(context.memory_statistics().used_bytes, before);
    assert_eq!(duplicate.as_gpu().unwrap().weight_bytes(), weight_bytes);
    let mut auto =
        Session::new_serial(auto_model.clone(), &shape, ExecutionLimits::default()).unwrap();
    assert_eq!(compiler.cached_packed_weight_bytes(), 0);
    let mut callbacks = 0;
    let error = auto
        .run_with_input(|_| {
            callbacks += 1;
            Err(RuntimeError::with_kind(
                ErrorKind::DeviceLost,
                "callback failure",
            ))
        })
        .unwrap_err();
    assert_eq!(callbacks, 1);
    assert_eq!(error.origin(), crate::error::ErrorOrigin::Callback);
    assert_eq!(auto.execution_backend(), ExecutionBackend::Gpu);
    let input = values(70, 1);
    std::thread::scope(|scope| {
        let a = scope.spawn(|| strict.run(&input).unwrap().data().to_vec());
        let b = scope.spawn(|| auto.run(&input).unwrap().data().to_vec());
        assert_close("concurrent outputs", &a.join().unwrap(), &b.join().unwrap());
    });
    assert_eq!(strict.statistics().production_submissions, 1);
    assert_eq!(strict.statistics().production_readbacks, 1);
    assert_eq!(strict.statistics().diagnostic_readbacks, 0);
    assert!(
        auto.reshape_with_limits(
            &shape,
            ExecutionLimits {
                max_activation_bytes: 1
            }
        )
        .is_err()
    );
    assert_eq!(auto.execution_backend(), ExecutionBackend::Gpu);
    context.fail("injected loss".into());
    assert_eq!(
        strict.run(&input).unwrap_err().kind(),
        ErrorKind::DeviceLost
    );
    let output = auto.run(&input).unwrap().data().to_vec();
    assert!(matches!(auto.execution_backend(), ExecutionBackend::Cpu(_)));
    assert!(
        auto.fallback_error()
            .unwrap()
            .to_string()
            .contains("injected loss")
    );
    assert!(
        compiler.cached_packed_weight_bytes() > 0,
        "fallback must compile CPU packs lazily"
    );
    assert_close(
        "lazy fallback",
        &output,
        &reference(model, &shape, &input).1,
    );
}

#[test]
#[ignore = "explicit bounded performance check; three inputs, one warmup, three measured calls"]
fn gpu_bounded_performance() {
    let mut cases = Vec::new();
    for (kernel, cin, cout, precision) in [
        (1, 65, 67, GpuPrecision::MixedF16),
        (3, 32, 64, GpuPrecision::F32),
    ] {
        let case = ConvCase {
            name: "measure",
            input: [1, cin, 32, 33],
            channels: cout,
            kernel: [kernel; 2],
            stride: [1; 2],
            dilation: [1; 2],
            pads: [kernel / 2; 4],
            group: 1,
            bias: true,
        };
        cases.push((
            if kernel == 1 { "pointwise" } else { "dense" },
            case.model(),
            case.input.to_vec(),
            precision,
        ));
    }
    let shape = [1, 4, 128, 32];
    cases.push((
        "attention",
        model(
            vec![
                tensor("k", &[1, 4, 32, 128], &values(4 * 32 * 128, 4)),
                tensor("v", &shape, &values(4 * 128 * 32, 5)),
                node("MatMul", &["x", "k"], "qk", &[]),
                node("Softmax", &["qk"], "p", &[integer("axis", 3)]),
                node("MatMul", &["p", "v"], "y", &[]),
            ],
            &shape,
            &shape,
        ),
        shape.to_vec(),
        GpuPrecision::F32,
    ));
    for (name, graph, shape, precision) in cases {
        let context = GpuContext::new_with_options(GpuContextOptions {
            backend: hardware().options().backend,
            shader_f16: precision == GpuPrecision::MixedF16,
            ..Default::default()
        })
        .unwrap();
        let compiled = Arc::new(
            GpuCompiledModel::compile_with_options(
                graph.clone(),
                context,
                super::super::CompileOptions {
                    gpu: GpuCompileOptions { precision },
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let input = values(shape.iter().product(), 1);
        let (_, expected) = reference(graph, &shape, &input);
        for optimized in [false, true] {
            let mut session = if optimized {
                GpuSession::new(compiled.clone(), &shape, ExecutionLimits::default())
            } else {
                GpuSession::new_unoptimized(compiled.clone(), &shape, ExecutionLimits::default())
            }
            .unwrap();
            let output = session.run(&input).unwrap(); // exactly one warmup, also correctness gate
            for (&a, &b) in output.data().iter().zip(&expected) {
                let (abs, rel) = if precision == GpuPrecision::F32 {
                    (1e-3, 1e-3)
                } else {
                    (5e-2, 1e-2)
                };
                assert!(a.is_finite() && (a - b).abs() <= abs + rel * b.abs());
            }
            let mut ms = Vec::new();
            for _ in 0..3 {
                let start = std::time::Instant::now();
                session.run(&input).unwrap();
                ms.push(start.elapsed().as_secs_f64() * 1000.);
            }
            eprintln!(
                "{name} precision={precision:?} optimized={optimized} milliseconds={ms:?} kernels={:?}",
                session.statistics().selected_kernels
            );
        }
    }
}

#[test]
#[ignore = "bounded pointwise experiment: three shapes, one correctness warmup and three wall/timestamp samples per candidate"]
fn gpu_pointwise_layout_candidates() {
    use operators::ShaderKind;
    use resident::Batch;
    let context = hardware();
    assert!(
        context
            .timestamp_features
            .contains(wgpu::Features::TIMESTAMP_QUERY)
    );
    let started = std::time::Instant::now();
    for (shape, channels) in [
        ([1, 128, 32, 33], 128),
        ([1, 1536, 3, 240], 768),
        ([2, 192, 6, 136], 1024),
    ] {
        let case = ConvCase {
            name: "pointwise candidate",
            input: shape,
            channels,
            kernel: [1; 2],
            stride: [1; 2],
            dilation: [1; 2],
            pads: [0; 4],
            group: 1,
            bias: true,
        };
        let model = case.model();
        let input = values(shape.iter().product(), 1);
        let (_, expected) = reference(model.clone(), &shape, &input);
        let compiled = Arc::new(GpuCompiledModel::compile(model, context.clone()).unwrap());
        let session =
            GpuSession::new(compiled.clone(), &shape, ExecutionLimits::default()).unwrap();
        assert_eq!(session.recipes.len(), 1);
        let recipe = &session.recipes[0];
        let baseline = match recipe.kind {
            operators::RecipeKind::Compute(ShaderKind::PointwiseNchw16) => ShaderKind::Conv16,
            operators::RecipeKind::Compute(
                ShaderKind::PointwiseNchw32 | ShaderKind::PointwiseBlocked32,
            ) => ShaderKind::Conv32,
            operators::RecipeKind::Compute(kind @ (ShaderKind::Conv16 | ShaderKind::Conv32)) => {
                kind
            }
            _ => panic!("expected tiled baseline"),
        };
        let pointwise = if baseline == ShaderKind::Conv32 {
            ShaderKind::PointwiseNchw32
        } else {
            ShaderKind::PointwiseNchw16
        };
        let a = context
            .create_buffer_init(
                "candidate input",
                bytemuck::cast_slice(&input),
                wgpu::BufferUsages::STORAGE,
            )
            .unwrap();
        let weights = compiled.weights.constants[recipe.inputs[1].0]
            .as_ref()
            .unwrap();
        let bias = compiled.weights.constants[recipe.inputs[2].0]
            .as_ref()
            .unwrap();
        let dummy = &compiled.weights.dummy;
        let source = |kind: ShaderKind| {
            let kernel = kind.source();
            let mut source =
                format!("{}\n{}", include_str!("common.wgsl"), kernel).replace("$ENABLE", "");
            for name in ["$A", "$B", "$C", "$D", "$E", "$O"] {
                source = source.replace(name, "f32");
            }
            source
        };
        for kind in [baseline, pointwise, ShaderKind::PointwiseNhwc] {
            let mut samples = Vec::new();
            let mut wall_samples = Vec::new();
            for iteration in 0..4 {
                assert!(
                    started.elapsed().as_secs() < 60,
                    "bounded experiment budget exhausted"
                );
                let timer = std::time::Instant::now();
                let mut batch = Batch::new(context.clone()).unwrap();
                batch.profiling = true;
                let output = batch
                    .storage("candidate output", expected.len() * 4)
                    .unwrap();
                let mut words = recipe.words;
                let packed;
                let input = if kind == ShaderKind::PointwiseNhwc {
                    let padded = shape[1].div_ceil(16) * 16;
                    let plane = shape[2] * shape[3];
                    let count = shape[0] * plane * padded;
                    packed = batch.storage("candidate NHWC", count * 4).unwrap();
                    let (x, y) = dispatch_grid(count, &context.limits).unwrap();
                    let mut convert = [0u32; 128];
                    convert[..6].copy_from_slice(&[
                        count as u32,
                        x * 64,
                        24,
                        shape[1] as u32,
                        plane as u32,
                        padded as u32,
                    ]);
                    batch
                        .dispatch(
                            "candidate_nhwc_conversion",
                            &source(ShaderKind::General),
                            &[
                                (&a, true),
                                (dummy, true),
                                (dummy, true),
                                (dummy, true),
                                (dummy, true),
                                (&packed, false),
                            ],
                            &convert,
                            [x, y, 1],
                        )
                        .unwrap();
                    words[32] = plane.div_ceil(16) as u32;
                    words[33] = channels.div_ceil(16) as u32;
                    words[35] = shape[0] as u32;
                    words[36] = padded as u32;
                    &packed
                } else {
                    &a
                };
                let (x, y) = dispatch_grid(
                    words[32] as usize * words[33] as usize * shape[0] * 64,
                    &context.limits,
                )
                .unwrap();
                words[34] = x;
                batch
                    .dispatch(
                        kind.name(),
                        &source(kind),
                        &[
                            (input, true),
                            (weights, true),
                            (bias, true),
                            (dummy, true),
                            (dummy, true),
                            (&output, false),
                        ],
                        &words,
                        [x, y, 1],
                    )
                    .unwrap();
                let (bytes, stats) = batch.read(&output, expected.len() * 4, false).unwrap();
                if iteration == 0 {
                    let actual: Vec<f32> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect();
                    assert_close(kind.name(), &actual, &expected);
                } else {
                    wall_samples.push(timer.elapsed().as_secs_f64() * 1e6);
                    samples.push(
                        stats
                            .profile
                            .unwrap()
                            .compute_ns()
                            .map(|ns| ns as f64 / 1000.),
                    );
                }
            }
            eprintln!(
                "POINTWISE {}",
                serde_json::json!({"input":shape,"channels":channels,"kernel":kind.name(),"samples_us":samples,"wall_samples_us":wall_samples,"includes_layout_conversion":kind==ShaderKind::PointwiseNhwc})
            );
        }
    }
}

fn pointwise_register_case(shape: [usize; 4], channels: usize, bias: bool, measured: usize) {
    use operators::ShaderKind;
    let case = ConvCase {
        name: "pointwise register tiles",
        input: shape,
        channels,
        kernel: [1; 2],
        stride: [1; 2],
        dilation: [1; 2],
        pads: [0; 4],
        group: 1,
        bias,
    };
    convolution_kernel_candidates(
        case,
        &[ShaderKind::PointwiseNchw32, ShaderKind::PointwiseBlocked32],
        measured,
    );
}

fn convolution_kernel_candidates(case: ConvCase, kinds: &[operators::ShaderKind], measured: usize) {
    use operators::ShaderKind;
    let context = hardware();
    let shape = case.input;
    let channels = case.channels;
    let bias = case.bias;
    let model = case.model();
    let input = values(shape.iter().product(), 1);
    let (_, expected) = reference(model.clone(), &shape, &input);
    let compiled = Arc::new(GpuCompiledModel::compile(model, context.clone()).unwrap());
    let session = GpuSession::new(compiled.clone(), &shape, ExecutionLimits::default()).unwrap();
    let recipe = &session.recipes[0];
    let input_buffer = context
        .create_buffer_init(
            "register input",
            bytemuck::cast_slice(&input),
            wgpu::BufferUsages::STORAGE,
        )
        .unwrap();
    let dummy = &compiled.weights.dummy;
    let weights = compiled.weights.constants[recipe.inputs[1].0]
        .as_ref()
        .unwrap();
    let bias_buffer = if bias {
        compiled.weights.constants[recipe.inputs[2].0]
            .as_ref()
            .unwrap()
    } else {
        dummy
    };
    for &kind in kinds {
        let mut source =
            format!("{}\n{}", include_str!("common.wgsl"), kind.source()).replace("$ENABLE", "");
        for token in ["$A", "$B", "$C", "$D", "$E", "$O"] {
            source = source.replace(token, "f32");
        }
        let mut words = recipe.words;
        let tile = if kind == ShaderKind::Conv16 { 16 } else { 32 };
        let output_shape = case.output();
        words[32] = channels.div_ceil(tile) as u32;
        words[33] = (output_shape[2] * output_shape[3]).div_ceil(tile) as u32;
        words[35] = shape[0] as u32;
        // Exercise destination offsets as well as channel, spatial and K tails.
        let offset = 7;
        words[60] = offset as u32;
        let (x, y) = dispatch_grid(
            if kind == ShaderKind::Convolution {
                expected.len()
            } else {
                words[32] as usize * words[33] as usize * shape[0] * 64
            },
            &context.limits,
        )
        .unwrap();
        words[1] = x * 64;
        words[34] = x;
        let mut wall_samples = Vec::new();
        let mut device_samples = Vec::new();
        for iteration in 0..=measured {
            let timer = std::time::Instant::now();
            let mut batch = resident::Batch::new(context.clone()).unwrap();
            batch.profiling = measured != 0;
            let output = batch
                .storage("register output", (expected.len() + offset) * 4)
                .unwrap();
            batch
                .dispatch(
                    kind.name(),
                    &source,
                    &[
                        (&input_buffer, true),
                        (weights, true),
                        (bias_buffer, true),
                        (dummy, true),
                        (dummy, true),
                        (&output, false),
                    ],
                    &words,
                    [x, y, 1],
                )
                .unwrap();
            let (bytes, stats) = batch
                .read(&output, (expected.len() + offset) * 4, false)
                .unwrap();
            let wall = timer.elapsed().as_secs_f64() * 1e6;
            if iteration == 0 {
                assert!(bytes[..offset * 4].iter().all(|&b| b == 0));
                let actual: Vec<f32> = bytes[offset * 4..]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                assert_close(kind.name(), &actual, &expected);
            } else {
                wall_samples.push(wall);
                device_samples.push(
                    stats
                        .profile
                        .unwrap()
                        .compute_ns()
                        .map(|ns| ns as f64 / 1000.),
                );
            }
        }
        if measured != 0 {
            eprintln!(
                "CONV_CANDIDATE {}",
                serde_json::json!({"input":shape,"channels":channels,"kernel_shape":case.kernel,"kernel":kind.name(),"wall_samples_us":wall_samples,"device_samples_us":device_samples})
            );
        }
    }
}

#[test]
#[ignore = "requires a physical GPU"]
fn gpu_hardware_pointwise_register_tiles_tails_and_offsets() {
    for (shape, channels, bias) in [
        ([2, 33, 3, 37], 65, true),
        ([1, 257, 1, 19], 129, false),
        ([1, 128, 6, 136], 384, true),
    ] {
        pointwise_register_case(shape, channels, bias, 0);
    }
}

#[test]
#[ignore = "requires a physical GPU"]
fn gpu_hardware_pointwise_register_fusion_and_selection() {
    let shape = [2, 128, 3, 271];
    let output = [2, 129, 3, 271];
    let graph = model(
        vec![
            tensor("w", &[129, 128, 1, 1], &values(129 * 128, 7)),
            tensor("b", &[129], &values(129, 11)),
            tensor("residual", &[1, 129, 1, 1], &values(129, 13)),
            node(
                "Conv",
                &["x", "w", "b"],
                "conv",
                &[integers("kernel_shape", &[1, 1])],
            ),
            node("Add", &["conv", "residual"], "sum", &[]),
            node("Relu", &["sum"], "y", &[]),
        ],
        &shape,
        &output,
    );
    compare_paths(
        graph.clone(),
        &shape,
        "register blocks with bias, broadcast residual and activation",
    );
    let compiled = Arc::new(GpuCompiledModel::compile(graph, hardware()).unwrap());
    let session = GpuSession::new(compiled, &shape, ExecutionLimits::default()).unwrap();
    assert_eq!(session.dispatches.len(), 1);
    assert_eq!(
        session
            .statistics()
            .specialized_kernels
            .iter()
            .any(|k| k.family == "pointwise_blocked32"),
        hardware().adapter_info().backend == wgpu::Backend::Metal,
    );
    // Tall detector maps retain the previously qualified pointwise policy.
    let case = ConvCase {
        name: "tall pointwise",
        input: [1, 128, 8, 136],
        channels: 129,
        kernel: [1; 2],
        stride: [1; 2],
        dilation: [1; 2],
        pads: [0; 4],
        group: 1,
        bias: true,
    };
    let compiled = Arc::new(GpuCompiledModel::compile(case.model(), hardware()).unwrap());
    let session = GpuSession::new(compiled, &case.input, ExecutionLimits::default()).unwrap();
    assert!(
        !session
            .statistics()
            .specialized_kernels
            .iter()
            .any(|k| k.family == "pointwise_blocked32")
    );
}

#[test]
#[ignore = "bounded pointwise register experiment: three shapes, one correctness warmup and three measured calls"]
fn gpu_pointwise_register_performance() {
    let timer = std::time::Instant::now();
    for (shape, channels) in [
        ([1, 1536, 3, 240], 768),
        ([1, 1024, 6, 240], 512),
        ([2, 192, 6, 136], 1024),
    ] {
        pointwise_register_case(shape, channels, true, 3);
        assert!(
            timer.elapsed().as_secs() < 60,
            "bounded experiment budget exhausted"
        );
    }
}

#[test]
#[ignore = "requires a physical GPU"]
fn gpu_hardware_large_spatial_tiles_padding_and_tails() {
    use operators::ShaderKind;
    for (input, channels, kernel, stride, pads) in [
        ([2, 17, 31, 35], 33, 5, [2, 1], [1, 2, 3, 1]),
        ([1, 33, 29, 39], 65, 7, [1, 2], [2, 3, 4, 1]),
        ([1, 65, 32, 33], 64, 3, [1, 1], [1, 1, 1, 1]),
    ] {
        let case = ConvCase {
            name: "large spatial tails",
            input,
            channels,
            kernel: [kernel; 2],
            stride,
            dilation: [1; 2],
            pads,
            group: 1,
            bias: true,
        };
        convolution_kernel_candidates(
            case,
            &[
                ShaderKind::Convolution,
                ShaderKind::Conv16,
                ShaderKind::Conv32,
            ],
            0,
        );
        let compiled = Arc::new(GpuCompiledModel::compile(case.model(), hardware()).unwrap());
        let session = GpuSession::new(compiled, &input, ExecutionLimits::default()).unwrap();
        let expected = if hardware().adapter_info().backend == wgpu::Backend::Metal {
            if kernel == 3 {
                ShaderKind::Conv32
            } else {
                ShaderKind::Conv16
            }
        } else if kernel == 3 {
            ShaderKind::Conv16
        } else {
            ShaderKind::Convolution
        };
        assert!(
            matches!(session.recipes[0].kind, operators::RecipeKind::Compute(kind) if kind==expected)
        );
    }
}

#[test]
#[ignore = "bounded spatial experiment: three shapes, one correctness warmup and three measured calls"]
fn gpu_spatial_tile_performance() {
    use operators::ShaderKind;
    let timer = std::time::Instant::now();
    for (cin, channels, kernel) in [(256, 64, 3), (32, 32, 5), (32, 32, 7)] {
        let kinds = if kernel == 3 {
            &[ShaderKind::Conv16, ShaderKind::Conv32][..]
        } else {
            &[
                ShaderKind::Convolution,
                ShaderKind::Conv16,
                ShaderKind::Conv32,
            ][..]
        };
        convolution_kernel_candidates(
            ConvCase {
                name: "spatial tile candidate",
                input: [1, cin, 128, 128],
                channels,
                kernel: [kernel; 2],
                stride: [1; 2],
                dilation: [1; 2],
                pads: [kernel / 2; 4],
                group: 1,
                bias: true,
            },
            kinds,
            3,
        );
        assert!(
            timer.elapsed().as_secs() < 60,
            "bounded experiment budget exhausted"
        );
    }
}

#[test]
#[ignore = "bounded resident performance experiment; requires models and generated corpus"]
fn benchmark_resident_candidates() {
    use crate::{
        ExecutionTarget, GpuPipelineMode, ModelPreset, OcrModelSet,
        image::{ImageView, PixelFormat},
    };
    let corpus = std::env::var("SIMD_OCR_FULL_GPU_CORPUS")
        .expect("set SIMD_OCR_FULL_GPU_CORPUS to generated fixture directory");
    let models =
        OcrModelSet::from_directory("../SimdPaddleOCR/models", ModelPreset::ChineseV6Tiny, true)
            .unwrap();
    let started = std::time::Instant::now();
    for case in ["sparse", "typical", "dense"] {
        let path = std::path::Path::new(&corpus).join(format!("{case}.png"));
        let pixels = image::ImageReader::open(path)
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();
        let image = ImageView::new(
            pixels.as_raw(),
            pixels.width() as usize,
            pixels.height() as usize,
            0,
            PixelFormat::Rgb24,
        )
        .unwrap();
        let mut reference = None;
        for (cache, tiled) in [(true, true), (false, true), (false, false)] {
            let context = GpuContext::new_with_options(GpuContextOptions {
                backend: hardware().options().backend,
                cache_resident_scratch: cache,
                tiled_labels: tiled,
                ..Default::default()
            })
            .unwrap();
            let mut request = models
                .clone()
                .builder()
                .execution_target(ExecutionTarget::Gpu)
                .gpu_context(context)
                .gpu_pipeline_mode(GpuPipelineMode::Full)
                .recognition_batch_size(2)
                .build()
                .unwrap()
                .request()
                .unwrap();
            let cold = request
                .run(
                    image,
                    crate::RunOptions {
                        capture_spans: true,
                    },
                )
                .unwrap()
                .try_to_owned()
                .unwrap();
            if let Some(expected) = &reference {
                assert_eq!(&cold.text, expected);
            } else {
                reference = Some(cold.text);
            }
            let mut samples = Vec::new();
            for _ in 0..3 {
                let clock = std::time::Instant::now();
                let result = request
                    .run(
                        image,
                        crate::RunOptions {
                            capture_spans: false,
                        },
                    )
                    .unwrap();
                samples.push(clock.elapsed().as_secs_f64() * 1000.);
                assert_eq!(&result.text, reference.as_ref().unwrap());
            }
            eprintln!("RESIDENT {case} cache={cache} tiled={tiled} ms={samples:?}");
            assert!(started.elapsed().as_secs() < 60);
        }
    }
}

#[test]
#[ignore = "requires a physical GPU"]
fn gpu_hardware_full_precision_variant_shares_context_and_isolates_weights() {
    let context = GpuContext::new_with_options(GpuContextOptions {
        backend: hardware().options().backend,
        shader_f16: true,
        ..Default::default()
    })
    .unwrap();
    let (source, _) = crate::ocr::tests::synthetic_sources();
    let model = Arc::new(Model::load_bytes(&source).unwrap());
    let mixed = Arc::new(
        GpuCompiledModel::compile_with_options(
            model,
            context.clone(),
            super::super::CompileOptions {
                gpu: GpuCompileOptions {
                    precision: GpuPrecision::MixedF16,
                },
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let full = mixed.with_full_precision().unwrap();
    assert!(Arc::ptr_eq(&context, full.context()));
    assert!(Arc::ptr_eq(&full, &mixed.with_full_precision().unwrap()));
    assert_eq!(mixed.precision, GpuPrecision::MixedF16);
    assert_eq!(full.precision, GpuPrecision::F32);
    assert!(!Arc::ptr_eq(&mixed.weights, &full.weights));
    let mut session = GpuSession::new(full, &[1, 3, 32, 32], ExecutionLimits::default()).unwrap();
    assert!(
        session
            .tensor_descriptors()
            .iter()
            .all(|d| d.precision == GpuPrecision::F32)
    );
    let input = vec![0.1234567; 3 * 32 * 32];
    let expected = 1.0 / (1.0 + (-0.1234567f32).exp());
    assert!(
        session
            .run(&input)
            .unwrap()
            .data()
            .iter()
            .all(|&v| (v - expected).abs() < 1e-6)
    );
    assert_eq!(session.statistics().effective_precision, "F32");
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_resident_storage_has_no_dense_transfers_and_reshape_rolls_back() {
    let compiled = Arc::new(GpuCompiledModel::compile(branch_model(), hardware()).unwrap());
    let shape = [1, 2, 5, 7];
    let mut resident =
        GpuSession::new_resident(compiled.clone(), &shape, ExecutionLimits::default(), None)
            .unwrap();
    let memory = resident.memory_statistics();
    assert_eq!(memory.host_input_bytes, 0);
    assert_eq!(memory.host_output_bytes, 0);
    assert_eq!(memory.upload_bytes, 0);
    assert_eq!(memory.readback_bytes, 0);
    assert!(memory.device_activation_bytes > 0);
    let before = resident.input_shape();
    assert!(
        resident
            .reshape_with_limits(
                &shape,
                ExecutionLimits {
                    max_activation_bytes: 1
                }
            )
            .is_err()
    );
    assert_eq!(resident.input_shape(), before);
    assert_eq!(resident.memory_statistics(), memory);
    let mut tensor = GpuSession::new(compiled, &shape, ExecutionLimits::default()).unwrap();
    assert!(tensor.memory_statistics().host_output_bytes > 0);
    tensor.run(&values(70, 1)).unwrap();
}
