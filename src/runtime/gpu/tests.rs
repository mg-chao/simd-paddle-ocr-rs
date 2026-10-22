use super::*;
use crate::model::{LoadOptions, OptimizationPolicy};
use crate::runtime::{
    BindScratch, CompileOptions, CompiledModel, Program, Session, ShapePlan, bind_into,
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
    let compiled = Arc::new(CompiledModel::compile(model, CompileOptions::reference()).unwrap());
    let mut session = Session::new_serial(compiled, shape, ExecutionLimits::default()).unwrap();
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
    assert_eq!(model.cached_packed_weight_bytes(), 0);
    let compiled = Arc::new(GpuCompiledModel::compile(model.clone(), hardware()).unwrap());
    assert_eq!(
        model.cached_packed_weight_bytes(),
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
    let compiled = Arc::new(GpuCompiledModel::compile(model.clone(), hardware()).unwrap());
    assert_eq!(model.cached_packed_weight_bytes(), 0);
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
    for model in [
        branch_model(),
        binary_model("Add", &[2, 3], &[3], true),
        binary_model("Mul", &[2, 3], &[], false),
    ] {
        let program = Program::compile(model.clone()).unwrap();
        validate_supported(&program).unwrap();
        assert_eq!(model.cached_packed_weight_bytes(), 0);
    }
    for op in ["Sigmoid", "Sqrt", "Erf"] {
        let model = model(vec![node(op, &["x"], "y", &[])], &[2, 3], &[2, 3]);
        let program = Program::compile(model.clone()).unwrap();
        validate_supported(&program).unwrap();
        assert_eq!(model.cached_packed_weight_bytes(), 0);
    }
    for op in ["Sub", "Div"] {
        let model = binary_model(op, &[2, 3], &[3], false);
        let program = Program::compile(model.clone()).unwrap();
        validate_supported(&program).unwrap();
        assert_eq!(model.cached_packed_weight_bytes(), 0);
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
            + memory.head_workspace_bytes
            + memory.diagnostic_bytes
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
    assert_eq!(
        dispatch_grid(WORKGROUP_SIZE as usize + 1, &limits).unwrap(),
        (2, 1)
    );
    assert_eq!(
        dispatch_grid(4 * WORKGROUP_SIZE as usize, &limits).unwrap(),
        (2, 2)
    );
    assert!(dispatch_grid(4 * WORKGROUP_SIZE as usize + 1, &limits).is_err());
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
