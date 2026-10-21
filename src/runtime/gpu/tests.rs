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
            let context = GpuContext::new().expect("explicit hardware tests require a working GPU");
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
        let error = validate_supported(&program)
            .err()
            .expect("unsupported GPU operator must fail before device access");
        assert!(error.to_string().contains(op), "{error}");
        assert_eq!(model.cached_packed_weight_bytes(), 0);
    }
    for op in ["Sub", "Div"] {
        let model = binary_model(op, &[2, 3], &[3], false);
        let program = Program::compile(model.clone()).unwrap();
        assert!(validate_supported(&program).is_err());
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
        memory.device_activation_bytes + memory.readback_bytes + memory.host_output_bytes
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
