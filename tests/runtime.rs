use simd_paddle_ocr_rs::{
    model::Model,
    runtime::{CompiledModel, ExecutionLimits, RuntimeError, Session},
    tensor::{Shape, TensorView},
};
use std::sync::Arc;

fn varint(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let low = (value & 127) as u8;
        value >>= 7;
        bytes.push(low | if value == 0 { 0 } else { 128 });
        if value == 0 {
            return bytes;
        }
    }
}
fn number(field: u64, value: i64) -> Vec<u8> {
    [varint(field << 3), varint(value as u64)].concat()
}
fn bytes(field: u64, value: Vec<u8>) -> Vec<u8> {
    [varint(field << 3 | 2), varint(value.len() as u64), value].concat()
}
fn string(field: u64, value: &str) -> Vec<u8> {
    bytes(field, value.as_bytes().to_vec())
}
fn info(name: &str, dims: &[i64]) -> Vec<u8> {
    [
        string(1, name),
        bytes(
            2,
            bytes(
                1,
                [
                    number(1, 1),
                    bytes(
                        2,
                        dims.iter()
                            .flat_map(|&d| {
                                bytes(
                                    1,
                                    if d < 0 {
                                        string(2, "opaque batch symbol")
                                    } else {
                                        number(1, d)
                                    },
                                )
                            })
                            .collect(),
                    ),
                ]
                .concat(),
            ),
        ),
    ]
    .concat()
}
fn tensor(name: &str, dims: &[i64], values: &[f32]) -> Vec<u8> {
    bytes(
        5,
        [
            dims.iter().flat_map(|&d| number(1, d)).collect(),
            number(2, 1),
            string(8, name),
            bytes(9, values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ]
        .concat(),
    )
}
fn integers(name: &str, values: &[i64]) -> Vec<u8> {
    bytes(
        5,
        [
            number(1, values.len() as i64),
            number(2, 7),
            string(8, name),
            bytes(9, values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ]
        .concat(),
    )
}
fn ints(name: &str, values: &[i64]) -> Vec<u8> {
    [
        string(1, name),
        number(20, 7),
        values.iter().flat_map(|&v| number(8, v)).collect(),
    ]
    .concat()
}
fn int(name: &str, value: i64) -> Vec<u8> {
    [string(1, name), number(20, 2), number(3, value)].concat()
}
fn float(name: &str, value: f32) -> Vec<u8> {
    [
        string(1, name),
        number(20, 1),
        varint(2 << 3 | 5),
        value.to_le_bytes().to_vec(),
    ]
    .concat()
}
fn node(name: &str, op: &str, inputs: &[&str], output: &str, attrs: Vec<Vec<u8>>) -> Vec<u8> {
    bytes(
        1,
        [
            inputs.iter().flat_map(|v| string(1, v)).collect(),
            string(2, output),
            string(3, name),
            string(4, op),
            attrs.into_iter().flat_map(|v| bytes(5, v)).collect(),
        ]
        .concat(),
    )
}
fn model_bytes(records: Vec<Vec<u8>>, input: &[i64], output: &[i64]) -> Vec<u8> {
    [
        number(1, 8),
        bytes(8, number(2, 7)),
        bytes(
            7,
            [
                records.concat(),
                bytes(11, info("x", input)),
                bytes(12, info("y", output)),
            ]
            .concat(),
        ),
    ]
    .concat()
}
fn model(records: Vec<Vec<u8>>, input: &[i64], output: &[i64]) -> Arc<Model> {
    Arc::new(Model::load_bytes(&model_bytes(records, input, output)).unwrap())
}
fn session(model: Arc<Model>, shape: &[usize]) -> Session {
    Session::new(
        Arc::new(CompiledModel::new(model).unwrap()),
        shape,
        ExecutionLimits::default(),
    )
    .unwrap()
}
fn classifier() -> Arc<Model> {
    model(
        vec![
            tensor("w", &[2, 1, 1, 1], &[1., -1.]),
            tensor("bias", &[2], &[0., 2.]),
            integers("minus_one", &[-1]),
            tensor("identity", &[2, 2], &[1., 0., 0., 1.]),
            tensor("head_bias", &[2], &[0.0625, -0.1875]),
            node(
                "convolution",
                "Conv",
                &["x", "w", "bias"],
                "conv",
                vec![ints("kernel_shape", &[1, 1])],
            ),
            node("relu", "Relu", &["conv"], "relu", vec![]),
            node(
                "gate",
                "HardSigmoid",
                &["relu"],
                "gate",
                vec![float("alpha", 0.25), float("beta", 0.5)],
            ),
            node("activation", "Mul", &["relu", "gate"], "activation", vec![]),
            node(
                "average",
                "ReduceMean",
                &["activation"],
                "mean",
                vec![ints("axes", &[2, 3]), int("keepdims", 1)],
            ),
            node("shape", "Shape", &["mean"], "shape", vec![]),
            node(
                "batch",
                "Slice",
                &["shape"],
                "batch",
                vec![ints("starts", &[0]), ints("ends", &[1]), ints("axes", &[0])],
            ),
            node(
                "target",
                "Concat",
                &["batch", "minus_one"],
                "target",
                vec![int("axis", 0)],
            ),
            node("flatten", "Reshape", &["mean", "target"], "flat", vec![]),
            node("head", "MatMul", &["flat", "identity"], "logits", vec![]),
            node("bias", "Add", &["logits", "head_bias"], "biased", vec![]),
            node(
                "probabilities",
                "Softmax",
                &["biased"],
                "y",
                vec![int("axis", 1)],
            ),
        ],
        &[-1, 1, 2, 2],
        &[-1, 2],
    )
}

#[test]
fn all_nine_ops_dynamic_shape_program_traces_and_repeated_runs() {
    let loaded = classifier();
    let compiled = Arc::new(CompiledModel::new(loaded).unwrap());
    let mut request =
        Session::new(compiled.clone(), &[1, 1, 2, 2], ExecutionLimits::default()).unwrap();
    let mut other = Session::new(compiled, &[1, 1, 2, 2], ExecutionLimits::default()).unwrap();
    let input = [1., -2., 3., 4.];
    let mut trace = Vec::new();
    let output = request
        .run_with_trace(&input, |node| {
            trace.push((
                node.index,
                node.key.to_owned(),
                node.shape,
                node.layout,
                node.values.to_vec(),
            ));
            Ok(())
        })
        .unwrap();
    assert_eq!(output.shape().dimensions(), &[1, 2]);
    assert!((output.data()[0] - 0.7310586).abs() < 1e-7);
    assert!((output.data()[1] - 0.26894143).abs() < 1e-7);
    let expected = output.data().to_vec();
    assert_eq!(trace.len(), 9);
    assert_eq!(
        trace.iter().map(|v| v.0).collect::<Vec<_>>(),
        (0..9).collect::<Vec<_>>()
    );
    assert_eq!(trace[5].1, "node:8");
    assert_eq!(trace[5].2.dimensions(), &[1, 2]);
    assert_eq!(trace[4].4, vec![1.9375, 1.1875]);
    assert_eq!(trace[0].3, "nchw");
    assert_eq!(trace[8].3, "row_major");
    assert!(request.activation_bytes() < request.naive_activation_bytes());
    for _ in 0..8 {
        assert_eq!(request.run(&input).unwrap().data(), expected);
    }
    let second_input = [4., 2., -3., 1.];
    let second_expected = other.run(&second_input).unwrap().data().to_vec();
    assert_ne!(expected, second_expected);
    request.reshape(&[2, 1, 2, 2]).unwrap();
    assert_eq!(request.output_shape().dimensions(), &[2, 2]);
    assert_eq!(
        request.run(&[input, second_input].concat()).unwrap().data(),
        [expected.clone(), second_expected.clone()].concat()
    );
    request.reshape(&[1, 1, 2, 2]).unwrap();
    assert_eq!(request.run(&input).unwrap().data(), expected);
    assert_eq!(other.run(&second_input).unwrap().data(), second_expected);
    assert_eq!(other.run(&input).unwrap().data(), expected);
}

#[test]
fn pointwise_tiling_preserves_bias_order_groups_batches_and_tail_channels() {
    let loaded = model(
        vec![
            tensor("w", &[7, 2, 1, 1], &[1.; 14]),
            tensor("b", &[7], &[1e8; 7]),
            node(
                "conv",
                "Conv",
                &["x", "w", "b"],
                "y",
                vec![ints("kernel_shape", &[1, 1])],
            ),
        ],
        &[1, 2, 1, 1],
        &[1, 7, 1, 1],
    );
    assert_eq!(
        session(loaded, &[1, 2, 1, 1])
            .run(&[-1e8, 1.])
            .unwrap()
            .data(),
        &[1.; 7]
    );

    let weights: Vec<_> = (1..=14).flat_map(|v| [v as f32, -(v as f32)]).collect();
    let bias: Vec<_> = (1..=14).map(|v| v as f32).collect();
    let loaded = model(
        vec![
            tensor("w", &[14, 2, 1, 1], &weights),
            tensor("b", &[14], &bias),
            node(
                "conv",
                "Conv",
                &["x", "w", "b"],
                "y",
                vec![ints("kernel_shape", &[1, 1]), int("group", 2)],
            ),
        ],
        &[2, 4, 1, 2],
        &[2, 14, 1, 2],
    );
    let input = [
        1., 2., 3., 4., 10., 20., 30., 40., 2., 4., 6., 8., 20., 40., 60., 80.,
    ];
    // Bias is channel index + 1. Input-channel differences are -2/-20
    // in the first batch and -4/-40 in the second batch.
    let expected: Vec<_> = (1..=7)
        .flat_map(|co| [-(co as f32); 2])
        .chain((8..=14).flat_map(|co| [-19. * co as f32; 2]))
        .chain((1..=7).flat_map(|co| [-3. * co as f32; 2]))
        .chain((8..=14).flat_map(|co| [-39. * co as f32; 2]))
        .collect();
    assert_eq!(
        session(loaded, &[2, 4, 1, 2]).run(&input).unwrap().data(),
        expected
    );
}

#[test]
fn malformed_empty_constants_and_graph_input_overwrites_are_rejected_before_execution() {
    for shape in [&[0][..], &[0, i64::MAX, i64::MAX][..]] {
        let source = model_bytes(
            vec![
                tensor("empty", shape, &[]),
                node("add", "Add", &["x", "empty"], "y", vec![]),
            ],
            &[1],
            &[],
        );
        let error = Model::load_bytes(&source).err().unwrap().to_string();
        assert!(error.contains("nonpositive dimension"), "{error}");
    }
    let loaded = model(
        vec![
            node("overwrite", "Relu", &["x"], "x", vec![]),
            node("output", "Relu", &["x"], "y", vec![]),
        ],
        &[1],
        &[1],
    );
    let error = CompiledModel::new(loaded).err().unwrap().to_string();
    assert!(
        error.contains("overwrites an already defined tensor"),
        "{error}"
    );
}

#[test]
fn tracing_failure_restores_reused_slot_and_request_can_run_again() {
    let mut request = session(classifier(), &[1, 1, 2, 2]);
    let error = request
        .run_with_trace(&[1., -2., 3., 4.], |trace| {
            if trace.index == 3 {
                Err(RuntimeError::new("writer failed"))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert_eq!(error.to_string(), "writer failed");
    assert!((request.run(&[1., -2., 3., 4.]).unwrap().data()[0] - 0.7310586).abs() < 1e-7);
}

#[test]
fn invalid_shape_and_inputs_are_errors_without_destroying_binding() {
    let mut request = session(classifier(), &[1, 1, 2, 2]);
    let bytes = request.activation_bytes();
    assert!(request.reshape(&[0, 1, 2, 2]).is_err());
    assert!(request.reshape(&[1, 1, 3, 2]).is_err());
    assert!(request.reshape(&[usize::MAX, 1, 2, 2]).is_err());
    assert_eq!(request.input_shape().dimensions(), &[1, 1, 2, 2]);
    assert_eq!(request.activation_bytes(), bytes);
    assert!(request.run(&[1.]).is_err());
    assert!(
        request
            .run(&[f32::NAN, 1., 2., 3.])
            .unwrap_err()
            .to_string()
            .contains("nonfinite input")
    );
    assert!(request.run(&[f32::INFINITY, 1., 2., 3.]).is_err());
    assert!(request.run(&[1., -2., 3., 4.]).is_ok());
    let compiled = Arc::new(CompiledModel::new(classifier()).unwrap());
    assert!(
        Session::new(
            compiled.clone(),
            &[1, 1, 2, 2],
            ExecutionLimits {
                max_activation_bytes: 0
            }
        )
        .is_err()
    );
    assert!(
        Session::new(
            compiled,
            &[1, 1, 2, 2],
            ExecutionLimits {
                max_activation_bytes: bytes - 1
            }
        )
        .is_err()
    );
}

#[test]
fn broadcasting_channel_left_scalar_and_generic_counter_paths() {
    let cases = [
        (
            vec![2, 2, 1, 2],
            vec![2, 2, 1, 1],
            vec![10., 20., 30., 40.],
            vec![10., 20., 60., 80., 150., 180., 280., 320.],
        ),
        (
            vec![2, 1],
            vec![1, 3],
            vec![10., 20., 30.],
            vec![10., 20., 30., 20., 40., 60.],
        ),
    ];
    for (input_shape, rhs_shape, rhs, expected) in cases {
        let records = vec![
            tensor("rhs", &rhs_shape, &rhs),
            node("multiply", "Mul", &["x", "rhs"], "y", vec![]),
        ];
        let loaded = model(records, &input_shape, &[]);
        let shape = input_shape.iter().map(|&v| v as usize).collect::<Vec<_>>();
        let input = (1..=shape.iter().product::<usize>())
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        assert_eq!(
            session(loaded, &shape).run(&input).unwrap().data(),
            expected
        );
    }
    let loaded = model(
        vec![
            tensor("scalar", &[], &[2.]),
            node("add", "Add", &["scalar", "x"], "y", vec![]),
        ],
        &[2],
        &[2],
    );
    assert_eq!(
        session(loaded, &[2]).run(&[3., 4.]).unwrap().data(),
        &[5., 6.]
    );
}

#[test]
fn generic_reduce_negative_axes_and_noop() {
    let loaded = model(
        vec![node(
            "average",
            "ReduceMean",
            &["x"],
            "y",
            vec![ints("axes", &[-2]), int("keepdims", 0)],
        )],
        &[2, 3],
        &[3],
    );
    assert_eq!(
        session(loaded, &[2, 3])
            .run(&[1., 2., 3., 5., 6., 7.])
            .unwrap()
            .data(),
        &[3., 4., 5.]
    );
    let loaded = model(
        vec![node(
            "noop",
            "ReduceMean",
            &["x"],
            "y",
            vec![int("noop_with_empty_axes", 1)],
        )],
        &[2],
        &[2],
    );
    assert_eq!(
        session(loaded, &[2]).run(&[1., 2.]).unwrap().data(),
        &[1., 2.]
    );
    let loaded = model(
        vec![node(
            "duplicate",
            "ReduceMean",
            &["x"],
            "y",
            vec![ints("axes", &[0, -2])],
        )],
        &[2, 2],
        &[2],
    );
    assert!(
        Session::new(
            Arc::new(CompiledModel::new(loaded).unwrap()),
            &[2, 2],
            ExecutionLimits::default()
        )
        .unwrap_err()
        .to_string()
        .contains("duplicate")
    );
}

#[test]
fn grouped_depthwise_stride_and_dense_dilation_convolution() {
    let loaded = model(
        vec![
            tensor("w", &[2, 1, 2, 2], &[1., 0., 0., 1., 2., 0., 0., -1.]),
            tensor("b", &[2], &[1., -1.]),
            node(
                "conv",
                "Conv",
                &["x", "w", "b"],
                "y",
                vec![
                    ints("kernel_shape", &[2, 2]),
                    ints("strides", &[2, 1]),
                    int("group", 2),
                ],
            ),
        ],
        &[1, 2, 3, 3],
        &[1, 2, 1, 2],
    );
    let input = (1..=18).map(|v| v as f32).collect::<Vec<_>>();
    assert_eq!(
        session(loaded, &[1, 2, 3, 3]).run(&input).unwrap().data(),
        &[7., 9., 5., 6.]
    );
    let loaded = model(
        vec![
            tensor("w", &[1, 1, 2, 2], &[1., 0., 0., 1.]),
            node(
                "conv",
                "Conv",
                &["x", "w"],
                "y",
                vec![ints("kernel_shape", &[2, 2]), ints("dilations", &[2, 2])],
            ),
        ],
        &[1, 1, 3, 3],
        &[1, 1, 1, 1],
    );
    assert_eq!(
        session(loaded, &[1, 1, 3, 3])
            .run(&(1..=9).map(|v| v as f32).collect::<Vec<_>>())
            .unwrap()
            .data(),
        &[10.]
    );
}

#[test]
fn reshape_checked_zero_copy_dimension_and_bad_targets() {
    for target in [&[0, -1][..], &[2, 2][..]] {
        let loaded = model(
            vec![
                integers("target", target),
                node("reshape", "Reshape", &["x", "target"], "y", vec![]),
            ],
            &[2, 1, 2],
            &[2, 2],
        );
        let mut request = session(loaded, &[2, 1, 2]);
        assert_eq!(request.output_shape().dimensions(), &[2, 2]);
        assert_eq!(
            request.run(&[1., 2., 3., 4.]).unwrap().data(),
            &[1., 2., 3., 4.]
        );
    }
    for target in [
        &[-1, -1][..],
        &[-2][..],
        &[3, -1][..],
        &[5][..],
        &[0, 0, 0, 0][..],
    ] {
        let loaded = model(
            vec![
                integers("target", target),
                node("reshape", "Reshape", &["x", "target"], "y", vec![]),
            ],
            &[2, 2],
            &[2, 2],
        );
        let compiled = Arc::new(CompiledModel::new(loaded).unwrap());
        assert!(Session::new(compiled, &[2, 2], ExecutionLimits::default()).is_err());
    }
}

#[test]
fn unsupported_ops_nonfinite_constants_and_nonfinite_output_are_errors() {
    let loaded = model(
        vec![node("sigmoid", "Sigmoid", &["x"], "y", vec![])],
        &[1],
        &[1],
    );
    assert!(
        CompiledModel::new(loaded)
            .err()
            .unwrap()
            .to_string()
            .contains("unsupported operator")
    );
    let loaded = model(
        vec![
            tensor("nan", &[], &[f32::NAN]),
            node("add", "Add", &["x", "nan"], "y", vec![]),
        ],
        &[1],
        &[1],
    );
    assert!(
        CompiledModel::new(loaded)
            .err()
            .unwrap()
            .to_string()
            .contains("nonfinite")
    );
    let loaded = model(
        vec![
            tensor("large", &[], &[f32::MAX]),
            node("mul", "Mul", &["x", "large"], "y", vec![]),
        ],
        &[1],
        &[1],
    );
    assert!(
        session(loaded, &[1])
            .run(&[2.])
            .unwrap_err()
            .to_string()
            .contains("nonfinite output")
    );
    let loaded = model(
        vec![node(
            "probability",
            "Softmax",
            &["x"],
            "y",
            vec![int("axis", 0)],
        )],
        &[2, 2],
        &[2, 2],
    );
    assert!(
        Session::new(
            Arc::new(CompiledModel::new(loaded).unwrap()),
            &[2, 2],
            ExecutionLimits::default()
        )
        .is_err()
    );
}

#[test]
fn view_constructor_checks_shape_without_allocating() {
    assert_eq!(
        TensorView::from_slice(Shape::new(&[2]).unwrap(), &[1, 2])
            .unwrap()
            .data(),
        &[1, 2]
    );
    assert!(TensorView::from_slice(Shape::new(&[3]).unwrap(), &[1, 2]).is_err());
}
