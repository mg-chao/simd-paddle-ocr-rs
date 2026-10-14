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
fn floats(name: &str, values: &[f32]) -> Vec<u8> {
    [
        string(1, name),
        number(20, 6),
        bytes(7, values.iter().flat_map(|v| v.to_le_bytes()).collect()),
    ]
    .concat()
}
fn text_attribute(name: &str, value: &str) -> Vec<u8> {
    [string(1, name), number(20, 3), string(4, value)].concat()
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

fn tiny_rec_tail() -> Arc<Model> {
    model(
        vec![
            tensor("scale", &[2], &[2., 3.]),
            tensor("bias", &[2], &[0.5, -0.25]),
            tensor("mean", &[2], &[1., -1.]),
            tensor("variance", &[2], &[0.75, 3.75]),
            tensor(
                "projection",
                &[2, 5],
                &[1., 0., -1., 0.5, 2., 0., 1., 1., -2., -0.5],
            ),
            tensor("head_bias", &[5], &[0.0625, -0.125, 0.25, -0.5, 1.]),
            node(
                "pool",
                "AveragePool",
                &["x"],
                "pool",
                vec![ints("kernel_shape", &[3, 2]), ints("strides", &[3, 2])],
            ),
            node(
                "squeeze",
                "Squeeze",
                &["pool"],
                "squeezed",
                vec![ints("axes", &[-2])],
            ),
            node(
                "batch_norm",
                "BatchNormalization",
                &["squeezed", "scale", "bias", "mean", "variance"],
                "normalized",
                vec![float("epsilon", 0.25)],
            ),
            node(
                "unsqueeze",
                "Unsqueeze",
                &["normalized"],
                "expanded",
                vec![ints("axes", &[-2])],
            ),
            node(
                "squeeze_again",
                "Squeeze",
                &["expanded"],
                "compressed",
                vec![ints("axes", &[2])],
            ),
            node(
                "transpose",
                "Transpose",
                &["compressed"],
                "feature",
                vec![ints("perm", &[0, 2, 1])],
            ),
            node(
                "projection",
                "MatMul",
                &["feature", "projection"],
                "logits",
                vec![],
            ),
            node("bias", "Add", &["logits", "head_bias"], "biased", vec![]),
            node(
                "probabilities",
                "Softmax",
                &["biased"],
                "y",
                vec![int("axis", -1)],
            ),
        ],
        &[-1, 2, 3, -1],
        &[-1, -1, 5],
    )
}

#[test]
fn tiny_rec_tail_runs_batched_pool_normalization_transpose_and_projection() {
    let mut request = session(tiny_rec_tail(), &[2, 2, 3, 6]);
    let input: Vec<_> = (0..72).map(|i| (i % 17) as f32 / 8. - 1.).collect();
    let mut expected_feature = Vec::new();
    for batch in 0..2 {
        for column in 0..3 {
            for channel in 0..2 {
                let mut sum = 0.0f32;
                for y in 0..3 {
                    for x in 0..2 {
                        sum += input[((batch * 2 + channel) * 3 + y) * 6 + column * 2 + x];
                    }
                }
                let pooled = sum / 6.;
                expected_feature.push(if channel == 0 {
                    (pooled - 1.) * 2. + 0.5
                } else {
                    (pooled + 1.) * 1.5 - 0.25
                });
            }
        }
    }
    let weights = [1., 0., -1., 0.5, 2., 0., 1., 1., -2., -0.5];
    let bias = [0.0625, -0.125, 0.25, -0.5, 1.];
    let mut expected_logits = Vec::new();
    let mut expected = Vec::new();
    for feature in expected_feature.chunks_exact(2) {
        let mut logits = [0.0f32; 5];
        for column in 0..5 {
            logits[column] += feature[0] * weights[column];
            logits[column] += feature[1] * weights[5 + column];
        }
        expected_logits.extend_from_slice(&logits);
        for (value, bias) in logits.iter_mut().zip(bias) {
            *value += bias;
        }
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials = logits.map(|value| (value - maximum).exp());
        let sum: f32 = exponentials.iter().sum();
        expected.extend(exponentials.map(|value| value / sum));
    }
    let mut count = 0;
    let output = request
        .run_with_trace(&input, |node| {
            count += 1;
            if node.name == "transpose" {
                assert_eq!(node.values, expected_feature);
            }
            if node.name == "projection" {
                assert_eq!(node.values, expected_logits);
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(count, 9);
    assert_eq!(output.shape().dimensions(), &[2, 3, 5]);
    for (&actual, &expected) in output.data().iter().zip(&expected) {
        assert!((actual - expected).abs() <= 1e-6);
    }
    let actual = output.data().to_vec();
    assert_eq!(request.run(&input).unwrap().data(), actual);
    request.reshape(&[1, 2, 3, 10]).unwrap();
    assert_eq!(request.output_shape().dimensions(), &[1, 5, 5]);
    request.run(&[0.; 60]).unwrap();
    assert!(request.reshape(&[1, 2, 3, 1]).is_err());
    assert_eq!(request.input_shape().dimensions(), &[1, 2, 3, 10]);
    request.run(&[0.; 60]).unwrap();
}

#[test]
fn recognition_axis_and_pool_parameters_are_validated_before_execution() {
    for (operator, attrs, shape, message) in [
        (
            "Squeeze",
            vec![ints("axes", &[0])],
            vec![2, 1, 3],
            "dimension one",
        ),
        (
            "Squeeze",
            vec![ints("axes", &[1, -2])],
            vec![2, 1, 3],
            "duplicated",
        ),
        (
            "Unsqueeze",
            vec![ints("axes", &[0, -4])],
            vec![2, 3],
            "duplicated",
        ),
        ("Unsqueeze", vec![ints("axes", &[4])], vec![2, 3], "axis"),
        (
            "Transpose",
            vec![ints("perm", &[0, 0, 2])],
            vec![2, 3, 4],
            "duplicated",
        ),
        (
            "Transpose",
            vec![ints("perm", &[0, 1])],
            vec![2, 3, 4],
            "rank mismatch",
        ),
        ("Softmax", vec![int("axis", 1)], vec![2, 3, 4], "last-axis"),
    ] {
        let dimensions = shape.iter().map(|&value| value as i64).collect::<Vec<_>>();
        let compiled = Arc::new(
            CompiledModel::new(model(
                vec![node("invalid", operator, &["x"], "y", attrs)],
                &dimensions,
                &dimensions,
            ))
            .unwrap(),
        );
        assert!(
            Session::new(compiled, &shape, ExecutionLimits::default())
                .unwrap_err()
                .to_string()
                .contains(message),
            "{operator}"
        );
    }
    let unsupported_pool = model(
        vec![node(
            "pool",
            "AveragePool",
            &["x"],
            "y",
            vec![
                ints("kernel_shape", &[3, 2]),
                ints("strides", &[3, 2]),
                int("count_include_pad", 1),
            ],
        )],
        &[1, 1, 3, 4],
        &[1, 1, 1, 2],
    );
    assert!(
        CompiledModel::new(unsupported_pool)
            .err()
            .unwrap()
            .to_string()
            .contains("AveragePool requires")
    );
}

#[test]
fn batch_normalization_preserves_subtract_multiply_add_and_rejects_bad_parameters() {
    let normalization = |channels: i64, variance: f32, training: i64| {
        model(
            vec![
                tensor("scale", &[channels], &vec![2.; channels as usize]),
                tensor("bias", &[channels], &vec![1.; channels as usize]),
                tensor("mean", &[channels], &vec![1e8; channels as usize]),
                tensor("variance", &[channels], &vec![variance; channels as usize]),
                node(
                    "batch_norm",
                    "BatchNormalization",
                    &["x", "scale", "bias", "mean", "variance"],
                    "y",
                    vec![float("epsilon", 0.), int("training_mode", training)],
                ),
            ],
            &[1, 2, 2],
            &[1, 2, 2],
        )
    };
    let mut request = session(normalization(2, 1., 0), &[1, 2, 2]);
    // Folding to x*2 + (1 - mean*2) would lose the +1 through cancellation.
    assert_eq!(request.run(&[1e8; 4]).unwrap().data(), &[1.; 4]);
    assert!(
        Session::new(
            Arc::new(CompiledModel::new(normalization(1, 1., 0)).unwrap()),
            &[1, 2, 2],
            ExecutionLimits::default()
        )
        .is_err()
    );
    assert!(
        CompiledModel::new(normalization(2, 1., 1))
            .err()
            .unwrap()
            .to_string()
            .contains("inference mode")
    );
    assert!(
        session(normalization(2, -1., 0), &[1, 2, 2])
            .run(&[0.; 4])
            .unwrap_err()
            .to_string()
            .contains("nonfinite output")
    );
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
fn dense_channel_tiles_preserve_groups_batches_padding_stride_dilation_and_bias_order() {
    let weights: Vec<_> = (1..=14)
        .flat_map(|channel| [channel as f32, 0., 0., channel as f32].repeat(2))
        .collect();
    let biases: Vec<_> = (1..=14).map(|channel| channel as f32).collect();
    let loaded = model(
        vec![
            tensor("w", &[14, 2, 2, 2], &weights),
            tensor("b", &[14], &biases),
            node(
                "conv",
                "Conv",
                &["x", "w", "b"],
                "y",
                vec![ints("kernel_shape", &[2, 2]), int("group", 2)],
            ),
        ],
        &[2, 4, 2, 3],
        &[2, 14, 1, 2],
    );
    let values: Vec<_> = [1., 10., 100., 1000.]
        .into_iter()
        .flat_map(|scale| (1..=6).map(move |value| value as f32 * scale))
        .collect();
    let input = [
        values.clone(),
        values.iter().map(|value| value * 2.).collect(),
    ]
    .concat();
    let expected: Vec<_> = (1..=7)
        .flat_map(|channel| [67. * channel as f32, 89. * channel as f32])
        .chain((8..=14).flat_map(|channel| [6601. * channel as f32, 8801. * channel as f32]))
        .chain((1..=7).flat_map(|channel| [133. * channel as f32, 177. * channel as f32]))
        .chain((8..=14).flat_map(|channel| [13201. * channel as f32, 17601. * channel as f32]))
        .collect();
    assert_eq!(
        session(loaded, &[2, 4, 2, 3]).run(&input).unwrap().data(),
        expected
    );
    let loaded = model(
        vec![
            tensor("w", &[5, 2, 2, 2], &[1.; 40]),
            tensor("b", &[5], &[1e8; 5]),
            node(
                "conv",
                "Conv",
                &["x", "w", "b"],
                "y",
                vec![ints("kernel_shape", &[2, 2]), ints("pads", &[1, 1, 0, 0])],
            ),
        ],
        &[1, 2, 1, 1],
        &[1, 5, 1, 1],
    );
    assert_eq!(
        session(loaded, &[1, 2, 1, 1])
            .run(&[-1e8, 1.])
            .unwrap()
            .data(),
        &[1.; 5]
    );
    let loaded = model(
        vec![
            tensor("w", &[5, 1, 3, 3], &[1.; 45]),
            node(
                "conv",
                "Conv",
                &["x", "w"],
                "y",
                vec![
                    ints("kernel_shape", &[3, 3]),
                    ints("strides", &[2, 2]),
                    ints("pads", &[1, 1, 1, 1]),
                ],
            ),
        ],
        &[1, 1, 2, 3],
        &[1, 5, 1, 2],
    );
    assert_eq!(
        session(loaded, &[1, 1, 2, 3])
            .run(&[1., 2., 3., 4., 5., 6.])
            .unwrap()
            .data(),
        [12., 16.].repeat(5)
    );
    let loaded = model(
        vec![
            tensor("w", &[5, 1, 2, 2], &[1.; 20]),
            node(
                "conv",
                "Conv",
                &["x", "w"],
                "y",
                vec![ints("kernel_shape", &[2, 2]), ints("dilations", &[2, 2])],
            ),
        ],
        &[1, 1, 3, 3],
        &[1, 5, 1, 1],
    );
    assert_eq!(
        session(loaded, &[1, 1, 3, 3])
            .run(&[1., 2., 3., 4., 5., 6., 7., 8., 9.])
            .unwrap()
            .data(),
        &[20.; 5]
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
        vec![
            tensor("exponent", &[], &[3.]),
            node("power", "Pow", &["x", "exponent"], "y", vec![]),
        ],
        &[1],
        &[1],
    );
    assert!(
        CompiledModel::new(loaded)
            .err()
            .unwrap()
            .to_string()
            .contains("exponent two only")
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

#[test]
fn variadic_concat_keeps_all_branches_repeated_inputs_and_batch_order() {
    let loaded = model(
        vec![
            tensor(
                "extra",
                &[2, 2, 1, 2],
                &(5..=12).map(|value| value as f32).collect::<Vec<_>>(),
            ),
            node("positive", "Relu", &["x"], "positive", vec![]),
            node(
                "join",
                "Concat",
                &["x", "positive", "x", "extra", "positive"],
                "y",
                vec![int("axis", -3)],
            ),
        ],
        &[2, 1, 1, 2],
        &[2, 6, 1, 2],
    );
    let mut request = session(loaded, &[2, 1, 1, 2]);
    let expected = [
        -1., 2., 0., 2., -1., 2., 5., 6., 7., 8., 0., 2., 3., -4., 3., 0., 3., -4., 9., 10., 11.,
        12., 3., 0.,
    ];
    let mut observed = 0;
    assert_eq!(
        request
            .run_with_trace(&[-1., 2., 3., -4.], |trace| {
                if trace.op_type == "Concat" {
                    assert_eq!(trace.shape.dimensions(), &[2, 6, 1, 2]);
                    assert_eq!(trace.values, expected);
                    observed += 1;
                }
                Ok(())
            })
            .unwrap()
            .data(),
        expected
    );
    assert_eq!(observed, 1);
    assert!(
        request
            .run_with_trace(&[4., 5., 6., 7.], |_| Err(RuntimeError::new(
                "trace failed"
            )))
            .is_err()
    );
    assert_eq!(request.run(&[-1., 2., 3., -4.]).unwrap().data(), expected);

    for axis in [1, 4, -5] {
        let loaded = model(
            vec![
                tensor("bad", &[1, 2, 1, 2], &[1.; 4]),
                node(
                    "join",
                    "Concat",
                    &["x", "bad"],
                    "y",
                    vec![int("axis", axis)],
                ),
            ],
            &[2, 1, 1, 2],
            &[-1, -1, -1, -1],
        );
        assert!(
            Session::new(
                Arc::new(CompiledModel::new(loaded).unwrap()),
                &[2, 1, 1, 2],
                ExecutionLimits::default()
            )
            .is_err()
        );
    }
}

fn pool_model(shape: &[i64]) -> Arc<Model> {
    model(
        vec![node(
            "pool",
            "MaxPool",
            &["x"],
            "y",
            vec![
                ints("kernel_shape", &[2, 2]),
                ints("strides", &[1, 1]),
                ints("pads", &[0, 0, 1, 1]),
            ],
        )],
        shape,
        shape,
    )
}

#[test]
fn max_pool_excludes_padding_and_handles_degenerate_rows_and_columns() {
    let mut request = session(pool_model(&[2, 1, 2, 3]), &[2, 1, 2, 3]);
    assert_eq!(
        request
            .run(&[-5., -2., -9., -1., -8., -3., 1., 7., 3., 5., 2., 4.])
            .unwrap()
            .data(),
        &[-1., -2., -3., -1., -3., -3., 7., 7., 4., 5., 4., 4.]
    );
    for (shape, values, expected) in [
        ([1, 1, 1, 1], vec![-5.], vec![-5.]),
        ([1, 1, 1, 3], vec![-5., -2., -9.], vec![-2., -2., -9.]),
        ([1, 1, 3, 1], vec![-5., -2., -9.], vec![-2., -2., -9.]),
    ] {
        let signed = shape.map(|dimension| dimension as i64);
        assert_eq!(
            session(pool_model(&signed), &shape)
                .run(&values)
                .unwrap()
                .data(),
            expected
        );
    }
}

fn resize_model(factor: f32, mode: &str, coordinate: &str, nearest: &str) -> Arc<Model> {
    model(
        vec![node(
            "resize",
            "Resize",
            &["x"],
            "y",
            vec![
                floats("scales", &[1., 1., factor, factor]),
                text_attribute("mode", mode),
                text_attribute("coordinate_transformation_mode", coordinate),
                text_attribute("nearest_mode", nearest),
            ],
        )],
        &[2, 1, 1, 2],
        &[2, 1, -1, -1],
    )
}

#[test]
fn nearest_resize_repeats_rectangular_batches_and_rejects_other_contracts() {
    for factor in [2usize, 4, 8] {
        let mut request = session(
            resize_model(factor as f32, "nearest", "asymmetric", "floor"),
            &[2, 1, 1, 2],
        );
        assert_eq!(
            request.output_shape().dimensions(),
            &[2, 1, factor, 2 * factor]
        );
        let mut expected = Vec::new();
        for pair in [[1., 2.], [-3., 4.]] {
            for _ in 0..factor {
                expected.extend(std::iter::repeat_n(pair[0], factor));
                expected.extend(std::iter::repeat_n(pair[1], factor));
            }
        }
        assert_eq!(request.run(&[1., 2., -3., 4.]).unwrap().data(), expected);
        assert!(
            request
                .run(&[0.; 4])
                .unwrap()
                .data()
                .iter()
                .all(|&value| value == 0.)
        );
    }
    for (factor, mode, coordinate, nearest) in [
        (1.5, "nearest", "asymmetric", "floor"),
        (2., "linear", "asymmetric", "floor"),
        (2., "nearest", "half_pixel", "floor"),
        (2., "nearest", "asymmetric", "round_prefer_floor"),
    ] {
        assert!(CompiledModel::new(resize_model(factor, mode, coordinate, nearest)).is_err());
    }
}

#[test]
fn transpose_convolution_rectangular_batches_accumulation_bias_and_overwrite() {
    let weights = [1., 10., 100., 1000., -1., -10., -100., -1000.];
    let loaded = model(
        vec![
            tensor("w", &[1, 2, 2, 2], &weights),
            tensor("b", &[2], &[1., -1.]),
            node(
                "upsample",
                "ConvTranspose",
                &["x", "w", "b"],
                "y",
                vec![ints("kernel_shape", &[2, 2]), ints("strides", &[2, 2])],
            ),
        ],
        &[2, 1, 1, 2],
        &[2, 2, 2, 4],
    );
    let mut request = session(loaded, &[2, 1, 1, 2]);
    let expected = [
        2., 11., 3., 21., 101., 1001., 201., 2001., -2., -11., -3., -21., -101., -1001., -201.,
        -2001., 4., 31., 5., 41., 301., 3001., 401., 4001., -4., -31., -5., -41., -301., -3001.,
        -401., -4001.,
    ];
    assert_eq!(request.run(&[1., 2., 3., 4.]).unwrap().data(), expected);
    let empty = request.run(&[0.; 4]).unwrap();
    for (channel, plane) in empty.data().chunks_exact(8).enumerate() {
        assert!(
            plane
                .iter()
                .all(|&value| value == if channel % 2 == 0 { 1. } else { -1. })
        );
    }
    let loaded = model(
        vec![
            tensor("w", &[2, 1, 2, 2], &[1.; 8]),
            tensor("b", &[1], &[1e8]),
            node(
                "upsample",
                "ConvTranspose",
                &["x", "w", "b"],
                "y",
                vec![ints("kernel_shape", &[2, 2]), ints("strides", &[2, 2])],
            ),
        ],
        &[1, 2, 1, 1],
        &[1, 1, 2, 2],
    );
    assert_eq!(
        session(loaded, &[1, 2, 1, 1])
            .run(&[-1e8, 1.])
            .unwrap()
            .data(),
        &[1.; 4]
    );
}

#[test]
fn spatial_ops_reject_unsupported_parameters_and_incompatible_weight_shapes() {
    let loaded = model(
        vec![node(
            "pool",
            "MaxPool",
            &["x"],
            "y",
            vec![
                ints("kernel_shape", &[2, 2]),
                ints("strides", &[2, 2]),
                ints("pads", &[0, 0, 1, 1]),
            ],
        )],
        &[1, 1, 2, 2],
        &[1, 1, 2, 2],
    );
    assert!(
        CompiledModel::new(loaded)
            .err()
            .unwrap()
            .to_string()
            .contains("MaxPool")
    );
    for weight_shape in [&[2, 1, 2, 2][..], &[1, 1, 1, 2][..]] {
        let count = weight_shape.iter().product::<i64>() as usize;
        let loaded = model(
            vec![
                tensor("w", weight_shape, &vec![1.; count]),
                node(
                    "up",
                    "ConvTranspose",
                    &["x", "w"],
                    "y",
                    vec![ints("kernel_shape", &[2, 2]), ints("strides", &[2, 2])],
                ),
            ],
            &[1, 1, 1, 2],
            &[1, 1, 2, 4],
        );
        let compiled = Arc::new(CompiledModel::new(loaded).unwrap());
        assert!(
            Session::new(compiled, &[1, 1, 1, 2], ExecutionLimits::default())
                .unwrap_err()
                .to_string()
                .contains("weights")
        );
    }
    let loaded = model(
        vec![
            tensor("w", &[1, 1, 2, 2], &[1.; 4]),
            node(
                "up",
                "ConvTranspose",
                &["x", "w"],
                "y",
                vec![ints("kernel_shape", &[2, 2]), ints("strides", &[1, 1])],
            ),
        ],
        &[1, 1, 1, 2],
        &[1, 1, 2, 4],
    );
    assert!(CompiledModel::new(loaded).is_err());
}

#[test]
fn div_preserves_operand_order_and_scalar_division_rounding() {
    for (inputs, values, expected) in [
        (["x", "scalar"], [9., -9., 5., -5.], [0.9, -0.9, 0.5, -0.5]),
        (["scalar", "x"], [2., 4., 5., 10.], [5., 2.5, 2., 1.]),
    ] {
        let loaded = model(
            vec![
                tensor("scalar", &[], &[10.]),
                node("divide", "Div", &inputs, "y", vec![]),
            ],
            &[4],
            &[4],
        );
        assert_eq!(session(loaded, &[4]).run(&values).unwrap().data(), expected);
    }
    let loaded = model(
        vec![
            tensor("zero", &[], &[0.]),
            node("divide", "Div", &["x", "zero"], "y", vec![]),
        ],
        &[1],
        &[1],
    );
    assert!(
        session(loaded, &[1])
            .run(&[1.])
            .unwrap_err()
            .to_string()
            .contains("nonfinite output")
    );
}

#[test]
fn erf_reference_values_and_sigmoid_saturation() {
    let input = [
        -3.,
        -1.,
        -0.5,
        -0.0001,
        0.,
        0.0001,
        0.5,
        1.,
        3.,
        f32::MAX,
        -f32::MAX,
    ];
    // Frozen outputs of the C# scalar polynomial, including its small-input rounding.
    let bits = [
        0xbf7ffe8d, 0xbf57bb3c, 0xbf053f7e, 0xb8ecc000, 0, 0x38ecc000, 0x3f053f7e, 0x3f57bb3c,
        0x3f7ffe8d, 0x3f800000, 0xbf800000,
    ];
    let loaded = model(vec![node("erf", "Erf", &["x"], "y", vec![])], &[11], &[11]);
    let mut request = session(loaded, &[11]);
    let output = request.run(&input).unwrap();
    for (index, (&actual, expected)) in output
        .data()
        .iter()
        .zip(bits.map(f32::from_bits))
        .enumerate()
    {
        if (3..=5).contains(&index) {
            assert_eq!(actual, expected);
        } else {
            assert!(
                (actual - expected).abs() <= 2e-7,
                "index {index}: {actual} vs {expected}"
            );
        }
    }
    let loaded = model(
        vec![node("sigmoid", "Sigmoid", &["x"], "y", vec![])],
        &[5],
        &[5],
    );
    let actual = session(loaded, &[5])
        .run(&[-1000., -1., 0., 1., 1000.])
        .unwrap()
        .data()
        .to_vec();
    for (&actual, expected) in actual.iter().zip([0., 0.26894143, 0.5, 0.7310586, 1.]) {
        assert!((actual - expected).abs() <= 1e-7);
    }
}

#[test]
fn fill_callback_errors_nonfinite_inputs_and_trace_errors_allow_reuse() {
    let mut request = session(classifier(), &[1, 1, 2, 2]);
    let mut calls = 0;
    assert!(
        request
            .run_with_input(|input| {
                calls += 1;
                input.fill(0.);
                Err(RuntimeError::new("preprocess failed"))
            })
            .unwrap_err()
            .to_string()
            .contains("preprocess failed")
    );
    assert_eq!(calls, 1);
    assert!(
        request
            .run_with_input(|input| {
                input.fill(f32::NAN);
                Ok(())
            })
            .unwrap_err()
            .to_string()
            .contains("nonfinite input")
    );
    assert!(
        request
            .run_with_input_and_trace(
                |input| {
                    input.copy_from_slice(&[1., -2., 3., 4.]);
                    Ok(())
                },
                |_| Err(RuntimeError::new("trace failed"))
            )
            .is_err()
    );
    let mut observed = 0;
    let output = request
        .run_with_input_and_trace(
            |input| {
                input.copy_from_slice(&[1., -2., 3., 4.]);
                Ok(())
            },
            |trace| {
                assert_eq!(trace.index, observed);
                observed += 1;
                Ok(())
            },
        )
        .unwrap();
    assert!((output.data()[0] - 0.7310586).abs() < 1e-7);
    assert_eq!(observed, 9);
}

#[test]
fn reshape_reuses_capacities_compacts_under_budget_and_keeps_failed_binding() {
    let loaded = model(
        vec![
            node(
                "columns",
                "ReduceMean",
                &["x"],
                "columns",
                vec![ints("axes", &[0])],
            ),
            node(
                "rows",
                "ReduceMean",
                &["x"],
                "rows",
                vec![ints("axes", &[1])],
            ),
            node("join", "Add", &["columns", "rows"], "y", vec![]),
        ],
        &[-1, -1],
        &[-1, -1],
    );
    let compiled = Arc::new(CompiledModel::new(loaded).unwrap());
    let mut request = Session::new(
        compiled,
        &[2, 8],
        ExecutionLimits {
            max_activation_bytes: 104,
        },
    )
    .unwrap();
    assert_eq!(request.activation_bytes(), 104);
    assert_eq!(request.retained_activation_bytes(), 104);
    request.reshape(&[8, 2]).unwrap();
    // Keeping both historical axis capacities would require 128 bytes.
    assert_eq!(request.retained_activation_bytes(), 104);
    let values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
    let mut previous_pointer = 0;
    assert_eq!(
        request
            .run_with_input(|input| {
                previous_pointer = input.as_ptr() as usize;
                input.copy_from_slice(&values);
                Ok(())
            })
            .unwrap()
            .data(),
        &(0..8)
            .flat_map(|row| [7.5 + 2. * row as f32, 8.5 + 2. * row as f32])
            .collect::<Vec<_>>()
    );
    assert!(request.reshape(&[9, 2]).is_err());
    assert!(request.reshape(&[usize::MAX, 2]).is_err());
    assert_eq!(request.input_shape().dimensions(), &[8, 2]);
    assert_eq!(request.retained_activation_bytes(), 104);
    request.reshape(&[4, 2]).unwrap();
    assert_eq!(request.activation_bytes(), 56);
    assert_eq!(request.retained_activation_bytes(), 104);
    let mut next_pointer = 0;
    let output = request
        .run_with_input(|input| {
            next_pointer = input.as_ptr() as usize;
            input.copy_from_slice(&values[..8]);
            Ok(())
        })
        .unwrap();
    assert_eq!(previous_pointer, next_pointer);
    assert_eq!(output.data(), &[3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]);
}

#[test]
#[ignore = "requires SIMDOCR_DET_REFERENCE pointing to an explicit canonical all-node bundle"]
fn shipped_detector_all_node_parity_from_explicit_reference() {
    use serde_json::Value;
    use std::{
        fs::{self, File},
        io::Read,
        path::Path,
    };
    let root =
        std::env::var("SIMDOCR_DET_REFERENCE").expect("SIMDOCR_DET_REFERENCE must be explicit");
    let root = Path::new(&root);
    simd_paddle_ocr_rs::validate_bundle(root).unwrap();
    let manifest: Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["contract"]["profile"], "canonical");
    assert_eq!(manifest["contract"]["nodes"], "all");
    let model = manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|asset| asset["id"] == "det")
        .unwrap();
    let model = Arc::new(Model::load_path(root.join(model["path"].as_str().unwrap())).unwrap());
    let inventory: Value =
        serde_json::from_slice(&fs::read(root.join("graph-inventory.json")).unwrap()).unwrap();
    let inventory = inventory
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["id"] == "det")
        .unwrap();
    assert_eq!(
        inventory["nodes"].as_array().unwrap().len(),
        model.graph().nodes.len()
    );
    for (saved, node) in inventory["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&model.graph().nodes)
    {
        assert_eq!(saved["name"], node.name);
        assert_eq!(saved["op_type"], node.op_type);
    }
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let shape = |artifact: &Value| {
        artifact["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|dimension| dimension.as_u64().unwrap() as usize)
            .collect::<Vec<_>>()
    };
    let compare =
        |actual: &[f32], actual_shape: Shape, artifact: &Value| -> Result<(), RuntimeError> {
            let expected_shape = shape(artifact);
            if actual_shape.dimensions() != expected_shape || artifact["layout"] != "nchw" {
                return Err(RuntimeError::new(format!(
                    "{} shape/layout differs",
                    artifact["id"]
                )));
            }
            if artifact["bytes"].as_u64().unwrap() != (actual.len() * 4) as u64 {
                return Err(RuntimeError::new("reference tensor length differs"));
            }
            let mut file = File::open(root.join(artifact["path"].as_str().unwrap())).unwrap();
            let mut buffer = [0u8; 64 * 1024];
            let mut position = 0;
            while position < actual.len() {
                let count = (actual.len() - position).min(buffer.len() / 4);
                file.read_exact(&mut buffer[..count * 4]).unwrap();
                for (offset, bytes) in buffer[..count * 4].chunks_exact(4).enumerate() {
                    let expected = f32::from_le_bytes(bytes.try_into().unwrap());
                    let value = actual[position + offset];
                    if !expected.is_finite()
                        || !value.is_finite()
                        || (f64::from(value) - f64::from(expected)).abs()
                            > 1e-5 + 1e-4 * f64::from(expected).abs()
                    {
                        return Err(RuntimeError::new(format!(
                            "{} element {}: {value} vs {expected}",
                            artifact["id"],
                            position + offset
                        )));
                    }
                }
                position += count;
            }
            Ok(())
        };
    let mut request: Option<Session> = None;
    let mut checked = 0;
    for case in manifest["cases"].as_array().unwrap() {
        let artifacts = case["artifacts"].as_array().unwrap();
        let input = artifacts
            .iter()
            .find(|artifact| artifact["id"] == "det.input")
            .unwrap();
        let dimensions = shape(input);
        match request.as_mut() {
            Some(request) => request.reshape(&dimensions).unwrap(),
            None => {
                request = Some(
                    Session::new(compiled.clone(), &dimensions, ExecutionLimits::default())
                        .unwrap(),
                )
            }
        }
        let values = fs::read(root.join(input["path"].as_str().unwrap()))
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let mut nodes = artifacts
            .iter()
            .filter_map(|artifact| {
                artifact["id"]
                    .as_str()
                    .unwrap()
                    .strip_prefix("det.node.")
                    .map(|ordinal| (ordinal.parse::<usize>().unwrap(), artifact))
            })
            .collect::<Vec<_>>();
        nodes.sort_by_key(|&(ordinal, _)| ordinal);
        assert_eq!(nodes.len(), compiled.node_count());
        for (index, &(ordinal, _)) in nodes.iter().enumerate() {
            assert_eq!(index, ordinal);
        }
        let mut observed = 0;
        let output = request
            .as_mut()
            .unwrap()
            .run_with_trace(&values, |trace| {
                assert_eq!(trace.index, observed);
                observed += 1;
                compare(trace.values, trace.shape, nodes[trace.index].1)
            })
            .unwrap_or_else(|error| panic!("case {}: {error}", case["id"]));
        assert_eq!(observed, compiled.node_count());
        let probabilities = artifacts
            .iter()
            .find(|artifact| artifact["id"] == "det.probabilities")
            .unwrap();
        compare(output.data(), output.shape(), probabilities).unwrap();
        checked += observed + 1;
        println!(
            "{}: {observed} nodes + output matched at atol=1e-5/rtol=1e-4",
            case["id"]
        );
    }
    assert!(checked > 0);
}

#[test]
fn typed_shape_recipes_resolve_data_aliases_and_reset_scalar_vector_caches() {
    let records = vec![
        tensor("zero", &[], &[0.]),
        integers("prefix_start", &[0]),
        integers("prefix_end", &[2]),
        integers("width_start", &[3]),
        integers("width_end", &[4]),
        integers("unit_step", &[1]),
        integers("minus_one", &[-1]),
        integers("zero_dim", &[0]),
        integers("one_dim", &[1]),
        integers("channels", &[2]),
        node("data", "Add", &["x", "zero"], "data", vec![]),
        node(
            "data_alias",
            "Identity",
            &["data"],
            "raw_data_alias",
            vec![],
        ),
        node("shape", "Shape", &["raw_data_alias"], "shape", vec![]),
        node("shape_alias", "Identity", &["shape"], "shape_alias", vec![]),
        node(
            "prefix",
            "Slice",
            &["shape_alias", "prefix_start", "prefix_end"],
            "prefix",
            vec![],
        ),
        node(
            "flatten_target",
            "Concat",
            &["prefix", "minus_one"],
            "flatten_target",
            vec![int("axis", 0)],
        ),
        node(
            "flatten",
            "Reshape",
            &["raw_data_alias", "flatten_target"],
            "flatten",
            vec![],
        ),
        node(
            "sequence",
            "Transpose",
            &["flatten"],
            "sequence",
            vec![ints("perm", &[0, 2, 1])],
        ),
        node(
            "width",
            "Slice",
            &["shape_alias", "width_start", "width_end", "", "unit_step"],
            "width",
            vec![],
        ),
        node(
            "scalar_width",
            "Squeeze",
            &["width"],
            "scalar_width",
            vec![ints("axes", &[0])],
        ),
        node(
            "width_alias",
            "Identity",
            &["scalar_width"],
            "width_alias",
            vec![],
        ),
        node(
            "width_alias2",
            "Identity",
            &["width_alias"],
            "width_alias2",
            vec![],
        ),
        node(
            "vector_width",
            "Unsqueeze",
            &["width_alias2"],
            "vector_width",
            vec![ints("axes", &[0])],
        ),
        node(
            "tail_target",
            "Concat",
            &["zero_dim", "one_dim", "vector_width", "channels"],
            "tail_target",
            vec![int("axis", 0)],
        ),
        node(
            "tail",
            "Reshape",
            &["sequence", "tail_target"],
            "tail",
            vec![],
        ),
        node(
            "restore",
            "Transpose",
            &["tail"],
            "y",
            vec![ints("perm", &[0, 3, 1, 2])],
        ),
    ];
    let source = model_bytes(records, &[-1, 2, 1, -1], &[-1, 2, 1, -1]);
    let loaded = Arc::new(Model::load_bytes(&source).unwrap());
    assert!(loaded.raw_snapshot().is_none());
    assert!(
        loaded
            .graph()
            .tensors
            .iter()
            .all(|tensor| tensor.name != "raw_data_alias")
    );
    let original_shape = loaded
        .shape_metadata()
        .nodes
        .iter()
        .find(|node| node.op_type == "Shape")
        .unwrap();
    assert_eq!(original_shape.inputs, ["raw_data_alias"]);
    let diagnostic = Model::load_bytes_with_options(
        &source,
        &simd_paddle_ocr_rs::model::LoadOptions {
            capture_raw_snapshot: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(loaded.normalized_snapshot().unwrap()).unwrap(),
        serde_json::to_value(diagnostic.normalized_snapshot().unwrap()).unwrap()
    );
    let compiled = Arc::new(CompiledModel::new(loaded).unwrap());
    let large = Session::new(compiled.clone(), &[2, 2, 1, 7], ExecutionLimits::default()).unwrap();
    let limit = large.activation_bytes();
    let mut request = Session::new(
        compiled,
        &[1, 2, 1, 3],
        ExecutionLimits {
            max_activation_bytes: limit,
        },
    )
    .unwrap();
    for dimensions in [[1, 2, 1, 3], [2, 2, 1, 7], [1, 2, 1, 4], [2, 2, 1, 7]] {
        request.reshape(&dimensions).unwrap();
        let values = (0..dimensions.iter().product::<usize>())
            .map(|i| i as f32 / 8. - 1.)
            .collect::<Vec<_>>();
        let mut reshapes = Vec::new();
        let output = request
            .run_with_trace(&values, |trace| {
                if trace.op_type == "Reshape" {
                    reshapes.push(trace.shape.dimensions().to_vec());
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(output.shape().dimensions(), dimensions);
        assert_eq!(output.data(), values);
        assert_eq!(
            reshapes,
            [
                vec![dimensions[0], 2, dimensions[3]],
                vec![dimensions[0], 1, dimensions[3], 2]
            ]
        );
    }
    assert!(request.reshape(&[2, 2, 1, 70]).is_err());
    assert_eq!(request.input_shape().dimensions(), [2, 2, 1, 7]);
    assert!(request.retained_activation_bytes() <= limit);
    let values = vec![0.5; 28];
    assert_eq!(request.run(&values).unwrap().data(), values);
}

#[test]
fn attention_qkv_slices_activation_matrices_and_shared_input_lifetimes_match_independent_calculation()
 {
    let mut records = vec![integers("axis", &[0]), integers("step", &[1])];
    for (index, name) in ["q", "k", "v"].into_iter().enumerate() {
        records.push(integers(&format!("start{index}"), &[index as i64]));
        records.push(integers(&format!("end{index}"), &[index as i64 + 1]));
        records.push(node(
            &format!("slice_{name}"),
            "Slice",
            &[
                "x",
                &format!("start{index}"),
                &format!("end{index}"),
                "axis",
                "step",
            ],
            &format!("slice_{name}"),
            vec![],
        ));
        records.push(node(
            &format!("squeeze_{name}"),
            "Squeeze",
            &[&format!("slice_{name}")],
            name,
            vec![ints("axes", &[0])],
        ));
    }
    records.extend([
        node(
            "keys",
            "Transpose",
            &["k"],
            "kt",
            vec![ints("perm", &[0, 1, 3, 2])],
        ),
        node("scores", "MatMul", &["q", "kt"], "scores", vec![]),
        node(
            "probabilities",
            "Softmax",
            &["scores"],
            "probabilities",
            vec![int("axis", 3)],
        ),
        node("context", "MatMul", &["probabilities", "v"], "y", vec![]),
    ]);
    let mut request = session(
        model(records, &[3, 1, 2, 4, 3], &[1, 2, 4, 3]),
        &[3, 1, 2, 4, 3],
    );
    let values = (0..72)
        .map(|i| ((i * 13 % 19) as f32 - 9.) / 16.)
        .collect::<Vec<_>>();
    let mut scores = [0.0f32; 32];
    let mut expected = [0.0f32; 24];
    for head in 0..2 {
        for row in 0..4 {
            for col in 0..4 {
                for inner in 0..3 {
                    scores[head * 16 + row * 4 + col] += values[head * 12 + row * 3 + inner]
                        * values[24 + head * 12 + col * 3 + inner];
                }
            }
            let row_scores = &scores[head * 16 + row * 4..head * 16 + (row + 1) * 4];
            let max = row_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let probabilities = row_scores
                .iter()
                .map(|&score| (score - max).exp())
                .collect::<Vec<_>>();
            let sum: f32 = probabilities.iter().sum();
            for col in 0..3 {
                for inner in 0..4 {
                    expected[head * 12 + row * 3 + col] +=
                        probabilities[inner] / sum * values[48 + head * 12 + inner * 3 + col];
                }
            }
        }
    }
    let output = request
        .run_with_trace(&values, |trace| {
            if trace.name == "scores" {
                assert_eq!(trace.values, scores);
            }
            if trace.op_type == "Slice" {
                assert_eq!(trace.shape.dimensions(), [1, 1, 2, 4, 3]);
                let block = match trace.name {
                    "slice_q" => 0,
                    "slice_k" => 1,
                    _ => 2,
                };
                assert_eq!(trace.values, &values[block * 24..(block + 1) * 24]);
            }
            Ok(())
        })
        .unwrap();
    for (&actual, expected) in output.data().iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs());
    }
    assert!(
        request
            .run_with_trace(&values, |trace| {
                if trace.name == "scores" {
                    Err(RuntimeError::new("stop after activation RHS"))
                } else {
                    Ok(())
                }
            })
            .is_err()
    );
    assert_eq!(request.run(&values).unwrap().data().len(), 24);
}

#[test]
fn activation_rhs_matmul_broadcasts_batches_and_rejects_incompatible_rebinds() {
    let weights = (0..45).map(|i| (i as f32 - 20.) / 8.).collect::<Vec<_>>();
    let loaded = model(
        vec![
            tensor("weights", &[1, 3, 3, 5], &weights),
            tensor("zero", &[], &[0.]),
            node(
                "activation_weights",
                "Add",
                &["weights", "zero"],
                "activation_weights",
                vec![],
            ),
            node(
                "product",
                "MatMul",
                &["x", "activation_weights"],
                "y",
                vec![],
            ),
        ],
        &[-1, -1, 4, 3],
        &[-1, 3, 4, 5],
    );
    let mut request = session(loaded, &[2, 1, 4, 3]);
    let values = (0..24).map(|i| (i as f32 - 10.) / 4.).collect::<Vec<_>>();
    let mut expected = Vec::new();
    for a_batch in 0..2 {
        for b_batch in 0..3 {
            for row in 0..4 {
                for col in 0..5 {
                    let mut value = 0.0f32;
                    for inner in 0..3 {
                        value += values[a_batch * 12 + row * 3 + inner]
                            * weights[b_batch * 15 + inner * 5 + col];
                    }
                    expected.push(value);
                }
            }
        }
    }
    assert_eq!(request.run(&values).unwrap().data(), expected);
    assert_eq!(request.output_shape().dimensions(), [2, 3, 4, 5]);
    assert!(
        request
            .reshape(&[2, 2, 4, 3])
            .unwrap_err()
            .to_string()
            .contains("do not broadcast")
    );
    assert_eq!(request.input_shape().dimensions(), [2, 1, 4, 3]);
    assert_eq!(request.run(&values).unwrap().data(), expected);
    // A rank-two dynamic RHS retains the original shared-weight path.
    let mut shared = session(
        model(
            vec![
                node(
                    "weights",
                    "ReduceMean",
                    &["x"],
                    "weights",
                    vec![ints("axes", &[0]), int("keepdims", 0)],
                ),
                node("product", "MatMul", &["x", "weights"], "y", vec![]),
            ],
            &[2, 2, 2],
            &[2, 2, 2],
        ),
        &[2, 2, 2],
    );
    assert_eq!(
        shared
            .run(&[1., 2., 3., 4., 5., 6., 7., 8.])
            .unwrap()
            .data(),
        [13., 16., 29., 36., 45., 56., 61., 76.]
    );
    let vector = model(
        vec![
            tensor("weights", &[3, 2], &[1.; 6]),
            node("product", "MatMul", &["x", "weights"], "y", vec![]),
        ],
        &[3],
        &[2],
    );
    assert!(
        Session::new(
            Arc::new(CompiledModel::new(vector).unwrap()),
            &[3],
            ExecutionLimits::default()
        )
        .is_err()
    );
}

#[test]
fn slice_bounds_optional_slots_and_failures_preserve_the_previous_binding() {
    let loaded = model(
        vec![
            integers("starts", &[1]),
            integers("ends", &[2]),
            integers("steps", &[1]),
            node(
                "slice",
                "Slice",
                &["x", "starts", "ends", "", "steps"],
                "y",
                vec![],
            ),
        ],
        &[-1, 3],
        &[1, 3],
    );
    let mut request = session(loaded, &[2, 3]);
    assert_eq!(
        request.run(&[1., 2., 3., 4., 5., 6.]).unwrap().data(),
        [4., 5., 6.]
    );
    assert!(request.reshape(&[1, 3]).is_err());
    assert_eq!(request.input_shape().dimensions(), [2, 3]);
    assert_eq!(
        request.run(&[1., 2., 3., 7., 8., 9.]).unwrap().data(),
        [7., 8., 9.]
    );
    for step in [0, -1, 2, i64::MAX] {
        let loaded = model(
            vec![
                integers("starts", &[0]),
                integers("ends", &[2]),
                integers("steps", &[step]),
                node(
                    "slice",
                    "Slice",
                    &["x", "starts", "ends", "", "steps"],
                    "y",
                    vec![],
                ),
            ],
            &[2, 3],
            &[2, 3],
        );
        assert!(
            CompiledModel::new(loaded)
                .err()
                .unwrap()
                .to_string()
                .contains("unit steps")
        );
    }
    let loaded = model(
        vec![
            integers("starts", &[0, 0]),
            integers("ends", &[2, 2]),
            integers("axes", &[0, -2]),
            node(
                "slice",
                "Slice",
                &["x", "starts", "ends", "axes"],
                "y",
                vec![],
            ),
        ],
        &[2, 3],
        &[2, 3],
    );
    assert!(
        Session::new(
            Arc::new(CompiledModel::new(loaded).unwrap()),
            &[2, 3],
            ExecutionLimits::default()
        )
        .err()
        .unwrap()
        .to_string()
        .contains("duplicated")
    );
    let mut clamped = session(
        model(
            vec![
                integers("starts", &[i64::MIN]),
                integers("ends", &[i64::MAX]),
                node("slice", "Slice", &["x", "starts", "ends"], "y", vec![]),
            ],
            &[2, 3],
            &[2, 3],
        ),
        &[2, 3],
    );
    assert_eq!(
        clamped.run(&[1., 2., 3., 4., 5., 6.]).unwrap().data(),
        [1., 2., 3., 4., 5., 6.]
    );
}

#[test]
fn pow_two_native_rounding_counterexamples_use_the_existing_tolerance_and_sqrt_recovers() {
    // Independently captured by .NET 10.0.12 MathF.Pow(x, 2f), seeded F32 bit
    // characterization. The normal result differs by one ULP; underflow differs
    // by one minimum subnormal. These are numerical, not byte-parity contracts.
    let cases = [(0xa96646f7, 0x134f2397), (0x1a20dd79, 0x00000000)];
    let mut request = session(
        model(
            vec![
                tensor("exponent", &[], &[2.]),
                node("square", "Pow", &["x", "exponent"], "y", vec![]),
            ],
            &[2],
            &[2],
        ),
        &[2],
    );
    let input = cases.map(|(input, _)| f32::from_bits(input));
    let output = request.run(&input).unwrap();
    for (&actual, (_, expected)) in output.data().iter().zip(cases) {
        let expected = f32::from_bits(expected);
        assert!(
            (f64::from(actual) - f64::from(expected)).abs()
                <= 1e-5 + 1e-4 * f64::from(expected).abs()
        );
    }
    assert_eq!(output.data()[0].to_bits(), 0x134f2398);
    assert_eq!(output.data()[1].to_bits(), 1);
    let mut square_root = session(
        model(vec![node("sqrt", "Sqrt", &["x"], "y", vec![])], &[2], &[2]),
        &[2],
    );
    assert!(square_root.run(&[-1., 4.]).is_err());
    assert_eq!(square_root.run(&[4., 9.]).unwrap().data(), [2., 3.]);
    let mut subtract = session(
        model(
            vec![
                tensor("rows", &[2, 1], &[10., 20.]),
                node("subtract", "Sub", &["rows", "x"], "y", vec![]),
            ],
            &[2, 3],
            &[2, 3],
        ),
        &[2, 3],
    );
    assert_eq!(
        subtract.run(&[1., 2., 3., 4., 5., 6.]).unwrap().data(),
        [9., 8., 7., 16., 15., 14.]
    );
}
