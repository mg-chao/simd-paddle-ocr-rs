use super::*;

fn node(op_type: &str, attributes: &[(&str, AttributeValue)]) -> RawNode {
    RawNode {
        source_indices: vec![0],
        name: "attribute_regression".into(),
        op_type: op_type.into(),
        domain: String::new(),
        inputs: vec!["x".into()],
        outputs: vec!["y".into()],
        attributes: attributes
            .iter()
            .map(|(name, value)| Attribute {
                name: (*name).into(),
                value: value.clone(),
            })
            .collect(),
    }
}

fn normalize_node(node: RawNode) -> Result<Model> {
    normalize_nodes(vec![node])
}

fn normalize_nodes(mut nodes: Vec<RawNode>) -> Result<Model> {
    for (index, node) in nodes.iter_mut().enumerate() {
        node.source_indices = vec![index];
    }
    let value = |name: &str| ValueInfo {
        name: name.into(),
        dtype: DType::F32,
        shape: ShapeSpec::UnknownRank,
    };
    normalize(
        RawModel {
            ir_version: 10,
            opsets: BTreeMap::from([(String::new(), 18)]),
            graph: RawGraph {
                nodes,
                inputs: vec![value("x")],
                outputs: vec![value("y")],
                ..Default::default()
            },
            source_bytes: 0,
        },
        &LoadLimits::default(),
    )
}

fn assert_rejected(op_type: &str, name: &str, value: AttributeValue) {
    let mut attributes = vec![("kernel_shape", AttributeValue::Ints(vec![2, 2]))];
    attributes.push((name, value));
    let error = normalize_node(node(op_type, &attributes)).expect_err("attribute must not be lost");
    let message = error.to_string();
    assert!(message.contains("attribute_regression"), "{message}");
    assert!(message.contains(name), "{message}");
}

#[test]
fn unsupported_spatial_attributes_are_not_silently_discarded() {
    for (op_type, name, value) in [
        (
            "ConvTranspose",
            "output_padding",
            AttributeValue::Ints(vec![1, 0]),
        ),
        (
            "ConvTranspose",
            "output_shape",
            AttributeValue::Ints(vec![9, 8]),
        ),
        ("MaxPool", "dilations", AttributeValue::Ints(vec![2, 2])),
        ("MaxPool", "storage_order", AttributeValue::Int(1)),
        (
            "MaxPool",
            "auto_pad",
            AttributeValue::String(b"SAME_LOWER".to_vec()),
        ),
        (
            "AveragePool",
            "auto_pad",
            AttributeValue::String(b"SAME_UPPER".to_vec()),
        ),
        ("AveragePool", "dilations", AttributeValue::Ints(vec![2, 1])),
        ("Conv", "unrecognized_semantics", AttributeValue::Int(1)),
    ] {
        assert_rejected(op_type, name, value);
    }
}

#[test]
fn removed_nodes_must_not_hide_unsupported_attributes() {
    for op_type in ["GlobalAveragePool", "Identity"] {
        let error = normalize_node(node(op_type, &[("unsupported", AttributeValue::Int(1))]))
            .expect_err("normalization must validate before removing attributes");
        assert!(error.to_string().contains("unsupported"));
        assert!(error.to_string().contains("attribute_regression"));
    }
}

#[test]
fn resize_attributes_that_change_sampling_are_rejected() {
    for (name, value) in [
        ("antialias", AttributeValue::Int(1)),
        ("exclude_outside", AttributeValue::Int(1)),
        ("axes", AttributeValue::Ints(vec![2, 3])),
        (
            "keep_aspect_ratio_policy",
            AttributeValue::String(b"not_larger".to_vec()),
        ),
    ] {
        let error = normalize_node(node(
            "Resize",
            &[
                ("scales", AttributeValue::Floats(vec![1.0, 1.0, 2.0, 2.0])),
                (name, value),
            ],
        ))
        .expect_err("sampling semantics must not be lost");
        assert!(error.to_string().contains(name));
    }
}

#[test]
fn neutral_spatial_defaults_preserve_typed_parameters() {
    for op_type in ["ConvTranspose", "MaxPool", "AveragePool"] {
        let mut attributes = vec![
            ("kernel_shape", AttributeValue::Ints(vec![2, 2])),
            ("auto_pad", AttributeValue::String(b"NOTSET".to_vec())),
        ];
        if op_type == "ConvTranspose" {
            attributes.push(("output_padding", AttributeValue::Ints(vec![0, 0])));
        } else {
            attributes.push(("dilations", AttributeValue::Ints(vec![1, 1])));
            if op_type == "MaxPool" {
                attributes.push(("storage_order", AttributeValue::Int(0)));
            }
        }
        let model = normalize_node(node(op_type, &attributes)).unwrap();
        match &model.graph.nodes[0].operator {
            Operator::ConvTranspose(parameters) => {
                assert_eq!(parameters.kernel_shape, [2, 2]);
                assert_eq!(parameters.dilations, [1, 1]);
                assert_eq!(parameters.pads, [0; 4]);
                assert_eq!(parameters.group, 1);
            }
            Operator::MaxPool(parameters) | Operator::AveragePool(parameters) => {
                assert_eq!(parameters.kernel_shape, [2, 2]);
                assert_eq!(parameters.pads, [0; 4]);
                assert_eq!(parameters.ceil_mode, 0);
                assert_eq!(parameters.count_include_pad, 0);
            }
            attributes => panic!("unexpected {op_type} parameters: {attributes:?}"),
        }
    }
}

#[test]
fn data_slice_rejects_attribute_form_parameters() {
    for name in ["starts", "ends", "axes", "steps"] {
        let error = normalize_node(node("Slice", &[(name, AttributeValue::Ints(vec![0]))]))
            .expect_err("Slice parameters must use constant integer inputs");
        assert!(error.to_string().contains(name));
    }
    let model = normalize_node(node("Slice", &[])).unwrap();
    assert!(matches!(model.graph.nodes[0].operator, Operator::Slice));
}

#[test]
fn shape_slice_attributes_survive_normalization_and_dynamic_rebinding() {
    use crate::runtime::{CpuCompiledModel, ExecutionLimits, Session};
    use std::sync::Arc;

    let mut shape = node("Shape", &[]);
    shape.outputs = vec!["shape".into()];
    let mut slice = node(
        "Slice",
        &[
            ("starts", AttributeValue::Ints(vec![0])),
            ("ends", AttributeValue::Ints(vec![2])),
            ("axes", AttributeValue::Ints(vec![0])),
            ("steps", AttributeValue::Ints(vec![1])),
        ],
    );
    slice.inputs = vec!["shape".into()];
    slice.outputs = vec!["target".into()];
    let mut reshape = node("Reshape", &[]);
    reshape.inputs = vec!["x".into(), "target".into()];
    let model = Arc::new(normalize_nodes(vec![shape, slice, reshape]).unwrap());
    assert_eq!(model.graph().nodes.len(), 1);
    assert_eq!(model.shape_recipes().len(), 1);
    let compiled = Arc::new(CpuCompiledModel::new(model).unwrap());
    let mut session = Session::new_serial(compiled, &[2, 3], ExecutionLimits::default()).unwrap();
    for width in [3, 5, 3] {
        session.reshape(&[2, width]).unwrap();
        let input: Vec<_> = (0..2 * width).map(|value| value as f32).collect();
        let output = session.run(&input).unwrap();
        assert_eq!(output.shape().dimensions(), &[2, width]);
        assert_eq!(output.data(), input);
    }
}

#[test]
fn unsupported_batch_normalization_training_is_rejected_before_folding() {
    let error = normalize_node(node(
        "BatchNormalization",
        &[("training_mode", AttributeValue::Int(1))],
    ))
    .expect_err("training mode must not be removed by normalization");
    assert!(error.to_string().contains("training_mode"));
    for name in ["spatial", "is_test"] {
        let error = normalize_node(node(
            "BatchNormalization",
            &[(name, AttributeValue::Int(0))],
        ))
        .expect_err("legacy nonspatial/training BN must not be treated as inference");
        assert!(error.to_string().contains(name));
    }
    normalize_node(node(
        "BatchNormalization",
        &[
            ("spatial", AttributeValue::Int(1)),
            ("is_test", AttributeValue::Int(1)),
        ],
    ))
    .unwrap();
}

#[test]
fn malformed_neutral_defaults_are_rejected_with_attribute_context() {
    for (op_type, name, value) in [
        (
            "ConvTranspose",
            "output_padding",
            AttributeValue::Ints(vec![0]),
        ),
        ("ConvTranspose", "output_padding", AttributeValue::Int(0)),
        ("MaxPool", "dilations", AttributeValue::Ints(vec![1])),
        ("MaxPool", "storage_order", AttributeValue::Float(0.0)),
        ("AveragePool", "auto_pad", AttributeValue::Int(0)),
    ] {
        assert_rejected(op_type, name, value);
    }
}

#[test]
fn supported_padding_is_normalized_without_dropping_other_semantics() {
    for op_type in ["Conv", "MaxPool"] {
        let model = normalize_node(node(
            op_type,
            &[
                ("kernel_shape", AttributeValue::Ints(vec![2, 2])),
                ("auto_pad", AttributeValue::String(b"SAME_UPPER".to_vec())),
            ],
        ))
        .unwrap();
        let pads = match &model.graph.nodes[0].operator {
            Operator::Conv(a) => a.pads,
            Operator::MaxPool(a) | Operator::AveragePool(a) => a.pads,
            _ => panic!("wrong effective attributes"),
        };
        assert_eq!(pads, [0, 0, 1, 1]);
        let error = normalize_node(node(
            op_type,
            &[
                ("kernel_shape", AttributeValue::Ints(vec![2, 2])),
                ("strides", AttributeValue::Ints(vec![2, 2])),
                ("auto_pad", AttributeValue::String(b"SAME_UPPER".to_vec())),
            ],
        ))
        .expect_err("stride-dependent padding is unsupported");
        assert!(error.to_string().contains("auto_pad"));
    }
    for op_type in ["Conv", "ConvTranspose", "MaxPool", "AveragePool"] {
        normalize_node(node(
            op_type,
            &[
                ("kernel_shape", AttributeValue::Ints(vec![2, 2])),
                ("auto_pad", AttributeValue::String(b"VALID".to_vec())),
            ],
        ))
        .unwrap();
        let error = normalize_node(node(
            op_type,
            &[
                ("kernel_shape", AttributeValue::Ints(vec![2, 2])),
                ("pads", AttributeValue::Ints(vec![1, 1, 1, 1])),
                ("auto_pad", AttributeValue::String(b"VALID".to_vec())),
            ],
        ))
        .expect_err("explicit padding conflicts with auto_pad");
        assert!(error.to_string().contains("auto_pad"));
    }
}

#[test]
fn resize_defaults_follow_the_declared_opset() {
    let model = normalize_node(node(
        "Resize",
        &[
            ("scales", AttributeValue::Floats(vec![1.0, 1.0, 2.0, 2.0])),
            ("antialias", AttributeValue::Int(0)),
            ("exclude_outside", AttributeValue::Int(0)),
            ("cubic_coeff_a", AttributeValue::Float(-0.75)),
            ("extrapolation_value", AttributeValue::Float(0.0)),
            (
                "keep_aspect_ratio_policy",
                AttributeValue::String(b"stretch".to_vec()),
            ),
            ("axes", AttributeValue::Ints(vec![0, 1, 2, 3])),
        ],
    ))
    .unwrap();
    let Operator::Resize(attributes) = &model.graph.nodes[0].operator else {
        panic!("expected resize attributes");
    };
    assert_eq!(attributes.scales, [1.0, 1.0, 2.0, 2.0]);
    assert_eq!(attributes.mode, b"nearest");
    assert_eq!(attributes.coordinate_transformation_mode, b"half_pixel");
    assert_eq!(attributes.nearest_mode, b"round_prefer_floor");
}

#[test]
fn constant_nodes_cannot_hide_extra_value_forms() {
    let error = normalize_node(node(
        "Constant",
        &[
            ("value_int", AttributeValue::Int(1)),
            ("value_float", AttributeValue::Float(2.0)),
        ],
    ))
    .expect_err("constant must have one authoritative value");
    assert!(error.to_string().contains("attribute"));
    assert!(error.to_string().contains("attribute_regression"));
}
