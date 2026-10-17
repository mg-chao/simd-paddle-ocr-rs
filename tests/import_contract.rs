use simd_paddle_ocr_rs::{
    error::ErrorKind,
    model::{LoadOptions, Model, Operator, OptimizationPolicy},
    onnx::LoadLimits,
    runtime::{CompileOptions, CompiledModel, ExecutionLimits, Session},
};
use std::sync::Arc;

fn var(mut v: u64) -> Vec<u8> {
    let mut b = vec![];
    loop {
        let x = (v & 127) as u8;
        v >>= 7;
        b.push(x | if v == 0 { 0 } else { 128 });
        if v == 0 {
            return b;
        }
    }
}
fn num(f: u64, v: u64) -> Vec<u8> {
    [var(f << 3), var(v)].concat()
}
fn bytes(f: u64, b: &[u8]) -> Vec<u8> {
    [var(f << 3 | 2), var(b.len() as u64), b.to_vec()].concat()
}
fn attr_s(k: &str, v: &str) -> Vec<u8> {
    bytes(
        5,
        &[bytes(1, k.as_bytes()), bytes(4, v.as_bytes()), num(20, 3)].concat(),
    )
}
fn attr_is(k: &str, v: &[i64]) -> Vec<u8> {
    bytes(
        5,
        &[
            bytes(1, k.as_bytes()),
            bytes(
                8,
                &v.iter().flat_map(|&x| var(x as u64)).collect::<Vec<_>>(),
            ),
            num(20, 7),
        ]
        .concat(),
    )
}
fn node(op: &str, inputs: &[&str], out: &str, domain: &str, attrs: Vec<u8>) -> Vec<u8> {
    bytes(
        1,
        &[
            inputs.iter().flat_map(|s| bytes(1, s.as_bytes())).collect(),
            bytes(2, out.as_bytes()),
            bytes(4, op.as_bytes()),
            bytes(7, domain.as_bytes()),
            attrs,
        ]
        .concat(),
    )
}
fn info(name: &str, dims: &[usize]) -> Vec<u8> {
    [
        bytes(1, name.as_bytes()),
        bytes(
            2,
            &bytes(
                1,
                &[
                    num(1, 1),
                    bytes(
                        2,
                        &dims
                            .iter()
                            .flat_map(|&d| bytes(1, &num(1, d as u64)))
                            .collect::<Vec<_>>(),
                    ),
                ]
                .concat(),
            ),
        ),
    ]
    .concat()
}
fn tensor(name: &str, dims: &[usize], data: &[f32]) -> Vec<u8> {
    bytes(
        5,
        &[
            dims.iter().flat_map(|&d| num(1, d as u64)).collect(),
            num(2, 1),
            bytes(8, name.as_bytes()),
            bytes(
                9,
                &data
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        ]
        .concat(),
    )
}
fn source(nodes: Vec<u8>, init: Vec<u8>, input: &[usize], output: &[usize]) -> Vec<u8> {
    [
        num(1, 8),
        bytes(8, &num(2, 13)),
        bytes(8, &[bytes(1, b"custom"), num(2, 1)].concat()),
        bytes(
            7,
            &[
                nodes,
                init,
                bytes(11, &info("x", input)),
                bytes(12, &info("y", output)),
            ]
            .concat(),
        ),
    ]
    .concat()
}

fn execute(model: Model, shape: &[usize], input: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let compiled =
        Arc::new(CompiledModel::compile(Arc::new(model), CompileOptions::reference()).unwrap());
    let mut session = Session::new_serial(compiled, shape, ExecutionLimits::default()).unwrap();
    let result = session.run(input).unwrap();
    (result.shape().dimensions().to_vec(), result.data().to_vec())
}

#[test]
fn deep_shape_graph_returns_an_error_on_a_normal_thread_stack() {
    let mut nodes = node("Shape", &["x"], "s0", "", vec![]);
    for i in 1..=20_000 {
        nodes.extend(node(
            "Squeeze",
            &[&format!("s{}", i - 1)],
            &format!("s{i}"),
            "",
            vec![],
        ));
    }
    nodes.extend(node("Reshape", &["x", "s20000"], "y", "", vec![]));
    let source = source(nodes, vec![], &[1], &[1]);
    let error = std::thread::spawn(move || Model::load_bytes(&source).unwrap_err())
        .join()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::ResourceLimit);
    assert!(error.to_string().contains("shape dependency"));
}

#[test]
fn long_alias_chains_resolve_without_recursion() {
    let mut nodes = node("Identity", &["x"], "s0", "", vec![]);
    for i in 1..=20_000 {
        nodes.extend(node(
            "Identity",
            &[&format!("s{}", i - 1)],
            &format!("s{i}"),
            "",
            vec![],
        ));
    }
    nodes.extend(node("Identity", &["s20000"], "y", "", vec![]));
    let model = Model::load_bytes(&source(nodes, vec![], &[1], &[1])).unwrap();
    assert!(model.graph().nodes.is_empty());
    assert_eq!(model.graph().inputs, model.graph().outputs);
}

#[test]
fn domains_are_checked_before_erasure_or_folding() {
    for op in ["Identity", "Constant"] {
        let source = source(node(op, &["x"], "y", "custom", vec![]), vec![], &[1], &[1]);
        assert_eq!(
            Model::load_bytes(&source).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
    }
    let source = bn_source("custom");
    assert_eq!(
        Model::load_bytes(&source).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
}

fn resize_source(coordinate: bool, nearest: bool, alias: bool) -> Vec<u8> {
    let mut attrs = vec![];
    if coordinate {
        attrs.extend(attr_s("coordinate_transformation_mode", "asymmetric"));
    }
    if nearest {
        attrs.extend(attr_s("nearest_mode", "floor"));
    }
    let mut nodes = if alias {
        node("Identity", &["scales"], "alias", "", vec![])
    } else {
        vec![]
    };
    nodes.extend(node(
        "Resize",
        &["x", "", if alias { "alias" } else { "scales" }],
        "y",
        "",
        attrs,
    ));
    source(
        nodes,
        tensor("scales", &[4], &[1., 1., 2., 2.]),
        &[1, 1, 1, 2],
        &[1, 1, 2, 4],
    )
}

#[test]
fn resize_uses_opset_defaults_and_rejects_unsupported_sampling() {
    for (coordinate, nearest) in [(false, true), (true, false), (false, false)] {
        let model = Model::load_bytes(&resize_source(coordinate, nearest, false)).unwrap();
        let Operator::Resize(attrs) = &model.graph().nodes[0].operator else {
            panic!()
        };
        if !coordinate {
            assert_eq!(attrs.coordinate_transformation_mode, b"half_pixel");
        }
        if !nearest {
            assert_eq!(attrs.nearest_mode, b"round_prefer_floor");
        }
        assert!(CompiledModel::compile(Arc::new(model), CompileOptions::reference()).is_err());
    }
}

#[test]
fn resize_constant_alias_has_the_same_samples_as_a_direct_initializer() {
    for alias in [false, true] {
        let model = Model::load_bytes(&resize_source(true, true, alias)).unwrap();
        let (shape, values) = execute(model, &[1, 1, 1, 2], &[10., 20.]);
        assert_eq!(shape, [1, 1, 2, 4]);
        assert_eq!(values, [10., 10., 20., 20., 10., 10., 20., 20.]);
    }
}

#[test]
fn global_average_pool_reduces_every_spatial_axis() {
    for rank in 3..=8 {
        let mut input_shape = vec![1; rank];
        input_shape[rank - 1] = 2;
        let output_shape = vec![1; rank];
        let model = Model::load_bytes(&source(
            node("GlobalAveragePool", &["x"], "y", "", vec![]),
            vec![],
            &input_shape,
            &output_shape,
        ))
        .unwrap();
        assert_eq!(
            execute(model, &input_shape, &[2., 4.]),
            (output_shape, vec![3.])
        );
    }
}

fn branches(kh: usize, kw: usize) -> Vec<u8> {
    let first = node(
        "Conv",
        &["x", "wh"],
        "a",
        "",
        [
            attr_is("kernel_shape", &[kh as i64, 1]),
            attr_is("pads", &[(kh / 2) as i64, 0, (kh / 2) as i64, 0]),
        ]
        .concat(),
    );
    let second = node(
        "Conv",
        &["x", "ww"],
        "b",
        "",
        [
            attr_is("kernel_shape", &[1, kw as i64]),
            attr_is("pads", &[0, (kw / 2) as i64, 0, (kw / 2) as i64]),
        ]
        .concat(),
    );
    source(
        [first, second, node("Add", &["a", "b"], "y", "", vec![])].concat(),
        [
            tensor("wh", &[1, 1, kh, 1], &vec![1.; kh]),
            tensor("ww", &[1, 1, 1, kw], &vec![1.; kw]),
        ]
        .concat(),
        &[1, 1, 3, 3],
        &[1, 1, 3, 3],
    )
}

#[test]
fn orthogonal_branches_do_not_amplify_constant_storage() {
    let source = branches(1023, 1023);
    let model = Model::load_bytes_with_limits(
        &source,
        &LoadLimits {
            max_input_bytes: source.len() as u64,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(model.graph().nodes.len(), 3);
    assert_eq!(model.normalized_weight_bytes(), 2 * 1023 * 4);
}

fn bn_source(domain: &str) -> Vec<u8> {
    source(
        [
            node(
                "Conv",
                &["x", "w"],
                "c",
                "",
                attr_is("kernel_shape", &[1, 1]),
            ),
            node(
                "BatchNormalization",
                &["c", "scale", "bias", "mean", "variance"],
                "y",
                domain,
                vec![],
            ),
        ]
        .concat(),
        [
            tensor("w", &[1, 1, 1, 1], &[2.]),
            tensor("scale", &[1], &[3.]),
            tensor("bias", &[1], &[4.]),
            tensor("mean", &[1], &[1.]),
            tensor("variance", &[1], &[4.]),
        ]
        .concat(),
        &[1, 1, 1, 2],
        &[1, 1, 1, 2],
    )
}

#[test]
fn arithmetic_preserving_import_is_an_independent_bn_reference() {
    let source = bn_source("");
    for policy in [
        OptimizationPolicy::Optimized,
        OptimizationPolicy::PreserveArithmetic,
    ] {
        let model = Model::load_bytes_with_options(
            &source,
            &LoadOptions {
                optimization: policy,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            model.graph().nodes.len(),
            if policy == OptimizationPolicy::Optimized {
                1
            } else {
                2
            }
        );
        let (_, values) = execute(model, &[1, 1, 1, 2], &[1., 2.]);
        for (value, x) in values.into_iter().zip([1_f64, 2.]) {
            let expected = ((2. * x - 1.) * 3. / (4_f64 + 1e-5).sqrt() + 4.) as f32;
            assert!((value - expected).abs() < 2e-6, "{value} vs {expected}");
        }
    }
}

#[test]
fn branch_merge_matches_the_original_arithmetic_and_known_values() {
    let source = branches(1, 3);
    for policy in [
        OptimizationPolicy::Optimized,
        OptimizationPolicy::PreserveArithmetic,
    ] {
        let model = Model::load_bytes_with_options(
            &source,
            &LoadOptions {
                optimization: policy,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            model.graph().nodes.len(),
            if policy == OptimizationPolicy::Optimized {
                1
            } else {
                3
            }
        );
        let (_, values) = execute(model, &[1, 1, 3, 3], &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        assert_eq!(values, [4., 8., 8., 13., 20., 17., 22., 32., 26.]);
    }
}

#[test]
fn budgets_reject_constants_and_skip_optional_rewrites_before_allocation() {
    let source = bn_source("");
    let limits = LoadLimits {
        max_constant_bytes: 19,
        ..Default::default()
    };
    assert_eq!(
        Model::load_bytes_with_limits(&source, &limits)
            .unwrap_err()
            .kind(),
        ErrorKind::ResourceLimit
    );
    for limits in [
        LoadLimits {
            max_constant_bytes: 20,
            ..Default::default()
        },
        LoadLimits {
            max_generated_bytes: 0,
            ..Default::default()
        },
    ] {
        let model = Model::load_bytes_with_limits(&source, &limits).unwrap();
        assert_eq!(model.graph().nodes.len(), 2);
        assert_eq!(model.normalized_weight_bytes(), 20);
    }
    let limits = LoadLimits {
        max_graph_nodes: 1,
        ..Default::default()
    };
    assert_eq!(
        Model::load_bytes_with_limits(&source, &limits)
            .unwrap_err()
            .kind(),
        ErrorKind::ResourceLimit
    );
}

#[test]
fn missing_opset_and_redefined_inputs_are_rejected_before_alias_erasure() {
    let graph = [
        node("Identity", &["x"], "y", "", vec![]),
        bytes(11, &info("x", &[1])),
        bytes(12, &info("y", &[1])),
    ]
    .concat();
    let missing = [num(1, 8), bytes(7, &graph)].concat();
    assert_eq!(
        Model::load_bytes(&missing).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    let duplicate = source(
        node("Identity", &["x"], "x", "", vec![]),
        vec![],
        &[1],
        &[1],
    );
    assert!(Model::load_bytes(&duplicate).is_err());
}

#[test]
fn alias_cycles_are_reported_and_legacy_resize_defaults_are_supported() {
    let cycle = source(
        [
            node("Identity", &["y"], "a", "", vec![]),
            node("Identity", &["a"], "y", "", vec![]),
        ]
        .concat(),
        vec![],
        &[1],
        &[1],
    );
    assert!(
        Model::load_bytes(&cycle)
            .unwrap_err()
            .to_string()
            .contains("alias cycle")
    );
    let graph = [
        node("Resize", &["x", "scales"], "y", "", vec![]),
        tensor("scales", &[4], &[1., 1., 2., 2.]),
        bytes(11, &info("x", &[1, 1, 1, 2])),
        bytes(12, &info("y", &[1, 1, 2, 4])),
    ]
    .concat();
    let legacy = [num(1, 8), bytes(8, &num(2, 10)), bytes(7, &graph)].concat();
    let model = Model::load_bytes(&legacy).unwrap();
    assert_eq!(
        execute(model, &[1, 1, 1, 2], &[10., 20.]).1,
        [10., 10., 20., 20., 10., 10., 20., 20.]
    );
}
