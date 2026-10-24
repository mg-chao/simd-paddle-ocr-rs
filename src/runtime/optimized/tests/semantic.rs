use super::*;

const INPUT_SHAPE: [usize; 4] = [2, 2, 2, 3];

fn input() -> Vec<f32> {
    (0..INPUT_SHAPE.iter().product())
        .map(|i| (i as f32 - 13.) / 7.)
        .collect()
}

fn constants() -> Vec<Vec<u8>> {
    vec![
        tensor("w", &[2, 2, 1, 1], &[1., 0.25, -0.5, 2.]),
        tensor("w2", &[2, 2, 1, 1], &[-0.75, 0.5, 1.25, 0.25]),
        tensor("bias", &[1, 2, 1, 1], &[0.2, -0.3]),
        tensor("scale", &[], &[-0.75]),
    ]
}

fn options() -> impl Iterator<Item = CompileOptions> {
    [LayoutPolicy::Nchw, LayoutPolicy::Auto]
        .into_iter()
        .flat_map(|layout| {
            [
                KernelPolicy::Scalar,
                KernelPolicy::Portable4,
                KernelPolicy::Auto,
            ]
            .into_iter()
            .map(move |kernels| CompileOptions {
                cpu: crate::runtime::CpuCompileOptions { kernels, layout },
                fusion: true,
                ..Default::default()
            })
        })
}

fn check(model: Arc<Model>, expected_fused: usize) {
    let values = input();
    let mut reference = session(model.clone(), CompileOptions::reference(), &INPUT_SHAPE);
    let expected = reference.run(&values).unwrap().data().to_vec();
    for options in options() {
        let mut optimized = session(model.clone(), options, &INPUT_SHAPE);
        compare(&expected, optimized.run(&values).unwrap().data());
        assert_eq!(optimized.statistics().fused_nodes, expected_fused);
    }
}

#[test]
fn gapped_bias_and_activation_keep_unrelated_live_nodes() {
    for (convolution, activation) in [("Conv", "Relu"), ("Conv", "Sigmoid")] {
        let mut parts = constants();
        parts.extend([
            node(convolution, &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
            node("Mul", &["x", "scale"], "branch", vec![]),
            node("Add", &["c", "bias"], "biased", vec![]),
            node("Relu", &["branch"], "other", vec![]),
            node(activation, &["biased"], "activated", vec![]),
            node("Concat", &["activated", "other"], "y", axis(1)),
        ]);
        check(model(parts, &[-1, 2, -1, -1], &[-1, 4, -1, -1]), 2);
    }
    let mut parts = constants();
    parts.push(tensor(
        "wt",
        &[2, 2, 2, 2],
        &(0..16).map(|i| (i as f32 - 7.) / 11.).collect::<Vec<_>>(),
    ));
    let attributes = [attr("kernel_shape", &[2, 2]), attr("strides", &[2, 2])].concat();
    parts.extend([
        node("ConvTranspose", &["x", "wt"], "c", attributes.clone()),
        node("ConvTranspose", &["x", "wt"], "branch", attributes),
        node("Add", &["c", "bias"], "biased", vec![]),
        node("Mul", &["branch", "scale"], "other", vec![]),
        node("Relu", &["biased"], "activated", vec![]),
        node("Concat", &["activated", "other"], "y", axis(1)),
    ]);
    check(model(parts, &[-1, 2, -1, -1], &[-1, 4, -1, -1]), 2);
}

fn residual_producer_model(extra_consumer: bool) -> Arc<Model> {
    let mut parts = constants();
    parts.extend([
        node("Conv", &["x", "w"], "a", attr("kernel_shape", &[1, 1])),
        node("Mul", &["x", "scale"], "gap", vec![]),
        node("Conv", &["x", "w2"], "b", attr("kernel_shape", &[1, 1])),
        node("Relu", &["b"], "residual", vec![]),
        node("Add", &["a", "residual"], "joined", vec![]),
    ]);
    if extra_consumer {
        parts.extend([
            node("Add", &["b", "gap"], "other", vec![]),
            node("Add", &["joined", "other"], "y", vec![]),
        ]);
    } else {
        parts.extend([
            node("Relu", &["joined"], "activated", vec![]),
            node("Add", &["activated", "gap"], "y", vec![]),
        ]);
    }
    model(parts, &[-1, 2, -1, -1], &[-1, 2, -1, -1])
}

#[test]
fn adjacent_residual_producer_cannot_absorb_reserved_semantic_members() {
    let model = residual_producer_model(false);
    check(model.clone(), 4);
    let optimized = session(model, CompileOptions::default(), &INPUT_SHAPE);
    let native = optimized
        .optimized
        .as_ref()
        .unwrap()
        .plan
        .steps
        .iter()
        .filter_map(|step| match &step.op {
            Operation::Native {
                index, epilogues, ..
            } => Some((*index, epilogues.len())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(native, [(1, 0), (2, 1), (0, 3)]);
}

#[test]
fn adjacent_use_checks_include_consumers_after_a_reserved_boundary() {
    // Node 2's output is read by node 5 as well as its adjacent Relu. Node 4
    // belongs to the delayed node-0 group and must not hide that later use.
    check(residual_producer_model(true), 2);
}

#[test]
fn shared_dataflow_keeps_prefix_exclusivity_and_selected_outputs() {
    let mut parts = constants();
    parts.extend([
        node("Conv", &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
        node("Mul", &["x", "scale"], "gap", vec![]),
        node("Add", &["c", "bias"], "biased", vec![]),
        node("Relu", &["biased"], "activated", vec![]),
        node("Add", &["c", "activated"], "joined", vec![]),
        node("Add", &["joined", "gap"], "y", vec![]),
    ]);
    let model = model(parts, &[-1, 2, -1, -1], &[-1, 2, -1, -1]);
    let values = input();
    let mut reference = session(model.clone(), CompileOptions::reference(), &INPUT_SHAPE);
    let mut expected = Vec::new();
    reference
        .run_with_trace(&values, |event| {
            expected.push(event.values.to_vec());
            Ok(())
        })
        .unwrap();
    for options in options() {
        let compiled = Arc::new(CpuCompiledModel::compile(model.clone(), options).unwrap());
        let mut full =
            CpuSession::new_serial(compiled.clone(), &INPUT_SHAPE, ExecutionLimits::default())
                .unwrap();
        let mut prefix = CpuSession::new_prefix(
            compiled.clone(),
            &INPUT_SHAPE,
            ExecutionLimits::default(),
            3,
        )
        .unwrap();
        // The suffix's second use of c prevents full-graph fusion, but must
        // not prevent the prefix from fusing Conv -> Add -> Relu across a gap.
        assert!(Arc::ptr_eq(&full.compiled, &prefix.compiled));
        assert_eq!(full.statistics().fused_nodes, 0);
        assert_eq!(prefix.statistics().fused_nodes, 2);
        compare(&expected[5], full.run(&values).unwrap().data());
        compare(&expected[3], prefix.run(&values).unwrap().data());

        // A selected intermediate remains available even when binding more
        // instructions than its producer; do not absorb it into an epilogue.
        let selected = compiled.program.instructions[2].output;
        let mut selected_session = CpuSession::new_selected(
            compiled.clone(),
            &INPUT_SHAPE,
            ExecutionLimits::default(),
            4,
            selected,
        )
        .unwrap();
        assert_eq!(selected_session.statistics().fused_nodes, 1);
        compare(&expected[2], selected_session.run(&values).unwrap().data());
        let graph_before = format!("{:?}", compiled.program.model.graph());
        let mut traced = 0;
        full.run_with_trace(&values, |event| {
            compare(&expected[event.index], event.values);
            traced += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(traced, expected.len());
        assert_eq!(
            graph_before,
            format!("{:?}", compiled.program.model.graph())
        );
    }
}

#[test]
fn externally_consumed_gapped_intermediates_remain_materialized() {
    let mut parts = constants();
    parts.extend([
        node("Conv", &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
        node("Mul", &["x", "scale"], "gap", vec![]),
        node("Add", &["c", "bias"], "biased", vec![]),
        node("Relu", &["c"], "external", vec![]),
        node("Relu", &["biased"], "activated", vec![]),
        node("Add", &["activated", "external"], "joined", vec![]),
        node("Add", &["joined", "gap"], "y", vec![]),
    ]);
    check(model(parts, &[-1, 2, -1, -1], &[-1, 2, -1, -1]), 0);
}

#[test]
fn gapped_self_add_stops_at_a_materialized_group_output() {
    let mut parts = constants();
    parts.extend([
        node("Conv", &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
        node("Mul", &["x", "scale"], "gap", vec![]),
        node("Add", &["c", "bias"], "biased", vec![]),
        node("Add", &["biased", "biased"], "doubled", vec![]),
        node("Add", &["doubled", "gap"], "y", vec![]),
    ]);
    check(model(parts, &[-1, 2, -1, -1], &[-1, 2, -1, -1]), 1);
}

#[test]
fn gapped_rank_promoting_add_preserves_its_output_shape() {
    let mut parts = constants();
    parts.extend([
        tensor("promote", &[1, 1, 1, 1, 1], &[0.5]),
        node("Conv", &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
        node("Mul", &["x", "scale"], "gap", vec![]),
        node("Add", &["c", "promote"], "promoted", vec![]),
        node("Sigmoid", &["promoted"], "activated", vec![]),
        node("Add", &["activated", "gap"], "y", vec![]),
    ]);
    let model = model(parts, &[-1, 2, -1, -1], &[1, -1, 2, -1, -1]);
    check(model.clone(), 0);
    let optimized = session(model, CompileOptions::default(), &INPUT_SHAPE);
    assert_eq!(optimized.output_shape().dimensions(), [1, 2, 2, 2, 3]);
}

#[test]
fn gapped_full_constant_residual_only_fuses_with_canonical_storage() {
    let mut parts = constants();
    parts.extend([
        tensor("full", &INPUT_SHAPE, &input()),
        node("Conv", &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
        node("Mul", &["x", "scale"], "gap", vec![]),
        node("Add", &["c", "full"], "biased", vec![]),
        node("Sigmoid", &["biased"], "activated", vec![]),
        node("Add", &["activated", "gap"], "y", vec![]),
    ]);
    let model = model(parts, &[-1, 2, -1, -1], &[-1, 2, -1, -1]);
    let values = input();
    let mut reference = session(model.clone(), CompileOptions::reference(), &INPUT_SHAPE);
    let expected = reference.run(&values).unwrap().data().to_vec();
    for options in options() {
        let expected_fused = if options.cpu.layout == LayoutPolicy::Nchw {
            3
        } else {
            0
        };
        let mut optimized = session(model.clone(), options, &INPUT_SHAPE);
        compare(&expected, optimized.run(&values).unwrap().data());
        assert_eq!(optimized.statistics().fused_nodes, expected_fused);
    }
}

#[test]
fn gapped_group_prefix_endpoints_and_trace_keep_each_normalized_value() {
    let model = residual_producer_model(false);
    let values = input();
    let mut reference = session(model.clone(), CompileOptions::reference(), &INPUT_SHAPE);
    let mut expected = Vec::new();
    reference
        .run_with_trace(&values, |event| {
            expected.push((event.output_id, event.shape, event.values.to_vec()));
            Ok(())
        })
        .unwrap();
    for options in options() {
        let mut optimized = session(model.clone(), options, &INPUT_SHAPE);
        let mut seen = 0;
        optimized
            .run_with_trace(&values, |event| {
                assert_eq!(event.index, seen);
                assert_eq!(event.layout, "nchw");
                assert_eq!(event.output_id, expected[seen].0);
                assert_eq!(event.shape, expected[seen].1);
                compare(&expected[seen].2, event.values);
                seen += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(seen, expected.len());
        for (last, (_, shape, expected)) in expected.iter().enumerate() {
            let mut prefix = CpuSession::new_prefix(
                optimized.compiled.clone(),
                &INPUT_SHAPE,
                ExecutionLimits::default(),
                last,
            )
            .unwrap();
            let output = prefix.run(&values).unwrap();
            assert_eq!(output.shape(), *shape);
            compare(expected, output.data());
        }
    }
}

#[test]
fn gapped_prefix_and_diagnostic_caches_keep_their_endpoints_across_reshapes() {
    let model = residual_producer_model(false);
    let mut optimized = session(model.clone(), CompileOptions::default(), &INPUT_SHAPE);
    for width in [3, 5, 3] {
        let shape = [2, 2, 2, width];
        let values = (0..shape.iter().product())
            .map(|i| (i as f32 - 11.) / 17.)
            .collect::<Vec<_>>();
        let mut reference = session(model.clone(), CompileOptions::reference(), &shape);
        let mut expected = Vec::new();
        reference
            .run_with_trace(&values, |event| {
                expected.push((event.shape, event.values.to_vec()));
                Ok(())
            })
            .unwrap();
        optimized.reshape(&shape).unwrap();
        for last in [0, 4, 2, 5, 4, 6, 0] {
            let output = optimized
                .run_prefix_with_input(last, |destination| {
                    destination.copy_from_slice(&values);
                    Ok(())
                })
                .unwrap();
            assert_eq!(output.shape(), expected[last].0);
            compare(&expected[last].1, output.data());
        }
        let final_expected = &expected.last().unwrap().1;
        compare(final_expected, optimized.run(&values).unwrap().data());
        compare(
            final_expected,
            optimized
                .run_with_trace(&values, |_| Ok(()))
                .unwrap()
                .data(),
        );
        compare(final_expected, optimized.run(&values).unwrap().data());
    }
}

#[test]
fn gapped_physical_limit_failure_preserves_the_previous_runnable_binding() {
    let model = residual_producer_model(false);
    let mut unlimited = session(model.clone(), CompileOptions::default(), &INPUT_SHAPE);
    let values = input();
    let expected = unlimited.run(&values).unwrap().data().to_vec();
    let mut limited = CpuSession::new_serial(
        unlimited.compiled.clone(),
        &INPUT_SHAPE,
        ExecutionLimits {
            max_activation_bytes: unlimited.activation_bytes(),
        },
    )
    .unwrap();
    let input_shape = limited.input_shape();
    let output_shape = limited.output_shape();
    assert!(limited.reshape(&[2, 2, 20, 30]).is_err());
    assert_eq!(limited.input_shape(), input_shape);
    assert_eq!(limited.output_shape(), output_shape);
    compare(&expected, limited.run(&values).unwrap().data());
}
