use super::*;

fn graph(axis_value: i64, axis_dimension: usize, constant: bool) -> Arc<Model> {
    let mut parts = vec![
        tensor(
            "w",
            &[16, 3, 1, 1],
            &(0..48).map(|i| (i as f32 - 24.) / 29.).collect::<Vec<_>>(),
        ),
        node("Conv", &["x", "w"], "a", attr("kernel_shape", &[1, 1])),
    ];
    if constant {
        parts.push(tensor("b", &[2, 16, 3, 5], &vec![0.25; 2 * 16 * 3 * 5]));
    } else {
        parts.push(node("Relu", &["a"], "b", vec![]));
    }
    parts.push(node("Concat", &["a", "b", "a"], "y", axis(axis_value)));
    let mut output = [2, 16, 3, 5];
    output[axis_dimension] *= 3;
    model(parts, &[2, 3, 3, 5], &output)
}

#[test]
fn channel_concat_continues_nhwc_with_multiple_batches_and_shared_inputs() {
    for axis_value in [1, -3] {
        let parsed = graph(axis_value, 1, false);
        let input: Vec<_> = (0..2 * 3 * 3 * 5).map(|i| (i as f32 - 31.) / 17.).collect();
        let mut reference = session(parsed.clone(), CompileOptions::reference(), &[2, 3, 3, 5]);
        let expected = reference.run(&input).unwrap().data().to_vec();
        let mut request = session(parsed, CompileOptions::default(), &[2, 3, 3, 5]);
        compare(request.run(&input).unwrap().data(), &expected);
        let state = request.optimized.as_ref().unwrap();
        let concat = state
            .plan
            .steps
            .iter()
            .find(|step| {
                matches!(
                    step.op,
                    Operation::Native {
                        kind: BoundKind::Concat(_),
                        ..
                    }
                )
            })
            .unwrap();
        assert_eq!(state.plan.values[concat.output].layout, Layout::Nhwc);
        assert!(
            concat
                .inputs
                .iter()
                .all(|&id| state.plan.values[id].layout == Layout::Nhwc)
        );
        compare(request.run(&input).unwrap().data(), &expected);
    }
}

#[test]
fn other_axes_and_constant_inputs_keep_canonical_concat() {
    for (axis_value, dimension, constant) in
        [(0, 0, false), (2, 2, false), (3, 3, false), (1, 1, true)]
    {
        let parsed = graph(axis_value, dimension, constant);
        let input: Vec<_> = (0..2 * 3 * 3 * 5).map(|i| (i as f32 - 31.) / 17.).collect();
        let mut reference = session(parsed.clone(), CompileOptions::reference(), &[2, 3, 3, 5]);
        let expected = reference.run(&input).unwrap().data().to_vec();
        let mut request = session(parsed, CompileOptions::default(), &[2, 3, 3, 5]);
        compare(request.run(&input).unwrap().data(), &expected);
        let state = request.optimized.as_ref().unwrap();
        let concat = state
            .plan
            .steps
            .iter()
            .find(|step| {
                matches!(
                    step.op,
                    Operation::Native {
                        kind: BoundKind::Concat(_),
                        ..
                    }
                )
            })
            .unwrap();
        assert_eq!(state.plan.values[concat.output].layout, Layout::Canonical);
    }
}

#[test]
fn mixed_layout_concat_preserves_unequal_channels_and_downstream_segment() {
    for channels in [[16usize, 24usize], [24, 8]] {
        let total = channels.iter().sum::<usize>();
        let source = model(
            vec![
                tensor("wa", &[channels[0], 3, 1, 1], &vec![0.125; channels[0] * 3]),
                tensor("wb", &[channels[1], 3, 1, 1], &vec![-0.25; channels[1] * 3]),
                tensor("wc", &[16, total, 1, 1], &vec![0.0625; total * 16]),
                node("Conv", &["x", "wa"], "a", attr("kernel_shape", &[1, 1])),
                node("Conv", &["x", "wb"], "b", attr("kernel_shape", &[1, 1])),
                node("Concat", &["a", "b"], "c", axis(1)),
                node("Conv", &["c", "wc"], "y", attr("kernel_shape", &[1, 1])),
            ],
            &[2, 3, 3, 5],
            &[2, 16, 3, 5],
        );
        let input: Vec<_> = (0..2 * 3 * 3 * 5).map(|i| (i as f32 - 31.) / 16.).collect();
        let mut reference = session(source.clone(), CompileOptions::reference(), &[2, 3, 3, 5]);
        let expected = reference.run(&input).unwrap().data().to_vec();
        let mut request = session(source, CompileOptions::default(), &[2, 3, 3, 5]);
        compare(request.run(&input).unwrap().data(), &expected);
        let state = request.optimized.as_ref().unwrap();
        let concat = state
            .plan
            .steps
            .iter()
            .find(|step| {
                matches!(
                    step.op,
                    Operation::Native {
                        kind: BoundKind::Concat(_),
                        ..
                    }
                )
            })
            .unwrap();
        let expected_layout = if channels[0] == 16 {
            Layout::Nhwc
        } else {
            Layout::Canonical
        };
        assert_eq!(state.plan.values[concat.output].layout, expected_layout);
        if expected_layout == Layout::Nhwc {
            assert!(
                concat
                    .inputs
                    .iter()
                    .all(|&id| state.plan.values[id].layout == Layout::Nhwc)
            );
            assert!(
                state
                    .plan
                    .steps
                    .iter()
                    .any(|step| step.inputs.contains(&concat.output)
                        && matches!(
                            step.op,
                            Operation::Native {
                                kind: BoundKind::Conv(_),
                                ..
                            }
                        )
                        && state.plan.values[step.output].layout == Layout::Nhwc)
            );
        }
    }
}
