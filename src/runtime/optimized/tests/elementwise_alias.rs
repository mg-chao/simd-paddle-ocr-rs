//! Operand ownership, view lifetimes and broadcast direction for exact aliases.
use super::*;

fn value(dimensions: &[usize]) -> Value {
    Value {
        shape: Shape::new(dimensions).unwrap(),
        layout: Layout::Canonical,
        constant: None,
        offset: 0,
        alias: None,
    }
}
fn binary_step(values: &[Value], inputs: [usize; 2], output: usize, op: BinaryOp) -> Step {
    Step {
        inputs: inputs.to_vec(),
        output,
        op: Operation::Native {
            index: 0,
            kind: BoundKind::Binary(
                kernels::Broadcast::new(
                    values[inputs[0]].shape,
                    values[inputs[1]].shape,
                    values[output].shape,
                ),
                op,
            ),
            epilogues: Vec::new(),
        },
    }
}
fn plan(values: Vec<Value>, steps: Vec<Step>, output: usize) -> Plan {
    Plan {
        values,
        steps,
        input: 0,
        output,
        elements: 0,
        fused: 0,
        conversions: 0,
    }
}
fn exact(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element {index}: {a} != {b}");
    }
}

#[test]
fn chooses_second_dying_operand_and_reuses_a_dead_full_range_view() {
    let mut values = vec![value(&[8]), value(&[8]), value(&[8]), value(&[8])];
    values[1].alias = Some((0, 0));
    let step = binary_step(&values, [2, 1], 3, BinaryOp::Sub);
    let mut p = plan(values, vec![step], 3);
    // The first operand is still needed later; the second is a dying full view.
    p.alias_elementwise(&[0, 1, 3, 3]);
    assert_eq!(p.values[3].alias, Some((1, 0)));
    assert_eq!(p.root(3), (0, 0));
}

#[test]
fn successive_elementwise_outputs_can_extend_the_same_dead_alias_group() {
    let values = vec![value(&[8]), value(&[1]), value(&[8]), value(&[8])];
    let first = binary_step(&values, [0, 1], 2, BinaryOp::Add);
    let second = binary_step(&values, [2, 1], 3, BinaryOp::Mul);
    let mut p = plan(values, vec![first, second], 3);
    p.alias_elementwise(&[1, 2, 2, 4]);
    assert_eq!(p.values[2].alias, Some((0, 0)));
    assert_eq!(p.values[3].alias, Some((2, 0)));
    assert_eq!(p.root(3), (0, 0));
}

#[test]
fn live_sibling_graph_output_constant_and_partial_views_block_aliasing() {
    // Every rejection has only one full-shaped mutable candidate.
    for condition in 0..6 {
        let mut values = vec![value(&[8]), value(&[8]), value(&[1]), value(&[8])];
        values[1].alias = Some((0, 0));
        values[2].constant = Some(TensorId(0));
        let mut end = vec![0, 1, 1, 3];
        match condition {
            0 => end[0] = 3, // A sibling view remains live after this operation.
            1 => end[1] = 3, // The candidate is a retained graph output.
            2 => {
                values[1].alias = None;
                values[1].constant = Some(TensorId(1));
            }
            3 => {
                values[0].shape = Shape::new(&[16]).unwrap();
                values[1].alias = Some((0, 4)); // Only a slice of a larger group.
            }
            4 => values[0].constant = Some(TensorId(1)), // A full view of constant storage.
            5 => {
                values[0].shape = Shape::new(&[16]).unwrap(); // A zero-offset prefix.
            }
            _ => unreachable!(),
        }
        let step = binary_step(&values, [1, 2], 3, BinaryOp::Add);
        let mut p = plan(values, vec![step], 3);
        p.alias_elementwise(&end);
        assert!(p.values[3].alias.is_none(), "condition {condition}");
    }
}

#[test]
fn partial_broadcast_self_view_is_rejected_but_exact_self_operand_is_safe() {
    let mut values = vec![value(&[2, 4]), value(&[4]), value(&[2, 4])];
    values[1].alias = Some((0, 0));
    let step = binary_step(&values, [0, 1], 2, BinaryOp::Add);
    let mut p = plan(values, vec![step], 2);
    p.alias_elementwise(&[1, 1, 3]);
    assert!(p.values[2].alias.is_none());

    let values = vec![value(&[2, 4]), value(&[2, 4])];
    let step = binary_step(&values, [0, 0], 1, BinaryOp::Div);
    let mut p = plan(values, vec![step], 1);
    p.alias_elementwise(&[1, 3]);
    assert_eq!(p.values[1].alias, Some((0, 0)));
}

fn physical(values: &[f32], shape: Shape, layout: Layout) -> Vec<f32> {
    if layout == Layout::Canonical {
        return values.to_vec();
    }
    let d = shape.dimensions();
    let plane = d[2] * d[3];
    (0..values.len())
        .map(|index| {
            let pixel = index / d[1];
            values[(pixel / plane * d[1] + index % d[1]) * plane + pixel % plane]
        })
        .collect()
}

#[test]
fn broadcast_alias_execution_matches_materialized_order_for_both_operands() {
    let cpu = CpuContext::serial();
    let cases: &[(&[usize], &[usize], Layout, Layout)] = &[
        (&[2, 3, 7], &[7], Layout::Canonical, Layout::Canonical),
        (&[2, 3, 7], &[2, 3, 1], Layout::Canonical, Layout::Canonical),
        (&[2, 3, 4], &[1, 3, 1], Layout::Canonical, Layout::Canonical),
        (
            &[2, 3, 2, 7],
            &[2, 3, 2, 1],
            Layout::Canonical,
            Layout::Nhwc,
        ),
        (
            &[2, 2, 3, 2, 5],
            &[1, 2, 1, 2, 1],
            Layout::Canonical,
            Layout::Canonical,
        ),
        (
            &[2, 5, 2, 3],
            &[2, 5, 1, 1],
            Layout::Canonical,
            Layout::Canonical,
        ),
        (&[2, 5, 2, 3], &[2, 5, 1, 1], Layout::Nhwc, Layout::Nhwc),
        (&[2, 3, 2, 4], &[1, 3, 1, 4], Layout::Nhwc, Layout::Nhwc),
        (
            &[2, 3, 2, 4],
            &[2, 3, 2, 4],
            Layout::Nhwc,
            Layout::Canonical,
        ),
    ];
    for &(full_dimensions, gate_dimensions, layout, gate_layout) in cases {
        let full = Shape::new(full_dimensions).unwrap();
        let gate = Shape::new(gate_dimensions).unwrap();
        let full_values: Vec<_> = (0..full.element_count())
            .map(|i| (i % 23 + 1) as f32 / 4.)
            .collect();
        let gate_values: Vec<_> = (0..gate.element_count())
            .map(|i| (i % 13 + 2) as f32 / 8.)
            .collect();
        for alias_slot in 0..2 {
            let shapes = if alias_slot == 0 {
                [full, gate]
            } else {
                [gate, full]
            };
            let layouts = if alias_slot == 0 {
                [layout, gate_layout]
            } else {
                [gate_layout, layout]
            };
            let logical_values = if alias_slot == 0 {
                [&full_values, &gate_values]
            } else {
                [&gate_values, &full_values]
            };
            let broadcast = kernels::Broadcast::new(shapes[0], shapes[1], full);
            for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
                let mut expected = vec![0.; full.element_count()];
                kernels::execute(
                    &BoundKind::Binary(broadcast.clone(), op),
                    &[TensorId(0), TensorId(1)],
                    |id| logical_values[id.0],
                    &mut expected,
                );
                let expected = physical(&expected, full, layout);
                for backend in [Backend::Scalar, Backend::Portable4] {
                    let mut actual = physical(&full_values, full, layout);
                    let other = physical(&gate_values, gate, gate_layout);
                    super::super::elementwise_alias::binary_inplace(
                        backend,
                        op,
                        &broadcast,
                        |slot| (shapes[slot], layouts[slot]),
                        alias_slot,
                        Some(&other),
                        &mut actual,
                        &cpu,
                        1,
                    );
                    exact(&actual, &expected);
                }
            }
        }
    }
}

#[test]
fn row_scalar_aliases_preserve_operand_order_tails_and_parallel_row_offsets() {
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 4 } else { 1 }).unwrap();
    let mut backends = vec![Backend::Scalar, Backend::Portable4];
    for (available, backend) in [
        (simd::avx_available(), Backend::Avx),
        (simd::avx_fma_available(), Backend::AvxFma),
        (simd::avx2_fma_available(), Backend::Avx2Fma),
        (simd::avx512_available(), Backend::Avx512),
    ] {
        if available {
            backends.push(backend);
        }
    }
    // The larger case crosses the parallel threshold with unequal row bands.
    for dimensions in [&[2, 3, 2, 5, 17][..], &[2, 3, 259, 173][..]] {
        let full = Shape::new(dimensions).unwrap();
        let mut gate_dimensions = dimensions.to_vec();
        *gate_dimensions.last_mut().unwrap() = 1;
        let gate = Shape::new(&gate_dimensions).unwrap();
        let full_values: Vec<_> = (0..full.element_count())
            .map(|i| (i % 23 + 1) as f32 / 4.)
            .collect();
        let gate_values: Vec<_> = (0..gate.element_count())
            .map(|i| (i % 13 + 2) as f32 / 8.)
            .collect();
        for alias_slot in 0..2 {
            let (shapes, inputs) = if alias_slot == 0 {
                ([full, gate], [&full_values, &gate_values])
            } else {
                ([gate, full], [&gate_values, &full_values])
            };
            let broadcast = kernels::Broadcast::new(shapes[0], shapes[1], full);
            for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
                let mut expected = vec![0.; full.element_count()];
                kernels::execute(
                    &BoundKind::Binary(broadcast.clone(), op),
                    &[TensorId(0), TensorId(1)],
                    |id| inputs[id.0],
                    &mut expected,
                );
                for &backend in &backends {
                    let mut actual = full_values.clone();
                    super::super::elementwise_alias::binary_inplace(
                        backend,
                        op,
                        &broadcast,
                        |slot| (shapes[slot], Layout::Canonical),
                        alias_slot,
                        Some(&gate_values),
                        &mut actual,
                        &cpu,
                        cpu.capacity(),
                    );
                    exact(&actual, &expected);
                }
            }
        }
    }
}

#[test]
fn row_scalar_sessions_keep_input_aliasing_and_warm_reuse() {
    let input: Vec<_> = (0..2 * 3 * 5 * 17)
        .map(|i| (i % 23 + 1) as f32 / 4.)
        .collect();
    let gate: Vec<_> = (0..2 * 3 * 5).map(|i| (i % 13 + 2) as f32 / 8.).collect();
    for op in ["Add", "Sub", "Mul", "Div"] {
        for operands in [["x", "gate"], ["gate", "x"]] {
            let source = model(
                vec![
                    tensor("gate", &[2, 3, 5, 1], &gate),
                    node(op, &operands, "y", Vec::new()),
                ],
                &[2, 3, 5, 17],
                &[2, 3, 5, 17],
            );
            let mut reference =
                session(source.clone(), CompileOptions::reference(), &[2, 3, 5, 17]);
            let expected = reference.run(&input).unwrap().data().to_vec();
            let mut request = session(source, CompileOptions::default(), &[2, 3, 5, 17]);
            for _ in 0..2 {
                exact(request.run(&input).unwrap().data(), &expected);
            }
            let state = request.optimized.as_ref().unwrap();
            assert_eq!(
                state.plan.values[state.plan.input].offset,
                state.plan.values[state.plan.output].offset
            );
        }
    }
}

#[test]
fn matmul_vector_bias_alias_preserves_unbiased_sum_and_warm_reuse() {
    for (op, operands) in [
        ("Add", ["m", "bias"]),
        ("Sub", ["m", "bias"]),
        ("Sub", ["bias", "m"]),
        ("Div", ["m", "bias"]),
        ("Div", ["bias", "m"]),
    ] {
        let source = model(
            vec![
                tensor("w", &[3, 8], &[1.; 24]),
                tensor("bias", &[8], &[1.; 8]),
                node("MatMul", &["x", "w"], "m", Vec::new()),
                node(op, &operands, "y", Vec::new()),
            ],
            &[2, 3],
            &[2, 8],
        );
        let mut reference = session(source.clone(), CompileOptions::reference(), &[2, 3]);
        let input = [1e20, -1e20, 3., 1e20, -1e20, 3.];
        let expected = reference.run(&input).unwrap().data().to_vec();
        let mut policies = vec![KernelPolicy::Auto, KernelPolicy::Portable4];
        for (available, policy) in [
            (simd::avx_available(), KernelPolicy::Avx),
            (simd::avx_fma_available(), KernelPolicy::AvxFma),
            (simd::avx2_fma_available(), KernelPolicy::Avx2Fma),
            (simd::avx512_available(), KernelPolicy::Avx512),
        ] {
            if available {
                policies.push(policy);
            }
        }
        for kernels in policies {
            let mut request = session(
                source.clone(),
                CompileOptions {
                    kernels,
                    ..Default::default()
                },
                &[2, 3],
            );
            for _ in 0..2 {
                exact(request.run(&input).unwrap().data(), &expected);
            }
            let state = request.optimized.as_ref().unwrap();
            let matmul = state
                .plan
                .steps
                .iter()
                .find(|step| {
                    matches!(
                        step.op,
                        Operation::Native {
                            kind: BoundKind::MatMul { .. },
                            ..
                        }
                    )
                })
                .unwrap();
            assert_eq!(
                state.plan.values[matmul.output].offset,
                state.plan.values[state.plan.output].offset
            );
        }
    }
}
