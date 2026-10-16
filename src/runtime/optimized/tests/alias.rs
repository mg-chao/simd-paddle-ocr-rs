//! Reachability regression for scalar Slice/Copy aliases versus full self operands.
//! Static inspection says Slice materializes, so no production fix is presumed.
use super::*;

fn integer_tensor(name: &str, value: i64) -> Vec<u8> {
    blob(
        5,
        &[
            num(1, 1),
            num(2, 7),
            blob(8, name.as_bytes()),
            blob(9, &value.to_le_bytes()),
        ]
        .concat(),
    )
}
fn policies() -> Vec<KernelPolicy> {
    let mut result = vec![KernelPolicy::Portable4];
    for (available, policy) in [
        (simd::avx_available(), KernelPolicy::Avx),
        (simd::avx_fma_available(), KernelPolicy::AvxFma),
        (simd::avx2_fma_available(), KernelPolicy::Avx2Fma),
        (simd::avx512_available(), KernelPolicy::Avx512),
    ] {
        if available {
            result.push(policy);
        }
    }
    result
}
fn assert_exact(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "element {index}");
    }
}

#[test]
fn first_and_last_scalar_slices_materialize_before_inplace_broadcast() {
    let input = [1., 2., 4., 8.];
    let cases: [(&str, usize, bool, [f32; 4]); 16] = [
        ("Add", 0, false, [2., 3., 5., 9.]),
        ("Add", 0, true, [2., 3., 5., 9.]),
        ("Add", 3, false, [9., 10., 12., 16.]),
        ("Add", 3, true, [9., 10., 12., 16.]),
        ("Mul", 0, false, [1., 2., 4., 8.]),
        ("Mul", 0, true, [1., 2., 4., 8.]),
        ("Mul", 3, false, [8., 16., 32., 64.]),
        ("Mul", 3, true, [8., 16., 32., 64.]),
        ("Sub", 0, false, [0., 1., 3., 7.]),
        ("Sub", 0, true, [0., -1., -3., -7.]),
        ("Sub", 3, false, [-7., -6., -4., 0.]),
        ("Sub", 3, true, [7., 6., 4., 0.]),
        ("Div", 0, false, [1., 2., 4., 8.]),
        ("Div", 0, true, [1., 0.5, 0.25, 0.125]),
        ("Div", 3, false, [0.125, 0.25, 0.5, 1.]),
        ("Div", 3, true, [8., 4., 2., 1.]),
    ];
    for (op, start, scalar_left, expected) in cases {
        let operands = if scalar_left {
            ["scalar_view", "x"]
        } else {
            ["x", "scalar_view"]
        };
        let parsed = model(
            vec![
                integer_tensor("starts", start as i64),
                integer_tensor("ends", (start + 1) as i64),
                integer_tensor("axes", 0),
                integer_tensor("one_shape", 1),
                node("Slice", &["x", "starts", "ends", "axes"], "scalar", vec![]),
                node("Reshape", &["scalar", "one_shape"], "scalar_view", vec![]),
                node(op, &operands, "y", vec![]),
            ],
            &[4],
            &[4],
        );
        for policy in policies() {
            let mut request = session(
                parsed.clone(),
                CompileOptions {
                    kernels: policy,
                    ..CompileOptions::default()
                },
                &[4],
            );
            assert_exact(request.run(&input).unwrap().data(), &expected);
            let state = request.optimized.as_ref().unwrap();
            let (slice_index, slice) = state
                .plan
                .steps
                .iter()
                .enumerate()
                .find(|(_, step)| {
                    matches!(
                        step.op,
                        Operation::Native {
                            kind: BoundKind::Transpose(_),
                            ..
                        }
                    )
                })
                .expect("Slice remains a materialized strided copy");
            let source = &state.plan.values[state.plan.input];
            let scalar = &state.plan.values[slice.output];
            let output = &state.plan.values[state.plan.output];
            assert!(scalar.alias.is_none());
            assert_eq!(scalar.shape.element_count(), 1);
            assert!(
                scalar.offset >= source.offset + source.shape.element_count()
                    || source.offset > scalar.offset,
                "Live scalar Slice storage must not overlap source"
            );
            assert!(
                state
                    .plan
                    .values
                    .iter()
                    .enumerate()
                    .any(|(id, value)| value.alias.is_some()
                        && state.plan.root(id).0 == slice.output
                        && value.shape.element_count() == 1)
            );
            let readers: Vec<_> = state
                .plan
                .steps
                .iter()
                .enumerate()
                .map(|(index, step)| (index, step.inputs.clone(), step.output))
                .collect();
            eprintln!(
                "{policy:?} {op} slice={start} scalar_left={scalar_left}: source=({},{}), scalar=({},1), output=({},{}), slice_step={slice_index}, readers={readers:?}",
                source.offset,
                source.shape.element_count(),
                scalar.offset,
                output.offset,
                output.shape.element_count()
            );
            assert_exact(request.run(&input).unwrap().data(), &expected);
        }
    }
}

#[test]
fn full_self_binary_aliases_use_each_current_element_in_both_directions() {
    let input = [1., 2., 4., 8.];
    for (op, expected) in [
        ("Add", [2., 4., 8., 16.]),
        ("Mul", [1., 4., 16., 64.]),
        ("Sub", [0.; 4]),
        ("Div", [1.; 4]),
    ] {
        let parsed = model(vec![node(op, &["x", "x"], "y", vec![])], &[4], &[4]);
        for policy in policies() {
            let mut request = session(
                parsed.clone(),
                CompileOptions {
                    kernels: policy,
                    ..CompileOptions::default()
                },
                &[4],
            );
            assert_exact(request.run(&input).unwrap().data(), &expected);
            let state = request.optimized.as_ref().unwrap();
            assert_eq!(
                state.plan.values[state.plan.input].offset,
                state.plan.values[state.plan.output].offset
            );
            assert_exact(request.run(&input).unwrap().data(), &expected);
        }
    }
}
