//! Allocation contracts over known lifetimes, plus session budget recovery.
use super::*;

fn value(length: usize) -> Value {
    Value {
        shape: Shape::new(&[length]).unwrap(),
        layout: Layout::Canonical,
        constant: None,
        offset: 0,
        alias: None,
    }
}

// Only placement runs here: these copy descriptors describe read/write
// lifetimes independently of an executable operator's shape constraints.
fn step(inputs: &[usize], output: usize) -> Step {
    Step {
        inputs: inputs.to_vec(),
        output,
        op: Operation::Native {
            index: 0,
            kind: BoundKind::Copy,
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

#[test]
fn placement_reuses_disjoint_parts_of_a_dead_larger_tensor() {
    // Input112 dies at step0. Later B64 and C48 overlap each other, but
    // together fit its dead region. A whole-block reuse scheme needed176
    // floats because B occupied the original block's entire capacity.
    let mut p = plan(
        vec![value(112), value(1), value(64), value(48), value(1)],
        vec![step(&[0], 1), step(&[], 2), step(&[], 3), step(&[2, 3], 4)],
        4,
    );
    p.place().unwrap();
    assert_eq!(p.elements, 113);
    assert_eq!(p.values[2].offset, 0);
    assert_eq!(p.values[3].offset, 64);
    assert_eq!(p.values[4].offset, 112);
    assert!(p.values.iter().all(|v| v.offset.is_multiple_of(16)));
}

#[test]
fn placement_keeps_last_read_and_same_step_write_disjoint() {
    // The input's final read and the output's first write occur in the same
    // step. Their closed lifetimes overlap, so this is not a reusable region.
    let mut p = plan(vec![value(32), value(32)], vec![step(&[0], 1)], 1);
    p.place().unwrap();
    assert_eq!(p.elements, 64);
    assert!(!ranges_overlap(
        p.values[0].offset,
        32,
        p.values[1].offset,
        32
    ));
}

#[test]
fn placement_aligns_region_starts_without_padding_the_final_tensor() {
    let mut p = plan(vec![value(3)], Vec::new(), 0);
    p.place().unwrap();
    assert_eq!(p.elements, 3);
    assert_eq!(p.values[0].offset, 0);
}

#[test]
fn placement_rejects_alignment_and_extent_overflow_without_allocating() {
    for length in [usize::MAX, usize::MAX - 15] {
        let mut p = plan(vec![value(length), value(32)], vec![step(&[0], 1)], 1);
        assert!(p.place().unwrap_err().to_string().contains("overflow"));
    }
}

#[test]
fn placement_extends_alias_root_lifetime_and_preserves_relative_offset() {
    let mut view = value(16);
    view.alias = Some((0, 16));
    let mut p = plan(
        vec![value(64), view, value(32)],
        vec![
            Step {
                inputs: Vec::new(),
                output: 1,
                op: Operation::View,
            },
            step(&[1], 2),
        ],
        2,
    );
    p.place().unwrap();
    assert_eq!(p.values[1].offset, p.values[0].offset + 16);
    assert!(!ranges_overlap(
        p.values[0].offset,
        64,
        p.values[2].offset,
        32
    ));
    assert_eq!(p.elements, 96);
}

#[test]
fn smaller_arena_plan_still_enforces_limits_and_rejected_growth_is_reusable() {
    let source = conv_model(false);
    let compiled = Arc::new(
        CpuCompiledModel::compile(
            source.clone(),
            CompileOptions {
                cpu: crate::runtime::CpuCompileOptions {
                    kernels: KernelPolicy::Portable4,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let shape = [1, 3, 8, 8];
    let unlimited =
        CpuSession::new_serial(compiled.clone(), &shape, ExecutionLimits::default()).unwrap();
    let bytes = unlimited.activation_bytes();
    drop(unlimited);
    assert!(
        CpuSession::new_serial(
            compiled.clone(),
            &shape,
            ExecutionLimits {
                max_activation_bytes: bytes - 1,
            },
        )
        .is_err()
    );
    let mut request = CpuSession::new_serial(
        compiled,
        &shape,
        ExecutionLimits {
            max_activation_bytes: bytes,
        },
    )
    .unwrap();
    let input = vec![0.25; shape.iter().product()];
    let expected = request.run(&input).unwrap().data().to_vec();
    let pointer = request.optimized.as_ref().unwrap().arena.as_ptr();
    assert!(request.reshape(&[1, 3, 64, 64]).is_err());
    assert_eq!(request.input_shape().dimensions(), shape);
    assert_eq!(request.activation_bytes(), bytes);
    assert_eq!(request.retained_activation_bytes(), bytes);
    assert_eq!(request.optimized.as_ref().unwrap().arena.as_ptr(), pointer);
    assert_eq!(request.run(&input).unwrap().data(), expected);
    request.reshape(&[1, 3, 4, 8]).unwrap();
    let input = vec![0.5; 3 * 4 * 8];
    let mut reference = session(source, CompileOptions::reference(), &[1, 3, 4, 8]);
    compare(
        request.run(&input).unwrap().data(),
        reference.run(&input).unwrap().data(),
    );
    assert!(request.retained_activation_bytes() <= bytes);
}
