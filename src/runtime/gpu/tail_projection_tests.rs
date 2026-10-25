use super::super::tests::{bind_program, integers, model, node, tensor};
use super::*;
#[cfg(target_os = "macos")]
use crate::runtime::CompileOptions;
#[cfg(target_os = "macos")]
use resident::Batch;

const N: usize = 2;
const K: usize = 129;
const C: usize = 73;
const T: usize = 67;
fn data(count: usize, seed: usize, divisor: f32) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 7 + seed) % 17) as f32 / divisor - 8.0 / divisor)
        .collect()
}
fn fixture(escape: bool) -> Arc<Model> {
    let mut parts = vec![
        tensor("rw", &[C, K, 1, 1], &data(C * K, 3, 256.0)),
        tensor("rb", &[C], &data(C, 5, 64.0)),
        tensor("w", &[C, K, 1, 1], &data(C * K, 7, 256.0)),
        tensor("b", &[C], &data(C, 11, 64.0)),
        node(
            "Conv",
            &["x", "rw", "rb"],
            "res0",
            &[integers("kernel_shape", &[1, 1])],
        ),
        node("Squeeze", &["res0"], "res1", &[integers("axes", &[2])]),
        node(
            "Transpose",
            &["res1"],
            "ntc",
            &[integers("perm", &[0, 2, 1])],
        ),
        node("Unsqueeze", &["ntc"], "expanded", &[integers("axes", &[1])]),
        node(
            "Transpose",
            &["expanded"],
            "residual",
            &[integers("perm", &[0, 3, 1, 2])],
        ),
        node(
            "Conv",
            &["x", "w", "b"],
            "dot",
            &[integers("kernel_shape", &[1, 1])],
        ),
        node("Sigmoid", &["dot"], "gate", &[]),
        node("Mul", &["dot", "gate"], "swish", &[]),
        node("Add", &["swish", "residual"], "joined", &[]),
        node(
            "Squeeze",
            &["joined"],
            "squeezed",
            &[integers("axes", &[2])],
        ),
        node(
            "Transpose",
            &["squeezed"],
            if escape { "result" } else { "y" },
            &[integers("perm", &[0, 2, 1])],
        ),
    ];
    if escape {
        parts.extend([
            node(
                "ReduceMean",
                &["expanded"],
                "escaped",
                &[
                    integers("axes", &[0, 1, 2, 3]),
                    super::super::tests::integer("keepdims", 0),
                ],
            ),
            node("Add", &["result", "escaped"], "y", &[]),
        ]);
    }
    model(parts, &[N, K, 1, T], &[N, T, C])
}
fn lower(
    program: &Program,
    selected: TensorId,
    precision: GpuPrecision,
    matrix: GpuMatrixPrecision,
    accept: bool,
) -> (ShapePlan, Vec<Recipe>) {
    let mut plan = bind_program(program, &[N, K, 1, T], ExecutionLimits::default()).unwrap();
    let mut descriptors = plan
        .shapes
        .iter()
        .map(|&s| GpuTensorDescriptor::canonical(s, precision))
        .collect();
    let recipes = physical::lower(
        program,
        &mut plan,
        selected,
        &crate::runtime::gpu::GpuLimits::default(),
        precision,
        &mut descriptors,
        matrix,
        &mut |_| Ok(accept),
    )
    .unwrap();
    (plan, recipes)
}
fn tails(recipes: &[Recipe]) -> usize {
    recipes
        .iter()
        .filter(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::TailProjection)))
        .count()
}

#[test]
fn tail_projection_proves_inverse_views_and_removes_only_private_movements() {
    let program = Program::compile(fixture(false)).unwrap();
    let (baseline_plan, baseline) = lower(
        &program,
        program.output,
        GpuPrecision::MixedF16,
        GpuMatrixPrecision::F16,
        false,
    );
    let (plan, recipes) = lower(
        &program,
        program.output,
        GpuPrecision::MixedF16,
        GpuMatrixPrecision::F16,
        true,
    );
    assert_eq!(tails(&baseline), 0);
    assert_eq!(tails(&recipes), 1);
    assert_eq!(recipes.len() + 2, baseline.len());
    let tail = recipes
        .iter()
        .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::TailProjection)))
        .unwrap();
    assert_eq!(tail.output, program.output);
    assert_eq!(program.model.graph().tensors[tail.inputs[3].0].name, "ntc");
    assert_eq!(tail.words[68..74], [8, 0, 0, 1, 2, 0]);
    assert!(plan.lengths.iter().sum::<usize>() <= baseline_plan.lengths.iter().sum::<usize>());
    for name in [
        "expanded", "residual", "dot", "gate", "swish", "joined", "squeezed",
    ] {
        let id = program
            .model
            .graph()
            .tensors
            .iter()
            .position(|t| t.name == name)
            .unwrap();
        assert_eq!(
            plan.slots[id],
            usize::MAX,
            "removed value {name} must not own an arena slot"
        );
        assert_eq!(
            tails(
                &lower(
                    &program,
                    TensorId(id),
                    GpuPrecision::MixedF16,
                    GpuMatrixPrecision::F16,
                    true
                )
                .1
            ),
            0,
            "selected private value {name}"
        );
    }
}

#[test]
fn tail_projection_keeps_escaping_values_and_default_precision() {
    let escaped = Program::compile(fixture(true)).unwrap();
    assert_eq!(
        tails(
            &lower(
                &escaped,
                escaped.output,
                GpuPrecision::MixedF16,
                GpuMatrixPrecision::F16,
                true
            )
            .1
        ),
        0
    );
    let program = Program::compile(fixture(false)).unwrap();
    for (precision, matrix) in [
        (GpuPrecision::F32, GpuMatrixPrecision::F16),
        (GpuPrecision::MixedF16, GpuMatrixPrecision::F32),
        (GpuPrecision::F32, GpuMatrixPrecision::F32),
    ] {
        assert_eq!(
            tails(&lower(&program, program.output, precision, matrix, true).1),
            0
        );
    }
}

#[test]
#[cfg(target_os = "macos")]
fn tail_projection_source_changes_exactly_the_two_qualified_indices() {
    let original = half_native::pointwise_epilogue_source();
    let source = source([GpuPrecision::MixedF16; 6]).unwrap();
    assert_eq!(
        source.replace(
            "const uint index = (batch * columns + column) * rows + row;",
            "const uint index = (batch * rows + row) * columns + column;"
        ),
        original
    );
    assert!(super::source([GpuPrecision::F32; 6]).is_err());
}

#[test]
fn tail_projection_rejection_is_atomic_and_near_miss_packets_keep_originals() {
    let program = Program::compile(fixture(false)).unwrap();
    let (plan, baseline) = lower(
        &program,
        program.output,
        GpuPrecision::MixedF16,
        GpuMatrixPrecision::F16,
        false,
    );
    let descriptors = plan
        .shapes
        .iter()
        .map(|&s| GpuTensorDescriptor::canonical(s, GpuPrecision::MixedF16))
        .collect::<Vec<_>>();
    let base = baseline
        .iter()
        .find(|r| program.model.graph().tensors[r.output.0].name == "joined")
        .unwrap();
    for (word, value) in [(19, 0), (60, 7), (64, 1), (68, 4), (71, 2), (72, 1)] {
        let mut wrong = base.clone();
        wrong.words[word] = value;
        assert!(
            candidate(
                &program,
                &plan,
                program.output,
                &wrong,
                &descriptors,
                &crate::runtime::gpu::GpuLimits::default()
            )
            .unwrap()
            .is_none(),
            "packet word {word}"
        );
    }
    let mut wrong = descriptors.clone();
    wrong[base.inputs[3].0].precision = GpuPrecision::F32;
    assert!(
        candidate(
            &program,
            &plan,
            program.output,
            base,
            &wrong,
            &crate::runtime::gpu::GpuLimits::default()
        )
        .unwrap()
        .is_none()
    );
    let mut recipes = baseline.clone();
    let mut calls = 0;
    select(
        &program,
        &plan,
        program.output,
        &mut recipes,
        &descriptors,
        &crate::runtime::gpu::GpuLimits::default(),
        &mut |_| {
            calls += 1;
            Ok(false)
        },
    )
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(recipes.len(), baseline.len());
    for (a, b) in recipes.iter().zip(&baseline) {
        assert_eq!(a.words, b.words);
        assert_eq!(a.inputs, b.inputs);
        assert_eq!(a.output, b.output);
        assert_eq!(a.nodes, b.nodes);
    }
    let error = select(
        &program,
        &plan,
        program.output,
        &mut recipes,
        &descriptors,
        &crate::runtime::gpu::GpuLimits::default(),
        &mut |_| {
            Err(RuntimeError::with_kind(
                ErrorKind::ResourceLimit,
                "injected quota",
            ))
        },
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResourceLimit);
    assert_eq!(recipes.len(), baseline.len());
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires a physical Metal device"]
fn gpu_hardware_tail_projection_native_closed_graph() {
    let input = data(N * K * T, 13, 32.0);
    let rw = data(C * K, 3, 256.0);
    let rb = data(C, 5, 64.0);
    let w = data(C * K, 7, 256.0);
    let b = data(C, 11, 64.0);
    let mut expected = vec![0.0; N * T * C];
    for n in 0..N {
        for c in 0..C {
            for t in 0..T {
                let mut residual = rb[c];
                let mut value = b[c];
                for k in 0..K {
                    let x = input[(n * K + k) * T + t];
                    residual += rw[c * K + k] * x;
                    value += w[c * K + k] * x;
                }
                let residual = half::f16::from_f32(residual).to_f32();
                let gate = if value >= 0.0 {
                    1.0 / (1.0 + (-value).exp())
                } else {
                    let v = value.exp();
                    v / (1.0 + v)
                };
                expected[(n * T + t) * C + c] =
                    half::f16::from_f32(value * gate + residual).to_f32();
            }
        }
    }
    for reject in [false, true] {
        let context = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
        let compiled = Arc::new(
            GpuCompiledModel::compile_with_options(
                fixture(false),
                context.clone(),
                CompileOptions {
                    gpu: GpuCompileOptions {
                        precision: GpuPrecision::MixedF16,
                        matrix_precision: GpuMatrixPrecision::F16,
                    },
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        if reject {
            compiled
                .pipelines
                .reject_tail_projection_for_test(RuntimeError::with_kind(
                    ErrorKind::GpuValidation,
                    "injected optional tail rejection",
                ));
        }
        let mut session =
            GpuSession::new_resident(compiled, &[N, K, 1, T], ExecutionLimits::default(), None)
                .unwrap();
        assert_eq!(tails(&session.recipes), usize::from(!reject));
        let expected_native = session.dispatches.len() as u64;
        context
            .upload_resident(
                &session.resident_input().buffer,
                &GpuPrecision::MixedF16.encode(&input),
            )
            .unwrap();
        let mut batch = Batch::new(context.clone()).unwrap();
        let output = batch.graph(&mut session, "token tail").unwrap();
        let (bytes, stats) = batch
            .read(
                &output.buffer,
                GpuPrecision::MixedF16.bytes(N * T * C),
                false,
            )
            .unwrap();
        assert_eq!(stats.native_graph_dispatches, expected_native);
        let mut actual = vec![0.0; N * T * C];
        GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                a.is_finite() && (a - e).abs() <= 1e-4 + 1e-4 * e.abs(),
                "reject={reject},i={i},actual={a},expected={e}"
            );
        }
    }
}
