use super::super::tests::{
    bind_program, int_tensor, integer, integers, model, node, tensor, values,
};
use super::*;
use crate::runtime::CompileOptions;

#[cfg(target_os = "macos")]
include!("attention_native_graph_tests.rs");

#[derive(Clone, Copy)]
struct Case {
    n: usize,
    t: usize,
    h: usize,
    d: usize,
    post_scale: bool,
    offset: bool,
}

#[derive(Clone, Copy)]
enum Escape {
    None,
    QueryScale,
    SharedView,
}

fn fixture(case: Case, escape: Escape, swap_kv: bool) -> Arc<Model> {
    let Case {
        n,
        t,
        h,
        d,
        post_scale,
        offset,
    } = case;
    let input = [n, t, 3 * h * d];
    let mut nodes = vec![
        int_tensor(
            "packed_shape",
            &[5],
            &[n as i64, t as i64, 3, h as i64, d as i64],
        ),
        int_tensor(
            "query_shape",
            &[4],
            &[n as i64, h as i64, t as i64, d as i64],
        ),
        int_tensor("slice_axes", &[1], &[0]),
        tensor("q_scale", &[1], &[1.0 / (d as f32).sqrt()]),
        node("Reshape", &["x", "packed_shape"], "packed", &[]),
        node(
            "Transpose",
            &["packed"],
            "shared",
            &[integers("perm", &[2, 0, 3, 1, 4])],
        ),
    ];
    if offset {
        nodes.extend([
            int_tensor("left_start", &[1], &[0]),
            int_tensor("left_end", &[1], &[(h * d) as i64]),
            int_tensor("left_axis", &[1], &[2]),
            int_tensor(
                "left_shape",
                &[4],
                &[n as i64, t as i64, h as i64, d as i64],
            ),
            node(
                "Slice",
                &["x", "left_start", "left_end", "left_axis"],
                "left_flat",
                &[],
            ),
            node("Reshape", &["left_flat", "left_shape"], "left_view", &[]),
            node("Relu", &["left_view"], "left", &[]),
        ]);
    }
    for (role, name) in [
        (0, "q"),
        (if swap_kv { 2 } else { 1 }, "k"),
        (if swap_kv { 1 } else { 2 }, "v"),
    ] {
        let start = format!("{name}_start");
        let end = format!("{name}_end");
        let slice = format!("{name}_slice");
        nodes.extend([
            int_tensor(&start, &[1], &[role]),
            int_tensor(&end, &[1], &[role + 1]),
            node(
                "Slice",
                &["shared", &start, &end, "slice_axes"],
                &slice,
                &[],
            ),
            node("Squeeze", &[&slice], name, &[integers("axes", &[0])]),
        ]);
    }
    nodes.extend([
        node("Mul", &["q_scale", "q"], "scaled_q", &[]),
        node("Reshape", &["scaled_q", "query_shape"], "query", &[]),
        node(
            "Transpose",
            &["k"],
            "key",
            &[integers("perm", &[0, 1, 3, 2])],
        ),
        node("MatMul", &["query", "key"], "scores", &[]),
    ]);
    let scores = if post_scale {
        nodes.extend([
            tensor("post_scale", &[1], &[0.7]),
            node("Mul", &["scores", "post_scale"], "scaled_scores", &[]),
        ]);
        "scaled_scores"
    } else {
        "scores"
    };
    nodes.extend([
        node("Softmax", &[scores], "probabilities", &[integer("axis", 3)]),
        node("MatMul", &["probabilities", "v"], "attended", &[]),
        node(
            "Transpose",
            &["attended"],
            "transposed",
            &[integers("perm", &[0, 2, 1, 3])],
        ),
    ]);
    let result = match escape {
        Escape::None => "transposed",
        Escape::QueryScale | Escape::SharedView => {
            let (source, axes): (&str, &[usize]) = match escape {
                Escape::QueryScale => ("scaled_q", &[0, 1, 2, 3]),
                Escape::SharedView => ("shared", &[0, 1, 2, 3, 4]),
                _ => unreachable!(),
            };
            nodes.extend([
                tensor("zero", &[1], &[0.0]),
                node(
                    "ReduceMean",
                    &[source],
                    "escaped_mean",
                    &[integers("axes", axes), integer("keepdims", 0)],
                ),
                node("Mul", &["escaped_mean", "zero"], "escaped_zero", &[]),
                node("Add", &["transposed", "escaped_zero"], "kept", &[]),
            ]);
            "kept"
        }
    };
    if offset {
        nodes.push(node(
            "Concat",
            &["left", result],
            "y",
            &[integer("axis", 0)],
        ));
    } else {
        nodes.push(node("Identity", &[result], "y", &[]));
    }
    model(nodes, &input, &[if offset { 2 * n } else { n }, t, h, d])
}

fn first_matmul(program: &Program) -> usize {
    program
        .instructions
        .iter()
        .position(|instruction| matches!(instruction.kind, Op::MatMul))
        .unwrap()
}

fn lowered(
    program: &Program,
    shape: &[usize],
    precision: GpuPrecision,
    matrix: GpuMatrixPrecision,
) -> (ShapePlan, Vec<Recipe>) {
    let mut plan = bind_program(program, shape, ExecutionLimits::default()).unwrap();
    let mut descriptors = plan
        .shapes
        .iter()
        .map(|&shape| GpuTensorDescriptor::canonical(shape, precision))
        .collect();
    let recipes = physical::lower(
        program,
        &mut plan,
        program.output,
        &crate::runtime::gpu::GpuLimits::default(),
        precision,
        &mut descriptors,
        matrix,
        &mut |_| Ok(false),
    )
    .unwrap();
    (plan, recipes)
}

#[test]
fn packed_attention_proves_shared_views_scale_binding_and_output_order() {
    let case = Case {
        n: 2,
        t: 17,
        h: 3,
        d: 15,
        post_scale: true,
        offset: false,
    };
    let shape = [case.n, case.t, 3 * case.h * case.d];
    let program = Program::compile(fixture(case, Escape::None, false)).unwrap();
    let (plan, recipes) = lowered(
        &program,
        &shape,
        GpuPrecision::MixedF16,
        GpuMatrixPrecision::F16,
    );
    assert_eq!(
        recipes.len(),
        1,
        "complete packed block must issue one dispatch"
    );
    let recipe = &recipes[0];
    assert!(matches!(
        recipe.kind,
        RecipeKind::Compute(ShaderKind::AttentionViews)
    ));
    assert_eq!(recipe.words[2], Q_SCALE | PACKED | TRANSPOSE_OUTPUT);
    assert_eq!(recipe.words[3..9], [17, 17, 15, 15, 0.7f32.to_bits(), 5]);
    assert_eq!(recipe.words[15], 3);
    assert_eq!(recipe.inputs[..3], [program.input; 3]);
    assert_eq!(recipe.inputs.len(), 4);
    assert_eq!(
        patterns::scalar(&program, recipe.inputs[3]),
        Some(1.0 / 15.0f32.sqrt())
    );
    assert_eq!(recipe.output, program.output);
    assert_eq!(
        recipe.nodes,
        (0..program.instructions.len()).collect::<Vec<_>>()
    );

    let start = first_matmul(&program);
    let base = fusions::replacement(
        &program,
        &plan,
        start,
        program.output,
        &crate::runtime::gpu::GpuLimits::default(),
        GpuMatrixPrecision::F32,
    )
    .unwrap()
    .unwrap();
    assert!(matches!(
        base.kind,
        RecipeKind::Compute(ShaderKind::Attention)
    ));
    let prefix = replacement(&program, &plan, start, base.output, &base).unwrap();
    assert_eq!(prefix.output, base.output);
    assert_eq!(prefix.words[2], Q_SCALE | PACKED);
    assert_eq!(prefix.node, base.node);
    for &internal in &recipe.nodes {
        let selected = program.instructions[internal].output;
        if selected == base.output || selected == recipe.output {
            continue;
        }
        assert!(
            replacement(&program, &plan, start, selected, &base).is_none(),
            "selected private tensor at node{internal} must remain materialized"
        );
    }
    let mut prefix_plan = ShapePlan::empty(
        program.model.graph().tensors.len(),
        program.instructions.len(),
    );
    let mut scratch = BindScratch::new(&program);
    bind_into(
        &program,
        &shape,
        ExecutionLimits::default(),
        base.node + 1,
        base.output,
        &mut prefix_plan,
        &mut scratch,
    )
    .unwrap();
    let prefix = fusions::replacement(
        &program,
        &prefix_plan,
        start,
        base.output,
        &crate::runtime::gpu::GpuLimits::default(),
        GpuMatrixPrecision::F16,
    )
    .unwrap()
    .unwrap();
    assert_eq!(prefix.output, base.output);
    assert_eq!(prefix.words[2], Q_SCALE | PACKED);
    assert_eq!(prefix.node, base.node);
    let (_, strict) = lowered(&program, &shape, GpuPrecision::F32, GpuMatrixPrecision::F32);
    assert!(!strict.iter().any(|r| matches!(
        r.kind,
        RecipeKind::Compute(ShaderKind::AttentionViews)
    ) && r.words[2] & PACKED != 0));
}

#[test]
fn packed_attention_rejects_escaping_views_and_wrong_slice_roles() {
    let case = Case {
        n: 1,
        t: 19,
        h: 2,
        d: 24,
        post_scale: false,
        offset: false,
    };
    let shape = [case.n, case.t, 3 * case.h * case.d];
    for (escape, swap) in [
        (Escape::QueryScale, false),
        (Escape::SharedView, false),
        (Escape::None, true),
    ] {
        let program = Program::compile(fixture(case, escape, swap)).unwrap();
        let (_, recipes) = lowered(
            &program,
            &shape,
            GpuPrecision::MixedF16,
            GpuMatrixPrecision::F16,
        );
        assert!(!recipes.iter().any(|r| matches!(
            r.kind,
            RecipeKind::Compute(ShaderKind::AttentionViews)
        ) && r.words[2] & PACKED != 0));
        assert!(
            recipes
                .iter()
                .any(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::Attention))),
            "retain existing online core fallback"
        );
    }
}

#[test]
#[ignore = "requires physical Metal GPU; packed QKV against materialized views and Q-scale"]
fn gpu_hardware_packed_attention_preserves_half_scale_tails_and_output_offset() {
    let context = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
    for (case, precision) in [
        (
            Case {
                n: 2,
                t: 17,
                h: 3,
                d: 15,
                post_scale: true,
                offset: false,
            },
            GpuPrecision::F32,
        ),
        (
            Case {
                n: 2,
                t: 17,
                h: 3,
                d: 15,
                post_scale: true,
                offset: false,
            },
            GpuPrecision::MixedF16,
        ),
        (
            Case {
                n: 1,
                t: 65,
                h: 2,
                d: 24,
                post_scale: false,
                offset: true,
            },
            GpuPrecision::MixedF16,
        ),
    ] {
        let shape = [case.n, case.t, 3 * case.h * case.d];
        let input = values(shape.iter().product(), 7);
        let session = |escape| {
            let compiled = Arc::new(
                GpuCompiledModel::compile_with_options(
                    fixture(case, escape, false),
                    context.clone(),
                    CompileOptions {
                        gpu: GpuCompileOptions {
                            precision,
                            matrix_precision: GpuMatrixPrecision::F16,
                        },
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
            GpuSession::new(compiled, &shape, ExecutionLimits::default()).unwrap()
        };
        let mut packed = session(Escape::None);
        let mut materialized = session(Escape::QueryScale);
        let selected = packed
            .recipes
            .iter()
            .find(|r| {
                matches!(r.kind, RecipeKind::Compute(ShaderKind::AttentionViews))
                    && r.words[2] & PACKED != 0
            })
            .unwrap();
        assert_eq!(selected.words[2], Q_SCALE | PACKED | TRANSPOSE_OUTPUT);
        assert_eq!(selected.inputs.len(), 4);
        if case.offset {
            assert_eq!(
                selected.words[60],
                (case.n * case.t * case.h * case.d) as u32
            );
        }
        assert!(!materialized.recipes.iter().any(|r| matches!(
            r.kind,
            RecipeKind::Compute(ShaderKind::AttentionViews)
        ) && r.words[2] & PACKED != 0));
        assert!(
            materialized
                .recipes
                .iter()
                .any(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::Attention)))
        );
        let actual = packed.run(&input).unwrap().data().to_vec();
        let expected = materialized.run(&input).unwrap().data().to_vec();
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-4 + 1e-4 * expected.abs(),
                "{precision:?} N{}T{}H{}D{} element{index}: {actual} != {expected}",
                case.n,
                case.t,
                case.h,
                case.d
            );
        }
    }
}
