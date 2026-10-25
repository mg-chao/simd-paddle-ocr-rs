use super::super::tests::{bind_program, model, node, tensor};
use super::*;
use crate::runtime::CompileOptions;
use backend::test_support::MetalTestBatch;
use backend::test_support::MetalTestShader;
use resident::Batch;

#[derive(Clone, Copy)]
struct Case {
    batch: usize,
    tokens: usize,
    inner: usize,
    columns: usize,
    mode: u32,
}
impl Case {
    fn shape(self) -> [usize; 3] {
        [self.batch, self.tokens, self.inner]
    }
    fn rows(self) -> usize {
        self.batch * self.tokens
    }
    fn count(self) -> usize {
        self.rows() * self.columns
    }
}
struct Values {
    input: Vec<f32>,
    weight: Vec<f32>,
    bias: Vec<f32>,
    residual: Vec<f32>,
}
fn values(case: Case) -> Values {
    let mut v = Values {
        input: (0..case.rows() * case.inner)
            .map(|i| ((i * 13 + 7) % 31) as f32 / 32.0 - 15.0 / 32.0)
            .collect(),
        weight: (0..case.inner * case.columns)
            .map(|i| ((i * 17 + 11) % 31) as f32 / 64.0 - 15.0 / 64.0)
            .collect(),
        bias: (0..case.columns)
            .map(|i| ((i * 7 + 3) % 15) as f32 / 128.0 - 7.0 / 128.0)
            .collect(),
        residual: (0..case.count())
            .map(|i| ((i * 5 + 2) % 15) as f32 / 128.0 - 7.0 / 128.0)
            .collect(),
    };
    for row in 0..case.rows() {
        v.input[row * case.inner..row * case.inner + 3].fill(1.0);
        v.residual[row * case.columns..row * case.columns + 3].fill(0.0);
    }
    for k in 0..case.inner {
        v.weight[k * case.columns..k * case.columns + 3].fill(0.0);
    }
    v.weight[0] = 2048.0;
    v.weight[case.columns] = 1.0;
    v.weight[2 * case.columns] = -2048.0;
    v.weight[1] = 1.0;
    v.weight[case.columns + 1] = 1.0 / 2048.0;
    v.weight[2] = 5.0 / 16.0;
    v.bias[..3].copy_from_slice(&[0.0, 1.0 / 2048.0, 0.0]);
    v
}
fn fixture(case: Case, v: &Values) -> Arc<Model> {
    let mut parts = vec![
        tensor("weight", &[case.inner, case.columns], &v.weight),
        tensor("bias", &[case.columns], &v.bias),
        node("MatMul", &["x", "weight"], "product", &[]),
    ];
    let mut current = "product";
    if case.mode != 0 {
        parts.push(node("Add", &[current, "bias"], "biased", &[]));
        current = "biased";
    }
    if case.mode == 2 {
        parts.push(node("Sigmoid", &[current], "gate", &[]));
        parts.push(node("Mul", &[current, "gate"], "activated", &[]));
        current = "activated";
    } else if case.mode == 3 {
        parts.push(tensor(
            "residual",
            &[case.batch, case.tokens, case.columns],
            &v.residual,
        ));
        parts.push(node("Add", &[current, "residual"], "joined", &[]));
        current = "joined";
    }
    parts.push(node("Identity", &[current], "y", &[]));
    model(
        parts,
        &case.shape(),
        &[case.batch, case.tokens, case.columns],
    )
}
fn bound(case: Case) -> (Program, ShapePlan, Vec<GpuTensorDescriptor>, Recipe) {
    let program = Program::compile(fixture(case, &values(case))).unwrap();
    let plan = bind_program(&program, &case.shape(), ExecutionLimits::default()).unwrap();
    let descriptors = plan
        .shapes
        .iter()
        .map(|&shape| GpuTensorDescriptor::canonical(shape, GpuPrecision::MixedF16))
        .collect::<Vec<_>>();
    let recipe = linear_native::replacement(
        &program,
        &plan,
        0,
        program.output,
        &limits(),
        &descriptors,
        GpuMatrixPrecision::F16,
    )
    .unwrap()
    .unwrap();
    (program, plan, descriptors, recipe)
}
fn limits() -> crate::runtime::gpu::GpuLimits {
    crate::runtime::gpu::GpuLimits {
        max_compute_workgroup_storage_size: 32 * 1024,
        ..Default::default()
    }
}
fn choose(manual: &Recipe, descriptors: &[GpuTensorDescriptor]) -> Option<Recipe> {
    candidate(manual, descriptors, &limits(), GpuMatrixPrecision::F16)
}

#[test]
fn linear_mpp_selection_is_bounded_and_retains_original_recipe() {
    for (rows, inner, columns, eligible) in [
        (31, 120, 240, false),
        (32, 120, 240, true),
        (65, 120, 360, true),
        (72, 240, 120, false),
        (72, 120, 120, false),
        (72, 192, 768, false),
    ] {
        let case = Case {
            batch: 1,
            tokens: rows,
            inner,
            columns,
            mode: 1,
        };
        let (_, _, descriptors, manual) = bound(case);
        let selected = choose(&manual, &descriptors);
        assert_eq!(selected.is_some(), eligible);
        if let Some(selected) = selected {
            assert!(matches!(
                selected.kind,
                RecipeKind::Compute(ShaderKind::LinearMpp)
            ));
            let mut old = manual.words;
            old[1] = selected.words[1];
            assert_eq!(selected.words, old);
            assert_eq!(selected.nodes, manual.nodes);
            assert_eq!(selected.inputs, manual.inputs);
            assert_eq!(selected.output, manual.output);
            assert_eq!(
                selected.x as usize * selected.y as usize,
                rows.div_ceil(32) * columns.div_ceil(64)
            );
        }

        assert!(candidate(&manual, &descriptors, &limits(), GpuMatrixPrecision::F32).is_none());
        let mut wrong = descriptors.clone();
        for d in &mut wrong {
            d.precision = GpuPrecision::F32;
        }
        assert!(choose(&manual, &wrong).is_none());
        let mut wrong = descriptors.clone();
        wrong[manual.inputs[0].0].layout = GpuLayout::PaddedNhwc {
            channels: inner.next_multiple_of(4),
        };
        assert!(choose(&manual, &wrong).is_none());
    }
    let case = Case {
        batch: 2,
        tokens: 33,
        inner: 120,
        columns: 360,
        mode: 2,
    };
    let (_, _, descriptors, manual) = bound(case);
    let mut overflow = manual.clone();
    overflow.words[60] = u32::MAX;
    assert!(choose(&overflow, &descriptors).is_none());
    let mut low = limits();
    low.max_compute_workgroups_per_dimension = 1;
    assert!(candidate(&manual, &descriptors, &low, GpuMatrixPrecision::F16).is_none());
    low = limits();
    low.max_compute_invocations_per_workgroup = 64;
    assert!(candidate(&manual, &descriptors, &low, GpuMatrixPrecision::F16).is_none());
    let mut calls = Vec::new();
    let fallback = admitted(
        manual.clone(),
        &descriptors,
        &limits(),
        GpuMatrixPrecision::F16,
        &mut |r| {
            calls.push(r.kind.clone());
            Ok(matches!(
                r.kind,
                RecipeKind::Compute(ShaderKind::LinearNative)
            ))
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(calls.len(), 2);
    assert!(matches!(
        calls[0],
        RecipeKind::Compute(ShaderKind::LinearMpp)
    ));
    assert!(matches!(
        fallback.kind,
        RecipeKind::Compute(ShaderKind::LinearNative)
    ));
    assert_eq!(fallback.words, manual.words);
    assert_eq!((fallback.x, fallback.y), (manual.x, manual.y));
    let mut calls = 0;
    let error = admitted(
        manual.clone(),
        &descriptors,
        &limits(),
        GpuMatrixPrecision::F16,
        &mut |_| {
            calls += 1;
            Err(resource("intentional allocation failure"))
        },
    )
    .err()
    .unwrap();
    assert_eq!(error.kind(), ErrorKind::ResourceLimit);
    assert_eq!(
        calls, 1,
        "allocation failure must not silently take manual fallback"
    );
    assert!(source([GpuPrecision::F32; 6]).is_err());
}

#[test]
fn linear_mpp_admission_precedes_physical_publication_and_preserves_prefixes() {
    let case = Case {
        batch: 2,
        tokens: 33,
        inner: 120,
        columns: 240,
        mode: 2,
    };
    let (program, mut plan, mut descriptors, manual) = bound(case);
    let descriptor_count = descriptors.len();
    let recipes = physical::lower(
        &program,
        &mut plan,
        program.output,
        &limits(),
        GpuPrecision::MixedF16,
        &mut descriptors,
        GpuMatrixPrecision::F16,
        &mut |r| {
            Ok(matches!(
                r.kind,
                RecipeKind::Compute(ShaderKind::LinearNative)
            ))
        },
    )
    .unwrap();
    let selected = recipes
        .iter()
        .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::LinearNative)))
        .unwrap();
    assert_eq!(selected.words, manual.words);
    assert_eq!((selected.x, selected.y), (manual.x, manual.y));
    assert_eq!(descriptors.len(), descriptor_count);
    assert!(
        !recipes
            .iter()
            .any(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::LinearMpp)))
    );
    let prefix = linear_native::replacement(
        &program,
        &plan,
        0,
        program.instructions[0].output,
        &limits(),
        &descriptors,
        GpuMatrixPrecision::F16,
    )
    .unwrap()
    .unwrap();
    assert_eq!(prefix.nodes, [0]);
    let prefix = choose(&prefix, &descriptors).unwrap();
    assert_eq!(prefix.words[19..22], [0, 0, 0]);
    assert_eq!(prefix.output, program.instructions[0].output);
}

fn oracle(case: Case, v: &Values) -> Vec<f32> {
    let half = |x: f32| half::f16::from_f32(x).to_f32();
    (0..case.count())
        .map(|i| {
            let row = i / case.columns;
            let column = i % case.columns;
            let mut dot = 0.0;
            for k in 0..case.inner {
                dot +=
                    half(v.input[row * case.inner + k]) * half(v.weight[k * case.columns + column]);
            }
            let mut x = half(dot);
            if case.mode != 0 {
                x = half(x + half(v.bias[column]));
            }
            if case.mode == 3 {
                x = half(x + half(v.residual[i]));
            }
            if case.mode == 2 {
                let gate = if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let e = x.exp();
                    e / (1.0 + e)
                };
                x *= half(gate);
            }
            half(x)
        })
        .collect()
}
fn close(actual: &[f32], expected: &[f32], case: Case) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= 1e-4 + 1e-4 * e.abs(),
            "{i}:GPU{a} CPU{e}"
        );
        if i % case.columns < 3 {
            assert_eq!(
                half::f16::from_f32(a).to_bits(),
                half::f16::from_f32(e).to_bits(),
                "semantic witness{i}"
            );
        }
    }
}
fn context() -> Arc<GpuContext> {
    GpuContext::new_with_options(GpuContextOptions::default()).unwrap()
}
fn compile(context: &Arc<GpuContext>, case: Case, v: &Values) -> Arc<GpuCompiledModel> {
    Arc::new(
        GpuCompiledModel::compile_with_options(
            fixture(case, v),
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
    )
}
fn hardware() {
    let context = context();
    for (number, case) in [
        Case {
            batch: 2,
            tokens: 17,
            inner: 120,
            columns: 240,
            mode: 1,
        },
        Case {
            batch: 1,
            tokens: 65,
            inner: 120,
            columns: 240,
            mode: 2,
        },
        Case {
            batch: 2,
            tokens: 33,
            inner: 120,
            columns: 360,
            mode: 3,
        },
    ]
    .into_iter()
    .enumerate()
    {
        let v = values(case);
        let expected = oracle(case, &v);
        let compiled = compile(&context, case, &v);
        let mut session =
            GpuSession::new_resident(compiled, &case.shape(), ExecutionLimits::default(), None)
                .unwrap();
        let recipe = session
            .recipes
            .iter()
            .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::LinearMpp)))
            .expect("actual admitted MPP family")
            .clone();
        assert_eq!(
            recipe.words[19..22],
            [1, u32::from(case.mode == 2), u32::from(case.mode == 3)]
        );
        assert!(
            session
                .descriptors
                .iter()
                .all(|d| d.precision == GpuPrecision::MixedF16)
        );
        context
            .upload_resident(
                &session.resident_input().buffer,
                &GpuPrecision::MixedF16.encode(&v.input),
            )
            .unwrap();
        let mut batch = Batch::new(context.clone()).unwrap();
        let output = batch.graph(&mut session, "linear-mpp").unwrap();
        let (bytes, stats) = batch
            .read(
                &output.buffer,
                GpuPrecision::MixedF16.bytes(case.count()),
                false,
            )
            .unwrap();
        assert_eq!(stats.native_graph_dispatches, 1);
        let mut actual = vec![0.0; case.count()];
        GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
        close(&actual, &expected, case);
        let mut batch = Batch::new(context.clone()).unwrap();
        let mut operands = Vec::new();
        for data in [&v.input, &v.weight, &v.bias, &v.residual, &v.input] {
            let bytes = GpuPrecision::MixedF16.encode(data);
            let b = batch.storage("linear MPP operand", bytes.len()).unwrap();
            context.upload_resident(&b, &bytes).unwrap();
            operands.push(b);
        }
        let prefix = 7;
        let suffix = 11;
        let sentinel = -777.0;
        let mut guarded = vec![sentinel; prefix + case.count() + suffix];
        let output = batch
            .storage(
                "linear MPP guards",
                GpuPrecision::MixedF16.bytes(guarded.len()),
            )
            .unwrap();
        context
            .upload_resident(&output, &GpuPrecision::MixedF16.encode(&guarded))
            .unwrap();
        let mut words = recipe.words;
        words[60] = prefix as u32;
        words[24] = u32::from(number == 2);
        let source = source([GpuPrecision::MixedF16; 6]).unwrap();
        let mut buffers = operands.iter().map(|b| (b, true)).collect::<Vec<_>>();
        buffers.push((&output, false));
        batch
            .dispatch_msl(
                "linear_mpp_guards",
                MetalTestShader {
                    source: &source,
                    entry_point: ENTRY,
                    workgroup_size: WORKGROUP_SIZE,
                },
                &buffers,
                &words,
                [recipe.x, recipe.y, 1],
            )
            .unwrap();
        let (bytes, stats) = batch.read(&output, output.bytes(), false).unwrap();
        assert_eq!(stats.native_resident_dispatches, 1);
        GpuPrecision::MixedF16.decode(&bytes, &mut guarded).unwrap();
        assert!(guarded[..prefix].iter().all(|&v| v == sentinel));
        assert!(
            guarded[prefix + case.count()..]
                .iter()
                .all(|&v| v == sentinel)
        );
        close(&guarded[prefix..prefix + case.count()], &expected, case);
    }
}

#[test]
#[ignore = "requires physical Metal; actual native MPP Half token projections and guarded scalar fallback"]
fn gpu_hardware_linear_mpp_native_graph_rounding_tails_and_offsets() {
    hardware();
}

#[test]
#[ignore = "requires physical Metal; scoped unsupported MPP retains exact manual splitK recipe and native PSO"]
fn gpu_hardware_linear_mpp_rejection_retains_manual_recipe_and_native_pipeline() {
    for kind in [ErrorKind::Unsupported, ErrorKind::GpuValidation] {
        let context = context();
        let case = Case {
            batch: 1,
            tokens: 65,
            inner: 120,
            columns: 240,
            mode: 2,
        };
        let v = values(case);
        let compiled = compile(&context, case, &v);
        compiled
            .pipelines
            .reject_linear_mpp_for_test(RuntimeError::with_kind(
                kind,
                "intentional scoped MPP rejection",
            ));
        let (_, _, _, manual) = bound(case);
        let mut session = GpuSession::new_resident(
            compiled.clone(),
            &case.shape(),
            ExecutionLimits::default(),
            None,
        )
        .unwrap();
        let selected = session
            .recipes
            .iter()
            .find(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::LinearNative)))
            .unwrap();
        assert_eq!(selected.words, manual.words);
        assert_eq!((selected.x, selected.y), (manual.x, manual.y));
        assert!(
            !session
                .recipes
                .iter()
                .any(|r| matches!(r.kind, RecipeKind::Compute(ShaderKind::LinearMpp)))
        );
        let pipeline = compiled
            .pipelines
            .get_for_recipe(
                &context,
                ShaderKind::LinearNative,
                [GpuPrecision::MixedF16; 6],
                &manual.words,
            )
            .unwrap();
        assert!(
            session
                .dispatches
                .iter()
                .any(|d| matches!(&d.command,Command::Compute {native,..} if native.pipeline_for_test()==pipeline))
        );
        context
            .upload_resident(
                &session.resident_input().buffer,
                &GpuPrecision::MixedF16.encode(&v.input),
            )
            .unwrap();
        let mut batch = Batch::new(context.clone()).unwrap();
        let output = batch.graph(&mut session, "manual-fallback").unwrap();
        let (bytes, stats) = batch
            .read(
                &output.buffer,
                GpuPrecision::MixedF16.bytes(case.count()),
                false,
            )
            .unwrap();
        assert_eq!(stats.native_graph_dispatches, 1);
        let mut actual = vec![0.0; case.count()];
        GpuPrecision::MixedF16.decode(&bytes, &mut actual).unwrap();
        close(&actual, &oracle(case, &v), case);
    }
}
