//! The oracle quantizes only matrix operands, then reproduces storage boundaries.
use super::super::tests::{bind_program, integer, model, node, tensor};
use super::*;
#[cfg(target_os = "macos")]
use crate::runtime::CompileOptions;

#[derive(Clone, Copy, Debug)]
struct Case {
    batch: usize,
    rows: usize,
    inner: usize,
    columns: usize,
    mode: u32,
}

fn values(count: usize, seed: usize) -> Vec<f32> {
    // Exactly representable operands make different FP32 split-K reductions
    // agree, exposing layout/tail/fusion errors independently of rounding noise.
    (0..count)
        .map(|i| (((i * 7 + seed * 11) % 15) as i32 - 7) as f32 / 32.0)
        .collect()
}

fn fixture(case: Case, concat: bool) -> Arc<Model> {
    let Case {
        batch,
        rows,
        inner,
        columns,
        mode,
    } = case;
    let mut parts = vec![
        tensor("w", &[inner, columns], &values(inner * columns, 7)),
        tensor("bias", &[columns], &values(columns, 11)),
        node("MatMul", &["x", "w"], "product", &[]),
    ];
    let mut current = "product";
    if mode != 0 {
        parts.push(node("Add", &[current, "bias"], "biased", &[]));
        current = "biased";
    }
    if mode == 2 {
        parts.push(node("Sigmoid", &[current], "gate", &[]));
        parts.push(node("Mul", &[current, "gate"], "activated", &[]));
        current = "activated";
    } else if mode == 3 {
        assert_eq!(inner, columns);
        parts.push(node("Add", &[current, "x"], "residual", &[]));
        current = "residual";
    }
    if concat {
        // Independent producers permit contiguous destination placement. The
        // second linear must honor word60, even with an odd element offset.
        parts.push(tensor(
            "w2",
            &[inner, columns],
            &values(inner * columns, 13),
        ));
        parts.push(node("MatMul", &["x", "w2"], "second", &[]));
        parts.push(node(
            "Concat",
            &[current, "second"],
            "y",
            &[integer("axis", 0)],
        ));
    } else {
        parts.push(node("Identity", &[current], "y", &[]));
    }
    model(
        parts,
        &[batch, rows, inner],
        &[batch * if concat { 2 } else { 1 }, rows, columns],
    )
}

fn limits() -> crate::runtime::gpu::GpuLimits {
    crate::runtime::gpu::GpuLimits {
        max_compute_workgroup_storage_size: 32 * 1024,
        ..Default::default()
    }
}

#[test]
fn linear_native_matches_constant_broadcasts_fusions_and_preserves_strict_default() {
    for mode in 0..=3 {
        let case = Case {
            batch: 2,
            rows: 5,
            inner: 67,
            columns: if mode == 3 { 67 } else { 33 },
            mode,
        };
        let program = Program::compile(fixture(case, false)).unwrap();
        let plan = bind_program(
            &program,
            &[case.batch, case.rows, case.inner],
            ExecutionLimits::default(),
        )
        .unwrap();
        for precision in [GpuPrecision::F32, GpuPrecision::MixedF16] {
            let descriptors: Vec<_> = plan
                .shapes
                .iter()
                .map(|&shape| GpuTensorDescriptor::canonical(shape, precision))
                .collect();
            let recipe = replacement(
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
            assert_eq!(recipe.words[3..6], [10, 67, case.columns as u32]);
            assert_eq!(recipe.words[19], u32::from(mode != 0));
            assert_eq!(recipe.words[20], u32::from(mode == 2));
            assert_eq!(recipe.words[21], u32::from(mode == 3));
            assert_eq!(recipe.words[23], 1);
            assert_eq!(
                recipe.words[38],
                u32::from(precision == GpuPrecision::MixedF16)
            );
            assert_eq!(
                recipe.nodes.len(),
                match mode {
                    0 => 1,
                    1 => 2,
                    2 => 4,
                    3 => 3,
                    _ => unreachable!(),
                }
            );
            assert!(
                replacement(
                    &program,
                    &plan,
                    0,
                    program.output,
                    &limits(),
                    &descriptors,
                    GpuMatrixPrecision::F32
                )
                .unwrap()
                .is_none()
            );
            let selected = program.instructions[0].output;
            let prefix = replacement(
                &program,
                &plan,
                0,
                selected,
                &limits(),
                &descriptors,
                GpuMatrixPrecision::F16,
            )
            .unwrap()
            .unwrap();
            assert_eq!(prefix.nodes, [0]);
            assert_eq!(prefix.output, selected);
        }
    }
}

#[test]
fn linear_native_admission_failure_preserves_the_generic_native_graph() {
    let case = Case {
        batch: 2,
        rows: 3,
        inner: 19,
        columns: 7,
        mode: 2,
    };
    let program = Program::compile(fixture(case, false)).unwrap();
    for admit in [false, true] {
        let mut plan = bind_program(
            &program,
            &[case.batch, case.rows, case.inner],
            ExecutionLimits::default(),
        )
        .unwrap();
        let mut descriptors: Vec<_> = plan
            .shapes
            .iter()
            .map(|&shape| GpuTensorDescriptor::canonical(shape, GpuPrecision::MixedF16))
            .collect();
        let recipes = super::super::physical::lower(
            &program,
            &mut plan,
            program.output,
            &limits(),
            GpuPrecision::MixedF16,
            &mut descriptors,
            GpuMatrixPrecision::F16,
            &mut |_| Ok(admit),
        )
        .unwrap();
        assert_eq!(
            recipes
                .iter()
                .filter(|recipe| matches!(
                    recipe.kind,
                    RecipeKind::Compute(ShaderKind::LinearNative)
                ))
                .count(),
            usize::from(admit)
        );
        if !admit {
            assert!(recipes.iter().any(|recipe| matches!(
                recipe.kind,
                RecipeKind::Compute(ShaderKind::General)
            ) && recipe.words[2] == 20));
        }
    }
}

#[cfg(target_os = "macos")]
fn quantize(value: f32) -> f32 {
    half::f16::from_f32(value).to_f32()
}
#[cfg(target_os = "macos")]
fn store(value: f32, precision: GpuPrecision) -> f32 {
    if precision == GpuPrecision::MixedF16 {
        quantize(value)
    } else {
        value
    }
}

#[cfg(target_os = "macos")]
fn oracle(
    case: Case,
    input: &[f32],
    precision: GpuPrecision,
    weight_seed: usize,
    mode: u32,
) -> Vec<f32> {
    let weights = values(case.inner * case.columns, weight_seed);
    let bias = values(case.columns, 11);
    let mut output = vec![0.0; case.batch * case.rows * case.columns];
    for row in 0..case.batch * case.rows {
        for column in 0..case.columns {
            let mut sum = 0.0f32;
            for k in 0..case.inner {
                sum += quantize(input[row * case.inner + k])
                    * quantize(weights[k * case.columns + column]);
            }
            sum = store(sum, precision);
            if mode != 0 {
                sum = store(sum + store(bias[column], precision), precision);
            }
            if mode == 3 {
                sum = store(
                    sum + store(input[row * case.inner + column], precision),
                    precision,
                );
            }
            if mode == 2 {
                let sigmoid = if sum >= 0.0 {
                    1.0 / (1.0 + (-sum).exp())
                } else {
                    let e = sum.exp();
                    e / (1.0 + e)
                };
                sum *= store(sigmoid, precision);
            }
            output[row * case.columns + column] = store(sum, precision);
        }
    }
    output
}

#[cfg(target_os = "macos")]
fn assert_close(actual: &[f32], expected: &[f32], case: Case, precision: GpuPrecision) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs() <= 1e-4 + 1e-4 * expected.abs(),
            "{case:?}/{precision:?}[{index}]: GPU={actual}, oracle={expected}"
        );
    }
}

#[cfg(target_os = "macos")]
fn context() -> Arc<GpuContext> {
    GpuContext::new_with_options(GpuContextOptions::default()).unwrap()
}

#[cfg(target_os = "macos")]
fn session(
    context: &Arc<GpuContext>,
    case: Case,
    precision: GpuPrecision,
    concat: bool,
) -> GpuSession {
    let compiled = Arc::new(
        GpuCompiledModel::compile_with_options(
            fixture(case, concat),
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
    GpuSession::new(
        compiled,
        &[case.batch, case.rows, case.inner],
        ExecutionLimits::default(),
    )
    .unwrap()
}

#[test]
#[ignore = "requires physical Metal; canonical token GEMM, split-K and all dimension/fusion tails"]
#[cfg(target_os = "macos")]
fn gpu_hardware_linear_native_quantized_oracle_and_storage_boundaries() {
    let context = context();
    let cases = [
        Case {
            batch: 2,
            rows: 3,
            inner: 19,
            columns: 7,
            mode: 0,
        },
        Case {
            batch: 2,
            rows: 3,
            inner: 19,
            columns: 7,
            mode: 2,
        },
        Case {
            batch: 2,
            rows: 5,
            inner: 67,
            columns: 33,
            mode: 1,
        },
        Case {
            batch: 2,
            rows: 5,
            inner: 67,
            columns: 33,
            mode: 2,
        },
        Case {
            batch: 2,
            rows: 5,
            inner: 67,
            columns: 67,
            mode: 3,
        },
    ];
    for precision in [GpuPrecision::F32, GpuPrecision::MixedF16] {
        for case in cases {
            let mut selected = session(&context, case, precision, false);
            let recipe = selected
                .recipes
                .iter()
                .find(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::LinearNative)))
                .expect("must qualify actual native token kernel");
            assert_eq!(recipe.words[23], u32::from(case.inner >= 64));
            assert_eq!(
                recipe.nodes.len(),
                match case.mode {
                    0 => 1,
                    1 => 2,
                    2 => 4,
                    3 => 3,
                    _ => unreachable!(),
                }
            );
            let input = values(case.batch * case.rows * case.inner, 3);
            let expected = oracle(case, &input, precision, 7, case.mode);
            assert_close(
                selected.run(&input).unwrap().data(),
                &expected,
                case,
                precision,
            );
        }
    }
}

#[test]
#[ignore = "requires physical Metal; native token writes into nonzero concatenation slice"]
#[cfg(target_os = "macos")]
fn gpu_hardware_linear_native_concat_offset() {
    let context = context();
    let case = Case {
        batch: 1,
        rows: 3,
        inner: 19,
        columns: 7,
        mode: 1,
    };
    for precision in [GpuPrecision::F32, GpuPrecision::MixedF16] {
        let mut selected = session(&context, case, precision, true);
        let recipes: Vec<_> = selected
            .recipes
            .iter()
            .filter(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::LinearNative)))
            .collect();
        assert_eq!(recipes.len(), 2);
        assert!(
            recipes.iter().any(|recipe| recipe.words[60] == 21),
            "must exercise odd output placement"
        );
        let input = values(case.batch * case.rows * case.inner, 3);
        let mut expected = oracle(case, &input, precision, 7, case.mode);
        expected.extend(oracle(case, &input, precision, 13, 0));
        assert_close(
            selected.run(&input).unwrap().data(),
            &expected,
            case,
            precision,
        );
    }
}

#[test]
#[ignore = "requires physical Metal; optimizer must preserve two distinct half-store ties"]
#[cfg(target_os = "macos")]
fn gpu_hardware_linear_native_preserves_half_tie_boundaries() {
    let context = context();
    let shape = [1, 3, 16];
    let mut weights = vec![0.0; 16 * 7];
    for column in 0..7 {
        weights[column] = 0.5;
        weights[7 + column] = 0.5;
    }
    let graph = model(
        vec![
            tensor("w", &[16, 7], &weights),
            tensor("bias", &[7], &[1.0 / 2048.0; 7]),
            node("MatMul", &["x", "w"], "product", &[]),
            node("Add", &["product", "bias"], "y", &[]),
        ],
        &shape,
        &[1, 3, 7],
    );
    let mut input = vec![0.0; shape.iter().product()];
    for row in 0..3 {
        input[row * 16] = 1.0 + 1.0 / 1024.0;
        input[row * 16 + 1] = 1.0;
    }
    // Matrix sum1.00048828125 is precisely between half1.0 and1.0009765625.
    // RNE stores1.0. Adding bias2^-11 creates the same tie and stores1.0.
    // Removing the first boundary instead produces half1.0009765625.
    assert_eq!(quantize(1.0 + 1.0 / 2048.0), 1.0);
    assert_eq!(quantize(1.0 + 2.0 / 2048.0), 1.0 + 1.0 / 1024.0);
    for precision in [GpuPrecision::F32, GpuPrecision::MixedF16] {
        let compiled = Arc::new(
            GpuCompiledModel::compile_with_options(
                graph.clone(),
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
        let mut selected = GpuSession::new(compiled, &shape, ExecutionLimits::default()).unwrap();
        let recipe = selected
            .recipes
            .iter()
            .find(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::LinearNative)))
            .unwrap();
        assert_eq!(recipe.nodes.len(), 2);
        let expected = if precision == GpuPrecision::MixedF16 {
            1.0
        } else {
            1.0 + 1.0 / 1024.0
        };
        assert_eq!(selected.run(&input).unwrap().data(), &[expected; 21]);
    }
}
