#[cfg(target_os = "macos")]
fn attention_online_oracle(case: Case, input: &[f32]) -> Vec<f32> {
    let round = |value: f32| half::f16::from_f32(value).to_f32();
    let input = input.iter().copied().map(round).collect::<Vec<_>>();
    let scale = round(1.0 / (case.d as f32).sqrt());
    let stride = 3 * case.h * case.d;
    let elements = case.n * case.t * case.h * case.d;
    let mut output = vec![0.; elements * if case.offset { 2 } else { 1 }];
    if case.offset {
        for n in 0..case.n {
            for r in 0..case.t {
                for h in 0..case.h {
                    for d in 0..case.d {
                        output[((n * case.t + r) * case.h + h) * case.d + d] =
                            input[(n * case.t + r) * stride + h * case.d + d].max(0.);
                    }
                }
            }
        }
    }
    for n in 0..case.n {
        for h in 0..case.h {
            for row in 0..case.t {
                let mut q = [0f32; 32];
                for (d, value) in q.iter_mut().enumerate().take(case.d) {
                    *value = round(input[(n * case.t + row) * stride + h * case.d + d] * scale)
                        * if case.post_scale { 0.7 } else { 1. };
                }
                let mut maximum = -f32::MAX;
                let mut denominator = 0f32;
                let mut accumulator = vec![0f32; case.d];
                for base in (0..case.t).step_by(16) {
                    let valid = 16.min(case.t - base);
                    let mut scores = [0f32; 16];
                    for (key, score) in scores[..valid].iter_mut().enumerate() {
                        for (d, &query) in q.iter().enumerate() {
                            let value = if d < case.d {
                                input[(n * case.t + base + key) * stride
                                    + case.h * case.d
                                    + h * case.d
                                    + d]
                            } else {
                                0.
                            };
                            *score += query * value;
                        }
                    }
                    let mut next = maximum;
                    for &score in &scores[..valid] {
                        next = next.max(score);
                    }
                    let correction = (maximum - next).exp();
                    let mut weights = [0f32; 16];
                    let mut total = 0f32;
                    for key in 0..valid {
                        weights[key] = (scores[key] - next).exp();
                        total += weights[key];
                    }
                    denominator = denominator * correction + total;
                    maximum = next;
                    for (d, accumulator) in accumulator.iter_mut().enumerate() {
                        *accumulator *= correction;
                        for (key, &weight) in weights[..valid].iter().enumerate() {
                            *accumulator += weight
                                * input[(n * case.t + base + key) * stride
                                    + 2 * case.h * case.d
                                    + h * case.d
                                    + d];
                        }
                        output[usize::from(case.offset) * elements
                            + ((n * case.t + row) * case.h + h) * case.d
                            + d] = round(*accumulator / denominator);
                    }
                }
            }
        }
    }
    output
}

#[cfg(target_os = "macos")]
fn attention_native_contract() {
    let context = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
    for case in [
        Case {
            n: 2,
            t: 17,
            h: 3,
            d: 15,
            post_scale: true,
            offset: false,
        },
        Case {
            n: 1,
            t: 65,
            h: 2,
            d: 24,
            post_scale: false,
            offset: true,
        },
    ] {
        let shape = [case.n, case.t, 3 * case.h * case.d];
        let input = values(shape.iter().product(), 7);
        let expected = attention_online_oracle(case, &input);
        let compiled = Arc::new(
            GpuCompiledModel::compile_with_options(
                fixture(case, Escape::None, false),
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
        let mut session =
            GpuSession::new_resident(compiled.clone(), &shape, ExecutionLimits::default(), None)
                .unwrap();
        let recipe = session
            .recipes
            .iter()
            .find(|r| {
                matches!(r.kind, RecipeKind::Compute(ShaderKind::AttentionViews)) && r.words[2] == 7
            })
            .unwrap();
        if case.offset {
            assert_eq!(recipe.words[60], (case.n * case.t * case.h * case.d) as u32);
        }
        let tensor = session.resident_input();
        context
            .upload_resident(&tensor.buffer, &GpuPrecision::MixedF16.encode(&input))
            .unwrap();
        for profiling in [false, true]
            .into_iter()
            .take(if !case.offset { 2 } else { 1 })
        {
            let mut batch = resident::Batch::new(context.clone()).unwrap();
            batch.profiling = profiling;
            let result = batch.graph(&mut session, "attention-contract").unwrap();
            let (bytes, statistics) = batch
                .read(&result.buffer, expected.len() * 2, false)
                .unwrap();
            for (i, (actual, &expected)) in
                bytes.as_chunks::<2>().0.iter().zip(&expected).enumerate()
            {
                let actual = half::f16::from_bits(u16::from_le_bytes(*actual)).to_f32();
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= 1e-4 + 1e-4 * expected.abs(),
                    "profile{profiling} element{i}: {actual} != {expected}"
                );
            }
            assert_eq!(statistics.diagnostic_readback_bytes, 0);
            assert!(statistics.cpu_compute_stages.is_empty());
            assert!(statistics.native_graph_dispatches > 0);
            if profiling {
                // Profiling measures the same admitted native SIMD source.
                assert!(
                    statistics
                        .profile
                        .unwrap()
                        .kernels
                        .iter()
                        .any(|k| k.kernel == ShaderKind::AttentionViews.name())
                );
            }
        }
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires physical Metal; same4-query Half attention native PSO, independent online16 oracle, tails/p60 and native dispatch profiling"]
fn gpu_attention_simd_contract_native_graph() {
    attention_native_contract();
}
