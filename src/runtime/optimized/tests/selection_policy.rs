//! Independent layout contracts from pinned LayoutPlanner.cs, not a second
//! implementation of the runtime's selector.
use super::*;

fn integer(name: &str, value: u64) -> Vec<u8> {
    blob(
        5,
        &[blob(1, name.as_bytes()), num(20, 2), num(3, value)].concat(),
    )
}
fn policies() -> Vec<KernelPolicy> {
    let mut policies = vec![KernelPolicy::Scalar, KernelPolicy::Portable4];
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
    policies
}
pub(super) fn input_layout_at(request: &CpuSession, index: usize) -> Layout {
    let state = request.optimized.as_ref().unwrap();
    let step = state
        .plan
        .steps
        .iter()
        .find(|step| {
            matches!(&step.op,
        Operation::Native { index: actual, .. } if *actual == index)
        })
        .unwrap();
    state.plan.values[step.inputs[0]].layout
}
fn single_conv(cin: usize, cout: usize, group: usize, dilation: usize) -> Arc<Model> {
    model(
        vec![
            tensor(
                "w",
                &[cout, cin / group, 1, 1],
                &vec![0.25; cout * cin / group],
            ),
            node(
                "Conv",
                &["x", "w"],
                "y",
                [
                    attr("kernel_shape", &[1, 1]),
                    integer("group", group as u64),
                    attr("dilations", &[dilation as i64; 2]),
                ]
                .concat(),
            ),
        ],
        &[-1, cin as i64, -1, -1],
        &[-1, cout as i64, -1, -1],
    )
}

#[test]
fn dense_pointwise_output_panels_and_neutral_inputs_follow_pinned_layout_gates() {
    // These expected decisions are fixed source-contract examples. In
    // particular 8/12/24/40 are not complete16-column panels.
    for (cin, cout, shape, expected) in [
        (16, 8, [1, 16, 3, 17], Layout::Canonical),
        (16, 12, [1, 16, 3, 17], Layout::Canonical),
        (16, 16, [1, 16, 3, 17], Layout::Nhwc),
        (16, 24, [1, 16, 3, 17], Layout::Canonical),
        (16, 32, [1, 16, 3, 17], Layout::Nhwc),
        (16, 40, [1, 16, 3, 17], Layout::Canonical),
        (16, 32, [2, 16, 1, 1], Layout::Canonical),
        (1, 16, [2, 1, 3, 17], Layout::Canonical),
    ] {
        for kernels in policies() {
            let request = session(
                single_conv(cin, cout, 1, 1),
                CompileOptions {
                    cpu: crate::runtime::CpuCompileOptions {
                        kernels,
                        ..CompileOptions::default().cpu
                    },
                    ..CompileOptions::default()
                },
                &shape,
            );
            assert_eq!(
                input_layout_at(&request, 0),
                expected,
                "{kernels:?} cin={cin} cout={cout} shape={shape:?}"
            );
        }
    }
    for (cin, cout, group, dilation) in [
        (16, 16, 16, 1),
        (8, 8, 8, 1),
        (16, 16, 2, 1),
        (16, 16, 1, 2),
    ] {
        let request = session(
            single_conv(cin, cout, group, dilation),
            CompileOptions::default(),
            &[1, cin, 3, 17],
        );
        assert_eq!(input_layout_at(&request, 0), Layout::Canonical);
    }
}

#[test]
fn neutral_layout_changes_are_storage_aliases_without_conversion_or_extra_arena() {
    for dimensions in [[2, 8, 1, 1], [2, 1, 3, 7], [1, 1, 1, 1]] {
        let shape = Shape::new(&dimensions).unwrap();
        let mut values = Vec::new();
        let input = add_value(&mut values, shape, Layout::Nhwc, None);
        let mut steps = Vec::new();
        let mut conversions = HashMap::new();
        let output = convert(
            &mut values,
            &mut steps,
            &mut conversions,
            input,
            Layout::Canonical,
        );
        assert!(steps.is_empty(), "neutral storage must not be transposed");
        assert_eq!(values[output].alias, Some((input, 0)));
        assert_eq!(values[output].layout, Layout::Canonical);
        let again = convert(
            &mut values,
            &mut steps,
            &mut conversions,
            input,
            Layout::Canonical,
        );
        assert_eq!(again, output, "reuse the layout-only alias");
        let mut plan = Plan {
            values,
            steps,
            input,
            output,
            elements: 0,
            fused: 0,
            conversions: 0,
        };
        plan.place().unwrap();
        assert_eq!(plan.root(output), plan.root(input));
        assert_eq!(plan.elements, shape.element_count());
    }
}

fn narrow_pad_end_segment() -> Arc<Model> {
    model(
        vec![
            tensor("a", &[16, 3, 1, 1], &[0.125; 48]),
            tensor("b", &[8, 16, 2, 2], &vec![0.0625; 512]),
            tensor("c", &[16, 8, 1, 1], &vec![0.125; 128]),
            node("Conv", &["x", "a"], "first", attr("kernel_shape", &[1, 1])),
            node("Relu", &["first"], "r0", vec![]),
            node(
                "Conv",
                &["r0", "b"],
                "narrow",
                [attr("kernel_shape", &[2, 2]), attr("pads", &[0, 0, 1, 1])].concat(),
            ),
            node("Relu", &["narrow"], "r1", vec![]),
            node("Conv", &["r1", "c"], "y", attr("kernel_shape", &[1, 1])),
        ],
        &[1, 3, 5, 17],
        &[1, 16, 5, 17],
    )
}

#[test]
fn a_narrow_pad_end_segment_has_two_internal_layout_transitions_and_restarts_at_dense() {
    let request = session(
        narrow_pad_end_segment(),
        CompileOptions {
            cpu: crate::runtime::CpuCompileOptions {
                kernels: KernelPolicy::Portable4,
                ..CompileOptions::default().cpu
            },
            fusion: false,
            ..CompileOptions::default()
        },
        &[1, 3, 5, 17],
    );
    assert_eq!(input_layout_at(&request, 0), Layout::Nhwc);
    assert_eq!(input_layout_at(&request, 2), Layout::Canonical);
    assert_eq!(input_layout_at(&request, 4), Layout::Nhwc);
    // Two internal conversions plus the required canonical public output.
    assert_eq!(request.statistics().layout_conversions, 3);
}

#[test]
fn neon_narrow_pad_end_keeps_a_segment_and_preserves_explicit_nchw() {
    if !simd::neon_available() {
        return;
    }
    for (layout, conversions) in [(LayoutPolicy::Auto, 1), (LayoutPolicy::Nchw, 0)] {
        let request = session(
            narrow_pad_end_segment(),
            CompileOptions {
                cpu: crate::runtime::CpuCompileOptions {
                    kernels: KernelPolicy::NeonFma,
                    layout,
                },
                fusion: false,
                ..Default::default()
            },
            &[1, 3, 5, 17],
        );
        if layout == LayoutPolicy::Auto {
            assert_eq!(input_layout_at(&request, 2), Layout::Nhwc);
        } else {
            assert!(request.optimized.is_none());
        }
        assert_eq!(request.statistics().layout_conversions, conversions);
        let (family, isa) = if layout == LayoutPolicy::Auto {
            ("nhwc_dense_6x16_k512", "NeonFma")
        } else {
            ("nchw_dense_2x2_pad_end", "Portable4")
        };
        assert!(
            request
                .statistics()
                .specialized_kernels
                .iter()
                .any(|kernel| { kernel.family == family && kernel.isa == isa })
        );
    }
}

#[test]
fn neon_narrow_pad_end_eligibility_is_limited_to_native_detector_geometry() {
    if !simd::neon_available() {
        return;
    }
    for (cin, cout) in [(16, 8), (24, 12), (12, 24), (48, 24), (96, 24)] {
        let parsed = model(
            vec![
                tensor("w", &[cout, cin, 2, 2], &vec![0.125; cout * cin * 4]),
                node(
                    "Conv",
                    &["x", "w"],
                    "y",
                    [attr("kernel_shape", &[2, 2]), attr("pads", &[0, 0, 1, 1])].concat(),
                ),
            ],
            &[2, cin as i64, 3, 7],
            &[2, cout as i64, 3, 7],
        );
        let request = session(
            parsed,
            CompileOptions {
                cpu: crate::runtime::CpuCompileOptions {
                    kernels: KernelPolicy::NeonFma,
                    ..CompileOptions::default().cpu
                },
                ..CompileOptions::default()
            },
            &[2, cin, 3, 7],
        );
        assert_eq!(input_layout_at(&request, 0), Layout::Nhwc);
        // The exception must not broaden native pointwise eligibility.
        let pointwise = session(
            single_conv(cin, cout, 1, 1),
            CompileOptions {
                cpu: crate::runtime::CpuCompileOptions {
                    kernels: KernelPolicy::NeonFma,
                    ..CompileOptions::default().cpu
                },
                ..CompileOptions::default()
            },
            &[2, cin, 3, 7],
        );
        assert_eq!(input_layout_at(&pointwise, 0), Layout::Canonical);
        let BoundKind::Conv(base) = &request.plan.steps[0].kind else {
            panic!("expected the pad-end convolution");
        };
        for variant in 0..7 {
            let mut changed = base.clone();
            match variant {
                0 => changed.kh = 3,
                1 => changed.sw = 2,
                2 => changed.pt = 1,
                3 => changed.dh = 2,
                4 => changed.ow -= 1,
                5 => changed.cout = 20,
                _ => {
                    changed.cin = 4;
                    changed.cpg = 4;
                }
            }
            assert!(!conv_nhwc_capable(&request.compiled, 0, &changed));
        }
    }
}

#[test]
fn neon_narrow_pad_end_session_keeps_fused_bias_channel_and_full_residual_results() {
    for (cin, cout) in [(16, 8), (24, 12), (12, 24)] {
        check_neon_narrow_session(cin, cout, 2, 1);
    }
}

#[test]
fn neon_narrow_three_eligibility_keeps_other_policies_and_geometry() {
    if !simd::neon_available() {
        return;
    }
    for stride in [1usize, 2] {
        let (cin, cout, height, width) = (32, 24, 9usize, 17usize);
        let parsed = model(
            vec![
                tensor("w", &[cout, cin, 3, 3], &vec![0.125; cout * cin * 9]),
                node(
                    "Conv",
                    &["x", "w"],
                    "y",
                    [
                        attr("kernel_shape", &[3, 3]),
                        attr("pads", &[1; 4]),
                        attr("strides", &[stride as i64; 2]),
                    ]
                    .concat(),
                ),
            ],
            &[2, cin as i64, height as i64, width as i64],
            &[
                2,
                cout as i64,
                height.div_ceil(stride) as i64,
                width.div_ceil(stride) as i64,
            ],
        );
        for kernels in policies().into_iter().chain([KernelPolicy::NeonFma]) {
            let request = session(
                parsed.clone(),
                CompileOptions {
                    cpu: crate::runtime::CpuCompileOptions {
                        kernels,
                        ..CompileOptions::default().cpu
                    },
                    ..CompileOptions::default()
                },
                &[2, cin, height, width],
            );
            assert_eq!(
                input_layout_at(&request, 0),
                if kernels == KernelPolicy::NeonFma {
                    Layout::Nhwc
                } else {
                    Layout::Canonical
                }
            );
            if kernels == KernelPolicy::NeonFma {
                let BoundKind::Conv(base) = &request.plan.steps[0].kind else {
                    panic!("expected a dense 3x3 convolution");
                };
                for variant in 0..8 {
                    let mut changed = base.clone();
                    match variant {
                        0 => {
                            changed.cin = 24;
                            changed.cpg = 24;
                        }
                        1 => changed.cout = 12,
                        2 => changed.kw = 5,
                        3 => {
                            changed.sh = 3;
                            changed.sw = 3;
                        }
                        4 => changed.sw = 3 - stride,
                        5 => changed.pl = 0,
                        6 => changed.oh -= 1,
                        _ => changed.dw = 2,
                    }
                    assert!(!conv_nhwc_capable(&request.compiled, 0, &changed));
                }
            }
        }
        let canonical = session(
            parsed,
            CompileOptions {
                cpu: crate::runtime::CpuCompileOptions {
                    kernels: KernelPolicy::NeonFma,
                    layout: LayoutPolicy::Nchw,
                },
                fusion: false,
                ..Default::default()
            },
            &[2, cin, height, width],
        );
        assert!(canonical.optimized.is_none());
    }
}

#[test]
fn neon_narrow_three_session_keeps_fused_results_for_both_strides() {
    for stride in [1, 2] {
        check_neon_narrow_session(32, 24, 3, stride);
    }
}

fn check_neon_narrow_session(cin: usize, cout: usize, kernel: usize, stride: usize) {
    if !simd::neon_available() {
        return;
    }
    let cpu = Arc::new(CpuContext::new(if cfg!(feature = "parallel") { 2 } else { 1 }).unwrap());
    // This spatial size reaches the dense two-worker threshold even for the
    // smallest head, while remaining small enough for focused correctness.
    let (batch, height, width): (usize, usize, usize) = if kernel == 2 {
        (2, 49, 41)
    } else {
        (2, 25, 25)
    };
    let (oh, ow) = (height.div_ceil(stride), width.div_ceil(stride));
    let pads = if kernel == 2 { [0, 0, 1, 1] } else { [1; 4] };
    let values = |length, seed| {
        (0..length)
            .map(|i| ((i * 31 + seed) % 103) as f32 / 79. - 0.625)
            .collect::<Vec<_>>()
    };
    let parsed = model(
        vec![
            tensor("r", &[cout, cin, 1, 1], &values(cout * cin, 7)),
            tensor(
                "w",
                &[cout, cin, kernel, kernel],
                &values(cout * cin * kernel * kernel, 11),
            ),
            tensor("b", &[cout], &values(cout, 13)),
            tensor("channel", &[1, cout, 1, 1], &values(cout, 17)),
            tensor("factor", &[], &[0.5]),
            node(
                "Conv",
                &["x", "r"],
                "residual",
                [
                    attr("kernel_shape", &[1, 1]),
                    attr("strides", &[stride as i64; 2]),
                ]
                .concat(),
            ),
            // Mul keeps the residual branch outside Conv epilogue groups,
            // so the tested fused store belongs to the narrow spatial head.
            node("Mul", &["residual", "factor"], "scaled", vec![]),
            node(
                "Conv",
                &["x", "w", "b"],
                "c",
                [
                    attr("kernel_shape", &[kernel as i64; 2]),
                    attr("pads", &pads),
                    attr("strides", &[stride as i64; 2]),
                ]
                .concat(),
            ),
            node("Add", &["c", "scaled"], "sum", vec![]),
            node("Add", &["sum", "channel"], "biased", vec![]),
            node("Relu", &["biased"], "y", vec![]),
        ],
        &[batch as i64, cin as i64, height as i64, width as i64],
        &[batch as i64, cout as i64, oh as i64, ow as i64],
    );
    let request = |fusion| {
        CpuSession::new(
            Arc::new(
                CpuCompiledModel::compile(
                    parsed.clone(),
                    CompileOptions {
                        cpu: crate::runtime::CpuCompileOptions {
                            kernels: KernelPolicy::NeonFma,
                            layout: LayoutPolicy::Auto,
                        },
                        fusion,
                        ..Default::default()
                    },
                )
                .unwrap(),
            ),
            cpu.clone(),
            &[batch, cin, height, width],
            ExecutionOptions::default(),
        )
        .unwrap()
    };
    let mut unfused = request(false);
    let mut fused = request(true);
    assert_eq!(input_layout_at(&fused, 2), Layout::Nhwc);
    assert_eq!(fused.statistics().store_fused_steps, 1);
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    let spatial_sme = kernel == 3 && std::arch::is_aarch64_feature_detected!("sme");
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    let spatial_sme = false;
    let family = if spatial_sme {
        "nhwc_dense_sme32x32_or_neon"
    } else {
        "nhwc_dense_6x16_k512"
    };
    assert!(
        fused
            .statistics()
            .specialized_kernels
            .iter()
            .any(|kernel| kernel.family == family && kernel.store_fused)
    );
    let input = values(batch * cin * height * width, 19);
    for budget in [1, 2] {
        unfused
            .set_intra_op_threads(budget.min(cpu.capacity()))
            .unwrap();
        fused
            .set_intra_op_threads(budget.min(cpu.capacity()))
            .unwrap();
        let expected = unfused.run(&input).unwrap().data().to_vec();
        let actual = fused.run(&input).unwrap().data();
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "cin={cin} cout={cout} kernel={kernel} stride={stride} budget={budget}"
        );
    }
}

#[test]
fn eight_channel_pad_end_matches_an_independent_ordered_oracle_for_all_budgets_and_tails() {
    for (batch, height, width) in [
        (1, 3, 7),
        (1, 3, 8),
        (1, 3, 9),
        (1, 3, 15),
        (1, 3, 16),
        (1, 3, 17),
        (2, 3, 9),
        (1, 64, 257),
    ] {
        let cin = 16;
        let cout = 8;
        // Dyadic values keep the complete ordered reduction exact in f32,
        // independently of available FMA and SIMD tail partitioning.
        let weights: Vec<_> = (0..cout * cin * 4)
            .map(|i| ((i % 7) as f32 - 3.) / 64.)
            .collect();
        let bias: Vec<_> = (0..cout).map(|i| (i as f32 - 3.) / 32.).collect();
        let input: Vec<_> = (0..batch * cin * height * width)
            .map(|i| ((i % 17) as f32 - 8.) / 32.)
            .collect();
        let mut expected = vec![0.; batch * cout * height * width];
        for n in 0..batch {
            for co in 0..cout {
                for y in 0..height {
                    for x in 0..width {
                        let mut sum = bias[co];
                        for ci in 0..cin {
                            for ky in 0..2 {
                                for kx in 0..2 {
                                    if y + ky < height && x + kx < width {
                                        sum += input
                                            [((n * cin + ci) * height + y + ky) * width + x + kx]
                                            * weights[((co * cin + ci) * 2 + ky) * 2 + kx];
                                    }
                                }
                            }
                        }
                        expected[((n * cout + co) * height + y) * width + x] = sum;
                    }
                }
            }
        }
        let parsed = model(
            vec![
                tensor("w", &[cout, cin, 2, 2], &weights),
                tensor("b", &[cout], &bias),
                node(
                    "Conv",
                    &["x", "w", "b"],
                    "y",
                    [attr("kernel_shape", &[2, 2]), attr("pads", &[0, 0, 1, 1])].concat(),
                ),
            ],
            &[batch as i64, cin as i64, height as i64, width as i64],
            &[batch as i64, cout as i64, height as i64, width as i64],
        );
        let cpu =
            Arc::new(CpuContext::new(if cfg!(feature = "parallel") { 8 } else { 1 }).unwrap());
        for kernels in policies() {
            let compiled = Arc::new(
                CpuCompiledModel::compile(
                    parsed.clone(),
                    CompileOptions {
                        cpu: crate::runtime::CpuCompileOptions {
                            kernels,
                            ..CompileOptions::default().cpu
                        },
                        ..CompileOptions::default()
                    },
                )
                .unwrap(),
            );
            for budget in [1, 2, 4, 8] {
                let mut request = CpuSession::new(
                    compiled.clone(),
                    cpu.clone(),
                    &[batch, cin, height, width],
                    ExecutionOptions {
                        intra_op_threads: budget.min(cpu.capacity()),
                        ..ExecutionOptions::default()
                    },
                )
                .unwrap();
                assert_eq!(input_layout_at(&request, 0), Layout::Canonical);
                let actual = request.run(&input).unwrap();
                assert_eq!(
                    actual
                        .data()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "{kernels:?} budget={budget} shape={batch}x{height}x{width}"
                );
            }
        }
    }
}

#[test]
fn batch_norm_continues_an_eligible_identity_stem_and_reuses_channel_workspace() {
    let channels = 16;
    let mut weights = vec![0.; channels * channels];
    for channel in 0..channels {
        weights[channel * channels + channel] = 1.;
    }
    let parsed = model(
        vec![
            tensor("w", &[channels, channels, 1, 1], &weights),
            tensor("scale", &[channels], &vec![0.5; channels]),
            tensor("bias", &[channels], &vec![0.125; channels]),
            tensor("mean", &[channels], &vec![0.25; channels]),
            tensor("variance", &[channels], &vec![0.75; channels]),
            node("Conv", &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
            // Relu prevents importer Conv+BN constant folding, keeping the
            // channels-last affine kernel independently reachable.
            node("Relu", &["c"], "r", vec![]),
            node(
                "BatchNormalization",
                &["r", "scale", "bias", "mean", "variance"],
                "y",
                vec![],
            ),
        ],
        &[2, 16, 3, 7],
        &[2, 16, 3, 7],
    );
    let input: Vec<_> = (0..2 * channels * 3 * 7)
        .map(|i| (i % 31) as f32 / 16. - 0.5)
        .collect();
    let mut reference = session(parsed.clone(), CompileOptions::reference(), &[2, 16, 3, 7]);
    let expected = reference.run(&input).unwrap().data().to_vec();
    let mut request = session(
        parsed,
        CompileOptions {
            cpu: crate::runtime::CpuCompileOptions {
                kernels: KernelPolicy::Portable4,
                ..CompileOptions::default().cpu
            },
            fusion: false,
            ..CompileOptions::default()
        },
        &[2, 16, 3, 7],
    );
    assert_eq!(input_layout_at(&request, 2), Layout::Nhwc);
    assert_eq!(
        request
            .run(&input)
            .unwrap()
            .data()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert!(
        request
            .statistics()
            .specialized_kernels
            .iter()
            .any(|s| s.family == "nhwc_batch_norm_affine")
    );
    let state = request.optimized.as_ref().unwrap();
    assert_eq!(state.workspaces[0].len(), channels);
    let address = state.workspaces[0].as_ptr();
    request.run(&input).unwrap();
    assert_eq!(
        request.optimized.as_ref().unwrap().workspaces[0].as_ptr(),
        address
    );
}

#[test]
fn a_fused_residual_does_not_start_a_segment_for_a_neutral_base_input() {
    let merged = model(
        vec![
            tensor("a", &[16, 1, 1, 1], &[0.25; 16]),
            tensor("b", &[16, 1, 1, 1], &[0.5; 16]),
            node("Conv", &["x", "a"], "r", attr("kernel_shape", &[1, 1])),
            node("Conv", &["x", "b"], "c", attr("kernel_shape", &[1, 1])),
            node("Add", &["c", "r"], "y", vec![]),
        ],
        &[1, 1, 3, 17],
        &[1, 16, 3, 17],
    );
    // The original parallel-linear fixture is intentionally normalized into
    // one Conv by model::merge_branches; there is no second physical Conv to
    // inspect. A nonlinear residual keeps both branches and the runtime Add.
    let merged_ops: Vec<_> = merged
        .graph()
        .nodes
        .iter()
        .map(|node| node.op_type())
        .collect();
    assert_eq!(merged_ops, ["Conv"]);
    let parsed = model(
        vec![
            tensor("a", &[16, 1, 1, 1], &[0.25; 16]),
            tensor("b", &[16, 1, 1, 1], &[0.5; 16]),
            node("Conv", &["x", "a"], "r", attr("kernel_shape", &[1, 1])),
            node("Relu", &["r"], "residual", vec![]),
            node("Conv", &["x", "b"], "c", attr("kernel_shape", &[1, 1])),
            node("Add", &["c", "residual"], "y", vec![]),
        ],
        &[1, 1, 3, 17],
        &[1, 16, 3, 17],
    );
    let preserved_ops: Vec<_> = parsed
        .graph()
        .nodes
        .iter()
        .map(|node| node.op_type())
        .collect();
    assert_eq!(preserved_ops, ["Conv", "Relu", "Conv", "Add"]);
    let input: Vec<_> = (0..3 * 17).map(|i| (i as f32 - 25.) / 32.).collect();
    let expected: Vec<_> = (0..16)
        .flat_map(|_| input.iter().map(|&x| 0.5 * x + (0.25 * x).max(0.)))
        .collect();
    for kernels in policies() {
        let mut request = session(
            parsed.clone(),
            CompileOptions {
                cpu: crate::runtime::CpuCompileOptions {
                    kernels,
                    ..CompileOptions::default().cpu
                },
                ..CompileOptions::default()
            },
            &[1, 1, 3, 17],
        );
        assert_eq!(input_layout_at(&request, 0), Layout::Canonical);
        assert_eq!(input_layout_at(&request, 2), Layout::Canonical);
        assert_eq!(request.statistics().fused_nodes, 2);
        assert_eq!(request.statistics().layout_conversions, 0);
        assert_eq!(
            request
                .run(&input)
                .unwrap()
                .data()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn neon_projection_metadata_distinguishes_four_and_six_row_tiles() {
    use crate::runtime::simd::Backend;
    // The wide packed eligibility shape must still name the actual four-row
    // ARM64 projection, rather than the AVX-512 eight/sixteen-row family.
    for (columns, packed, allow_pointwise, expected) in [
        (1040, true, true, "matmul_4x16"),
        (128, true, false, "matmul_4x16"),
        (128, true, true, "nhwc_constant_matmul_6x16"),
        (128, false, true, "matmul_4x16"),
        (33, true, true, "matmul_4x16"),
    ] {
        let kind = BoundKind::MatMul {
            rows: 8,
            inner: 64,
            columns,
        };
        assert_eq!(
            native_selection(&kind, Backend::NeonFma, packed, 1, allow_pointwise),
            (expected, "NeonFma".into()),
            "columns={columns} packed={packed} pointwise={allow_pointwise}",
        );
    }
}

#[test]
fn neon_canonical_metadata_reports_portable_arithmetic_for_both_worker_policies() {
    use crate::runtime::{
        bound::{Conv, ConvTranspose},
        simd::Backend,
    };
    let kind = BoundKind::Conv(Conv {
        batch: 1,
        cin: 3,
        cout: 16,
        height: 5,
        width: 9,
        oh: 5,
        ow: 9,
        cpg: 3,
        opg: 16,
        kh: 3,
        kw: 3,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
        pt: 1,
        pl: 1,
    });
    for (threads, expected) in [(1, "nchw_packed_3x3"), (4, "nchw_dense_3x3")] {
        assert_eq!(
            native_selection(&kind, Backend::NeonFma, true, threads, false),
            (expected, "Portable4".into()),
        );
        assert!(native_store_eligible(
            &kind,
            Backend::NeonFma,
            true,
            threads
        ));
        assert!(native_consumes_packed(&kind, Backend::NeonFma, true, threads) == (threads == 1));
    }
    let transpose = BoundKind::ConvTranspose(ConvTranspose {
        batch: 1,
        cin: 3,
        cout: 17,
        height: 3,
        width: 9,
    });
    assert_eq!(
        native_selection(&transpose, Backend::NeonFma, false, 1, false),
        ("nchw_transpose2x2_spatial", "Portable4".into()),
    );
    assert!(native_store_eligible(
        &transpose,
        Backend::NeonFma,
        false,
        1
    ));
}

#[test]
fn neon_depthwise_metadata_names_packed_two_pixel_channel_tiles() {
    use crate::runtime::{bound::Conv, simd::Backend};
    for (channels, kernel, packed, expected) in [
        (32, 3, true, "nhwc_depthwise_neon_2x32"),
        (35, 3, true, "nhwc_depthwise_neon_2x32"),
        (64, 7, true, "nhwc_depthwise_neon_2x32"),
        (31, 3, true, "nhwc_depthwise_pixel8"),
        (32, 3, false, "nhwc_depthwise_pixel8"),
        (32, 7, false, "nhwc_depthwise_strip"),
    ] {
        let kind = BoundKind::Conv(Conv {
            batch: 1,
            cin: channels,
            cout: channels,
            height: 9,
            width: 17,
            oh: 9,
            ow: 17,
            cpg: 1,
            opg: 1,
            kh: kernel,
            kw: kernel,
            sh: 1,
            sw: 1,
            dh: 1,
            dw: 1,
            pt: kernel / 2,
            pl: kernel / 2,
        });
        assert_eq!(
            crate::runtime::nhwc::selection(&kind, Backend::NeonFma, packed),
            (expected, Backend::NeonFma),
            "channels={channels} kernel={kernel} packed={packed}",
        );
    }
}

#[test]
fn neon_canonical_fallback_preserves_fused_store_results_and_statistics() {
    if !simd::neon_available() {
        return;
    }
    let parsed = model(
        vec![
            tensor("w", &[17, 3, 1, 1], &[0.125; 51]),
            node("Conv", &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
            node("Relu", &["c"], "y", vec![]),
        ],
        &[1, 3, 3, 9],
        &[1, 17, 3, 9],
    );
    let shape = [1, 3, 3, 9];
    let input = (0..81).map(|i| (i as f32 - 40.) / 32.).collect::<Vec<_>>();
    let mut control = session(
        parsed.clone(),
        CompileOptions {
            cpu: crate::runtime::CpuCompileOptions {
                kernels: KernelPolicy::Portable4,
                layout: LayoutPolicy::Nchw,
            },
            fusion: false,
            ..Default::default()
        },
        &shape,
    );
    let expected = control.run(&input).unwrap().data().to_vec();
    let mut request = session(
        parsed,
        CompileOptions {
            cpu: crate::runtime::CpuCompileOptions {
                kernels: KernelPolicy::NeonFma,
                layout: LayoutPolicy::Nchw,
            },
            fusion: true,
            ..Default::default()
        },
        &shape,
    );
    let actual = request.run(&input).unwrap().data();
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let statistics = request.statistics();
    assert_eq!(statistics.store_fused_steps, 1);
    assert!(
        statistics
            .specialized_kernels
            .iter()
            .any(|kernel| kernel.operator == "Conv"
                && kernel.family == "nchw_pointwise_spatial"
                && kernel.isa == "Portable4"
                && kernel.store_fused)
    );
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[test]
fn sme_convolution_reserves_reuses_and_accounts_for_request_scratch() {
    if !std::arch::is_aarch64_feature_detected!("sme") {
        return;
    }
    for (channels, output_channels) in [
        (384, 384),
        (192, 384),
        (384, 192),
        (96, 192),
        (192, 96),
        (128, 256),
        (256, 128),
    ] {
        let shape = [1, channels, 1, 65];
        let input: Vec<_> = (0..shape.iter().product())
            .map(|i| ((i % 31) as f32 - 15.) / 32.)
            .collect();
        let model = single_conv(channels, output_channels, 1, 1);
        let mut reference = session(model.clone(), CompileOptions::reference(), &shape);
        let expected = reference.run(&input).unwrap().data().to_vec();
        let mut request = session(model, CompileOptions::default(), &shape);
        let before = request.retained_scratch_bytes();
        assert_eq!(request.run(&input).unwrap().data(), expected);
        assert!(
            request
                .statistics()
                .specialized_kernels
                .iter()
                .any(|s| s.family == "nhwc_pointwise_sme16x16_or_neon")
        );
        let state = request.optimized.as_ref().unwrap();
        assert_eq!(state.workspaces[0].len(), 64 * channels);
        let address = state.workspaces[0].as_ptr();
        let retained = request.retained_scratch_bytes();
        // The run also materializes a canonical output buffer. Packing storage
        // must be included in the reported scratch total alongside that buffer.
        assert!(retained - before >= 64 * channels * size_of::<f32>());
        for width in [33, 65] {
            request.reshape(&[1, channels, 1, width]).unwrap();
            let input = vec![0.25; channels * width];
            assert!(
                request
                    .run(&input)
                    .unwrap()
                    .data()
                    .iter()
                    .all(|&v| v == channels as f32 / 16.)
            );
            assert_eq!(
                request.optimized.as_ref().unwrap().workspaces[0].as_ptr(),
                address
            );
            assert!(request.retained_scratch_bytes() >= retained);
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[test]
fn spatial_sme_reserves_and_reuses_accounted_scratch_across_shapes() {
    if !std::arch::is_aarch64_feature_detected!("sme") {
        return;
    }
    for kernel in [2, 3, 5, 7] {
        let channels = 64;
        let model = model(
            vec![
                tensor(
                    "w",
                    &[channels, channels, kernel, kernel],
                    &vec![0.125; channels * channels * kernel * kernel],
                ),
                node("Relu", &["x"], "positive", vec![]),
                node(
                    "Conv",
                    &["positive", "w"],
                    "y",
                    [
                        attr("kernel_shape", &[kernel as i64; 2]),
                        attr(
                            "pads",
                            &[
                                (kernel as i64 - 1) / 2,
                                (kernel as i64 - 1) / 2,
                                kernel as i64 / 2,
                                kernel as i64 / 2,
                            ],
                        ),
                    ]
                    .concat(),
                ),
            ],
            &[-1, channels as i64, -1, -1],
            &[-1, channels as i64, -1, -1],
        );
        let shape = [2, channels, 5, 17];
        let threads = if cfg!(feature = "parallel") { 4 } else { 1 };
        let mut reference = session(model.clone(), CompileOptions::reference(), &shape);
        let mut request = session(model, CompileOptions::default(), &shape);
        request
            .set_execution_context(Arc::new(CpuContext::new(threads).unwrap()), threads)
            .unwrap();
        let before = request.retained_scratch_bytes();
        let mut addresses = None;
        let mut retained = None;
        for width in [17, 9, 17] {
            let shape = [2, channels, 5, width];
            reference.reshape(&shape).unwrap();
            request.reshape(&shape).unwrap();
            let input = vec![0.25; shape.iter().product()];
            let expected = reference.run(&input).unwrap().data().to_vec();
            assert_eq!(request.run(&input).unwrap().data(), expected);
            assert!(
                request
                    .statistics()
                    .specialized_kernels
                    .iter()
                    .any(|s| s.family == "nhwc_dense_sme32x32_or_neon")
            );
            let state = request.optimized.as_ref().unwrap();
            assert_eq!(state.workspaces.len(), threads);
            assert!(
                state
                    .workspaces
                    .iter()
                    .all(|s| s.len() == 64 * channels * kernel * kernel)
            );
            let current: Vec<_> = state.workspaces.iter().map(Vec::as_ptr).collect();
            assert!(
                request.retained_scratch_bytes() - before
                    >= threads * 64 * channels * kernel * kernel * size_of::<f32>()
            );
            if let Some(expected) = &addresses {
                assert_eq!(&current, expected);
            }
            if let Some(expected) = retained {
                assert_eq!(request.retained_scratch_bytes(), expected);
            }
            addresses = Some(current);
            retained = Some(request.retained_scratch_bytes());
        }
    }
}
