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
fn input_layout_at(request: &Session, index: usize) -> Layout {
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
                    kernels,
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

#[test]
fn a_narrow_pad_end_segment_has_two_internal_layout_transitions_and_restarts_at_dense() {
    let parsed = model(
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
    );
    let request = session(
        parsed,
        CompileOptions {
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
                CompiledModel::compile(
                    parsed.clone(),
                    CompileOptions {
                        kernels,
                        ..CompileOptions::default()
                    },
                )
                .unwrap(),
            );
            for budget in [1, 2, 4, 8] {
                let mut request = Session::new(
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
            fusion: false,
            kernels: KernelPolicy::Portable4,
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
        .map(|node| node.op_type.as_str())
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
        .map(|node| node.op_type.as_str())
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
                kernels,
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
        kernels::{Conv, ConvTranspose},
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
    use crate::runtime::{kernels::Conv, simd::Backend};
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
            kernels: KernelPolicy::Portable4,
            layout: LayoutPolicy::Nchw,
            fusion: false,
        },
        &shape,
    );
    let expected = control.run(&input).unwrap().data().to_vec();
    let mut request = session(
        parsed,
        CompileOptions {
            kernels: KernelPolicy::NeonFma,
            layout: LayoutPolicy::Nchw,
            fusion: true,
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
