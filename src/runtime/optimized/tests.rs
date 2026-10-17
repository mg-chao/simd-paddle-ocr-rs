use super::*;
mod alias;
mod concat;
mod elementwise_alias;
mod nchw_epilogue;
mod performance;
mod placement;
mod selection_policy;
mod semantic;
fn var(mut n: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let b = (n & 127) as u8;
        n >>= 7;
        out.push(b | if n == 0 { 0 } else { 128 });
        if n == 0 {
            return out;
        }
    }
}
fn num(f: u64, n: u64) -> Vec<u8> {
    [var(f * 8), var(n)].concat()
}
fn blob(f: u64, b: &[u8]) -> Vec<u8> {
    [var(f * 8 + 2), var(b.len() as u64), b.to_vec()].concat()
}
fn info(name: &str, dims: &[i64]) -> Vec<u8> {
    let dims = dims
        .iter()
        .enumerate()
        .flat_map(|(i, &d)| {
            blob(
                1,
                &if d < 0 {
                    blob(2, format!("dim{i}").as_bytes())
                } else {
                    num(1, d as u64)
                },
            )
        })
        .collect::<Vec<_>>();
    [
        blob(1, name.as_bytes()),
        blob(2, &blob(1, &[num(1, 1), blob(2, &dims)].concat())),
    ]
    .concat()
}
fn tensor(name: &str, dims: &[usize], values: &[f32]) -> Vec<u8> {
    blob(
        5,
        &[
            dims.iter()
                .flat_map(|&n| num(1, n as u64))
                .collect::<Vec<_>>(),
            num(2, 1),
            blob(8, name.as_bytes()),
            blob(
                9,
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        ]
        .concat(),
    )
}
fn attr(name: &str, values: &[i64]) -> Vec<u8> {
    blob(
        5,
        &[
            blob(1, name.as_bytes()),
            num(20, 7),
            values
                .iter()
                .flat_map(|&v| num(8, v as u64))
                .collect::<Vec<_>>(),
        ]
        .concat(),
    )
}
fn axis(value: i64) -> Vec<u8> {
    blob(
        5,
        &[blob(1, b"axis"), num(20, 2), num(3, value as u64)].concat(),
    )
}
fn node(op: &str, inputs: &[&str], output: &str, attrs: Vec<u8>) -> Vec<u8> {
    blob(
        1,
        &[
            inputs
                .iter()
                .flat_map(|n| blob(1, n.as_bytes()))
                .collect::<Vec<_>>(),
            blob(2, output.as_bytes()),
            blob(4, op.as_bytes()),
            attrs,
        ]
        .concat(),
    )
}
fn model(parts: Vec<Vec<u8>>, input: &[i64], output: &[i64]) -> Arc<Model> {
    let mut graph = parts.concat();
    graph.extend(blob(11, &info("x", input)));
    graph.extend(blob(12, &info("y", output)));
    let source = [num(1, 8), blob(8, &num(2, 13)), blob(7, &graph)].concat();
    Arc::new(Model::load_bytes(&source).unwrap())
}
fn conv_model(extra_consumer: bool) -> Arc<Model> {
    let mut parts = vec![
        tensor(
            "w",
            &[5, 3, 1, 1],
            &(0..15).map(|i| (i as f32 - 7.) / 13.).collect::<Vec<_>>(),
        ),
        tensor("b", &[1, 5, 1, 1], &[0.2, -0.1, 0.3, 0.4, -0.5]),
        node("Conv", &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
        node("Add", &["c", "b"], "biased", Vec::new()),
        node(
            "Relu",
            &["biased"],
            if extra_consumer { "activated" } else { "y" },
            Vec::new(),
        ),
    ];
    if extra_consumer {
        parts.push(node("Add", &["c", "activated"], "y", Vec::new()));
    }
    model(parts, &[-1, 3, -1, -1], &[-1, 5, -1, -1])
}
fn compare(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() <= 1e-4 + 1e-4 * a.abs(), "{i}: {a} != {b}");
    }
}
fn session(model: Arc<Model>, options: CompileOptions, shape: &[usize]) -> Session {
    Session::new_serial(
        Arc::new(CompiledModel::compile(model, options).unwrap()),
        shape,
        ExecutionLimits::default(),
    )
    .unwrap()
}

#[test]
fn fusion_layout_prefix_and_trace_keep_normalized_identities() {
    let model = conv_model(false);
    let original_graph = format!("{:?}", model.graph());
    let shape = [2, 3, 4, 5];
    let input = (0..shape.iter().product())
        .map(|i| (i as f32 - 31.) / 19.)
        .collect::<Vec<_>>();
    let mut reference = session(model.clone(), CompileOptions::reference(), &shape);
    let expected = reference.run(&input).unwrap().data().to_vec();
    for layout in [LayoutPolicy::Nchw, LayoutPolicy::Auto] {
        for kernels in [
            KernelPolicy::Scalar,
            KernelPolicy::Portable4,
            KernelPolicy::Auto,
        ] {
            let mut optimized = session(
                model.clone(),
                CompileOptions {
                    kernels,
                    layout,
                    fusion: true,
                },
                &shape,
            );
            compare(&expected, optimized.run(&input).unwrap().data());
            assert_eq!(optimized.statistics().fused_nodes, 2);
            let mut events = Vec::new();
            optimized
                .run_with_trace(&input, |event| {
                    assert_eq!(event.layout, "nchw");
                    events.push((
                        event.index,
                        event.output_id,
                        event.shape,
                        event.values.to_vec(),
                    ));
                    Ok(())
                })
                .unwrap();
            assert_eq!(events.len(), model.graph().nodes.len());
            for (i, event) in events.iter().enumerate() {
                assert_eq!(event.0, i);
                assert_eq!(Some(event.1), model.graph().nodes[i].outputs[0]);
            }
            for (last, event) in events.iter().enumerate() {
                let compiled = optimized.compiled.clone();
                let mut prefix =
                    Session::new_prefix(compiled, &shape, ExecutionLimits::default(), last)
                        .unwrap();
                let result = prefix.run(&input).unwrap();
                compare(&event.3, result.data());
            }
        }
    }
    assert_eq!(original_graph, format!("{:?}", model.graph()));
}
#[test]
fn externally_consumed_fusion_intermediate_stays_materialized() {
    let model = conv_model(true);
    let shape = [1, 3, 3, 7];
    let input = (0..63).map(|i| i as f32 / 31. - 1.).collect::<Vec<_>>();
    let mut reference = session(model.clone(), CompileOptions::reference(), &shape);
    let expected = reference.run(&input).unwrap().data().to_vec();
    let mut optimized = session(model, CompileOptions::default(), &shape);
    compare(&expected, optimized.run(&input).unwrap().data());
    assert_eq!(optimized.statistics().fused_nodes, 0);
}

#[test]
fn convolution_store_programs_preserve_order_tails_and_worker_bands() {
    let integer = |name: &str, value: u64| {
        blob(
            5,
            &[blob(1, name.as_bytes()), num(20, 2), num(3, value)].concat(),
        )
    };
    // Pointwise, im2row dense, large depthwise, scalar transpose, and packed
    // transpose each reach a different store, including non-vector channel tails.
    for (transpose, cin, cout, kernel, groups, stem) in [
        (false, 3, 17, 1, 1, false),
        (false, 3, 16, 1, 1, false),
        (false, 3, 17, 3, 1, false),
        (false, 3, 16, 3, 1, false),
        (false, 17, 17, 7, 17, false),
        (false, 16, 16, 7, 16, false),
        // Depthwise only continues an NHWC segment. Preserve standalone
        // canonical arithmetic above and independently reach NHWC stores via
        // complete16-/32-channel identity stems. These shapes exceed the1M
        // depthwise threshold, so Auto also exercises whole-row worker bands.
        (false, 16, 16, 7, 16, true),
        (false, 32, 32, 7, 32, true),
        (true, 3, 1, 2, 1, false),
        (true, 3, 16, 2, 1, false),
        (true, 3, 17, 2, 1, false),
    ] {
        let (height, width) = if stem { (33, 21) } else { (11, 19) };
        let (oh, ow) = if transpose {
            (height * 2, width * 2)
        } else {
            (height, width)
        };
        let count = cin * cout / groups * kernel * kernel;
        let weights = (0..count)
            .map(|i| (i as f32 % 19. - 9.) / 37.)
            .collect::<Vec<_>>();
        let channels = (0..cout).map(|i| (i as f32 - 3.) / 13.).collect::<Vec<_>>();
        let weight_shape = if transpose {
            [cin, cout, kernel, kernel]
        } else {
            [cout, cin / groups, kernel, kernel]
        };
        let attrs = if transpose {
            [attr("kernel_shape", &[2, 2]), attr("strides", &[2, 2])].concat()
        } else {
            [
                attr("kernel_shape", &[kernel as i64; 2]),
                attr("pads", &[kernel as i64 / 2; 4]),
                integer("group", groups as u64),
            ]
            .concat()
        };
        let mut parts = vec![
            tensor("w", &weight_shape, &weights),
            tensor("bias", &[cout], &channels),
            tensor("b", &[1, cout, 1, 1], &channels),
            tensor("s", &[1], &[-0.125]),
        ];
        if stem {
            let mut identity = vec![0.; cin * cin];
            for channel in 0..cin {
                identity[channel * cin + channel] = 1.;
            }
            parts.extend([
                tensor("stem_w", &[cin, cin, 1, 1], &identity),
                node(
                    "Conv",
                    &["x", "stem_w"],
                    "stem",
                    attr("kernel_shape", &[1, 1]),
                ),
            ]);
            assert!(2 * height * width * cout * kernel * kernel >= 1_000_000);
        }
        let source = if stem { "stem" } else { "x" };
        let operands = [source, "w", "bias"];
        parts.extend([
            node(
                if transpose { "ConvTranspose" } else { "Conv" },
                &operands[..if cout == 16 { 3 } else { 2 }],
                "c",
                attrs,
            ),
            node("Add", &["c", "b"], "b1", vec![]),
            node("Sigmoid", &["b1"], "a1", vec![]),
            node("Add", &["a1", "s"], "b2", vec![]),
            node("Relu", &["b2"], "y", vec![]),
        ]);
        let model = model(
            parts,
            &[2, cin as i64, height as i64, width as i64],
            &[2, cout as i64, oh as i64, ow as i64],
        );
        let shape = [2, cin, height, width];
        let input = (0..shape.iter().product())
            .map(|i| (i as f32 % 31. - 15.) / 23.)
            .collect::<Vec<_>>();
        let capacity = if cfg!(feature = "parallel") { 4 } else { 1 };
        let cpu = Arc::new(CpuContext::new(capacity).unwrap());
        let mut policies = vec![KernelPolicy::Scalar, KernelPolicy::Portable4];
        if simd::avx_available() {
            policies.push(KernelPolicy::Avx);
        }
        if simd::avx_fma_available() {
            policies.push(KernelPolicy::AvxFma);
        }
        if simd::avx2_fma_available() {
            policies.push(KernelPolicy::Avx2Fma);
        }
        if simd::avx512_available() {
            policies.push(KernelPolicy::Avx512);
        }
        for (kernels, layout) in policies.into_iter().flat_map(|policy| {
            [LayoutPolicy::Nchw, LayoutPolicy::Auto].map(|layout| (policy, layout))
        }) {
            let options = CompileOptions {
                kernels,
                layout,
                ..CompileOptions::default()
            };
            let mut unfused = session(
                model.clone(),
                CompileOptions {
                    fusion: false,
                    ..options
                },
                &shape,
            );
            let expected = unfused.run(&input).unwrap().data().to_vec();
            let mut fused = Session::new(
                Arc::new(CompiledModel::compile(model.clone(), options).unwrap()),
                cpu.clone(),
                &shape,
                ExecutionOptions {
                    intra_op_threads: capacity,
                    ..ExecutionOptions::default()
                },
            )
            .unwrap();
            let actual = fused.run(&input).unwrap().data();
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{kernels:?} {layout:?} transpose={transpose} cin={cin} cout={cout} kernel={kernel}"
            );
            let expected_fused = kernels != KernelPolicy::Scalar
                || (layout == LayoutPolicy::Auto
                    && ((transpose && cin > 1)
                        || (!transpose
                            && cout >= 16
                            && cout.is_multiple_of(16)
                            && (groups == 1 || stem))));
            assert_eq!(
                fused.statistics().store_fused_steps,
                usize::from(expected_fused)
            );
            assert_eq!(
                fused
                    .statistics()
                    .specialized_kernels
                    .iter()
                    .any(|kernel| kernel.store_fused
                        && kernel.operator == if transpose { "ConvTranspose" } else { "Conv" }),
                expected_fused,
            );
            if stem && layout == LayoutPolicy::Auto {
                assert!(fused.statistics().specialized_kernels.iter().any(|kernel|
                    kernel.family == "nhwc_depthwise_strip" && kernel.store_fused));
            }
            fused.set_intra_op_threads(1).unwrap();
            assert!(
                fused
                    .run(&input)
                    .unwrap()
                    .data()
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
}

#[test]
fn packed_dense_does_not_retain_an_im2row_tensor() {
    for channels in [17usize, 32] {
        let weights = (0..channels * channels * 9)
            .map(|i| (i as f32 % 17. - 8.) / 31.)
            .collect::<Vec<_>>();
        let model = model(
            vec![
                tensor("w", &[channels, channels, 3, 3], &weights),
                node(
                    "Conv",
                    &["x", "w"],
                    "y",
                    [attr("kernel_shape", &[3, 3]), attr("pads", &[1; 4])].concat(),
                ),
            ],
            &[1, channels as i64, 5, 13],
            &[1, channels as i64, 5, 13],
        );
        let mut request = session(
            model,
            CompileOptions {
                kernels: KernelPolicy::Portable4,
                ..CompileOptions::default()
            },
            &[1, channels, 5, 13],
        );
        request.run(&vec![0.25; channels * 5 * 13]).unwrap();
        let physical = request.optimized.as_ref().unwrap();
        let nhwc = request
            .statistics()
            .specialized_kernels
            .iter()
            .any(|kernel| {
                kernel.family == "nhwc_dense_6x8_pair2_k512"
                    && kernel.isa == "Portable4"
                    && !kernel.store_fused
            });
        assert_eq!(nhwc, channels == 32);
        assert!(
            physical
                .workspaces
                .iter()
                .all(|workspace| workspace.capacity() == 0)
        );
    }
}
#[test]
fn standalone_batch_norm_keeps_canonical_storage_without_channel_workspace() {
    let channels = 17;
    let scale = (0..channels)
        .map(|i| 0.5 + i as f32 / 31.)
        .collect::<Vec<_>>();
    let mean = (0..channels)
        .map(|i| (i as f32 - 8.) / 19.)
        .collect::<Vec<_>>();
    let model = model(
        vec![
            tensor("scale", &[channels], &scale),
            tensor("bias", &[channels], &[0.125; 17]),
            tensor("mean", &[channels], &mean),
            tensor("variance", &[channels], &[0.75; 17]),
            node(
                "BatchNormalization",
                &["x", "scale", "bias", "mean", "variance"],
                "y",
                vec![],
            ),
        ],
        &[2, 17, 3, 7],
        &[2, 17, 3, 7],
    );
    let input = (0..2 * 17 * 3 * 7)
        .map(|i| (i as f32 % 31. - 15.) / 17.)
        .collect::<Vec<_>>();
    let mut reference = session(model.clone(), CompileOptions::reference(), &[2, 17, 3, 7]);
    let expected = reference.run(&input).unwrap().data().to_vec();
    let mut optimized = session(
        model,
        CompileOptions {
            kernels: KernelPolicy::Portable4,
            ..CompileOptions::default()
        },
        &[2, 17, 3, 7],
    );
    assert!(
        optimized
            .run(&input)
            .unwrap()
            .data()
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        optimized
            .statistics()
            .specialized_kernels
            .iter()
            .any(|s| s.family == "canonical_native" && s.operator == "BatchNormalization")
    );
    let workspace = &optimized.optimized.as_ref().unwrap().workspaces[0];
    assert_eq!(workspace.len(), 0);
    let address = workspace.as_ptr();
    optimized.run(&input).unwrap();
    assert_eq!(
        optimized.optimized.as_ref().unwrap().workspaces[0].as_ptr(),
        address
    );
}

#[test]
fn channel_pool_and_resize_row_shards_match_one_thread() {
    for resize in [false, true] {
        let (shape, parts, output_shape) = if resize {
            (
                [2, 17, 64, 65],
                vec![
                    tensor("scales", &[4], &[1., 1., 2., 2.]),
                    node(
                        "Resize",
                        &["x", "", "scales"],
                        "y",
                        [
                            blob(
                                5,
                                &[
                                    blob(1, b"coordinate_transformation_mode"),
                                    num(20, 3),
                                    blob(4, b"asymmetric"),
                                ]
                                .concat(),
                            ),
                            blob(
                                5,
                                &[blob(1, b"nearest_mode"), num(20, 3), blob(4, b"floor")].concat(),
                            ),
                        ]
                        .concat(),
                    ),
                ],
                [2, 17, 128, 130],
            )
        } else {
            (
                [2, 17, 128, 129],
                vec![node(
                    "MaxPool",
                    &["x"],
                    "y",
                    [attr("kernel_shape", &[2, 2]), attr("pads", &[0, 0, 1, 1])].concat(),
                )],
                [2, 17, 128, 129],
            )
        };
        let model = model(
            parts,
            &shape.map(|x| x as i64),
            &output_shape.map(|x| x as i64),
        );
        let input = (0..shape.iter().product())
            .map(|i| (i as f32 % 37. - 18.) / 31.)
            .collect::<Vec<_>>();
        let mut request = session(model.clone(), CompileOptions::default(), &shape);
        let expected = request.run(&input).unwrap().data().to_vec();
        if cfg!(feature = "parallel") {
            let cpu = Arc::new(CpuContext::new(4).unwrap());
            let mut parallel = Session::new(
                Arc::new(CompiledModel::new(model).unwrap()),
                cpu,
                &shape,
                ExecutionOptions {
                    intra_op_threads: 4,
                    ..ExecutionOptions::default()
                },
            )
            .unwrap();
            assert!(
                parallel
                    .run(&input)
                    .unwrap()
                    .data()
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
}

#[test]
fn store_epilogue_reads_live_full_residual_and_preserves_intermediate_consumers() {
    let weights = (0..17 * 17)
        .map(|i| (i as f32 % 11. - 5.) / 31.)
        .collect::<Vec<_>>();
    for extra_consumer in [false, true] {
        let mut parts = vec![
            tensor("w", &[17, 17, 1, 1], &weights),
            node("Relu", &["x"], "r", vec![]),
            node("Conv", &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
            node("Add", &["c", "r"], "sum", vec![]),
            node(
                "Sigmoid",
                &["sum"],
                if extra_consumer { "a" } else { "y" },
                vec![],
            ),
        ];
        if extra_consumer {
            parts.push(node("Add", &["a", "c"], "y", vec![]));
        }
        let model = model(parts, &[2, 17, 3, 7], &[2, 17, 3, 7]);
        let input = (0..2 * 17 * 3 * 7)
            .map(|i| (i as f32 % 23. - 11.) / 19.)
            .collect::<Vec<_>>();
        for layout in [LayoutPolicy::Nchw, LayoutPolicy::Auto] {
            let options = CompileOptions {
                layout,
                kernels: KernelPolicy::Portable4,
                ..CompileOptions::default()
            };
            let mut unfused = session(
                model.clone(),
                CompileOptions {
                    fusion: false,
                    ..options
                },
                &[2, 17, 3, 7],
            );
            let expected = unfused.run(&input).unwrap().data().to_vec();
            let mut fused = session(model.clone(), options, &[2, 17, 3, 7]);
            assert!(
                fused
                    .run(&input)
                    .unwrap()
                    .data()
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert_eq!(
                fused.statistics().store_fused_steps,
                usize::from(!extra_consumer)
            );
        }
    }
}
#[test]
fn packing_statistics_report_consumed_panels_for_each_operator() {
    let model = model(
        vec![
            tensor("w", &[2, 2], &[1., 2., 3., 4.]),
            node("MatMul", &["x", "w"], "m", vec![]),
            node("Add", &["m", "w"], "y", vec![]),
        ],
        &[2, 2],
        &[2, 2],
    );
    for layout in [LayoutPolicy::Nchw, LayoutPolicy::Auto] {
        let mut request = session(
            model.clone(),
            CompileOptions {
                kernels: KernelPolicy::Portable4,
                layout,
                fusion: false,
            },
            &[2, 2],
        );
        compare(
            &[8., 12., 18., 26.],
            request.run(&[1., 2., 3., 4.]).unwrap().data(),
        );
        let stats = request.statistics();
        assert!(
            stats
                .selected_kernels
                .iter()
                .any(|s| s.operator == "MatMul" && s.packed)
        );
        assert!(
            stats
                .selected_kernels
                .iter()
                .any(|s| s.operator == "Add" && !s.packed)
        );
        assert!(
            !stats
                .selected_kernels
                .iter()
                .any(|s| s.operator == "Add" && s.packed)
        );
    }
}
#[test]
fn packing_statistics_exclude_prepared_panels_from_canonical_convolution() {
    let model = model(
        vec![
            tensor("a", &[1, 2, 2, 3], &[0.5; 12]),
            tensor("w", &[2, 2, 1, 1], &[1., 2., 3., 4.]),
            node("Conv", &["a", "w"], "c", attr("kernel_shape", &[1, 1])),
            node("Add", &["c", "x"], "y", vec![]),
        ],
        &[1, 2, 2, 3],
        &[1, 2, 2, 3],
    );
    let mut request = session(
        model,
        CompileOptions {
            kernels: KernelPolicy::Portable4,
            ..CompileOptions::default()
        },
        &[1, 2, 2, 3],
    );
    compare(
        &[vec![2.5; 6], vec![4.5; 6]].concat(),
        request.run(&[1.; 12]).unwrap().data(),
    );
    let stats = request.statistics();
    assert!(stats.packed_weight_bytes > 0);
    assert!(
        stats
            .selected_kernels
            .iter()
            .any(|s| s.operator == "Conv" && s.layout == "canonical" && !s.packed)
    );
}
#[test]
fn transpose_panels_are_shared_across_nodes_sessions_and_reshape() {
    let model = model(
        vec![
            tensor(
                "w",
                &[5, 17, 2, 2],
                &(0..5 * 17 * 4)
                    .map(|i| (i as f32 % 23. - 11.) / 47.)
                    .collect::<Vec<_>>(),
            ),
            node(
                "ConvTranspose",
                &["x", "w"],
                "left",
                [attr("kernel_shape", &[2, 2]), attr("strides", &[2, 2])].concat(),
            ),
            node(
                "ConvTranspose",
                &["x", "w"],
                "right",
                [attr("kernel_shape", &[2, 2]), attr("strides", &[2, 2])].concat(),
            ),
            node("Add", &["left", "right"], "y", vec![]),
        ],
        &[-1, 5, -1, -1],
        &[-1, 17, -1, -1],
    );
    let compiled = Arc::new(
        CompiledModel::compile(
            model.clone(),
            CompileOptions {
                kernels: KernelPolicy::Portable4,
                ..CompileOptions::default()
            },
        )
        .unwrap(),
    );
    assert!(Arc::ptr_eq(
        &compiled.transpose_packed[&0],
        &compiled.transpose_packed[&1]
    ));
    assert_eq!(compiled.packed_weight_bytes(), 4 * 5 * 32 * 4);
    let panels = compiled.transpose_packed[&0].clone();
    let mut first =
        Session::new_serial(compiled.clone(), &[2, 5, 3, 5], ExecutionLimits::default()).unwrap();
    let mut second =
        Session::new_serial(compiled.clone(), &[2, 5, 3, 5], ExecutionLimits::default()).unwrap();
    for width in [5, 7, 5] {
        let shape = [2, 5, 3, width];
        let input = (0..shape.iter().product())
            .map(|i| (i as f32 % 31. - 15.) / 19.)
            .collect::<Vec<_>>();
        let mut reference = session(model.clone(), CompileOptions::reference(), &shape);
        let expected = reference.run(&input).unwrap().data().to_vec();
        for request in [&mut first, &mut second] {
            request.reshape(&shape).unwrap();
            compare(&expected, request.run(&input).unwrap().data());
            assert!(Arc::ptr_eq(&panels, &request.compiled.transpose_packed[&0]));
            assert_eq!(request.statistics().packed_weight_bytes, 4 * 5 * 32 * 4);
        }
    }
}
#[test]
fn concat_views_and_dying_binary_operands_share_only_legal_regions() {
    let model = model(
        vec![
            tensor("two", &[], &[2.]),
            node("Relu", &["x"], "left", vec![]),
            node("Mul", &["x", "two"], "right", vec![]),
            node("Concat", &["left", "right"], "joined", axis(0)),
            node("Relu", &["joined"], "y", vec![]),
        ],
        &[-1, 2],
        &[-1, 2],
    );
    let shape = [3, 2];
    let input = [-1., 2., 3., -4., 5., -6.];
    let mut reference = session(model.clone(), CompileOptions::reference(), &shape);
    let expected = reference.run(&input).unwrap().data().to_vec();
    let mut optimized = session(model, CompileOptions::default(), &shape);
    compare(&expected, optimized.run(&input).unwrap().data());
    let state = optimized.optimized.as_ref().unwrap();
    assert!(
        state
            .plan
            .steps
            .iter()
            .any(|s| matches!(s.op, Operation::View))
    );
    let ptr = state.arena.as_ptr();
    optimized.run(&input).unwrap();
    assert_eq!(ptr, optimized.optimized.as_ref().unwrap().arena.as_ptr());
    // Concat views deliberately have no operands after arena placement.
    assert_eq!(optimized.statistics().store_fused_steps, 0);
    let stats = optimized.statistics();
    assert_eq!(
        stats
            .specialized_kernels
            .iter()
            .map(|kernel| kernel.steps)
            .sum::<usize>(),
        stats.physical_steps
    );
}
#[test]
fn metadata_cache_reuses_eight_shapes_and_failed_growth_keeps_previous_run() {
    let model = conv_model(false);
    let mut optimized = session(model, CompileOptions::default(), &[1, 3, 8, 8]);
    for width in 1..=8 {
        optimized.reshape(&[1, 3, 8, width]).unwrap();
    }
    let pointer = optimized.optimized.as_ref().unwrap().arena.as_ptr();
    let miss = optimized.statistics().plan_cache_misses;
    for width in (1..=8).rev() {
        optimized.reshape(&[1, 3, 8, width]).unwrap();
        optimized.run(&vec![0.5; 3 * 8 * width]).unwrap();
    }
    assert_eq!(
        pointer,
        optimized.optimized.as_ref().unwrap().arena.as_ptr()
    );
    assert_eq!(miss, optimized.statistics().plan_cache_misses);
    let before = optimized.input_shape();
    assert!(optimized.reshape(&[1, 3, usize::MAX, 3]).is_err());
    assert_eq!(before, optimized.input_shape());
    optimized.run(&vec![0.5; before.element_count()]).unwrap();
    optimized
        .run_with_trace(&vec![0.5; before.element_count()], |_| {
            Err(RuntimeError::new("observer"))
        })
        .unwrap_err();
    optimized.run(&vec![0.5; before.element_count()]).unwrap();
}

#[test]
fn initialized_arena_keeps_dirty_high_water_through_alternating_shapes() {
    let model = conv_model(true);
    let dirty = f32::from_bits(0x7fc0_1234);
    for layout in [LayoutPolicy::Nchw, LayoutPolicy::Auto] {
        let mut optimized = session(
            model.clone(),
            CompileOptions {
                kernels: KernelPolicy::Portable4,
                layout,
                fusion: true,
            },
            &[1, 3, 8, 64],
        );
        let state = optimized.optimized.as_ref().unwrap();
        let high_water = state.arena.len();
        let retained = state.arena.capacity() * size_of::<f32>();
        let pointer = state.arena.as_ptr();
        for width in [3, 64, 5, 64, 1, 64] {
            // Poison the entire initialized allocation, including inactive
            // high-water elements. Rebinding must preserve it byte for byte;
            // execution must overwrite every input and produced live value.
            optimized.optimized.as_mut().unwrap().arena.fill(dirty);
            let shape = [1, 3, 8, width];
            optimized.reshape(&shape).unwrap();
            let state = optimized.optimized.as_ref().unwrap();
            let active = state.plan.elements;
            assert_eq!(state.arena.as_ptr(), pointer);
            assert_eq!(state.arena.len(), high_water);
            assert!(state.arena.iter().all(|v| v.to_bits() == dirty.to_bits()));
            assert_eq!(optimized.activation_bytes(), active * size_of::<f32>());
            assert_eq!(optimized.retained_activation_bytes(), retained);
            if width < 64 {
                assert!(optimized.activation_bytes() < retained);
            }
            let input = (0..shape.iter().product())
                .map(|i| (i as f32 % 47. - 23.) / 29.)
                .collect::<Vec<_>>();
            let mut reference = session(model.clone(), CompileOptions::reference(), &shape);
            let expected = reference.run(&input).unwrap().data().to_vec();
            compare(&expected, optimized.run(&input).unwrap().data());
            let state = optimized.optimized.as_ref().unwrap();
            assert!(
                state.arena[active..]
                    .iter()
                    .all(|v| v.to_bits() == dirty.to_bits())
            );
            assert_eq!(optimized.retained_activation_bytes(), retained);
        }
    }
}

#[test]
fn rejected_arena_growth_keeps_the_active_plan_output_and_initialized_storage() {
    let model = conv_model(true);
    let compiled = Arc::new(
        CompiledModel::compile(
            model,
            CompileOptions {
                kernels: KernelPolicy::Portable4,
                ..CompileOptions::default()
            },
        )
        .unwrap(),
    );
    let wide = [1, 3, 8, 64];
    let probe = Session::new_serial(compiled.clone(), &wide, ExecutionLimits::default()).unwrap();
    let limit = probe.retained_activation_bytes();
    drop(probe);
    let mut optimized = Session::new_serial(
        compiled,
        &wide,
        ExecutionLimits {
            max_activation_bytes: limit,
        },
    )
    .unwrap();
    optimized.reshape(&[1, 3, 8, 3]).unwrap();
    let input = vec![0.5; 3 * 8 * 3];
    let expected = optimized.run(&input).unwrap().data().to_vec();
    let state = optimized.optimized.as_ref().unwrap();
    let plan = state.plan.clone();
    let pointer = state.arena.as_ptr();
    let storage = state.arena.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let active = optimized.activation_bytes();
    let retained = optimized.retained_activation_bytes();
    assert!(optimized.reshape(&[1, 3, 8, 128]).is_err());
    assert_eq!(optimized.input_shape().dimensions(), [1, 3, 8, 3]);
    let state = optimized.optimized.as_ref().unwrap();
    assert!(Arc::ptr_eq(&state.plan, &plan));
    assert_eq!(state.arena.as_ptr(), pointer);
    assert_eq!(
        state.arena.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        storage
    );
    assert_eq!(optimized.activation_bytes(), active);
    assert_eq!(optimized.retained_activation_bytes(), retained);
    assert_eq!(state.output().unwrap().data(), expected);
    compare(&expected, optimized.run(&input).unwrap().data());
}

#[test]
fn tighter_activation_budget_replaces_the_initialized_high_water_arena() {
    let model = conv_model(false);
    let options = CompileOptions {
        kernels: KernelPolicy::Portable4,
        ..CompileOptions::default()
    };
    let mut optimized = session(model.clone(), options, &[1, 3, 8, 64]);
    let narrow = [1, 3, 8, 3];
    let probe = session(model.clone(), options, &narrow);
    let limit = probe.retained_activation_bytes();
    let active = probe.activation_bytes();
    let previous = optimized.optimized.as_ref().unwrap().arena.as_ptr();
    optimized
        .reshape_with_limits(
            &narrow,
            ExecutionLimits {
                max_activation_bytes: limit,
            },
        )
        .unwrap();
    let state = optimized.optimized.as_ref().unwrap();
    assert_ne!(state.arena.as_ptr(), previous);
    assert_eq!(state.arena.len(), state.plan.elements);
    assert_eq!(optimized.activation_bytes(), active);
    assert!(optimized.retained_activation_bytes() <= limit);
    let input = vec![0.5; 3 * 8 * 3];
    let mut reference = session(model, CompileOptions::reference(), &narrow);
    let expected = reference.run(&input).unwrap().data().to_vec();
    compare(&expected, optimized.run(&input).unwrap().data());
}

fn norm_model(keep: bool, column_gate: bool) -> Arc<Model> {
    let mean_attrs = [
        attr("axes", &[-1]),
        if keep {
            vec![]
        } else {
            blob(5, &[blob(1, b"keepdims"), num(20, 2), num(3, 0)].concat())
        },
    ]
    .concat();
    model(
        vec![
            tensor("two", &[], &[2.]),
            tensor("epsilon", &[], &[0.001]),
            tensor(
                "gamma",
                if column_gate { &[2, 1] } else { &[2] },
                &[0.75, 1.25],
            ),
            tensor("beta", &[2], &[0.1, -0.2]),
            node("ReduceMean", &["x"], "mean", mean_attrs.clone()),
            node("Sub", &["x", "mean"], "centered", vec![]),
            node("Pow", &["centered", "two"], "square", vec![]),
            node("ReduceMean", &["square"], "variance", mean_attrs),
            node("Add", &["variance", "epsilon"], "v", vec![]),
            node("Sqrt", &["v"], "sd", vec![]),
            node("Div", &["centered", "sd"], "normal", vec![]),
            node("Mul", &["normal", "gamma"], "scaled", vec![]),
            node("Add", &["scaled", "beta"], "y", vec![]),
        ],
        &[2, 2],
        &[2, 2],
    )
}
#[test]
fn layer_norm_fuses_only_kept_feature_axis_and_trailing_affine_vectors() {
    for (keep, column_gate, fused) in [(true, false, 8), (false, false, 0), (true, true, 0)] {
        let model = norm_model(keep, column_gate);
        let input = [1., 2., 4., 8.];
        let mut reference = session(model.clone(), CompileOptions::reference(), &[2, 2]);
        let expected = reference.run(&input).unwrap().data().to_vec();
        let mut optimized = session(model, CompileOptions::default(), &[2, 2]);
        compare(&expected, optimized.run(&input).unwrap().data());
        assert_eq!(optimized.statistics().fused_nodes, fused);
    }
}
#[test]
fn activation_chains_match_unfused_values_through_tails_and_signed_zero() {
    for gelu in [false, true] {
        let mut parts = vec![];
        if gelu {
            parts.extend([
                tensor("sqrt2", &[], &[std::f32::consts::SQRT_2]),
                tensor("one", &[], &[1.]),
                tensor("half", &[], &[0.5]),
                node("Div", &["x", "sqrt2"], "v", vec![]),
                node("Erf", &["v"], "e", vec![]),
                node("Add", &["e", "one"], "a", vec![]),
                node("Mul", &["a", "x"], "m", vec![]),
                node("Mul", &["m", "half"], "y", vec![]),
            ]);
        } else {
            let attrs = [
                blob(
                    5,
                    &[
                        blob(1, b"alpha"),
                        num(20, 1),
                        [vec![2 * 8 + 5], 0.16666667f32.to_le_bytes().to_vec()].concat(),
                    ]
                    .concat(),
                ),
                blob(
                    5,
                    &[
                        blob(1, b"beta"),
                        num(20, 1),
                        [vec![2 * 8 + 5], 0.5f32.to_le_bytes().to_vec()].concat(),
                    ]
                    .concat(),
                ),
            ]
            .concat();
            parts.extend([
                node("HardSigmoid", &["x"], "gate", attrs),
                node("Mul", &["x", "gate"], "y", vec![]),
            ]);
        }
        let model = model(parts, &[3, 9], &[3, 9]);
        let input = (0..27).map(|i| (i as f32 - 13.) / 3.).collect::<Vec<_>>();
        let mut reference = session(model.clone(), CompileOptions::reference(), &[3, 9]);
        let expected = reference.run(&input).unwrap().data().to_vec();
        let mut optimized = session(model, CompileOptions::default(), &[3, 9]);
        compare(&expected, optimized.run(&input).unwrap().data());
        assert_eq!(optimized.statistics().fused_nodes, if gelu { 4 } else { 1 });
    }
}

#[test]
fn compound_activations_execute_inside_convolution_stores_in_both_layouts() {
    // Independent exact-rational binary32 evaluation of the pinned C# native
    // GELU expression for (i-27)/3. Both FMA policies give these same grid
    // values; unary/store policy tests separately distinguish FMA rounding.
    const GELU_REFERENCE_BITS: [u32; 54] = [
        0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000,
        0x80000000, 0x80000000, 0x80000000, 0x80000000, 0xb42aaaab, 0xb5c80000, 0xb6f2aaaa,
        0xb8054000, 0xb904e000, 0xb9ec2800, 0xbabb7600, 0xbb84b280, 0xbc275a00, 0xbcbb9dea,
        0xbd3a5e70, 0xbda31fd8, 0xbdf91140, 0xbe227684, 0xbe2c5e44, 0xbdfc348e, 0x00000000,
        0x3e573b0f, 0x3eff2633, 0x3f57625f, 0x3f9b1997, 0x3fcb2358, 0x3ffa2d0c, 0x4013de19,
        0x402a0351, 0x403fbda7, 0x40553de7, 0x406aa34a, 0x407ffdec, 0x408aaa68, 0x40955546,
        0x409ffffd, 0x40aaaaab, 0x40b55555, 0x40c00000, 0x40caaaab, 0x40d55555, 0x40e00000,
        0x40eaaaab, 0x40f55555, 0x41000000, 0x41055555, 0x410aaaab,
    ];
    for gelu in [false, true] {
        let mut parts = vec![
            tensor("w", &[3, 3, 1, 1], &[1., 0., 0., 0., 1., 0., 0., 0., 1.]),
            node("Conv", &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
        ];
        if gelu {
            parts.extend([
                tensor("sqrt2", &[], &[std::f32::consts::SQRT_2]),
                tensor("one", &[], &[1.]),
                tensor("half", &[], &[0.5]),
                node("Div", &["c", "sqrt2"], "v", vec![]),
                node("Erf", &["v"], "e", vec![]),
                node("Add", &["e", "one"], "a", vec![]),
                node("Mul", &["a", "c"], "m", vec![]),
                node("Mul", &["m", "half"], "y", vec![]),
            ]);
        } else {
            let attribute = |name: &str, value: f32| {
                blob(
                    5,
                    &[
                        blob(1, name.as_bytes()),
                        num(20, 1),
                        [vec![2 * 8 + 5], value.to_le_bytes().to_vec()].concat(),
                    ]
                    .concat(),
                )
            };
            parts.extend([
                node(
                    "HardSigmoid",
                    &["c"],
                    "gate",
                    [attribute("alpha", 0.16666667), attribute("beta", 0.5)].concat(),
                ),
                node("Mul", &["c", "gate"], "y", vec![]),
            ]);
        }
        let model = model(parts, &[2, 3, 1, 9], &[2, 3, 1, 9]);
        let input = (0..54).map(|i| (i as f32 - 27.) / 3.).collect::<Vec<_>>();
        for layout in [LayoutPolicy::Nchw, LayoutPolicy::Auto] {
            for kernels in [KernelPolicy::Portable4, KernelPolicy::Auto] {
                let options = CompileOptions {
                    kernels,
                    layout,
                    ..CompileOptions::default()
                };
                let mut unfused = session(
                    model.clone(),
                    CompileOptions {
                        fusion: false,
                        ..options
                    },
                    &[2, 3, 1, 9],
                );
                let expected = unfused.run(&input).unwrap().data().to_vec();
                let mut fused = session(model.clone(), options, &[2, 3, 1, 9]);
                let actual = fused.run(&input).unwrap().data();
                if gelu {
                    assert_eq!(
                        actual
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        GELU_REFERENCE_BITS,
                        "reference GELU {layout:?} {kernels:?}"
                    );
                    // Reference GELU multiplies by rounded inverse-sqrt2. The
                    // unfused ONNX graph divides by rounded sqrt2, so their
                    // intermediates are not bit-identical. Keep the fixed
                    // roadmap tensor contract as well as the exact oracle.
                    assert!(actual.iter().zip(&expected).all(|(a, b)| {
                        a.is_finite() && b.is_finite() && (a - b).abs() <= 0.001 + 0.001 * b.abs()
                    }));
                } else {
                    assert!(
                        actual
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{layout:?} {kernels:?} hard-swish"
                    );
                }
                assert_eq!(fused.statistics().store_fused_steps, 1);
            }
        }
    }
}
#[test]
fn constant_shape_views_materialize_values_instead_of_aliasing_unwritten_arena() {
    let model = model(
        vec![
            tensor("w", &[1, 2], &[2., 3.]),
            tensor("axes", &[1], &[0.]),
            node("Squeeze", &["w"], "v", attr("axes", &[0])),
            node("Add", &["x", "v"], "y", vec![]),
        ],
        &[2],
        &[2],
    );
    let mut reference = session(model.clone(), CompileOptions::reference(), &[2]);
    let expected = reference.run(&[1., 4.]).unwrap().data().to_vec();
    let mut optimized = session(model, CompileOptions::default(), &[2]);
    assert_eq!(&expected, &[3., 7.]);
    compare(&expected, optimized.run(&[1., 4.]).unwrap().data());
}
#[test]
fn constant_activation_and_full_constant_residual_keep_their_physical_layout() {
    for constant_activation in [false, true] {
        let parts = vec![
            tensor("w", &[2, 2, 1, 1], &[1., 0.25, -0.5, 2.]),
            tensor("c", &[1, 2, 2, 2], &[0., 1., 2., 3., 4., 5., 6., 7.]),
            node(
                "Conv",
                &[if constant_activation { "c" } else { "x" }, "w"],
                "conv",
                attr("kernel_shape", &[1, 1]),
            ),
            node(
                "Add",
                &[if constant_activation { "x" } else { "c" }, "conv"],
                "y",
                vec![],
            ),
        ];
        let model = model(parts, &[1, 2, 2, 2], &[1, 2, 2, 2]);
        let input = [1., 2., 3., 4., 5., 6., 7., 8.];
        let mut reference = session(model.clone(), CompileOptions::reference(), &[1, 2, 2, 2]);
        let expected = reference.run(&input).unwrap().data().to_vec();
        let mut optimized = session(model, CompileOptions::default(), &[1, 2, 2, 2]);
        compare(&expected, optimized.run(&input).unwrap().data());
    }
}
#[test]
fn diagnostic_and_conversion_storage_is_included_in_retained_accounting() {
    let model = model(
        vec![
            tensor("w", &[16, 3, 1, 1], &[0.125; 48]),
            node("Conv", &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
            node("Relu", &["c"], "y", vec![]),
        ],
        &[-1, 3, -1, -1],
        &[-1, 16, -1, -1],
    );
    let mut optimized = session(model, CompileOptions::default(), &[1, 3, 4, 5]);
    let input = vec![0.5; 60];
    optimized.run(&input).unwrap();
    let before = optimized.retained_activation_bytes();
    let planning = optimized.retained_planning_bytes();
    assert!(optimized.retained_scratch_bytes() >= 240);
    optimized.run_with_trace(&input, |_| Ok(())).unwrap();
    assert!(optimized.retained_activation_bytes() > before);
    assert!(optimized.retained_planning_bytes() > planning);
    let expected = optimized.optimized.as_ref().unwrap().retained_bytes()
        + optimized
            .diagnostic
            .as_ref()
            .unwrap()
            .retained_activation_bytes();
    assert_eq!(optimized.retained_activation_bytes(), expected);
}
#[test]
fn self_add_epilogue_is_executed_without_an_undefined_fusion_operand() {
    let model = model(
        vec![
            tensor("w", &[2, 2, 1, 1], &[1., 0.25, -0.5, 2.]),
            node("Conv", &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
            node("Add", &["c", "c"], "y", vec![]),
        ],
        &[1, 2, 2, 2],
        &[1, 2, 2, 2],
    );
    let input = [1., 2., 3., 4., 5., 6., 7., 8.];
    let mut reference = session(model.clone(), CompileOptions::reference(), &[1, 2, 2, 2]);
    let expected = reference.run(&input).unwrap().data().to_vec();
    let mut optimized = session(model, CompileOptions::default(), &[1, 2, 2, 2]);
    compare(&expected, optimized.run(&input).unwrap().data());
}
#[test]
fn binary_rank_promotion_canonicalizes_nhwc_before_writing_rank_five_output() {
    let model = model(
        vec![
            tensor("w", &[2, 2, 1, 1], &[1., 0.25, -0.5, 2.]),
            tensor("bias", &[1, 1, 1, 1, 1], &[0.5]),
            node("Conv", &["x", "w"], "c", attr("kernel_shape", &[1, 1])),
            node("Add", &["c", "bias"], "y", vec![]),
        ],
        &[1, 2, 2, 2],
        &[1, 1, 2, 2, 2],
    );
    let input = [1., 2., 3., 4., 5., 6., 7., 8.];
    let mut reference = session(model.clone(), CompileOptions::reference(), &[1, 2, 2, 2]);
    let expected = reference.run(&input).unwrap().data().to_vec();
    let mut optimized = session(model, CompileOptions::default(), &[1, 2, 2, 2]);
    compare(&expected, optimized.run(&input).unwrap().data());
    assert_eq!(optimized.output_shape().dimensions(), [1, 1, 2, 2, 2]);
}
#[test]
fn activation_limit_is_checked_against_allocated_physical_storage() {
    let model = norm_model(true, false);
    let shape = [2, 2];
    let mut unlimited = session(model.clone(), CompileOptions::default(), &shape);
    let bytes = unlimited.activation_bytes();
    let reference = session(model.clone(), CompileOptions::reference(), &shape);
    assert!(bytes < reference.activation_bytes());
    let compiled = Arc::new(CompiledModel::compile(model, CompileOptions::default()).unwrap());
    let mut limited = Session::new_serial(
        compiled,
        &shape,
        ExecutionLimits {
            max_activation_bytes: bytes,
        },
    )
    .unwrap();
    let input = [1., 2., 4., 8.];
    compare(
        unlimited.run(&input).unwrap().data(),
        limited.run(&input).unwrap().data(),
    );
}

#[test]
fn constant_matmul_reuses_nhwc_panels_only_under_reference_eligibility_and_keeps_bias_separate() {
    for columns in [15, 16, 17, 32, 1008, 1024] {
        let parsed = model(
            vec![
                tensor("w", &[3, columns], &vec![1.; 3 * columns]),
                tensor("bias", &[columns], &vec![1.; columns]),
                node("MatMul", &["x", "w"], "m", vec![]),
                node("Add", &["m", "bias"], "y", vec![]),
            ],
            &[-1, 3],
            &[-1, columns as i64],
        );
        for layout in [LayoutPolicy::Nchw, LayoutPolicy::Auto] {
            let options = CompileOptions {
                kernels: KernelPolicy::Portable4,
                layout,
                fusion: true,
            };
            let mut request = session(parsed.clone(), options, &[7, 3]);
            let cached = parsed.cached_packed_weight_bytes();
            let input: Vec<_> = (0..7).flat_map(|_| [1e20, -1e20, 3.]).collect();
            for rows in [7, 6, 7] {
                request.reshape(&[rows, 3]).unwrap();
                let actual = request.run(&input[..rows * 3]).unwrap().data().to_vec();
                assert!(
                    actual
                        .iter()
                        .all(|&value| value.to_bits() == 4f32.to_bits()),
                    "bias Add must follow unbiased sum"
                );
                assert_eq!(
                    parsed.cached_packed_weight_bytes(),
                    cached,
                    "execution/reshape must reuse immutable panels"
                );
                let selected = request
                    .statistics()
                    .specialized_kernels
                    .into_iter()
                    .any(|kernel| kernel.family == "nhwc_constant_matmul_6x8_pair2");
                assert_eq!(
                    selected,
                    layout == LayoutPolicy::Auto && columns.is_multiple_of(16) && columns < 1024
                );
            }
        }
    }
}
