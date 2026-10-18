//! Compare panel-wide fused stores with independently executed graph operators.
use super::*;

fn activation_nodes(operation: &str, input: &str, output: &str) -> Vec<Vec<u8>> {
    if operation != "HardSwish" {
        return vec![node(operation, &[input], output, vec![])];
    }
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
    vec![
        node(
            "HardSigmoid",
            &[input],
            "gate",
            [attribute("alpha", 0.16666667), attribute("beta", 0.5)].concat(),
        ),
        node("Mul", &[input, "gate"], output, vec![]),
    ]
}

#[test]
fn neon_panel_stores_preserve_activation_order_residuals_and_batch_tails() {
    if !simd::neon_available() {
        return;
    }
    // Narrow NEON pointwise, SME-eligible pointwise, and spatial NEON all
    // produce complete channel panels. Two batches and 102 spatial rows
    // exercise short row groups and nonzero output-band offsets.
    for (channels, kernel) in [
        (32, 1),
        (192, 1),
        (32, 2),
        (32, 3),
        (64, 3),
        (32, 5),
        (32, 7),
    ] {
        let shape = [2, channels, 3, 17];
        let weights: Vec<_> = (0..channels * channels * kernel * kernel)
            .map(|i| (i % 17) as f32 / 64. - 0.125)
            .collect();
        let bias: Vec<_> = (0..channels).map(|i| i as f32 / 32. - 2.).collect();
        for activation in ["Relu", "HardSwish", "Sigmoid"] {
            for activation_first in [false, true] {
                let mut parts = vec![
                    tensor("w", &[channels, channels, kernel, kernel], &weights),
                    tensor("b", &[1, channels, 1, 1], &bias),
                    node("Relu", &["x"], "residual", vec![]),
                    node(
                        "Conv",
                        &["x", "w"],
                        "conv",
                        [
                            attr("kernel_shape", &[kernel as i64; 2]),
                            attr(
                                "pads",
                                &[
                                    ((kernel - 1) / 2) as i64,
                                    ((kernel - 1) / 2) as i64,
                                    (kernel / 2) as i64,
                                    (kernel / 2) as i64,
                                ],
                            ),
                        ]
                        .concat(),
                    ),
                ];
                if activation_first {
                    parts.extend(activation_nodes(activation, "conv", "a"));
                    parts.push(node("Add", &["a", "b"], "biased", vec![]));
                } else {
                    parts.push(node("Add", &["conv", "b"], "a", vec![]));
                    parts.extend(activation_nodes(activation, "a", "biased"));
                }
                parts.push(node("Add", &["biased", "residual"], "y", vec![]));
                let dims = shape.map(|v| v as i64);
                let model = model(parts, &dims, &dims);
                let options = CompileOptions {
                    kernels: KernelPolicy::NeonFma,
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
                let mut fused = session(model, options, &shape);
                let threads = if cfg!(feature = "parallel") { 2 } else { 1 };
                fused
                    .set_execution_context(Arc::new(CpuContext::new(threads).unwrap()), threads)
                    .unwrap();
                assert_eq!(selection_policy::input_layout_at(&fused, 1), Layout::Nhwc);
                assert!(fused.statistics().store_fused_steps > 0);
                for shift in [0, 7] {
                    let input: Vec<_> = (0..shape.iter().product())
                        .map(|i| ((i + shift) % 31) as f32 / 16. - 1.)
                        .collect();
                    let expected = unfused.run(&input).unwrap().data().to_vec();
                    let actual = fused.run(&input).unwrap().data();
                    assert_eq!(actual.len(), expected.len());
                    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "channels={channels} kernel={kernel} activation={activation} first={activation_first} shift={shift} index={i}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn neon_direct_transpose_stores_preserve_fused_graph_order_and_channel_tails() {
    if !cfg!(target_os = "macos") || !simd::neon_available() {
        return;
    }
    for channels in [64, 65] {
        let shape = [2, channels, 3, 17];
        let output_shape = [2, channels, 6, 34];
        let weights: Vec<_> = (0..channels * channels * 4)
            .map(|i| (i % 19) as f32 / 37. - 0.25)
            .collect();
        let bias: Vec<_> = (0..channels).map(|i| i as f32 / 13. - 2.).collect();
        let attributes = [attr("kernel_shape", &[2, 2]), attr("strides", &[2, 2])].concat();
        for activation in ["Relu", "HardSwish", "Sigmoid"] {
            for activation_first in [false, true] {
                let mut parts = vec![
                    tensor("w", &[channels, channels, 2, 2], &weights),
                    tensor("b", &[1, channels, 1, 1], &bias),
                    node("ConvTranspose", &["x", "w"], "branch", attributes.clone()),
                    node("Relu", &["branch"], "residual", vec![]),
                    node("ConvTranspose", &["x", "w"], "conv", attributes.clone()),
                ];
                if activation_first {
                    parts.extend(activation_nodes(activation, "conv", "a"));
                    parts.push(node("Add", &["a", "b"], "biased", vec![]));
                } else {
                    parts.push(node("Add", &["conv", "b"], "a", vec![]));
                    parts.extend(activation_nodes(activation, "a", "biased"));
                }
                parts.push(node("Add", &["biased", "residual"], "y", vec![]));
                let model = model(
                    parts,
                    &shape.map(|v| v as i64),
                    &output_shape.map(|v| v as i64),
                );
                let options = CompileOptions {
                    kernels: KernelPolicy::NeonFma,
                    ..Default::default()
                };
                let mut unfused = session(
                    model.clone(),
                    CompileOptions {
                        fusion: false,
                        ..options
                    },
                    &shape,
                );
                let mut fused = session(model, options, &shape);
                let threads = if cfg!(feature = "parallel") { 3 } else { 1 };
                fused
                    .set_execution_context(Arc::new(CpuContext::new(threads).unwrap()), threads)
                    .unwrap();
                for shift in [0, 7] {
                    let input: Vec<_> = (0..shape.iter().product())
                        .map(|i| ((i + shift) % 31) as f32 / 23. - 1.)
                        .collect();
                    let expected = unfused.run(&input).unwrap().data().to_vec();
                    let actual = fused.run(&input).unwrap().data();
                    assert_eq!(actual.len(), expected.len());
                    assert!(
                        actual
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "channels={channels} activation={activation} first={activation_first} shift={shift}"
                    );
                    assert!(
                        fused
                            .statistics()
                            .specialized_kernels
                            .iter()
                            .any(|s| s.family == "nhwc_transpose_4tap_direct" && s.store_fused)
                    );
                }
            }
        }
    }
}
