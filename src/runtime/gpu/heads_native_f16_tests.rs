use super::*;
use crate::runtime::gpu::tests::{integer, integers, model, node, tensor};
use crate::runtime::{CompileOptions, GpuCompileOptions};

fn context() -> Arc<GpuContext> {
    GpuContext::new_with_options(GpuContextOptions::default()).unwrap()
}

fn dense(count: usize, salt: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * salt + 7) % modulus) as f32 * 2. + 1. - modulus as f32) / scale)
        .collect()
}

fn fixture(
    context: &Arc<GpuContext>,
    [rows, inner, classes]: [usize; 3],
    precision: GpuPrecision,
    matrix_precision: GpuMatrixPrecision,
    [features, weights, bias]: [&[f32]; 3],
) -> (Arc<GpuCompiledModel>, Arc<Head>, Tensor) {
    let graph = model(
        vec![
            tensor("weights", &[inner, classes], weights),
            tensor("bias", &[classes], bias),
            tensor("zero", &[1], &[0.]),
            node("Add", &["x", "zero"], "features", &[]),
            node("MatMul", &["features", "weights"], "logits", &[]),
            node("Add", &["logits", "bias"], "biased", &[]),
            node("Softmax", &["biased"], "y", &[]),
        ],
        &[rows, 1, inner],
        &[rows, 1, classes],
    );
    let compiled = Arc::new(
        GpuCompiledModel::compile_with_options(
            graph,
            context.clone(),
            CompileOptions {
                gpu: GpuCompileOptions {
                    precision,
                    matrix_precision,
                },
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let head = Head::shared(&compiled, classes, 3, None).unwrap();
    let feature = Tensor::allocate(
        context.clone(),
        Shape::new(&[rows, 1, inner]).unwrap(),
        precision,
    )
    .unwrap();
    context
        .upload_resident(&feature.buffer, &precision.encode(features))
        .unwrap();
    (compiled, head, feature)
}

fn oracle(
    [rows, inner, classes]: [usize; 3],
    features: &[f32],
    weights: &[f32],
    bias: &[f32],
) -> Vec<(u32, f64, f64)> {
    (0..rows)
        .map(|row| {
            let logits: Vec<_> = (0..classes)
                .map(|class| {
                    f64::from(bias[class])
                        + (0..inner)
                            .map(|k| {
                                f64::from(features[row * inner + k])
                                    * f64::from(weights[k * classes + class])
                            })
                            .sum::<f64>()
                })
                .collect();
            let mut winner = 0;
            for class in 1..classes {
                if logits[class] > logits[winner] {
                    winner = class;
                }
            }
            let maximum = logits[winner];
            let probability = logits
                .iter()
                .map(|&value| (value - maximum).exp())
                .sum::<f64>()
                .recip();
            (winner as u32, maximum, probability)
        })
        .collect()
}

fn rounded(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&value| half::f16::from_f32(value).to_f32())
        .collect()
}
fn word(record: &[u8], index: usize) -> u32 {
    u32::from_le_bytes(record[index * 4..index * 4 + 4].try_into().unwrap())
}

fn read(
    head: &Head,
    features: &Tensor,
    probability: bool,
    half: bool,
    widen: bool,
) -> (Vec<u8>, Vec<u8>) {
    let rows = features.descriptor.shape.element_count() / head.metadata.inner;
    let mut batch = Batch::new(head.context.clone()).unwrap();
    let winners = head.project(&mut batch, features, probability).unwrap();
    assert_eq!(
        batch
            .statistics
            .kernels
            .iter()
            .any(|name| name == NATIVE_PROJECTION_F16_NAME),
        half
    );
    assert_eq!(
        batch
            .statistics
            .kernels
            .iter()
            .any(|name| name == "projection_features_F16_F32"),
        widen
    );
    let decoded = head
        .decode(&mut batch, &winners, 1, rows, true, usize::MAX)
        .unwrap();
    let winner_bytes = rows * 16;
    let decoded_bytes = (rows + 1) * 24;
    let combined = batch
        .storage("Half projection contract", winner_bytes + decoded_bytes)
        .unwrap();
    batch.copy(&winners, 0, &combined, 0, winner_bytes as u64);
    batch.copy(
        &decoded,
        0,
        &combined,
        winner_bytes as u64,
        decoded_bytes as u64,
    );
    let (mut actual, statistics) = batch
        .read(&combined, winner_bytes + decoded_bytes, false)
        .unwrap();
    assert_eq!(statistics.diagnostic_readback_bytes, 0);
    assert!(statistics.native_resident_dispatches > 0);
    assert_eq!(
        statistics.native_resident_dispatches, statistics.dispatches,
        "projection, reduction and CTC must all use the native resident path"
    );
    let decoded = actual.split_off(winner_bytes);
    (actual, decoded)
}

fn assert_oracle(actual: &[u8], expected: &[(u32, f64, f64)], probability: bool) {
    for (row, record) in actual.as_chunks::<16>().0.iter().enumerate() {
        assert_eq!(word(record, 0), expected[row].0, "winner row{row}");
        assert_eq!(word(record, 3), 0, "finite row{row}");
        let reference = if probability {
            expected[row].2
        } else {
            expected[row].1
        };
        let score = f64::from(f32::from_bits(word(record, 1)));
        // Same score gate as the original native F32 head, now against an
        // independent Half-operand oracle. The F32 head contract is unchanged.
        assert!(
            score.is_finite() && (score - reference).abs() <= 1e-6 + 2e-5 * reference.abs(),
            "row{row}, probability{probability}: GPU{score} versus oracle{reference}"
        );
    }
}

#[test]
#[ignore = "requires physical Metal; optional Half vocabulary projection with dense row/class/K tails"]
fn gpu_native_half_projection_dense_tails_and_score_modes() {
    let context = context();
    for dimensions @ [rows, inner, classes] in [[8, 7, 4097], [32, 16, 4160], [33, 17, 4099]] {
        let features = dense(rows * inner, 13, 63, 131.);
        let weights = dense(inner * classes, 17, 127, 257.);
        let mut bias = dense(classes, 7, 31, 251.);
        bias[classes - 1] = 10.;
        let expected = oracle(dimensions, &rounded(&features), &rounded(&weights), &bias);
        let (compiled, head, feature) = fixture(
            &context,
            dimensions,
            GpuPrecision::MixedF16,
            GpuMatrixPrecision::F16,
            [&features, &weights, &bias],
        );
        let half_weights = head
            .half_weights
            .as_ref()
            .expect("Half projection must be qualified");
        assert!(
            Arc::ptr_eq(
                &half_weights.reservation,
                &compiled.weights.constants[head.metadata.weights.0]
                    .as_ref()
                    .unwrap()
                    .reservation
            ),
            "Half head must reuse the model constant allocation"
        );
        for probability in [false, true] {
            let (actual, decoded) = read(&head, &feature, probability, true, false);
            assert_oracle(&actual, &expected, probability);
            let decoded = Recognition::parse(&decoded, 1, rows, classes).unwrap();
            let token = decoded[0].tokens.iter().next().unwrap();
            assert_eq!(
                (token.class, token.start, token.end),
                (classes - 1, 0, rows)
            );
        }
    }
}

#[test]
#[ignore = "requires physical Metal; Half-quantized cross-tile ties and nonfinite propagation"]
fn gpu_native_half_projection_ties_and_nonfinite() {
    let context = context();
    let dimensions @ [rows, inner, classes] = [33, 8, 4097];
    let mut features = vec![1.; rows * inner];
    for row in 0..rows {
        features[row * inner] = if row % 2 == 0 { 1. } else { -1. };
    }
    let mut weights = vec![0.; inner * classes];
    // Distinct original F32 weights become an exact Half tie across tiles.
    // This also proves that the fast path reads its reused Half constant.
    weights[63] = 2.0001;
    weights[64] = 2.0002;
    weights[classes - 1] = -2.;
    let mut bias = vec![0.; classes];
    for class in [63, 64, classes - 1] {
        bias[class] = 4.;
    }
    let expected = oracle(dimensions, &rounded(&features), &rounded(&weights), &bias);
    assert_eq!(expected[0].0, 63);
    assert_eq!(oracle(dimensions, &features, &weights, &bias)[0].0, 64);
    let (_, head, feature) = fixture(
        &context,
        dimensions,
        GpuPrecision::MixedF16,
        GpuMatrixPrecision::F16,
        [&features, &weights, &bias],
    );
    for probability in [false, true] {
        let (actual, decoded) = read(&head, &feature, probability, true, false);
        assert_oracle(&actual, &expected, probability);
        Recognition::parse(&decoded, 1, rows, classes).unwrap();
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 65520.] {
        for location in 0..4 {
            let mut case_weights = weights.clone();
            let mut case_features = features.clone();
            if location < 2 {
                case_weights[if location == 0 {
                    65
                } else {
                    inner * classes - 1
                }] = invalid;
            } else {
                case_features[if location == 2 { 0 } else { rows * inner - 1 }] = invalid;
            }
            let defensive_weights = location < 2 && !invalid.is_finite();
            // The loader correctly rejects nonfinite F32 constants. Only
            // finite65520 overflow uses an actual compiled Half constant;
            // unreachable NaN/Inf operands exercise the defensive shader path
            // through a separately owned test buffer, never a model mutation.
            let model_weights = if defensive_weights {
                &weights
            } else {
                &case_weights
            };
            let (compiled, case_head, case_feature) = fixture(
                &context,
                dimensions,
                GpuPrecision::MixedF16,
                GpuMatrixPrecision::F16,
                [&case_features, model_weights, &bias],
            );
            let defensive_head = if defensive_weights {
                let owned = context
                    .create_buffer_init(
                        "defensive Half projection weights",
                        &GpuPrecision::MixedF16.encode(&case_weights),
                        BufferUsage::STORAGE | BufferUsage::COPY_DST,
                    )
                    .unwrap();
                let mut local = Head::new(&compiled, classes, 3, None).unwrap();
                assert!(!Arc::ptr_eq(
                    &owned.reservation,
                    &compiled.weights.constants[local.metadata.weights.0]
                        .as_ref()
                        .unwrap()
                        .reservation,
                ));
                local.owned_bytes += owned.bytes();
                local.half_weights = Some(owned);
                Some(local)
            } else {
                assert!(Arc::ptr_eq(
                    &case_head.half_weights.as_ref().unwrap().reservation,
                    &compiled.weights.constants[case_head.metadata.weights.0]
                        .as_ref()
                        .unwrap()
                        .reservation,
                ));
                None
            };
            let active_head = defensive_head.as_ref().unwrap_or(case_head.as_ref());
            for probability in [false, true] {
                let (actual, decoded) = read(active_head, &case_feature, probability, true, false);
                for (row, record) in actual.as_chunks::<16>().0.iter().enumerate() {
                    assert_eq!(
                        word(record, 3) != 0,
                        location < 2
                            || (location == 2 && row == 0)
                            || (location == 3 && row == rows - 1),
                        "location{location}, input{invalid}, row{row}, probability{probability}"
                    );
                }
                assert!(Recognition::parse(&decoded, 1, rows, classes).is_err());
            }
        }
    }
}

#[test]
#[ignore = "requires physical Metal; Half weight sharing, strict/opt-out fallback and actual feature dtype"]
fn gpu_native_half_projection_policy_fallback_and_weight_sharing() {
    let dimensions @ [rows, inner, classes] = [8, 7, 4097];
    let features = dense(rows * inner, 13, 63, 131.);
    let weights = dense(inner * classes, 17, 127, 257.);
    let mut bias = dense(classes, 7, 31, 251.);
    bias[classes - 1] = 10.;
    for matrix in [GpuMatrixPrecision::F16, GpuMatrixPrecision::F32] {
        let context = context();
        let (compiled, head, feature) = fixture(
            &context,
            dimensions,
            GpuPrecision::MixedF16,
            matrix,
            [&features, &weights, &bias],
        );
        let use_half = matrix == GpuMatrixPrecision::F16;
        assert_eq!(head.half_weights.is_some(), use_half);
        // No additional vocabulary Half allocation belongs to Head. Only its
        // existing F32 fallback weights/bias and token lengths are owned here.
        assert_eq!(head.retained_bytes(), (inner * classes + 2 * classes) * 4);
        let before = context.memory_statistics().used_bytes;
        let repeated = Head::shared(&compiled, classes, 3, None).unwrap();
        assert!(Arc::ptr_eq(&head, &repeated));
        assert_eq!(context.memory_statistics().used_bytes, before);
        let expected_weights = if use_half {
            rounded(&weights)
        } else {
            weights.clone()
        };
        let expected = oracle(dimensions, &rounded(&features), &expected_weights, &bias);
        let (actual, _) = read(&head, &feature, false, use_half, !use_half);
        assert_oracle(&actual, &expected, false);
        if use_half {
            // The explicit mode alone cannot reinterpret F32 feature storage.
            let full =
                Tensor::allocate(context.clone(), feature.descriptor.shape, GpuPrecision::F32)
                    .unwrap();
            context
                .upload_resident(&full.buffer, bytemuck::cast_slice(&features))
                .unwrap();
            let (actual, _) = read(&head, &full, false, false, false);
            assert_oracle(
                &actual,
                &oracle(dimensions, &features, &weights, &bias),
                false,
            );
            // Rows below native admission still widen and use the F32 head.
            let small = Tensor::allocate(
                context.clone(),
                Shape::new(&[7, 1, inner]).unwrap(),
                GpuPrecision::MixedF16,
            )
            .unwrap();
            context
                .upload_resident(
                    &small.buffer,
                    &GpuPrecision::MixedF16.encode(&features[..7 * inner]),
                )
                .unwrap();
            let (actual, _) = read(&head, &small, false, false, true);
            assert_oracle(
                &actual,
                &oracle(
                    [7, inner, classes],
                    &rounded(&features[..7 * inner]),
                    &weights,
                    &bias,
                ),
                false,
            );
        }
    }
}

fn classifier_weights() -> Vec<f32> {
    (0..8)
        .flat_map(|channel| {
            // Deliberately not Half-exact: the two-class fallback must retain
            // the original F32 projection instead of reusing Half constants.
            let weight = 0.40013 + channel as f32 / 128.;
            [weight, -weight]
        })
        .collect()
}

fn classifier_model(with_bias: bool) -> Arc<Model> {
    let mut parts = vec![
        tensor(
            "conv_weights",
            &[8, 3, 1, 1],
            &[0.125, 0.25, 0.125].repeat(8),
        ),
        tensor(
            "conv_bias",
            &[8],
            &(0..8)
                .map(|channel| channel as f32 / 64. - 0.0625)
                .collect::<Vec<_>>(),
        ),
        tensor("projection_weights", &[8, 2], &classifier_weights()),
        node(
            "Conv",
            &["x", "conv_weights", "conv_bias"],
            "convolved",
            &[integers("kernel_shape", &[1, 1])],
        ),
        node(
            "ReduceMean",
            &["convolved"],
            "features",
            &[integers("axes", &[2, 3]), integer("keepdims", 0)],
        ),
        node("MatMul", &["features", "projection_weights"], "logits", &[]),
    ];
    if with_bias {
        parts.push(tensor("projection_bias", &[2], &[0.0625, -0.0625]));
        parts.push(node("Add", &["logits", "projection_bias"], "biased", &[]));
    }
    parts.push(node(
        "Softmax",
        &[if with_bias { "biased" } else { "logits" }],
        "y",
        &[],
    ));
    model(parts, &[1, 3, 80, 160], &[1, 2])
}

fn classifier_oracle(pixel: u8, with_bias: bool, half: bool) -> Vec<(u32, f64, f64)> {
    let round = |value| {
        if half {
            half::f16::from_f32(value).to_f32()
        } else {
            value
        }
    };
    let input = round(f32::from(pixel) * (2.0f32 / 255.0f32) - 1.);
    let features = (0..8)
        .map(|channel| {
            let mut value = channel as f32 / 64. - 0.0625;
            for weight in [0.125, 0.25, 0.125] {
                value += input * weight;
            }
            // Every spatial position is identical. The dyadic Half convolution
            // and the 64-lane mean reduction have exact F32 sums, so the pooled
            // feature is the stored convolution value, with no resize oracle.
            round(value)
        })
        .collect::<Vec<_>>();
    let bias = if with_bias {
        [0.0625, -0.0625]
    } else {
        [0.; 2]
    };
    oracle([1, 8, 2], &features, &classifier_weights(), &bias)
}

fn assert_classifier_native_path(statistics: &GpuPipelineStatistics, half: bool) {
    assert!(statistics.native_resident_dispatches > 0);
    assert!(statistics.native_graph_dispatches > 0);
    assert_eq!(
        statistics.native_resident_dispatches + statistics.native_graph_dispatches,
        statistics.dispatches,
        "native classifier coverage: {} resident + {} neural versus {} total dispatches; kernels {:?}",
        statistics.native_resident_dispatches,
        statistics.native_graph_dispatches,
        statistics.dispatches,
        statistics.kernels
    );
    assert!(statistics.cpu_compute_stages.is_empty());
    assert_eq!(statistics.image_processing, GpuPlacement::Gpu);
    assert_eq!(statistics.projection, GpuPlacement::Gpu);
    assert_eq!(statistics.diagnostic_readback_bytes, 0);
    assert_eq!(
        statistics
            .kernels
            .iter()
            .any(|name| name == "projection_features_F16_F32"),
        half,
        "only actual Half classifier features need widening for the small F32 projection"
    );
    assert!(
        !statistics
            .kernels
            .iter()
            .any(|name| { name == NATIVE_PROJECTION_F16_NAME || name == NATIVE_PROJECTION_NAME })
    );
}

#[test]
#[ignore = "requires physical Metal; explicit Half classifier policy, default F32 and native image execution"]
fn gpu_native_half_classifier_policy_and_public_results() {
    use crate::{
        image::PixelFormat,
        orientation::{Classifier, parse_gpu_classification, should_rotate},
        runtime::CpuCompiledModel,
    };

    let context = context();
    for with_bias in [false, true] {
        let graph = classifier_model(with_bias);
        let cpu = Arc::new(
            CpuCompiledModel::compile(graph.clone(), CompileOptions::reference()).unwrap(),
        );
        let mut reference = Classifier::new(cpu, ExecutionLimits::default()).unwrap();
        for (precision, matrix_precision) in [
            (GpuPrecision::F32, GpuMatrixPrecision::F32),
            (GpuPrecision::MixedF16, GpuMatrixPrecision::F32),
            (GpuPrecision::MixedF16, GpuMatrixPrecision::F16),
        ] {
            let compiled = Arc::new(
                GpuCompiledModel::compile_with_options(
                    graph.clone(),
                    context.clone(),
                    CompileOptions {
                        gpu: GpuCompileOptions {
                            precision,
                            matrix_precision,
                        },
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
            let mut classifier =
                Classification::new(compiled.clone(), ExecutionLimits::default()).unwrap();
            let half = precision == GpuPrecision::MixedF16;
            assert_eq!(classifier.session.compiled.precision, precision);
            assert_eq!(
                classifier.session.compiled.matrix_precision,
                matrix_precision
            );
            assert_eq!(classifier.session.input_precision(), precision);
            assert_eq!(classifier.session.output_precision(), precision);
            assert_eq!(classifier.session.output_shape().dimensions(), &[1, 8]);
            assert!(Arc::ptr_eq(&classifier.session.compiled, &compiled));
            assert!(Arc::ptr_eq(
                &classifier.session.compiled.model_arc(),
                &compiled.model_arc()
            ));
            assert!(Arc::ptr_eq(
                &classifier.session.compiled.weights,
                &compiled.weights
            ));
            let repeated =
                Classification::new(compiled.clone(), ExecutionLimits::default()).unwrap();
            assert!(Arc::ptr_eq(
                &classifier.session.compiled,
                &repeated.session.compiled
            ));
            assert!(classifier.head.half_weights.is_none());
            assert_eq!(classifier.head.weights.bytes(), 8 * 2 * 4);
            let model_weights = classifier.session.compiled.weights.constants
                [classifier.head.metadata.weights.0]
                .as_ref()
                .unwrap();
            assert_eq!(
                Arc::ptr_eq(
                    &classifier.head.weights.reservation,
                    &model_weights.reservation
                ),
                !half,
                "small-class projection must use original F32 weights"
            );

            let mut labels = Vec::new();
            for (index, pixel) in [8, 247].into_iter().enumerate() {
                let pixels = [pixel, pixel, pixel, 255].repeat(80 * 160);
                let image = ImageView::new(&pixels, 160, 80, 0, PixelFormat::Bgra32).unwrap();
                let expected = reference.classify(image).unwrap();
                let oracle = classifier_oracle(pixel, with_bias, half);
                let (bytes, width, statistics) = if index == 0 {
                    let (bytes, width) = classifier.run(image, false).unwrap();
                    (bytes, width, classifier.statistics.clone())
                } else {
                    let mut batch = Batch::new(context.clone()).unwrap();
                    let uploaded = Image::upload(&mut batch, image).unwrap();
                    let (winners, width) = classifier.record(&mut batch, &uploaded).unwrap();
                    let (bytes, statistics) = batch.read(&winners, 16, false).unwrap();
                    (bytes, width, statistics)
                };
                assert_oracle(&bytes, &oracle, true);
                let actual = parse_gpu_classification(&bytes, width).unwrap();
                assert_eq!(actual.label, expected.label);
                assert_eq!(actual.resized_width, expected.resized_width);
                assert_eq!(actual.resized_width, 160);
                assert!(actual.score.is_finite() && actual.score > 0.9 && actual.score <= 1.);
                assert!(
                    (actual.score - expected.score).abs() <= 1e-3,
                    "pixel{pixel}, bias{with_bias}, matrix{matrix_precision:?}: GPU {} vs CPU {}",
                    actual.score,
                    expected.score
                );
                assert_eq!(should_rotate(actual, 0.5), expected.label == 1);
                assert_classifier_native_path(&statistics, half);
                labels.push(actual.label);
            }
            assert_eq!(labels, [1, 0], "both orientation classes must be exercised");
        }
    }
}
