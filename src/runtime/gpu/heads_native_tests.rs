use super::*;

fn native_context() -> Arc<GpuContext> {
    GpuContext::new_with_options(GpuContextOptions::default()).unwrap()
}

fn word(record: &[u8], index: usize) -> u32 {
    u32::from_le_bytes(record[index * 4..index * 4 + 4].try_into().unwrap())
}

fn read_projection(head: &Head, features: &Tensor, probability: bool) -> (Vec<u8>, Vec<u8>) {
    let rows = features.descriptor.shape.dimensions()[0];
    let mut batch = Batch::new(head.context.clone()).unwrap();
    let winners = head.project(&mut batch, features, probability).unwrap();
    assert!(
        batch
            .statistics
            .kernels
            .iter()
            .any(|name| name == NATIVE_PROJECTION_NAME)
    );
    read_winners_and_ctc(head, batch, &winners, rows)
}

fn read_winners_and_ctc(
    head: &Head,
    mut batch: Batch,
    winners: &GpuBuffer,
    rows: usize,
) -> (Vec<u8>, Vec<u8>) {
    let decoded = head
        .decode(&mut batch, winners, 1, rows, true, usize::MAX)
        .unwrap();
    let winner_bytes = rows * 16;
    let decoded_bytes = (rows + 1) * 24;
    let combined = batch
        .storage("native projection contract", winner_bytes + decoded_bytes)
        .unwrap();
    batch.copy(winners, 0, &combined, 0, winner_bytes as u64);
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
    let decoded = actual.split_off(winner_bytes);
    (actual, decoded)
}

fn oracle(
    [rows, inner, classes]: [usize; 3],
    features: &[f32],
    weights: &[f32],
    bias: &[f32],
) -> Vec<(u32, f64, f64)> {
    (0..rows)
        .map(|row| {
            let logits: Vec<f64> = (0..classes)
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
            for (class, &score) in logits.iter().enumerate().skip(1) {
                if score > logits[winner] {
                    winner = class;
                }
            }
            let maximum = logits[winner];
            let denominator: f64 = logits.iter().map(|&value| (value - maximum).exp()).sum();
            (winner as u32, maximum, denominator.recip())
        })
        .collect()
}

fn assert_oracle(actual: &[u8], expected: &[(u32, f64, f64)], probability: bool) {
    for (row, record) in actual.as_chunks::<16>().0.iter().enumerate() {
        assert_eq!(word(record, 0), expected[row].0);
        let reference = if probability {
            expected[row].2
        } else {
            expected[row].1
        };
        let score = f64::from(f32::from_bits(word(record, 1)));
        // Same tolerance as the native dense-tail contract.
        assert!(score.is_finite());
        assert!(
            (score - reference).abs() <= 1e-6 + 2e-5 * reference.abs(),
            "row {row}, probability {probability}: {score} versus {reference}"
        );
        assert_eq!(word(record, 3), 0);
    }
}

fn dense_values(count: usize, salt: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * salt + 7) % modulus) as f32 * 2. + 1. - modulus as f32) / scale)
        .collect()
}

#[test]
#[ignore = "requires a physical Metal tensor GPU; native FP32 dense row/class/K tails"]
fn gpu_native_projection_dense_tails_and_score_modes() {
    let context = native_context();
    for dimensions @ [rows, inner, classes] in [[8, 7, 4097], [32, 16, 4160], [33, 17, 4099]] {
        let feature_values = dense_values(rows * inner, 13, 63, 131.);
        let weight_values = dense_values(inner * classes, 17, 127, 257.);
        let mut bias_values = dense_values(classes, 7, 31, 251.);
        bias_values[classes - 1] = 10.;
        let expected = oracle(dimensions, &feature_values, &weight_values, &bias_values);
        let (head, features) = projection_fixture(
            &context,
            dimensions,
            GpuPrecision::F32,
            [&feature_values, &weight_values, &bias_values],
        );
        assert!(
            head.native_projection,
            "qualification device must support the MPP arithmetic kernel"
        );
        for probability in [false, true] {
            let (actual, decoded) = read_projection(&head, &features, probability);
            assert_oracle(&actual, &expected, probability);
            let decoded = Recognition::parse(&decoded, 1, rows, classes).unwrap();
            assert_eq!(decoded[0].tokens.len(), 1);
            let token = decoded[0].tokens.iter().next().unwrap();
            assert_eq!(
                (token.class, token.start, token.end),
                (classes - 1, 0, rows)
            );
        }
    }
}

#[test]
#[ignore = "requires a physical Metal tensor GPU; native ties and nonfinite boundaries"]
fn gpu_native_projection_ties_and_nonfinite() {
    let [rows, inner, classes] = [33usize, 8, 4097];
    let mut feature_values = vec![1f32; rows * inner];
    for row in 0..rows {
        feature_values[row * inner] = if row % 2 == 0 { 1. } else { -1. };
    }
    let mut weight_values = vec![0f32; inner * classes];
    weight_values[63] = 2.;
    weight_values[64] = 2.;
    weight_values[classes - 1] = -2.;
    let mut bias_values = vec![0f32; classes];
    for class in [63, 64, classes - 1] {
        bias_values[class] = 4.;
    }
    {
        let context = native_context();
        let (head, features) = projection_fixture(
            &context,
            [rows, inner, classes],
            GpuPrecision::F32,
            [&feature_values, &weight_values, &bias_values],
        );
        for probability in [false, true] {
            let (actual, decoded) = read_projection(&head, &features, probability);
            for (row, record) in actual.as_chunks::<16>().0.iter().enumerate() {
                assert_eq!(
                    word(record, 0),
                    if row % 2 == 0 {
                        63
                    } else {
                        (classes - 1) as u32
                    }
                );
                let reference = if probability {
                    let high = if row % 2 == 0 { 2. } else { 1. };
                    let low = 3. - high;
                    1. / (high + low * (-4f32).exp() + (classes - 3) as f32 * (-6f32).exp())
                } else {
                    6.
                };
                assert!(
                    (f32::from_bits(word(record, 1)) - reference).abs()
                        <= 1e-6 + 1e-5 * reference.abs()
                );
                assert_eq!(word(record, 3), 0);
            }
            Recognition::parse(&decoded, 1, rows, classes).unwrap();
            // Exercise interior and last row/class/K coordinates. Flags must
            // survive both local partial reduction and the shared final head.
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                for location in 0..4 {
                    if location < 2 {
                        let mut values = weight_values.clone();
                        values[if location == 0 {
                            65
                        } else {
                            inner * classes - 1
                        }] = invalid;
                        context
                            .upload_resident(&head.weights, bytemuck::cast_slice(&values))
                            .unwrap();
                    } else {
                        let mut values = feature_values.clone();
                        values[if location == 2 { 0 } else { rows * inner - 1 }] = invalid;
                        context
                            .upload_resident(&features.buffer, bytemuck::cast_slice(&values))
                            .unwrap();
                    }
                    let (actual, decoded) = read_projection(&head, &features, probability);
                    for (row, record) in actual.as_chunks::<16>().0.iter().enumerate() {
                        assert_eq!(
                            word(record, 3) != 0,
                            location < 2
                                || (location == 2 && row == 0)
                                || (location == 3 && row == rows - 1),
                            "location {location}, invalid {invalid:?}, row {row}, probability {probability}"
                        );
                    }
                    assert!(Recognition::parse(&decoded, 1, rows, classes).is_err());
                    context
                        .upload_resident(&head.weights, bytemuck::cast_slice(&weight_values))
                        .unwrap();
                    context
                        .upload_resident(&features.buffer, bytemuck::cast_slice(&feature_values))
                        .unwrap();
                }
            }
        }
    }
}

#[test]
#[ignore = "requires physical Metal; forces the GPU scalar execution-width fallback"]
fn gpu_native_projection_scalar_width_fallback() {
    let context = native_context();
    let dimensions @ [rows, inner, classes] = [9, 7, 4097];
    let feature_values = dense_values(rows * inner, 13, 63, 131.);
    let weight_values = dense_values(inner * classes, 17, 127, 257.);
    let mut bias_values = dense_values(classes, 7, 31, 251.);
    bias_values[classes - 1] = 10.;
    let expected = oracle(dimensions, &feature_values, &weight_values, &bias_values);
    let (head, features) = projection_fixture(
        &context,
        dimensions,
        GpuPrecision::F32,
        [&feature_values, &weight_values, &bias_values],
    );
    // This qualification GPU has width 32. Recompile the exact fallback body
    // behind a forced uniform condition, without assuming a nonexistent device.
    let shader = KernelKey::new(KernelId::ProjectionScalarProbe);
    let tiles = classes.div_ceil(64);
    for probability in [false, true] {
        let mut batch = Batch::new(context.clone()).unwrap();
        let partial = batch
            .storage("native fallback partial", rows * tiles * 16)
            .unwrap();
        let winners = batch.storage("native fallback winners", rows * 16).unwrap();
        let words = [
            rows as u32,
            inner as u32,
            classes as u32,
            tiles as u32,
            u32::from(probability),
        ];
        batch
            .dispatch(
                "projection_mpp_scalar_F32",
                shader,
                &[
                    (&features.buffer, true),
                    (&head.weights, true),
                    (&head.bias, true),
                    (&partial, false),
                ],
                &words,
                [tiles as u32, rows.div_ceil(32) as u32, 1],
            )
            .unwrap();
        batch
            .dispatch(
                "projection_winners",
                KernelKey::new(KernelId::HeadReduce),
                &[(&partial, true), (&winners, false)],
                &words,
                [(rows as u32).div_ceil(64), 1, 1],
            )
            .unwrap();
        let (actual, decoded) = read_winners_and_ctc(&head, batch, &winners, rows);
        assert_oracle(&actual, &expected, probability);
        Recognition::parse(&decoded, 1, rows, classes).unwrap();
    }
}
