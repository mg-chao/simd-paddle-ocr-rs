use simd_paddle_ocr_rs::{
    detector::{DetectionTrace, Detector, DetectorOptions},
    image::{ImageError, ImageView, PixelFormat},
    lines::{LineOptions, LineTrace, OrientedLineExtractor},
    model::Model,
    ocr::{Ocr, OcrOptions, OcrTrace},
    recognizer::{Dictionary, RecognitionTrace, Recognizer, RecognizerOptions},
    runtime::{CompiledModel, ExecutionLimits},
};
use std::sync::Arc;

fn varint(mut value: u64) -> Vec<u8> {
    let mut result = Vec::new();
    loop {
        let b = (value & 127) as u8;
        value >>= 7;
        result.push(b | if value == 0 { 0 } else { 128 });
        if value == 0 {
            return result;
        }
    }
}
fn number(field: u64, value: i64) -> Vec<u8> {
    [varint(field << 3), varint(value as u64)].concat()
}
fn bytes(field: u64, value: Vec<u8>) -> Vec<u8> {
    [varint(field << 3 | 2), varint(value.len() as u64), value].concat()
}
fn string(field: u64, value: &str) -> Vec<u8> {
    bytes(field, value.as_bytes().to_vec())
}
fn info(name: &str, dims: &[i64]) -> Vec<u8> {
    [
        string(1, name),
        bytes(
            2,
            bytes(
                1,
                [
                    number(1, 1),
                    bytes(
                        2,
                        dims.iter()
                            .flat_map(|&d| {
                                bytes(
                                    1,
                                    if d < 0 {
                                        string(2, "dynamic")
                                    } else {
                                        number(1, d)
                                    },
                                )
                            })
                            .collect(),
                    ),
                ]
                .concat(),
            ),
        ),
    ]
    .concat()
}
fn tensor(name: &str, dims: &[i64], values: &[f32]) -> Vec<u8> {
    bytes(
        5,
        [
            dims.iter().flat_map(|&d| number(1, d)).collect(),
            number(2, 1),
            string(8, name),
            bytes(9, values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ]
        .concat(),
    )
}
fn int(name: &str, value: i64) -> Vec<u8> {
    [string(1, name), number(20, 2), number(3, value)].concat()
}
fn ints(name: &str, values: &[i64]) -> Vec<u8> {
    [
        string(1, name),
        number(20, 7),
        values.iter().flat_map(|&v| number(8, v)).collect(),
    ]
    .concat()
}
fn node(op: &str, inputs: &[&str], output: &str, attrs: Vec<Vec<u8>>) -> Vec<u8> {
    bytes(
        1,
        [
            inputs.iter().flat_map(|v| string(1, v)).collect(),
            string(2, output),
            string(3, output),
            string(4, op),
            attrs.into_iter().flat_map(|a| bytes(5, a)).collect(),
        ]
        .concat(),
    )
}
fn compile(records: Vec<Vec<u8>>, input: &[i64], output: &[i64]) -> Arc<CompiledModel> {
    let graph = [
        records.concat(),
        bytes(11, info("x", input)),
        bytes(12, info("y", output)),
    ]
    .concat();
    let bytes = [number(1, 8), bytes(8, number(2, 7)), bytes(7, graph)].concat();
    Arc::new(CompiledModel::new(Arc::new(Model::load_bytes(&bytes).unwrap())).unwrap())
}
fn engine(classify: bool, blank_rec: bool, options: OcrOptions) -> Ocr {
    let det = compile(
        vec![
            tensor("w", &[1, 3, 1, 1], &[1., 0., 0.]),
            node(
                "Conv",
                &["x", "w"],
                "map",
                vec![ints("kernel_shape", &[1, 1])],
            ),
            node("Sigmoid", &["map"], "y", vec![]),
        ],
        &[1, 3, -1, -1],
        &[1, 1, -1, -1],
    );
    let classifier = classify.then(|| {
        compile(
            vec![
                tensor("w", &[3, 2], &[0.; 6]),
                tensor("b", &[2], &[0., 2.]),
                node(
                    "ReduceMean",
                    &["x"],
                    "features",
                    vec![ints("axes", &[2, 3]), int("keepdims", 0)],
                ),
                node("MatMul", &["features", "w"], "logits", vec![]),
                node("Add", &["logits", "b"], "biased", vec![]),
                node("Softmax", &["biased"], "y", vec![int("axis", 1)]),
            ],
            &[1, 3, 80, 160],
            &[1, 2],
        )
    });
    let bias = if blank_rec {
        [3., 0., 0., 0.]
    } else {
        [0., 2., 2., 0.]
    };
    let rec = compile(
        vec![
            tensor("w", &[3, 4], &[0.; 12]),
            tensor("b", &[4], &bias),
            node(
                "ReduceMean",
                &["x"],
                "mean",
                vec![ints("axes", &[2]), int("keepdims", 0)],
            ),
            node(
                "Transpose",
                &["mean"],
                "features",
                vec![ints("perm", &[0, 2, 1])],
            ),
            node("MatMul", &["features", "w"], "logits", vec![]),
            node("Add", &["logits", "b"], "biased", vec![]),
            node("Softmax", &["biased"], "y", vec![int("axis", 2)]),
        ],
        &[1, 3, 48, -1],
        &[1, -1, 4],
    );
    let limits = ExecutionLimits::default();
    let det = Detector::new(det, DetectorOptions::default(), limits).unwrap();
    let extractor = OrientedLineExtractor::new(det, classifier, LineOptions::default()).unwrap();
    let rec = Recognizer::new(
        rec,
        Arc::new(Dictionary::from_utf8("你好\n👩‍🔬".as_bytes()).unwrap()),
        RecognizerOptions::default(),
        limits,
    )
    .unwrap();
    Ocr::new(extractor, rec, options).unwrap()
}
fn page() -> Vec<u8> {
    let mut result = vec![0; 128 * 96 * 3];
    for (x0, x1, y0, y1) in [(10, 110, 10, 25), (10, 70, 60, 85)] {
        for y in y0..y1 {
            for x in x0..x1 {
                result[(y * 128 + x) * 3..(y * 128 + x) * 3 + 3].fill(255);
            }
        }
    }
    result
}
fn image(data: &[u8]) -> ImageView<'_> {
    ImageView::new(data, 128, 96, 0, PixelFormat::Bgr24).unwrap()
}

#[test]
fn ordered_lines_joined_text_alignment_and_owned_views_survive_empty_reuse() {
    let data = page();
    let mut ocr = engine(false, false, OcrOptions::default());
    let result = ocr.run(image(&data), true).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result.detected_count, 2);
    assert_eq!(result.text, "你好\n你好");
    assert_eq!(
        (
            result.detector_resized_width,
            result.detector_resized_height
        ),
        (128, 96)
    );
    let first = result.get(0).unwrap();
    let second = result.get(1).unwrap();
    assert!(first.detection.points[0][1] < second.detection.points[0][1]);
    for line in result.iter() {
        assert_eq!(
            (
                line.classification_label,
                line.classification_score,
                line.applied_rotation_degrees
            ),
            (0, 0., 0)
        );
        assert_eq!(line.recognition_score, 2.);
        assert_eq!(line.emitted_count, 1);
        let span = line.ctc_spans.unwrap().get(0).unwrap();
        assert_eq!(span.text, "你好");
        assert_eq!(
            (span.start_column, span.end_column),
            (0, line.recognition_time_steps)
        );
    }
    let owned = result.try_to_owned().unwrap();
    assert_eq!(owned.as_view().try_to_owned().unwrap(), owned);
    let blank = vec![0; data.len()];
    let result = ocr.run(image(&blank), true).unwrap();
    assert!(result.is_empty());
    assert_eq!(result.detected_count, 0);
    assert_eq!(result.text, "");
    assert_eq!(
        (
            result.detector_resized_width,
            result.detector_resized_height
        ),
        (128, 96)
    );
    assert_eq!(owned.text, "你好\n你好");
    let result = ocr.run(image(&data), false).unwrap();
    assert!(result.iter().all(|line| line.ctc_spans.is_none()));
    assert_eq!(result.text, owned.text);
}
#[test]
fn classified_rotations_and_blank_recognition_are_preserved() {
    let data = page();
    let mut classified = engine(true, false, OcrOptions::default());
    let result = classified.run(image(&data), true).unwrap();
    assert_eq!(result.len(), 2);
    for line in result.iter() {
        assert_eq!(line.classification_label, 1);
        assert!(line.classification_score > 0.8);
        assert_eq!(line.applied_rotation_degrees, 180);
    }
    let mut blank = engine(false, true, OcrOptions::default());
    let result = blank.run(image(&data), true).unwrap();
    assert_eq!(result.detected_count, 2);
    assert_eq!(result.len(), 2);
    assert_eq!(result.text, "\n");
    for line in result.iter() {
        assert_eq!(line.text, "");
        assert_eq!(line.recognition_score, 0.);
        assert_eq!(line.emitted_count, 0);
        assert!(line.ctc_spans.unwrap().is_empty());
    }
}
fn event_kind(event: &OcrTrace<'_>) -> usize {
    match event {
        OcrTrace::Detection(DetectionTrace::Input { .. }) => 0,
        OcrTrace::Detection(DetectionTrace::Node(_)) => 1,
        OcrTrace::Detection(DetectionTrace::Boxes(_)) => 2,
        OcrTrace::Line(LineTrace::Crop { .. }) => 3,
        OcrTrace::Line(LineTrace::ClsInput { .. }) => 4,
        OcrTrace::Line(LineTrace::ClsNode { .. }) => 5,
        OcrTrace::Line(LineTrace::Classification { .. }) => 6,
        OcrTrace::Line(LineTrace::OrientedCrop { .. }) => 7,
        OcrTrace::Recognition {
            event: RecognitionTrace::Input { .. },
            ..
        } => 8,
        OcrTrace::Recognition {
            event: RecognitionTrace::Node(_),
            ..
        } => 9,
        OcrTrace::Recognition {
            event: RecognitionTrace::Probabilities { .. },
            ..
        } => 10,
        OcrTrace::Recognition {
            event: RecognitionTrace::DecodedDense(_),
            ..
        } => 11,
        OcrTrace::Recognition {
            event: RecognitionTrace::Public(_),
            ..
        } => 12,
        OcrTrace::Result(_) => 13,
        _ => 14,
    }
}
#[test]
fn every_trace_stage_error_and_total_budget_failure_can_recover() {
    let data = page();
    let mut ocr = engine(true, false, OcrOptions::default());
    let expected = ocr.run(image(&data), true).unwrap().try_to_owned().unwrap();
    for kind in 0..15 {
        let error = ocr
            .run_with_trace(image(&data), true, |event| {
                if event_kind(&event) == kind {
                    Err(ImageError::new("injected OCR trace error"))
                } else {
                    Ok(())
                }
            })
            .err()
            .unwrap();
        assert!(error.to_string().contains("injected"));
        assert_eq!(
            ocr.run(image(&data), true).unwrap().try_to_owned().unwrap(),
            expected
        );
    }
    let mut limited = engine(
        false,
        false,
        OcrOptions {
            max_total_result_bytes: 1,
        },
    );
    let error = limited.run(image(&data), true).err().unwrap();
    assert!(error.to_string().contains("max_total_result_bytes"));
    let blank = vec![0; data.len()];
    assert!(limited.run(image(&blank), false).unwrap().is_empty());
}
