//! Independent fixtures exported by the pinned C# character-box implementation.
use serde_json::Value;
use simd_paddle_ocr_rs::{
    characters,
    detector::DetectionBox,
    ocr::{OwnedOcr, OwnedOcrLine},
    recognizer::{OwnedCtcSpan, OwnedRecognition},
};

fn number(value: &Value) -> usize {
    value.as_u64().unwrap() as usize
}

#[test]
fn packed_text_and_hash_match_independent_unicode_and_nul_contracts() {
    let oracle: Value = serde_json::from_str(include_str!("data/character-oracle.json")).unwrap();
    let cases = oracle["packed_text_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let lines: Vec<_> = case["lines"]
            .as_array()
            .unwrap()
            .iter()
            .map(|text| OwnedOcrLine {
                detection: DetectionBox {
                    points: [[0.; 2]; 4],
                    score: 1.,
                },
                text: text.as_str().unwrap().to_owned(),
                recognition_score: 0.75,
                classification_score: 1.,
                classification_label: 0,
                applied_rotation_degrees: 0,
                emitted_count: 0,
                recognition_content_width: 0,
                recognition_tensor_width: 0,
                recognition_time_steps: 0,
                ctc_spans: None,
            })
            .collect();
        let result = OwnedOcr {
            text: case["text"].as_str().unwrap().to_owned(),
            detected_count: lines.len(),
            lines,
            detector_resized_width: 128,
            detector_resized_height: 64,
            width_ratio: 1.,
            height_ratio: 1.,
        };
        let packed = result.packed_text().unwrap();
        assert_eq!(packed, case["packed_text"].as_str().unwrap(), "{id}");
        let hex: String = packed.bytes().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(hex, case["packed_utf8_hex"].as_str().unwrap(), "{id}");
        assert_eq!(
            result.packed_text_hash(),
            case["packed_text_hash"].as_u64().unwrap(),
            "{id}"
        );
        assert_eq!(result.as_view().packed_text().unwrap(), packed, "{id}");
        assert_eq!(
            result.as_view().packed_text_hash(),
            result.packed_text_hash(),
            "{id}"
        );
    }
}

#[test]
fn character_geometry_and_validation_match_the_independent_csharp_oracle() {
    let oracle: Value = serde_json::from_str(include_str!("data/character-oracle.json")).unwrap();
    let lock: Value = serde_json::from_str(include_str!("../tools/reference-lock.json")).unwrap();
    assert_eq!(oracle["reference_revision"], lock["revision"]);
    assert_eq!(oracle["schema_version"], 1);
    assert_eq!(
        oracle["reference_assembly_sha256"].as_str().unwrap().len(),
        64
    );
    let cases = oracle["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 18);
    let mut successes = 0;
    let mut errors = 0;
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let detection = DetectionBox {
            points: std::array::from_fn(|point| {
                std::array::from_fn(|axis| {
                    case["detection"]["points"][point][axis].as_f64().unwrap() as f32
                })
            }),
            score: case["detection"]["score"].as_f64().unwrap() as f32,
        };
        let spans = case["spans"].as_array().map(|values| {
            values
                .iter()
                .map(|span| OwnedCtcSpan {
                    text: span["text"].as_str().unwrap().to_owned(),
                    score: span["score"].as_f64().unwrap() as f32,
                    start_column: number(&span["start_column"]),
                    end_column: number(&span["end_column"]),
                })
                .collect::<Vec<_>>()
        });
        let recognition = OwnedRecognition {
            text: spans.as_ref().map_or_else(String::new, |values| {
                values.iter().map(|span| span.text.as_str()).collect()
            }),
            score: 1.,
            emitted_count: spans.as_ref().map_or(0, Vec::len),
            resized_width: number(&case["content_width"]),
            tensor_width: number(&case["tensor_width"]),
            time_steps: number(&case["time_steps"]),
            ctc_spans: spans,
        };
        let rotation = number(&case["applied_rotation_degrees"]) as u16;
        let actual = characters::estimate_recognition(&detection, recognition.as_view(), rotation);
        if !case["error"].is_null() {
            assert!(actual.is_err(), "{id}: expected {}", case["error"]);
            errors += 1;
            continue;
        }
        let actual = actual.unwrap_or_else(|error| panic!("{id}: {error}"));
        let expected = case["expected"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len(), "{id}");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.text,
                expected["text"].as_str().unwrap(),
                "{id}/{index}"
            );
            assert_eq!(
                actual.score.to_bits(),
                (expected["score"].as_f64().unwrap() as f32).to_bits(),
                "{id}/{index}"
            );
            for point in 0..4 {
                for axis in 0..2 {
                    let expected = expected["points"][point][axis].as_f64().unwrap() as f32;
                    assert!(
                        (actual.points[point][axis] - expected).abs() <= 0.00001,
                        "{id}/{index} point {point} axis {axis}: {} != {expected}",
                        actual.points[point][axis]
                    );
                }
            }
            let owned = actual.try_to_owned().unwrap();
            assert_eq!(owned.as_view(), *actual, "{id}/{index}: owned roundtrip");
        }
        let line = OwnedOcrLine {
            detection,
            text: recognition.text.clone(),
            recognition_score: recognition.score,
            classification_score: 1.,
            classification_label: u32::from(rotation == 180),
            applied_rotation_degrees: rotation,
            emitted_count: recognition.emitted_count,
            recognition_content_width: recognition.resized_width,
            recognition_tensor_width: recognition.tensor_width,
            recognition_time_steps: recognition.time_steps,
            ctc_spans: recognition.ctc_spans.clone(),
        };
        assert_eq!(
            line.estimate_character_boxes().unwrap(),
            actual,
            "{id}: owned line"
        );
        assert_eq!(
            line.as_view().estimate_character_boxes().unwrap(),
            actual,
            "{id}: borrowed line"
        );
        successes += 1;
    }
    assert_eq!((successes, errors), (9, 9));
}
