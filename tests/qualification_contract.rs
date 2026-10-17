#[path = "../tools/baseline/mod.rs"]
mod baseline;
#[path = "../tools/qualification/mod.rs"]
mod qualification;
use baseline::{CaseResult, Frame, Line, Manifest};
use qualification::*;
use std::{collections::BTreeSet, path::Path};

fn profile() -> Profile {
    profiles(&["sample".into()])
        .into_iter()
        .find(|p| p.id == "alignment")
        .unwrap()
}
fn input() -> Input {
    Input {
        id: "sample".into(),
        fixture: "sample".into(),
        path: "unused".into(),
        width: 10,
        height: 10,
        stride: 30,
        format: "Bgr24".into(),
        expected_error: None,
    }
}
fn sample() -> Response {
    Response {
        id: "sample".into(),
        result: Some(CaseResult {
            id: "sample".into(),
            text: "A".into(),
            detected_count: 1,
            detector_resized_width: 32,
            detector_resized_height: 32,
            lines: vec![Line {
                points: [[0., 0.], [10., 0.], [10., 10.], [0., 10.]],
                detection_score: 0.9,
                text: "A".into(),
                recognition_score: 0.9,
                classification_score: 0.8,
                classification_label: 0,
                applied_rotation_degrees: 0,
                emitted_count: 1,
                recognition_content_width: 48,
                recognition_tensor_width: 64,
                recognition_time_steps: 16,
            }],
        }),
        alignments: vec![Alignment {
            spans: Some(vec![Span {
                text: "A".into(),
                score: 0.9,
                start_column: 2,
                end_column: 4,
            }]),
            character_boxes: Some(vec![CharacterBox {
                text: "A".into(),
                score: 0.9,
                points: [[0., 0.], [10., 0.], [10., 10.], [0., 10.]],
            }]),
        }],
        error: None,
    }
}
#[test]
fn matrix_preserves_full_parallel_corpus_and_bounded_independent_profiles() {
    let manifest: Manifest = baseline::read_json(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/baseline/manifest.json"),
    )
    .unwrap();
    let ids: Vec<_> = manifest.fixtures.iter().map(|f| f.id.clone()).collect();
    let plan = profiles(&ids);
    assert_eq!(plan.len(), 16);
    assert_eq!(plan[0].id, "default-parallel");
    assert_eq!(plan[0].fixtures, ids);
    assert_eq!(plan.iter().map(|p| p.fixtures.len()).sum::<usize>(), 71);
    assert_eq!(
        plan.iter().map(|p| &p.id).collect::<BTreeSet<_>>().len(),
        plan.len()
    );
    for p in &plan {
        assert!(p.fixtures.iter().all(|f| ids.contains(f)));
    }
    assert!(plan.iter().find(|p| p.id == "alignment").unwrap().capture);
    for (id, capacity) in [("serial", 1), ("workers-2", 2), ("workers-4", 4)] {
        let p = plan.iter().find(|p| p.id == id).unwrap();
        assert_eq!(
            (
                p.capacity,
                p.line_workers,
                p.preprocess_workers,
                p.detector_threads
            ),
            (capacity, capacity, capacity, capacity)
        );
    }
    let reuse = plan.iter().find(|p| p.id == "reuse").unwrap();
    assert_eq!(reuse.fixtures.first(), reuse.fixtures.last());
    let recovery = plan.iter().find(|p| p.id == "recovery").unwrap();
    assert!(720 * 1150 > recovery.max_image_pixels);
    assert!(896 * 528 < recovery.max_image_pixels);
}
#[test]
fn layouts_preserve_color_and_poison_stride_padding_and_alpha() {
    let frame = Frame {
        id: "sample".into(),
        width: 2,
        height: 2,
        pixels: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    };
    for (format, bpp, first) in [
        ("Bgr24", 3, vec![1, 2, 3]),
        ("Rgb24", 3, vec![3, 2, 1]),
        ("Bgra32", 4, vec![1, 2, 3]),
        ("Rgba32", 4, vec![3, 2, 1]),
    ] {
        let mut p = profile();
        p.format = format.into();
        p.padding = 13;
        let (bytes, stride) = pixels(&frame, &p).unwrap();
        assert_eq!(stride, 2 * bpp + 13);
        assert_eq!(bytes.len(), stride * 2);
        assert_eq!(&bytes[..3], first);
        assert!(bytes[2 * bpp..stride].iter().all(|b| *b == 0xa5));
        assert_eq!(
            &bytes[stride..stride + 3],
            if format.starts_with("Rgb") {
                &[9, 8, 7]
            } else {
                &[7, 8, 9]
            }
        );
        if bpp == 4 {
            assert_eq!(bytes[3], 0);
            assert_eq!(bytes[stride + 3], 31);
        }
    }
}
#[test]
fn alignment_contract_rejects_changed_columns_tokens_missing_capture_and_nonfinite() {
    let expected = sample();
    let input = input();
    let profile = profile();
    assert!(compare("tiny", &expected, &expected, &input, &profile).is_empty());
    let mut actual = expected.clone();
    actual.alignments[0].spans.as_mut().unwrap()[0].start_column += 1;
    assert!(
        compare("tiny", &expected, &actual, &input, &profile)
            .iter()
            .any(|e| e.contains("time columns"))
    );
    let mut actual = expected.clone();
    actual.alignments[0].spans = None;
    assert!(validate(&actual, &input, &profile).is_err());
    let mut actual = expected.clone();
    actual.alignments[0].spans.as_mut().unwrap()[0].end_column = 17;
    assert!(validate(&actual, &input, &profile).is_err());
    let mut actual = expected.clone();
    actual.alignments[0].character_boxes.as_mut().unwrap()[0].text = "B".into();
    assert!(validate(&actual, &input, &profile).is_err());
    let mut actual = expected.clone();
    actual.alignments[0].character_boxes.as_mut().unwrap()[0].points[0][0] = f64::NAN;
    assert!(validate(&actual, &input, &profile).is_err());
    let mut off = profile;
    off.capture = false;
    assert!(validate(&expected, &input, &off).is_err());
    let mut absent = expected;
    absent.alignments[0] = Alignment {
        spans: None,
        character_boxes: None,
    };
    validate(&absent, &input, &off).unwrap();
}
#[test]
fn character_and_span_tolerances_are_exactly_the_canonical_contract() {
    let mut expected = sample();
    expected.alignments[0].spans.as_mut().unwrap()[0].score = 0.;
    let mut actual = expected.clone();
    actual.alignments[0].spans.as_mut().unwrap()[0].score = baseline::SCORE_ABSOLUTE;
    actual.alignments[0].character_boxes.as_mut().unwrap()[0].points[0][0] =
        baseline::COORDINATE_TOLERANCE;
    assert!(compare("tiny", &expected, &actual, &input(), &profile()).is_empty());
    actual.alignments[0].spans.as_mut().unwrap()[0].score += 1e-9;
    assert!(
        compare("tiny", &expected, &actual, &input(), &profile())
            .iter()
            .any(|e| e.contains("span[0].score"))
    );
    actual = expected.clone();
    actual.alignments[0].character_boxes.as_mut().unwrap()[0].points[0][0] =
        baseline::COORDINATE_TOLERANCE + 1e-9;
    assert!(
        compare("tiny", &expected, &actual, &input(), &profile())
            .iter()
            .any(|e| e.contains("character[0].points"))
    );
}
#[test]
fn recovery_requires_the_expected_rejection_and_a_real_success_afterward() {
    let mut input = input();
    input.expected_error = Some("invalid_stride".into());
    let error = Response {
        id: input.id.clone(),
        result: None,
        alignments: vec![],
        error: Some("Source stride is too small".into()),
    };
    validate(&error, &input, &profile()).unwrap();
    assert!(validate(&sample(), &input, &profile()).is_err());
    let mut wrong = error.clone();
    wrong.error = Some("file missing".into());
    assert!(validate(&wrong, &input, &profile()).is_err());
    input.expected_error = Some("pixel_limit".into());
    wrong.error = Some("Source image exceeds MaxImagePixels".into());
    validate(&wrong, &input, &profile()).unwrap();
    input.expected_error = None;
    assert!(validate(&wrong, &input, &profile()).is_err());
    validate(&sample(), &input, &profile()).unwrap();
}
