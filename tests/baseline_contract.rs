#[path = "../tools/baseline/mod.rs"]
mod baseline;
use baseline::*;
use std::{fs, path::PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/baseline")
}
fn sample() -> CaseResult {
    CaseResult {
        id: "sample".into(),
        text: "中文 OCR".into(),
        detected_count: 1,
        detector_resized_width: 512,
        detector_resized_height: 512,
        lines: vec![Line {
            points: [[0., 0.], [100., 0.], [100., 20.], [0., 20.]],
            detection_score: 0.9,
            text: "中文 OCR".into(),
            recognition_score: 0.9,
            classification_score: 0.9,
            classification_label: 0,
            applied_rotation_degrees: 0,
            emitted_count: 6,
            recognition_content_width: 240,
            recognition_tensor_width: 256,
            recognition_time_steps: 64,
        }],
    }
}
#[test]
fn pinned_corpus_pixels_and_three_goldens_are_complete() {
    let path = root().join("manifest.json");
    let manifest: Manifest = read_json(&path).unwrap();
    manifest.validate().unwrap();
    let frames = decode_frames(&manifest, &root(), true).unwrap();
    assert_eq!(frames.len(), 22);
    let blank = frames.iter().find(|f| f.id == "blank").unwrap();
    assert!(blank.pixels.iter().all(|&b| b == 255));
    let original = frames.iter().find(|f| f.id == "imgs_11").unwrap();
    let quarter = frames.iter().find(|f| f.id == "imgs_11_rotate90").unwrap();
    assert_eq!(
        (quarter.width, quarter.height),
        (original.height, original.width)
    );
    for model in &manifest.models {
        let golden: Golden =
            read_json(&root().join(format!("expected/{}.json", model.id))).unwrap();
        golden
            .validate(&model.id, &manifest, &hash(&fs::read(&path).unwrap()))
            .unwrap();
        let blank = golden.cases.iter().find(|c| c.id == "blank").unwrap();
        assert_eq!(blank.detected_count, 0);
    }
}
#[test]
fn sha256_checks_reject_corruption_and_missing_files() {
    assert_eq!(
        hash(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("target/baseline-contract-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model-or-dictionary");
    fs::write(&path, b"abc").unwrap();
    check_file(&path, &hash(b"abc")).unwrap();
    fs::write(&path, b"abd").unwrap();
    assert!(
        check_file(&path, &hash(b"abc"))
            .unwrap_err()
            .to_string()
            .contains("SHA-256 mismatch")
    );
    fs::remove_file(&path).unwrap();
    assert!(check_file(&path, &hash(b"abc")).is_err());
    fs::remove_dir(&dir).unwrap();
}
#[test]
fn dictionary_checkout_forms_are_individually_pinned_without_normalization() {
    use simd_paddle_ocr_rs::recognizer::Dictionary;
    let lf = "中\nA\n".as_bytes();
    let crlf = "中\r\nA\r\n".as_bytes();
    let asset = Asset {
        file: "ppocr_keys.txt".into(),
        sha256: hash(lf),
        sha256_crlf: Some(hash(crlf)),
    };
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("target/baseline-newlines-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let path = directory.join("ppocr_keys.txt");
    for bytes in [lf, crlf] {
        fs::write(&path, bytes).unwrap();
        asset.check(&path).unwrap();
    }
    // Mixed line endings decode to the same labels, but their unpinned bytes
    // must still be rejected by the provenance check.
    fs::write(&path, "中\r\nA\n".as_bytes()).unwrap();
    assert!(asset.check(&path).is_err());
    let left = Dictionary::from_utf8(lf).unwrap();
    let right = Dictionary::from_utf8(crlf).unwrap();
    assert_eq!(left.class_count(), right.class_count());
    for class in 0..left.class_count() {
        assert_eq!(left.token(class), right.token(class));
    }
    fs::remove_file(path).unwrap();
    fs::remove_dir(directory).unwrap();
}
#[test]
fn corpus_and_bundle_omissions_are_rejected() {
    let manifest: Manifest = read_json(&root().join("manifest.json")).unwrap();
    let mut broken = manifest.clone();
    broken.fixtures.pop();
    assert!(broken.validate().is_err());
    let mut broken = manifest.clone();
    broken.fixtures[1].id = broken.fixtures[0].id.clone();
    assert!(broken.validate().is_err());
    let mut broken = manifest.clone();
    broken.assets.pop();
    assert!(broken.validate().is_err());
    let mut broken = manifest.clone();
    broken.models[1].dictionary = "../unverified.txt".into();
    assert!(broken.validate().is_err());
    let mut broken = manifest.clone();
    broken.fixtures[0].file = Some("../outside.jpg".into());
    assert!(broken.validate().is_err());
    let mut broken = manifest.clone();
    broken.fixtures[0].url = Some("https://example.com/moving.jpg".into());
    assert!(broken.validate().is_err());
    let mut broken = manifest;
    broken.settings.cpu_threads = 2;
    assert!(broken.validate().is_err());
}
#[test]
fn text_order_structure_rotation_and_dimensions_are_exact() {
    let expected = sample();
    assert!(compare("tiny", &expected, &expected).is_empty());
    for change in [
        "text",
        "classification_label",
        "applied_rotation_degrees",
        "emitted_count",
        "recognition_content_width",
        "recognition_tensor_width",
        "recognition_time_steps",
    ] {
        let mut value = serde_json::to_value(&expected).unwrap();
        if change == "text" {
            value["lines"][0][change] = "changed".into();
            value["text"] = "changed".into();
        } else {
            value["lines"][0][change] = (value["lines"][0][change].as_u64().unwrap() + 1).into();
        }
        let actual = serde_json::from_value(value).unwrap();
        assert!(
            compare("tiny", &expected, &actual)
                .iter()
                .any(|e| e.contains(change)),
            "{change}"
        );
    }
    let mut actual = expected.clone();
    actual.lines.clear();
    actual.text.clear();
    actual.detected_count = 0;
    assert!(
        compare("small", &expected, &actual)
            .iter()
            .any(|e| e.contains("lines.length"))
    );
    let mut two = expected.clone();
    let mut second = two.lines[0].clone();
    second.text = "second".into();
    two.lines.push(second);
    two.detected_count = 2;
    two.text = "中文 OCR\nsecond".into();
    let mut actual = two.clone();
    actual.lines.swap(0, 1);
    actual.text = "second\n中文 OCR".into();
    assert!(!compare("medium", &two, &actual).is_empty());
}
#[test]
fn float_tolerances_have_boundaries_and_never_accept_nonfinite_values() {
    let mut expected = sample();
    expected.lines[0].recognition_score = 0.;
    let mut actual = expected.clone();
    actual.lines[0].points[0][0] = 0.5;
    actual.lines[0].recognition_score = 1e-4;
    assert!(compare("tiny", &expected, &actual).is_empty());
    actual.lines[0].points[0][0] = 0.500001;
    assert!(
        compare("tiny", &expected, &actual)
            .iter()
            .any(|e| e.contains("points[0][0]"))
    );
    actual = expected.clone();
    actual.lines[0].recognition_score = 0.000100001;
    assert!(
        compare("tiny", &expected, &actual)
            .iter()
            .any(|e| e.contains("recognition_score"))
    );
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        actual = expected.clone();
        actual.lines[0].detection_score = value;
        assert!(actual.validate().is_err());
        assert!(!compare("tiny", &expected, &actual).is_empty());
    }
}
#[test]
fn malformed_results_and_changed_golden_provenance_are_rejected() {
    let mut value = serde_json::to_value(sample()).unwrap();
    value["unexpected"] = true.into();
    assert!(serde_json::from_value::<CaseResult>(value).is_err());
    let mut value = serde_json::to_value(sample()).unwrap();
    value["lines"][0]["detection_score"] = serde_json::Value::Null;
    assert!(serde_json::from_value::<CaseResult>(value).is_err());
    let mut value = serde_json::to_value(sample()).unwrap();
    value.as_object_mut().unwrap().remove("lines");
    assert!(serde_json::from_value::<CaseResult>(value).is_err());
    let manifest: Manifest = read_json(&root().join("manifest.json")).unwrap();
    let digest = hash(&fs::read(root().join("manifest.json")).unwrap());
    let golden: Golden = read_json(&root().join("expected/tiny.json")).unwrap();
    let mut changed = golden.clone();
    changed.cases.pop();
    assert!(changed.validate("tiny", &manifest, &digest).is_err());
    let mut changed = golden;
    changed.reference_commit = "0".repeat(40);
    assert!(changed.validate("tiny", &manifest, &digest).is_err());
}
