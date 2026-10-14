//! Stateful public pipeline contracts against the prepared native mini corpus.
use serde_json::Value;
use simd_paddle_ocr_rs::{
    detector::{DetectionTrace, Detector, DetectorOptions},
    image::ImageError,
    lines::{LineOptions, LineTrace, OrientedLineExtractor, OwnedOrientedLines},
    model::Model,
    runtime::{CompiledModel, ExecutionLimits},
    step12::{FrozenImages, load_images},
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
fn fixture_root() -> PathBuf {
    std::env::var_os("SIMDOCR_STEP12_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/step12-fixtures"))
}
fn compiled(path: &Path) -> Arc<CompiledModel> {
    Arc::new(CompiledModel::new(Arc::new(Model::load_path(path).unwrap())).unwrap())
}
fn detector(model: &Arc<CompiledModel>) -> Detector {
    Detector::new(
        model.clone(),
        DetectorOptions::default(),
        ExecutionLimits::default(),
    )
    .unwrap()
}
fn extractor(
    det: &Arc<CompiledModel>,
    cls: Option<Arc<CompiledModel>>,
    options: LineOptions,
) -> OrientedLineExtractor {
    OrientedLineExtractor::new(detector(det), cls, options).unwrap()
}
fn summary(root: &Path, case_id: &str) -> Value {
    let manifest: Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let case = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == case_id)
        .unwrap();
    let artifact = case["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "step12.public")
        .unwrap();
    serde_json::from_slice(&fs::read(root.join(artifact["path"].as_str().unwrap())).unwrap())
        .unwrap()
}
fn assert_snapshot(
    output: &simd_paddle_ocr_rs::lines::OrientedLines<'_>,
    expected: &OwnedOrientedLines,
) {
    assert_eq!(output.pixels, expected.pixels);
    assert_eq!(
        serde_json::to_value(output.lines).unwrap(),
        serde_json::to_value(&expected.lines).unwrap()
    );
    assert_eq!(
        (output.resized_width, output.resized_height),
        (expected.resized_width, expected.resized_height)
    );
}
#[test]
#[ignore = "requires tools/prepare-step12-fixtures.py; CI runs this stateful public API gate"]
fn real_prefix_full_public_outputs_and_owned_copies_survive_reuse() {
    let root = fixture_root();
    let mut total_lines = 0;
    for variant in ["tiny", "small", "medium"] {
        let path = root.join(variant);
        let corpus = load_images(&path).unwrap();
        let det = compiled(&corpus.detector_model_path);
        let cls = compiled(&corpus.classifier_model_path);
        let mut engine = extractor(&det, Some(cls.clone()), LineOptions::default());
        let expected_threshold = if variant == "tiny" { 0.4 } else { 0.45 };
        assert_eq!(
            engine.detector_mut().resolved_options().box_threshold,
            Some(expected_threshold)
        );
        let mut snapshots = Vec::new();
        for image in &corpus.images {
            let expected = engine
                .extract(image.image().unwrap())
                .unwrap()
                .try_to_owned()
                .unwrap();
            let native = summary(&path, &image.case_id);
            assert_eq!(
                expected.lines.len(),
                native["detected_count"].as_u64().unwrap() as usize
            );
            for (line, native) in expected
                .lines
                .iter()
                .zip(native["lines"].as_array().unwrap())
            {
                let cls = line.classification.unwrap();
                assert_eq!(
                    cls.label,
                    native["classification_label"].as_u64().unwrap() as u32
                );
                assert!(
                    (cls.score - native["classification_score"].as_f64().unwrap() as f32).abs()
                        <= 1e-5
                );
                assert_eq!(
                    line.applied_rotation_degrees,
                    native["applied_rotation_degrees"].as_u64().unwrap() as u16
                );
            }
            let (mut nodes, mut public, mut probabilities) = (0, 0, 0);
            let output = engine
                .extract_with_trace(
                    image.image().unwrap(),
                    |_| Ok(()),
                    |event| {
                        match event {
                            LineTrace::ClsNode { .. } => nodes += 1,
                            LineTrace::Classification { .. } => public += 1,
                            LineTrace::ClsProbabilities { .. } => probabilities += 1,
                            _ => {}
                        }
                        Ok(())
                    },
                )
                .unwrap();
            assert_snapshot(&output, &expected);
            assert_eq!(nodes, expected.lines.len() * cls.node_count());
            assert_eq!(public, expected.lines.len());
            assert_eq!(probabilities, expected.lines.len());
            assert!(expected.as_view().get(expected.lines.len()).is_none());
            assert_eq!(expected.as_view().iter().count(), expected.lines.len());
            total_lines += expected.lines.len();
            snapshots.push(expected);
        }
        for (image, expected) in corpus.images.iter().zip(&snapshots).rev() {
            assert_snapshot(&engine.extract(image.image().unwrap()).unwrap(), expected);
        }
        for (image, expected) in corpus.images.iter().zip(&snapshots) {
            assert_snapshot(&engine.extract(image.image().unwrap()).unwrap(), expected);
        }
    }
    assert!(
        total_lines > 0,
        "prepared mini corpus must exercise real crop and CLS execution"
    );
}
fn nonempty_medium() -> (
    FrozenImages,
    usize,
    usize,
    Arc<CompiledModel>,
    Arc<CompiledModel>,
) {
    let path = fixture_root().join("medium");
    let corpus = load_images(&path).unwrap();
    let full = corpus
        .images
        .iter()
        .position(|i| {
            summary(&path, &i.case_id)["detected_count"]
                .as_u64()
                .unwrap()
                > 0
        })
        .expect("medium mini corpus needs detected line");
    let empty = corpus
        .images
        .iter()
        .position(|i| summary(&path, &i.case_id)["detected_count"] == 0)
        .expect("medium mini corpus needs empty page");
    let det = compiled(&corpus.detector_model_path);
    let cls = compiled(&corpus.classifier_model_path);
    (corpus, full, empty, det, cls)
}
#[test]
#[ignore = "requires tools/prepare-step12-fixtures.py; CI runs this stateful public API gate"]
fn observer_failures_recover_and_disabled_classifier_returns_original_crops() {
    let (corpus, index, _, det, cls) = nonempty_medium();
    let image = corpus.images[index].image().unwrap();
    let mut engine = extractor(&det, Some(cls), LineOptions::default());
    let expected = engine.extract(image).unwrap().try_to_owned().unwrap();
    for fail_at in 0..4 {
        let error = engine
            .extract_with_trace(
                image,
                |event| {
                    let fail = match event {
                        DetectionTrace::Input { .. } => fail_at == 0,
                        DetectionTrace::Node(node) => fail_at == 1 && node.index == 0,
                        DetectionTrace::Probabilities { .. } => fail_at == 2,
                        DetectionTrace::Boxes(_) => fail_at == 3,
                    };
                    if fail {
                        Err(ImageError::new("injected detector observer failure"))
                    } else {
                        Ok(())
                    }
                },
                |_| Ok(()),
            )
            .err()
            .expect("observer must fail");
        assert!(error.to_string().contains("injected detector"));
        assert_snapshot(&engine.extract(image).unwrap(), &expected);
    }
    for fail_at in 0..6 {
        let error = engine
            .extract_with_trace(
                image,
                |_| Ok(()),
                |event| {
                    let fail = match event {
                        LineTrace::Crop { .. } => fail_at == 0,
                        LineTrace::ClsInput { .. } => fail_at == 1,
                        LineTrace::ClsNode { node, .. } => fail_at == 2 && node.index == 0,
                        LineTrace::ClsProbabilities { .. } => fail_at == 3,
                        LineTrace::Classification { .. } => fail_at == 4,
                        LineTrace::OrientedCrop { .. } => fail_at == 5,
                    };
                    if fail {
                        Err(ImageError::new("injected line observer failure"))
                    } else {
                        Ok(())
                    }
                },
            )
            .err()
            .expect("observer must fail");
        assert!(error.to_string().contains("injected line"));
        assert_snapshot(&engine.extract(image).unwrap(), &expected);
    }
    let mut raw = Vec::new();
    engine
        .extract_with_trace(
            image,
            |_| Ok(()),
            |event| {
                if let LineTrace::Crop { image, .. } = event {
                    raw.push(image.pixels().to_vec());
                }
                Ok(())
            },
        )
        .unwrap();
    let mut disabled = extractor(
        &det,
        None,
        LineOptions {
            max_classifier_image_pixels: 1,
            ..Default::default()
        },
    );
    let output = disabled.extract(image).unwrap();
    assert_eq!(output.lines.len(), expected.lines.len());
    for (line, raw) in output.iter().zip(raw) {
        assert_eq!(line.image.pixels(), raw);
        assert!(line.metadata.classification.is_none());
        assert_eq!(line.metadata.applied_rotation_degrees, 0);
    }
    assert_eq!(disabled.classifier_activation_bytes(), 0);
}
#[test]
#[ignore = "requires tools/prepare-step12-fixtures.py; CI runs this stateful public API gate"]
fn crop_budget_failures_precede_slab_writes_and_empty_requests_can_recover() {
    let (corpus, index, empty, det, cls) = nonempty_medium();
    let image = corpus.images[index].image().unwrap();
    for (options, message) in [
        (
            LineOptions {
                max_crop_pixels: 1,
                ..Default::default()
            },
            "max_crop_pixels",
        ),
        (
            LineOptions {
                max_total_crop_bytes: 1,
                ..Default::default()
            },
            "max_total_crop_bytes",
        ),
        (
            LineOptions {
                max_classifier_image_pixels: 1,
                ..Default::default()
            },
            "max_classifier_image_pixels",
        ),
    ] {
        let mut engine = extractor(&det, Some(cls.clone()), options);
        let error = engine.extract(image).err().expect("budget must fail");
        assert!(error.to_string().contains(message));
        assert_eq!(
            engine.crop_capacity_bytes(),
            0,
            "failed preflight must not allocate crop slab"
        );
        let output = engine
            .extract(corpus.images[empty].image().unwrap())
            .unwrap();
        assert!(output.lines.is_empty());
        assert!(output.pixels.is_empty());
    }
    assert!(
        OrientedLineExtractor::new(
            detector(&det),
            Some(cls),
            LineOptions {
                classifier_execution_limits: ExecutionLimits {
                    max_activation_bytes: 0
                },
                ..Default::default()
            }
        )
        .is_err()
    );
    for options in [
        LineOptions {
            classifier_threshold: f32::NAN,
            ..Default::default()
        },
        LineOptions {
            max_crop_pixels: 0,
            ..Default::default()
        },
        LineOptions {
            max_total_crop_bytes: 0,
            ..Default::default()
        },
    ] {
        assert!(OrientedLineExtractor::new(detector(&det), None, options).is_err());
    }
}
