//! Shipped-model public API contracts; no tracing or dense diagnostic logits.
use serde_json::Value;
use simd_paddle_ocr_rs::{
    detector::{Detector, DetectorOptions},
    full_ocr::public_json,
    image::{ImageView, PixelFormat},
    lines::{LineOptions, OrientedLineExtractor},
    model::Model,
    ocr::{Ocr, OcrOptions},
    recognizer::{Dictionary, OwnedRecognition, Recognizer, RecognizerOptions},
    runtime::{CompiledModel, ExecutionLimits},
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone)]
struct Pixels {
    bytes: Vec<u8>,
    width: usize,
    height: usize,
    stride: usize,
    format: PixelFormat,
}
impl Pixels {
    fn view(&self) -> ImageView<'_> {
        ImageView::new(
            &self.bytes,
            self.width,
            self.height,
            self.stride,
            self.format,
        )
        .unwrap()
    }
    fn reformatted(&self, format: PixelFormat) -> Self {
        assert_eq!(self.format, PixelFormat::Bgr24);
        let bpp = format.bytes_per_pixel();
        let stride = self.width * bpp + 13;
        let mut bytes = vec![239; self.height * stride];
        for y in 0..self.height {
            for x in 0..self.width {
                let from = y * self.stride + x * 3;
                let to = y * stride + x * bpp;
                let p = &self.bytes[from..from + 3];
                if matches!(format, PixelFormat::Rgb24 | PixelFormat::Rgba32) {
                    bytes[to..to + 3].copy_from_slice(&[p[2], p[1], p[0]]);
                } else {
                    bytes[to..to + 3].copy_from_slice(p);
                }
                if bpp == 4 {
                    bytes[to + 3] = (x.wrapping_mul(17) + y.wrapping_mul(31)) as u8;
                }
            }
        }
        Self {
            bytes,
            width: self.width,
            height: self.height,
            stride,
            format,
        }
    }
    // Derive arbitrary widths from already-exported pixels, without a decoder.
    fn width_sample(&self, width: usize) -> Self {
        let mut bytes = Vec::with_capacity(width * 48 * 3);
        for y in 0..48 {
            for x in 0..width {
                let offset = (y % self.height) * self.stride + (x % self.width) * 3;
                bytes.extend_from_slice(&self.bytes[offset..offset + 3]);
            }
        }
        Self {
            bytes,
            width,
            height: 48,
            stride: width * 3,
            format: PixelFormat::Bgr24,
        }
    }
}

struct Models {
    det: Arc<CompiledModel>,
    cls: Arc<CompiledModel>,
    rec: Arc<CompiledModel>,
    dictionary: Arc<Dictionary>,
}
impl Models {
    fn load(root: &Path, manifest: &Value) -> Self {
        let path = |id: &str| {
            let model = manifest["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|model| model["id"] == id)
                .unwrap();
            root.join(model["path"].as_str().unwrap())
        };
        let compile = |id| {
            Arc::new(CompiledModel::new(Arc::new(Model::load_path(path(id)).unwrap())).unwrap())
        };
        Self {
            det: compile("det"),
            cls: compile("cls"),
            rec: compile("rec"),
            dictionary: Arc::new(
                Dictionary::from_utf8(&fs::read(path("dictionary")).unwrap()).unwrap(),
            ),
        }
    }
    fn rec(&self, options: RecognizerOptions, limits: ExecutionLimits) -> Recognizer {
        Recognizer::new(self.rec.clone(), self.dictionary.clone(), options, limits).unwrap()
    }
    fn ocr(
        &self,
        det_options: DetectorOptions,
        line_options: LineOptions,
        options: OcrOptions,
    ) -> Ocr {
        let detector =
            Detector::new(self.det.clone(), det_options, ExecutionLimits::default()).unwrap();
        let extractor =
            OrientedLineExtractor::new(detector, Some(self.cls.clone()), line_options).unwrap();
        Ocr::new(
            extractor,
            self.rec(RecognizerOptions::default(), ExecutionLimits::default()),
            options,
        )
        .unwrap()
    }
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn artifact<'a>(case: &'a Value, id: &str) -> &'a Value {
    case["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == id)
        .unwrap()
}
fn json_artifact(root: &Path, case: &Value, id: &str) -> Value {
    read_json(&root.join(artifact(case, id)["path"].as_str().unwrap()))
}
fn case_pixels(root: &Path, case: &Value) -> Pixels {
    let metadata = json_artifact(root, case, "input.metadata");
    assert_eq!(metadata["pixel_format"], "bgr24");
    Pixels {
        bytes: fs::read(root.join(artifact(case, "input.pixels")["path"].as_str().unwrap()))
            .unwrap(),
        width: metadata["width"].as_u64().unwrap() as usize,
        height: metadata["height"].as_u64().unwrap() as usize,
        stride: metadata["stride"].as_u64().unwrap() as usize,
        format: PixelFormat::Bgr24,
    }
}
fn line_pixels(root: &Path, case: &Value) -> Pixels {
    let a = artifact(case, "line.000.rec.pixels");
    let shape = a["shape"].as_array().unwrap();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[2], 3);
    let (height, width) = (
        shape[0].as_u64().unwrap() as usize,
        shape[1].as_u64().unwrap() as usize,
    );
    Pixels {
        bytes: fs::read(root.join(a["path"].as_str().unwrap())).unwrap(),
        width,
        height,
        stride: width * 3,
        format: PixelFormat::Bgr24,
    }
}
// Exact structure/integers/text; the existing canonical tolerance for floats.
fn assert_native(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
            for (key, value) in b {
                assert_native(&a[key], value);
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                assert_native(a, b);
            }
        }
        (Value::Number(a), Value::Number(b)) if !(a.is_u64() && b.is_u64()) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!(
                a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5 + 1e-4 * b.abs(),
                "{a} != {b}"
            );
        }
        _ => assert_eq!(actual, expected),
    }
}
fn recognize(rec: &mut Recognizer, pixels: &Pixels, capture: bool) -> OwnedRecognition {
    rec.recognize(pixels.view(), capture)
        .unwrap()
        .try_to_owned()
        .unwrap()
}

#[test]
#[ignore = "requires verified tools/prepare-step12-fixtures.py compact native bundles"]
fn all_shipped_families_public_api_reuse_and_recovery() {
    let fixtures = std::env::var_os("SIMDOCR_STEP12_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/step12-fixtures")
        });
    for family in ["tiny", "small", "medium"] {
        let root = fixtures.join(format!("{family}-ocr-ci-v1"));
        let manifest = read_json(&root.join("manifest.json"));
        assert_eq!(manifest["contract"]["model"], family);
        assert_eq!(
            manifest["contract"]["options"]["recognizer"]["adaptive_width"],
            true
        );
        let models = Models::load(&root, &manifest);
        let cases = manifest["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 4, "complete compact corpus is required");
        let crop = line_pixels(&root, &cases[0]);
        let mut rec = models.rec(RecognizerOptions::default(), ExecutionLimits::default());
        let original = recognize(&mut rec, &crop, true);
        assert!(
            original.emitted_count > 0,
            "native crop must exercise result budgets"
        );
        assert_native(
            &serde_json::to_value(&original).unwrap(),
            &json_artifact(&root, &cases[0], "line.000.rec.public"),
        );
        let mut unaligned = original.clone();
        unaligned.ctc_spans = None;
        assert_eq!(recognize(&mut rec, &crop, false), unaligned);
        assert_eq!(recognize(&mut rec, &crop, true), original);
        for format in [
            PixelFormat::Bgr24,
            PixelFormat::Rgb24,
            PixelFormat::Bgra32,
            PixelFormat::Rgba32,
        ] {
            assert_eq!(
                recognize(&mut rec, &crop.reformatted(format), true),
                original,
                "{family} {format:?}"
            );
        }

        let widths = [32, 64, 96, 160, 224, 320, 480, 576, 992, 32, 992];
        let sequence = widths.map(|width| crop.width_sample(width));
        let expected = sequence
            .iter()
            .map(|pixels| {
                let mut fresh =
                    models.rec(RecognizerOptions::default(), ExecutionLimits::default());
                recognize(&mut fresh, pixels, true)
            })
            .collect::<Vec<_>>();
        for (pixels, expected) in sequence.iter().zip(&expected) {
            assert_eq!(
                recognize(&mut rec, pixels, true),
                *expected,
                "{family} width {}",
                pixels.width
            );
            assert_eq!(expected.tensor_width, pixels.width);
        }
        let mut changed = sequence[0].clone();
        for byte in &mut changed.bytes {
            *byte = 255 - *byte;
        }
        assert_ne!(changed.bytes, sequence[0].bytes);
        let changed_expected = recognize(
            &mut models.rec(RecognizerOptions::default(), ExecutionLimits::default()),
            &changed,
            true,
        );
        assert_eq!(recognize(&mut rec, &changed, true), changed_expected);
        assert_eq!(recognize(&mut rec, &sequence[0], true), expected[0]);
        assert!(rec.select_width(0, 48).is_err());
        assert!(rec.select_width(usize::MAX, 48).is_err());
        assert!(
            ImageView::new(&crop.bytes, crop.width, crop.height, 1, PixelFormat::Bgr24).is_err()
        );
        assert_eq!(recognize(&mut rec, &crop, true), original);

        let fixed_options = RecognizerOptions {
            adaptive_width: false,
            target_width: 320,
            ..Default::default()
        };
        let mut fixed = models.rec(fixed_options, ExecutionLimits::default());
        let fixed_size = fixed.select_width(992, 48).unwrap();
        assert_eq!(
            (fixed_size.resized_width, fixed_size.tensor_width),
            (320, 320)
        );
        let fixed_expected = recognize(
            &mut models.rec(fixed_options, ExecutionLimits::default()),
            &sequence[8],
            true,
        );
        assert_eq!(recognize(&mut fixed, &sequence[8], true), fixed_expected);
        assert_eq!(
            (fixed_expected.resized_width, fixed_expected.tensor_width),
            (320, 320)
        );
        recognize(&mut fixed, &sequence[0], false);
        assert_eq!(recognize(&mut fixed, &sequence[8], true), fixed_expected);

        // Reject a wide rebind, then reproduce the previous narrow public result.
        let narrow_bytes = models
            .rec(RecognizerOptions::default(), ExecutionLimits::default())
            .activation_bytes();
        let mut capped = models.rec(
            RecognizerOptions::default(),
            ExecutionLimits {
                max_activation_bytes: narrow_bytes,
            },
        );
        let narrow = recognize(&mut capped, &sequence[0], true);
        assert!(capped.recognize(sequence[8].view(), true).is_err());
        assert_eq!(recognize(&mut capped, &sequence[0], true), narrow);
        let mut image_capped = models.rec(
            RecognizerOptions {
                max_image_pixels: 32 * 48,
                ..Default::default()
            },
            ExecutionLimits::default(),
        );
        let narrow = recognize(&mut image_capped, &sequence[0], true);
        assert!(image_capped.recognize(sequence[8].view(), true).is_err());
        assert_eq!(recognize(&mut image_capped, &sequence[0], true), narrow);
        // Same nonempty crop fits text alone, but cannot fit added alignment records.
        let mut result_capped = models.rec(
            RecognizerOptions {
                max_result_bytes: original.text.len(),
                ..Default::default()
            },
            ExecutionLimits::default(),
        );
        assert_eq!(recognize(&mut result_capped, &crop, false), unaligned);
        assert!(result_capped.recognize(crop.view(), true).is_err());
        assert_eq!(recognize(&mut result_capped, &crop, false), unaligned);
        assert_eq!(rec.diagnostic_retained_activation_bytes(), 0);
        assert_eq!(rec.diagnostic_retained_workspace_bytes(), 0);
        assert_eq!(fixed.diagnostic_retained_activation_bytes(), 0);

        let mut ocr = models.ocr(
            DetectorOptions::default(),
            LineOptions::default(),
            OcrOptions::default(),
        );
        let images = cases
            .iter()
            .map(|case| case_pixels(&root, case))
            .collect::<Vec<_>>();
        assert_eq!(
            (images[0].width, images[0].height),
            (images[1].width, images[1].height)
        );
        assert_ne!(images[0].bytes, images[1].bytes);
        let mut snapshots = Vec::new();
        for (case, pixels) in cases.iter().zip(&images) {
            let output = ocr
                .run(pixels.view(), true)
                .unwrap()
                .try_to_owned()
                .unwrap();
            assert_native(
                &public_json(output.as_view()),
                &json_artifact(&root, case, "result.public"),
            );
            snapshots.push(output);
        }
        assert!(!snapshots[0].lines.is_empty() && !snapshots[1].lines.is_empty());
        assert!(snapshots[2].lines.is_empty() && snapshots[3].lines.is_empty());
        for format in [
            PixelFormat::Bgr24,
            PixelFormat::Rgb24,
            PixelFormat::Bgra32,
            PixelFormat::Rgba32,
        ] {
            let formatted = images[0].reformatted(format);
            assert_eq!(
                ocr.run(formatted.view(), true)
                    .unwrap()
                    .try_to_owned()
                    .unwrap(),
                snapshots[0]
            );
        }
        let mut unaligned_ocr = snapshots[0].clone();
        for line in &mut unaligned_ocr.lines {
            line.ctc_spans = None;
        }
        assert_eq!(
            ocr.run(images[0].view(), false)
                .unwrap()
                .try_to_owned()
                .unwrap(),
            unaligned_ocr
        );
        assert_eq!(
            ocr.run(images[1].view(), true)
                .unwrap()
                .try_to_owned()
                .unwrap(),
            snapshots[1]
        );
        assert_eq!(
            ocr.run(images[0].view(), true)
                .unwrap()
                .try_to_owned()
                .unwrap(),
            snapshots[0]
        );
        // Public image validation does not invalidate the existing reusable engine.
        assert!(
            ImageView::new(&images[0].bytes, 0, images[0].height, 0, PixelFormat::Bgr24).is_err()
        );
        assert_eq!(
            ocr.run(images[0].view(), true)
                .unwrap()
                .try_to_owned()
                .unwrap(),
            snapshots[0]
        );
        let budget_variants = [
            (
                DetectorOptions {
                    max_image_pixels: images[0].width * images[0].height - 1,
                    ..Default::default()
                },
                LineOptions::default(),
                OcrOptions::default(),
            ),
            (
                DetectorOptions::default(),
                LineOptions {
                    max_crop_pixels: 1,
                    ..Default::default()
                },
                OcrOptions::default(),
            ),
            (
                DetectorOptions::default(),
                LineOptions::default(),
                OcrOptions {
                    max_total_result_bytes: 1,
                },
            ),
        ];
        for (det_options, line_options, options) in budget_variants {
            let mut limited = models.ocr(det_options, line_options, options);
            assert!(
                limited.run(images[0].view(), true).is_err(),
                "{family} budget must reject a nonempty page"
            );
            assert_eq!(
                limited
                    .run(images[2].view(), true)
                    .unwrap()
                    .try_to_owned()
                    .unwrap(),
                snapshots[2]
            );
            assert_eq!(
                limited
                    .run(images[3].view(), false)
                    .unwrap()
                    .try_to_owned()
                    .unwrap(),
                snapshots[3]
            );
            assert_eq!(limited.diagnostic_retained_activation_bytes(), 0);
        }
        assert_eq!(ocr.diagnostic_retained_activation_bytes(), 0);
        eprintln!(
            "PASS real public API contracts: {family}; adaptive widths={widths:?}; fixed=320; formats=4; alignment toggles; budget recovery"
        );
    }
}
