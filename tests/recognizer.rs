use simd_paddle_ocr_rs::{
    image::{ImageError, ImageView, PixelFormat},
    model::Model,
    preprocess::{self, ResizeWorkspace},
    recognizer::{Dictionary, RecognitionTrace, Recognizer, RecognizerOptions},
    runtime::{CompiledModel, ExecutionLimits},
};
use std::sync::Arc;
use std::{fs, path::PathBuf};

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
                                        string(2, "width")
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
fn compiled() -> Arc<CompiledModel> {
    let graph = [
        tensor(
            "weights",
            &[3, 4],
            &[4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ),
        tensor("bias", &[4], &[0., 2., 2., 0.]),
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
        node("MatMul", &["features", "weights"], "logits", vec![]),
        node("Add", &["logits", "bias"], "biased", vec![]),
        node("Softmax", &["biased"], "y", vec![int("axis", 2)]),
        bytes(11, info("x", &[1, 3, 48, -1])),
        bytes(12, info("y", &[1, -1, 4])),
    ]
    .concat();
    let model = [number(1, 8), bytes(8, number(2, 7)), bytes(7, graph)].concat();
    Arc::new(CompiledModel::new(Arc::new(Model::load_bytes(&model).unwrap())).unwrap())
}
fn recognizer(options: RecognizerOptions) -> Recognizer {
    Recognizer::new(
        compiled(),
        Arc::new(Dictionary::from_utf8("你好\n👩‍🔬".as_bytes()).unwrap()),
        options,
        ExecutionLimits::default(),
    )
    .unwrap()
}
fn pixels(w: usize, h: usize) -> Vec<u8> {
    (0..w * h * 3)
        .map(|i| ((i * 73 + 19) % 256) as u8)
        .collect()
}

#[cfg(feature = "profiling")]
#[test]
fn production_profile_preserves_results_and_never_initializes_dense_storage() {
    let mut engine = recognizer(RecognizerOptions::default());
    for width in [32, 320, 64, 992, 32] {
        let data = pixels(width, 48);
        let image = ImageView::new(&data, width, 48, 0, PixelFormat::Bgr24).unwrap();
        for capture in [false, true] {
            let expected = engine
                .recognize(image, capture)
                .unwrap()
                .try_to_owned()
                .unwrap();
            let (result, profile) = engine.recognize_profiled(image, capture).unwrap();
            assert_eq!(result.try_to_owned().unwrap(), expected);
            assert_eq!(profile.recognized_lines, 1);
            assert!(
                profile.total_ns > 0 && profile.rec_backbone_ns > 0 && profile.projection_ns > 0
            );
            assert_eq!(profile.det_inference_ns, 0);
            assert_eq!(engine.diagnostic_retained_activation_bytes(), 0);
            assert_eq!(engine.diagnostic_retained_workspace_bytes(), 0);
        }
    }
}

#[test]
fn strict_dictionary_preserves_native_class_numbering_and_string_tokens() {
    let dictionary = Dictionary::from_utf8("\u{feff}你好\r\n\n👩‍🔬\r\n你好\n\n".as_bytes()).unwrap();
    assert_eq!(dictionary.class_count(), 7);
    assert_eq!(dictionary.token(0), None);
    assert_eq!(dictionary.token(1), Some("你好"));
    assert_eq!(dictionary.token(2), Some(""));
    assert_eq!(dictionary.token(3), Some("👩‍🔬"));
    assert_eq!(dictionary.token(4), Some("你好"));
    assert_eq!(dictionary.token(5), Some(""));
    assert_eq!(dictionary.token(6), Some(" "));
    assert_eq!(dictionary.token(7), None);
    for bytes in [&b""[..], &b"\n\r\n"[..], &[0xff][..], "\u{feff}".as_bytes()] {
        assert!(Dictionary::from_utf8(bytes).is_err());
    }
    for text in ["a", "a\n", "a\r\n"] {
        let d = Dictionary::from_utf8(text.as_bytes()).unwrap();
        assert_eq!(d.class_count(), 3);
        assert_eq!(d.token(1), Some("a"));
    }
    assert!(
        Recognizer::new(
            compiled(),
            Arc::new(Dictionary::from_utf8(b"a").unwrap()),
            RecognizerOptions::default(),
            ExecutionLimits::default()
        )
        .is_err()
    );
}
#[test]
fn width_selection_has_no_adaptive_cap_and_fixed_mode_resizes_whole_line() {
    let adaptive = recognizer(RecognizerOptions::default());
    for (width, target) in [
        (1, 32),
        (31, 32),
        (32, 32),
        (33, 64),
        (319, 320),
        (320, 320),
        (321, 352),
        (992, 992),
    ] {
        let s = adaptive.select_width(width, 48).unwrap();
        assert_eq!((s.resized_width, s.tensor_width), (width, target));
    }
    assert!(adaptive.select_width(0, 48).is_err());
    assert!(adaptive.select_width(i32::MAX as usize, 1).is_err());
    let fixed = recognizer(RecognizerOptions {
        adaptive_width: false,
        ..Default::default()
    });
    let s = fixed.select_width(992, 48).unwrap();
    assert_eq!((s.resized_width, s.tensor_width), (320, 320));
    let data = pixels(333, 51);
    let image = ImageView::new(&data, 333, 51, 0, PixelFormat::Bgr24).unwrap();
    let mut input = vec![9.; 3 * 48 * 320];
    let mut work = ResizeWorkspace::default();
    assert_eq!(
        preprocess::rec(image, 320, &mut input, &mut work).unwrap(),
        314
    );
    for c in 0..3 {
        for y in 0..48 {
            assert!(
                input[c * 48 * 320 + y * 320 + 314..c * 48 * 320 + (y + 1) * 320]
                    .iter()
                    .all(|&v| v == 0.)
            );
        }
    }
}
#[test]
fn rec_formats_stride_padding_and_reused_pixels_are_identical() {
    let (w, h) = (37, 29);
    let data = pixels(w, h);
    let base = ImageView::new(&data, w, h, 0, PixelFormat::Bgr24).unwrap();
    let mut work = ResizeWorkspace::default();
    let mut expected = vec![0.; 3 * 48 * 64];
    preprocess::rec(base, 64, &mut expected, &mut work).unwrap();
    for format in [
        PixelFormat::Bgr24,
        PixelFormat::Rgb24,
        PixelFormat::Bgra32,
        PixelFormat::Rgba32,
    ] {
        let bpp = format.bytes_per_pixel();
        let stride = w * bpp + 13;
        let mut data2 = vec![243; (h - 1) * stride + w * bpp];
        for y in 0..h {
            for x in 0..w {
                let a = (y * w + x) * 3;
                let b = y * stride + x * bpp;
                let p = &data[a..a + 3];
                if matches!(format, PixelFormat::Rgb24 | PixelFormat::Rgba32) {
                    data2[b..b + 3].copy_from_slice(&[p[2], p[1], p[0]]);
                } else {
                    data2[b..b + 3].copy_from_slice(p);
                }
                if bpp == 4 {
                    data2[b + 3] = (x * y) as u8;
                }
            }
        }
        let view = ImageView::new(&data2, w, h, stride, format).unwrap();
        let mut actual = vec![9.; expected.len()];
        preprocess::rec(view, 64, &mut actual, &mut work).unwrap();
        assert_eq!(actual, expected);
    }
    let black = vec![0; w * h * 3];
    let mut actual = expected.clone();
    preprocess::rec(
        ImageView::new(&black, w, h, 0, PixelFormat::Bgr24).unwrap(),
        64,
        &mut actual,
        &mut work,
    )
    .unwrap();
    assert_ne!(actual, expected);
    assert_eq!(actual[0], -1.);
}
#[test]
fn public_owned_trace_and_mixed_width_reuse_preserve_score_contracts() {
    let mut rec = recognizer(RecognizerOptions::default());
    let data = pixels(33, 48);
    let image = ImageView::new(&data, 33, 48, 0, PixelFormat::Bgr24).unwrap();
    let owned = rec.recognize(image, true).unwrap().try_to_owned().unwrap();
    assert_eq!(owned.text, "你好");
    assert_eq!(owned.score, 2.);
    assert_eq!(owned.emitted_count, 1);
    assert_eq!(
        (owned.resized_width, owned.tensor_width, owned.time_steps),
        (33, 64, 64)
    );
    let span = &owned.ctc_spans.as_ref().unwrap()[0];
    assert_eq!((span.start_column, span.end_column), (0, 64));
    assert_eq!(rec.diagnostic_retained_activation_bytes(), 0);
    let mut nodes = 0;
    let mut dense = None;
    let mut public = None;
    let traced = rec
        .recognize_with_trace(image, true, |event| {
            match event {
                RecognitionTrace::Node(_) => nodes += 1,
                RecognitionTrace::DecodedDense(r) => dense = Some(r.try_to_owned()?),
                RecognitionTrace::Public(r) => public = Some(r.try_to_owned()?),
                _ => {}
            }
            Ok(())
        })
        .unwrap()
        .try_to_owned()
        .unwrap();
    assert_eq!(nodes, 5);
    assert_eq!(traced, owned);
    assert_eq!(public.unwrap(), owned);
    let dense = dense.unwrap();
    assert_eq!(dense.text, owned.text);
    assert!(dense.score > 0. && dense.score < 1.);
    assert!(rec.diagnostic_retained_activation_bytes() > 0);
    for width in [992, 32, 224, 992, 32, 224] {
        let bytes = pixels(width, 48);
        let out = rec
            .recognize(
                ImageView::new(&bytes, width, 48, 0, PixelFormat::Bgr24).unwrap(),
                false,
            )
            .unwrap();
        assert_eq!(out.text, "你好");
        assert!(out.ctc_spans.is_none());
    }
    assert_eq!(owned.as_view().text, "你好");
    assert_eq!(
        owned.as_view().ctc_spans.unwrap().get(0).unwrap().text,
        "你好"
    );
}
#[test]
fn observer_and_resource_failures_leave_recognizer_reusable() {
    let mut rec = recognizer(RecognizerOptions::default());
    let data = pixels(33, 48);
    let image = ImageView::new(&data, 33, 48, 0, PixelFormat::Bgr24).unwrap();
    for fail_at in 0..9 {
        let mut event_index = 0;
        let error = rec
            .recognize_with_trace(image, true, |_| {
                let i = event_index;
                event_index += 1;
                if i == fail_at {
                    Err(ImageError::new("injected observer error"))
                } else {
                    Ok(())
                }
            })
            .err()
            .unwrap();
        assert!(error.to_string().contains("injected"));
        assert_eq!(rec.recognize(image, false).unwrap().text, "你好");
    }
    let mut limited = recognizer(RecognizerOptions {
        target_width: 33,
        adaptive_width: false,
        max_result_bytes: 1,
        ..Default::default()
    });
    assert!(limited.recognize(image, false).is_err());
    let white = vec![255; data.len()];
    let empty = limited
        .recognize(
            ImageView::new(&white, 33, 48, 0, PixelFormat::Bgr24).unwrap(),
            true,
        )
        .unwrap();
    assert_eq!(empty.text, "");
    assert_eq!(empty.score, 0.);
    assert_eq!(empty.emitted_count, 0);
    assert!(empty.ctc_spans.unwrap().is_empty());
    assert!(
        Recognizer::new(
            compiled(),
            Arc::new(Dictionary::from_utf8(b"a\nb").unwrap()),
            RecognizerOptions {
                target_width: 0,
                ..Default::default()
            },
            ExecutionLimits::default()
        )
        .is_err()
    );
    let mut capped = recognizer(RecognizerOptions {
        max_image_pixels: 1,
        ..Default::default()
    });
    assert!(capped.recognize(image, false).is_err());
    let pixel = [0; 3];
    assert_eq!(
        capped
            .recognize(
                ImageView::new(&pixel, 1, 1, 0, PixelFormat::Bgr24).unwrap(),
                false
            )
            .unwrap()
            .text,
        "你好"
    );
}

#[test]
fn independently_captured_native_rec_preprocessing_is_bit_exact() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recognition-v1");
    let validation = simd_paddle_ocr_rs::validate_bundle(&root).unwrap();
    assert_eq!(validation.checked_cases, 56);
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    let mut workspace = ResizeWorkspace::default();
    let mut output = Vec::new();
    // Reversing the same corpus exercises changing widths and cached coefficient
    // keys while stale output values are replaced on every call.
    for pass in 0..2 {
        for ordinal in 0..cases.len() {
            let case = &cases[if pass == 0 {
                ordinal
            } else {
                cases.len() - 1 - ordinal
            }];
            let artifact = |id: &str| {
                case["artifacts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|a| a["id"] == id)
                    .unwrap()
            };
            let read =
                |id: &str| fs::read(root.join(artifact(id)["path"].as_str().unwrap())).unwrap();
            let fixture: serde_json::Value = serde_json::from_slice(&read("fixture")).unwrap();
            let dimension = |field: &str| fixture[field].as_u64().unwrap() as usize;
            let format = match fixture["pixel_format"].as_str().unwrap() {
                "bgr24" => PixelFormat::Bgr24,
                "rgb24" => PixelFormat::Rgb24,
                "bgra32" => PixelFormat::Bgra32,
                "rgba32" => PixelFormat::Rgba32,
                other => panic!("unexpected native pixel format {other}"),
            };
            let recognizer = recognizer(RecognizerOptions {
                target_width: dimension("fixed_width"),
                adaptive_width: fixture["adaptive_width"].as_bool().unwrap(),
                ..Default::default()
            });
            let size = recognizer
                .select_width(dimension("width"), dimension("height"))
                .unwrap();
            assert_eq!(
                size.tensor_width,
                dimension("target_width"),
                "{}",
                case["id"]
            );
            assert_eq!(
                size.resized_width,
                dimension("actual_width"),
                "{}",
                case["id"]
            );
            let source = read("pixels");
            let image = ImageView::new(
                &source,
                dimension("width"),
                dimension("height"),
                dimension("stride"),
                format,
            )
            .unwrap();
            let expected = read("rec.input");
            output.resize(expected.len() / 4, 42.);
            output.fill(42.);
            let actual_width =
                preprocess::rec(image, size.tensor_width, &mut output, &mut workspace).unwrap();
            assert_eq!(actual_width, size.resized_width, "{}", case["id"]);
            for (index, (&actual, bytes)) in output.iter().zip(expected.chunks_exact(4)).enumerate()
            {
                assert_eq!(
                    actual.to_bits(),
                    u32::from_le_bytes(bytes.try_into().unwrap()),
                    "native preprocessing {} differs at element {index} (pass {pass})",
                    case["id"]
                );
            }
        }
    }
}
