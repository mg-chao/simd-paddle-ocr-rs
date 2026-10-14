use serde_json::Value;
use simd_paddle_ocr_rs::{
    crop::{CropTransform, rotate180},
    detector::{DetectionBox, Detector, DetectorOptions},
    image::{ImageView, PixelFormat},
    model::Model,
    preprocess::{self, ResizeWorkspace},
    runtime::{CompiledModel, ExecutionLimits},
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

fn pixels(w: usize, h: usize) -> Vec<u8> {
    (0..w * h * 3)
        .map(|i| ((i * 73 + 19) % 256) as u8)
        .collect()
}
fn reformat(bgr: &[u8], w: usize, h: usize, f: PixelFormat, pad: usize) -> Vec<u8> {
    let bpp = f.bytes_per_pixel();
    let stride = w * bpp + pad;
    let mut result = vec![243; (h - 1) * stride + w * bpp];
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * 3;
            let dst = y * stride + x * bpp;
            let p = &bgr[src..src + 3];
            if matches!(f, PixelFormat::Rgb24 | PixelFormat::Rgba32) {
                result[dst..dst + 3].copy_from_slice(&[p[2], p[1], p[0]]);
            } else {
                result[dst..dst + 3].copy_from_slice(p);
            }
            if bpp == 4 {
                result[dst + 3] = ((x * 17 + y * 31) % 256) as u8;
            }
        }
    }
    result
}
#[test]
fn checked_images_accept_unpadded_final_row_and_reject_bad_bounds() {
    let data = [0u8; 16];
    assert!(ImageView::new(&data, 2, 2, 10, PixelFormat::Bgr24).is_ok());
    assert!(ImageView::new(&data[..15], 2, 2, 10, PixelFormat::Bgr24).is_err());
    assert!(ImageView::new(&data, 0, 2, 0, PixelFormat::Bgr24).is_err());
    assert!(ImageView::new(&data, 2, 2, 5, PixelFormat::Bgr24).is_err());
    assert_eq!(
        ImageView::new(&data, 2, 2, 0, PixelFormat::Bgra32)
            .unwrap()
            .stride(),
        8
    );
}
#[test]
fn formats_stride_and_alpha_preserve_preprocessed_values_and_crop_bytes() {
    let (w, h) = (37, 29);
    let bgr = pixels(w, h);
    let image = ImageView::new(&bgr, w, h, 0, PixelFormat::Bgr24).unwrap();
    let mut workspace = ResizeWorkspace::default();
    let mut det = vec![0.; 3 * 32 * 64];
    preprocess::det(image, 64, 32, &mut det, &mut workspace).unwrap();
    let mut cls = vec![0.; 3 * 80 * 160];
    let cw = preprocess::cls(image, &mut cls, &mut workspace).unwrap();
    let transform = CropTransform::new(&DetectionBox {
        points: [[3., 4.], [32., 5.], [31., 24.], [4., 23.]],
        score: 1.,
    })
    .unwrap();
    let mut crop = vec![0; transform.byte_count().unwrap()];
    transform.extract_into(image, &mut crop).unwrap();
    for format in [
        PixelFormat::Bgr24,
        PixelFormat::Rgb24,
        PixelFormat::Bgra32,
        PixelFormat::Rgba32,
    ] {
        let bytes = reformat(&bgr, w, h, format, 13);
        let image =
            ImageView::new(&bytes, w, h, w * format.bytes_per_pixel() + 13, format).unwrap();
        let mut d = vec![0.; det.len()];
        preprocess::det(image, 64, 32, &mut d, &mut workspace).unwrap();
        assert_eq!(d, det);
        let mut c = vec![0.; cls.len()];
        assert_eq!(preprocess::cls(image, &mut c, &mut workspace).unwrap(), cw);
        assert_eq!(c, cls);
        let mut p = vec![0; crop.len()];
        transform.extract_into(image, &mut p).unwrap();
        assert_eq!(p, crop);
    }
}
#[test]
fn classifier_left_window_padding_and_workspace_pixels_are_refreshed() {
    let (w, h) = (120, 20);
    let bgr = pixels(w, h);
    let mut painted = bgr.clone();
    for y in 0..h {
        painted[(y * w + 4 * h) * 3..(y + 1) * w * 3].fill(255);
    }
    let mut a = vec![0.; 3 * 80 * 160];
    let mut b = a.clone();
    let mut scratch = ResizeWorkspace::default();
    assert_eq!(
        preprocess::cls(
            ImageView::new(&bgr, w, h, 0, PixelFormat::Bgr24).unwrap(),
            &mut a,
            &mut scratch
        )
        .unwrap(),
        160
    );
    preprocess::cls(
        ImageView::new(&painted, w, h, 0, PixelFormat::Bgr24).unwrap(),
        &mut b,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(a, b);
    let short = pixels(10, 20);
    let width = preprocess::cls(
        ImageView::new(&short, 10, 20, 0, PixelFormat::Bgr24).unwrap(),
        &mut b,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(width, 40);
    for c in 0..3 {
        for y in 0..80 {
            assert!(
                b[c * 80 * 160 + y * 160 + 40..c * 80 * 160 + (y + 1) * 160]
                    .iter()
                    .all(|&v| v == -1.)
            );
        }
    }
    let white = vec![255; short.len()];
    preprocess::cls(
        ImageView::new(&white, 10, 20, 0, PixelFormat::Bgr24).unwrap(),
        &mut b,
        &mut scratch,
    )
    .unwrap();
    assert_ne!(a, b);
    assert_eq!(b[0], 1.);
}
#[test]
fn detector_sizes_use_truncation_then_round_and_tiny_padding() {
    assert_eq!(
        (
            preprocess::compute_det_size(1, 1, 960).unwrap().width,
            preprocess::compute_det_size(1, 1, 960).unwrap().height
        ),
        (32, 32)
    );
    let size = preprocess::compute_det_size(1500, 500, 960).unwrap();
    assert_eq!((size.width, size.height), (960, 320));
    assert_eq!(size.width_ratio, 0.64);
    assert_eq!(size.height_ratio, 0.64);
    assert_eq!(
        (
            preprocess::compute_det_size(48, 48, 960).unwrap().width,
            preprocess::compute_det_size(48, 48, 960).unwrap().height
        ),
        (64, 64)
    );
    assert!(preprocess::compute_det_size(0, 2, 960).is_err());
}
#[test]
fn crop_floor_vertical_rotation_and_odd_pixel_180_are_explicit() {
    let b = DetectionBox {
        points: [[1., 1.], [5., 1.], [5., 7.], [1., 7.]],
        score: 1.,
    };
    let t = CropTransform::new(&b).unwrap();
    assert!(t.vertical_rotation());
    assert_eq!((t.width(), t.height()), (6, 4));
    let t = CropTransform::new(&DetectionBox {
        points: [[1., 1.], [5.9, 1.], [5.9, 6.], [1., 6.]],
        score: 1.,
    })
    .unwrap();
    assert_eq!((t.unrotated_width(), t.unrotated_height()), (4, 5));
    assert!(!t.vertical_rotation());
    let mut p = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    rotate180(&mut p, 3, 1).unwrap();
    assert_eq!(p, [7, 8, 9, 4, 5, 6, 1, 2, 3]);
    assert!(rotate180(&mut p, 2, 1).is_err());
    assert!(
        CropTransform::new(&DetectionBox {
            points: [[0., 0.]; 4],
            score: 1.
        })
        .is_err()
    );
}
fn floats(path: &Path) -> Vec<f32> {
    fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn artifact<'a>(case: &'a Value, id: &str) -> &'a Value {
    case["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == id)
        .unwrap()
}
#[test]
fn native_synthetic_resize_and_crop_goldens_are_byte_exact() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/geometry-v1");
    let manifest: Value = serde_json::from_slice(
        &fs::read(root.join("manifest.json")).expect("committed native geometry fixtures"),
    )
    .unwrap();
    let mut scratch = ResizeWorkspace::default();
    let mut checked = 0;
    for case in manifest["cases"].as_array().unwrap() {
        let kind = case["kind"].as_str().unwrap();
        if kind != "resize" && kind != "crop" {
            continue;
        }
        let meta: Value = serde_json::from_slice(
            &fs::read(root.join(artifact(case, "fixture")["path"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let format = match meta["pixel_format"].as_str().unwrap() {
            "bgr24" => PixelFormat::Bgr24,
            "rgb24" => PixelFormat::Rgb24,
            "bgra32" => PixelFormat::Bgra32,
            "rgba32" => PixelFormat::Rgba32,
            _ => panic!("unknown fixture format"),
        };
        let bytes =
            fs::read(root.join(artifact(case, "pixels")["path"].as_str().unwrap())).unwrap();
        let image = ImageView::new(
            &bytes,
            meta["width"].as_u64().unwrap() as usize,
            meta["height"].as_u64().unwrap() as usize,
            meta["stride"].as_u64().unwrap() as usize,
            format,
        )
        .unwrap();
        if kind == "resize" {
            let w = meta["det_width"].as_u64().unwrap() as usize;
            let h = meta["det_height"].as_u64().unwrap() as usize;
            let mut output = vec![0.; 3 * w * h];
            preprocess::det(image, w, h, &mut output, &mut scratch).unwrap();
            assert_eq!(
                output,
                floats(&root.join(artifact(case, "det.input")["path"].as_str().unwrap())),
                "{} DET",
                case["id"]
            );
            let mut output = vec![0.; 3 * 80 * 160];
            let width = preprocess::cls(image, &mut output, &mut scratch).unwrap();
            assert_eq!(width, meta["cls_content_width"].as_u64().unwrap() as usize);
            assert_eq!(
                output,
                floats(&root.join(artifact(case, "cls.input")["path"].as_str().unwrap())),
                "{} CLS",
                case["id"]
            );
        } else {
            let box_ = DetectionBox {
                points: serde_json::from_value(meta["box"]["points"].clone()).unwrap(),
                score: meta["box"]["score"].as_f64().unwrap() as f32,
            };
            let transform = CropTransform::new(&box_).unwrap();
            assert_eq!(
                transform.width(),
                meta["crop_width"].as_u64().unwrap() as usize
            );
            assert_eq!(
                transform.height(),
                meta["crop_height"].as_u64().unwrap() as usize
            );
            let mut output = vec![0; transform.byte_count().unwrap()];
            transform.extract_into(image, &mut output).unwrap();
            assert_eq!(
                output,
                fs::read(root.join(artifact(case, "crop")["path"].as_str().unwrap())).unwrap(),
                "{} crop",
                case["id"]
            );
            rotate180(&mut output, transform.width(), transform.height()).unwrap();
            assert_eq!(
                output,
                fs::read(root.join(artifact(case, "rotated")["path"].as_str().unwrap())).unwrap(),
                "{} 180",
                case["id"]
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 44);
}
#[test]
#[ignore = "requires SIMDOCR_IMAGE_REFERENCE pointing at an authoritative canonical archive"]
fn frozen_image_stages_match_exact_preprocess_geometry_and_crop_bytes() {
    let root = PathBuf::from(
        std::env::var("SIMDOCR_IMAGE_REFERENCE").expect("set SIMDOCR_IMAGE_REFERENCE"),
    );
    let manifest: Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let model = Arc::new(Model::load_path(root.join("models/det.onnx")).unwrap());
    let mut detector = Detector::new(
        Arc::new(CompiledModel::new(model).unwrap()),
        DetectorOptions {
            box_threshold: manifest["contract"]["options"]["detector"]["box_threshold"]
                .as_f64()
                .map(|v| v as f32),
            ..Default::default()
        },
        ExecutionLimits::default(),
    )
    .unwrap();
    let mut scratch = ResizeWorkspace::default();
    let mut count = 0;
    for case in manifest["cases"].as_array().unwrap() {
        let meta: Value = serde_json::from_slice(
            &fs::read(root.join(artifact(case, "input.metadata")["path"].as_str().unwrap()))
                .unwrap(),
        )
        .unwrap();
        let (w, h, stride) = (
            meta["width"].as_u64().unwrap() as usize,
            meta["height"].as_u64().unwrap() as usize,
            meta["stride"].as_u64().unwrap() as usize,
        );
        let bytes =
            fs::read(root.join(artifact(case, "input.pixels")["path"].as_str().unwrap())).unwrap();
        let image = ImageView::new(&bytes, w, h, stride, PixelFormat::Bgr24).unwrap();
        let input = artifact(case, "det.input");
        let shape = input["shape"].as_array().unwrap();
        let (rh, rw) = (
            shape[2].as_u64().unwrap() as usize,
            shape[3].as_u64().unwrap() as usize,
        );
        if w + h >= 64 {
            let mut values = vec![0.; 3 * rw * rh];
            preprocess::det(image, rw, rh, &mut values, &mut scratch).unwrap();
            assert_eq!(
                values,
                floats(&root.join(input["path"].as_str().unwrap())),
                "DET preprocess {}",
                case["id"]
            );
        }
        let map = floats(
            &root.join(
                artifact(case, "det.probabilities")["path"]
                    .as_str()
                    .unwrap(),
            ),
        );
        let saved: Value = serde_json::from_slice(
            &fs::read(root.join(artifact(case, "det.boxes")["path"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let result = detector
            .postprocess(
                &map,
                rw,
                rh,
                image,
                saved["width_ratio"].as_f64().unwrap() as f32,
                saved["height_ratio"].as_f64().unwrap() as f32,
            )
            .unwrap();
        let expected = saved["boxes"].as_array().unwrap();
        assert_eq!(result.boxes.len(), expected.len());
        for (i, (b, expected)) in result.boxes.iter().zip(expected).enumerate() {
            let points: [[f32; 2]; 4] = serde_json::from_value(expected["points"].clone()).unwrap();
            assert_eq!(b.points, points, "DB geometry {}/{i}", case["id"]);
            assert_eq!(b.score, expected["score"].as_f64().unwrap() as f32);
            let transform = CropTransform::new(b).unwrap();
            let crop = artifact(case, &format!("line.{i:03}.crop"));
            let mut output = vec![0; transform.byte_count().unwrap()];
            transform.extract_into(image, &mut output).unwrap();
            assert_eq!(
                output,
                fs::read(root.join(crop["path"].as_str().unwrap())).unwrap(),
                "crop {}/{i}",
                case["id"]
            );
            let input = artifact(case, &format!("line.{i:03}.cls.input"));
            let mut values = vec![0.; 3 * 80 * 160];
            preprocess::cls(
                ImageView::new(
                    &output,
                    transform.width(),
                    transform.height(),
                    0,
                    PixelFormat::Bgr24,
                )
                .unwrap(),
                &mut values,
                &mut scratch,
            )
            .unwrap();
            assert_eq!(
                values,
                floats(&root.join(input["path"].as_str().unwrap())),
                "CLS preprocess {}/{i}",
                case["id"]
            );
            count += 1;
        }
    }
    assert!(count > 0);
}
