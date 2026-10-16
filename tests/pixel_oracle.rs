//! Replays committed little-endian tensors exported by the pinned C# oracle.
//! No model files, .NET, fonts or image codecs are needed for these CI gates.
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{
    crop::CropTransform,
    detector::DetectionBox,
    image::{ImageView, PixelFormat},
    preprocess::{self, ResizeWorkspace},
};

#[test]
fn frozen_tensor_files_match_the_recorded_csharp_checksums_and_shapes() {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("data/pixel-oracle/manifest.json")).unwrap();
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("../tools/reference-lock.json")).unwrap();
    assert_eq!(manifest["reference_revision"], reference["revision"]);
    assert_eq!(manifest["files"].as_object().unwrap().len(), 4);
    for (name, bytes) in [
        (
            "det.f32",
            include_bytes!("data/pixel-oracle/det.f32").as_slice(),
        ),
        (
            "cls.f32",
            include_bytes!("data/pixel-oracle/cls.f32").as_slice(),
        ),
        (
            "rec.f32",
            include_bytes!("data/pixel-oracle/rec.f32").as_slice(),
        ),
        (
            "crop.f32",
            include_bytes!("data/pixel-oracle/crop.f32").as_slice(),
        ),
    ] {
        let entry = &manifest["files"][name];
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            entry["sha256"].as_str().unwrap(),
            "{name}"
        );
        let elements: u64 = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap())
            .product();
        assert_eq!(bytes.len() as u64, elements * 4, "{name}");
    }
}

fn assert_frozen(actual: &[f32], expected: &[u8], label: &str) {
    assert_eq!(actual.len() * 4, expected.len(), "{label} length");
    for (index, (&value, bytes)) in actual
        .iter()
        .zip(expected.as_chunks::<4>().0.iter())
        .enumerate()
    {
        assert_eq!(
            &value.to_le_bytes(),
            bytes,
            "{label} element {index}: {value}"
        );
    }
}

#[test]
fn tiny_raw_image_matches_original_det_cls_rec_and_perspective_samples_bitwise() {
    let pixels: Vec<u8> = (0..7 * 9 * 3).map(|i| (i * 17 % 256) as u8).collect();
    let image = ImageView::new(&pixels, 7, 9, 0, PixelFormat::Bgr24).unwrap();
    let mut workspace = ResizeWorkspace::default();
    let mut det = vec![0.; 3 * 32 * 32];
    preprocess::det(image, 32, 32, &mut det, &mut workspace).unwrap();
    assert_frozen(&det, include_bytes!("data/pixel-oracle/det.f32"), "DET");
    let mut cls = vec![0.; 3 * 80 * 160];
    preprocess::cls(image, &mut cls, &mut workspace).unwrap();
    assert_frozen(&cls, include_bytes!("data/pixel-oracle/cls.f32"), "CLS");
    let mut rec = vec![0.; 3 * 48 * 320];
    preprocess::rec(image, 320, &mut rec, &mut workspace).unwrap();
    assert_frozen(&rec, include_bytes!("data/pixel-oracle/rec.f32"), "REC");
    let crop = CropTransform::new(&DetectionBox {
        points: [[0., 0.], [6., 0.], [6., 8.], [0., 8.]],
        score: 1.,
    })
    .unwrap();
    let mut output = vec![0; crop.byte_count().unwrap()];
    crop.extract_into(image, &mut output).unwrap();
    assert_frozen(
        &output.into_iter().map(f32::from).collect::<Vec<_>>(),
        include_bytes!("data/pixel-oracle/crop.f32"),
        "crop",
    );
}
