use super::*;
use crate::{cpu::CpuContext, image::PixelFormat};

#[test]
fn minimum_side_scaling_keeps_narrow_images_runnable() {
    let mut pipeline = crate::ocr::tests::synthetic_ocr(CpuContext::serial());
    pipeline
        .extractor_mut()
        .detector_mut()
        .set_resize_options(DetResizePolicy::Min, 736)
        .unwrap();
    for (width, height) in [(192, 32), (32, 192)] {
        let pixels = vec![255; width * height * 3];
        let image = ImageView::new(&pixels, width, height, 0, PixelFormat::Bgr24).unwrap();
        let result = pipeline
            .run(
                image,
                crate::RunOptions {
                    capture_spans: false,
                },
            )
            .unwrap();
        assert_eq!(result.detected_count, 1);
        assert!(!result.get(0).unwrap().text.is_empty());
        assert!(result.detector_resized_width <= 4096);
        assert!(result.detector_resized_height <= 4096);
    }
}

#[test]
fn implicit_padding_preserves_boxes_crops_trace_and_replay() {
    let mut pipeline = crate::ocr::tests::synthetic_ocr(CpuContext::serial());
    for (width, height) in [(46, 17), (17, 46), (17, 11)] {
        let padded_width = width.max(32);
        let padded_height = height.max(32);
        let mut pixels = vec![0; width * height * 3];
        let mut padded = vec![0; padded_width * padded_height * 3];
        for y in 3..height - 3 {
            for x in 3..width - 3 {
                pixels[(y * width + x) * 3..(y * width + x) * 3 + 3].fill(255);
            }
        }
        for y in 0..height {
            padded[y * padded_width * 3..y * padded_width * 3 + width * 3]
                .copy_from_slice(&pixels[y * width * 3..(y + 1) * width * 3]);
        }
        let image = ImageView::new(&pixels, width, height, 0, PixelFormat::Bgr24).unwrap();
        let control =
            ImageView::new(&padded, padded_width, padded_height, 0, PixelFormat::Bgr24).unwrap();
        let detector = pipeline.extractor_mut().detector_mut();
        let mut expected = detector.detect(control).unwrap().try_to_owned().unwrap();
        assert_eq!(expected.boxes.len(), 1);
        for point in &mut expected.boxes[0].points {
            point[0] = point[0].clamp(0., width as f32);
            point[1] = point[1].clamp(0., height as f32);
        }
        let actual = detector.detect(image).unwrap().try_to_owned().unwrap();
        assert_eq!(actual.boxes, expected.boxes, "{width}x{height}");
        assert_eq!(actual.width_ratio, expected.width_ratio);
        assert_eq!(actual.height_ratio, expected.height_ratio);

        let mut probabilities = Vec::new();
        let traced = detector
            .detect_with_trace(image, |event| {
                if let DetectionTrace::Probabilities { values, .. } = event {
                    probabilities.extend_from_slice(values);
                }
                Ok(())
            })
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert_eq!(traced.boxes, expected.boxes);
        let replayed = detector
            .postprocess(
                &probabilities,
                traced.resized_width,
                traced.resized_height,
                image,
                traced.width_ratio,
                traced.height_ratio,
            )
            .unwrap();
        assert_eq!(replayed.boxes, expected.boxes);

        // The same source-space box must produce the same recognition pixels,
        // whether minimum-size padding was implicit or supplied by the caller.
        let actual_crop = crate::crop::CropTransform::new(&actual.boxes[0]).unwrap();
        let expected_crop = crate::crop::CropTransform::new(&expected.boxes[0]).unwrap();
        let mut actual_pixels = vec![0; actual_crop.byte_count().unwrap()];
        let mut expected_pixels = vec![0; expected_crop.byte_count().unwrap()];
        actual_crop.extract_into(image, &mut actual_pixels).unwrap();
        expected_crop
            .extract_into(control, &mut expected_pixels)
            .unwrap();
        assert_eq!(actual_pixels, expected_pixels);
    }
}
