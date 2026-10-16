//! Model-independent public pixel/crop contracts, required on every CI run.
use simd_paddle_ocr_rs::{
    crop::{CropTransform, rotate180},
    detector::DetectionBox,
    image::{ImageView, PixelFormat},
    preprocess::{self, ResizeWorkspace},
};

fn encoded(format: PixelFormat, width: usize, height: usize, padding: usize) -> (Vec<u8>, usize) {
    let bytes = format.bytes_per_pixel();
    let stride = width * bytes + padding;
    // The final row intentionally has no padding, as ImageView permits.
    let mut pixels = vec![0xE7; (height - 1) * stride + width * bytes];
    for y in 0..height {
        for x in 0..width {
            let b = ((x * 17 + y * 3) % 256) as u8;
            let g = ((x * 5 + y * 29) % 256) as u8;
            let r = ((x * 31 + y * 7) % 256) as u8;
            let offset = y * stride + x * bytes;
            let channels = if matches!(format, PixelFormat::Rgb24 | PixelFormat::Rgba32) {
                [r, g, b]
            } else {
                [b, g, r]
            };
            pixels[offset..offset + 3].copy_from_slice(&channels);
            if bytes == 4 {
                pixels[offset + 3] = (x ^ y) as u8;
            }
        }
    }
    (pixels, stride)
}

fn normalized(image: ImageView<'_>) -> Vec<Vec<u32>> {
    let mut workspace = ResizeWorkspace::default();
    let mut det = vec![0.; 3 * 32 * 64];
    let mut cls = vec![0.; 3 * 80 * 160];
    let mut rec = vec![0.; 3 * 48 * 96];
    preprocess::det(image, 64, 32, &mut det, &mut workspace).unwrap();
    preprocess::cls(image, &mut cls, &mut workspace).unwrap();
    preprocess::rec(image, 96, &mut rec, &mut workspace).unwrap();
    [det, cls, rec]
        .into_iter()
        .map(|values| values.into_iter().map(f32::to_bits).collect())
        .collect()
}

#[test]
fn formats_alpha_and_noncontiguous_rows_have_identical_normalized_pixels() {
    let (pixels, stride) = encoded(PixelFormat::Bgr24, 17, 11, 0);
    let expected = normalized(ImageView::new(&pixels, 17, 11, stride, PixelFormat::Bgr24).unwrap());
    for format in [
        PixelFormat::Bgr24,
        PixelFormat::Rgb24,
        PixelFormat::Bgra32,
        PixelFormat::Rgba32,
    ] {
        let (pixels, stride) = encoded(format, 17, 11, 9);
        let source = ImageView::new(&pixels, 17, 11, stride, format).unwrap();
        assert_eq!(normalized(source), expected, "{format:?}");
    }
}

#[test]
fn recognition_zero_padding_and_classifier_black_padding_are_distinct() {
    let pixels = vec![255; 3 * 5 * 20];
    let image = ImageView::new(&pixels, 5, 20, 0, PixelFormat::Bgr24).unwrap();
    let mut workspace = ResizeWorkspace::default();
    let mut cls = vec![f32::NAN; 3 * 80 * 160];
    let mut rec = vec![f32::NAN; 3 * 48 * 64];
    assert_eq!(
        preprocess::cls(image, &mut cls, &mut workspace).unwrap(),
        20
    );
    assert_eq!(
        preprocess::rec(image, 64, &mut rec, &mut workspace).unwrap(),
        12
    );
    for channel in 0..3 {
        for row in 0..80 {
            assert!(
                cls[channel * 80 * 160 + row * 160..channel * 80 * 160 + row * 160 + 20]
                    .iter()
                    .all(|&x| x == 1.)
            );
            assert!(
                cls[channel * 80 * 160 + row * 160 + 20..channel * 80 * 160 + (row + 1) * 160]
                    .iter()
                    .all(|&x| x == -1.)
            );
        }
        for row in 0..48 {
            assert!(
                rec[channel * 48 * 64 + row * 64..channel * 48 * 64 + row * 64 + 12]
                    .iter()
                    .all(|&x| x == 1.)
            );
            assert!(
                rec[channel * 48 * 64 + row * 64 + 12..channel * 48 * 64 + (row + 1) * 64]
                    .iter()
                    .all(|&x| x.to_bits() == 0)
            );
        }
    }
}

#[test]
fn classifier_truncates_a_wide_source_to_the_historical_left_window() {
    let (pixels, stride) = encoded(PixelFormat::Bgr24, 101, 10, 7);
    let full = ImageView::new(&pixels, 101, 10, stride, PixelFormat::Bgr24).unwrap();
    let left = ImageView::new(&pixels, 40, 10, stride, PixelFormat::Bgr24).unwrap();
    let mut workspace = ResizeWorkspace::default();
    let mut a = vec![0.; 3 * 80 * 160];
    let mut b = a.clone();
    assert_eq!(preprocess::cls(full, &mut a, &mut workspace).unwrap(), 160);
    preprocess::cls(left, &mut b, &mut workspace).unwrap();
    assert_eq!(a, b);
}

#[test]
fn perspective_crop_format_conversion_and_sentinel_bounds_are_exact() {
    let crop = CropTransform::new(&DetectionBox {
        points: [[1.25, 2.], [14., 1.5], [15., 9.], [0.5, 10.]],
        score: 1.,
    })
    .unwrap();
    let mut expected = None;
    for format in [
        PixelFormat::Bgr24,
        PixelFormat::Rgb24,
        PixelFormat::Bgra32,
        PixelFormat::Rgba32,
    ] {
        let (pixels, stride) = encoded(format, 17, 12, 13);
        let image = ImageView::new(&pixels, 17, 12, stride, format).unwrap();
        let count = crop.byte_count().unwrap();
        let mut output = vec![0xAA; count + 23];
        crop.extract_into(image, &mut output).unwrap();
        assert_eq!(&output[count..], &[0xAA; 23]);
        let active = output[..count].to_vec();
        if let Some(expected) = &expected {
            assert_eq!(&active, expected);
        } else {
            expected = Some(active);
        }
    }
}

#[test]
fn rotation_reverses_pixel_order_and_is_an_involution_for_odd_and_even_counts() {
    for (width, height) in [(1, 1), (3, 3), (4, 3), (1, 7), (8, 1)] {
        let original: Vec<_> = (0..width * height)
            .flat_map(|i| [i as u8, (i + 20) as u8, (i + 40) as u8])
            .collect();
        let mut pixels = original.clone();
        rotate180(&mut pixels, width, height).unwrap();
        for (a, b) in pixels
            .as_chunks::<3>()
            .0
            .iter()
            .zip(original.as_chunks::<3>().0.iter().rev())
        {
            assert_eq!(a, b);
        }
        rotate180(&mut pixels, width, height).unwrap();
        assert_eq!(pixels, original);
    }
}

#[test]
fn invalid_images_and_degenerate_crops_fail_before_writing() {
    assert!(ImageView::new(&[0; 8], 3, 1, 0, PixelFormat::Bgr24).is_err());
    assert!(ImageView::new(&[0; 64], 3, 2, 8, PixelFormat::Bgr24).is_err());
    assert!(
        CropTransform::new(&DetectionBox {
            points: [[0., 0.]; 4],
            score: 1.
        })
        .is_err()
    );
    assert!(
        CropTransform::new(&DetectionBox {
            points: [[f32::NAN, 0.], [2., 0.], [2., 2.], [0., 2.]],
            score: 1.
        })
        .is_err()
    );
    let mut output = [9; 6];
    assert!(rotate180(&mut output, 3, 2).is_err());
    assert_eq!(output, [9; 6]);
}
