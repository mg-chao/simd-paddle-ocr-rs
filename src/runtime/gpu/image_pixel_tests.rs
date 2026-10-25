//! Exact byte resize through the public resident recorder and CPU resize oracle.
use super::*;

#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    format: PixelFormat,
    image: [usize; 2],
    stride_extra: usize,
    source: [usize; 2],
    content_width: usize,
    output: [usize; 2],
    detector: bool,
}

const CASES: [Case; 8] = [
    Case {
        name: "bgr_down_odd_stride_padding",
        format: PixelFormat::Bgr24,
        image: [17, 13],
        stride_extra: 5,
        source: [17, 13],
        content_width: 9,
        output: [13, 7],
        detector: true,
    },
    Case {
        name: "rgb_up_odd_stride",
        format: PixelFormat::Rgb24,
        image: [7, 5],
        stride_extra: 1,
        source: [7, 5],
        content_width: 13,
        output: [17, 9],
        detector: false,
    },
    Case {
        name: "bgra_down_alpha_ignored",
        format: PixelFormat::Bgra32,
        image: [11, 7],
        stride_extra: 3,
        source: [11, 7],
        content_width: 7,
        output: [9, 5],
        detector: true,
    },
    Case {
        name: "rgba_up_tail",
        format: PixelFormat::Rgba32,
        image: [5, 3],
        stride_extra: 2,
        source: [5, 3],
        content_width: 17,
        output: [19, 11],
        detector: false,
    },
    Case {
        name: "virtual_source_zero_padding",
        format: PixelFormat::Bgr24,
        image: [3, 2],
        stride_extra: 7,
        source: [7, 5],
        content_width: 7,
        output: [11, 5],
        detector: true,
    },
    Case {
        name: "single_pixel_classifier_negative_padding",
        format: PixelFormat::Rgba32,
        image: [1, 1],
        stride_extra: 5,
        source: [1, 1],
        content_width: 1,
        output: [5, 80],
        detector: false,
    },
    Case {
        name: "coefficient_ties_both_even_directions",
        format: PixelFormat::Rgb24,
        image: [5, 3],
        stride_extra: 1,
        source: [5, 3],
        content_width: 2048,
        output: [2051, 3],
        detector: false,
    },
    Case {
        name: "wide_coordinate_product_overflows_u32",
        format: PixelFormat::Bgra32,
        image: [70001, 1],
        stride_extra: 3,
        source: [70001, 1],
        content_width: 70003,
        output: [70005, 1],
        detector: true,
    },
];

#[test]
fn image_pixel_dispatch_maps_2d_tails_once_to_each_nchw_channel() {
    for [height, width] in [[1, 1], [3, 5], [7, 13], [3, 2051], [80, 160], [1, 70005]] {
        let plane = height * width;
        let prefix = 3 * plane;
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: if plane > 17 * 17 * 64 { 127 } else { 17 },
            ..Default::default()
        };
        let (gx, gy) = dispatch_grid(plane, &limits).unwrap();
        let mut writes = vec![0u8; 9 * plane];
        for group_y in 0..gy as usize {
            for group_x in 0..gx as usize {
                for lane in 0..64 {
                    let pixel = group_x * 64 + lane + group_y * gx as usize * 64;
                    if pixel >= plane {
                        continue;
                    }
                    let [y, x] = [pixel / width, pixel % width];
                    for channel in 0..3 {
                        // Independent N,C,Y,X address with sample1 selected.
                        let index = ((3 + channel) * height + y) * width + x;
                        writes[index] += 1;
                    }
                }
            }
        }
        assert!(writes[..prefix].iter().all(|&count| count == 0));
        assert!(writes[prefix..2 * prefix].iter().all(|&count| count == 1));
        assert!(writes[2 * prefix..].iter().all(|&count| count == 0));
        if plane > 17 * 64 {
            assert!(gy > 1, "{height}x{width} must exercise a true2D grid");
        }
    }
    // These are semantic witnesses, not just ordinary resize dimensions.
    let wide = CASES
        .iter()
        .find(|case| case.name == "wide_coordinate_product_overflows_u32")
        .unwrap();
    assert!((2 * (wide.content_width as u64 - 1) + 1) * wide.source[0] as u64 > u32::MAX as u64);
    let mut tie_directions = [false; 2];
    for position in 0..2048u64 {
        let denominator = 4096;
        let remainder = ((2 * position + 1) * 5 + denominator - 2048) % denominator;
        let whole = remainder * 2048 / denominator;
        if remainder * 2048 % denominator * 2 == denominator {
            tie_directions[(whole & 1) as usize] = true;
        }
    }
    assert_eq!(tie_directions, [true; 2]);
}

fn source_pixels(case: Case) -> Vec<u8> {
    let [width, height] = case.image;
    let channels = case.format.bytes_per_pixel();
    let stride = width * channels + case.stride_extra;
    let mut pixels = vec![0xA5; stride * height];
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                pixels[y * stride + x * channels + c] =
                    ((x * 17 + y * 29 + c * 71 + x * y % 131 + 3) % 256) as u8;
            }
        }
    }
    pixels
}

fn cpu_reference(case: Case, pixels: &[u8]) -> Vec<f32> {
    let channels = case.format.bytes_per_pixel();
    let input_stride = case.image[0] * channels + case.stride_extra;
    let [source_width, source_height] = case.source;
    let stride = source_width * channels;
    // Materialize the shader's virtual padded source independently, then use
    // the existing CPU resize implementation, with its own coefficient cache,
    // horizontal row construction and final normalized-byte LUT.
    let mut padded = vec![0; stride * source_height];
    for y in 0..source_height.min(case.image[1]) {
        let count = source_width.min(case.image[0]) * channels;
        padded[y * stride..y * stride + count]
            .copy_from_slice(&pixels[y * input_stride..y * input_stride + count]);
    }
    let image = ImageView::new(&padded, source_width, source_height, stride, case.format).unwrap();
    let [width, height] = case.output;
    let mut resized = vec![0.; 3 * height * case.content_width];
    let mut workspace = crate::preprocess::ResizeWorkspace::default();
    if case.detector {
        crate::preprocess::det(
            image,
            case.content_width,
            height,
            &mut resized,
            &mut workspace,
        )
        .unwrap();
    } else {
        crate::preprocess::resize_for_gpu_test(
            image,
            case.content_width,
            height,
            case.content_width,
            &mut resized,
            &mut workspace,
        )
        .unwrap();
    }
    let padding = if !case.detector && height == 80 {
        -1.
    } else {
        0.
    };
    let mut expected = vec![padding; 3 * height * width];
    for c in 0..3 {
        for y in 0..height {
            let output = (c * height + y) * width;
            let source = (c * height + y) * case.content_width;
            expected[output..output + case.content_width]
                .copy_from_slice(&resized[source..source + case.content_width]);
        }
    }
    expected
}

fn hardware() {
    let context = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
    for precision in [GpuPrecision::F32, GpuPrecision::MixedF16] {
        for case in CASES {
            let [width, height] = case.image;
            let pixels = source_pixels(case);
            let view = ImageView::new(
                &pixels,
                width,
                height,
                width * case.format.bytes_per_pixel() + case.stride_extra,
                case.format,
            )
            .unwrap();
            let [width, height] = case.output;
            let count = 3 * width * height;
            let total = 3 * count + 11;
            let reference = cpu_reference(case, &pixels);
            let mut expected = vec![-777.; total];
            expected[count..2 * count].copy_from_slice(&reference);
            let mut initialized = vec![-777.; total];
            initialized[count..2 * count].fill(f32::NAN);
            let mut batch = Batch::new(context.clone()).unwrap();
            let image = Image::upload(&mut batch, view).unwrap();
            let output = Tensor {
                context: context.clone(),
                buffer: storage(
                    &context,
                    "pixel resize guarded output",
                    precision.bytes(total),
                )
                .unwrap(),
                descriptor: GpuTensorDescriptor::canonical(
                    Shape::new(&[3, 3, height, width]).unwrap(),
                    precision,
                ),
            };
            context
                .upload_resident(&output.buffer, &precision.encode(&initialized))
                .unwrap();
            preprocess_at(
                &mut batch,
                &image,
                &output,
                case.source,
                case.content_width,
                case.detector,
                1,
            )
            .unwrap();
            let (actual, statistics) = batch
                .read(&output.buffer, precision.bytes(total), false)
                .unwrap();
            let expected = precision.encode(&expected);
            assert_eq!(actual.len(), expected.len());
            if let Some(index) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
                panic!(
                    "Metal {precision:?} {}: byte{index}, actual{}, CPU{}; includes NCHW sample1/p12, sample0/2 and11 suffix guards",
                    case.name, actual[index], expected[index]
                );
            }
            assert_eq!(statistics.dispatches, 1, "{case:?}");
            assert_eq!(statistics.submissions, 1, "{case:?}");
            assert_eq!(statistics.native_resident_dispatches, 1);
            assert_eq!(statistics.diagnostic_readback_bytes, 0);
            assert!(statistics.cpu_compute_stages.is_empty());
        }
    }
}

#[test]
#[ignore = "requires a physical GPU with F16"]
fn gpu_resident_image_pixel_reuse_matches_cpu_and_batch_offset() {
    hardware();
}
