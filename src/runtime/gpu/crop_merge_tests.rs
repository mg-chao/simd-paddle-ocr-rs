use super::*;

fn bytes(rows: &[[u32; 6]]) -> Vec<u8> {
    rows.iter()
        .flat_map(|row| row.iter().flat_map(|word| word.to_le_bytes()))
        .collect()
}

#[test]
fn crop_metadata_preserves_dimensions_limits_and_error_order() {
    let metadata = bytes(&[[12, 10, 12, 10, 0, 0], [8, 20, 20, 8, 1, 0]]);
    let options = crate::lines::LineOptions {
        max_crop_pixels: 160,
        max_total_crop_bytes: 840,
        ..Default::default()
    };
    let shapes = parse_crop_metadata(&metadata, 2, options).unwrap();
    assert_eq!(shapes.sizes, [[12, 10], [8, 20]]);
    assert_eq!(shapes.original, [[12, 10, 0], [20, 8, 1]]);
    for (options, expected) in [
        (
            crate::lines::LineOptions {
                max_crop_pixels: 159,
                ..options
            },
            "crop exceeds max_crop_pixels",
        ),
        (
            crate::lines::LineOptions {
                max_total_crop_bytes: 839,
                ..options
            },
            "crops exceed max_total_crop_bytes",
        ),
    ] {
        assert_eq!(
            parse_crop_metadata(&metadata, 2, options)
                .err()
                .unwrap()
                .to_string(),
            expected
        );
    }
    let restrictive = crate::lines::LineOptions {
        max_crop_pixels: 0,
        max_total_crop_bytes: 0,
        ..Default::default()
    };
    for row in [
        [0, 10, 0, 10, 0, 0],
        [10, 0, 10, 0, 0, 0],
        [10, 10, 10, 10, 0, 1],
    ] {
        assert_eq!(
            parse_crop_metadata(&bytes(&[row]), 1, restrictive)
                .err()
                .unwrap()
                .to_string(),
            "degenerate GPU perspective crop"
        );
    }
    assert!(
        parse_crop_metadata(&[], 0, restrictive)
            .unwrap()
            .sizes
            .is_empty()
    );
    assert!(parse_crop_metadata(&metadata[..23], 1, options).is_err());
    assert!(parse_crop_metadata(&metadata, 1, options).is_err());
    // Rejected capacity rows may contain arbitrary metadata; only the accepted
    // prefix is passed to the shared validator by the merged readback path.
    let mut capacity = metadata.clone();
    capacity.extend_from_slice(&bytes(&[[0, 0, 0, 0, 0, 1]]));
    assert_eq!(
        parse_crop_metadata(&capacity[..48], 2, options)
            .unwrap()
            .sizes,
        shapes.sizes
    );
}

#[test]
fn crop_metadata_capacity_falls_back_for_quota_limits_and_overflow() {
    assert!(merged_crop_layout(0, usize::MAX, false).is_none());
    assert!(merged_crop_layout(17, usize::MAX, true).is_none());
    assert!(merged_crop_layout(usize::MAX, usize::MAX, false).is_none());
    assert!(merged_crop_layout(1, 151, false).is_none());
    let one = merged_crop_layout(1, 152, false).unwrap();
    assert_eq!(
        (
            one.box_bytes,
            one.metadata_bytes,
            one.transform_bytes,
            one.total_bytes
        ),
        (128, 24, 128, 152)
    );
    assert!(merged_crop_layout(17, 2175, false).is_none());
    let many = merged_crop_layout(17, 2176, false).unwrap();
    assert_eq!(
        (
            many.box_bytes,
            many.metadata_bytes,
            many.transform_bytes,
            many.total_bytes
        ),
        (1152, 408, 2176, 1560)
    );
}

fn hardware(quota: bool) -> Arc<GpuContext> {
    let context = GpuContext::new_with_options(GpuContextOptions {
        max_device_bytes: quota.then_some(256 << 20),
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(
        context.adapter_info().device_type,
        crate::runtime::gpu::GpuDeviceType::DiscreteGpu
            | crate::runtime::gpu::GpuDeviceType::IntegratedGpu
    ));
    context
}

fn prediction(context: &Arc<GpuContext>, precision: GpuPrecision, case: usize) -> Tensor {
    let mut map = vec![0.0f32; 96 * 96];
    if case != 0 {
        for y in 10..27 {
            for x in 10..39 {
                map[y * 96 + x] = 0.9;
            }
        }
        for y in 50..67 {
            for x in 52..81 {
                map[y * 96 + x] = 0.5;
            }
        }
    }
    let tensor = Tensor::allocate(
        context.clone(),
        Shape::new(&[1, 1, 96, 96]).unwrap(),
        precision,
    )
    .unwrap();
    if precision == GpuPrecision::F32 {
        context
            .upload_resident(&tensor.buffer, bytemuck::cast_slice(&map))
            .unwrap();
    } else {
        let values: Vec<_> = map
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        context
            .upload_resident(&tensor.buffer, bytemuck::cast_slice(&values))
            .unwrap();
    }
    tensor
}

fn batch(context: &Arc<GpuContext>, profiling: bool) -> Batch {
    let mut batch = Batch::new(context.clone()).unwrap();
    batch.profiling = profiling;
    batch
}

#[test]
#[ignore = "requires a physical GPU"]
fn gpu_resident_db_crop_merge_matches_standalone_coefficients_and_pixels() {
    let context = hardware(false);
    let geometry = DetSourceGeometry::new(96, 96).unwrap();
    let pixels: Vec<u8> = (0..96 * 96 * 3)
        .map(|i| ((i * 53 + i / 7) % 256) as u8)
        .collect();
    let source = ImageView::new(&pixels, 96, 96, 96 * 3, crate::image::PixelFormat::Bgr24).unwrap();
    let image = Image::upload(&mut Batch::new(context.clone()).unwrap(), source).unwrap();
    for precision in [GpuPrecision::F32, GpuPrecision::MixedF16] {
        for (case, threshold, profiling) in [
            (0, 0.8, false),
            (1, 0.95, false),
            (1, 0.8, false),
            (1, 0.8, true),
        ] {
            let prediction = prediction(&context, precision, case);
            let options = DetectorOptions {
                box_threshold: Some(threshold),
                max_candidates: 100,
                ..Default::default()
            };
            let (standalone_boxes, standalone_buffer, standalone_stats) =
                postprocess(batch(&context, profiling), &prediction, options, geometry).unwrap();
            let standalone = Detection {
                image: image.clone(),
                boxes: standalone_boxes,
                device_boxes: standalone_buffer,
                size: geometry
                    .resize(96, crate::preprocess::DetResizePolicy::Max)
                    .unwrap(),
                statistics: standalone_stats,
            };
            let standalone_crops = (!standalone.boxes.is_empty())
                .then(|| Crops::prepare(&standalone, Default::default()).unwrap());
            // Keep all standalone views alive across later allocations and both merged
            // submissions; the allocation reservation must prevent clobbering.
            let (boxes, _, stats, crops) = postprocess_inner(
                batch(&context, profiling),
                &prediction,
                options,
                geometry,
                true,
            )
            .unwrap();
            assert_eq!(boxes.len(), standalone.boxes.len());
            for (a, b) in boxes.iter().zip(&standalone.boxes) {
                assert_eq!(a.score.to_bits(), b.score.to_bits());
                assert!(
                    a.points
                        .iter()
                        .flatten()
                        .zip(b.points.iter().flatten())
                        .all(|(x, y)| x.to_bits() == y.to_bits())
                );
            }
            assert_eq!(stats.submissions, standalone.statistics.submissions);
            assert_eq!(
                stats.output_readback_bytes,
                standalone.statistics.output_readback_bytes
            );
            assert!(stats.cpu_compute_stages.is_empty());
            if case == 0 {
                assert_eq!(stats.submissions, 1);
                assert!(!stats.kernels.iter().any(|n| n == "perspective_transform"));
            } else {
                assert_eq!(stats.submissions, 2);
                let capacity = stats.output_readback_bytes as usize / 64 - 1;
                assert_eq!(stats.control_readback_bytes, 16 + (capacity * 24) as u64);
            }
            let (Some(old), Some(new)) = (standalone_crops, crops) else {
                assert!(boxes.is_empty());
                continue;
            };
            let new = new.finish(Default::default()).unwrap();
            assert_eq!(old.sizes, new.sizes);
            assert_eq!(old.original, new.original);
            assert_eq!(new.statistics.submissions, 0);
            assert_eq!(new.statistics.control_readback_bytes, 0);
            let bytes = boxes.len() * 128;
            let mut compare = batch(&context, false);
            let output = compare
                .storage("both crop coefficient sets", bytes * 2)
                .unwrap();
            compare.copy(&old.transforms, 0, &output, 0, bytes as u64);
            compare.copy(&new.transforms, 0, &output, bytes as u64, bytes as u64);
            let (coefficients, _) = compare.read(&output, bytes * 2, true).unwrap();
            assert_eq!(&coefficients[..bytes], &coefficients[bytes..]);
            let mut crop_batch = batch(&context, false);
            let crop = new.record(&mut crop_batch, &image, 0).unwrap();
            let (actual, _) = crop_batch
                .read(&crop.buffer, crop.width * crop.height * 4, false)
                .unwrap();
            let transform = crate::crop::CropTransform::new(&boxes[0]).unwrap();
            let mut expected = vec![0; transform.byte_count().unwrap()];
            transform.extract_into(source, &mut expected).unwrap();
            for (a, b) in actual
                .as_chunks::<4>()
                .0
                .iter()
                .zip(expected.as_chunks::<3>().0)
            {
                assert_eq!(&a[..3], b);
            }
        }
    }
}

#[test]
#[ignore = "requires a physical GPU"]
fn gpu_resident_crop_header_guard_preserves_explicit_count_and_capacity_tails() {
    let context = hardware(false);
    let capacity = 3;
    let mut words = vec![0u32; (capacity + 1) * 16];
    words[0] = 1;
    let points = [
        [3.25f32, 4.5],
        [36.125, 6.75],
        [30.5, 38.25],
        [6.75, 32.125],
    ];
    for (i, value) in points.iter().flatten().enumerate() {
        words[16 + i] = value.to_bits();
    }
    for word in &mut words[32..] {
        *word = f32::NAN.to_bits();
    }
    let boxes = context
        .create_buffer_init(
            "guarded box capacity",
            bytemuck::cast_slice(&words),
            BufferUsage::STORAGE,
        )
        .unwrap();
    let merged = storage(&context, "guarded merged transforms", capacity * 128).unwrap();
    context
        .upload_resident(
            &merged,
            bytemuck::cast_slice(&vec![0xdeadbeefu32; capacity * 32]),
        )
        .unwrap();
    let mut work = Batch::new(context.clone()).unwrap();
    record_crop_transforms(&mut work, &boxes, &merged, capacity, true).unwrap();
    let (merged_bytes, _) = work.read(&merged, capacity * 128, true).unwrap();
    assert!(
        merged_bytes[128..]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|v| u32::from_le_bytes(*v) == 0xdeadbeef)
    );
    // Standalone callers do not need to populate the header at all.
    words[0] = 0;
    let standalone_boxes = context
        .create_buffer_init(
            "standalone explicit box count",
            bytemuck::cast_slice(&words),
            BufferUsage::STORAGE,
        )
        .unwrap();
    let standalone = storage(&context, "standalone transforms", 128).unwrap();
    context
        .upload_resident(&standalone, bytemuck::cast_slice(&[0xdeadbeefu32; 32]))
        .unwrap();
    let mut work = Batch::new(context.clone()).unwrap();
    record_crop_transforms(&mut work, &standalone_boxes, &standalone, 1, false).unwrap();
    let (standalone_bytes, _) = work.read(&standalone, 128, true).unwrap();
    assert_eq!(&merged_bytes[..128], &standalone_bytes);
}

#[test]
#[ignore = "requires a physical GPU"]
fn gpu_resident_merged_crop_limits_and_device_quota_preserve_standalone_path() {
    let geometry = DetSourceGeometry::new(96, 96).unwrap();
    for quota in [false, true] {
        let context = hardware(quota);
        let prediction = prediction(&context, GpuPrecision::F32, 1);
        let options = DetectorOptions {
            box_threshold: Some(0.8),
            max_candidates: 100,
            ..Default::default()
        };
        let (boxes, device_boxes, statistics) =
            postprocess(batch(&context, false), &prediction, options, geometry).unwrap();
        assert!(
            !statistics
                .kernels
                .iter()
                .any(|name| name == "perspective_transform")
        );
        let image = Image {
            context: context.clone(),
            buffer: storage(&context, "source image", 96 * 96 * 4).unwrap(),
            width: 96,
            height: 96,
            stride: 96 * 4,
            format: crate::image::PixelFormat::Bgra32,
        };
        let detection = Detection {
            image,
            boxes,
            device_boxes,
            size: geometry
                .resize(96, crate::preprocess::DetResizePolicy::Max)
                .unwrap(),
            statistics,
        };
        for limits in [
            crate::lines::LineOptions {
                max_crop_pixels: 1,
                ..Default::default()
            },
            crate::lines::LineOptions {
                max_total_crop_bytes: 1,
                ..Default::default()
            },
        ] {
            let expected = Crops::prepare(&detection, limits).err().unwrap();
            let merged =
                postprocess_inner(batch(&context, false), &prediction, options, geometry, true)
                    .unwrap();
            if quota {
                let (_, _, stats, crops) = merged;
                assert!(crops.is_none());
                assert_eq!(stats.control_readback_bytes, 16);
            } else {
                // Metadata preparation does not validate crop quotas early;
                // Full OCR first gets to enforce its result-byte budget.
                let (_, _, _, crops) = merged;
                let actual = crops.unwrap().finish(limits).err().unwrap();
                assert_eq!(actual.kind(), expected.kind());
                assert_eq!(actual.to_string(), expected.to_string());
            }
        }
    }
}
