//! GPU image stages and canonical DB quadrilaterals.
use super::{resident::*, *};
use crate::{
    detector::{DetectionBox, DetectorOptions},
    image::ImageView,
    preprocess::{DetSize, DetSourceGeometry},
};

pub(crate) struct Detection {
    pub scratch: Option<Arc<ScratchPool>>,
    pub image: Image,
    pub boxes: Vec<DetectionBox>,
    pub device_boxes: GpuBuffer,
    pub size: DetSize,
    pub statistics: GpuPipelineStatistics,
}
pub(crate) struct Detector {
    pub scratch: Arc<ScratchPool>,
    pub session: GpuSession,
    pub context: Arc<GpuContext>,
}
impl Detector {
    pub fn new(compiled: Arc<GpuCompiledModel>, limits: ExecutionLimits) -> RuntimeResult<Self> {
        let compiled = compiled.with_full_precision()?;
        compiled.prepare_images()?;
        Ok(Self {
            scratch: Arc::default(),
            context: compiled.context.clone(),
            session: GpuSession::new_resident(compiled, &[1, 3, 32, 32], limits, None)?,
        })
    }
    pub fn run(
        &mut self,
        image: ImageView<'_>,
        options: DetectorOptions,
        profiling: bool,
    ) -> RuntimeResult<Detection> {
        if image.pixel_count().map_err(RuntimeError::from)? > options.max_image_pixels {
            return Err(resource("source image exceeds max_image_pixels"));
        }
        let geometry =
            DetSourceGeometry::new(image.width(), image.height()).map_err(RuntimeError::from)?;
        let size = geometry
            .resize(options.limit_side_length, options.resize_policy)
            .map_err(RuntimeError::from)?;
        self.session.reshape(&[1, 3, size.height, size.width])?;
        if self.session.output_shape().dimensions() != [1, 1, size.height, size.width] {
            return Err(RuntimeError::new(
                "detector output shape differs from input",
            ));
        }
        let mut batch = Batch::new(self.context.clone())?;
        batch.profiling = profiling;
        batch.scratch = Some(self.scratch.clone());
        let image = Image::upload(&mut batch, image)?;
        preprocess(
            &mut batch,
            &image,
            &self.session.resident_input(),
            [geometry.padded_width, geometry.padded_height],
            size.width,
            true,
        )?;
        let prediction = batch.graph(&mut self.session, "detector")?;
        let (boxes, device_boxes, statistics) = postprocess(batch, &prediction, options, geometry)?;
        Ok(Detection {
            scratch: Some(self.scratch.clone()),
            image,
            boxes,
            device_boxes,
            size,
            statistics,
        })
    }
}

pub(crate) fn merge_statistics(
    destination: &mut GpuPipelineStatistics,
    source: GpuPipelineStatistics,
) {
    if destination.adapter.is_empty() {
        destination.adapter = source.adapter;
        destination.backend = source.backend;
        destination.driver = source.driver;
        destination.driver_info = source.driver_info;
    }
    destination.image_processing = destination.image_processing.merge(source.image_processing);
    destination.geometry = destination.geometry.merge(source.geometry);
    destination.projection = destination.projection.merge(source.projection);
    destination.decoding = destination.decoding.merge(source.decoding);
    destination.uploaded_bytes += source.uploaded_bytes;
    destination.output_readback_bytes += source.output_readback_bytes;
    destination.control_readback_bytes += source.control_readback_bytes;
    destination.diagnostic_readback_bytes += source.diagnostic_readback_bytes;
    destination.submissions += source.submissions;
    destination.peak_device_bytes = destination.peak_device_bytes.max(source.peak_device_bytes);
    for name in source.kernels {
        if !destination.kernels.contains(&name) {
            destination.kernels.push(name);
        }
    }
    for name in source.cpu_compute_stages {
        if !destination.cpu_compute_stages.contains(&name) {
            destination.cpu_compute_stages.push(name);
        }
    }
    destination.submission_ns += source.submission_ns;
    destination.completion_wait_ns += source.completion_wait_ns;
    destination.readback_ns += source.readback_ns;
    destination.projection_encoding_ns += source.projection_encoding_ns;
    destination.encoding_ns += source.encoding_ns;
    destination.admission_wait_ns += source.admission_wait_ns;
    destination.dispatches += source.dispatches;
    destination.layout_conversions += source.layout_conversions;
    destination.scratch_allocations += source.scratch_allocations;
    destination.scratch_allocated_bytes += source.scratch_allocated_bytes;
    destination.scratch_reuses += source.scratch_reuses;
    destination.profiling_readback_bytes += source.profiling_readback_bytes;
    if let Some(profile) = source.profile {
        destination
            .profile
            .get_or_insert_with(Default::default)
            .kernels
            .extend(profile.kernels);
    }
}

pub(crate) struct Crops {
    pub transforms: GpuBuffer,
    pub sizes: Vec<[usize; 2]>,
    pub original: Vec<[usize; 3]>,
    pub statistics: GpuPipelineStatistics,
}
impl Crops {
    pub fn prepare(
        detection: &Detection,
        options: crate::lines::LineOptions,
    ) -> RuntimeResult<Self> {
        let context = detection.image.context.clone();
        let count = detection.boxes.len();
        let transforms = storage(&context, "perspective transforms", count.max(1) * 128)?;
        if count == 0 {
            return Ok(Self {
                transforms,
                sizes: Vec::new(),
                original: Vec::new(),
                statistics: Default::default(),
            });
        }
        let mut batch = Batch::new(context)?;
        batch.profiling = detection.statistics.profile.is_some();
        batch.scratch = detection.scratch.clone();
        let source = format!(
            "{PARAMETERS}{}{}",
            shader("precise.wgsl", include_str!("precise.wgsl")),
            shader("crop_meta.wgsl", include_str!("crop_meta.wgsl"))
        );
        batch.dispatch(
            "perspective_transform",
            &source,
            &[(&detection.device_boxes, true), (&transforms, false)],
            &[count as u32],
            [(count as u32).div_ceil(64), 1, 1],
        )?;
        // Only shape/validation metadata crosses the scheduling boundary. Coefficients stay on device.
        let metadata = batch.storage("crop shape metadata", count * 24)?;
        for index in 0..count {
            batch.copy(
                &transforms,
                (index * 128) as u64,
                &metadata,
                (index * 24) as u64,
                24,
            );
        }
        let (bytes, statistics) = batch.read(&metadata, count * 24, true)?;
        let mut sizes = Vec::with_capacity(count);
        let mut original = Vec::with_capacity(count);
        let mut total = 0usize;
        for row in bytes.as_chunks::<24>().0 {
            let word =
                |i: usize| u32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
            let (w, h) = (word(0), word(1));
            if word(5) != 0 || w == 0 || h == 0 {
                return Err(RuntimeError::new("degenerate GPU perspective crop"));
            }
            let pixels = w
                .checked_mul(h)
                .filter(|&n| n <= options.max_crop_pixels)
                .ok_or_else(|| resource("crop exceeds max_crop_pixels"))?;
            total = total
                .checked_add(
                    pixels
                        .checked_mul(3)
                        .ok_or_else(|| resource("crop byte overflow"))?,
                )
                .filter(|&n| n <= options.max_total_crop_bytes)
                .ok_or_else(|| resource("crops exceed max_total_crop_bytes"))?;
            sizes.push([w, h]);
            original.push([word(2), word(3), word(4)]);
        }
        Ok(Self {
            transforms,
            sizes,
            original,
            statistics,
        })
    }
    pub fn record(&self, batch: &mut Batch, source: &Image, index: usize) -> RuntimeResult<Image> {
        batch.ensure_context(&source.context)?;
        let [width, height] = self.sizes[index];
        let pixels = width * height;
        let output = batch.storage(
            "perspective crop BGRA",
            pixels
                .checked_mul(4)
                .ok_or_else(|| resource("crop storage overflow"))?,
        )?;
        let shader = format!(
            "{PARAMETERS}{}{}",
            shader("precise.wgsl", include_str!("precise.wgsl")),
            shader("crop.wgsl", include_str!("crop.wgsl"))
        );
        let (x, y) = dispatch_grid(pixels, &batch.context.limits)?;
        let swap = matches!(
            source.format,
            crate::image::PixelFormat::Rgb24 | crate::image::PixelFormat::Rgba32
        );
        batch.dispatch(
            "perspective_bicubic",
            &shader,
            &[
                (&source.buffer, true),
                (&self.transforms, true),
                (&output, false),
            ],
            &[
                source.width as u32,
                source.height as u32,
                source.stride as u32,
                source.format.bytes_per_pixel() as u32,
                u32::from(swap),
                index as u32,
            ],
            [x, y, 1],
        )?;
        Ok(Image {
            context: batch.context.clone(),
            buffer: output,
            width,
            height,
            stride: width * 4,
            format: crate::image::PixelFormat::Bgra32,
        })
    }
}

pub(crate) fn pack_bgr(batch: &mut Batch, image: &Image) -> RuntimeResult<GpuBuffer> {
    let bytes = image.width * image.height * 3;
    let output = batch.storage("requested BGR crop", bytes)?;
    let source = format!(
        "{PARAMETERS}{}",
        shader("pack_bgr.wgsl", include_str!("pack_bgr.wgsl"))
    );
    let (x, y) = dispatch_grid(bytes.div_ceil(4), &batch.context.limits)?;
    batch.dispatch(
        "image_pack_bgr",
        &source,
        &[(&image.buffer, true), (&output, false)],
        &[bytes as u32],
        [x, y, 1],
    )?;
    Ok(output)
}

pub(crate) fn orient(
    batch: &mut Batch,
    image: &Image,
    winner: &GpuBuffer,
    threshold: f32,
) -> RuntimeResult<Image> {
    orient_at(batch, image, winner, 0, threshold)
}
pub(crate) fn orient_at(
    batch: &mut Batch,
    image: &Image,
    winners: &GpuBuffer,
    row: usize,
    threshold: f32,
) -> RuntimeResult<Image> {
    if row
        .checked_add(1)
        .and_then(|n| n.checked_mul(16))
        .is_none_or(|n| n as u64 > winners.size())
    {
        return Err(RuntimeError::new("classifier winner row out of range"));
    }
    let output = batch.storage("oriented crop", image.width * image.height * 4)?;
    let source = format!(
        "{PARAMETERS}{}",
        shader("rotate.wgsl", include_str!("rotate.wgsl"))
    );
    let (x, y) = dispatch_grid(image.width * image.height, &batch.context.limits)?;
    batch.dispatch(
        "classifier_rotation",
        &source,
        &[(&image.buffer, true), (winners, true), (&output, false)],
        &[
            (image.width * image.height) as u32,
            threshold.to_bits(),
            row as u32,
        ],
        [x, y, 1],
    )?;
    Ok(Image {
        buffer: output,
        ..image.clone()
    })
}
pub(crate) fn postprocess(
    mut batch: Batch,
    prediction: &Tensor,
    options: DetectorOptions,
    geometry: DetSourceGeometry,
) -> RuntimeResult<(Vec<DetectionBox>, GpuBuffer, GpuPipelineStatistics)> {
    batch.ensure_context(&prediction.context)?;
    let profiling = batch.profiling;
    let scratch = batch.scratch.clone();
    let context = batch.context.clone();
    let dims = prediction.descriptor.shape.dimensions();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 1 {
        return Err(RuntimeError::new("invalid GPU DB map shape"));
    }
    let (h, w) = (dims[2], dims[3]);
    let pixels = w * h;
    let candidates = options.max_candidates;
    let mask = batch.storage("DB bitmap", pixels * 4)?;
    let parents = batch.storage("DB component parents", pixels * 4)?;
    let counts = batch.storage("DB component counts", pixels * 4)?;
    let ranks = batch.storage("DB component ranks", pixels * 4)?;
    let block_count = pixels.div_ceil(64);
    let blocks = batch.storage(
        "DB prefix blocks",
        (block_count + block_count.div_ceil(64)) * 16,
    )?;
    let meta = batch.storage("DB candidate extents", (16 + candidates * 8) * 4)?;
    let precision = prediction.descriptor.precision;
    let source = format!(
        "{}{}{}",
        if precision == GpuPrecision::MixedF16 {
            "enable f16;\n"
        } else {
            ""
        },
        PARAMETERS,
        shader("db_label.wgsl", include_str!("db_label.wgsl")).replace(
            "$INPUT",
            if precision == GpuPrecision::MixedF16 {
                "f16"
            } else {
                "f32"
            }
        )
    );
    #[cfg(test)]
    let tiled = context.options.tiled_labels;
    #[cfg(not(test))]
    let tiled = true;
    let (gx, gy) = dispatch_grid(pixels, &context.limits)?;
    for mode in [0, 1, 2, 3, 4, 9, 5, 6, 7, 8] {
        if mode == 1 && !tiled {
            continue;
        }
        let groups = if mode == 5 || mode == 8 {
            [1, 1, 1]
        } else if mode == 9 {
            let (x, y) = dispatch_grid(block_count, &context.limits)?;
            [x, y, 1]
        } else if mode == 1 {
            let (tx, ty) = dispatch_grid(w.div_ceil(8) * h.div_ceil(8) * 64, &context.limits)?;
            [tx, ty, 1]
        } else {
            [gx, gy, 1]
        };
        batch.dispatch(
            &format!("db_label_{precision:?}"),
            &source,
            &[
                (&prediction.buffer, true),
                (&mask, false),
                (&parents, false),
                (&counts, false),
                (&ranks, false),
                (&blocks, false),
                (&meta, false),
            ],
            &[
                w as u32,
                h as u32,
                options.bitmap_threshold.to_bits(),
                candidates as u32,
                mode,
                u32::from(options.use_dilation),
                u32::from(tiled),
            ],
            groups,
        )?;
    }
    let (metadata, mut statistics) = batch.read(&meta, 16, true)?;
    let word = |i: usize| u32::from_le_bytes(metadata[i * 4..i * 4 + 4].try_into().unwrap());
    if word(2) != 0 {
        return Err(RuntimeError::new("nonfinite GPU detector probability"));
    }
    let count = word(1) as usize;
    let row_count = word(3) as usize;
    if count > candidates || row_count > pixels {
        return Err(RuntimeError::new("invalid GPU component metadata"));
    }
    let mut batch = Batch::new(context.clone())?;
    batch.profiling = profiling;
    batch.scratch = scratch;
    let compact = batch.storage("DB sorted output", (count + 1) * 64)?;
    if count == 0 {
        return Ok((Vec::new(), compact, statistics));
    }
    let rows = batch.storage("DB row extremes", row_count * 8)?;
    let hull = batch.storage(
        "DB hull and offset scratch",
        (row_count * 4 + count * 1024) * 8,
    )?;
    let raw_boxes = batch.storage("DB candidate boxes", count * 64)?;
    let source = format!(
        "{PARAMETERS}{}",
        shader("db_rows.wgsl", include_str!("db_rows.wgsl"))
    );
    for (mode, length) in [(0, row_count), (1, pixels)] {
        let (x, y) = dispatch_grid(length, &context.limits)?;
        batch.dispatch(
            "db_row_extremes",
            &source,
            &[
                (&parents, true),
                (&ranks, true),
                (&meta, true),
                (&rows, false),
            ],
            &[w as u32, h as u32, mode],
            [x, y, 1],
        )?;
    }
    let source = format!(
        "{}{}{}",
        if precision == GpuPrecision::MixedF16 {
            "enable f16;\n"
        } else {
            ""
        },
        format_args!(
            "{PARAMETERS}{}",
            shader("precise.wgsl", include_str!("precise.wgsl"))
        ),
        shader("db_geometry.wgsl", include_str!("db_geometry.wgsl")).replace(
            "$INPUT",
            if precision == GpuPrecision::MixedF16 {
                "f16"
            } else {
                "f32"
            }
        )
    );
    batch.dispatch(
        &format!("db_geometry_{precision:?}"),
        &source,
        &[
            (&prediction.buffer, true),
            (&meta, true),
            (&rows, true),
            (&hull, false),
            (&raw_boxes, false),
        ],
        &[
            w as u32,
            h as u32,
            options.box_threshold.unwrap_or(0.6).to_bits(),
            options.unclip_ratio.to_bits(),
            geometry.original_width as u32,
            geometry.original_height as u32,
            geometry.padded_width as u32,
            geometry.padded_height as u32,
        ],
        [count as u32, 1, 1],
    )?;
    let source = format!(
        "{PARAMETERS}{}",
        shader("db_sort.wgsl", include_str!("db_sort.wgsl"))
    );
    batch.dispatch(
        "db_reading_order",
        &source,
        &[(&raw_boxes, true), (&compact, false)],
        &[count as u32],
        [1, 1, 1],
    )?;
    let (data, stats) = batch.read(&compact, (count + 1) * 64, false)?;
    merge_statistics(&mut statistics, stats);
    let word = |i: usize| u32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
    let accepted = word(0) as usize;
    if accepted > count {
        return Err(RuntimeError::new("invalid GPU detection count"));
    }
    let mut boxes = Vec::with_capacity(accepted);
    for i in 0..accepted {
        let base = (i + 1) * 16;
        let points = std::array::from_fn(|j| {
            [
                f32::from_bits(word(base + j * 2)),
                f32::from_bits(word(base + j * 2 + 1)),
            ]
        });
        let score = f32::from_bits(word(base + 8));
        if !score.is_finite() || points.iter().flatten().any(|v| !v.is_finite()) {
            return Err(RuntimeError::new("invalid GPU detection box"));
        }
        boxes.push(DetectionBox { points, score });
    }
    Ok((boxes, compact, statistics))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hardware() -> Arc<GpuContext> {
        let backend = match std::env::var("SIMD_OCR_GPU_BACKEND").as_deref() {
            Ok("dx12") => GpuBackendPreference::Dx12,
            Ok("metal") => GpuBackendPreference::Metal,
            _ => GpuBackendPreference::Vulkan,
        };
        let context = GpuContext::new_with_options(GpuContextOptions {
            backend,
            ..Default::default()
        })
        .unwrap();
        assert!(matches!(
            context.adapter_info().device_type,
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
        ));
        context
    }
    #[test]
    #[ignore = "requires a physical GPU"]
    fn gpu_resident_reading_order_equal_keys() {
        let context = hardware();
        for count in [16, 17, 33, 96] {
            let mut boxes = Vec::new();
            let mut words = vec![0u32; count * 16];
            for i in 0..count {
                let x = ((i * 7) % 5) as f32 * 13.;
                let y = ((i * 11) % 3) as f32 * 9.;
                let points = [[x, y], [x + 11., y], [x + 11., y + 5.], [x, y + 5.]];
                boxes.push(DetectionBox {
                    points,
                    score: i as f32,
                });
                for (k, v) in points.iter().flatten().enumerate() {
                    words[i * 16 + k] = v.to_bits();
                }
                words[i * 16 + 8] = (i as f32).to_bits();
                words[i * 16 + 9] = 1;
            }
            crate::db::sort_reading_order(&mut boxes);
            let raw = context
                .create_buffer_init(
                    "sort fixtures",
                    bytemuck::cast_slice(&words),
                    wgpu::BufferUsages::STORAGE,
                )
                .unwrap();
            let mut batch = Batch::new(context.clone()).unwrap();
            let output = batch.storage("sort output", (count + 1) * 64).unwrap();
            batch
                .dispatch(
                    "db_reading_order",
                    &format!("{PARAMETERS}{}", include_str!("db_sort.wgsl")),
                    &[(&raw, true), (&output, false)],
                    &[count as u32],
                    [1, 1, 1],
                )
                .unwrap();
            let (data, _) = batch.read(&output, (count + 1) * 64, false).unwrap();
            for (i, b) in boxes.iter().enumerate() {
                let offset = (i + 1) * 64 + 32;
                assert_eq!(
                    f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()),
                    b.score,
                    "count {count}, item {i}"
                );
            }
        }
    }
    #[test]
    #[ignore = "requires a physical GPU"]
    fn gpu_resident_db_geometry_contract() {
        let context = hardware();
        let (w, h) = (96, 96);
        let geometry = DetSourceGeometry::new(w, h).unwrap();
        for case in 0..9 {
            let mut map = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let yes = match case {
                        0 => false,
                        1 => (8..70).contains(&x) && (9..20).contains(&y),
                        2 => {
                            (6..85).contains(&x)
                                && (6..85).contains(&y)
                                && !((24..65).contains(&x) && (24..65).contains(&y))
                        }
                        3 => {
                            ((8..22).contains(&x) && (8..22).contains(&y))
                                || ((22..40).contains(&x) && (22..40).contains(&y))
                        }
                        4 => x % 24 < 13 && y % 24 < 9,
                        5 => x < 18 && y < 35,
                        6 => x > 8 && x < 80 && y > x / 3 + 8 && y < x / 3 + 23,
                        7 => x > 15 && x < 80 && y > 10 && y < 80 && !(x > 35 && x < 50 && y < 55),
                        _ => x > 40 && x < 45 && y > 10 && y < 80,
                    };
                    if yes {
                        map[y * w + x] = 0.9;
                    }
                }
            }
            for dilation in [false, true] {
                for max_candidates in [2, 100] {
                    let options = DetectorOptions {
                        box_threshold: Some(0.4),
                        use_dilation: dilation,
                        max_candidates,
                        ..Default::default()
                    };
                    let tensor = Tensor::allocate(
                        context.clone(),
                        Shape::new(&[1, 1, h, w]).unwrap(),
                        GpuPrecision::F32,
                    )
                    .unwrap();
                    context
                        .queue
                        .write_buffer(&tensor.buffer, 0, bytemuck::cast_slice(&map));
                    let (actual, _, statistics) = postprocess(
                        Batch::new(context.clone()).unwrap(),
                        &tensor,
                        options,
                        geometry,
                    )
                    .unwrap();
                    let mut expected = crate::db::Workspace::default();
                    crate::db::run(&map, w, h, &options, geometry, &mut expected).unwrap();
                    assert_eq!(
                        actual.len(),
                        expected.boxes.len(),
                        "case {case}, dilation {dilation}, candidates {max_candidates}"
                    );
                    for (a, b) in actual.iter().zip(&expected.boxes) {
                        assert!(
                            (a.score - b.score).abs() <= 1e-3 + 1e-3 * b.score.abs(),
                            "score case {case}: {a:?} vs {b:?}"
                        );
                        for (x, y) in a.points.iter().flatten().zip(b.points.iter().flatten()) {
                            assert!(
                                (x - y).abs() <= 0.5,
                                "coordinate case {case}: {a:?} vs {b:?}"
                            );
                        }
                    }
                    assert!(statistics.cpu_compute_stages.is_empty());
                    assert_eq!(statistics.diagnostic_readback_bytes, 0);
                }
            }
        }
    }
    #[test]
    #[ignore = "requires a physical GPU"]
    fn gpu_resident_perspective_matches_cpu_bytes() {
        let context = hardware();
        let quads = [
            [[2., 3.], [38., 3.], [38., 24.], [2., 24.]],
            [[4., 2.], [14., 2.], [14., 40.], [4., 40.]],
            [[-2.5, 1.25], [37.5, -2.], [40.25, 29.75], [1.5, 32.5]],
            [[3.25, 4.5], [36.125, 6.75], [30.5, 38.25], [6.75, 32.125]],
        ];
        for format in [
            crate::image::PixelFormat::Bgr24,
            crate::image::PixelFormat::Rgb24,
            crate::image::PixelFormat::Bgra32,
            crate::image::PixelFormat::Rgba32,
        ] {
            let stride = 43 * format.bytes_per_pixel() + 7;
            let pixels: Vec<u8> = (0..stride * 47)
                .map(|i| ((i * 53 + i / 7) % 256) as u8)
                .collect();
            let source = ImageView::new(&pixels, 43, 47, stride, format).unwrap();
            for points in quads {
                let box_ = DetectionBox { points, score: 1. };
                let mut batch = Batch::new(context.clone()).unwrap();
                let image = Image::upload(&mut batch, source).unwrap();
                let mut words = vec![0u32; 32];
                words[0] = 1;
                for (i, &v) in points.iter().flatten().enumerate() {
                    words[16 + i] = v.to_bits();
                }
                let device_boxes = context
                    .create_buffer_init(
                        "test boxes",
                        bytemuck::cast_slice(&words),
                        wgpu::BufferUsages::STORAGE,
                    )
                    .unwrap();
                let detection = Detection {
                    scratch: None,
                    image,
                    boxes: vec![box_],
                    device_boxes,
                    size: DetSourceGeometry::new(43, 47)
                        .unwrap()
                        .resize(96, crate::preprocess::DetResizePolicy::Max)
                        .unwrap(),
                    statistics: Default::default(),
                };
                let crops = Crops::prepare(&detection, Default::default()).unwrap();
                let image = crops.record(&mut batch, &detection.image, 0).unwrap();
                let (data, _) = batch
                    .read(&image.buffer, image.width * image.height * 4, false)
                    .unwrap();
                let transform = crate::crop::CropTransform::new(&box_).unwrap();
                assert_eq!(
                    [image.width, image.height],
                    [transform.width(), transform.height()]
                );
                let mut expected = vec![0; transform.byte_count().unwrap()];
                transform.extract_into(source, &mut expected).unwrap();
                for (i, (actual, reference)) in data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(expected.as_chunks::<3>().0.iter())
                    .enumerate()
                {
                    assert_eq!(&actual[..3], reference, "{format:?} pixel {i}, {points:?}");
                }
            }
        }
    }
}
