//! GPU image stages and canonical DB quadrilaterals.
use super::{resident::*, *};
use crate::{
    detector::{DetectionBox, DetectorOptions},
    image::ImageView,
    preprocess::{DetSize, DetSourceGeometry},
};

#[cfg(all(test, target_os = "macos"))]
#[path = "detector_precision_tests.rs"]
mod detector_precision_tests;

pub(crate) struct Detection {
    pub image: Image,
    pub boxes: Vec<DetectionBox>,
    pub device_boxes: GpuBuffer,
    pub size: DetSize,
    pub statistics: GpuPipelineStatistics,
}
pub(crate) struct Detector {
    pub session: GpuSession,
    pub context: Arc<GpuContext>,
}
impl Detector {
    pub fn new(compiled: Arc<GpuCompiledModel>, limits: ExecutionLimits) -> RuntimeResult<Self> {
        let compiled = compiled.for_resident_detector()?;
        compiled.prepare_images()?;
        Ok(Self {
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
        self.run_inner(image, options, profiling, false)
            .map(|(detection, _)| detection)
    }
    pub fn run_with_crops(
        &mut self,
        image: ImageView<'_>,
        options: DetectorOptions,
        profiling: bool,
    ) -> RuntimeResult<(Detection, Option<PreparedCrops>)> {
        self.run_inner(image, options, profiling, true)
    }
    fn run_inner(
        &mut self,
        image: ImageView<'_>,
        options: DetectorOptions,
        profiling: bool,
        prepare_crops: bool,
    ) -> RuntimeResult<(Detection, Option<PreparedCrops>)> {
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
        let (boxes, device_boxes, statistics, crops) = postprocess_with_math(
            batch,
            &prediction,
            options,
            geometry,
            prepare_crops,
            self.session.compiled.matrix_precision,
        )?;
        Ok((
            Detection {
                image,
                boxes,
                device_boxes,
                size,
                statistics,
            },
            crops,
        ))
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
    destination.native_graph_dispatches += source.native_graph_dispatches;
    destination.native_resident_dispatches += source.native_resident_dispatches;
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
pub(crate) struct PreparedCrops {
    transforms: GpuBuffer,
    readback: Vec<u8>,
    metadata_offset: usize,
    count: usize,
}
impl PreparedCrops {
    pub fn finish(self, options: crate::lines::LineOptions) -> RuntimeResult<Crops> {
        let metadata = &self.readback[self.metadata_offset..self.metadata_offset + self.count * 24];
        let CropShapes { sizes, original } = parse_crop_metadata(metadata, self.count, options)?;
        Ok(Crops {
            transforms: self.transforms,
            sizes,
            original,
            // Its GPU work/readback is already included in Detection statistics.
            statistics: Default::default(),
        })
    }
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
        record_crop_transforms(
            &mut batch,
            &detection.device_boxes,
            &transforms,
            count,
            false,
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
        let CropShapes { sizes, original } = parse_crop_metadata(&bytes, count, options)?;
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
        let (x, y) = dispatch_grid(pixels, &batch.context.limits)?;
        let swap = matches!(
            source.format,
            crate::image::PixelFormat::Rgb24 | crate::image::PixelFormat::Rgba32
        );
        batch.dispatch(
            "perspective_bicubic",
            KernelKey::new(KernelId::Crop),
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

fn record_crop_transforms(
    batch: &mut Batch,
    boxes: &GpuBuffer,
    transforms: &GpuBuffer,
    capacity: usize,
    header_count: bool,
) -> RuntimeResult<()> {
    batch.dispatch(
        "perspective_transform",
        KernelKey::new(KernelId::CropMeta),
        &[(boxes, true), (transforms, false)],
        &[capacity as u32, u32::from(header_count)],
        [(capacity as u32).div_ceil(64), 1, 1],
    )
}

struct CropShapes {
    sizes: Vec<[usize; 2]>,
    original: Vec<[usize; 3]>,
}

fn parse_crop_metadata(
    bytes: &[u8],
    count: usize,
    options: crate::lines::LineOptions,
) -> RuntimeResult<CropShapes> {
    if count.checked_mul(24) != Some(bytes.len()) {
        return Err(RuntimeError::new("invalid GPU crop metadata size"));
    }
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
    Ok(CropShapes { sizes, original })
}

pub(crate) fn pack_bgr(batch: &mut Batch, image: &Image) -> RuntimeResult<GpuBuffer> {
    let bytes = image.width * image.height * 3;
    let output = batch.storage("requested BGR crop", bytes)?;
    let (x, y) = dispatch_grid(bytes.div_ceil(4), &batch.context.limits)?;
    batch.dispatch(
        "image_pack_bgr",
        KernelKey::new(KernelId::PackBgr),
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
    let (x, y) = dispatch_grid(image.width * image.height, &batch.context.limits)?;
    batch.dispatch(
        "classifier_rotation",
        KernelKey::new(KernelId::Rotate),
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
    batch: Batch,
    prediction: &Tensor,
    options: DetectorOptions,
    geometry: DetSourceGeometry,
) -> RuntimeResult<(Vec<DetectionBox>, GpuBuffer, GpuPipelineStatistics)> {
    let (boxes, buffer, statistics, _) =
        postprocess_inner(batch, prediction, options, geometry, false)?;
    Ok((boxes, buffer, statistics))
}

struct CropReadbackLayout {
    box_bytes: usize,
    metadata_bytes: usize,
    transform_bytes: usize,
    total_bytes: usize,
}

fn merged_crop_layout(
    count: usize,
    binding_limit: usize,
    device_quota: bool,
) -> Option<CropReadbackLayout> {
    // Speculating on rejected candidates must not add a failure under an
    // explicit quota or exceed a binding that the accepted-only path can fit.
    if count == 0 || device_quota {
        return None;
    }
    let box_bytes = count.checked_add(1)?.checked_mul(64)?;
    let metadata_bytes = count.checked_mul(24)?;
    let transform_bytes = count.checked_mul(128)?;
    let total_bytes = box_bytes.checked_add(metadata_bytes)?;
    if transform_bytes > binding_limit || total_bytes > binding_limit {
        return None;
    }
    Some(CropReadbackLayout {
        box_bytes,
        metadata_bytes,
        transform_bytes,
        total_bytes,
    })
}

type PostprocessResult = (
    Vec<DetectionBox>,
    GpuBuffer,
    GpuPipelineStatistics,
    Option<PreparedCrops>,
);

fn postprocess_inner(
    batch: Batch,
    prediction: &Tensor,
    options: DetectorOptions,
    geometry: DetSourceGeometry,
    prepare_crops: bool,
) -> RuntimeResult<PostprocessResult> {
    // Explicit postprocess replay retains its original precise contract.
    postprocess_with_math(
        batch,
        prediction,
        options,
        geometry,
        prepare_crops,
        GpuMatrixPrecision::F32,
    )
}

fn native_trig_eligible(
    matrix: GpuMatrixPrecision,
    storage: GpuPrecision,
    dimensions: [usize; 6],
    ratio: f32,
) -> bool {
    matrix == GpuMatrixPrecision::F16
        && storage == GpuPrecision::MixedF16
        && dimensions.into_iter().all(|n| (1..=4096).contains(&n))
        && (0.25..=2.0).contains(&ratio)
}

fn geometry_kernel(
    context: &GpuContext,
    precision: GpuPrecision,
    native_trig: bool,
) -> RuntimeResult<(String, Arc<Kernel>)> {
    geometry_kernel_with(context, precision, native_trig, Kernel::get)
}
fn geometry_kernel_with(
    context: &GpuContext,
    precision: GpuPrecision,
    native_trig: bool,
    mut compile: impl FnMut(&GpuContext, KernelKey, &[bool]) -> RuntimeResult<Arc<Kernel>>,
) -> RuntimeResult<(String, Arc<Kernel>)> {
    let layout = &[true, true, true, false, false];
    if native_trig {
        match compile(
            context,
            KernelKey::storage(KernelId::DbGeometryFast, precision),
            layout,
        ) {
            Ok(kernel) => return Ok((format!("db_geometry_native_trig_{precision:?}"), kernel)),
            Err(error) if matches!(error.kind(), ErrorKind::Unsupported) => {
                context.check_alive()?
            }
            Err(error) => return Err(error),
        }
    }
    Ok((
        format!("db_geometry_{precision:?}"),
        compile(
            context,
            KernelKey::storage(KernelId::DbGeometry, precision),
            layout,
        )?,
    ))
}

fn postprocess_with_math(
    mut batch: Batch,
    prediction: &Tensor,
    options: DetectorOptions,
    geometry: DetSourceGeometry,
    prepare_crops: bool,
    matrix: GpuMatrixPrecision,
) -> RuntimeResult<PostprocessResult> {
    batch.ensure_context(&prediction.context)?;
    let profiling = batch.profiling;
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
            KernelKey::storage(KernelId::DbLabel, precision),
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
    let crop_layout = prepare_crops
        .then(|| {
            merged_crop_layout(
                count,
                context.limits.max_storage_buffer_binding_size as usize,
                context.options.max_device_bytes.is_some(),
            )
        })
        .flatten();
    let compact_bytes = crop_layout
        .as_ref()
        .map_or((count + 1) * 64, |layout| layout.total_bytes);
    let compact = batch.storage("DB sorted output", compact_bytes)?;
    if count == 0 {
        return Ok((Vec::new(), compact, statistics, None));
    }
    let rows = batch.storage("DB row extremes", row_count * 8)?;
    let hull = batch.storage(
        "DB hull and offset scratch",
        (row_count * 4 + count * 1024) * 8,
    )?;
    let raw_boxes = batch.storage("DB candidate boxes", count * 64)?;
    for (mode, length) in [(0, row_count), (1, pixels)] {
        let (x, y) = dispatch_grid(length, &context.limits)?;
        batch.dispatch(
            "db_row_extremes",
            KernelKey::new(KernelId::DbRows),
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
    let native_trig = native_trig_eligible(
        matrix,
        precision,
        [
            w,
            h,
            geometry.original_width,
            geometry.original_height,
            geometry.padded_width,
            geometry.padded_height,
        ],
        options.unclip_ratio,
    );
    let (name, kernel) = geometry_kernel(&context, precision, native_trig)?;
    batch.dispatch_prepared(
        &name,
        &kernel,
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
    batch.dispatch(
        "db_reading_order",
        KernelKey::new(KernelId::DbSort),
        &[(&raw_boxes, true), (&compact, false)],
        &[count as u32],
        [1, 1, 1],
    )?;
    let transforms = if let Some(layout) = &crop_layout {
        let transforms = storage(&context, "perspective transforms", layout.transform_bytes)?;
        record_crop_transforms(&mut batch, &compact, &transforms, count, true)?;
        for index in 0..count {
            batch.copy(
                &transforms,
                (index * 128) as u64,
                &compact,
                (layout.box_bytes + index * 24) as u64,
                24,
            );
        }
        Some(transforms)
    } else {
        None
    };
    let (data, mut stats) = batch.read(&compact, compact_bytes, false)?;
    if let Some(layout) = &crop_layout {
        // The single physical readback carries two semantic kinds of data.
        // Count all capacity rows actually copied, including rejected tails.
        stats.output_readback_bytes -= layout.metadata_bytes as u64;
        stats.control_readback_bytes += layout.metadata_bytes as u64;
    }
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
    let crops = if accepted != 0
        && let (Some(transforms), Some(layout)) = (transforms, crop_layout)
    {
        Some(PreparedCrops {
            transforms,
            readback: data,
            metadata_offset: layout.box_bytes,
            count: accepted,
        })
    } else {
        None
    };
    Ok((boxes, compact, statistics, crops))
}

#[cfg(test)]
#[path = "crop_merge_tests.rs"]
mod crop_merge_tests;

#[cfg(test)]
#[path = "geometry_native_trig_tests.rs"]
mod geometry_native_trig_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn count_fixture(case: usize, w: usize, h: usize, dilation: bool) -> Vec<bool> {
        let original: Vec<_> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                match case {
                    0 => false,
                    1 => x == w / 2 && y == h / 2,
                    2 => true,
                    3 => {
                        (3..w - 3).contains(&x)
                            && (3..h - 3).contains(&y)
                            && !((7..w - 7).contains(&x) && (6..h - 6).contains(&y))
                    }
                    4 => (x < 4 && y < 8) || (x >= w - 2 && y >= h - 4),
                    5 => x == y,
                    _ => x % 2 == 1 && y % 2 == 1,
                }
            })
            .collect();
        if !dilation {
            return original;
        }
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                (y..=(y + 1).min(h - 1))
                    .any(|yy| (x..=(x + 1).min(w - 1)).any(|xx| original[yy * w + xx]))
            })
            .collect()
    }

    fn count_reference(
        mask: &[bool],
        w: usize,
        h: usize,
        omit_zero_adds: bool,
    ) -> (Vec<u32>, Vec<u32>, usize) {
        // Independent flood fill: foreground has eight-neighbor connectivity,
        // background four. Ascending seeds give each component its minimum
        // pixel index, independently of the GPU's tile/atomic union algorithm.
        let mut parents = vec![u32::MAX; w * h];
        for seed in 0..mask.len() {
            if parents[seed] != u32::MAX {
                continue;
            }
            let mut todo = vec![seed];
            parents[seed] = seed as u32;
            while let Some(i) = todo.pop() {
                let (x, y) = ((i % w) as i32, (i / w) as i32);
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if (dx == 0 && dy == 0) || (!mask[seed] && dx.abs() + dy.abs() != 1) {
                            continue;
                        }
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let j = ny as usize * w + nx as usize;
                        if parents[j] == u32::MAX && mask[j] == mask[seed] {
                            parents[j] = seed as u32;
                            todo.push(j);
                        }
                    }
                }
            }
        }
        let mut counts = vec![0u32; mask.len()];
        let mut additions = 0;
        for i in 0..mask.len() {
            let (x, y) = (i % w, i / w);
            let root = parents[i] as usize;
            let border = x == 0 || y == 0 || x + 1 == w || y + 1 == h;
            let boundary = if mask[i] {
                u32::from(
                    border
                        || (y.saturating_sub(1)..=(y + 1).min(h - 1)).any(|yy| {
                            (x.saturating_sub(1)..=(x + 1).min(w - 1)).any(|xx| !mask[yy * w + xx])
                        }),
                )
            } else {
                if border {
                    counts[root] |= 0x8000_0000;
                }
                u32::from(x > 0 && mask[i - 1])
                    + u32::from(x + 1 < w && mask[i + 1])
                    + u32::from(y > 0 && mask[i - w])
                    + u32::from(y + 1 < h && mask[i + w])
            };
            if !omit_zero_adds || boundary != 0 {
                counts[root] = counts[root].wrapping_add(boundary);
                additions += 1;
            }
        }
        (parents, counts, additions)
    }

    #[test]
    fn db_zero_boundary_adds_preserve_component_counts_and_border_flags() {
        let (w, h) = (19, 17);
        for case in 0..7 {
            for dilation in [false, true] {
                let mask = count_fixture(case, w, h, dilation);
                let (parents, counts, all_additions) = count_reference(&mask, w, h, false);
                let (guarded_parents, guarded_counts, nonzero_additions) =
                    count_reference(&mask, w, h, true);
                assert_eq!(parents, guarded_parents);
                assert_eq!(counts, guarded_counts);
                assert_eq!(all_additions, w * h);
                assert!(nonzero_additions <= all_additions);
                for i in 0..mask.len() {
                    if mask[i] {
                        assert_eq!(counts[parents[i] as usize] & 0x8000_0000, 0);
                    }
                }
                if case == 0 {
                    assert_eq!(nonzero_additions, 0);
                    assert_eq!(counts[0], 0x8000_0000);
                } else if case == 2 {
                    assert_eq!(nonzero_additions, 2 * w + 2 * h - 4);
                    assert_eq!(counts[0], nonzero_additions as u32);
                } else if case == 1 && !dilation {
                    assert_eq!(nonzero_additions, 5);
                    assert_eq!(counts[w * (h / 2) + w / 2], 1);
                    assert_eq!(counts[0], 0x8000_0004);
                }
                if case == 3 && !dilation {
                    let hole = 7 * w + 8;
                    assert!(!mask[hole]);
                    assert_eq!(counts[parents[hole] as usize] & 0x8000_0000, 0);
                    assert!(counts[parents[hole] as usize] >= 3);
                }
            }
        }
    }

    #[test]
    #[ignore = "requires physical Vulkan; exact DB parent/count flags and instrumented zero-add traffic for F32/Half"]
    fn gpu_resident_db_nonzero_boundary_counts() {
        let (w, h) = (19usize, 17usize);
        let pixels = w * h;
        {
            let context = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
            for precision in [GpuPrecision::F32, GpuPrecision::MixedF16] {
                let dispatch_name = format!("db_label_boundary_{precision:?}");
                let source = KernelKey::storage(KernelId::DbLabel, precision);
                for case in 0..7 {
                    for dilation in [false, true] {
                        let original = count_fixture(case, w, h, false);
                        let mask = count_fixture(case, w, h, dilation);
                        let (parents, counts, _additions) = count_reference(&mask, w, h, true);
                        let map: Vec<f32> = original
                            .iter()
                            .map(|&yes| if yes { 0.75 } else { 0.0 })
                            .collect();
                        let prediction = context
                            .create_buffer_init(
                                "DB boundary probability fixture",
                                &precision.encode(&map),
                                BufferUsage::STORAGE,
                            )
                            .unwrap();
                        let mut batch = Batch::new(context.clone()).unwrap();
                        let state: Vec<_> = (0..5)
                            .map(|_| batch.storage("DB boundary state", pixels * 4).unwrap())
                            .collect();
                        let meta = batch.storage("DB boundary count metadata", 16).unwrap();
                        let buffers = [
                            (&prediction, true),
                            (&state[0], false),
                            (&state[1], false),
                            (&state[2], false),
                            (&state[3], false),
                            (&state[4], false),
                            (&meta, false),
                        ];
                        let groups = pixels.div_ceil(64) as u32;
                        for mode in 0..=3 {
                            batch
                                .dispatch(
                                    &dispatch_name,
                                    source,
                                    &buffers,
                                    &[
                                        w as u32,
                                        h as u32,
                                        0.5f32.to_bits(),
                                        pixels as u32,
                                        mode,
                                        u32::from(dilation),
                                        1,
                                    ],
                                    [
                                        if mode == 1 {
                                            (w.div_ceil(8) * h.div_ceil(8)) as u32
                                        } else {
                                            groups
                                        },
                                        1,
                                        1,
                                    ],
                                )
                                .unwrap();
                        }
                        let output = batch
                            .storage("DB boundary fixture readback", (pixels * 3 + 4) * 4)
                            .unwrap();
                        for (index, buffer) in state[..3].iter().enumerate() {
                            batch.copy(
                                buffer,
                                0,
                                &output,
                                (index * pixels * 4) as u64,
                                (pixels * 4) as u64,
                            );
                        }
                        batch.copy(&meta, 0, &output, (pixels * 3 * 4) as u64, 16);
                        let (bytes, _) = batch.read(&output, output.bytes(), false).unwrap();
                        let actual: Vec<_> = bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|b| u32::from_le_bytes(*b))
                            .collect();
                        for index in 0..pixels {
                            assert_eq!(
                                actual[index],
                                u32::from(mask[index]),
                                "mask precision={precision:?} case={case} dilation={dilation} index={index}"
                            );
                            assert_eq!(
                                actual[pixels + index],
                                parents[index],
                                "parent precision={precision:?} case={case} dilation={dilation} index={index}"
                            );
                            assert_eq!(
                                actual[pixels * 2 + index],
                                counts[index],
                                "count precision={precision:?} case={case} dilation={dilation} index={index}"
                            );
                        }
                        assert_eq!(
                            actual[pixels * 3 + 2],
                            0,
                            "nonfinite flag precision={precision:?} case={case} dilation={dilation}"
                        );
                        assert_eq!(
                            actual[pixels * 3 + 3] as usize,
                            0,
                            "unused metadata precision={precision:?} case={case} dilation={dilation}"
                        );
                    }
                }
            }
        }
    }

    fn hardware() -> Arc<GpuContext> {
        let context = GpuContext::new_with_options(GpuContextOptions::default()).unwrap();
        assert!(matches!(
            context.adapter_info().device_type,
            crate::runtime::gpu::GpuDeviceType::DiscreteGpu
                | crate::runtime::gpu::GpuDeviceType::IntegratedGpu
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
                    BufferUsage::STORAGE,
                )
                .unwrap();
            let mut batch = Batch::new(context.clone()).unwrap();
            let output = batch.storage("sort output", (count + 1) * 64).unwrap();
            batch
                .dispatch(
                    "db_reading_order",
                    KernelKey::new(KernelId::DbSort),
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
                        .upload(&tensor.buffer, 0, bytemuck::cast_slice(&map))
                        .unwrap();
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
    fn gpu_resident_geometry_mapping_rounds_halfway_to_even() {
        let context = hardware();
        let (w, h) = (64, 64);
        let options = DetectorOptions {
            box_threshold: Some(0.5),
            ..Default::default()
        };
        for origin in [12, 13] {
            let mut map = vec![0.0f32; w * h];
            for y in 10..25 {
                map[y * w + origin..y * w + origin + 20].fill(0.75);
            }
            let tensor = Tensor::allocate(
                context.clone(),
                Shape::new(&[1, 1, h, w]).unwrap(),
                GpuPrecision::F32,
            )
            .unwrap();
            context
                .upload(&tensor.buffer, 0, bytemuck::cast_slice(&map))
                .unwrap();
            // Both scales map odd integer corners onto exact half-integers.
            for side in [96, 160] {
                let geometry = DetSourceGeometry::new(side, side).unwrap();
                let (actual, _, _) = postprocess(
                    Batch::new(context.clone()).unwrap(),
                    &tensor,
                    options,
                    geometry,
                )
                .unwrap();
                let mut expected = crate::db::Workspace::default();
                crate::db::run(&map, w, h, &options, geometry, &mut expected).unwrap();
                assert_eq!(actual.len(), 1);
                assert_eq!(
                    actual[0].points, expected.boxes[0].points,
                    "origin {origin}, source side {side}"
                );
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
                        BufferUsage::STORAGE,
                    )
                    .unwrap();
                let detection = Detection {
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
