//! Reusable pixel-to-oriented-crop pipeline. Recognition is a separate stage.
use crate::{
    cpu::{CpuBudgets, CpuContext, CpuOptions},
    crop::{self, CropTransform},
    detector::{DetectionBox, DetectionTrace, Detector},
    image::{ImageError, ImageResult, ImageView, PixelFormat},
    orientation::{
        ClassificationResult, Classifier, ClassifierOptions, ClassifierTrace, should_rotate,
    },
    runtime::{CompiledModel, ExecutionLimits, NodeTrace},
};
use serde::Serialize;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct LineOptions {
    pub classifier_threshold: f32,
    pub max_classifier_image_pixels: usize,
    pub max_crop_pixels: usize,
    pub max_total_crop_bytes: usize,
    pub classifier_execution_limits: ExecutionLimits,
}
impl Default for LineOptions {
    fn default() -> Self {
        Self {
            classifier_threshold: 0.,
            max_classifier_image_pixels: 40_000_000,
            max_crop_pixels: 16_000_000,
            max_total_crop_bytes: i32::MAX as usize,
            classifier_execution_limits: ExecutionLimits::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LineMetadata {
    pub detection: DetectionBox,
    pub offset: usize,
    pub bytes: usize,
    pub width: usize,
    pub height: usize,
    pub unrotated_width: usize,
    pub unrotated_height: usize,
    pub vertical_rotation: bool,
    pub classification: Option<ClassificationResult>,
    pub applied_rotation_degrees: u16,
}

pub struct LineView<'a> {
    pub metadata: &'a LineMetadata,
    pub image: ImageView<'a>,
}

pub struct OrientedLines<'a> {
    pub resized_width: usize,
    pub resized_height: usize,
    pub width_ratio: f32,
    pub height_ratio: f32,
    pub lines: &'a [LineMetadata],
    pub pixels: &'a [u8],
}
impl<'a> OrientedLines<'a> {
    pub fn get(&self, index: usize) -> Option<LineView<'a>> {
        let metadata = self.lines.get(index)?;
        let end = metadata.offset.checked_add(metadata.bytes)?;
        let pixels = self.pixels.get(metadata.offset..end)?;
        let image = ImageView::new(
            pixels,
            metadata.width,
            metadata.height,
            0,
            PixelFormat::Bgr24,
        )
        .ok()?;
        Some(LineView { metadata, image })
    }
    pub fn iter(&self) -> impl Iterator<Item = LineView<'a>> + '_ {
        (0..self.lines.len()).filter_map(|index| self.get(index))
    }
    pub fn try_to_owned(&self) -> ImageResult<OwnedOrientedLines> {
        let mut lines = Vec::new();
        lines
            .try_reserve_exact(self.lines.len())
            .map_err(|e| ImageError::new(format!("line copy allocation: {e}")))?;
        lines.extend_from_slice(self.lines);
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(self.pixels.len())
            .map_err(|e| ImageError::new(format!("crop copy allocation: {e}")))?;
        pixels.extend_from_slice(self.pixels);
        Ok(OwnedOrientedLines {
            resized_width: self.resized_width,
            resized_height: self.resized_height,
            width_ratio: self.width_ratio,
            height_ratio: self.height_ratio,
            lines,
            pixels,
        })
    }
}
pub struct OwnedOrientedLines {
    pub resized_width: usize,
    pub resized_height: usize,
    pub width_ratio: f32,
    pub height_ratio: f32,
    pub lines: Vec<LineMetadata>,
    pub pixels: Vec<u8>,
}
impl OwnedOrientedLines {
    pub fn as_view(&self) -> OrientedLines<'_> {
        OrientedLines {
            resized_width: self.resized_width,
            resized_height: self.resized_height,
            width_ratio: self.width_ratio,
            height_ratio: self.height_ratio,
            lines: &self.lines,
            pixels: &self.pixels,
        }
    }
}

pub enum LineTrace<'a> {
    Crop {
        index: usize,
        image: ImageView<'a>,
    },
    ClsInput {
        index: usize,
        values: &'a [f32],
        resized_width: usize,
    },
    ClsNode {
        index: usize,
        node: NodeTrace<'a>,
    },
    ClsProbabilities {
        index: usize,
        values: &'a [f32],
    },
    Classification {
        index: usize,
        result: ClassificationResult,
    },
    OrientedCrop {
        index: usize,
        image: ImageView<'a>,
        applied_rotation_degrees: u16,
    },
}

pub struct OrientedLineExtractor {
    detector: Detector,
    classifier: Option<Classifier>,
    options: LineOptions,
    transforms: Vec<CropTransform>,
    metadata: Vec<LineMetadata>,
    slab: Vec<u8>,
    cpu: Arc<CpuContext>,
    cpu_options: CpuOptions,
    budgets: CpuBudgets,
    classifier_workers: Vec<Classifier>,
}
impl OrientedLineExtractor {
    pub fn detector(&self) -> &Detector {
        &self.detector
    }
    pub(crate) fn classifier_model(&self) -> Option<&Arc<CompiledModel>> {
        self.classifier.as_ref().map(Classifier::compiled)
    }
    pub fn new(
        detector: Detector,
        classifier: Option<Arc<CompiledModel>>,
        options: LineOptions,
    ) -> ImageResult<Self> {
        if !options.classifier_threshold.is_finite()
            || !(0.0..=1.0).contains(&options.classifier_threshold)
        {
            return Err(ImageError::new(
                "classifier_threshold must be finite and within [0,1]",
            ));
        }
        if options.max_crop_pixels == 0
            || options.max_classifier_image_pixels == 0
            || options.max_total_crop_bytes == 0
        {
            return Err(ImageError::new("image and crop limits must be positive"));
        }
        let classifier = classifier
            .map(|model| {
                Classifier::new_with_options(
                    model,
                    ClassifierOptions {
                        max_image_pixels: options.max_classifier_image_pixels,
                    },
                    options.classifier_execution_limits,
                )
            })
            .transpose()?;
        Ok(Self {
            detector,
            classifier,
            options,
            transforms: Vec::new(),
            metadata: Vec::new(),
            slab: Vec::new(),
            cpu: CpuContext::serial(),
            cpu_options: CpuOptions::default(),
            budgets: CpuOptions::default().resolve(1, true)?,
            classifier_workers: Vec::new(),
        })
    }
    pub fn fork(&self) -> ImageResult<Self> {
        Ok(Self {
            detector: self.detector.fork()?,
            classifier: self.classifier.as_ref().map(Classifier::fork).transpose()?,
            options: self.options,
            transforms: Vec::new(),
            metadata: Vec::new(),
            slab: Vec::new(),
            cpu: self.cpu.clone(),
            cpu_options: self.cpu_options,
            budgets: self.budgets,
            classifier_workers: Vec::new(),
        })
    }
    pub fn set_execution_context(
        &mut self,
        cpu: Arc<CpuContext>,
        options: CpuOptions,
    ) -> ImageResult<()> {
        let budgets = options.resolve(
            cpu.capacity(),
            matches!(
                self.detector.compiled().options().layout,
                crate::runtime::LayoutPolicy::Auto
            ),
        )?;
        self.detector
            .set_execution_context(cpu.clone(), budgets.detector_threads)?;
        if let Some(classifier) = &mut self.classifier {
            classifier.set_execution_context(cpu.clone(), 1)?;
        }
        for classifier in &mut self.classifier_workers {
            classifier.set_execution_context(cpu.clone(), 1)?;
        }
        self.cpu = cpu;
        self.cpu_options = options;
        self.budgets = budgets;
        Ok(())
    }
    pub fn detector_mut(&mut self) -> &mut Detector {
        &mut self.detector
    }
    pub fn options(&self) -> LineOptions {
        self.options
    }
    pub fn extract(&mut self, image: ImageView<'_>) -> ImageResult<OrientedLines<'_>> {
        self.extract_profiled::<false>(image, &mut crate::profile::StageProfile::default())
    }
    pub(crate) fn extract_profiled<const PROFILE: bool>(
        &mut self,
        image: ImageView<'_>,
        profile: &mut crate::profile::StageProfile,
    ) -> ImageResult<OrientedLines<'_>> {
        let cpu = self.cpu.clone();
        cpu.install(|| {
            self.extract_impl::<false, PROFILE, _, _>(
                image,
                &mut |_| Ok(()),
                &mut |_| Ok(()),
                profile,
            )
        })
    }
    pub fn extract_with_trace<D, L>(
        &mut self,
        image: ImageView<'_>,
        mut det_observer: D,
        mut line_observer: L,
    ) -> ImageResult<OrientedLines<'_>>
    where
        D: for<'a> FnMut(DetectionTrace<'a>) -> ImageResult<()>,
        L: for<'a> FnMut(LineTrace<'a>) -> ImageResult<()>,
    {
        self.extract_impl::<true, false, _, _>(
            image,
            &mut det_observer,
            &mut line_observer,
            &mut crate::profile::StageProfile::default(),
        )
    }
    fn extract_impl<const TRACE: bool, const PROFILE: bool, D, L>(
        &mut self,
        image: ImageView<'_>,
        det_observer: &mut D,
        line_observer: &mut L,
        profile: &mut crate::profile::StageProfile,
    ) -> ImageResult<OrientedLines<'_>>
    where
        D: for<'a> FnMut(DetectionTrace<'a>) -> ImageResult<()>,
        L: for<'a> FnMut(LineTrace<'a>) -> ImageResult<()>,
    {
        #[cfg(test)]
        if !TRACE {
            self.cpu.assert_production_thread("crop extraction");
        }
        let detection = if TRACE {
            self.detector.detect_with_trace(image, det_observer)?
        } else {
            self.detector.detect_profiled::<PROFILE>(image, profile)?
        };
        let crop_timer = crate::profile::Timer::<PROFILE>::start();
        self.transforms.clear();
        self.metadata.clear();
        self.transforms
            .try_reserve(detection.boxes.len())
            .map_err(|e| ImageError::new(format!("crop transform allocation: {e}")))?;
        self.metadata
            .try_reserve(detection.boxes.len())
            .map_err(|e| ImageError::new(format!("line metadata allocation: {e}")))?;
        let mut total = 0usize;
        // Every transform and budget is checked before writing any crop bytes.
        for detection_box in detection.boxes {
            let transform = CropTransform::new(detection_box)?;
            let pixels = transform
                .width()
                .checked_mul(transform.height())
                .ok_or_else(|| ImageError::new("crop pixel overflow"))?;
            if pixels > self.options.max_crop_pixels {
                return Err(ImageError::new("crop exceeds max_crop_pixels"));
            }
            if self.classifier.is_some() && pixels > self.options.max_classifier_image_pixels {
                return Err(ImageError::new("crop exceeds max_classifier_image_pixels"));
            }
            let bytes = pixels
                .checked_mul(3)
                .ok_or_else(|| ImageError::new("crop byte overflow"))?;
            let offset = total;
            total = total
                .checked_add(bytes)
                .filter(|&n| n <= self.options.max_total_crop_bytes && n <= isize::MAX as usize)
                .ok_or_else(|| ImageError::new("crops exceed max_total_crop_bytes"))?;
            self.metadata.push(LineMetadata {
                detection: *detection_box,
                offset,
                bytes,
                width: transform.width(),
                height: transform.height(),
                unrotated_width: transform.unrotated_width(),
                unrotated_height: transform.unrotated_height(),
                vertical_rotation: transform.vertical_rotation(),
                classification: None,
                applied_rotation_degrees: 0,
            });
            self.transforms.push(transform);
        }
        self.slab
            .try_reserve_exact(total.saturating_sub(self.slab.len()))
            .map_err(|e| ImageError::new(format!("crop slab allocation: {e}")))?;
        self.slab.resize(total, 0);
        if PROFILE {
            profile.crop_ns = profile.crop_ns.saturating_add(crop_timer.elapsed());
        }
        if !TRACE && self.cpu.capacity() > 1 && !self.metadata.is_empty() {
            extract_parallel::<PROFILE>(
                &self.cpu,
                self.budgets,
                image,
                &self.transforms,
                &mut self.metadata,
                &mut self.slab,
                &self.classifier,
                &mut self.classifier_workers,
                self.options.classifier_threshold,
                profile,
            )?;
            return Ok(OrientedLines {
                resized_width: detection.resized_width,
                resized_height: detection.resized_height,
                width_ratio: detection.width_ratio,
                height_ratio: detection.height_ratio,
                lines: &self.metadata,
                pixels: &self.slab,
            });
        }
        for (index, (metadata, transform)) in
            self.metadata.iter_mut().zip(&self.transforms).enumerate()
        {
            let crop_timer = crate::profile::Timer::<PROFILE>::start();
            let pixels = &mut self.slab[metadata.offset..metadata.offset + metadata.bytes];
            transform.extract_into(image, pixels)?;
            let crop = ImageView::new(
                pixels,
                metadata.width,
                metadata.height,
                0,
                PixelFormat::Bgr24,
            )?;
            if TRACE {
                line_observer(LineTrace::Crop { index, image: crop })?;
            }
            if PROFILE {
                let elapsed = crop_timer.elapsed();
                profile.crop_ns = profile.crop_ns.saturating_add(elapsed);
                profile.crop_work_ns = profile.crop_work_ns.saturating_add(elapsed);
            }
            let cls_timer = crate::profile::Timer::<PROFILE>::start();
            if let Some(classifier) = &mut self.classifier {
                let result = if TRACE {
                    classifier.classify_with_trace(crop, |event| {
                        let event = match event {
                            ClassifierTrace::Input {
                                values,
                                resized_width,
                            } => LineTrace::ClsInput {
                                index,
                                values,
                                resized_width,
                            },
                            ClassifierTrace::Node(node) => LineTrace::ClsNode { index, node },
                            ClassifierTrace::Probabilities(values) => {
                                LineTrace::ClsProbabilities { index, values }
                            }
                            ClassifierTrace::Public(result) => {
                                LineTrace::Classification { index, result }
                            }
                        };
                        line_observer(event)
                    })?
                } else {
                    classifier.classify(crop)?
                };
                metadata.classification = Some(result);
                if should_rotate(result, self.options.classifier_threshold) {
                    crop::rotate180(pixels, metadata.width, metadata.height)?;
                    metadata.applied_rotation_degrees = 180;
                }
            }
            if PROFILE {
                let elapsed = cls_timer.elapsed();
                profile.cls_orientation_ns = profile.cls_orientation_ns.saturating_add(elapsed);
                profile.classification_wall_ns =
                    profile.classification_wall_ns.saturating_add(elapsed);
            }
            if TRACE {
                line_observer(LineTrace::OrientedCrop {
                    index,
                    image: ImageView::new(
                        pixels,
                        metadata.width,
                        metadata.height,
                        0,
                        PixelFormat::Bgr24,
                    )?,
                    applied_rotation_degrees: metadata.applied_rotation_degrees,
                })?;
            }
        }
        Ok(OrientedLines {
            resized_width: detection.resized_width,
            resized_height: detection.resized_height,
            width_ratio: detection.width_ratio,
            height_ratio: detection.height_ratio,
            lines: &self.metadata,
            pixels: &self.slab,
        })
    }
    pub fn crop_capacity_bytes(&self) -> usize {
        self.slab.capacity()
    }
    pub fn line_capacity_bytes(&self) -> usize {
        self.metadata.capacity() * std::mem::size_of::<LineMetadata>()
            + self.transforms.capacity() * std::mem::size_of::<CropTransform>()
    }
    pub fn classifier_activation_bytes(&self) -> usize {
        self.classifier
            .as_ref()
            .map_or(0, Classifier::activation_bytes)
            + self
                .classifier_workers
                .iter()
                .map(Classifier::activation_bytes)
                .sum::<usize>()
    }
    pub fn classifier_retained_activation_bytes(&self) -> usize {
        self.classifier
            .as_ref()
            .map_or(0, Classifier::retained_activation_bytes)
            + self
                .classifier_workers
                .iter()
                .map(Classifier::retained_activation_bytes)
                .sum::<usize>()
    }
    pub fn retained_workspace_bytes(&self) -> usize {
        let buffers = self.detector.retained_workspace_bytes()
            + self
                .classifier
                .as_ref()
                .map_or(0, Classifier::retained_workspace_bytes)
            + self.crop_capacity_bytes()
            + self.line_capacity_bytes()
            + self
                .classifier_workers
                .iter()
                .map(Classifier::retained_workspace_bytes)
                .sum::<usize>();
        buffers.saturating_add(
            self.classifier_workers
                .capacity()
                .saturating_mul(std::mem::size_of::<Classifier>()),
        )
    }
    /// Retained kernel buffers across DET and every classifier request.
    pub fn retained_kernel_scratch_bytes(&self) -> usize {
        self.detector.retained_kernel_scratch_bytes()
            + self
                .classifier
                .as_ref()
                .map_or(0, Classifier::retained_kernel_scratch_bytes)
            + self
                .classifier_workers
                .iter()
                .map(Classifier::retained_kernel_scratch_bytes)
                .sum::<usize>()
    }
}

struct CropJob<'a> {
    transform: CropTransform,
    first_row: usize,
    pixels: &'a mut [u8],
    error: Option<ImageError>,
    elapsed: u64,
}

struct ClassifyJob<'a> {
    index: usize,
    metadata: &'a mut LineMetadata,
    pixels: &'a mut [u8],
}
struct ClassifyTask<'a> {
    classifier: &'a mut Classifier,
    jobs: Vec<ClassifyJob<'a>>,
    error: Option<(usize, ImageError)>,
    work_ns: u64,
}
#[allow(clippy::too_many_arguments)]
fn extract_parallel<const PROFILE: bool>(
    cpu: &Arc<CpuContext>,
    budgets: CpuBudgets,
    image: ImageView<'_>,
    transforms: &[CropTransform],
    metadata: &mut [LineMetadata],
    slab: &mut [u8],
    classifier: &Option<Classifier>,
    classifiers: &mut Vec<Classifier>,
    threshold: f32,
    profile: &mut crate::profile::StageProfile,
) -> ImageResult<()> {
    let crop_timer = crate::profile::Timer::<PROFILE>::start();
    let mut jobs = Vec::new();
    let bands = metadata
        .iter()
        .try_fold(0usize, |sum, line| {
            sum.checked_add(line.height.div_ceil(16))
        })
        .ok_or_else(|| ImageError::new("crop band count overflow"))?;
    jobs.try_reserve_exact(bands)
        .map_err(|e| ImageError::new(format!("crop bands allocation: {e}")))?;
    let mut remaining = &mut *slab;
    for (transform, line) in transforms.iter().zip(metadata.iter()) {
        let (pixels, tail) = remaining.split_at_mut(line.bytes);
        remaining = tail;
        let row_bytes = line.width * 3;
        for (band, pixels) in pixels.chunks_mut(row_bytes * 16).enumerate() {
            jobs.push(CropJob {
                transform: *transform,
                first_row: band * 16,
                pixels,
                error: None,
                elapsed: 0,
            });
        }
    }
    cpu.for_each_mut(&mut jobs, budgets.preprocess_workers, |_, job| {
        let timer = crate::profile::Timer::<PROFILE>::start();
        job.error = job
            .transform
            .extract_rows_into(image, job.first_row, job.pixels)
            .err();
        job.elapsed = timer.elapsed();
    });
    if PROFILE {
        profile.crop_ns = profile.crop_ns.saturating_add(crop_timer.elapsed());
        profile.crop_work_ns = profile
            .crop_work_ns
            .saturating_add(jobs.iter().map(|j| j.elapsed).sum::<u64>());
    }
    for job in jobs {
        if let Some(error) = job.error {
            return Err(error);
        }
    }
    let Some(classifier) = classifier else {
        return Ok(());
    };
    let cls_timer = crate::profile::Timer::<PROFILE>::start();
    let active = budgets.line_workers.min(metadata.len());
    classifiers
        .try_reserve(active.saturating_sub(classifiers.len()))
        .map_err(|e| ImageError::new(format!("classifier workers allocation: {e}")))?;
    while classifiers.len() < active {
        let mut worker = classifier.fork()?;
        worker.set_execution_context(cpu.clone(), 1)?;
        classifiers.push(worker);
    }
    let mut tasks = Vec::new();
    tasks
        .try_reserve_exact(active)
        .map_err(|e| ImageError::new(format!("classification tasks allocation: {e}")))?;
    for classifier in &mut classifiers[..active] {
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(metadata.len().div_ceil(active))
            .map_err(|e| ImageError::new(format!("classification jobs allocation: {e}")))?;
        tasks.push(ClassifyTask {
            classifier,
            jobs,
            error: None,
            work_ns: 0,
        });
    }
    let mut remaining = &mut *slab;
    for (index, line) in metadata.iter_mut().enumerate() {
        let (pixels, tail) = remaining.split_at_mut(line.bytes);
        remaining = tail;
        tasks[index % active].jobs.push(ClassifyJob {
            index,
            metadata: line,
            pixels,
        });
    }
    cpu.for_each_mut(&mut tasks, active, |_, task| {
        for job in &mut task.jobs {
            let timer = crate::profile::Timer::<PROFILE>::start();
            let outcome = (|| {
                let crop = ImageView::new(
                    job.pixels,
                    job.metadata.width,
                    job.metadata.height,
                    0,
                    PixelFormat::Bgr24,
                )?;
                let result = task.classifier.classify(crop)?;
                job.metadata.classification = Some(result);
                if should_rotate(result, threshold) {
                    crop::rotate180(job.pixels, job.metadata.width, job.metadata.height)?;
                    job.metadata.applied_rotation_degrees = 180;
                }
                Ok(())
            })();
            task.work_ns = task.work_ns.saturating_add(timer.elapsed());
            if let Err(error) = outcome
                && task
                    .error
                    .as_ref()
                    .is_none_or(|(earlier, _)| job.index < *earlier)
            {
                task.error = Some((job.index, error));
            }
        }
    });
    let mut error = None;
    for task in tasks {
        if PROFILE {
            profile.cls_orientation_ns = profile.cls_orientation_ns.saturating_add(task.work_ns);
        }
        if let Some((index, cause)) = task.error
            && error.as_ref().is_none_or(|(earlier, _)| index < *earlier)
        {
            error = Some((index, cause));
        }
    }
    if PROFILE {
        profile.classification_wall_ns = profile
            .classification_wall_ns
            .saturating_add(cls_timer.elapsed());
    }
    if let Some((_, error)) = error {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_workspace_counts_reserved_classifier_worker_storage() {
        let mut template = crate::ocr::tests::synthetic_ocr(CpuContext::serial());
        let detector = template.extractor_mut().detector().fork().unwrap();
        let mut extractor =
            OrientedLineExtractor::new(detector, None, LineOptions::default()).unwrap();
        let before = extractor.retained_workspace_bytes();
        extractor.classifier_workers.try_reserve_exact(3).unwrap();
        let added = extractor.classifier_workers.capacity() * std::mem::size_of::<Classifier>();
        assert!(added > 0);
        assert_eq!(extractor.retained_workspace_bytes(), before + added);
    }
}
