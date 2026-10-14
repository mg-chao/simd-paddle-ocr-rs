//! Reusable pixel-to-oriented-crop pipeline. Recognition is a separate stage.
use crate::{
    crop::{self, CropTransform},
    detector::{DetectionBox, DetectionTrace, Detector},
    image::{ImageError, ImageResult, ImageView, PixelFormat},
    orientation::{ClassificationResult, Classifier, ClassifierTrace, should_rotate},
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
}
impl OrientedLineExtractor {
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
            .map(|model| Classifier::new(model, options.classifier_execution_limits))
            .transpose()?;
        Ok(Self {
            detector,
            classifier,
            options,
            transforms: Vec::new(),
            metadata: Vec::new(),
            slab: Vec::new(),
        })
    }
    pub fn detector_mut(&mut self) -> &mut Detector {
        &mut self.detector
    }
    pub fn options(&self) -> LineOptions {
        self.options
    }
    pub fn extract(&mut self, image: ImageView<'_>) -> ImageResult<OrientedLines<'_>> {
        self.extract_impl::<false, _, _>(image, &mut |_| Ok(()), &mut |_| Ok(()))
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
        self.extract_impl::<true, _, _>(image, &mut det_observer, &mut line_observer)
    }
    fn extract_impl<const TRACE: bool, D, L>(
        &mut self,
        image: ImageView<'_>,
        det_observer: &mut D,
        line_observer: &mut L,
    ) -> ImageResult<OrientedLines<'_>>
    where
        D: for<'a> FnMut(DetectionTrace<'a>) -> ImageResult<()>,
        L: for<'a> FnMut(LineTrace<'a>) -> ImageResult<()>,
    {
        let detection = if TRACE {
            self.detector.detect_with_trace(image, det_observer)?
        } else {
            self.detector.detect(image)?
        };
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
        for (index, (metadata, transform)) in
            self.metadata.iter_mut().zip(&self.transforms).enumerate()
        {
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
    }
    pub fn classifier_retained_activation_bytes(&self) -> usize {
        self.classifier
            .as_ref()
            .map_or(0, Classifier::retained_activation_bytes)
    }
    pub fn retained_workspace_bytes(&self) -> usize {
        self.detector.retained_workspace_bytes()
            + self
                .classifier
                .as_ref()
                .map_or(0, Classifier::retained_workspace_bytes)
            + self.crop_capacity_bytes()
            + self.line_capacity_bytes()
    }
}
