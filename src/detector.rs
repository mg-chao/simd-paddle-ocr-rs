//! Reusable serial image detector with borrowed results and synchronous tracing.
pub use crate::preprocess::DetResizePolicy;
use crate::profile::{StageProfile, Timer};
use crate::{
    db,
    image::{ImageError, ImageResult, ImageView, resize_vec},
    model::Operator,
    preprocess::{DET_MAX_SIDE, DetSourceGeometry, ResizeWorkspace},
    runtime::{CompiledModel, ExecutionLimits, NodeTrace, RuntimeError, Session},
    tensor::Shape,
};
use serde::Serialize;
use std::{cell::RefCell, sync::Arc};

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct DetectionBox {
    pub points: [[f32; 2]; 4],
    pub score: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct DetectorOptions {
    pub limit_side_length: usize,
    pub resize_policy: DetResizePolicy,
    pub max_candidates: usize,
    pub use_dilation: bool,
    pub bitmap_threshold: f32,
    pub box_threshold: Option<f32>,
    pub unclip_ratio: f32,
    pub max_image_pixels: usize,
}
impl Default for DetectorOptions {
    fn default() -> Self {
        Self {
            limit_side_length: 960,
            resize_policy: DetResizePolicy::Max,
            max_candidates: 3000,
            use_dilation: false,
            bitmap_threshold: 0.2,
            box_threshold: None,
            unclip_ratio: 1.4,
            max_image_pixels: 40_000_000,
        }
    }
}
impl DetectorOptions {
    fn validate(self) -> ImageResult<()> {
        if !(32..=DET_MAX_SIDE).contains(&self.limit_side_length)
            || !(1..=10_000).contains(&self.max_candidates)
            || self.max_image_pixels == 0
            || !self.bitmap_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.bitmap_threshold)
            || self
                .box_threshold
                .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
            || !self.unclip_ratio.is_finite()
            || self.unclip_ratio <= 0.
            || self.unclip_ratio > 10.
        {
            return Err(ImageError::new("invalid detector options"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct DetectionView<'a> {
    pub boxes: &'a [DetectionBox],
    pub resized_width: usize,
    pub resized_height: usize,
    pub width_ratio: f32,
    pub height_ratio: f32,
}
impl DetectionView<'_> {
    pub fn try_to_owned(self) -> ImageResult<OwnedDetection> {
        let mut boxes = Vec::new();
        boxes.try_reserve_exact(self.boxes.len()).map_err(|e| {
            ImageError::caused_by(
                crate::error::ErrorKind::Allocation,
                "detection copy allocation failed",
                e,
            )
        })?;
        boxes.extend_from_slice(self.boxes);
        Ok(OwnedDetection {
            boxes,
            resized_width: self.resized_width,
            resized_height: self.resized_height,
            width_ratio: self.width_ratio,
            height_ratio: self.height_ratio,
        })
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct OwnedDetection {
    pub boxes: Vec<DetectionBox>,
    pub resized_width: usize,
    pub resized_height: usize,
    pub width_ratio: f32,
    pub height_ratio: f32,
}
impl OwnedDetection {
    pub fn as_view(&self) -> DetectionView<'_> {
        DetectionView {
            boxes: &self.boxes,
            resized_width: self.resized_width,
            resized_height: self.resized_height,
            width_ratio: self.width_ratio,
            height_ratio: self.height_ratio,
        }
    }
}
pub enum DetectionTrace<'a> {
    Input { shape: Shape, values: &'a [f32] },
    Node(NodeTrace<'a>),
    Probabilities { shape: Shape, values: &'a [f32] },
    Boxes(DetectionView<'a>),
}
pub struct Detector {
    compiled: Arc<CompiledModel>,
    options: DetectorOptions,
    session: Session,
    resize: ResizeWorkspace,
    db: db::Workspace,
    padded: Vec<u8>,
}
impl Detector {
    pub fn fork(&self) -> ImageResult<Self> {
        let mut fork = Self::new(
            self.compiled.clone(),
            self.options,
            self.session.execution_limits(),
        )?;
        fork.set_execution_context(
            self.session.cpu_context().clone(),
            self.session.intra_op_threads(),
        )?;
        Ok(fork)
    }
    pub fn set_execution_context(
        &mut self,
        cpu: Arc<crate::cpu::CpuContext>,
        threads: usize,
    ) -> ImageResult<()> {
        self.session
            .set_execution_context(cpu, threads)
            .map_err(Into::into)
    }
    pub fn new(
        compiled: Arc<CompiledModel>,
        mut options: DetectorOptions,
        limits: ExecutionLimits,
    ) -> ImageResult<Self> {
        options.validate()?;
        if options.box_threshold.is_none() {
            let channels = compiled
                .model()
                .graph()
                .nodes
                .iter()
                .find(|n| matches!(n.operator, Operator::Conv(_)))
                .and_then(|n| n.inputs.get(1).copied().flatten())
                .and_then(|id| compiled.model().graph().tensors[id.0].constant.as_ref())
                .and_then(|w| w.dimensions().first().copied());
            options.box_threshold = Some(if channels.is_some_and(|c| c <= 16) {
                0.4
            } else {
                0.45
            });
        }
        let session = Session::new_serial(compiled.clone(), &[1, 3, 32, 32], limits)?;
        if session.output_shape().dimensions() != [1, 1, 32, 32] {
            return Err(ImageError::new(
                "detector requires a [1,1,H,W] probability output matching input spatial dimensions",
            ));
        }
        Ok(Self {
            compiled,
            options,
            session,
            resize: ResizeWorkspace::default(),
            db: db::Workspace::default(),
            padded: Vec::new(),
        })
    }
    pub fn resolved_options(&self) -> DetectorOptions {
        self.options
    }
    /// Change image-dependent resizing without recompiling the model. Invalid
    /// options leave the reusable detector unchanged.
    pub fn set_resize_options(&mut self, policy: DetResizePolicy, limit: usize) -> ImageResult<()> {
        let options = DetectorOptions {
            resize_policy: policy,
            limit_side_length: limit,
            ..self.options
        };
        options.validate()?;
        self.options = options;
        Ok(())
    }
    pub fn compiled(&self) -> &Arc<CompiledModel> {
        &self.compiled
    }
    pub fn execution_limits(&self) -> ExecutionLimits {
        self.session.execution_limits()
    }
    pub fn activation_bytes(&self) -> usize {
        self.session.activation_bytes()
    }
    pub fn retained_activation_bytes(&self) -> usize {
        self.session.retained_activation_bytes()
    }
    pub fn retained_workspace_bytes(&self) -> usize {
        self.resize.retained_bytes()
            + self.db.retained_bytes()
            + self.padded.capacity()
            + (self.session.retained_planning_bytes() + self.session.retained_scratch_bytes())
    }
    /// Retained kernel buffers, already included in retained workspace bytes.
    pub fn retained_kernel_scratch_bytes(&self) -> usize {
        self.session.retained_scratch_bytes()
    }
    /// Replay a frozen detector map through the same reusable DB workspace,
    /// including implicit source padding, without inference.
    #[allow(clippy::too_many_arguments)]
    pub fn postprocess(
        &mut self,
        probabilities: &[f32],
        map_width: usize,
        map_height: usize,
        source: ImageView<'_>,
        width_ratio: f32,
        height_ratio: f32,
    ) -> ImageResult<DetectionView<'_>> {
        if source.pixel_count()? > self.options.max_image_pixels
            || !width_ratio.is_finite()
            || !height_ratio.is_finite()
            || width_ratio <= 0.
            || height_ratio <= 0.
        {
            return Err(ImageError::new(
                "invalid postprocess image or resize ratios",
            ));
        }
        db::run(
            probabilities,
            map_width,
            map_height,
            &self.options,
            DetSourceGeometry::new(source.width(), source.height())?,
            &mut self.db,
        )?;
        Ok(DetectionView {
            boxes: &self.db.boxes,
            resized_width: map_width,
            resized_height: map_height,
            width_ratio,
            height_ratio,
        })
    }
    pub fn detect(&mut self, image: ImageView<'_>) -> ImageResult<DetectionView<'_>> {
        self.detect_profiled::<false>(image, &mut StageProfile::default())
    }
    pub(crate) fn detect_profiled<const PROFILE: bool>(
        &mut self,
        image: ImageView<'_>,
        profile: &mut StageProfile,
    ) -> ImageResult<DetectionView<'_>> {
        let cpu = self.session.cpu_context().clone();
        cpu.install(|| self.detect_impl::<false, PROFILE, _>(image, |_| Ok(()), profile))
    }
    pub fn detect_with_trace<F>(
        &mut self,
        image: ImageView<'_>,
        observer: F,
    ) -> ImageResult<DetectionView<'_>>
    where
        F: for<'a> FnMut(DetectionTrace<'a>) -> ImageResult<()>,
    {
        self.detect_impl::<true, false, F>(image, observer, &mut StageProfile::default())
    }
    fn detect_impl<const TRACE: bool, const PROFILE: bool, F>(
        &mut self,
        image: ImageView<'_>,
        observer: F,
        profile: &mut StageProfile,
    ) -> ImageResult<DetectionView<'_>>
    where
        F: for<'a> FnMut(DetectionTrace<'a>) -> ImageResult<()>,
    {
        #[cfg(test)]
        if !TRACE {
            self.session.cpu_context().assert_production_thread("DET");
        }
        let binding_timer = Timer::<PROFILE>::start();
        if image.pixel_count()? > self.options.max_image_pixels {
            return Err(ImageError::new("source image exceeds max_image_pixels"));
        }
        let geometry = DetSourceGeometry::new(image.width(), image.height())?;
        let size = geometry.resize(self.options.limit_side_length, self.options.resize_policy)?;
        // Reshape is transactional; no processing starts before a valid binding exists.
        if PROFILE {
            profile.binding_ns = profile.binding_ns.saturating_add(binding_timer.elapsed());
        }
        let reshape_timer = Timer::<PROFILE>::start();
        let reshape = self.session.reshape(&[1, 3, size.height, size.width]);
        if PROFILE {
            profile.reshape_ns = profile.reshape_ns.saturating_add(reshape_timer.elapsed());
        }
        reshape?;
        let binding_timer = Timer::<PROFILE>::start();
        if self.session.output_shape().dimensions() != [1, 1, size.height, size.width] {
            return Err(ImageError::new(
                "detector output spatial shape differs from resized input",
            ));
        }
        let source = if geometry.requires_padding() {
            let w = geometry.padded_width;
            let h = geometry.padded_height;
            let bpp = image.format().bytes_per_pixel();
            let stride = w * bpp;
            let bytes = stride
                .checked_mul(h)
                .ok_or_else(|| ImageError::new("padded source size overflow"))?;
            resize_vec(&mut self.padded, bytes)?;
            self.padded[..bytes].fill(0);
            for y in 0..geometry.original_height {
                let row = geometry.original_width * bpp;
                self.padded[y * stride..y * stride + row]
                    .copy_from_slice(&image.pixels()[y * image.stride()..y * image.stride() + row]);
            }
            ImageView::new(&self.padded[..bytes], w, h, stride, image.format())?
        } else {
            image
        };
        if PROFILE {
            profile.binding_ns = profile.binding_ns.saturating_add(binding_timer.elapsed());
        }
        let observer = RefCell::new(observer);
        let input_shape = self.session.input_shape();
        let mut preprocessing_ns = 0;
        let cpu = self.session.cpu_context().clone();
        let threads = self.session.intra_op_threads();
        let mut fill = |input: &mut [f32], nhwc: bool| {
            let timer = Timer::<PROFILE>::start();
            crate::preprocess::det_with_context_layout(
                source,
                size.width,
                size.height,
                input,
                &mut self.resize,
                &cpu,
                threads,
                nhwc,
            )
            .map_err(RuntimeError::from)?;
            preprocessing_ns = timer.elapsed();
            if TRACE {
                observer.borrow_mut()(DetectionTrace::Input {
                    shape: input_shape,
                    values: input,
                })
                .map_err(RuntimeError::from)?;
            }
            Ok(())
        };
        let inference_timer = Timer::<PROFILE>::start();
        let prediction = if TRACE {
            self.session.run_with_input_and_trace(
                |input| fill(input, false),
                |node| {
                    observer.borrow_mut()(DetectionTrace::Node(node)).map_err(RuntimeError::from)
                },
            )?
        } else {
            self.session.run_with_layout_input(fill)?
        };
        if PROFILE {
            profile.det_preprocess_ns = profile.det_preprocess_ns.saturating_add(preprocessing_ns);
            profile.det_inference_ns = profile
                .det_inference_ns
                .saturating_add(inference_timer.elapsed().saturating_sub(preprocessing_ns));
        }
        if TRACE {
            observer.borrow_mut()(DetectionTrace::Probabilities {
                shape: prediction.shape(),
                values: prediction.data(),
            })?;
        }
        let postprocess_timer = Timer::<PROFILE>::start();
        db::run(
            prediction.data(),
            size.width,
            size.height,
            &self.options,
            geometry,
            &mut self.db,
        )?;
        if PROFILE {
            profile.det_postprocess_ns = profile
                .det_postprocess_ns
                .saturating_add(postprocess_timer.elapsed());
        }
        let view = DetectionView {
            boxes: &self.db.boxes,
            resized_width: size.width,
            resized_height: size.height,
            width_ratio: size.width_ratio,
            height_ratio: size.height_ratio,
        };
        if TRACE {
            observer.borrow_mut()(DetectionTrace::Boxes(view))?;
        }
        Ok(view)
    }
}

#[cfg(test)]
mod tests;
