//! Native public orientation classification, distinct from dense graph output.
use crate::{
    image::{ImageError, ImageResult, ImageView},
    math::exp_f32,
    model::Graph,
    preprocess::{self, ResizeWorkspace},
    profile::{StageProfile, Timer},
    projection::ProjectionHead,
    runtime::{CompiledModel, ExecutionLimits, NodeTrace, RuntimeError, Session},
    tensor::TensorData,
};
use serde::Serialize;
use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct ClassificationResult {
    pub label: u32,
    pub score: f32,
    pub resized_width: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ClassifierOptions {
    pub max_image_pixels: usize,
}
impl Default for ClassifierOptions {
    fn default() -> Self {
        Self {
            max_image_pixels: 40_000_000,
        }
    }
}
impl ClassificationResult {
    pub fn orientation_degrees(self) -> u16 {
        if self.label == 0 { 0 } else { 180 }
    }
}

pub(crate) enum ClassifierTrace<'a> {
    Input {
        values: &'a [f32],
        resized_width: usize,
    },
    Node(NodeTrace<'a>),
    Probabilities(&'a [f32]),
    Public(ClassificationResult),
}

impl ProjectionHead {
    fn evaluate(
        &self,
        graph: &Graph,
        feature: &[f32],
        resized_width: usize,
    ) -> ImageResult<ClassificationResult> {
        if feature.len() != self.inner {
            return Err(ImageError::new("classifier feature must contain one row"));
        }
        let weights = match &graph.tensors[self.weights.0].constant {
            Some(TensorData::F32(t)) => t.data(),
            _ => unreachable!(),
        };
        let bias = self.bias.map(|id| match &graph.tensors[id.0].constant {
            Some(TensorData::F32(t)) => t.data(),
            _ => unreachable!(),
        });
        public_head(feature, weights, bias, resized_width)
    }
}

fn public_head(
    feature: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    resized_width: usize,
) -> ImageResult<ClassificationResult> {
    if weights.len()
        != feature
            .len()
            .checked_mul(2)
            .ok_or_else(|| ImageError::new("classifier head size overflow"))?
        || bias.is_some_and(|b| b.len() != 2)
    {
        return Err(ImageError::new("invalid classifier head dimensions"));
    }
    let mut logits = [0f32; 2];
    for column in 0..2 {
        let mut sum = bias.map_or(0., |b| b[column]);
        for (index, &value) in feature.iter().enumerate() {
            sum += value * weights[index * 2 + column];
        }
        logits[column] = sum;
    }
    if logits.iter().any(|v| !v.is_finite()) {
        return Err(ImageError::new("nonfinite public classifier logits"));
    }
    let label = u32::from(logits[1] > logits[0]);
    let maximum = logits[0].max(logits[1]);
    let e0 = exp_f32(logits[0] - maximum);
    let e1 = exp_f32(logits[1] - maximum);
    let mut sum = 0f32;
    sum += e0;
    sum += e1;
    let score = if label == 0 { e0 } else { e1 } / sum;
    Ok(ClassificationResult {
        label,
        score,
        resized_width,
    })
}

#[derive(Clone)]
pub struct ClassifierPlan {
    execution: crate::runtime::stage::RequestSettings,
    compiled: CompiledModel,
    options: ClassifierOptions,
    limits: ExecutionLimits,
    head: ProjectionHead,
}
pub struct Classifier {
    plan: Arc<ClassifierPlan>,
    #[cfg(feature = "gpu")]
    gpu: Option<crate::runtime::gpu::heads::Classification>,
    session: Session,
    diagnostic: Option<Session>,
    resize: ResizeWorkspace,
}
impl Classifier {
    pub fn plan(&self) -> &Arc<ClassifierPlan> {
        &self.plan
    }
    pub fn uses_gpu(&self) -> bool {
        self.session.is_gpu()
    }
    #[cfg(feature = "gpu")]
    pub(crate) fn is_full_gpu(&self) -> bool {
        self.session.is_gpu()
    }
    #[cfg(feature = "gpu")]
    pub(crate) fn resident(
        &mut self,
    ) -> crate::runtime::RuntimeResult<&mut crate::runtime::gpu::heads::Classification> {
        if self.gpu.is_none() {
            self.gpu = Some(crate::runtime::gpu::heads::Classification::new(
                self.plan.compiled.as_gpu().unwrap().clone(),
                self.session.execution_limits(),
            )?);
        }
        Ok(self.gpu.as_mut().unwrap())
    }
    #[cfg(feature = "gpu")]
    pub(crate) fn reset_run_statistics(&mut self) {
        if let Some(gpu) = &mut self.gpu {
            gpu.statistics = Default::default();
        }
    }
    #[cfg(feature = "gpu")]
    pub fn gpu_pipeline_statistics(&self) -> Option<&crate::runtime::gpu::GpuPipelineStatistics> {
        self.gpu.as_ref().map(|g| &g.statistics)
    }
    pub fn compiled(&self) -> &CompiledModel {
        &self.plan.compiled
    }
    pub(crate) fn counter_snapshot(&self) -> crate::runtime::StatisticsCounters {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu {
            return gpu.session.counter_snapshot();
        }
        self.session.counter_snapshot()
    }
    pub fn execution_statistics(&self) -> crate::runtime::ExecutionStatistics {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu {
            return gpu.session.statistics();
        }
        self.session.statistics()
    }
    pub fn new(compiled: impl Into<CompiledModel>, limits: ExecutionLimits) -> ImageResult<Self> {
        Self::new_with_options(compiled, ClassifierOptions::default(), limits)
    }
    fn validate(
        compiled: impl Into<CompiledModel>,
        options: ClassifierOptions,
        limits: ExecutionLimits,
    ) -> ImageResult<ClassifierPlan> {
        let compiled = compiled.into();
        if options.max_image_pixels == 0 {
            return Err(ImageError::new(
                "classifier max_image_pixels must be positive",
            ));
        }
        let head = ProjectionHead::compile(compiled.model().graph(), 2, 2, "classifier")?;
        let output = compiled.validate_shape(&[1, 3, 80, 160], limits, Some(head.feature_node))?;
        if output.element_count() != head.inner {
            return Err(ImageError::new("classifier feature must contain one row"));
        }
        Ok(ClassifierPlan {
            execution: Default::default(),
            compiled,
            options,
            limits,
            head,
        })
    }
    pub fn new_with_options(
        compiled: impl Into<CompiledModel>,
        options: ClassifierOptions,
        limits: ExecutionLimits,
    ) -> ImageResult<Self> {
        ClassifierPlan::new(compiled, options, limits)?.request()
    }
    fn from_plan(plan: Arc<ClassifierPlan>) -> ImageResult<Self> {
        let mut session = Session::deferred(
            plan.compiled.clone(),
            &[1, 3, 80, 160],
            plan.limits,
            Some(plan.head.feature_node),
        )?;
        session.set_execution_context(plan.execution.cpu.clone(), plan.execution.threads)?;
        Ok(Self {
            plan,
            #[cfg(feature = "gpu")]
            gpu: None,
            session,
            diagnostic: None,
            resize: ResizeWorkspace::default(),
        })
    }
    pub fn fork(&self) -> ImageResult<Self> {
        self.plan.request()
    }
    pub fn set_execution_context(
        &mut self,
        cpu: Arc<crate::cpu::CpuContext>,
        threads: usize,
    ) -> ImageResult<()> {
        if threads == 0 || threads > cpu.capacity() {
            return Err(ImageError::new(
                "intra-op threads must be within CPU context capacity",
            ));
        }
        self.session.set_execution_context(cpu.clone(), 1)?;
        if let Some(diagnostic) = &mut self.diagnostic {
            diagnostic.set_execution_context(cpu.clone(), 1)?;
        }
        let execution = &mut Arc::make_mut(&mut self.plan).execution;
        execution.cpu = cpu;
        execution.threads = 1;
        Ok(())
    }
    pub fn classify(&mut self, image: ImageView<'_>) -> ImageResult<ClassificationResult> {
        self.classify_profiled::<false>(image, &mut StageProfile::default())
    }
    pub(crate) fn classify_profiled<const PROFILE: bool>(
        &mut self,
        image: ImageView<'_>,
        profile: &mut StageProfile,
    ) -> ImageResult<ClassificationResult> {
        let cpu = self.session.cpu_context().clone();
        if self.session.is_gpu() {
            return self.classify_impl::<PROFILE>(image, profile);
        }
        cpu.install(|| self.classify_impl::<PROFILE>(image, profile))
    }
    fn classify_impl<const PROFILE: bool>(
        &mut self,
        image: ImageView<'_>,
        profile: &mut StageProfile,
    ) -> ImageResult<ClassificationResult> {
        self.check_image(image)?;
        #[cfg(feature = "gpu")]
        if self.session.is_gpu() {
            let candidate = (|| -> crate::runtime::RuntimeResult<_> {
                if self.gpu.is_none() {
                    self.gpu = Some(crate::runtime::gpu::heads::Classification::new(
                        self.plan.compiled.as_gpu().unwrap().clone(),
                        self.session.execution_limits(),
                    )?);
                }
                self.gpu
                    .as_mut()
                    .unwrap()
                    .run(image, PROFILE)
                    .and_then(|(bytes, width)| parse_gpu_classification(&bytes, width))
            })();
            let result = match candidate {
                Ok(result) => result,
                Err(error) => {
                    let compiled = self.plan.compiled.resident_fallback(error)?;
                    let mut replacement = Self::new_with_options(
                        compiled,
                        self.plan.options,
                        self.session.execution_limits(),
                    )?;
                    replacement.set_execution_context(
                        self.session.cpu_context().clone(),
                        self.session.intra_op_threads(),
                    )?;
                    *self = replacement;
                    return self.classify_profiled::<PROFILE>(image, profile);
                }
            };
            let gpu = self.gpu.as_ref().unwrap();
            if PROFILE {
                profile.gpu_submission_ns += gpu.statistics.submission_ns;
                profile.gpu_completion_wait_ns += gpu.statistics.completion_wait_ns;
                profile.gpu_readback_ns += gpu.statistics.readback_ns;
            }
            return Ok(result);
        }
        #[cfg(test)]
        if !self.session.is_gpu() {
            self.session.cpu_context().assert_production_thread("CLS");
        }
        let mut width = 0;
        let resize = &mut self.resize;
        let backbone_timer = Timer::<PROFILE>::start();
        let mut preprocessing_ns = 0;
        let feature =
            self.session
                .run_with_layout_input_profiled::<PROFILE, _>(|input, nhwc| {
                    let timer = Timer::<PROFILE>::start();
                    width = preprocess::cls_with_layout(image, input, resize, nhwc)
                        .map_err(RuntimeError::from)?;
                    preprocessing_ns = timer.elapsed();
                    Ok(())
                })?;
        if PROFILE {
            profile.cls_preprocess_ns = profile.cls_preprocess_ns.saturating_add(preprocessing_ns);
            profile.cls_backbone_ns = profile
                .cls_backbone_ns
                .saturating_add(backbone_timer.elapsed().saturating_sub(preprocessing_ns));
        }
        let head_timer = Timer::<PROFILE>::start();
        let result =
            self.plan
                .head
                .evaluate(self.plan.compiled.model().graph(), feature.data(), width);
        self.session.accumulate_gpu_profile::<PROFILE>(profile);
        if PROFILE {
            profile.cls_head_ns = profile.cls_head_ns.saturating_add(head_timer.elapsed());
        }
        result
    }
    pub(crate) fn classify_with_trace<F>(
        &mut self,
        image: ImageView<'_>,
        observer: F,
    ) -> ImageResult<ClassificationResult>
    where
        F: for<'a> FnMut(ClassifierTrace<'a>) -> ImageResult<()>,
    {
        self.check_image(image)?;
        if self.diagnostic.is_none() {
            let mut diagnostic = Session::new_diagnostic(
                self.plan.compiled.clone(),
                &[1, 3, 80, 160],
                self.session.execution_limits(),
            )?;
            diagnostic.set_execution_context(
                self.session.cpu_context().clone(),
                self.session.intra_op_threads(),
            )?;
            self.diagnostic = Some(diagnostic);
        }
        let observer = RefCell::new(observer);
        let width = Cell::new(0);
        let public = Cell::new(None);
        let resize = &mut self.resize;
        let head = &self.plan.head;
        let graph = self.plan.compiled.model().graph();
        let probabilities = self.diagnostic.as_mut().unwrap().run_with_input_and_trace(
            |input| {
                let actual = preprocess::cls(image, input, resize).map_err(RuntimeError::from)?;
                width.set(actual);
                observer.borrow_mut()(ClassifierTrace::Input {
                    values: input,
                    resized_width: actual,
                })
                .map_err(RuntimeError::from)?;
                Ok(())
            },
            |node| {
                if node.output_id == head.feature {
                    public.set(Some(
                        head.evaluate(graph, node.values, width.get())
                            .map_err(RuntimeError::from)?,
                    ));
                }
                observer.borrow_mut()(ClassifierTrace::Node(node)).map_err(RuntimeError::from)
            },
        )?;
        observer.borrow_mut()(ClassifierTrace::Probabilities(probabilities.data()))?;
        let result = public
            .get()
            .ok_or_else(|| ImageError::new("classifier feature was not observed"))?;
        observer.borrow_mut()(ClassifierTrace::Public(result))?;
        Ok(result)
    }
    pub(crate) fn activation_bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        let resident = self
            .gpu
            .as_ref()
            .map_or(0, |g| g.session.activation_bytes());
        #[cfg(not(feature = "gpu"))]
        let resident = 0;
        resident + self.session.activation_bytes()
    }
    pub fn options(&self) -> ClassifierOptions {
        self.plan.options
    }
    fn check_image(&self, image: ImageView<'_>) -> ImageResult<()> {
        if image.pixel_count()? > self.plan.options.max_image_pixels {
            return Err(ImageError::new(
                "source image exceeds classifier max_image_pixels",
            ));
        }
        Ok(())
    }
    pub fn retained_device_bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        let resident = self.gpu.as_ref().map_or(0, |g| g.retained_bytes());
        #[cfg(not(feature = "gpu"))]
        let resident = 0;
        resident
            + self.session.retained_device_bytes()
            + self
                .diagnostic
                .as_ref()
                .map_or(0, Session::retained_device_bytes)
    }
    pub(crate) fn retained_activation_bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        let resident = self
            .gpu
            .as_ref()
            .map_or(0, |g| g.session.retained_activation_bytes());
        #[cfg(not(feature = "gpu"))]
        let resident = 0;
        resident
            + self.session.retained_activation_bytes()
            + self
                .diagnostic
                .as_ref()
                .map_or(0, Session::retained_activation_bytes)
    }
    pub(crate) fn retained_workspace_bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        let resident = self
            .gpu
            .as_ref()
            .map_or(0, |g| g.session.retained_planning_bytes());
        #[cfg(not(feature = "gpu"))]
        let resident = 0;
        resident
            + self.resize.retained_bytes()
            + (self.session.retained_planning_bytes() + self.session.retained_scratch_bytes())
            + self.diagnostic.as_ref().map_or(0, |s| {
                s.retained_planning_bytes() + s.retained_scratch_bytes()
            })
    }
    /// Retained kernel buffers, excluding fixed local stack tiles.
    pub fn retained_kernel_scratch_bytes(&self) -> usize {
        self.session.retained_scratch_bytes()
            + self
                .diagnostic
                .as_ref()
                .map_or(0, Session::retained_scratch_bytes)
    }
}

pub fn should_rotate(result: ClassificationResult, threshold: f32) -> bool {
    result.label == 1 && result.score > threshold
}

impl ClassifierPlan {
    pub fn new(
        compiled: impl Into<CompiledModel>,
        options: ClassifierOptions,
        limits: ExecutionLimits,
    ) -> ImageResult<Arc<Self>> {
        Ok(Arc::new(Classifier::validate(compiled, options, limits)?))
    }
    pub fn request(self: &Arc<Self>) -> ImageResult<Classifier> {
        Classifier::from_plan(self.clone())
    }
    pub fn options(&self) -> ClassifierOptions {
        self.options
    }
    pub fn compiled(&self) -> &CompiledModel {
        &self.compiled
    }
}

#[cfg(feature = "gpu")]
pub(crate) fn parse_gpu_classification(
    bytes: &[u8],
    resized_width: usize,
) -> crate::runtime::RuntimeResult<ClassificationResult> {
    let label = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let score = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if label > 1 || !score.is_finite() || u32::from_le_bytes(bytes[12..16].try_into().unwrap()) != 0
    {
        return Err(RuntimeError::new("invalid GPU classifier result"));
    }
    Ok(crate::orientation::ClassificationResult {
        label,
        score,
        resized_width,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn standalone_limits_apply_before_production_or_diagnostic_work_and_survive_forks() {
        let compiled = crate::ocr::tests::synthetic_classifier();
        assert!(
            Classifier::new_with_options(
                compiled.clone(),
                ClassifierOptions {
                    max_image_pixels: 0
                },
                ExecutionLimits::default()
            )
            .is_err()
        );
        let mut classifier = Classifier::new_with_options(
            compiled,
            ClassifierOptions {
                max_image_pixels: 1,
            },
            ExecutionLimits::default(),
        )
        .unwrap();
        let pixels = [0u8; 6];
        let image = ImageView::new(&pixels, 2, 1, 0, crate::image::PixelFormat::Bgr24).unwrap();
        assert!(classifier.classify(image).is_err());
        assert!(
            classifier
                .classify_with_trace(image, |_| panic!("invalid input must not invoke observers"))
                .is_err()
        );
        assert!(classifier.diagnostic.is_none());
        let mut fork = classifier.fork().unwrap();
        assert_eq!(fork.options().max_image_pixels, 1);
        assert!(fork.classify(image).is_err());
        let valid =
            ImageView::new(&pixels[..3], 1, 1, 0, crate::image::PixelFormat::Bgr24).unwrap();
        assert_eq!(
            classifier.classify(valid).unwrap(),
            fork.classify(valid).unwrap()
        );
    }
    #[test]
    fn classifier_validates_requests_and_keeps_production_and_diagnostics_single_threaded() {
        let mut classifier = Classifier::new(
            crate::ocr::tests::synthetic_classifier(),
            ExecutionLimits::default(),
        )
        .unwrap();
        #[cfg(feature = "parallel")]
        let (capacity, requests, invalid) = (32, &[1, 16, 17, 32][..], [0, 33]);
        #[cfg(not(feature = "parallel"))]
        let (capacity, requests, invalid) = (1, &[1][..], [0, 2]);
        let cpu = Arc::new(crate::cpu::CpuContext::new(capacity).unwrap());
        let original = classifier.session.cpu_context().clone();
        for requested in invalid {
            assert!(
                classifier
                    .set_execution_context(cpu.clone(), requested)
                    .is_err()
            );
            assert!(Arc::ptr_eq(classifier.session.cpu_context(), &original));
            assert_eq!(classifier.session.intra_op_threads(), 1);
            assert!(classifier.diagnostic.is_none());
        }
        classifier
            .set_execution_context(cpu.clone(), capacity)
            .unwrap();
        let pixels = crate::ocr::tests::two_lines();
        let image = ImageView::new(&pixels, 96, 96, 0, crate::image::PixelFormat::Bgr24).unwrap();
        classifier.classify_with_trace(image, |_| Ok(())).unwrap();
        for &requested in requests {
            classifier
                .set_execution_context(cpu.clone(), requested)
                .unwrap();
            assert_eq!(classifier.session.intra_op_threads(), 1);
            assert!(Arc::ptr_eq(classifier.session.cpu_context(), &cpu));
            let diagnostic = classifier.diagnostic.as_ref().unwrap();
            assert_eq!(diagnostic.intra_op_threads(), 1);
            assert!(Arc::ptr_eq(diagnostic.cpu_context(), &cpu));
        }
        let serial = crate::cpu::CpuContext::serial();
        assert!(classifier.set_execution_context(serial, 2).is_err());
        assert!(Arc::ptr_eq(classifier.session.cpu_context(), &cpu));
        assert_eq!(classifier.session.intra_op_threads(), 1);
        let diagnostic = classifier.diagnostic.as_ref().unwrap();
        assert!(Arc::ptr_eq(diagnostic.cpu_context(), &cpu));
        assert_eq!(diagnostic.intra_op_threads(), 1);
        let production = classifier.classify(image).unwrap();
        let traced = classifier.classify_with_trace(image, |_| Ok(())).unwrap();
        assert_eq!(production, traced);
    }
    #[test]
    fn public_head_preserves_bias_order_and_logit_winner() {
        let r = public_head(&[1., 1.], &[1e8, 0., -1e8, 0.], Some(&[1., 0.5]), 160).unwrap();
        assert_eq!(r.label, 1);
        let r = public_head(&[1.], &[0., 1e-8], None, 80).unwrap();
        assert_eq!(r.label, 1);
        assert_eq!(r.score, 0.5);
        assert!(!should_rotate(r, 0.5));
        assert!(should_rotate(r, f32::from_bits(0.5f32.to_bits() - 1)));
        assert_eq!(public_head(&[1.], &[1., 1.], None, 80).unwrap().label, 0);
    }
    #[test]
    fn malformed_or_nonfinite_heads_fail() {
        assert!(public_head(&[1.], &[1.], None, 160).is_err());
        assert!(public_head(&[1.], &[1., 1.], Some(&[]), 160).is_err());
        assert!(public_head(&[f32::INFINITY], &[1., 1.], None, 160).is_err());
    }
}
