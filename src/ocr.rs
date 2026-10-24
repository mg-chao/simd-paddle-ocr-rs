//! Pixel-to-text composition with borrowed results and bounded CPU workers.
#[cfg(feature = "gpu")]
pub(crate) mod gpu;
use crate::profile::{StageProfile, Timer};
use crate::{
    cpu::{CpuBudgets, CpuContext, CpuOptions},
    detector::{DetectionBox, DetectionTrace},
    image::{ImageError, ImageResult, ImageView},
    lines::{LineTrace, OrientedLineExtractor, OrientedLines},
    recognizer::{
        CtcRun, CtcSpans, OwnedCtcSpan, RecognitionTrace, RecognitionView, Recognizer, copy_spans,
        copy_text,
    },
};
use serde::Serialize;
use std::{
    cell::RefCell,
    mem::size_of,
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

#[derive(Clone, Copy, Debug)]
pub struct OcrOptions {
    /// Active joined UTF-8, line metadata and optional span records.
    pub max_total_result_bytes: usize,
}
impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            max_total_result_bytes: i32::MAX as usize,
        }
    }
}
#[derive(Clone, Copy, Debug)]
struct LineRecord {
    detection: DetectionBox,
    text_start: usize,
    text_end: usize,
    recognition_score: f32,
    classification_score: f32,
    classification_label: u32,
    applied_rotation_degrees: u16,
    emitted_count: usize,
    recognition_content_width: usize,
    recognition_tensor_width: usize,
    recognition_time_steps: usize,
    spans: Option<(usize, usize)>,
}
#[cfg(feature = "gpu")]
pub(crate) fn line_record_bytes() -> usize {
    size_of::<LineRecord>()
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct OcrLineView<'a> {
    pub detection: DetectionBox,
    pub text: &'a str,
    pub recognition_score: f32,
    pub classification_score: f32,
    pub classification_label: u32,
    pub applied_rotation_degrees: u16,
    pub emitted_count: usize,
    pub recognition_content_width: usize,
    pub recognition_tensor_width: usize,
    pub recognition_time_steps: usize,
    pub ctc_spans: Option<CtcSpans<'a>>,
}
impl OcrLineView<'_> {
    pub fn try_to_owned(self) -> ImageResult<OwnedOcrLine> {
        Ok(OwnedOcrLine {
            detection: self.detection,
            text: copy_text(self.text)?,
            recognition_score: self.recognition_score,
            classification_score: self.classification_score,
            classification_label: self.classification_label,
            applied_rotation_degrees: self.applied_rotation_degrees,
            emitted_count: self.emitted_count,
            recognition_content_width: self.recognition_content_width,
            recognition_tensor_width: self.recognition_tensor_width,
            recognition_time_steps: self.recognition_time_steps,
            ctc_spans: copy_spans(self.ctc_spans)?,
        })
    }
}
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct OwnedOcrLine {
    pub detection: DetectionBox,
    pub text: String,
    pub recognition_score: f32,
    pub classification_score: f32,
    pub classification_label: u32,
    pub applied_rotation_degrees: u16,
    pub emitted_count: usize,
    pub recognition_content_width: usize,
    pub recognition_tensor_width: usize,
    pub recognition_time_steps: usize,
    pub ctc_spans: Option<Vec<OwnedCtcSpan>>,
}
impl OwnedOcrLine {
    pub fn as_view(&self) -> OcrLineView<'_> {
        OcrLineView {
            detection: self.detection,
            text: &self.text,
            recognition_score: self.recognition_score,
            classification_score: self.classification_score,
            classification_label: self.classification_label,
            applied_rotation_degrees: self.applied_rotation_degrees,
            emitted_count: self.emitted_count,
            recognition_content_width: self.recognition_content_width,
            recognition_tensor_width: self.recognition_tensor_width,
            recognition_time_steps: self.recognition_time_steps,
            ctc_spans: self.ctc_spans.as_deref().map(CtcSpans::from_owned),
        }
    }
}
#[derive(Clone, Copy, Debug)]
enum LineSource<'a> {
    Slab(&'a [LineRecord], &'a str, &'a [CtcRun]),
    Owned(&'a [OwnedOcrLine]),
}
#[derive(Clone, Copy, Debug)]
pub struct OcrView<'a> {
    /// Every detected line joined by LF, including empty recognition results.
    pub text: &'a str,
    pub detected_count: usize,
    pub detector_resized_width: usize,
    pub detector_resized_height: usize,
    pub width_ratio: f32,
    pub height_ratio: f32,
    source: LineSource<'a>,
}
impl<'a> OcrView<'a> {
    pub fn text(self) -> &'a str {
        self.text
    }
    pub fn detected_count(self) -> usize {
        self.detected_count
    }
    pub fn detector_resized_width(self) -> usize {
        self.detector_resized_width
    }
    pub fn detector_resized_height(self) -> usize {
        self.detector_resized_height
    }
    pub fn len(self) -> usize {
        match self.source {
            LineSource::Slab(lines, _, _) => lines.len(),
            LineSource::Owned(lines) => lines.len(),
        }
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    pub fn get(self, index: usize) -> Option<OcrLineView<'a>> {
        match self.source {
            LineSource::Owned(lines) => lines.get(index).map(OwnedOcrLine::as_view),
            LineSource::Slab(lines, text, runs) => {
                let l = lines.get(index)?;
                Some(OcrLineView {
                    detection: l.detection,
                    text: &text[l.text_start..l.text_end],
                    recognition_score: l.recognition_score,
                    classification_score: l.classification_score,
                    classification_label: l.classification_label,
                    applied_rotation_degrees: l.applied_rotation_degrees,
                    emitted_count: l.emitted_count,
                    recognition_content_width: l.recognition_content_width,
                    recognition_tensor_width: l.recognition_tensor_width,
                    recognition_time_steps: l.recognition_time_steps,
                    ctc_spans: l.spans.map(|(a, b)| CtcSpans::from_runs(text, &runs[a..b])),
                })
            }
        }
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = OcrLineView<'a>> {
        (0..self.len()).map(move |i| self.get(i).expect("validated line index"))
    }
    pub fn try_to_owned(self) -> ImageResult<OwnedOcr> {
        let mut lines = Vec::new();
        lines.try_reserve_exact(self.len()).map_err(allocation)?;
        for line in self.iter() {
            lines.push(line.try_to_owned()?);
        }
        Ok(OwnedOcr {
            text: copy_text(self.text)?,
            lines,
            detected_count: self.detected_count,
            detector_resized_width: self.detector_resized_width,
            detector_resized_height: self.detector_resized_height,
            width_ratio: self.width_ratio,
            height_ratio: self.height_ratio,
        })
    }
}
impl Serialize for OcrView<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        struct Lines<'a>(OcrView<'a>);
        impl Serialize for Lines<'_> {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.collect_seq(self.0.iter())
            }
        }
        let mut result = serializer.serialize_struct("OcrView", 7)?;
        result.serialize_field("text", self.text)?;
        result.serialize_field("detected_count", &self.detected_count)?;
        result.serialize_field("detector_resized_width", &self.detector_resized_width)?;
        result.serialize_field("detector_resized_height", &self.detector_resized_height)?;
        result.serialize_field("width_ratio", &self.width_ratio)?;
        result.serialize_field("height_ratio", &self.height_ratio)?;
        result.serialize_field("lines", &Lines(*self))?;
        result.end()
    }
}
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct OwnedOcr {
    pub text: String,
    pub lines: Vec<OwnedOcrLine>,
    pub detected_count: usize,
    pub detector_resized_width: usize,
    pub detector_resized_height: usize,
    pub width_ratio: f32,
    pub height_ratio: f32,
}
impl OwnedOcr {
    pub fn as_view(&self) -> OcrView<'_> {
        OcrView {
            text: &self.text,
            detected_count: self.detected_count,
            detector_resized_width: self.detector_resized_width,
            detector_resized_height: self.detector_resized_height,
            width_ratio: self.width_ratio,
            height_ratio: self.height_ratio,
            source: LineSource::Owned(&self.lines),
        }
    }
}
pub enum OcrTrace<'a> {
    Detection(DetectionTrace<'a>),
    Line(LineTrace<'a>),
    Recognition {
        index: usize,
        event: RecognitionTrace<'a>,
    },
    Result(OcrView<'a>),
}
fn allocation(error: std::collections::TryReserveError) -> ImageError {
    ImageError::caused_by(
        crate::error::ErrorKind::Allocation,
        "OCR result allocation failed",
        error,
    )
}

#[derive(Default)]
struct ResultWorkspace {
    lines: Vec<LineRecord>,
    text: String,
    runs: Vec<CtcRun>,
    detected_count: usize,
    detector_resized_width: usize,
    detector_resized_height: usize,
    width_ratio: f32,
    height_ratio: f32,
}
#[cfg(feature = "gpu")]
impl ResultWorkspace {
    fn clear(&mut self) {
        self.lines.clear();
        self.text.clear();
        self.runs.clear();
    }
}

/// Immutable OCR configuration and shared stage resources. Cloning never clones arenas.
#[derive(Clone)]
pub struct OcrPlan {
    extractor: Arc<crate::lines::OrientedLinePlan>,
    recognizer: Arc<crate::recognizer::RecognizerPlan>,
    options: OcrOptions,
    cpu: Arc<CpuContext>,
    cpu_options: CpuOptions,
    recognition_batch_size: usize,
    #[cfg(feature = "gpu")]
    gpu_mode: crate::runtime::gpu::GpuPipelineMode,
}
impl OcrPlan {
    pub fn new(
        extractor: Arc<crate::lines::OrientedLinePlan>,
        recognizer: Arc<crate::recognizer::RecognizerPlan>,
        options: OcrOptions,
        cpu: Arc<CpuContext>,
        cpu_options: CpuOptions,
        recognition_batch_size: usize,
    ) -> ImageResult<Self> {
        if options.max_total_result_bytes == 0 {
            return Err(ImageError::new("max_total_result_bytes must be positive"));
        }
        validate_batch_size(recognition_batch_size)?;
        cpu_options.resolve(
            cpu.capacity(),
            extractor
                .detector()
                .compiled()
                .as_cpu()
                .is_some_and(|c| c.options().cpu.layout == crate::runtime::LayoutPolicy::Auto),
        )?;
        let plan = Self {
            extractor,
            recognizer,
            options,
            cpu,
            cpu_options,
            recognition_batch_size,
            #[cfg(feature = "gpu")]
            gpu_mode: Default::default(),
        };
        Ok(plan)
    }
    pub fn with_extractor(
        mut self,
        extractor: Arc<crate::lines::OrientedLinePlan>,
    ) -> ImageResult<Self> {
        self.extractor = extractor;
        #[cfg(feature = "gpu")]
        self.validate_contexts()?;
        Ok(self)
    }
    pub fn with_recognizer(
        mut self,
        recognizer: Arc<crate::recognizer::RecognizerPlan>,
    ) -> ImageResult<Self> {
        self.recognizer = recognizer;
        #[cfg(feature = "gpu")]
        self.validate_contexts()?;
        Ok(self)
    }
    pub fn request(&self) -> ImageResult<OcrRequest> {
        #[cfg(feature = "gpu")]
        self.validate_contexts()?;
        OcrRequest::from_plan(self.clone())
    }
    /// Shared coefficient and packing storage. Excluded from request eviction totals.
    pub fn shared_weights(&self) -> crate::runtime::MemorySnapshot {
        let mut memory = crate::runtime::MemorySnapshot::default();
        let mut models = std::collections::HashSet::new();
        let mut packs = std::collections::HashSet::new();
        #[cfg(feature = "gpu")]
        let mut device_resources = std::collections::HashSet::new();
        for compiled in [
            Some(self.extractor.detector().compiled()),
            Some(self.recognizer.compiled()),
            self.extractor.classifier().map(|c| c.compiled()),
        ]
        .into_iter()
        .flatten()
        {
            if models.insert(compiled.model() as *const _ as usize) {
                memory.host_bytes += compiled.model().normalized_weight_bytes() as usize;
            }
            if packs.insert(Arc::as_ptr(&compiled.packed) as usize) {
                memory.host_bytes += compiled.packed.bytes();
            }
            #[cfg(feature = "gpu")]
            if let Some(gpu) = compiled.as_gpu() {
                memory.device_bytes += gpu.unique_weight_bytes(&mut device_resources);
            }
        }
        memory
    }
    pub fn cpu_context(&self) -> &Arc<CpuContext> {
        &self.cpu
    }
    pub fn extractor(&self) -> &Arc<crate::lines::OrientedLinePlan> {
        &self.extractor
    }
    pub fn recognizer(&self) -> &Arc<crate::recognizer::RecognizerPlan> {
        &self.recognizer
    }
    #[cfg(feature = "gpu")]
    fn validate_contexts(&self) -> ImageResult<()> {
        use crate::runtime::gpu::GpuPipelineMode;
        if self.gpu_mode != GpuPipelineMode::Full {
            return Ok(());
        }
        let detector = self.extractor.detector().compiled().as_gpu();
        let classifier = self
            .extractor
            .classifier()
            .and_then(|p| p.compiled().as_gpu());
        if let (Some(d), Some(c)) = (detector, classifier)
            && !Arc::ptr_eq(d.context(), c.context())
        {
            return Err(ImageError::new(
                "resident extraction requires one GPU context",
            ));
        }
        if let (Some(d), Some(r)) = (detector, self.recognizer.compiled().as_gpu())
            && self
                .extractor
                .classifier()
                .is_none_or(|c| c.compiled().is_gpu())
            && !Arc::ptr_eq(d.context(), r.context())
        {
            return Err(ImageError::new("resident OCR requires one GPU context"));
        }
        Ok(())
    }
    #[cfg(feature = "gpu")]
    pub fn with_gpu_pipeline_mode(
        mut self,
        mode: crate::runtime::gpu::GpuPipelineMode,
    ) -> ImageResult<Self> {
        self.gpu_mode = mode;
        self.validate_contexts()?;
        Ok(self)
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct RunOptions {
    pub capture_spans: bool,
}
fn validate_batch_size(size: usize) -> ImageResult<()> {
    if size == 0 || size > i32::MAX as usize {
        return Err(ImageError::new(
            "recognition_batch_size must be within 1..=i32::MAX",
        ));
    }
    Ok(())
}

/// Production orchestration selected by the active request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum OcrExecution {
    Cpu,
    Hybrid,
    Resident,
}
enum ActiveExecution {
    Cpu,
    Hybrid,
    #[cfg(feature = "gpu")]
    Resident(Box<gpu::Pipeline>),
}
impl ActiveExecution {
    fn staged(gpu: bool) -> Self {
        if gpu { Self::Hybrid } else { Self::Cpu }
    }
    fn kind(&self) -> OcrExecution {
        match self {
            Self::Cpu => OcrExecution::Cpu,
            Self::Hybrid => OcrExecution::Hybrid,
            #[cfg(feature = "gpu")]
            Self::Resident(_) => OcrExecution::Resident,
        }
    }
    #[cfg(feature = "gpu")]
    fn resident(&self) -> Option<&gpu::Pipeline> {
        match self {
            Self::Resident(p) => Some(p),
            _ => None,
        }
    }
    #[cfg(feature = "gpu")]
    fn resident_mut(&mut self) -> Option<&mut gpu::Pipeline> {
        match self {
            Self::Resident(p) => Some(p),
            _ => None,
        }
    }
}
/// Placement, timings and traffic for exactly one invocation, including diagnostics.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RunStatistics {
    pub execution: OcrExecution,
    pub succeeded: bool,
    pub stages: std::collections::BTreeMap<
        crate::error::OcrStage,
        Vec<crate::runtime::ExecutionStatistics>,
    >,
    #[cfg(feature = "gpu")]
    pub gpu: Option<crate::runtime::gpu::GpuPipelineStatistics>,
}

impl RunStatistics {
    fn retained_bytes(&self) -> usize {
        let stages = self
            .stages
            .values()
            .map(|sessions| {
                sessions.capacity() * size_of::<crate::runtime::ExecutionStatistics>()
                    + sessions
                        .iter()
                        .map(crate::runtime::ExecutionStatistics::retained_bytes)
                        .sum::<usize>()
            })
            .sum::<usize>();
        #[cfg(feature = "gpu")]
        let gpu = self.gpu.as_ref().map_or(
            0,
            crate::runtime::gpu::GpuPipelineStatistics::retained_bytes,
        );
        #[cfg(not(feature = "gpu"))]
        let gpu = 0;
        stages + gpu
    }
}
/// Exclusive reusable execution and result storage.
///
/// ```compile_fail,E0515
/// use simd_paddle_ocr_rs::{OcrPlan, RunOptions, image::ImageView, ocr::OcrView};
/// fn escape<'a>(plan: &OcrPlan, image: ImageView<'_>) -> OcrView<'a> {
///     let mut request = plan.request().unwrap();
///     request.run(image, RunOptions::default()).unwrap()
/// }
/// ```
///
/// ```compile_fail,E0499
/// use simd_paddle_ocr_rs::{OcrRequest, RunOptions, image::ImageView};
/// fn overwrite(request: &mut OcrRequest, image: ImageView<'_>) {
///     let first = request.run(image, RunOptions::default()).unwrap();
///     request.run(image, RunOptions::default()).unwrap();
///     println!("{}", first.text);
/// }
/// ```
pub struct OcrRequest {
    plan: OcrPlan,
    execution: ActiveExecution,
    latest_run: Option<RunStatistics>,
    counters: crate::runtime::ExecutionCounters,
    extractor: OrientedLineExtractor,
    recognizer: Recognizer,
    result: ResultWorkspace,
    pinned_cpu: Option<Arc<CpuContext>>,
    budgets: CpuBudgets,
    workers: Vec<RecognitionWorker>,
    recognition_results: Vec<Mutex<RecognitionSlot>>,
    recognition_order: Vec<usize>,
    unwind_state: Arc<AtomicBool>,
}
struct RequestUnwindGuard(Arc<AtomicBool>);
impl Drop for RequestUnwindGuard {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.0.store(false, Ordering::Relaxed);
        }
    }
}
struct RecognitionWorker {
    recognizer: Recognizer,
    jobs: Vec<Range<usize>>,
    scheduled_cost: usize,
    error: Option<(usize, ImageError)>,
    profile: StageProfile,
    #[cfg(feature = "profiling")]
    owner_profile: crate::profile::RecognitionOwnerProfile,
}
impl OcrRequest {
    #[cfg(test)]
    pub(crate) fn new_with_execution(
        extractor: OrientedLineExtractor,
        recognizer: Recognizer,
        options: OcrOptions,
        cpu: Arc<CpuContext>,
        cpu_options: CpuOptions,
    ) -> ImageResult<Self> {
        OcrPlan::new(
            extractor.plan().clone(),
            recognizer.plan().clone(),
            options,
            cpu,
            cpu_options,
            1,
        )?
        .request()
    }
    fn from_plan(plan: OcrPlan) -> ImageResult<Self> {
        #[cfg(feature = "gpu")]
        let mut extractor = plan
            .extractor
            .with_gpu_pipeline_mode(plan.gpu_mode)?
            .request()?;
        #[cfg(not(feature = "gpu"))]
        let mut extractor = plan.extractor.request()?;
        let mut recognizer = plan.recognizer.request()?;
        let options = plan.options;
        let cpu = plan.cpu.clone();
        let cpu_options = plan.cpu_options;
        #[cfg(feature = "gpu")]
        {
            extractor.set_gpu_pipeline_mode(plan.gpu_mode)?;
            recognizer.set_gpu_pipeline_mode(plan.gpu_mode);
        }
        if options.max_total_result_bytes == 0 {
            return Err(ImageError::new("max_total_result_bytes must be positive"));
        }
        let budgets = cpu_options.resolve(
            cpu.capacity(),
            extractor
                .detector()
                .compiled()
                .as_cpu()
                .is_some_and(|c| c.options().cpu.layout == crate::runtime::LayoutPolicy::Auto),
        )?;
        extractor.set_execution_context(cpu.clone(), cpu_options)?;
        recognizer.set_execution_context(cpu.clone(), budgets.recognizer_base_threads())?;
        Ok(Self {
            plan,
            execution: ActiveExecution::staged(extractor.uses_gpu() || recognizer.uses_gpu()),
            latest_run: None,
            counters: Default::default(),
            extractor,
            recognizer,
            result: ResultWorkspace::default(),
            pinned_cpu: None,
            budgets,
            workers: Vec::new(),
            recognition_results: Vec::new(),
            recognition_order: Vec::new(),
            unwind_state: Arc::new(AtomicBool::new(false)),
        })
    }
    /// Replace configuration transactionally. A rejected plan leaves this request usable.
    pub fn reconfigure(&mut self, plan: OcrPlan) -> ImageResult<()> {
        if self
            .pinned_cpu
            .as_ref()
            .is_some_and(|cpu| !Arc::ptr_eq(cpu, plan.cpu_context()))
        {
            return Err(ImageError::new("CPU context is pinned to this OCR engine"));
        }
        let mut replacement = plan.request()?;
        replacement.pinned_cpu = self.pinned_cpu.clone();
        replacement.result = std::mem::take(&mut self.result);
        replacement.counters = self.counters;
        *self = replacement;
        Ok(())
    }
    pub fn plan(&self) -> &OcrPlan {
        &self.plan
    }
    pub fn extractor(&self) -> &OrientedLineExtractor {
        &self.extractor
    }
    pub fn recognizer(&self) -> &Recognizer {
        &self.recognizer
    }
    /// A fresh request with independent storage and the same immutable plan.
    pub fn fork(&self) -> ImageResult<Self> {
        self.plan.request()
    }
    pub fn cpu_context(&self) -> &Arc<CpuContext> {
        &self.plan.cpu
    }
    /// Whether any neural stage uses the GPU. Host image work still uses the CPU context.
    pub fn uses_gpu(&self) -> bool {
        self.extractor.uses_gpu() || self.recognizer.uses_gpu()
    }
    #[cfg(feature = "gpu")]
    pub fn set_gpu_pipeline_mode(
        &mut self,
        mode: crate::runtime::gpu::GpuPipelineMode,
    ) -> ImageResult<()> {
        let plan = self.plan.clone().with_gpu_pipeline_mode(mode)?;
        self.extractor.set_gpu_pipeline_mode(mode)?;
        self.plan = plan;
        self.execution = ActiveExecution::staged(self.uses_gpu());
        self.recognizer.set_gpu_pipeline_mode(mode);
        for worker in &mut self.workers {
            worker.recognizer.set_gpu_pipeline_mode(mode);
        }
        Ok(())
    }
    #[cfg(feature = "gpu")]
    fn collect_gpu_statistics(
        &self,
        stages: &std::collections::BTreeMap<
            crate::error::OcrStage,
            Vec<crate::runtime::ExecutionStatistics>,
        >,
        resident: bool,
    ) -> Option<crate::runtime::gpu::GpuPipelineStatistics> {
        use crate::runtime::gpu::{GpuPipelineMode, GpuPipelineStatistics, GpuPlacement};
        if resident && let Some(pipeline) = self.execution.resident() {
            let mut stats = pipeline.statistics.clone();
            stats.diagnostic_readback_bytes = stages
                .values()
                .flatten()
                .map(|s| s.diagnostic_readback_bytes)
                .sum();
            return Some(stats);
        }
        if !self.uses_gpu() {
            return None;
        }
        let full = resident && self.plan.gpu_mode == GpuPipelineMode::Full;
        let det = full && self.extractor.detector().uses_gpu();
        let rec = full && self.recognizer.uses_gpu();
        let cls = full && self.extractor.classifier().is_none_or(|c| c.uses_gpu());
        let extraction = self.extractor.gpu_pipeline_statistics();
        let mut stats = GpuPipelineStatistics {
            mode: self.plan.gpu_mode,
            image_processing: if extraction.is_some() {
                GpuPlacement::NotRun
            } else {
                GpuPlacement::Cpu
            },
            geometry: if det {
                GpuPlacement::Gpu
            } else {
                GpuPlacement::Cpu
            },
            projection: if self.extractor.classifier().is_none() || rec == cls {
                if rec {
                    GpuPlacement::Gpu
                } else {
                    GpuPlacement::Cpu
                }
            } else {
                GpuPlacement::Hybrid
            },
            decoding: if rec {
                GpuPlacement::Gpu
            } else {
                GpuPlacement::Cpu
            },
            ..Default::default()
        };
        if extraction.is_none() {
            stats
                .cpu_compute_stages
                .push("perspective crops and orientation".into());
        }
        if !det {
            stats
                .cpu_compute_stages
                .extend(["detector preprocessing".into(), "DB geometry".into()]);
        }
        if !rec {
            stats.cpu_compute_stages.extend([
                "recognizer preprocessing".into(),
                "recognition projection".into(),
                "CTC decoding".into(),
            ]);
            stats.image_processing = GpuPlacement::Cpu;
        }
        if !cls && self.extractor.classifier().is_some() {
            stats.cpu_compute_stages.extend([
                "classifier preprocessing".into(),
                "classifier projection".into(),
            ]);
        }
        for (name, model) in [
            ("detector", Some(self.extractor.detector().uses_gpu())),
            ("recognizer", Some(self.recognizer.uses_gpu())),
            (
                "classifier",
                self.extractor.classifier().map(|c| c.uses_gpu()),
            ),
        ] {
            if model == Some(false) {
                stats.cpu_compute_stages.push(format!("{name} inference"));
            }
        }
        let detector = extraction.or_else(|| self.extractor.detector().gpu_pipeline_statistics());
        let classifier = if extraction.is_some() {
            None
        } else {
            self.extractor
                .classifier()
                .and_then(|c| c.gpu_pipeline_statistics())
        };
        let recognizer = self.recognizer.gpu_pipeline_statistics();
        for source in [detector, recognizer, classifier].into_iter().flatten() {
            crate::runtime::gpu::vision::merge_statistics(&mut stats, source.clone());
        }
        for (name, stages) in stages {
            for stage in stages {
                let resident = match name {
                    crate::error::OcrStage::Detector => detector.is_some(),
                    crate::error::OcrStage::Classifier => {
                        classifier.is_some() || extraction.is_some()
                    }
                    crate::error::OcrStage::Recognizer => recognizer.is_some(),
                    _ => false,
                };
                if !resident {
                    stats.submissions += stage.production_submissions;
                    stats.uploaded_bytes += stage.uploaded_bytes;
                    stats.output_readback_bytes += stage.output_readback_bytes;
                }
                stats.diagnostic_readback_bytes += stage.diagnostic_readback_bytes;
            }
        }
        Some(stats)
    }
    /// Statistics from each mutable stage, including CPU worker requests.
    pub fn execution_statistics(
        &self,
    ) -> std::collections::BTreeMap<crate::error::OcrStage, Vec<crate::runtime::ExecutionStatistics>>
    {
        self.stage_statistics(
            self.latest_run
                .as_ref()
                .is_none_or(|run| run.execution == OcrExecution::Resident),
        )
    }
    fn stage_statistics(
        &self,
        resident: bool,
    ) -> std::collections::BTreeMap<crate::error::OcrStage, Vec<crate::runtime::ExecutionStatistics>>
    {
        #[cfg(not(feature = "gpu"))]
        let _ = resident;
        let mut result = std::collections::BTreeMap::new();
        result.insert(
            crate::error::OcrStage::Detector,
            vec![self.extractor.detector().execution_statistics()],
        );
        result.insert(
            crate::error::OcrStage::Recognizer,
            std::iter::once(self.recognizer.execution_statistics())
                .chain(
                    self.workers
                        .iter()
                        .map(|w| w.recognizer.execution_statistics()),
                )
                .collect(),
        );
        let classifiers = self.extractor.classifier_statistics();
        if !classifiers.is_empty() {
            result.insert(crate::error::OcrStage::Classifier, classifiers);
        }
        #[cfg(feature = "gpu")]
        if resident && let Some(pipeline) = self.execution.resident() {
            let diagnostics = result;
            result = pipeline.execution_statistics();
            for (stage, sessions) in diagnostics {
                if let Some(active) = result.get_mut(&stage).and_then(|s| s.first_mut()) {
                    active.diagnostic_readbacks +=
                        sessions.iter().map(|s| s.diagnostic_readbacks).sum::<u64>();
                    active.diagnostic_readback_bytes += sessions
                        .iter()
                        .map(|s| s.diagnostic_readback_bytes)
                        .sum::<u64>();
                }
            }
        }
        result
    }
    #[cfg(all(test, feature = "gpu"))]
    pub(crate) fn workers_len_for_gpu_test(&self) -> usize {
        self.workers.len()
    }
    pub(crate) fn unwound(&self) -> bool {
        self.unwind_state.load(Ordering::Relaxed)
    }
    pub fn cpu_budgets(&self) -> CpuBudgets {
        self.budgets
    }
    pub fn set_cpu_context(&mut self, cpu: Arc<CpuContext>) -> ImageResult<()> {
        self.set_execution_context(cpu, self.plan.cpu_options)
    }
    pub(crate) fn pin_cpu_context(&mut self, cpu: Arc<CpuContext>) -> ImageResult<()> {
        self.set_cpu_context(cpu.clone())?;
        self.pinned_cpu = Some(cpu);
        Ok(())
    }
    pub fn set_execution_context(
        &mut self,
        cpu: Arc<CpuContext>,
        options: CpuOptions,
    ) -> ImageResult<()> {
        if self
            .pinned_cpu
            .as_ref()
            .is_some_and(|pinned| !Arc::ptr_eq(pinned, &cpu))
        {
            return Err(ImageError::new("CPU context is pinned to this OCR engine"));
        }
        let budgets = options.resolve(
            cpu.capacity(),
            self.extractor
                .detector()
                .compiled()
                .as_cpu()
                .is_some_and(|c| c.options().cpu.layout == crate::runtime::LayoutPolicy::Auto),
        )?;
        self.extractor.set_execution_context(cpu.clone(), options)?;
        self.recognizer
            .set_execution_context(cpu.clone(), budgets.recognizer_base_threads())?;
        for worker in &mut self.workers {
            worker
                .recognizer
                .set_execution_context(cpu.clone(), budgets.recognizer_base_threads())?;
        }
        self.plan.cpu = cpu;
        self.plan.cpu_options = options;
        self.budgets = budgets;
        Ok(())
    }
    pub fn options(&self) -> OcrOptions {
        self.plan.options
    }
    /// Maximum crops per exact selected-width recognition batch. Defaults to one.
    pub fn recognition_batch_size(&self) -> usize {
        self.plan.recognition_batch_size
    }
    /// Opt into same-width batching. Larger batches increase activation memory;
    /// each worker remains subject to its recognizer's execution limits.
    pub fn set_recognition_batch_size(&mut self, size: usize) -> ImageResult<()> {
        if size == 0 || size > i32::MAX as usize {
            return Err(ImageError::new(
                "recognition_batch_size must be within 1..=i32::MAX",
            ));
        }
        self.plan.recognition_batch_size = size;
        Ok(())
    }
    pub fn set_detector_resize(
        &mut self,
        policy: crate::detector::DetResizePolicy,
        limit: usize,
    ) -> ImageResult<()> {
        self.extractor.set_resize_options(policy, limit)?;
        self.plan.extractor = self.extractor.plan().clone();
        #[cfg(feature = "gpu")]
        {
            self.execution = ActiveExecution::staged(self.uses_gpu());
        }
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn recognizer_mut(&mut self) -> &mut Recognizer {
        #[cfg(feature = "gpu")]
        {
            self.execution = ActiveExecution::staged(self.uses_gpu());
        }
        // A caller can replace the recognizer through this reference. Worker
        // sessions must then be rebuilt from the new immutable model/options.
        self.workers.clear();
        &mut self.recognizer
    }
    #[cfg(test)]
    pub(crate) fn extractor_mut(&mut self) -> &mut OrientedLineExtractor {
        #[cfg(feature = "gpu")]
        {
            self.execution = ActiveExecution::staged(self.uses_gpu());
        }
        &mut self.extractor
    }
    pub fn activation_bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        let resident = self
            .execution
            .resident()
            .map_or(0, |g| g.activation_bytes());
        #[cfg(not(feature = "gpu"))]
        let resident = 0;
        resident
            + self.extractor.detector().activation_bytes()
            + self.extractor.classifier_activation_bytes()
            + self.recognizer.activation_bytes()
            + self
                .workers
                .iter()
                .map(|w| w.recognizer.activation_bytes())
                .sum::<usize>()
    }
    pub fn retained_activation_bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        let resident = self
            .execution
            .resident()
            .map_or(0, |g| g.activation_bytes());
        #[cfg(not(feature = "gpu"))]
        let resident = 0;
        resident
            + self.extractor.detector().retained_activation_bytes()
            + self.extractor.classifier_retained_activation_bytes()
            + self.recognizer.retained_activation_bytes()
            + self
                .workers
                .iter()
                .map(|w| w.recognizer.retained_activation_bytes())
                .sum::<usize>()
    }
    pub fn diagnostic_retained_activation_bytes(&self) -> usize {
        self.recognizer.diagnostic_retained_activation_bytes()
            + self
                .workers
                .iter()
                .map(|w| w.recognizer.diagnostic_retained_activation_bytes())
                .sum::<usize>()
    }
    /// Retained request-owned GPU buffers. Shared uploaded weights are excluded.
    pub fn retained_device_bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        let resident = self.execution.resident().map_or(0, |g| g.retained_bytes());
        #[cfg(not(feature = "gpu"))]
        let resident = 0;
        resident
            + self.extractor.retained_device_bytes()
            + self.recognizer.retained_device_bytes()
            + self
                .workers
                .iter()
                .map(|w| w.recognizer.retained_device_bytes())
                .sum::<usize>()
    }
    pub fn retained_workspace_bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        let resident = self.execution.resident().map_or(0, |g| g.workspace_bytes());
        #[cfg(not(feature = "gpu"))]
        let resident = 0;
        let buffers = self
            .latest_run
            .as_ref()
            .map_or(0, RunStatistics::retained_bytes)
            + resident
            + self.extractor.retained_workspace_bytes()
            + self.recognizer.retained_workspace_bytes()
            + self.result.lines.capacity() * size_of::<LineRecord>()
            + self.result.text.capacity()
            + self.result.runs.capacity() * size_of::<CtcRun>()
            + self.recognition_order.capacity() * size_of::<usize>()
            + self.recognition_results.capacity() * size_of::<Mutex<RecognitionSlot>>()
            + self
                .recognition_results
                .iter()
                .map(|slot| {
                    slot.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .retained_bytes()
                })
                .sum::<usize>()
            + self
                .workers
                .iter()
                .map(|w| {
                    w.recognizer.retained_workspace_bytes()
                        + w.jobs.capacity() * size_of::<Range<usize>>()
                })
                .sum::<usize>();
        buffers.saturating_add(
            self.workers
                .capacity()
                .saturating_mul(size_of::<RecognitionWorker>()),
        )
    }
    /// Kernel buffers retained by all stages and workers, already included in
    /// retained workspace bytes. Fixed local stack tiles are not heap capacity.
    pub fn retained_kernel_scratch_bytes(&self) -> usize {
        self.extractor.retained_kernel_scratch_bytes()
            + self.recognizer.retained_kernel_scratch_bytes()
            + self
                .workers
                .iter()
                .map(|worker| worker.recognizer.retained_kernel_scratch_bytes())
                .sum::<usize>()
    }
    /// Reuse one DET/optional CLS/REC pipeline in detector reading order.
    /// Errors expose no partial result; a later valid call can reuse the engine.
    pub fn run(&mut self, image: ImageView<'_>, options: RunOptions) -> ImageResult<OcrView<'_>> {
        let capture = options.capture_spans;
        let cpu = self.plan.cpu.clone();
        let _demand = cpu.register_request();
        if self.uses_gpu() {
            return self.run_impl::<false, false, _>(
                image,
                capture,
                |_| Ok(()),
                &mut StageProfile::default(),
            );
        }
        cpu.install(|| {
            self.run_impl::<false, false, _>(
                image,
                capture,
                |_| Ok(()),
                &mut StageProfile::default(),
            )
        })
    }
    /// Profile production OCR without dense diagnostic recognition execution.
    #[cfg(feature = "profiling")]
    pub fn run_profiled(
        &mut self,
        image: ImageView<'_>,
        capture: bool,
    ) -> ImageResult<(OcrView<'_>, StageProfile)> {
        let timer = Timer::<true>::start();
        let mut profile = StageProfile::default();
        let cpu = self.plan.cpu.clone();
        let _demand = cpu.register_request();
        let result = if self.uses_gpu() {
            self.run_impl::<false, true, _>(image, capture, |_| Ok(()), &mut profile)
        } else {
            cpu.install(|| {
                self.run_impl::<false, true, _>(image, capture, |_| Ok(()), &mut profile)
            })
        }?;
        profile.total_ns = timer.elapsed();
        Ok((result, profile))
    }
    /// Observe normalized execution synchronously on the calling thread.
    /// Trace orchestration and callbacks run outside the production worker pool.
    pub fn run_with_trace<F>(
        &mut self,
        image: ImageView<'_>,
        capture: bool,
        observer: F,
    ) -> ImageResult<OcrView<'_>>
    where
        F: for<'a> FnMut(OcrTrace<'a>) -> ImageResult<()>,
    {
        let cpu = self.plan.cpu.clone();
        let _demand = cpu.register_request();
        self.run_impl::<true, false, F>(image, capture, observer, &mut StageProfile::default())
    }
    pub fn latest_run(&self) -> Option<&RunStatistics> {
        self.latest_run.as_ref()
    }
    pub fn counters(&self) -> crate::runtime::ExecutionCounters {
        self.counters
    }
    #[cfg(feature = "gpu")]
    pub fn gpu_pipeline_statistics(&self) -> Option<crate::runtime::gpu::GpuPipelineStatistics> {
        self.latest_run.as_ref().and_then(|r| r.gpu.clone())
    }
    /// Retained capacity used for eviction; shared weights and transient peaks are excluded.
    pub fn memory_snapshot(&self) -> crate::runtime::MemorySnapshot {
        let total = self
            .retained_activation_bytes()
            .saturating_add(self.diagnostic_retained_activation_bytes())
            .saturating_add(self.retained_workspace_bytes());
        let device_bytes = self.retained_device_bytes();
        crate::runtime::MemorySnapshot {
            host_bytes: total.saturating_sub(device_bytes),
            device_bytes,
        }
    }
    fn run_impl<const TRACE: bool, const PROFILE: bool, F>(
        &mut self,
        image: ImageView<'_>,
        capture: bool,
        observer: F,
        profile: &mut StageProfile,
    ) -> ImageResult<OcrView<'_>>
    where
        F: for<'a> FnMut(OcrTrace<'a>) -> ImageResult<()>,
    {
        let before = self.stage_statistics(!TRACE);
        #[cfg(feature = "gpu")]
        {
            self.extractor.reset_run_statistics();
            self.recognizer.reset_run_statistics();
            for worker in &mut self.workers {
                worker.recognizer.reset_run_statistics();
            }
        }
        let mut observer = observer;
        let result = self.execute_inner::<TRACE, PROFILE, _>(
            image,
            capture,
            |event| observer(event).map_err(ImageError::callback),
            profile,
        );
        let mut stages = self.stage_statistics(!TRACE);
        for (stage, sessions) in &mut stages {
            for (i, session) in sessions.iter_mut().enumerate() {
                if let Some(previous) = before.get(stage).and_then(|s| s.get(i)) {
                    session.since(previous);
                }
            }
        }
        #[cfg(feature = "gpu")]
        let mut gpu = self.collect_gpu_statistics(&stages, !TRACE);
        if result.is_ok() && self.result.detected_count == 0 {
            for stage in [
                crate::error::OcrStage::Recognizer,
                crate::error::OcrStage::Classifier,
            ] {
                if let Some(sessions) = stages.get_mut(&stage) {
                    for session in sessions {
                        session.execution_backend = None;
                    }
                }
            }
            #[cfg(feature = "gpu")]
            if let Some(stats) = &mut gpu {
                stats.projection = crate::GpuPlacement::NotRun;
                stats.decoding = crate::GpuPlacement::NotRun;
            }
        }
        let execution = if TRACE || self.execution.kind() != OcrExecution::Resident {
            if self.uses_gpu() {
                OcrExecution::Hybrid
            } else {
                OcrExecution::Cpu
            }
        } else {
            OcrExecution::Resident
        };
        self.counters.runs += 1;
        self.counters.failures += u64::from(result.is_err());
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &gpu {
            self.counters.submissions += gpu.submissions;
            self.counters.uploaded_bytes += gpu.uploaded_bytes;
            self.counters.output_readback_bytes += gpu.output_readback_bytes;
            self.counters.control_readback_bytes += gpu.control_readback_bytes;
            self.counters.diagnostic_readback_bytes += gpu.diagnostic_readback_bytes;
        }
        self.latest_run = Some(RunStatistics {
            execution,
            succeeded: result.is_ok(),
            stages,
            #[cfg(feature = "gpu")]
            gpu,
        });
        result?;
        Ok(self.view())
    }
    fn execute_inner<const TRACE: bool, const PROFILE: bool, F>(
        &mut self,
        image: ImageView<'_>,
        capture: bool,
        observer: F,
        profile: &mut StageProfile,
    ) -> ImageResult<()>
    where
        F: for<'a> FnMut(OcrTrace<'a>) -> ImageResult<()>,
    {
        let inline_cpu = self.plan.cpu.clone();
        let _inline = (!self.uses_gpu()).then(|| inline_cpu.enter_inline());
        let _active_request = inline_cpu.enter_request();
        #[cfg(test)]
        if !TRACE && !self.uses_gpu() {
            self.plan.cpu.assert_production_thread("OCR");
        }
        if self.unwound() {
            return Err(ImageError::new(
                "OCR request unwound; create a fresh request",
            ));
        }
        self.unwind_state.store(true, Ordering::Relaxed);
        let _unwind = RequestUnwindGuard(self.unwind_state.clone());
        #[cfg(feature = "gpu")]
        if !TRACE
            && self.plan.gpu_mode == crate::runtime::gpu::GpuPipelineMode::Full
            && self.extractor.detector().uses_gpu()
            && self.recognizer.uses_gpu()
            && self.extractor.classifier().is_none_or(|c| c.uses_gpu())
        {
            let candidate = (|| -> Result<_, crate::ocr::gpu::Failure> {
                if self.execution.resident().is_none() {
                    self.execution = ActiveExecution::Resident(Box::new(
                        crate::ocr::gpu::Pipeline::new(&self.plan)?,
                    ));
                }
                self.execution.resident_mut().unwrap().run(
                    image,
                    &self.plan,
                    RunOptions {
                        capture_spans: capture,
                    },
                    PROFILE,
                    &mut self.result,
                )
            })();
            match candidate {
                Ok(()) => (),
                Err(failure) => {
                    use crate::ocr::gpu::Stage;
                    let fallback = |model: &crate::runtime::CompiledModel, stage| {
                        if failure.stage == stage || failure.stage == Stage::Shared {
                            model.resident_fallback(failure.error.clone())
                        } else {
                            Ok(model.clone())
                        }
                    };
                    let det = self.plan.extractor.detector();
                    let detector = crate::detector::DetectorPlan::new(
                        fallback(det.compiled(), Stage::Detector)?,
                        det.options(),
                        det.execution_limits(),
                    )?;
                    let classifier = self
                        .plan
                        .extractor
                        .classifier()
                        .map(|stage| fallback(stage.compiled(), Stage::Classifier))
                        .transpose()?;
                    let extractor = crate::lines::OrientedLinePlan::new(
                        detector,
                        classifier,
                        self.plan.extractor.options(),
                    )?;
                    let rec = &self.plan.recognizer;
                    let recognizer = crate::recognizer::RecognizerPlan::new(
                        fallback(rec.compiled(), Stage::Recognizer)?,
                        rec.dictionary().clone(),
                        rec.options(),
                        rec.execution_limits(),
                    )?;
                    let mut replacement = OcrPlan::new(
                        extractor,
                        recognizer,
                        self.plan.options,
                        self.plan.cpu.clone(),
                        self.plan.cpu_options,
                        self.plan.recognition_batch_size,
                    )?
                    .with_gpu_pipeline_mode(self.plan.gpu_mode)?
                    .request()?;
                    replacement.pinned_cpu = self.pinned_cpu.clone();
                    replacement.result = std::mem::take(&mut self.result);
                    replacement.counters = self.counters;
                    *self = replacement;
                    let cpu = self.plan.cpu.clone();
                    let ignore: for<'a> fn(OcrTrace<'a>) -> ImageResult<()> = |_| Ok(());
                    return cpu.install(|| {
                        self.execute_inner::<false, PROFILE, _>(image, capture, ignore, profile)
                    });
                }
            };
            let pipeline = self.execution.resident().unwrap();
            if PROFILE {
                profile.recognized_lines += self.result.lines.len();
                profile.gpu_submission_ns += pipeline.statistics.submission_ns;
                profile.gpu_completion_wait_ns += pipeline.statistics.completion_wait_ns;
                profile.gpu_readback_ns += pipeline.statistics.readback_ns;
                profile.projection_ns += pipeline.statistics.projection_encoding_ns;
            }
            return Ok(());
        }
        self.budgets = self.plan.cpu_options.resolve(
            self.plan.cpu.capacity(),
            self.extractor
                .detector()
                .compiled()
                .as_cpu()
                .is_some_and(|c| c.options().cpu.layout == crate::runtime::LayoutPolicy::Auto),
        )?;
        self.extractor.set_execution_context(
            self.plan.cpu.clone(),
            self.budgets.extraction_options(
                self.plan.cpu.request_capacity(),
                self.plan.cpu_options.line_workers == 0,
            ),
        )?;
        let observer = RefCell::new(observer);
        let oriented = if TRACE {
            self.extractor.extract_with_trace(
                image,
                |event| observer.borrow_mut()(OcrTrace::Detection(event)),
                |event| observer.borrow_mut()(OcrTrace::Line(event)),
            )?
        } else {
            self.extractor.extract_profiled::<PROFILE>(image, profile)?
        };
        let storage_timer = Timer::<PROFILE>::start();
        let count = oriented.len();
        let metadata_bytes = count
            .checked_mul(size_of::<LineRecord>())
            .filter(|&n| n <= self.plan.options.max_total_result_bytes)
            .ok_or_else(|| ImageError::new("OCR exceeds max_total_result_bytes"))?;
        // All crop recognition shapes are checked before the first REC call.
        for line in oriented.iter() {
            if line.image.pixel_count()? > self.recognizer.resolved_options().max_image_pixels {
                return Err(ImageError::new("crop exceeds recognizer max_image_pixels"));
            }
            self.recognizer
                .select_width(line.image.width(), line.image.height())?;
        }
        self.result.lines.clear();
        self.result.text.clear();
        self.result.runs.clear();
        self.result
            .lines
            .try_reserve_exact(count)
            .map_err(allocation)?;
        if PROFILE {
            profile.result_storage_ns = profile
                .result_storage_ns
                .saturating_add(storage_timer.elapsed());
        }
        let recognition_timer = Timer::<PROFILE>::start();
        let scheduled = !TRACE
            && count > 1
            && (self.plan.cpu.capacity() > 1 || self.plan.recognition_batch_size > 1);
        let result_budget = self
            .plan
            .options
            .max_total_result_bytes
            .checked_sub(metadata_bytes)
            .and_then(|bytes| bytes.checked_sub(count.saturating_sub(1)))
            .ok_or_else(|| ImageError::new("OCR exceeds max_total_result_bytes"))?;
        if scheduled && self.recognizer.uses_gpu() {
            recognize_gpu::<PROFILE>(
                &mut self.recognizer,
                &mut self.recognition_results,
                &mut self.recognition_order,
                self.plan.recognition_batch_size,
                &oriented,
                capture,
                result_budget,
                profile,
            )?;
        } else if scheduled {
            recognize_parallel::<PROFILE>(
                &self.plan.cpu,
                &mut self.budgets,
                self.plan.cpu_options.line_workers == 0,
                &self.recognizer,
                &mut self.workers,
                &mut self.recognition_results,
                &mut self.recognition_order,
                self.plan.recognition_batch_size,
                &oriented,
                capture,
                result_budget,
                profile,
            )?;
        }
        if !scheduled {
            self.recognizer.set_execution_context(
                self.plan.cpu.clone(),
                self.budgets
                    .recognizer_threads_for_capacity(count, self.plan.cpu.request_capacity()),
            )?;
        }
        for (index, line) in oriented.iter().enumerate() {
            let cached = if scheduled {
                Some(
                    self.recognition_results[index]
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                )
            } else {
                None
            };
            let recognition = if let Some(result) = &cached {
                result.as_view()
            } else if TRACE {
                self.recognizer
                    .recognize_with_trace(line.image, capture, |event| {
                        observer.borrow_mut()(OcrTrace::Recognition { index, event })
                    })?
            } else if self.recognizer.uses_gpu() {
                self.recognizer
                    .recognize_impl::<PROFILE>(line.image, capture, profile)?
            } else {
                inline_cpu.install(|| {
                    self.recognizer
                        .recognize_impl::<PROFILE>(line.image, capture, profile)
                })?
            };
            let storage_timer = Timer::<PROFILE>::start();
            let text_length = self
                .result
                .text
                .len()
                .checked_add(usize::from(index != 0))
                .and_then(|n| n.checked_add(recognition.text.len()))
                .ok_or_else(|| ImageError::new("OCR text size overflow"))?;
            let span_count = self
                .result
                .runs
                .len()
                .checked_add(recognition.ctc_spans.map_or(0, CtcSpans::len))
                .ok_or_else(|| ImageError::new("OCR alignment count overflow"))?;
            let required = span_count
                .checked_mul(size_of::<CtcRun>())
                .and_then(|n| n.checked_add(text_length))
                .and_then(|n| n.checked_add(metadata_bytes))
                .filter(|&n| {
                    n <= self.plan.options.max_total_result_bytes && n <= isize::MAX as usize
                })
                .ok_or_else(|| ImageError::new("OCR exceeds max_total_result_bytes"))?;
            let _ = required;
            self.result
                .text
                .try_reserve_exact(text_length - self.result.text.len())
                .map_err(allocation)?;
            self.result
                .runs
                .try_reserve_exact(span_count - self.result.runs.len())
                .map_err(allocation)?;
            if index != 0 {
                self.result.text.push('\n');
            }
            let text_start = self.result.text.len();
            self.result.text.push_str(recognition.text);
            let spans = recognition.ctc_spans.map(|spans| {
                let start = self.result.runs.len();
                let mut text_offset = text_start;
                for s in spans.iter() {
                    self.result.runs.push(CtcRun {
                        text_start: text_offset,
                        text_end: text_offset + s.text.len(),
                        score: s.score,
                        start_column: s.start_column,
                        end_column: s.end_column,
                    });
                    text_offset += s.text.len();
                }
                (start, self.result.runs.len())
            });
            self.result.lines.push(LineRecord {
                detection: line.metadata.detection,
                text_start,
                text_end: self.result.text.len(),
                recognition_score: recognition.score,
                classification_score: line.metadata.classification.map_or(0., |c| c.score),
                classification_label: line.metadata.classification.map_or(0, |c| c.label),
                applied_rotation_degrees: line.metadata.applied_rotation_degrees,
                emitted_count: recognition.emitted_count,
                recognition_content_width: recognition.resized_width,
                recognition_tensor_width: recognition.tensor_width,
                recognition_time_steps: recognition.time_steps,
                spans,
            });
            if PROFILE {
                profile.result_storage_ns = profile
                    .result_storage_ns
                    .saturating_add(storage_timer.elapsed());
            }
        }
        if PROFILE {
            profile.recognition_wall_ns = profile
                .recognition_wall_ns
                .saturating_add(recognition_timer.elapsed());
        }
        self.result.detected_count = count;
        self.result.detector_resized_width = oriented.resized_width();
        self.result.detector_resized_height = oriented.resized_height();
        self.result.width_ratio = oriented.width_ratio();
        self.result.height_ratio = oriented.height_ratio();
        let output = self.view();
        if TRACE {
            observer.borrow_mut()(OcrTrace::Result(output))?;
        }
        Ok(())
    }
    fn view(&self) -> OcrView<'_> {
        OcrView {
            text: &self.result.text,
            detected_count: self.result.detected_count,
            detector_resized_width: self.result.detector_resized_width,
            detector_resized_height: self.result.detector_resized_height,
            width_ratio: self.result.width_ratio,
            height_ratio: self.result.height_ratio,
            source: LineSource::Slab(&self.result.lines, &self.result.text, &self.result.runs),
        }
    }
}

/// One GPU owner executes width-homogeneous batches on the calling thread.
/// Results are copied into the existing slots to restore detector reading order.
#[allow(clippy::too_many_arguments)]
fn recognize_gpu<const PROFILE: bool>(
    recognizer: &mut Recognizer,
    results: &mut Vec<Mutex<RecognitionSlot>>,
    order: &mut Vec<usize>,
    batch_size: usize,
    lines: &OrientedLines<'_>,
    capture: bool,
    result_budget: usize,
    profile: &mut StageProfile,
) -> ImageResult<()> {
    let count = lines.len();
    order.clear();
    order.try_reserve_exact(count).map_err(allocation)?;
    order.extend(0..count);
    results
        .try_reserve(count.saturating_sub(results.len()))
        .map_err(allocation)?;
    results.resize_with(count.max(results.len()), || {
        Mutex::new(RecognitionSlot::new())
    });
    sort_recognition_order(order, recognizer, lines.metadata());
    let result_bytes = AtomicUsize::new(0);
    let exceeded = AtomicBool::new(false);
    let mut start = 0;
    while start < count {
        let end = recognition_unit_end(order, start, batch_size, recognizer, lines.metadata());
        if end - start > 1 {
            let batch = recognizer.recognize_batch_impl::<PROFILE, _>(
                end - start,
                |offset| {
                    lines
                        .get(order[start + offset])
                        .map(|line| line.image)
                        .ok_or_else(|| ImageError::new("invalid oriented line metadata"))
                },
                capture,
                profile,
            )?;
            for (offset, recognition) in batch.iter().enumerate() {
                store_recognition::<PROFILE>(
                    order[start + offset],
                    recognition,
                    results,
                    &result_bytes,
                    &exceeded,
                    result_budget,
                    profile,
                )?;
            }
        } else {
            let line = lines
                .get(order[start])
                .ok_or_else(|| ImageError::new("invalid oriented line metadata"))?;
            let recognition = recognizer.recognize_impl::<PROFILE>(line.image, capture, profile)?;
            store_recognition::<PROFILE>(
                order[start],
                recognition,
                results,
                &result_bytes,
                &exceeded,
                result_budget,
                profile,
            )?;
        }
        if exceeded.load(Ordering::Relaxed) {
            return Err(ImageError::new("OCR exceeds max_total_result_bytes"));
        }
        start = end;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn recognize_parallel<const PROFILE: bool>(
    cpu: &Arc<CpuContext>,
    budgets: &mut CpuBudgets,
    automatic_line_workers: bool,
    template: &Recognizer,
    workers: &mut Vec<RecognitionWorker>,
    results: &mut Vec<Mutex<RecognitionSlot>>,
    order: &mut Vec<usize>,
    batch_size: usize,
    lines: &OrientedLines<'_>,
    capture: bool,
    result_budget: usize,
    profile: &mut StageProfile,
) -> ImageResult<()> {
    let count = lines.len();
    order.clear();
    order.try_reserve_exact(count).map_err(allocation)?;
    order.extend(0..count);
    results
        .try_reserve(count.saturating_sub(results.len()))
        .map_err(allocation)?;
    if results.len() < count {
        results.resize_with(count, || Mutex::new(RecognitionSlot::new()));
    }
    sort_recognition_order(order, template, lines.metadata());
    let mut units = 0;
    let mut position = 0;
    while position < count {
        position = recognition_unit_end(order, position, batch_size, template, lines.metadata());
        units += 1;
    }
    let backend = template
        .compiled()
        .as_cpu()
        .ok_or_else(|| ImageError::new("CPU recognition scheduling requires a CPU model"))?
        .backend();
    *budgets = budgets.for_recognition_units(units, automatic_line_workers, backend);
    let active = budgets.line_workers().min(units).max(1);
    workers
        .try_reserve(active.saturating_sub(workers.len()))
        .map_err(allocation)?;
    while workers.len() < active {
        workers.push(RecognitionWorker {
            recognizer: template.fork()?,
            jobs: Vec::new(),
            scheduled_cost: 0,
            error: None,
            profile: StageProfile::default(),
            #[cfg(feature = "profiling")]
            owner_profile: crate::profile::RecognitionOwnerProfile::default(),
        });
    }
    assign_recognition_jobs(
        &mut workers[..active],
        order,
        batch_size,
        template,
        lines.metadata(),
    )?;
    let request_capacity = cpu.request_capacity();
    let threads = budgets.recognizer_threads_for_capacity(units, request_capacity);
    let borrowing_owner = recognition_borrowing_owner(
        *budgets,
        automatic_line_workers,
        backend,
        request_capacity,
        workers[..active].iter().map(|worker| {
            (
                worker.scheduled_cost,
                if worker.jobs.len() == 1 {
                    worker.jobs[0].len()
                } else {
                    0
                },
            )
        }),
    );
    for (index, worker) in workers[..active].iter_mut().enumerate() {
        let kernel_threads = if borrowing_owner == Some(index) {
            2
        } else {
            threads
        };
        worker
            .recognizer
            .set_execution_context(cpu.clone(), kernel_threads)?;
        worker.error = None;
        worker.profile = StageProfile::default();
        #[cfg(feature = "profiling")]
        {
            worker.owner_profile = if PROFILE {
                crate::profile::RecognitionOwnerProfile {
                    kernel_threads,
                    ..Default::default()
                }
            } else {
                Default::default()
            };
        }
    }
    #[cfg(feature = "profiling")]
    if PROFILE {
        for (index, worker) in workers[..active].iter_mut().enumerate() {
            worker.owner_profile = crate::profile::RecognitionOwnerProfile {
                owner_index: index,
                job_count: worker.jobs.len(),
                line_count: worker.jobs.iter().map(Range::len).sum(),
                scheduled_width_cost: worker.scheduled_cost,
                maximum_tensor_width: worker
                    .jobs
                    .iter()
                    .map(|job| {
                        let line = &lines.metadata()[order[job.start]];
                        template
                            .select_width(line.width, line.height)
                            .expect("recognition width was preflighted")
                            .tensor_width
                    })
                    .max()
                    .unwrap_or(0),
                kernel_threads: worker.owner_profile.kernel_threads,
                ..Default::default()
            };
        }
    }
    let result_bytes = AtomicUsize::new(0);
    let budget_exceeded = AtomicBool::new(false);
    #[cfg(feature = "profiling")]
    let dispatch_timer = Timer::<PROFILE>::start();
    cpu.for_each_mut(&mut workers[..active], active, |_, worker| {
        #[cfg(feature = "profiling")]
        let owner_start = if PROFILE {
            worker.owner_profile.rayon_worker_index = crate::profile::rayon_worker_index();
            Some((
                dispatch_timer.elapsed(),
                crate::profile::thread_cpu_ns::<PROFILE>(),
            ))
        } else {
            None
        };
        for job in &worker.jobs {
            let position = job.start;
            let index = order[position];
            let end = job.end;
            let outcome = (|| {
                if end - position > 1 {
                    let batch = worker.recognizer.recognize_batch_impl::<PROFILE, _>(
                        end - position,
                        |offset| {
                            lines
                                .get(order[position + offset])
                                .map(|line| line.image)
                                .ok_or_else(|| ImageError::new("invalid oriented line metadata"))
                        },
                        capture,
                        &mut worker.profile,
                    )?;
                    for (offset, recognition) in batch.iter().enumerate() {
                        store_recognition::<PROFILE>(
                            order[position + offset],
                            recognition,
                            results,
                            &result_bytes,
                            &budget_exceeded,
                            result_budget,
                            &mut worker.profile,
                        )?;
                    }
                    return Ok(());
                }
                let line = lines
                    .get(index)
                    .ok_or_else(|| ImageError::new("invalid oriented line metadata"))?;
                let recognition = worker.recognizer.recognize_impl::<PROFILE>(
                    line.image,
                    capture,
                    &mut worker.profile,
                )?;
                store_recognition::<PROFILE>(
                    index,
                    recognition,
                    results,
                    &result_bytes,
                    &budget_exceeded,
                    result_budget,
                    &mut worker.profile,
                )
            })();
            match outcome {
                Ok(()) => {}
                Err(error) => remember_error(&mut worker.error, index, error),
            }
        }
        #[cfg(feature = "profiling")]
        if let Some((start, cpu_start)) = owner_start {
            let cpu_end = crate::profile::thread_cpu_ns::<PROFILE>();
            let end = dispatch_timer.elapsed();
            worker.owner_profile.dispatch_start_offset_ns = start;
            worker.owner_profile.dispatch_end_offset_ns = end;
            worker.owner_profile.wall_ns = end.saturating_sub(start);
            worker.owner_profile.thread_cpu_ns = cpu_start
                .zip(cpu_end)
                .and_then(|(start, end)| end.checked_sub(start));
        }
    });
    let mut error = None;
    for worker in &mut workers[..active] {
        if PROFILE {
            profile.merge(worker.profile);
        }
        if let Some((index, cause)) = worker.error.take() {
            remember_error(&mut error, index, cause);
        }
    }
    #[cfg(feature = "profiling")]
    if PROFILE {
        debug_assert!(active <= profile.recognition_owners.len());
        profile.recognition_owner_count = active.min(profile.recognition_owners.len());
        profile.recognition_owners.fill(Default::default());
        for (record, worker) in profile
            .recognition_owners
            .iter_mut()
            .zip(&workers[..active])
        {
            *record = worker.owner_profile;
        }
    }
    if let Some((_, error)) = error {
        return Err(error);
    }
    if budget_exceeded.load(Ordering::Relaxed) {
        return Err(ImageError::new("OCR exceeds max_total_result_bytes"));
    }
    Ok(())
}

/// A long individual crop can outlast every other owner even after descending
/// width assignment. Let at most one such owner expose two kernel bands to
/// the same bounded pool, so completed owners can help its remaining work.
/// Geometry alone selects the hint; ownership and warm buffers never migrate.
fn recognition_borrowing_owner(
    budgets: CpuBudgets,
    automatic_line_workers: bool,
    backend: crate::runtime::Backend,
    request_capacity: usize,
    owners: impl ExactSizeIterator<Item = (usize, usize)>,
) -> Option<usize> {
    let active = owners.len();
    if !automatic_line_workers
        || backend != crate::runtime::Backend::NeonFma
        || budgets.recognizer_base_threads() != 1
        || active < 8
        || request_capacity != budgets.capacity()
        || request_capacity < active
    {
        return None;
    }
    let (mut total, mut maximum, mut selected, mut tied) = (0u128, 0usize, None, false);
    for (index, (cost, single_job_lines)) in owners.enumerate() {
        total += cost as u128;
        if cost > maximum {
            maximum = cost;
            selected = (single_job_lines == 1).then_some(index);
            tied = false;
        } else if cost == maximum {
            tied = true;
        }
    }
    // Widen before either sum or multiplication: theoretical width costs near
    // usize::MAX must not wrap into a false dominant-owner decision.
    (!tied && maximum as u128 * active as u128 > total)
        .then_some(selected)
        .flatten()
}

fn assign_recognition_jobs(
    workers: &mut [RecognitionWorker],
    order: &[usize],
    batch_size: usize,
    recognizer: &Recognizer,
    lines: &[crate::lines::LineMetadata],
) -> ImageResult<()> {
    for worker in workers.iter_mut() {
        worker.jobs.clear();
        worker.scheduled_cost = 0;
    }
    let mut start = 0;
    while start < order.len() {
        let end = recognition_unit_end(order, start, batch_size, recognizer, lines);
        let line = &lines[order[start]];
        let width = recognizer
            .select_width(line.width, line.height)?
            .tensor_width;
        // Keep immutable ownership of width-sorted units for this page. A work
        // stealing cursor lets identical pages grow different worker-private
        // arenas, shape caches and resize buffers after warming has completed.
        // Width * batch count balances cost without splitting a batch; equal
        // costs select the first worker, independently of execution timing.
        let worker = workers
            .iter_mut()
            .min_by_key(|worker| worker.scheduled_cost)
            .expect("recognition has at least one worker");
        worker.jobs.try_reserve(1).map_err(allocation)?;
        worker.jobs.push(start..end);
        worker.scheduled_cost = worker
            .scheduled_cost
            .saturating_add(width.saturating_mul(end - start));
        start = end;
    }
    Ok(())
}

fn store_recognition<const PROFILE: bool>(
    index: usize,
    recognition: RecognitionView<'_>,
    results: &[Mutex<RecognitionSlot>],
    result_bytes: &AtomicUsize,
    budget_exceeded: &AtomicBool,
    result_budget: usize,
    profile: &mut StageProfile,
) -> ImageResult<()> {
    let bytes = recognition
        .ctc_spans
        .map_or(0, CtcSpans::len)
        .checked_mul(size_of::<CtcRun>())
        .and_then(|n| n.checked_add(recognition.text.len()));
    if bytes.is_none_or(|bytes| {
        result_bytes
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current
                    .checked_add(bytes)
                    .filter(|&total| total <= result_budget)
            })
            .is_err()
    }) {
        budget_exceeded.store(true, Ordering::Relaxed);
        return Ok(());
    }
    let timer = Timer::<PROFILE>::start();
    let result = results[index]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .assign(recognition);
    if PROFILE {
        profile.result_storage_ns = profile.result_storage_ns.saturating_add(timer.elapsed());
    }
    result
}

fn recognition_unit_end(
    order: &[usize],
    start: usize,
    batch_size: usize,
    recognizer: &Recognizer,
    lines: &[crate::lines::LineMetadata],
) -> usize {
    let width = |position: usize| {
        let line = &lines[order[position]];
        recognizer
            .select_width(line.width, line.height)
            .expect("recognition width was preflighted")
            .tensor_width
    };
    let target = width(start);
    let limit = start.saturating_add(batch_size).min(order.len());
    let mut end = start + 1;
    while end < limit && width(end) == target {
        end += 1;
    }
    end
}

fn sort_recognition_order(
    order: &mut [usize],
    recognizer: &Recognizer,
    lines: &[crate::lines::LineMetadata],
) {
    order.sort_unstable_by(|&a, &b| {
        // run_impl validates every crop's selected width before workers start.
        let width = |index: usize| {
            recognizer
                .select_width(lines[index].width, lines[index].height)
                .expect("recognition width was preflighted")
                .tensor_width
        };
        width(b).cmp(&width(a)).then_with(|| a.cmp(&b))
    });
}

struct RecognitionSlot {
    text: String,
    runs: Vec<CtcRun>,
    capture: bool,
    score: f32,
    emitted_count: usize,
    resized_width: usize,
    tensor_width: usize,
    time_steps: usize,
}
impl RecognitionSlot {
    fn new() -> Self {
        Self {
            text: String::new(),
            runs: Vec::new(),
            capture: false,
            score: 0.,
            emitted_count: 0,
            resized_width: 0,
            tensor_width: 0,
            time_steps: 0,
        }
    }
    fn as_view(&self) -> RecognitionView<'_> {
        RecognitionView {
            text: &self.text,
            score: self.score,
            emitted_count: self.emitted_count,
            resized_width: self.resized_width,
            tensor_width: self.tensor_width,
            time_steps: self.time_steps,
            ctc_spans: self
                .capture
                .then(|| CtcSpans::from_runs(&self.text, &self.runs)),
        }
    }
    fn retained_bytes(&self) -> usize {
        self.text.capacity() + self.runs.capacity() * size_of::<CtcRun>()
    }
    fn assign(&mut self, source: RecognitionView<'_>) -> ImageResult<()> {
        self.text.clear();
        self.text
            .try_reserve_exact(source.text.len())
            .map_err(allocation)?;
        self.text.push_str(source.text);
        self.runs.clear();
        self.capture = source.ctc_spans.is_some();
        if let Some(source_spans) = source.ctc_spans {
            self.runs
                .try_reserve_exact(source_spans.len())
                .map_err(allocation)?;
            let mut text_start = 0;
            for span in source_spans.iter() {
                let text_end = text_start + span.text.len();
                self.runs.push(CtcRun {
                    text_start,
                    text_end,
                    score: span.score,
                    start_column: span.start_column,
                    end_column: span.end_column,
                });
                text_start = text_end;
            }
        }
        self.score = source.score;
        self.emitted_count = source.emitted_count;
        self.resized_width = source.resized_width;
        self.tensor_width = source.tensor_width;
        self.time_steps = source.time_steps;
        Ok(())
    }
}
fn remember_error(destination: &mut Option<(usize, ImageError)>, index: usize, error: ImageError) {
    if destination
        .as_ref()
        .is_none_or(|(earlier, _)| index < *earlier)
    {
        *destination = Some((index, error));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        detector::{Detector, DetectorOptions},
        image::PixelFormat,
        lines::LineOptions,
        model::Model,
        recognizer::{Dictionary, RecognizerOptions},
        runtime::{CpuCompiledModel, ExecutionLimits},
    };
    #[test]
    fn plan_reconfiguration_reports_and_borrowed_storage_are_independent() {
        let mut request = synthetic_ocr(CpuContext::serial());
        let pixels = two_lines();
        let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let expected = request
            .run(
                image,
                RunOptions {
                    capture_spans: true,
                },
            )
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert!(request.set_recognition_batch_size(0).is_err());
        assert!(
            request
                .set_detector_resize(crate::detector::DetResizePolicy::Max, 0)
                .is_err()
        );
        assert_eq!(
            request
                .run(
                    image,
                    RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert_eq!(request.counters().runs, 2);
        assert_eq!(request.latest_run().unwrap().execution, OcrExecution::Cpu);
        let mut other = request.fork().unwrap();
        assert_eq!(other.activation_bytes(), 0);
        let view = request.run(image, RunOptions::default()).unwrap();
        other
            .run(
                image,
                RunOptions {
                    capture_spans: true,
                },
            )
            .unwrap();
        assert_eq!(view.text, expected.text);
        assert!(view.iter().all(|line| line.ctc_spans.is_none()));
        let memory = request.memory_snapshot();
        assert_eq!(memory.device_bytes, 0);
        assert!(memory.host_bytes >= request.retained_activation_bytes());
        let mut calls = 0;
        let error = request
            .run_with_trace(image, false, |_| {
                calls += 1;
                Err(ImageError::with_kind(
                    crate::error::ErrorKind::DeviceLost,
                    "callback failure",
                ))
            })
            .err()
            .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(error.origin(), crate::error::ErrorOrigin::Callback);
        assert!(!request.latest_run().unwrap().succeeded);
        assert_eq!(request.counters().failures, 1);
    }
    fn varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (value & 127) as u8;
            value >>= 7;
            out.push(byte | if value == 0 { 0 } else { 128 });
            if value == 0 {
                return out;
            }
        }
    }
    fn number(field: u64, value: u64) -> Vec<u8> {
        [varint(field << 3), varint(value)].concat()
    }
    fn bytes(field: u64, value: &[u8]) -> Vec<u8> {
        [
            varint(field << 3 | 2),
            varint(value.len() as u64),
            value.to_vec(),
        ]
        .concat()
    }
    fn ints(name: &str, values: &[u64]) -> Vec<u8> {
        bytes(
            5,
            &[
                bytes(1, name.as_bytes()),
                values.iter().flat_map(|&v| number(8, v)).collect(),
                number(20, 7),
            ]
            .concat(),
        )
    }
    fn integer(name: &str, value: u64) -> Vec<u8> {
        bytes(
            5,
            &[bytes(1, name.as_bytes()), number(3, value), number(20, 2)].concat(),
        )
    }
    fn node(op: &str, input: &[&str], output: &str, attrs: Vec<u8>) -> Vec<u8> {
        bytes(
            1,
            &[
                input.iter().flat_map(|v| bytes(1, v.as_bytes())).collect(),
                bytes(2, output.as_bytes()),
                bytes(4, op.as_bytes()),
                attrs,
            ]
            .concat(),
        )
    }
    fn info(name: &str, shape: &[Option<u64>]) -> Vec<u8> {
        bytes(
            if name == "x" { 11 } else { 12 },
            &[
                bytes(1, name.as_bytes()),
                bytes(
                    2,
                    &bytes(
                        1,
                        &[
                            number(1, 1),
                            bytes(
                                2,
                                &shape
                                    .iter()
                                    .enumerate()
                                    .flat_map(|(i, v)| {
                                        bytes(
                                            1,
                                            &v.map_or_else(
                                                || bytes(2, format!("d{i}").as_bytes()),
                                                |n| number(1, n),
                                            ),
                                        )
                                    })
                                    .collect::<Vec<_>>(),
                            ),
                        ]
                        .concat(),
                    ),
                ),
            ]
            .concat(),
        )
    }
    fn source(graph: Vec<u8>) -> Vec<u8> {
        [number(1, 8), bytes(8, &number(2, 7)), bytes(7, &graph)].concat()
    }
    fn compile(graph: Vec<u8>) -> Arc<CpuCompiledModel> {
        Arc::new(
            CpuCompiledModel::new(Arc::new(Model::load_bytes(&source(graph)).unwrap())).unwrap(),
        )
    }
    pub(crate) fn synthetic_sources() -> (Vec<u8>, Vec<u8>) {
        let detector = source(
            [
                node(
                    "ReduceMean",
                    &["x"],
                    "mean",
                    [ints("axes", &[1]), integer("keepdims", 1)].concat(),
                ),
                node("Sigmoid", &["mean"], "y", vec![]),
                info("x", &[Some(1), Some(3), None, None]),
                info("y", &[Some(1), Some(1), None, None]),
            ]
            .concat(),
        );
        let weights: Vec<f32> = (0..9).map(|i| if i % 3 == 1 { 1. } else { 0. }).collect();
        let tensor = bytes(
            5,
            &[
                number(1, 3),
                number(1, 3),
                number(2, 1),
                bytes(8, b"weights"),
                bytes(
                    9,
                    &weights
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            ]
            .concat(),
        );
        let rec = source(
            [
                tensor,
                node(
                    "ReduceMean",
                    &["x"],
                    "mean",
                    [ints("axes", &[2]), integer("keepdims", 0)].concat(),
                ),
                node("Transpose", &["mean"], "feature", ints("perm", &[0, 2, 1])),
                node("MatMul", &["feature", "weights"], "logits", vec![]),
                node("Softmax", &["logits"], "y", integer("axis", 2)),
                info("x", &[None, Some(3), Some(48), None]),
                info("y", &[None, None, Some(3)]),
            ]
            .concat(),
        );
        (detector, rec)
    }
    pub(crate) fn synthetic_ocr(cpu: Arc<CpuContext>) -> OcrRequest {
        let (detector, rec) = synthetic_sources();
        let load = |bytes: &[u8]| {
            Arc::new(CpuCompiledModel::new(Arc::new(Model::load_bytes(bytes).unwrap())).unwrap())
        };
        let detector = Detector::new(
            load(&detector),
            DetectorOptions {
                limit_side_length: 96,
                ..Default::default()
            },
            ExecutionLimits::default(),
        )
        .unwrap();
        let extractor = OrientedLineExtractor::new(detector, None, LineOptions::default()).unwrap();
        let recognizer = Recognizer::new(
            load(&rec),
            Arc::new(Dictionary::from_utf8(b"A\n").unwrap()),
            RecognizerOptions::default(),
            ExecutionLimits::default(),
        )
        .unwrap();
        OcrRequest::new_with_execution(
            extractor,
            recognizer,
            OcrOptions::default(),
            cpu,
            CpuOptions::default(),
        )
        .unwrap()
    }
    pub(crate) fn two_lines() -> Vec<u8> {
        let mut pixels = vec![0; 96 * 96 * 3];
        for (left, right, top, bottom) in [(12, 52, 10, 23), (10, 84, 58, 72)] {
            for y in top..bottom {
                for x in left..right {
                    pixels[(y * 96 + x) * 3..(y * 96 + x) * 3 + 3].fill(255);
                }
            }
        }
        pixels
    }
    pub(crate) fn synthetic_classifier() -> Arc<CpuCompiledModel> {
        let weights = [-0.25f32, 0.5, -0.5, 0.25, -0.75, 0.75];
        let tensor = bytes(
            5,
            &[
                number(1, 3),
                number(1, 2),
                number(2, 1),
                bytes(8, b"weights"),
                bytes(
                    9,
                    &weights
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            ]
            .concat(),
        );
        compile(
            [
                tensor,
                node(
                    "ReduceMean",
                    &["x"],
                    "feature",
                    [ints("axes", &[2, 3]), integer("keepdims", 0)].concat(),
                ),
                node("MatMul", &["feature", "weights"], "logits", vec![]),
                node("Softmax", &["logits"], "y", integer("axis", 1)),
                info("x", &[Some(1), Some(3), Some(80), Some(160)]),
                info("y", &[Some(1), Some(2)]),
            ]
            .concat(),
        )
    }
    #[test]
    fn standalone_production_stages_enter_their_context_and_cls_trace_stays_on_caller() {
        #[cfg(feature = "parallel")]
        let capacities = [1, 2, 4, 8];
        #[cfg(not(feature = "parallel"))]
        let capacities = [1];
        let pixels = two_lines();
        let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        for capacity in capacities {
            let cpu = Arc::new(CpuContext::new(capacity).unwrap());
            let mut pipeline = synthetic_ocr(cpu.clone());
            assert!(!cpu.is_worker_thread());
            let detected = pipeline.extractor.detector_mut().detect(image).unwrap();
            assert_eq!(detected.boxes.len(), 2);
            let crops = pipeline.extractor.extract(image).unwrap();
            let first = crops.get(0).unwrap();
            assert!(pipeline.recognizer.recognize(first.image, true).is_ok());
            #[cfg(feature = "profiling")]
            {
                let (_, profile) = pipeline
                    .recognizer
                    .recognize_profiled(first.image, true)
                    .unwrap();
                assert!(profile.total_ns > 0);
                let (result, profile) = pipeline.run_profiled(image, true).unwrap();
                assert_eq!(result.detected_count, 2);
                assert!(profile.total_ns > 0);
            }
            let mut classifier = crate::orientation::Classifier::new(
                synthetic_classifier(),
                ExecutionLimits::default(),
            )
            .unwrap();
            classifier.set_execution_context(cpu.clone(), 1).unwrap();
            assert!(classifier.classify(image).unwrap().score.is_finite());
            let caller = std::thread::current().id();
            let calls = std::rc::Rc::new(std::cell::Cell::new(0));
            classifier
                .classify_with_trace(image, |_| {
                    assert_eq!(std::thread::current().id(), caller);
                    calls.set(calls.get() + 1);
                    Ok(())
                })
                .unwrap();
            assert!(calls.get() > 0);
            assert!(!cpu.is_worker_thread());
        }
    }
    #[cfg(feature = "parallel")]
    fn float_tensor(name: &str, dimensions: &[u64], values: &[f32]) -> Vec<u8> {
        bytes(
            5,
            &[
                dimensions.iter().flat_map(|&d| number(1, d)).collect(),
                number(2, 1),
                bytes(8, name.as_bytes()),
                bytes(
                    9,
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            ]
            .concat(),
        )
    }
    #[test]
    fn recognition_slot_reuses_compact_unicode_runs_through_shrink_and_capture_toggles() {
        fn assign(slot: &mut RecognitionSlot, tokens: &[&str], capture: bool) {
            let text = tokens.concat();
            let spans = tokens
                .iter()
                .enumerate()
                .map(|(index, text)| OwnedCtcSpan {
                    text: (*text).to_owned(),
                    score: 0.5 + index as f32 / 16.,
                    start_column: index * 3,
                    end_column: index * 3 + 2,
                })
                .collect::<Vec<_>>();
            let source = RecognitionView {
                text: &text,
                score: 0.25 + tokens.len() as f32 / 32.,
                emitted_count: tokens.len(),
                resized_width: 17 + tokens.len(),
                tensor_width: 32,
                time_steps: 64,
                ctc_spans: capture.then(|| CtcSpans::from_owned(&spans)),
            };
            let expected = serde_json::to_value(source).unwrap();
            slot.assign(source).unwrap();
            assert_eq!(serde_json::to_value(slot.as_view()).unwrap(), expected);
            assert_eq!(slot.as_view().ctc_spans.is_some(), capture);
            assert_eq!(slot.runs.len(), if capture { tokens.len() } else { 0 });
            if capture {
                let mut offset = 0;
                for (run, token) in slot.runs.iter().zip(tokens) {
                    assert_eq!(run.text_start, offset);
                    offset += token.len();
                    assert_eq!(run.text_end, offset);
                    assert_eq!(&slot.text[run.text_start..run.text_end], *token);
                }
            }
        }
        let mut slot = RecognitionSlot::new();
        assign(&mut slot, &["A"], true);
        let large = ["A", "汉", "🙂", "é", "字"];
        assign(&mut slot, &large, true);
        let storage = |slot: &RecognitionSlot| {
            (
                slot.text.as_ptr() as usize,
                slot.text.capacity(),
                slot.runs.as_ptr() as usize,
                slot.runs.capacity(),
                slot.retained_bytes(),
            )
        };
        let high_water = storage(&slot);
        for (tokens, capture) in [
            (&large[..3], true),
            (&large[1..2], true),
            (&large[..], false),
            (&large[..0], false),
            (&large[..0], true),
            (&large[..], true),
        ] {
            assign(&mut slot, tokens, capture);
            assert_eq!(storage(&slot), high_water);
        }
    }
    #[cfg(feature = "parallel")]
    fn classified_ocr(cpu: Arc<CpuContext>, weights: [f32; 6], bias: [f32; 2]) -> OcrRequest {
        let classifier = compile(
            [
                float_tensor("weights", &[3, 2], &weights),
                float_tensor("bias", &[2], &bias),
                node(
                    "ReduceMean",
                    &["x"],
                    "feature",
                    [ints("axes", &[2, 3]), integer("keepdims", 0)].concat(),
                ),
                node("MatMul", &["feature", "weights"], "logits0", vec![]),
                node("Add", &["logits0", "bias"], "logits", vec![]),
                node("Softmax", &["logits"], "y", integer("axis", 1)),
                info("x", &[Some(1), Some(3), Some(80), Some(160)]),
                info("y", &[Some(1), Some(2)]),
            ]
            .concat(),
        );
        let mut ocr = synthetic_ocr(cpu.clone());
        let extractor = OrientedLineExtractor::new(
            ocr.extractor.detector().fork().unwrap(),
            Some(classifier.into()),
            crate::lines::LineOptions {
                classifier_threshold: 0.5,
                ..Default::default()
            },
        )
        .unwrap();
        ocr.reconfigure(
            ocr.plan()
                .clone()
                .with_extractor(extractor.plan().clone())
                .unwrap(),
        )
        .unwrap();
        ocr.set_execution_context(cpu, CpuOptions::default())
            .unwrap();
        ocr
    }
    #[cfg(feature = "parallel")]
    fn mixed_lines(color: [u8; 3]) -> Vec<u8> {
        let mut pixels = vec![0; 96 * 96 * 3];
        for (left, right, top, bottom) in [
            (8, 55, 8, 16),
            (10, 85, 31, 40),
            (6, 35, 54, 63),
            (18, 75, 77, 86),
        ] {
            for y in top..bottom {
                for x in left..right {
                    pixels[(y * 96 + x) * 3..(y * 96 + x) * 3 + 3].copy_from_slice(&color);
                }
            }
        }
        pixels
    }
    #[cfg(feature = "parallel")]
    #[test]
    fn staged_line_cache_reuses_width_owners_and_buffers_after_budget_errors() {
        let cpu = Arc::new(CpuContext::new(8).unwrap());
        let mut parallel = classified_ocr(cpu, [0.; 6], [0., 5.]);
        let mut serial = parallel.fork().unwrap();
        serial.set_cpu_context(CpuContext::serial()).unwrap();
        let first = mixed_lines([255; 3]);
        let second = mixed_lines([230; 3]);
        let images = [&first, &second]
            .map(|pixels| ImageView::new(pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap());
        let expected = images.map(|image| {
            serial
                .run(
                    image,
                    crate::RunOptions {
                        capture_spans: true,
                    },
                )
                .unwrap()
                .try_to_owned()
                .unwrap()
        });
        for (image, expected) in images.into_iter().zip(&expected) {
            assert_eq!(
                &parallel
                    .run(
                        image,
                        crate::RunOptions {
                            capture_spans: true
                        }
                    )
                    .unwrap()
                    .try_to_owned()
                    .unwrap(),
                expected
            );
        }
        let worker_storage = |request: &OcrRequest| {
            request
                .workers
                .iter()
                .map(|worker| {
                    (
                        worker.jobs.clone(),
                        worker.jobs.as_ptr(),
                        worker.jobs.capacity(),
                        worker.recognizer.retained_activation_bytes(),
                        worker.recognizer.retained_workspace_bytes(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let owners = worker_storage(&parallel);
        let buffers = parallel
            .recognition_results
            .iter()
            .map(|slot| {
                let slot = slot.lock().unwrap();
                (slot.text.as_ptr() as usize, slot.retained_bytes())
            })
            .collect::<Vec<_>>();
        let activations = parallel.retained_activation_bytes();
        let workspace = parallel.retained_workspace_bytes();
        let classifier_bytes = parallel.extractor.classifier_retained_activation_bytes();
        for (image, expected) in images.into_iter().zip(&expected) {
            assert_eq!(
                &parallel
                    .run(
                        image,
                        crate::RunOptions {
                            capture_spans: true
                        }
                    )
                    .unwrap()
                    .try_to_owned()
                    .unwrap(),
                expected
            );
            assert_eq!(worker_storage(&parallel), owners);
            assert_eq!(parallel.retained_activation_bytes(), activations);
            assert_eq!(parallel.retained_workspace_bytes(), workspace);
            assert_eq!(
                parallel.extractor.classifier_retained_activation_bytes(),
                classifier_bytes
            );
            for (slot, expected_buffer) in parallel.recognition_results.iter().zip(&buffers) {
                let slot = slot.lock().unwrap();
                assert_eq!(
                    (slot.text.as_ptr() as usize, slot.retained_bytes()),
                    *expected_buffer
                );
            }
        }
        let limit = parallel.plan.options.max_total_result_bytes;
        for failed_limit in [1, 4 * size_of::<LineRecord>() + 3] {
            parallel.plan.options.max_total_result_bytes = failed_limit;
            assert!(
                parallel
                    .run(
                        images[0],
                        crate::RunOptions {
                            capture_spans: true
                        }
                    )
                    .unwrap_err()
                    .to_string()
                    .contains("max_total_result_bytes")
            );
            parallel.plan.options.max_total_result_bytes = limit;
            assert_eq!(
                parallel
                    .run(
                        images[0],
                        crate::RunOptions {
                            capture_spans: true
                        }
                    )
                    .unwrap()
                    .try_to_owned()
                    .unwrap(),
                expected[0]
            );
            assert_eq!(worker_storage(&parallel), owners);
        }
        let inactive_slots = parallel.recognition_results[2..]
            .iter()
            .map(|slot| {
                let slot = slot.lock().unwrap();
                (
                    slot.text.as_ptr() as usize,
                    slot.text.capacity(),
                    slot.runs.as_ptr() as usize,
                    slot.runs.capacity(),
                )
            })
            .collect::<Vec<_>>();
        let fewer = two_lines();
        let fewer = ImageView::new(&fewer, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let fewer_expected = serial
            .run(
                fewer,
                crate::RunOptions {
                    capture_spans: true,
                },
            )
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert_eq!(fewer_expected.lines.len(), 2);
        assert_eq!(
            parallel
                .run(
                    fewer,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            fewer_expected
        );
        assert_eq!(parallel.recognition_results.len(), 4);
        for (slot, expected_storage) in parallel.recognition_results[2..].iter().zip(inactive_slots)
        {
            let slot = slot.lock().unwrap();
            assert_eq!(
                (
                    slot.text.as_ptr() as usize,
                    slot.text.capacity(),
                    slot.runs.as_ptr() as usize,
                    slot.runs.capacity()
                ),
                expected_storage
            );
        }
        assert_eq!(
            parallel
                .run(
                    images[0],
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected[0]
        );
    }
    #[test]
    fn recognition_order_uses_padded_tensor_width_and_original_index_ties() {
        let ocr = synthetic_ocr(CpuContext::serial());
        let lines = [80, 95, 64, 96, 32, 65].map(|width| crate::lines::LineMetadata {
            detection: DetectionBox {
                points: [[0.; 2]; 4],
                score: 1.,
            },
            offset: 0,
            bytes: 0,
            width,
            height: 48,
            unrotated_width: width,
            unrotated_height: 48,
            vertical_rotation: false,
            classification: None,
            applied_rotation_degrees: 0,
        });
        let mut order = [5, 4, 3, 2, 1, 0];
        sort_recognition_order(&mut order, &ocr.recognizer, &lines);
        assert_eq!(order, [0, 1, 3, 5, 2, 4]);
        let mut units = Vec::new();
        let mut start = 0;
        while start < order.len() {
            let end = recognition_unit_end(&order, start, 2, &ocr.recognizer, &lines);
            units.push(order[start..end].to_vec());
            start = end;
        }
        assert_eq!(units, [vec![0, 1], vec![3, 5], vec![2], vec![4]]);
        let mut workers: Vec<_> = (0..3)
            .map(|_| RecognitionWorker {
                recognizer: ocr.recognizer.fork().unwrap(),
                jobs: Vec::new(),
                scheduled_cost: 0,
                error: None,
                profile: StageProfile::default(),
                #[cfg(feature = "profiling")]
                owner_profile: crate::profile::RecognitionOwnerProfile::default(),
            })
            .collect();
        assign_recognition_jobs(&mut workers, &order, 2, &ocr.recognizer, &lines).unwrap();
        assert_eq!(workers[0].jobs.len(), 1);
        assert_eq!(workers[0].jobs.first(), Some(&(0..2)));
        assert_eq!(workers[1].jobs.len(), 1);
        assert_eq!(workers[1].jobs.first(), Some(&(2..4)));
        assert_eq!(workers[2].jobs, [4..5, 5..6]);
        assert_eq!(
            workers.iter().map(|w| w.scheduled_cost).collect::<Vec<_>>(),
            [192, 192, 96]
        );
        let storage: Vec<_> = workers
            .iter()
            .map(|w| (w.jobs.as_ptr(), w.jobs.capacity()))
            .collect();
        assign_recognition_jobs(&mut workers, &order, 2, &ocr.recognizer, &lines).unwrap();
        assert_eq!(
            workers
                .iter()
                .map(|w| (w.jobs.as_ptr(), w.jobs.capacity()))
                .collect::<Vec<_>>(),
            storage
        );
        assign_recognition_jobs(&mut workers[..2], &order, 1, &ocr.recognizer, &lines).unwrap();
        assert_eq!(workers[0].jobs, [0..1, 2..3, 4..5]);
        assert_eq!(workers[1].jobs, [1..2, 3..4, 5..6]);
        assert_eq!(workers[0].scheduled_cost, 256);
        assert_eq!(workers[1].scheduled_cost, 224);
    }

    #[test]
    fn batch_configuration_is_validated_and_preserved_by_forks_and_factories() {
        let mut ocr = synthetic_ocr(CpuContext::serial());
        assert_eq!(ocr.recognition_batch_size(), 1);
        assert!(ocr.set_recognition_batch_size(0).is_err());
        assert!(ocr.set_recognition_batch_size(usize::MAX).is_err());
        assert_eq!(ocr.recognition_batch_size(), 1);
        ocr.set_recognition_batch_size(7).unwrap();
        assert_eq!(ocr.fork().unwrap().recognition_batch_size(), 7);
        assert_eq!(ocr.plan().request().unwrap().recognition_batch_size(), 7);
    }

    #[cfg(feature = "profiling")]
    #[test]
    fn unprofiled_recognition_clears_owner_diagnostics_on_reuse() {
        let mut ocr = synthetic_ocr(CpuContext::serial());
        ocr.set_recognition_batch_size(2).unwrap();
        let pixels = two_lines();
        let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let (_, profile) = ocr.run_profiled(image, true).unwrap();
        assert_eq!(profile.recognition_owner_count, 1);
        ocr.run(
            image,
            crate::RunOptions {
                capture_spans: true,
            },
        )
        .unwrap();
        for worker in &ocr.workers {
            assert_eq!(
                worker.owner_profile,
                crate::profile::RecognitionOwnerProfile::default()
            );
            assert_eq!(worker.profile.recognition_owner_count, 0);
            assert_eq!(worker.profile.rec_backbone_ns, 0);
        }
    }

    #[test]
    fn same_width_ocr_batches_preserve_order_trace_results_and_recovery() {
        let mut pixels = vec![0; 96 * 96 * 3];
        for top in [10, 58] {
            for y in top..top + 13 {
                for x in 12..52 {
                    pixels[(y * 96 + x) * 3..(y * 96 + x) * 3 + 3].fill(255);
                }
            }
        }
        let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let mut serial = synthetic_ocr(CpuContext::serial());
        let expected = serde_json::to_value(
            serial
                .run(
                    image,
                    crate::RunOptions {
                        capture_spans: true,
                    },
                )
                .unwrap(),
        )
        .unwrap();
        assert_eq!(serial.result.detected_count, 2);
        assert_eq!(
            serial.result.lines[0].recognition_tensor_width,
            serial.result.lines[1].recognition_tensor_width
        );
        #[cfg(feature = "parallel")]
        let capacities = [1, 3];
        #[cfg(not(feature = "parallel"))]
        let capacities = [1];
        for capacity in capacities {
            let mut ocr = synthetic_ocr(Arc::new(CpuContext::new(capacity).unwrap()));
            ocr.set_recognition_batch_size(2).unwrap();
            assert_eq!(
                serde_json::to_value(
                    ocr.run(
                        image,
                        crate::RunOptions {
                            capture_spans: true
                        }
                    )
                    .unwrap()
                )
                .unwrap(),
                expected
            );
            assert_eq!(ocr.workers.len(), 1);
            let retained = ocr.retained_workspace_bytes();
            ocr.run(
                image,
                crate::RunOptions {
                    capture_spans: false,
                },
            )
            .unwrap();
            assert_eq!(
                serde_json::to_value(
                    ocr.run(
                        image,
                        crate::RunOptions {
                            capture_spans: true
                        }
                    )
                    .unwrap()
                )
                .unwrap(),
                expected
            );
            assert_eq!(ocr.retained_workspace_bytes(), retained);
            let mut traced_inputs = 0;
            let traced = ocr
                .run_with_trace(image, true, |event| {
                    if let OcrTrace::Recognition {
                        event: RecognitionTrace::Input { shape, .. },
                        ..
                    } = event
                    {
                        assert_eq!(shape.dimensions()[0], 1);
                        traced_inputs += 1;
                    }
                    Ok(())
                })
                .unwrap();
            assert_eq!(serde_json::to_value(traced).unwrap(), expected);
            assert_eq!(traced_inputs, 2);
            assert!(
                ocr.run_with_trace(image, true, |_| Err(ImageError::new("observer stopped")))
                    .is_err()
            );
            assert_eq!(
                serde_json::to_value(
                    ocr.run(
                        image,
                        crate::RunOptions {
                            capture_spans: true
                        }
                    )
                    .unwrap()
                )
                .unwrap(),
                expected
            );
            let budget = ocr.plan.options.max_total_result_bytes;
            ocr.plan.options.max_total_result_bytes = 2 * size_of::<LineRecord>() + 1;
            assert!(
                ocr.run(
                    image,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap_err()
                .to_string()
                .contains("max_total_result_bytes")
            );
            ocr.plan.options.max_total_result_bytes = budget;
            assert_eq!(
                serde_json::to_value(
                    ocr.run(
                        image,
                        crate::RunOptions {
                            capture_spans: true
                        }
                    )
                    .unwrap()
                )
                .unwrap(),
                expected
            );
            #[cfg(feature = "profiling")]
            {
                let (_, profile) = ocr.run_profiled(image, true).unwrap();
                assert_eq!(profile.recognized_lines, 2);
            }
        }
    }
    #[test]
    fn retained_workspace_counts_reserved_recognition_worker_storage() {
        let mut request = synthetic_ocr(CpuContext::serial());
        let before = request.retained_workspace_bytes();
        request.workers.try_reserve_exact(3).unwrap();
        let added = request.workers.capacity() * size_of::<RecognitionWorker>();
        assert!(added > 0);
        assert_eq!(request.retained_workspace_bytes(), before + added);
    }
    #[test]
    fn recognition_borrowing_requires_automatic_neon_and_a_unique_dominant_crop() {
        use crate::runtime::Backend;
        let budgets = CpuOptions::default()
            .resolve(8, true)
            .unwrap()
            .for_recognition_units(16, true, Backend::NeonFma);
        let owners = [
            (512, 1),
            (128, 2),
            (128, 2),
            (128, 2),
            (128, 2),
            (128, 2),
            (128, 2),
            (128, 2),
        ];
        let select = |budgets, automatic, backend, share, owners: &[(usize, usize)]| {
            recognition_borrowing_owner(budgets, automatic, backend, share, owners.iter().copied())
        };
        assert_eq!(select(budgets, true, Backend::NeonFma, 8, &owners), Some(0));
        assert_eq!(select(budgets, false, Backend::NeonFma, 8, &owners), None);
        for backend in [
            Backend::Scalar,
            Backend::Portable4,
            Backend::Avx,
            Backend::AvxFma,
            Backend::Avx2Fma,
            Backend::Avx512,
        ] {
            assert_eq!(select(budgets, true, backend, 8, &owners), None);
        }
        for share in [1, 4, 7, 9] {
            assert_eq!(
                select(budgets, true, Backend::NeonFma, share, &owners),
                None
            );
        }
        assert_eq!(
            select(budgets, true, Backend::NeonFma, 8, &owners[..7]),
            None
        );
        let nested = CpuOptions::default()
            .resolve(16, true)
            .unwrap()
            .for_recognition_units(16, true, Backend::NeonFma);
        assert_eq!(select(nested, true, Backend::NeonFma, 16, &owners), None);
        let mut divisible = owners;
        divisible[0].1 = 2;
        assert_eq!(select(budgets, true, Backend::NeonFma, 8, &divisible), None);
        assert_eq!(
            select(budgets, true, Backend::NeonFma, 8, &[(128, 1); 8]),
            None
        );
        assert_eq!(
            select(budgets, true, Backend::NeonFma, 8, &[(0, 1); 8]),
            None
        );
        let mut moved = owners;
        moved.swap(0, 4);
        assert_eq!(select(budgets, true, Backend::NeonFma, 8, &moved), Some(4));
        // Tied maxima provide no unique owner to borrow an additional band.
        let mut tied = owners;
        tied[1] = (512, 1);
        assert_eq!(select(budgets, true, Backend::NeonFma, 8, &tied), None);
        let mut large = [(usize::MAX - 1, 2); 8];
        large[0] = (usize::MAX, 1);
        assert_eq!(select(budgets, true, Backend::NeonFma, 8, &large), Some(0));
    }

    #[cfg(all(feature = "parallel", target_arch = "aarch64"))]
    #[test]
    fn automatic_recognition_borrowing_preserves_ownership_outputs_and_warm_storage() {
        fn lines(widths: &[usize]) -> crate::lines::OwnedOrientedLines {
            let mut pixels = Vec::new();
            let metadata = widths
                .iter()
                .map(|&width| {
                    let offset = pixels.len();
                    let bytes = width * 48 * 3;
                    pixels.resize(offset + bytes, 255);
                    crate::lines::LineMetadata {
                        detection: DetectionBox {
                            points: [[0., 0.], [width as f32, 0.], [width as f32, 48.], [0., 48.]],
                            score: 1.,
                        },
                        offset,
                        bytes,
                        width,
                        height: 48,
                        unrotated_width: width,
                        unrotated_height: 48,
                        vertical_rotation: false,
                        classification: None,
                        applied_rotation_degrees: 0,
                    }
                })
                .collect();
            crate::lines::OwnedOrientedLines::new(
                *widths.iter().max().unwrap(),
                48,
                1.,
                1.,
                metadata,
                pixels,
            )
            .unwrap()
        }
        fn run(
            request: &mut OcrRequest,
            lines: &crate::lines::OwnedOrientedLines,
            batch: usize,
        ) -> StageProfile {
            let cpu = request.cpu.clone();
            request.budgets = request.cpu_options.resolve(cpu.capacity(), true).unwrap();
            let mut profile = StageProfile::default();
            recognize_parallel::<{ cfg!(feature = "profiling") }>(
                &cpu,
                &mut request.budgets,
                request.cpu_options.line_workers == 0,
                &request.recognizer,
                &mut request.workers,
                &mut request.recognition_results,
                &mut request.recognition_order,
                batch,
                &lines.as_view(),
                true,
                usize::MAX,
                &mut profile,
            )
            .unwrap();
            let mut reference = request.recognizer.fork().unwrap();
            reference
                .set_execution_context(CpuContext::serial(), 1)
                .unwrap();
            for (slot, line) in request
                .recognition_results
                .iter()
                .zip(lines.as_view().iter())
            {
                assert_eq!(
                    serde_json::to_value(slot.lock().unwrap().as_view()).unwrap(),
                    serde_json::to_value(reference.recognize(line.image, true).unwrap()).unwrap(),
                );
            }
            profile
        }
        let threads = |request: &OcrRequest| {
            request
                .workers
                .iter()
                .map(|worker| worker.recognizer.projection_statistics().unwrap().threads)
                .collect::<Vec<_>>()
        };
        let storage = |request: &OcrRequest| {
            request
                .workers
                .iter()
                .map(|worker| {
                    (
                        worker.jobs.clone(),
                        worker.jobs.as_ptr() as usize,
                        worker.jobs.capacity(),
                        worker.recognizer.retained_activation_bytes(),
                        worker.recognizer.retained_workspace_bytes(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let cpu = Arc::new(CpuContext::new(8).unwrap());
        // Direct recognition dispatch normally runs inside OcrRequest's registered
        // request. Keep that same demand accounting while bypassing detection.
        let _active_request = cpu.register_request();
        let mut request = synthetic_ocr(cpu.clone());
        assert_eq!(
            request.recognizer.compiled().as_cpu().unwrap().backend(),
            crate::runtime::Backend::NeonFma
        );
        let mixed = lines(&[512, 64, 64, 64, 64, 64, 64, 64, 64]);
        let profile = run(&mut request, &mixed, 1);
        assert_eq!(threads(&request), [2, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(request.workers[0].jobs.len(), 1);
        assert_eq!(request.workers[0].jobs[0], 0..1);
        #[cfg(feature = "profiling")]
        {
            assert_eq!(profile.recognition_owner_count, 8);
            assert_eq!(profile.recognition_owners[0].kernel_threads, 2);
            assert!(
                profile.recognition_owners[1..8]
                    .iter()
                    .all(|owner| owner.kernel_threads == 1)
            );
        }
        #[cfg(not(feature = "profiling"))]
        let _ = profile;
        let jobs = request
            .workers
            .iter()
            .map(|worker| worker.jobs.clone())
            .collect::<Vec<_>>();
        let warmed = storage(&request);
        run(&mut request, &mixed, 1);
        assert_eq!(storage(&request), warmed);

        let balanced = lines(&[64; 9]);
        run(&mut request, &balanced, 1);
        assert_eq!(threads(&request), [1; 8]);
        run(&mut request, &mixed, 1);
        assert_eq!(threads(&request), [2, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(
            request
                .workers
                .iter()
                .map(|worker| worker.jobs.clone())
                .collect::<Vec<_>>(),
            jobs
        );
        let rewarmed = storage(&request);
        let waiting = cpu.register_request();
        run(&mut request, &mixed, 1);
        assert_eq!(threads(&request), [1; 8]);
        assert_eq!(storage(&request), rewarmed);
        drop(waiting);
        run(&mut request, &mixed, 1);
        assert_eq!(threads(&request), [2, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(storage(&request), rewarmed);

        request
            .set_execution_context(
                cpu.clone(),
                CpuOptions {
                    line_workers: 8,
                    ..Default::default()
                },
            )
            .unwrap();
        run(&mut request, &mixed, 1);
        assert_eq!(threads(&request), [1; 8]);
        assert_eq!(storage(&request), rewarmed);
        request
            .set_execution_context(cpu.clone(), CpuOptions::default())
            .unwrap();
        // A batch is already independent work over several crops. Eight units
        // remain eight owners, but a multi-crop unit does not borrow a thread.
        let batched = lines(&[
            512, 512, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        ]);
        run(&mut request, &batched, 2);
        assert_eq!(threads(&request), [1; 8]);
        assert!(
            request
                .workers
                .iter()
                .all(|worker| worker.jobs.len() == 1 && worker.jobs[0].len() == 2)
        );
        let batch_warmed = storage(&request);
        run(&mut request, &batched, 2);
        assert_eq!(storage(&request), batch_warmed);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn automatic_recognition_preserves_dense_owners_and_counts_batches() {
        let cpu = Arc::new(CpuContext::new(8).unwrap());
        let mut parallel = synthetic_ocr(cpu.clone());
        let mut serial = synthetic_ocr(CpuContext::serial());
        let mut pixels = vec![0; 96 * 96 * 3];
        for top in [8, 36, 64] {
            for left in [8, 36, 64] {
                for y in top..top + 8 {
                    for x in left..left + 16 {
                        pixels[(y * 96 + x) * 3..(y * 96 + x) * 3 + 3].fill(255);
                    }
                }
            }
        }
        let dense = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let expected = serial
            .run(
                dense,
                crate::RunOptions {
                    capture_spans: true,
                },
            )
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert_eq!(expected.lines.len(), 9);
        assert!(expected.lines.iter().all(|line| {
            line.recognition_tensor_width == expected.lines[0].recognition_tensor_width
        }));
        let owners = if matches!(
            parallel.recognizer.compiled().as_cpu().unwrap().backend(),
            crate::runtime::Backend::Avx2Fma | crate::runtime::Backend::NeonFma
        ) {
            8
        } else {
            4
        };
        assert_eq!(
            parallel
                .run(
                    dense,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), owners);
        assert_eq!(parallel.workers.len(), owners);
        let storage = |request: &OcrRequest| {
            request
                .workers
                .iter()
                .map(|worker| {
                    (
                        worker.jobs.clone(),
                        worker.jobs.as_ptr(),
                        worker.jobs.capacity(),
                        worker.recognizer.retained_activation_bytes(),
                        worker.recognizer.retained_workspace_bytes(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let warmed = storage(&parallel);
        // A waiting caller changes the request's CPU share, not line ownership
        // or retained worker storage. The fixed executor still enforces its cap.
        let waiting = cpu.register_request();
        assert_eq!(
            parallel
                .run(
                    dense,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), owners);
        assert_eq!(storage(&parallel), warmed);
        drop(waiting);
        assert_eq!(
            parallel
                .run(
                    dense,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert_eq!(storage(&parallel), warmed);

        let sparse_pixels = two_lines();
        let sparse = ImageView::new(&sparse_pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let sparse_expected = serial
            .run(
                sparse,
                crate::RunOptions {
                    capture_spans: true,
                },
            )
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert_eq!(
            parallel
                .run(
                    sparse,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            sparse_expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 4);
        assert_eq!(parallel.workers.len(), owners); // Retain warm inactive owners.

        parallel.set_recognition_batch_size(3).unwrap();
        assert_eq!(
            parallel
                .run(
                    dense,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 4);
        assert_eq!(
            parallel.workers[..3]
                .iter()
                .map(|worker| worker.jobs.len())
                .sum::<usize>(),
            3
        );
        assert!(
            parallel.workers[..3]
                .iter()
                .flat_map(|worker| &worker.jobs)
                .all(|job| job.len() == 3)
        );

        parallel
            .set_execution_context(
                cpu,
                CpuOptions {
                    line_workers: 2,
                    ..Default::default()
                },
            )
            .unwrap();
        parallel.set_recognition_batch_size(1).unwrap();
        assert_eq!(
            parallel
                .run(
                    dense,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 2);
    }
    #[cfg(all(feature = "parallel", target_arch = "x86_64", target_os = "windows"))]
    #[test]
    fn automatic_sixteen_recognition_owners_reuse_storage_across_page_and_budget_changes() {
        let mut serial = synthetic_ocr(CpuContext::serial());
        if serial.recognizer.compiled().as_cpu().unwrap().backend()
            != crate::runtime::Backend::Avx2Fma
        {
            return;
        }
        let cpu = Arc::new(CpuContext::new(24).unwrap());
        let mut parallel = serial.fork().unwrap();
        parallel.set_cpu_context(cpu.clone()).unwrap();
        let mut pixels = vec![0; 96 * 96 * 3];
        for row in 0..6 {
            for column in 0..6 {
                let top = 5 + row * 15;
                let left = 5 + column * 15;
                for y in top..top + 6 {
                    for x in left..left + 9 {
                        pixels[(y * 96 + x) * 3..(y * 96 + x) * 3 + 3].fill(255);
                    }
                }
            }
        }
        let dense = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let expected = serial
            .run(
                dense,
                crate::RunOptions {
                    capture_spans: true,
                },
            )
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert_eq!(expected.lines.len(), 36);
        assert!(expected.lines.iter().all(|line| {
            !line.text.is_empty()
                && line.recognition_tensor_width == expected.lines[0].recognition_tensor_width
        }));
        assert_eq!(
            parallel
                .run(
                    dense,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 16);
        assert_eq!(parallel.cpu_budgets().recognizer_base_threads(), 1);
        assert_eq!(parallel.workers.len(), 16);
        let storage = |request: &OcrRequest| {
            request
                .workers
                .iter()
                .map(|worker| {
                    (
                        worker.jobs.clone(),
                        worker.jobs.as_ptr(),
                        worker.jobs.capacity(),
                        worker.recognizer.retained_activation_bytes(),
                        worker.recognizer.retained_workspace_bytes(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let warmed = storage(&parallel);
        let waiting = cpu.register_request();
        assert_eq!(
            parallel
                .run(
                    dense,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 16);
        assert_eq!(storage(&parallel), warmed);
        drop(waiting);

        let sparse_pixels = two_lines();
        let sparse = ImageView::new(&sparse_pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let sparse_expected = serial
            .run(
                sparse,
                crate::RunOptions {
                    capture_spans: true,
                },
            )
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert_eq!(
            parallel
                .run(
                    sparse,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            sparse_expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 4);
        assert_eq!(parallel.workers.len(), 16);
        assert_eq!(
            parallel
                .run(
                    dense,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 16);

        // Thirty-six lines become eighteen equal-width batches. The policy
        // counts these indivisible jobs rather than the number of text boxes.
        parallel.set_recognition_batch_size(2).unwrap();
        assert_eq!(
            parallel
                .run(
                    dense,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 8);
        assert_eq!(parallel.workers.len(), 16);
        assert_eq!(
            parallel.workers[..8]
                .iter()
                .map(|worker| worker.jobs.len())
                .sum::<usize>(),
            18
        );
        assert!(
            parallel.workers[..8]
                .iter()
                .flat_map(|worker| &worker.jobs)
                .all(|job| job.len() == 2)
        );
        parallel.set_recognition_batch_size(1).unwrap();
        assert_eq!(
            parallel
                .run(
                    dense,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 16);
        // Sparse and batched shapes may legitimately grow private buffers.
        // After visiting those shapes, a failed result-budget attempt and its
        // successful retry must retain the same owners and storage capacities.
        let warmed = storage(&parallel);
        let activation_bytes = parallel.retained_activation_bytes();
        let workspace_bytes = parallel.retained_workspace_bytes();
        let limit = parallel.plan.options.max_total_result_bytes;
        parallel.plan.options.max_total_result_bytes =
            expected.lines.len() * size_of::<LineRecord>() + expected.lines.len() - 1;
        assert!(
            parallel
                .run(
                    dense,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap_err()
                .to_string()
                .contains("max_total_result_bytes")
        );
        parallel.plan.options.max_total_result_bytes = limit;
        assert_eq!(
            parallel
                .run(
                    dense,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 16);
        assert_eq!(storage(&parallel), warmed);
        assert_eq!(parallel.retained_activation_bytes(), activation_bytes);
        assert_eq!(parallel.retained_workspace_bytes(), workspace_bytes);
    }
    #[test]
    fn factory_drops_the_warm_request_and_shares_only_immutable_configuration() {
        let cpu = CpuContext::serial();
        let mut template = synthetic_ocr(cpu.clone());
        let pixels = two_lines();
        let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let expected = serde_json::to_value(
            template
                .run(
                    image,
                    crate::RunOptions {
                        capture_spans: true,
                    },
                )
                .unwrap(),
        )
        .unwrap();
        template.run_with_trace(image, true, |_| Ok(())).unwrap();
        assert!(template.diagnostic_retained_activation_bytes() > 0);
        let request_lifetime = Arc::downgrade(&template.unwind_state);
        let recognizer_model = template.recognizer.compiled().clone();
        let dictionary = template.recognizer.dictionary().clone();
        let detector_limit = template.extractor.detector().execution_limits();
        let recognizer_limit = template.recognizer.execution_limits();
        let plan = template.plan().clone();
        drop(template);
        let factory = move || plan.request();
        assert!(request_lifetime.upgrade().is_none());
        let mut first = factory().unwrap();
        let mut second = factory().unwrap();
        for request in [&mut first, &mut second] {
            assert!(Arc::ptr_eq(request.cpu_context(), &cpu));
            assert!(Arc::ptr_eq(
                request.recognizer.compiled().as_cpu().unwrap(),
                recognizer_model.as_cpu().unwrap()
            ));
            assert!(Arc::ptr_eq(request.recognizer.dictionary(), &dictionary));
            assert_eq!(request.diagnostic_retained_activation_bytes(), 0);
            assert_eq!(request.result.text.capacity(), 0);
            assert!(request.workers.is_empty());
            assert!(request.recognition_results.is_empty());
            assert_eq!(
                request
                    .extractor
                    .detector()
                    .execution_limits()
                    .max_activation_bytes,
                detector_limit.max_activation_bytes
            );
            assert_eq!(
                request.recognizer.execution_limits().max_activation_bytes,
                recognizer_limit.max_activation_bytes
            );
            assert_eq!(
                serde_json::to_value(
                    request
                        .run(
                            image,
                            crate::RunOptions {
                                capture_spans: true
                            }
                        )
                        .unwrap()
                )
                .unwrap(),
                expected
            );
        }
        assert!(!Arc::ptr_eq(&first.unwind_state, &second.unwind_state));
    }
    #[test]
    fn synchronous_trace_keeps_non_send_observer_on_the_calling_thread() {
        #[cfg(feature = "parallel")]
        let cpu = Arc::new(CpuContext::new(3).unwrap());
        #[cfg(not(feature = "parallel"))]
        let cpu = CpuContext::serial();
        let mut ocr = synthetic_ocr(cpu);
        let pixels = two_lines();
        let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let calls = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let caller = std::thread::current().id();
        let result = ocr
            .run_with_trace(image, true, |_| {
                assert_eq!(std::thread::current().id(), caller);
                calls.set(calls.get() + 1);
                Ok(())
            })
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert!(calls.get() > 0);
        assert_eq!(result.detected_count, 2);
        assert_eq!(result.lines.len(), 2);
    }
    #[cfg(feature = "parallel")]
    #[test]
    fn width_priority_workers_match_serial_order_and_recover_after_observer_error() {
        let cpu = Arc::new(CpuContext::new(3).unwrap());
        let mut parallel = synthetic_ocr(cpu);
        let mut serial = synthetic_ocr(CpuContext::serial());
        let pixels = two_lines();
        let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let expected = serde_json::to_value(
            serial
                .run(
                    image,
                    crate::RunOptions {
                        capture_spans: true,
                    },
                )
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(
                parallel
                    .run(
                        image,
                        crate::RunOptions {
                            capture_spans: true
                        }
                    )
                    .unwrap()
            )
            .unwrap(),
            expected
        );
        assert_eq!(parallel.workers.len(), 2);
        let worker_storage = |request: &OcrRequest| {
            request
                .workers
                .iter()
                .map(|w| {
                    (
                        w.jobs.clone(),
                        w.jobs.as_ptr(),
                        w.jobs.capacity(),
                        w.recognizer.retained_activation_bytes(),
                        w.recognizer.retained_workspace_bytes(),
                        w.recognizer.projection_statistics().unwrap().rows,
                    )
                })
                .collect::<Vec<_>>()
        };
        let worker_buffers = worker_storage(&parallel);
        assert_eq!(worker_buffers[0].0.len(), 1);
        assert_eq!(worker_buffers[0].0.first(), Some(&(0..1)));
        assert_eq!(worker_buffers[1].0.len(), 1);
        assert_eq!(worker_buffers[1].0.first(), Some(&(1..2)));
        let buffers: Vec<_> = parallel
            .recognition_results
            .iter()
            .map(|slot| {
                let slot = slot.lock().unwrap();
                (slot.text.as_ptr() as usize, slot.retained_bytes())
            })
            .collect();
        let retained = parallel.retained_workspace_bytes();
        parallel
            .run(
                image,
                crate::RunOptions {
                    capture_spans: false,
                },
            )
            .unwrap();
        parallel
            .run(
                image,
                crate::RunOptions {
                    capture_spans: true,
                },
            )
            .unwrap();
        assert_eq!(parallel.retained_workspace_bytes(), retained);
        assert_eq!(worker_storage(&parallel), worker_buffers);
        for (slot, (pointer, bytes)) in parallel.recognition_results.iter().zip(buffers) {
            let slot = slot.lock().unwrap();
            assert_eq!(slot.text.as_ptr() as usize, pointer);
            assert_eq!(slot.retained_bytes(), bytes);
        }
        let original_budget = parallel.plan.options.max_total_result_bytes;
        parallel.plan.options.max_total_result_bytes = 2 * size_of::<LineRecord>() + 1;
        assert!(
            parallel
                .run(
                    image,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap_err()
                .to_string()
                .contains("max_total_result_bytes")
        );
        parallel.plan.options.max_total_result_bytes = original_budget;
        assert_eq!(
            serde_json::to_value(
                parallel
                    .run(
                        image,
                        crate::RunOptions {
                            capture_spans: true
                        }
                    )
                    .unwrap()
            )
            .unwrap(),
            expected
        );
        assert_eq!(worker_storage(&parallel), worker_buffers);
        assert!(
            parallel
                .run_with_trace(image, true, |_| Err(ImageError::new("observer stopped")))
                .is_err()
        );
        assert_eq!(
            serde_json::to_value(
                parallel
                    .run(
                        image,
                        crate::RunOptions {
                            capture_spans: true
                        }
                    )
                    .unwrap()
            )
            .unwrap(),
            expected
        );
        assert_eq!(worker_storage(&parallel), worker_buffers);
        let bad = ImageView::new(&[0; 3], 1, 1, 0, PixelFormat::Bgr24).unwrap();
        assert!(
            parallel
                .run(
                    bad,
                    crate::RunOptions {
                        capture_spans: false
                    }
                )
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            serde_json::to_value(
                parallel
                    .run(
                        image,
                        crate::RunOptions {
                            capture_spans: true
                        }
                    )
                    .unwrap()
            )
            .unwrap(),
            expected
        );
        assert_eq!(worker_storage(&parallel), worker_buffers);

        // Classification can rotate crop pixels by 180 degrees without changing
        // the recognition shape. Ownership and high-water storage must survive
        // that orientation change, including when the contents change.
        let oriented = parallel
            .extractor
            .extract(image)
            .unwrap()
            .try_to_owned()
            .unwrap();
        let geometry = oriented.as_view();
        let mut metadata = oriented.metadata().to_vec();
        let mut pixels = oriented.pixels().to_vec();
        for line in &mut metadata {
            crate::crop::rotate180(
                &mut pixels[line.offset..line.offset + line.bytes],
                line.width,
                line.height,
            )
            .unwrap();
            line.applied_rotation_degrees = 180;
        }
        let oriented = crate::lines::OwnedOrientedLines::new(
            geometry.resized_width(),
            geometry.resized_height(),
            geometry.width_ratio(),
            geometry.height_ratio(),
            metadata,
            pixels,
        )
        .unwrap();
        let rotated_expected: Vec<_> = oriented
            .as_view()
            .iter()
            .map(|line| {
                serde_json::to_value(serial.recognizer.recognize(line.image, true).unwrap())
                    .unwrap()
            })
            .collect();
        let cpu = parallel.cpu_context().clone();
        recognize_parallel::<false>(
            &cpu,
            &mut parallel.budgets,
            parallel.plan.cpu_options.line_workers == 0,
            &parallel.recognizer,
            &mut parallel.workers,
            &mut parallel.recognition_results,
            &mut parallel.recognition_order,
            parallel.plan.recognition_batch_size,
            &oriented.as_view(),
            true,
            original_budget,
            &mut StageProfile::default(),
        )
        .unwrap();
        for (slot, expected) in parallel.recognition_results.iter().zip(rotated_expected) {
            assert_eq!(
                serde_json::to_value(slot.lock().unwrap().as_view()).unwrap(),
                expected
            );
        }
        assert_eq!(worker_storage(&parallel), worker_buffers);
        assert_eq!(
            serde_json::to_value(
                parallel
                    .run(
                        image,
                        crate::RunOptions {
                            capture_spans: true
                        }
                    )
                    .unwrap()
            )
            .unwrap(),
            expected
        );
        assert_eq!(worker_storage(&parallel), worker_buffers);

        // A deliberate budget change can bind new shapes to a worker once.
        // Subsequent identical pages must reuse that binding and its storage.
        for line_workers in [1, 2] {
            parallel
                .set_execution_context(
                    parallel.cpu_context().clone(),
                    CpuOptions {
                        line_workers,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(
                serde_json::to_value(
                    parallel
                        .run(
                            image,
                            crate::RunOptions {
                                capture_spans: true
                            }
                        )
                        .unwrap()
                )
                .unwrap(),
                expected
            );
            let storage = worker_storage(&parallel);
            let retained = parallel.retained_workspace_bytes();
            parallel
                .run(
                    image,
                    crate::RunOptions {
                        capture_spans: false,
                    },
                )
                .unwrap();
            assert_eq!(
                serde_json::to_value(
                    parallel
                        .run(
                            image,
                            crate::RunOptions {
                                capture_spans: true
                            }
                        )
                        .unwrap()
                )
                .unwrap(),
                expected
            );
            assert_eq!(worker_storage(&parallel), storage);
            assert_eq!(parallel.retained_workspace_bytes(), retained);
        }
    }
}
