//! Pixel-to-text composition with borrowed results and bounded CPU workers.
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

pub struct Ocr {
    extractor: OrientedLineExtractor,
    recognizer: Recognizer,
    options: OcrOptions,
    lines: Vec<LineRecord>,
    text: String,
    runs: Vec<CtcRun>,
    detected_count: usize,
    detector_resized_width: usize,
    detector_resized_height: usize,
    width_ratio: f32,
    height_ratio: f32,
    cpu: Arc<CpuContext>,
    pinned_cpu: Option<Arc<CpuContext>>,
    cpu_options: CpuOptions,
    budgets: CpuBudgets,
    workers: Vec<RecognitionWorker>,
    recognition_results: Vec<Mutex<RecognitionSlot>>,
    recognition_order: Vec<usize>,
    recognition_batch_size: usize,
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
}
impl Ocr {
    pub fn new(
        extractor: OrientedLineExtractor,
        recognizer: Recognizer,
        options: OcrOptions,
    ) -> ImageResult<Self> {
        Self::new_with_execution(
            extractor,
            recognizer,
            options,
            CpuContext::shared_default()?,
            CpuOptions::default(),
        )
    }
    pub fn new_with_execution(
        mut extractor: OrientedLineExtractor,
        mut recognizer: Recognizer,
        options: OcrOptions,
        cpu: Arc<CpuContext>,
        cpu_options: CpuOptions,
    ) -> ImageResult<Self> {
        if options.max_total_result_bytes == 0 {
            return Err(ImageError::new("max_total_result_bytes must be positive"));
        }
        let budgets = cpu_options.resolve(
            cpu.capacity(),
            matches!(
                extractor.detector().compiled().options().layout,
                crate::runtime::LayoutPolicy::Auto
            ),
        )?;
        extractor.set_execution_context(cpu.clone(), cpu_options)?;
        recognizer.set_execution_context(cpu.clone(), budgets.recognizer_base_threads())?;
        Ok(Self {
            extractor,
            recognizer,
            options,
            lines: Vec::new(),
            text: String::new(),
            runs: Vec::new(),
            detected_count: 0,
            detector_resized_width: 0,
            detector_resized_height: 0,
            width_ratio: 0.,
            height_ratio: 0.,
            cpu,
            pinned_cpu: None,
            cpu_options,
            budgets,
            workers: Vec::new(),
            recognition_results: Vec::new(),
            recognition_order: Vec::new(),
            recognition_batch_size: 1,
            unwind_state: Arc::new(AtomicBool::new(false)),
        })
    }
    /// A fresh mutable request sharing compiled models, packs and the CPU cap.
    pub fn fork(&self) -> ImageResult<Self> {
        let mut fork = Self::new_with_execution(
            self.extractor.fork()?,
            self.recognizer.fork()?,
            self.options,
            self.cpu.clone(),
            self.cpu_options,
        )?;
        fork.recognition_batch_size = self.recognition_batch_size;
        Ok(fork)
    }
    /// Consume a request into an arena-free factory sharing its immutable
    /// models, packed weights, dictionary and CPU context. Warm activations,
    /// crop slabs, decoding buffers and diagnostic sessions are released.
    pub fn into_factory(self) -> impl Fn() -> ImageResult<Self> + Send + Sync + 'static {
        let detector = self.extractor.detector();
        let detector_model = detector.compiled().clone();
        let detector_options = detector.resolved_options();
        let detector_limits = detector.execution_limits();
        let classifier_model = self.extractor.classifier_model().cloned();
        let line_options = self.extractor.options();
        let recognizer_model = self.recognizer.compiled().clone();
        let dictionary = self.recognizer.dictionary().clone();
        let recognizer_options = self.recognizer.resolved_options();
        let recognizer_limits = self.recognizer.execution_limits();
        let cpu = self.cpu.clone();
        let cpu_options = self.cpu_options;
        let options = self.options;
        let recognition_batch_size = self.recognition_batch_size;
        drop(self);
        move || {
            let detector = crate::detector::Detector::new(
                detector_model.clone(),
                detector_options,
                detector_limits,
            )?;
            let extractor =
                OrientedLineExtractor::new(detector, classifier_model.clone(), line_options)?;
            let recognizer = Recognizer::new(
                recognizer_model.clone(),
                dictionary.clone(),
                recognizer_options,
                recognizer_limits,
            )?;
            let mut request =
                Self::new_with_execution(extractor, recognizer, options, cpu.clone(), cpu_options)?;
            request.recognition_batch_size = recognition_batch_size;
            Ok(request)
        }
    }
    pub fn cpu_context(&self) -> &Arc<CpuContext> {
        &self.cpu
    }
    pub(crate) fn unwound(&self) -> bool {
        self.unwind_state.load(Ordering::Relaxed)
    }
    pub fn cpu_budgets(&self) -> CpuBudgets {
        self.budgets
    }
    pub fn set_cpu_context(&mut self, cpu: Arc<CpuContext>) -> ImageResult<()> {
        self.set_execution_context(cpu, self.cpu_options)
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
            matches!(
                self.extractor.detector().compiled().options().layout,
                crate::runtime::LayoutPolicy::Auto
            ),
        )?;
        self.extractor.set_execution_context(cpu.clone(), options)?;
        self.recognizer
            .set_execution_context(cpu.clone(), budgets.recognizer_base_threads())?;
        for worker in &mut self.workers {
            worker
                .recognizer
                .set_execution_context(cpu.clone(), budgets.recognizer_base_threads())?;
        }
        self.cpu = cpu;
        self.cpu_options = options;
        self.budgets = budgets;
        Ok(())
    }
    pub fn options(&self) -> OcrOptions {
        self.options
    }
    /// Maximum crops per exact selected-width recognition batch. Defaults to one.
    pub fn recognition_batch_size(&self) -> usize {
        self.recognition_batch_size
    }
    /// Opt into same-width batching. Larger batches increase activation memory;
    /// each worker remains subject to its recognizer's execution limits.
    pub fn set_recognition_batch_size(&mut self, size: usize) -> ImageResult<()> {
        if size == 0 || size > i32::MAX as usize {
            return Err(ImageError::new(
                "recognition_batch_size must be within 1..=i32::MAX",
            ));
        }
        self.recognition_batch_size = size;
        Ok(())
    }
    pub fn recognizer_mut(&mut self) -> &mut Recognizer {
        // A caller can replace the recognizer through this reference. Worker
        // sessions must then be rebuilt from the new immutable model/options.
        self.workers.clear();
        &mut self.recognizer
    }
    pub fn extractor_mut(&mut self) -> &mut OrientedLineExtractor {
        &mut self.extractor
    }
    pub fn activation_bytes(&self) -> usize {
        self.extractor.detector().activation_bytes()
            + self.extractor.classifier_activation_bytes()
            + self.recognizer.activation_bytes()
            + self
                .workers
                .iter()
                .map(|w| w.recognizer.activation_bytes())
                .sum::<usize>()
    }
    pub fn retained_activation_bytes(&self) -> usize {
        self.extractor.detector().retained_activation_bytes()
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
    pub fn retained_workspace_bytes(&self) -> usize {
        let buffers = self.extractor.retained_workspace_bytes()
            + self.recognizer.retained_workspace_bytes()
            + self.lines.capacity() * size_of::<LineRecord>()
            + self.text.capacity()
            + self.runs.capacity() * size_of::<CtcRun>()
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
    pub fn run(&mut self, image: ImageView<'_>, capture: bool) -> ImageResult<OcrView<'_>> {
        let cpu = self.cpu.clone();
        let _demand = cpu.register_request();
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
        let cpu = self.cpu.clone();
        let _demand = cpu.register_request();
        let result = cpu.install(|| {
            self.run_impl::<false, true, _>(image, capture, |_| Ok(()), &mut profile)
        })?;
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
        let cpu = self.cpu.clone();
        let _demand = cpu.register_request();
        self.run_impl::<true, false, F>(image, capture, observer, &mut StageProfile::default())
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
        let inline_cpu = self.cpu.clone();
        let _inline = inline_cpu.enter_inline();
        let _active_request = inline_cpu.enter_request();
        #[cfg(test)]
        if !TRACE {
            self.cpu.assert_production_thread("OCR");
        }
        if self.unwound() {
            return Err(ImageError::new(
                "OCR request unwound; create a fresh request",
            ));
        }
        self.unwind_state.store(true, Ordering::Relaxed);
        let _unwind = RequestUnwindGuard(self.unwind_state.clone());
        self.budgets = self.cpu_options.resolve(
            self.cpu.capacity(),
            matches!(
                self.extractor.detector().compiled().options().layout,
                crate::runtime::LayoutPolicy::Auto
            ),
        )?;
        self.extractor.set_execution_context(
            self.cpu.clone(),
            self.budgets.extraction_options(
                self.cpu.request_capacity(),
                self.cpu_options.line_workers == 0,
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
            .filter(|&n| n <= self.options.max_total_result_bytes)
            .ok_or_else(|| ImageError::new("OCR exceeds max_total_result_bytes"))?;
        // All crop recognition shapes are checked before the first REC call.
        for line in oriented.iter() {
            if line.image.pixel_count()? > self.recognizer.resolved_options().max_image_pixels {
                return Err(ImageError::new("crop exceeds recognizer max_image_pixels"));
            }
            self.recognizer
                .select_width(line.image.width(), line.image.height())?;
        }
        self.lines.clear();
        self.text.clear();
        self.runs.clear();
        self.lines.try_reserve_exact(count).map_err(allocation)?;
        if PROFILE {
            profile.result_storage_ns = profile
                .result_storage_ns
                .saturating_add(storage_timer.elapsed());
        }
        let recognition_timer = Timer::<PROFILE>::start();
        let scheduled =
            !TRACE && count > 1 && (self.cpu.capacity() > 1 || self.recognition_batch_size > 1);
        if scheduled {
            recognize_parallel::<PROFILE>(
                &self.cpu,
                &mut self.budgets,
                self.cpu_options.line_workers == 0,
                &self.recognizer,
                &mut self.workers,
                &mut self.recognition_results,
                &mut self.recognition_order,
                self.recognition_batch_size,
                &oriented,
                capture,
                self.options
                    .max_total_result_bytes
                    .checked_sub(metadata_bytes)
                    .and_then(|bytes| bytes.checked_sub(count.saturating_sub(1)))
                    .ok_or_else(|| ImageError::new("OCR exceeds max_total_result_bytes"))?,
                profile,
            )?;
        }
        if !scheduled {
            self.recognizer.set_execution_context(
                self.cpu.clone(),
                self.budgets
                    .recognizer_threads_for_capacity(count, self.cpu.request_capacity()),
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
            } else {
                self.recognizer
                    .recognize_impl::<PROFILE>(line.image, capture, profile)?
            };
            let storage_timer = Timer::<PROFILE>::start();
            let text_length = self
                .text
                .len()
                .checked_add(usize::from(index != 0))
                .and_then(|n| n.checked_add(recognition.text.len()))
                .ok_or_else(|| ImageError::new("OCR text size overflow"))?;
            let span_count = self
                .runs
                .len()
                .checked_add(recognition.ctc_spans.map_or(0, CtcSpans::len))
                .ok_or_else(|| ImageError::new("OCR alignment count overflow"))?;
            let required = span_count
                .checked_mul(size_of::<CtcRun>())
                .and_then(|n| n.checked_add(text_length))
                .and_then(|n| n.checked_add(metadata_bytes))
                .filter(|&n| n <= self.options.max_total_result_bytes && n <= isize::MAX as usize)
                .ok_or_else(|| ImageError::new("OCR exceeds max_total_result_bytes"))?;
            let _ = required;
            self.text
                .try_reserve_exact(text_length - self.text.len())
                .map_err(allocation)?;
            self.runs
                .try_reserve_exact(span_count - self.runs.len())
                .map_err(allocation)?;
            if index != 0 {
                self.text.push('\n');
            }
            let text_start = self.text.len();
            self.text.push_str(recognition.text);
            let spans = recognition.ctc_spans.map(|spans| {
                let start = self.runs.len();
                let mut text_offset = text_start;
                for s in spans.iter() {
                    self.runs.push(CtcRun {
                        text_start: text_offset,
                        text_end: text_offset + s.text.len(),
                        score: s.score,
                        start_column: s.start_column,
                        end_column: s.end_column,
                    });
                    text_offset += s.text.len();
                }
                (start, self.runs.len())
            });
            self.lines.push(LineRecord {
                detection: line.metadata.detection,
                text_start,
                text_end: self.text.len(),
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
        self.detected_count = count;
        self.detector_resized_width = oriented.resized_width();
        self.detector_resized_height = oriented.resized_height();
        self.width_ratio = oriented.width_ratio();
        self.height_ratio = oriented.height_ratio();
        let output = self.view();
        if TRACE {
            observer.borrow_mut()(OcrTrace::Result(output))?;
        }
        Ok(output)
    }
    fn view(&self) -> OcrView<'_> {
        OcrView {
            text: &self.text,
            detected_count: self.detected_count,
            detector_resized_width: self.detector_resized_width,
            detector_resized_height: self.detector_resized_height,
            width_ratio: self.width_ratio,
            height_ratio: self.height_ratio,
            source: LineSource::Slab(&self.lines, &self.text, &self.runs),
        }
    }
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
    *budgets =
        budgets.for_recognition_units(units, automatic_line_workers, template.compiled().backend());
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
        });
    }
    let threads = budgets.recognizer_threads_for_capacity(units, cpu.request_capacity());
    for worker in &mut workers[..active] {
        worker
            .recognizer
            .set_execution_context(cpu.clone(), threads)?;
        worker.error = None;
        worker.profile = StageProfile::default();
    }
    assign_recognition_jobs(
        &mut workers[..active],
        order,
        batch_size,
        template,
        lines.metadata(),
    )?;
    let result_bytes = AtomicUsize::new(0);
    let budget_exceeded = AtomicBool::new(false);
    cpu.for_each_mut(&mut workers[..active], active, |_, worker| {
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
    if let Some((_, error)) = error {
        return Err(error);
    }
    if budget_exceeded.load(Ordering::Relaxed) {
        return Err(ImageError::new("OCR exceeds max_total_result_bytes"));
    }
    Ok(())
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
        runtime::{CompiledModel, ExecutionLimits},
    };
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
    fn compile(graph: Vec<u8>) -> Arc<CompiledModel> {
        Arc::new(CompiledModel::new(Arc::new(Model::load_bytes(&source(graph)).unwrap())).unwrap())
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
    pub(crate) fn synthetic_ocr(cpu: Arc<CpuContext>) -> Ocr {
        let (detector, rec) = synthetic_sources();
        let load = |bytes: &[u8]| {
            Arc::new(CompiledModel::new(Arc::new(Model::load_bytes(bytes).unwrap())).unwrap())
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
        Ocr::new_with_execution(
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
    pub(crate) fn synthetic_classifier() -> Arc<CompiledModel> {
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
    fn classified_ocr(cpu: Arc<CpuContext>, weights: [f32; 6], bias: [f32; 2]) -> Ocr {
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
        ocr.extractor = OrientedLineExtractor::new(
            ocr.extractor.detector().fork().unwrap(),
            Some(classifier),
            crate::lines::LineOptions {
                classifier_threshold: 0.5,
                ..Default::default()
            },
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
        let expected = images.map(|image| serial.run(image, true).unwrap().try_to_owned().unwrap());
        for (image, expected) in images.into_iter().zip(&expected) {
            assert_eq!(
                &parallel.run(image, true).unwrap().try_to_owned().unwrap(),
                expected
            );
        }
        let worker_storage = |request: &Ocr| {
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
                &parallel.run(image, true).unwrap().try_to_owned().unwrap(),
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
        let limit = parallel.options.max_total_result_bytes;
        for failed_limit in [1, 4 * size_of::<LineRecord>() + 3] {
            parallel.options.max_total_result_bytes = failed_limit;
            assert!(
                parallel
                    .run(images[0], true)
                    .unwrap_err()
                    .to_string()
                    .contains("max_total_result_bytes")
            );
            parallel.options.max_total_result_bytes = limit;
            assert_eq!(
                parallel
                    .run(images[0], true)
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
        let fewer_expected = serial.run(fewer, true).unwrap().try_to_owned().unwrap();
        assert_eq!(fewer_expected.lines.len(), 2);
        assert_eq!(
            parallel.run(fewer, true).unwrap().try_to_owned().unwrap(),
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
                .run(images[0], true)
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
        assert_eq!(ocr.into_factory()().unwrap().recognition_batch_size(), 7);
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
        let expected = serde_json::to_value(serial.run(image, true).unwrap()).unwrap();
        assert_eq!(serial.detected_count, 2);
        assert_eq!(
            serial.lines[0].recognition_tensor_width,
            serial.lines[1].recognition_tensor_width
        );
        #[cfg(feature = "parallel")]
        let capacities = [1, 3];
        #[cfg(not(feature = "parallel"))]
        let capacities = [1];
        for capacity in capacities {
            let mut ocr = synthetic_ocr(Arc::new(CpuContext::new(capacity).unwrap()));
            ocr.set_recognition_batch_size(2).unwrap();
            assert_eq!(
                serde_json::to_value(ocr.run(image, true).unwrap()).unwrap(),
                expected
            );
            assert_eq!(ocr.workers.len(), 1);
            let retained = ocr.retained_workspace_bytes();
            ocr.run(image, false).unwrap();
            assert_eq!(
                serde_json::to_value(ocr.run(image, true).unwrap()).unwrap(),
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
                serde_json::to_value(ocr.run(image, true).unwrap()).unwrap(),
                expected
            );
            let budget = ocr.options.max_total_result_bytes;
            ocr.options.max_total_result_bytes = 2 * size_of::<LineRecord>() + 1;
            assert!(
                ocr.run(image, true)
                    .unwrap_err()
                    .to_string()
                    .contains("max_total_result_bytes")
            );
            ocr.options.max_total_result_bytes = budget;
            assert_eq!(
                serde_json::to_value(ocr.run(image, true).unwrap()).unwrap(),
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
        let expected = serial.run(dense, true).unwrap().try_to_owned().unwrap();
        assert_eq!(expected.lines.len(), 9);
        assert!(expected.lines.iter().all(|line| {
            line.recognition_tensor_width == expected.lines[0].recognition_tensor_width
        }));
        let owners = if matches!(
            parallel.recognizer.compiled().backend(),
            crate::runtime::Backend::Avx2Fma | crate::runtime::Backend::NeonFma
        ) {
            8
        } else {
            4
        };
        assert_eq!(
            parallel.run(dense, true).unwrap().try_to_owned().unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), owners);
        assert_eq!(parallel.workers.len(), owners);
        let storage = |request: &Ocr| {
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
            parallel.run(dense, true).unwrap().try_to_owned().unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), owners);
        assert_eq!(storage(&parallel), warmed);
        drop(waiting);
        assert_eq!(
            parallel.run(dense, true).unwrap().try_to_owned().unwrap(),
            expected
        );
        assert_eq!(storage(&parallel), warmed);

        let sparse_pixels = two_lines();
        let sparse = ImageView::new(&sparse_pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let sparse_expected = serial.run(sparse, true).unwrap().try_to_owned().unwrap();
        assert_eq!(
            parallel.run(sparse, true).unwrap().try_to_owned().unwrap(),
            sparse_expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 4);
        assert_eq!(parallel.workers.len(), owners); // Retain warm inactive owners.

        parallel.set_recognition_batch_size(3).unwrap();
        assert_eq!(
            parallel.run(dense, true).unwrap().try_to_owned().unwrap(),
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
            parallel.run(dense, true).unwrap().try_to_owned().unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 2);
    }
    #[cfg(all(feature = "parallel", target_arch = "x86_64", target_os = "windows"))]
    #[test]
    fn automatic_sixteen_recognition_owners_reuse_storage_across_page_and_budget_changes() {
        let mut serial = synthetic_ocr(CpuContext::serial());
        if serial.recognizer.compiled().backend() != crate::runtime::Backend::Avx2Fma {
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
        let expected = serial.run(dense, true).unwrap().try_to_owned().unwrap();
        assert_eq!(expected.lines.len(), 36);
        assert!(expected.lines.iter().all(|line| {
            !line.text.is_empty()
                && line.recognition_tensor_width == expected.lines[0].recognition_tensor_width
        }));
        assert_eq!(
            parallel.run(dense, true).unwrap().try_to_owned().unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 16);
        assert_eq!(parallel.cpu_budgets().recognizer_base_threads(), 1);
        assert_eq!(parallel.workers.len(), 16);
        let storage = |request: &Ocr| {
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
            parallel.run(dense, true).unwrap().try_to_owned().unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 16);
        assert_eq!(storage(&parallel), warmed);
        drop(waiting);

        let sparse_pixels = two_lines();
        let sparse = ImageView::new(&sparse_pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let sparse_expected = serial.run(sparse, true).unwrap().try_to_owned().unwrap();
        assert_eq!(
            parallel.run(sparse, true).unwrap().try_to_owned().unwrap(),
            sparse_expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 4);
        assert_eq!(parallel.workers.len(), 16);
        assert_eq!(
            parallel.run(dense, true).unwrap().try_to_owned().unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 16);

        // Thirty-six lines become eighteen equal-width batches. The policy
        // counts these indivisible jobs rather than the number of text boxes.
        parallel.set_recognition_batch_size(2).unwrap();
        assert_eq!(
            parallel.run(dense, true).unwrap().try_to_owned().unwrap(),
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
            parallel.run(dense, true).unwrap().try_to_owned().unwrap(),
            expected
        );
        assert_eq!(parallel.cpu_budgets().line_workers(), 16);
        // Sparse and batched shapes may legitimately grow private buffers.
        // After visiting those shapes, a failed result-budget attempt and its
        // successful retry must retain the same owners and storage capacities.
        let warmed = storage(&parallel);
        let activation_bytes = parallel.retained_activation_bytes();
        let workspace_bytes = parallel.retained_workspace_bytes();
        let limit = parallel.options.max_total_result_bytes;
        parallel.options.max_total_result_bytes =
            expected.lines.len() * size_of::<LineRecord>() + expected.lines.len() - 1;
        assert!(
            parallel
                .run(dense, true)
                .unwrap_err()
                .to_string()
                .contains("max_total_result_bytes")
        );
        parallel.options.max_total_result_bytes = limit;
        assert_eq!(
            parallel.run(dense, true).unwrap().try_to_owned().unwrap(),
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
        let expected = serde_json::to_value(template.run(image, true).unwrap()).unwrap();
        template.run_with_trace(image, true, |_| Ok(())).unwrap();
        assert!(template.diagnostic_retained_activation_bytes() > 0);
        let request_lifetime = Arc::downgrade(&template.unwind_state);
        let recognizer_model = template.recognizer.compiled().clone();
        let dictionary = template.recognizer.dictionary().clone();
        let detector_limit = template.extractor.detector().execution_limits();
        let recognizer_limit = template.recognizer.execution_limits();
        let factory = template.into_factory();
        assert!(request_lifetime.upgrade().is_none());
        let mut first = factory().unwrap();
        let mut second = factory().unwrap();
        for request in [&mut first, &mut second] {
            assert!(Arc::ptr_eq(request.cpu_context(), &cpu));
            assert!(Arc::ptr_eq(
                request.recognizer.compiled(),
                &recognizer_model
            ));
            assert!(Arc::ptr_eq(request.recognizer.dictionary(), &dictionary));
            assert_eq!(request.diagnostic_retained_activation_bytes(), 0);
            assert_eq!(request.text.capacity(), 0);
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
                serde_json::to_value(request.run(image, true).unwrap()).unwrap(),
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
        let expected = serde_json::to_value(serial.run(image, true).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(parallel.run(image, true).unwrap()).unwrap(),
            expected
        );
        assert_eq!(parallel.workers.len(), 2);
        let worker_storage = |request: &Ocr| {
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
        parallel.run(image, false).unwrap();
        parallel.run(image, true).unwrap();
        assert_eq!(parallel.retained_workspace_bytes(), retained);
        assert_eq!(worker_storage(&parallel), worker_buffers);
        for (slot, (pointer, bytes)) in parallel.recognition_results.iter().zip(buffers) {
            let slot = slot.lock().unwrap();
            assert_eq!(slot.text.as_ptr() as usize, pointer);
            assert_eq!(slot.retained_bytes(), bytes);
        }
        let original_budget = parallel.options.max_total_result_bytes;
        parallel.options.max_total_result_bytes = 2 * size_of::<LineRecord>() + 1;
        assert!(
            parallel
                .run(image, true)
                .unwrap_err()
                .to_string()
                .contains("max_total_result_bytes")
        );
        parallel.options.max_total_result_bytes = original_budget;
        assert_eq!(
            serde_json::to_value(parallel.run(image, true).unwrap()).unwrap(),
            expected
        );
        assert_eq!(worker_storage(&parallel), worker_buffers);
        assert!(
            parallel
                .run_with_trace(image, true, |_| Err(ImageError::new("observer stopped")))
                .is_err()
        );
        assert_eq!(
            serde_json::to_value(parallel.run(image, true).unwrap()).unwrap(),
            expected
        );
        assert_eq!(worker_storage(&parallel), worker_buffers);
        let bad = ImageView::new(&[0; 3], 1, 1, 0, PixelFormat::Bgr24).unwrap();
        assert!(parallel.run(bad, false).unwrap().is_empty());
        assert_eq!(
            serde_json::to_value(parallel.run(image, true).unwrap()).unwrap(),
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
            parallel.cpu_options.line_workers == 0,
            &parallel.recognizer,
            &mut parallel.workers,
            &mut parallel.recognition_results,
            &mut parallel.recognition_order,
            parallel.recognition_batch_size,
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
            serde_json::to_value(parallel.run(image, true).unwrap()).unwrap(),
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
                serde_json::to_value(parallel.run(image, true).unwrap()).unwrap(),
                expected
            );
            let storage = worker_storage(&parallel);
            let retained = parallel.retained_workspace_bytes();
            parallel.run(image, false).unwrap();
            assert_eq!(
                serde_json::to_value(parallel.run(image, true).unwrap()).unwrap(),
                expected
            );
            assert_eq!(worker_storage(&parallel), storage);
            assert_eq!(parallel.retained_workspace_bytes(), retained);
        }
    }
}
