//! Serial pixel-to-text composition with borrowed results and reusable output slabs.
use crate::{
    detector::{DetectionBox, DetectionTrace},
    image::{ImageError, ImageResult, ImageView},
    lines::{LineTrace, OrientedLineExtractor},
    recognizer::{
        CtcRun, CtcSpans, OwnedCtcSpan, RecognitionTrace, Recognizer, copy_spans, copy_text,
    },
};
use serde::Serialize;
use std::{cell::RefCell, mem::size_of};

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
fn allocation(error: impl std::fmt::Display) -> ImageError {
    ImageError::new(format!("OCR result allocation failed: {error}"))
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
}
impl Ocr {
    pub fn new(
        extractor: OrientedLineExtractor,
        recognizer: Recognizer,
        options: OcrOptions,
    ) -> ImageResult<Self> {
        if options.max_total_result_bytes == 0 {
            return Err(ImageError::new("max_total_result_bytes must be positive"));
        }
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
        })
    }
    pub fn options(&self) -> OcrOptions {
        self.options
    }
    pub fn recognizer_mut(&mut self) -> &mut Recognizer {
        &mut self.recognizer
    }
    pub fn extractor_mut(&mut self) -> &mut OrientedLineExtractor {
        &mut self.extractor
    }
    pub fn activation_bytes(&self) -> usize {
        self.extractor.detector().activation_bytes()
            + self.extractor.classifier_activation_bytes()
            + self.recognizer.activation_bytes()
    }
    pub fn retained_activation_bytes(&self) -> usize {
        self.extractor.detector().retained_activation_bytes()
            + self.extractor.classifier_retained_activation_bytes()
            + self.recognizer.retained_activation_bytes()
    }
    pub fn diagnostic_retained_activation_bytes(&self) -> usize {
        self.recognizer.diagnostic_retained_activation_bytes()
    }
    pub fn retained_workspace_bytes(&self) -> usize {
        self.extractor.retained_workspace_bytes()
            + self.recognizer.retained_workspace_bytes()
            + self.lines.capacity() * size_of::<LineRecord>()
            + self.text.capacity()
            + self.runs.capacity() * size_of::<CtcRun>()
    }
    /// Reuse one DET/optional CLS/REC pipeline in detector reading order.
    /// Errors expose no partial result; a later valid call can reuse the engine.
    pub fn run(&mut self, image: ImageView<'_>, capture: bool) -> ImageResult<OcrView<'_>> {
        self.run_impl::<false, _>(image, capture, |_| Ok(()))
    }
    pub fn run_with_trace<F>(
        &mut self,
        image: ImageView<'_>,
        capture: bool,
        observer: F,
    ) -> ImageResult<OcrView<'_>>
    where
        F: for<'a> FnMut(OcrTrace<'a>) -> ImageResult<()>,
    {
        self.run_impl::<true, F>(image, capture, observer)
    }
    fn run_impl<const TRACE: bool, F>(
        &mut self,
        image: ImageView<'_>,
        capture: bool,
        observer: F,
    ) -> ImageResult<OcrView<'_>>
    where
        F: for<'a> FnMut(OcrTrace<'a>) -> ImageResult<()>,
    {
        let observer = RefCell::new(observer);
        let oriented = if TRACE {
            self.extractor.extract_with_trace(
                image,
                |event| observer.borrow_mut()(OcrTrace::Detection(event)),
                |event| observer.borrow_mut()(OcrTrace::Line(event)),
            )?
        } else {
            self.extractor.extract(image)?
        };
        let count = oriented.lines.len();
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
        for (index, line) in oriented.iter().enumerate() {
            let recognition = if TRACE {
                self.recognizer
                    .recognize_with_trace(line.image, capture, |event| {
                        observer.borrow_mut()(OcrTrace::Recognition { index, event })
                    })?
            } else {
                self.recognizer.recognize(line.image, capture)?
            };
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
        }
        self.detected_count = count;
        self.detector_resized_width = oriented.resized_width;
        self.detector_resized_height = oriented.resized_height;
        self.width_ratio = oriented.width_ratio;
        self.height_ratio = oriented.height_ratio;
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
