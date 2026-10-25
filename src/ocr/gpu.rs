//! Request-owned, bounded resident OCR composition.
use crate::runtime::gpu::{
    heads::{Classification, Recognition},
    resident::*,
    resource,
    vision::{self, Crops},
};
use crate::{
    image::ImageView,
    ocr::{LineRecord, OcrPlan, RecognitionSlot, ResultWorkspace, RunOptions},
    recognizer::{CtcRun, RecognitionScoreMode},
    runtime::{RuntimeError, RuntimeResult},
};

use std::sync::Arc;

fn recognition_wave_size(
    requested: usize,
    units: usize,
    device_quota: bool,
    activation_limit: usize,
) -> usize {
    let memory_constrained = device_quota
        || activation_limit < crate::runtime::ExecutionLimits::default().max_activation_bytes;
    if memory_constrained {
        1
    } else {
        requested.min(units)
    }
}

pub(crate) use crate::error::OcrStage as Stage;
pub(crate) struct Failure {
    pub stage: Stage,
    pub error: RuntimeError,
}
impl Failure {
    fn at(stage: Stage) -> impl FnOnce(RuntimeError) -> Self {
        move |error| Self {
            stage,
            error: error.at_stage(stage),
        }
    }
}
pub(crate) struct Pipeline {
    pub failed_stage: Stage,
    detector: vision::Detector,
    classifier: Option<Classification>,
    recognition: Vec<Recognition>,
    pub statistics: GpuPipelineStatistics,
    results: Vec<RecognitionSlot>,
}
impl Pipeline {
    pub fn new(plan: &OcrPlan) -> Result<Self, Failure> {
        Self::create(plan).map_err(|(stage, error)| Failure {
            stage,
            error: error.at_stage(stage),
        })
    }
    fn create(plan: &OcrPlan) -> Result<Self, (Stage, RuntimeError)> {
        let extractor = plan.extractor();
        let recognizer = plan.recognizer();
        let mut stage = Stage::Detector;
        let result = (|| -> RuntimeResult<Self> {
            let det = extractor.detector();
            let compiled = det
                .compiled()
                .as_gpu()
                .ok_or_else(|| RuntimeError::new("resident detector requires GPU"))?
                .clone();
            let context = compiled.context().clone();
            stage = Stage::Recognizer;
            let rec = recognizer
                .compiled()
                .as_gpu()
                .ok_or_else(|| RuntimeError::new("resident recognizer requires GPU"))?
                .clone();
            if !Arc::ptr_eq(&context, rec.context()) {
                return Err(RuntimeError::new(
                    "OCR models belong to different GPU contexts",
                ));
            }
            stage = Stage::Classifier;
            let classifier = extractor
                .classifier()
                .map(|c| c.compiled())
                .map(|model| {
                    let cls = model
                        .as_gpu()
                        .ok_or_else(|| RuntimeError::new("resident classifier requires GPU"))?
                        .clone();
                    if !Arc::ptr_eq(&context, cls.context()) {
                        return Err(RuntimeError::new(
                            "OCR models belong to different GPU contexts",
                        ));
                    }
                    Classification::new(cls, extractor.options().classifier_execution_limits)
                })
                .transpose()?;
            stage = Stage::Detector;
            let detector = vision::Detector::new(compiled, det.execution_limits())?;
            stage = Stage::Recognizer;
            Ok(Self {
                failed_stage: Stage::Shared,
                detector,
                classifier,
                recognition: vec![Recognition::new(
                    rec,
                    recognizer.dictionary(),
                    32,
                    recognizer.execution_limits(),
                )?],
                statistics: Default::default(),
                results: Vec::new(),
            })
        })();
        result.map_err(|error| (stage, error))
    }
    pub fn retained_bytes(&self) -> usize {
        self.detector.session.retained_device_bytes()
            + self
                .classifier
                .as_ref()
                .map_or(0, Classification::retained_bytes)
            + self
                .recognition
                .iter()
                .map(Recognition::retained_bytes)
                .sum::<usize>()
    }
    pub fn activation_bytes(&self) -> usize {
        self.detector.session.retained_activation_bytes()
            + self
                .classifier
                .as_ref()
                .map_or(0, |c| c.session.retained_activation_bytes())
            + self
                .recognition
                .iter()
                .map(|r| r.session.retained_activation_bytes())
                .sum::<usize>()
    }
    pub fn workspace_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.recognition.capacity() * std::mem::size_of::<Recognition>()
            + self.statistics.retained_bytes()
            + self.results.capacity() * std::mem::size_of::<RecognitionSlot>()
            + self
                .results
                .iter()
                .map(RecognitionSlot::retained_bytes)
                .sum::<usize>()
            + self.detector.session.retained_planning_bytes()
            + self
                .classifier
                .as_ref()
                .map_or(0, |c| c.session.retained_planning_bytes())
            + self
                .recognition
                .iter()
                .map(|r| r.session.retained_planning_bytes())
                .sum::<usize>()
    }
    pub fn counter_snapshots(
        &self,
    ) -> std::collections::BTreeMap<Stage, Vec<crate::runtime::StatisticsCounters>> {
        let mut output = std::collections::BTreeMap::new();
        output.insert(
            Stage::Detector,
            vec![self.detector.session.counter_snapshot()],
        );
        output.insert(
            Stage::Recognizer,
            self.recognition
                .iter()
                .map(Recognition::counter_snapshot)
                .collect(),
        );
        if let Some(cls) = &self.classifier {
            output.insert(Stage::Classifier, vec![cls.session.counter_snapshot()]);
        }
        output
    }
    pub fn execution_statistics(
        &self,
    ) -> std::collections::BTreeMap<crate::error::OcrStage, Vec<crate::runtime::ExecutionStatistics>>
    {
        let mut output = std::collections::BTreeMap::new();
        output.insert(
            crate::error::OcrStage::Detector,
            vec![self.detector.session.statistics()],
        );
        output.insert(
            crate::error::OcrStage::Recognizer,
            self.recognition
                .iter()
                .map(Recognition::execution_statistics)
                .collect(),
        );
        if let Some(cls) = &self.classifier {
            output.insert(
                crate::error::OcrStage::Classifier,
                vec![cls.session.statistics()],
            );
        }
        output
    }
    pub(super) fn run(
        &mut self,
        image: ImageView<'_>,
        plan: &OcrPlan,
        options: RunOptions,
        profiling: bool,
        result: &mut ResultWorkspace,
    ) -> Result<(), Failure> {
        self.run_inner(image, plan, options, profiling, result)
            .map_err(Failure::at(self.failed_stage))
    }
    fn run_inner(
        &mut self,
        image: ImageView<'_>,
        plan: &OcrPlan,
        run_options: RunOptions,
        profiling: bool,
        result: &mut ResultWorkspace,
    ) -> RuntimeResult<()> {
        let extractor = plan.extractor();
        let recognizer = plan.recognizer();
        let options = plan.options;
        let batch_size = plan.recognition_batch_size;
        let capture = run_options.capture_spans;
        self.statistics = Default::default();
        self.failed_stage = Stage::Detector;
        let line_options = extractor.options();
        let (detection, prepared_crops) =
            self.detector
                .run_with_crops(image, extractor.detector().options(), profiling)?;
        vision::merge_statistics(&mut self.statistics, detection.statistics.clone());
        let count = detection.boxes.len();
        let rec_options = recognizer.options();
        result.clear();
        result.detected_count = count;
        result.detector_resized_width = detection.size.width;
        result.detector_resized_height = detection.size.height;
        result.width_ratio = detection.size.width_ratio;
        result.height_ratio = detection.size.height_ratio;
        let mut used = count
            .checked_mul(crate::ocr::line_record_bytes())
            .and_then(|n| n.checked_add(count.saturating_sub(1)))
            .filter(|&n| n <= options.max_total_result_bytes)
            .ok_or_else(|| resource("OCR exceeds max_total_result_bytes"))?;
        if count == 0 {
            return Ok(());
        }
        let crops = match prepared_crops {
            Some(crops) => crops.finish(line_options)?,
            None => Crops::prepare(&detection, line_options)?,
        };
        vision::merge_statistics(&mut self.statistics, crops.statistics.clone());
        let mut sizes = Vec::with_capacity(count);
        for &[width, height] in &crops.sizes {
            if width * height > rec_options.max_image_pixels {
                return Err(resource("crop exceeds recognizer max_image_pixels"));
            }
            if self.classifier.is_some()
                && width * height > line_options.max_classifier_image_pixels
            {
                return Err(resource("crop exceeds classifier max_image_pixels"));
            }
            sizes.push(
                recognizer
                    .select_width(width, height)
                    .map_err(RuntimeError::from)?,
            );
        }
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by_key(|&i| (sizes[i].tensor_width, i));
        let mut units: Vec<Vec<usize>> = Vec::new();
        for &index in &order {
            if let Some(last) = units.last_mut()
                && last.len() < batch_size
                && sizes[last[0]].tensor_width == sizes[index].tensor_width
            {
                last.push(index);
                continue;
            }
            units.push(vec![index]);
        }
        let context_options = self.detector.context.options();
        // Admission counts independent requests. Within one request, a bounded
        // wave owns disjoint recognition arenas so widths can share a submission
        // without padding or changing the model's time-step contract. Preserve
        // single-arena execution for explicitly constrained memory budgets.
        let bound = recognition_wave_size(
            context_options.recognition_wave_size,
            units.len(),
            context_options.max_device_bytes.is_some(),
            recognizer.execution_limits().max_activation_bytes,
        );
        while self.recognition.len() < bound {
            self.recognition.push(self.recognition[0].fork()?);
        }
        if self.results.len() < count {
            self.results.resize_with(count, RecognitionSlot::new);
        }
        result.lines.try_reserve(count).map_err(|e| {
            RuntimeError::caused_by(
                crate::error::ErrorKind::Allocation,
                "OCR line allocation",
                e,
            )
        })?;
        for detection in &detection.boxes {
            result.lines.push(LineRecord {
                detection: *detection,
                text_start: 0,
                text_end: 0,
                recognition_score: 0.,
                classification_score: 0.,
                classification_label: 0,
                applied_rotation_degrees: 0,
                emitted_count: 0,
                recognition_content_width: 0,
                recognition_tensor_width: 0,
                recognition_time_steps: 0,
                spans: None,
            });
        }
        let context = self.detector.context.clone();
        // Classification has a fixed input shape, so combine adjacent crops
        // independently of recognition width. Only one bounded lookahead is
        // retained in addition to the submitted waves, never all page crops.
        // A global device quota also covers the lookahead crop pixels, whose
        // sizes are independent of classifier activation storage. Preserve the
        // original four-crop bound under that quota even if reshape admits 16.
        let classification_limit = if context_options.max_device_bytes.is_some() {
            context_options.classification_batch_size.min(4)
        } else {
            context_options.classification_batch_size
        };
        let mut classification_batch = if self.classifier.is_some() {
            count.min(classification_limit)
        } else {
            1
        };
        if let Some(classifier) = &mut self.classifier {
            self.failed_stage = Stage::Classifier;
            classification_batch = classifier.prepare_batch(classification_batch)?;
        }
        let mut prepared = std::collections::VecDeque::new();
        let mut next_crop = 0;
        let admission = profiling.then(std::time::Instant::now);
        let gate = context.acquire_gate()?;
        if let Some(timer) = admission {
            self.statistics.admission_wait_ns += timer.elapsed().as_nanos() as u64;
        }
        let mut pending = std::collections::VecDeque::new();
        // Each width in a wave has its own activation and parameter arena. Batch
        // retains immutable command snapshots and interleaves independent graph
        // dispatches by dependency depth. Submit before reshaping those sessions
        // for the next wave; at most two waves retain scratch/readback data.
        for wave in units.chunks(bound).map(Some).chain(std::iter::once(None)) {
            if let Some(wave) = wave {
                let mut batch = Batch::new(self.detector.context.clone())?;
                batch.profiling = profiling;
                let mut outputs = Vec::new();
                let mut total = 0usize;
                for (slot, indices) in wave.iter().enumerate() {
                    let mut images = Vec::with_capacity(indices.len());
                    let mut classes = Vec::new();
                    for &index in indices {
                        if prepared.is_empty() {
                            let end = (next_crop + classification_batch).min(count);
                            self.failed_stage = Stage::Detector;
                            let raw = order[next_crop..end]
                                .iter()
                                .map(|&i| crops.record(&mut batch, &detection.image, i))
                                .collect::<RuntimeResult<Vec<_>>>()?;
                            if let Some(classifier) = &mut self.classifier {
                                self.failed_stage = Stage::Classifier;
                                let (winners, widths) =
                                    classifier.record_batch(&mut batch, &raw)?;
                                for (row, image) in raw.iter().enumerate() {
                                    let image = vision::orient_at(
                                        &mut batch,
                                        image,
                                        &winners,
                                        row,
                                        line_options.classifier_threshold,
                                    )?;
                                    prepared.push_back((
                                        order[next_crop + row],
                                        image,
                                        Some((winners.clone(), row, widths[row])),
                                    ));
                                }
                            } else {
                                prepared.push_back((
                                    order[next_crop],
                                    raw.into_iter().next().unwrap(),
                                    None,
                                ));
                            }
                            next_crop = end;
                        }
                        let (prepared_index, image, class) = prepared.pop_front().unwrap();
                        debug_assert_eq!(prepared_index, index);
                        images.push(image);
                        if let Some(class) = class {
                            classes.push(class);
                        }
                    }
                    let shapes = indices.iter().map(|&i| sizes[i]).collect::<Vec<_>>();
                    self.failed_stage = Stage::Recognizer;
                    let (tokens, steps) = self.recognition[slot].record(
                        &mut batch,
                        &images,
                        &shapes,
                        rec_options.score_mode == RecognitionScoreMode::Probability,
                        capture,
                        rec_options
                            .max_result_bytes
                            .min(options.max_total_result_bytes - used),
                    )?;
                    let bytes = (steps + 1) * indices.len() * 24;
                    let offset = total;
                    total = total
                        .checked_add(bytes + classes.len() * 16)
                        .ok_or_else(|| resource("OCR compact result overflow"))?;
                    outputs.push((tokens, steps, classes, offset, bytes));
                }
                let compact = batch.storage("OCR compact output", total)?;
                for (tokens, _, classes, offset, bytes) in &outputs {
                    batch.copy(tokens, 0, &compact, *offset as u64, *bytes as u64);
                    for (j, (winner, row, _)) in classes.iter().enumerate() {
                        batch.copy(
                            winner,
                            (row * 16) as u64,
                            &compact,
                            (*offset + *bytes + j * 16) as u64,
                            16,
                        );
                    }
                }
                self.failed_stage = Stage::Shared;
                pending.push_back((wave, outputs, batch.submit(&compact, total, false, &gate)?));
                if pending.len() < 2 {
                    continue;
                }
            }
            let Some((wave, outputs, submitted)) = pending.pop_front() else {
                continue;
            };
            self.failed_stage = Stage::Shared;
            let (data, statistics) = submitted.finish()?;
            vision::merge_statistics(&mut self.statistics, statistics);
            for (indices, (_, steps, classes, offset, bytes)) in wave.iter().zip(outputs) {
                let decoded = Recognition::parse(
                    &data[offset..offset + bytes],
                    indices.len(),
                    steps,
                    recognizer.dictionary().class_count(),
                )?;
                let batch_bytes = decoded
                    .iter()
                    .try_fold(0usize, |sum, line| {
                        sum.checked_add(line.text_bytes)?.checked_add(if capture {
                            line.tokens.len() * std::mem::size_of::<crate::recognizer::CtcRun>()
                        } else {
                            0
                        })
                    })
                    .filter(|&n| n <= rec_options.max_result_bytes);
                if batch_bytes.is_none() {
                    return Err(resource("recognition batch exceeds max_result_bytes"));
                }
                for (j, (&index, decoded)) in indices.iter().zip(decoded).enumerate() {
                    let spans_bytes = if capture {
                        decoded.tokens.len() * std::mem::size_of::<crate::recognizer::CtcRun>()
                    } else {
                        0
                    };
                    used = used
                        .checked_add(decoded.text_bytes)
                        .and_then(|n| n.checked_add(spans_bytes))
                        .filter(|&n| n <= options.max_total_result_bytes)
                        .ok_or_else(|| resource("OCR exceeds max_total_result_bytes"))?;
                    let slot = &mut self.results[index];
                    slot.text.clear();
                    slot.runs.clear();
                    slot.text.try_reserve(decoded.text_bytes).map_err(|e| {
                        RuntimeError::caused_by(
                            crate::error::ErrorKind::Allocation,
                            "OCR text allocation",
                            e,
                        )
                    })?;
                    if capture {
                        slot.runs.try_reserve(decoded.tokens.len()).map_err(|e| {
                            RuntimeError::caused_by(
                                crate::error::ErrorKind::Allocation,
                                "OCR span allocation",
                                e,
                            )
                        })?;
                    }
                    let emitted_count = decoded.tokens.len();
                    for token in decoded.tokens.iter() {
                        let label = recognizer
                            .dictionary()
                            .token(token.class)
                            .ok_or_else(|| RuntimeError::new("GPU token out of range"))?;
                        let start = slot.text.len();
                        slot.text.push_str(label);
                        if capture {
                            slot.runs.push(CtcRun {
                                text_start: start,
                                text_end: slot.text.len(),
                                score: token.score,
                                start_column: token.start,
                                end_column: token.end,
                            });
                        }
                    }
                    if slot.text.len() != decoded.text_bytes {
                        return Err(RuntimeError::new("GPU token byte count mismatch"));
                    }
                    let (label, score) = if classes.is_empty() {
                        (0, 0.0)
                    } else {
                        let start = offset + bytes + j * 16;
                        let class = crate::orientation::parse_gpu_classification(
                            &data[start..start + 16],
                            classes[j].2,
                        )?;
                        (class.label, class.score)
                    };
                    result.lines[index] = LineRecord {
                        detection: detection.boxes[index],
                        text_start: 0,
                        text_end: 0,
                        recognition_score: decoded.score,
                        classification_score: score,
                        classification_label: label,
                        applied_rotation_degrees: if label == 1
                            && score > line_options.classifier_threshold
                        {
                            180
                        } else {
                            0
                        },
                        emitted_count,
                        recognition_content_width: sizes[index].resized_width,
                        recognition_tensor_width: sizes[index].tensor_width,
                        recognition_time_steps: steps,
                        spans: None,
                    };
                }
            }
        }
        let text_bytes: usize = self.results[..count]
            .iter()
            .map(|s| s.text.len())
            .sum::<usize>()
            + count.saturating_sub(1);
        let run_count: usize = self.results[..count].iter().map(|s| s.runs.len()).sum();
        result.text.try_reserve(text_bytes).map_err(|e| {
            RuntimeError::caused_by(
                crate::error::ErrorKind::Allocation,
                "OCR text allocation",
                e,
            )
        })?;
        result.runs.try_reserve(run_count).map_err(|e| {
            RuntimeError::caused_by(
                crate::error::ErrorKind::Allocation,
                "OCR span allocation",
                e,
            )
        })?;
        for (index, (line, slot)) in result.lines.iter_mut().zip(&self.results).enumerate() {
            if index > 0 {
                result.text.push('\n');
            }
            line.text_start = result.text.len();
            result.text.push_str(&slot.text);
            line.text_end = result.text.len();
            if capture {
                let start = result.runs.len();
                result.runs.extend(slot.runs.iter().map(|r| CtcRun {
                    text_start: r.text_start + line.text_start,
                    text_end: r.text_end + line.text_start,
                    ..*r
                }));
                line.spans = Some((start, result.runs.len()));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod scheduling_tests {
    use super::*;

    #[test]
    fn recognition_waves_are_independent_of_request_admission_and_preserve_quotas() {
        let default_limit = crate::runtime::ExecutionLimits::default().max_activation_bytes;
        assert_eq!(recognition_wave_size(4, 13, false, default_limit), 4);
        assert_eq!(recognition_wave_size(4, 2, false, default_limit), 2);
        assert_eq!(recognition_wave_size(1, 13, false, default_limit), 1);
        assert_eq!(recognition_wave_size(4, 13, true, default_limit), 1);
        assert_eq!(recognition_wave_size(4, 13, false, default_limit - 1), 1);
    }
}
