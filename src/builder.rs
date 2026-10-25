//! Convenient model loading and composition over the low-level CPU APIs.
use crate::{
    cpu::{CpuContext, CpuOptions},
    detector::{DetectorOptions, DetectorPlan},
    engine::{EngineLimits, OcrEngine},
    image::{ImageError, ImageResult},
    lines::{LineOptions, OrientedLinePlan},
    model::Model,
    ocr::{OcrOptions, OcrPlan},
    recognizer::{Dictionary, RecognizerOptions, RecognizerPlan},
    runtime::{CompileOptions, CompiledModel, ExecutionLimits, ExecutionTarget},
};
use std::{io::BufRead, path::Path, sync::Arc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelPreset {
    ChineseV6Tiny,
    ChineseV6Small,
    ChineseV6Medium,
}
impl ModelPreset {
    pub fn file_names(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::ChineseV6Tiny => ("det.onnx", "rec.onnx", "ppocr_keys.txt"),
            Self::ChineseV6Small => ("small_det.onnx", "small_rec.onnx", "rec_keys.txt"),
            Self::ChineseV6Medium => ("medium_det.onnx", "medium_rec.onnx", "rec_keys.txt"),
        }
    }
}

/// Immutable parsed models and dictionary. No raw ONNX bytes are retained.
#[derive(Clone)]
pub struct OcrModelSet {
    pub detection: Arc<Model>,
    pub recognition: Arc<Model>,
    pub dictionary: Arc<Dictionary>,
    pub classification: Option<Arc<Model>>,
}
impl OcrModelSet {
    pub fn new(
        detection: Arc<Model>,
        recognition: Arc<Model>,
        dictionary: Arc<Dictionary>,
        classification: Option<Arc<Model>>,
    ) -> Self {
        Self {
            detection,
            recognition,
            dictionary,
            classification,
        }
    }
    pub fn from_paths(
        detection: impl AsRef<Path>,
        recognition: impl AsRef<Path>,
        dictionary: impl AsRef<Path>,
        classification: Option<&Path>,
    ) -> ImageResult<Self> {
        fn load(path: impl AsRef<Path>, stage: &str) -> ImageResult<Arc<Model>> {
            Model::load_path(path)
                .map(Arc::new)
                .map_err(|e| ImageError::caused_by(e.kind(), format!("{stage} model"), e))
        }
        let detection = load(detection, "detection")?;
        let recognition = load(recognition, "recognition")?;
        let bytes = std::fs::read(dictionary)
            .map_err(|e| ImageError::caused_by(crate::error::ErrorKind::Io, "dictionary", e))?;
        let dictionary = Arc::new(Dictionary::from_utf8(&bytes)?);
        let classification = classification
            .map(|path| load(path, "classification"))
            .transpose()?;
        Ok(Self::new(
            detection,
            recognition,
            dictionary,
            classification,
        ))
    }
    /// Readers are consumed and released during parsing; they need not be seekable.
    pub fn from_readers<D: BufRead, R: BufRead, K: BufRead>(
        detection: D,
        recognition: R,
        mut dictionary: K,
        classification: Option<&mut dyn BufRead>,
    ) -> ImageResult<Self> {
        fn load(reader: impl BufRead, stage: &str) -> ImageResult<Arc<Model>> {
            Model::load_reader(reader)
                .map(Arc::new)
                .map_err(|e| ImageError::caused_by(e.kind(), format!("{stage} model"), e))
        }
        let detection = load(detection, "detection")?;
        let recognition = load(recognition, "recognition")?;
        let mut bytes = Vec::new();
        dictionary
            .read_to_end(&mut bytes)
            .map_err(|e| ImageError::caused_by(crate::error::ErrorKind::Io, "dictionary", e))?;
        let dictionary = Arc::new(Dictionary::from_utf8(&bytes)?);
        let classification = classification
            .map(|reader| load(reader, "classification"))
            .transpose()?;
        Ok(Self::new(
            detection,
            recognition,
            dictionary,
            classification,
        ))
    }
    /// Load the reference project's directory layout. `classification` controls
    /// whether `cls.onnx` is required and loaded. No files are downloaded.
    pub fn from_directory(
        directory: impl AsRef<Path>,
        preset: ModelPreset,
        classification: bool,
    ) -> ImageResult<Self> {
        let directory = directory.as_ref();
        let (det, rec, dict) = preset.file_names();
        Self::from_paths(
            directory.join(det),
            directory.join(rec),
            directory.join(dict),
            classification
                .then(|| directory.join("cls.onnx"))
                .as_deref(),
        )
    }
    pub fn builder(self) -> OcrBuilder {
        OcrBuilder::new(self)
    }
}

/// Configure stages before compiling and creating a reusable mutable OCR request.
/// Requests forked from the result share its compiled weights and CPU context.
pub struct OcrBuilder {
    models: OcrModelSet,
    compile: CompileOptions,
    compiler: Option<Arc<crate::runtime::Compiler>>,
    target: ExecutionTarget,
    stage_targets: [Option<ExecutionTarget>; 3],
    #[cfg(feature = "gpu")]
    gpu: Option<Arc<crate::runtime::gpu::GpuContext>>,
    #[cfg(feature = "gpu")]
    gpu_options: crate::runtime::gpu::GpuContextOptions,
    cpu: Option<Arc<CpuContext>>,
    cpu_options: CpuOptions,
    detector: DetectorOptions,
    recognizer: RecognizerOptions,
    lines: LineOptions,
    ocr: OcrOptions,
    detector_limits: ExecutionLimits,
    recognizer_limits: ExecutionLimits,
    classification: bool,
    batch_size: usize,
}
impl OcrBuilder {
    pub fn new(models: OcrModelSet) -> Self {
        Self {
            models,
            compile: CompileOptions::default(),
            compiler: None,
            target: ExecutionTarget::default(),
            stage_targets: [None; 3],
            #[cfg(feature = "gpu")]
            gpu: None,
            #[cfg(feature = "gpu")]
            gpu_options: Default::default(),
            cpu: None,
            cpu_options: CpuOptions::default(),
            detector: DetectorOptions::default(),
            recognizer: RecognizerOptions::default(),
            lines: LineOptions::default(),
            ocr: OcrOptions::default(),
            detector_limits: ExecutionLimits::default(),
            recognizer_limits: ExecutionLimits::default(),
            classification: true,
            batch_size: 1,
        }
    }
    pub fn compiler(mut self, compiler: Arc<crate::runtime::Compiler>) -> Self {
        self.compiler = Some(compiler);
        self
    }
    pub fn compile_options(mut self, options: CompileOptions) -> Self {
        self.compile = options;
        self
    }
    /// Select inference placement explicitly. CPU options continue to configure
    /// image processing when inference uses a GPU.
    pub fn execution_target(mut self, target: ExecutionTarget) -> Self {
        self.target = target;
        self
    }
    pub fn detector_target(mut self, target: ExecutionTarget) -> Self {
        self.stage_targets[0] = Some(target);
        self
    }
    pub fn recognizer_target(mut self, target: ExecutionTarget) -> Self {
        self.stage_targets[1] = Some(target);
        self
    }
    pub fn classifier_target(mut self, target: ExecutionTarget) -> Self {
        self.stage_targets[2] = Some(target);
        self
    }
    #[cfg(feature = "gpu")]
    pub fn gpu_context(mut self, context: Arc<crate::runtime::gpu::GpuContext>) -> Self {
        self.gpu = Some(context);
        self
    }
    #[cfg(feature = "gpu")]
    pub fn gpu_options(mut self, options: crate::runtime::gpu::GpuContextOptions) -> Self {
        self.gpu_options = options;
        self
    }
    pub fn cpu_context(mut self, cpu: Arc<CpuContext>) -> Self {
        self.cpu = Some(cpu);
        self
    }
    pub fn cpu_options(mut self, options: CpuOptions) -> Self {
        self.cpu_options = options;
        self
    }
    pub fn detector_options(mut self, options: DetectorOptions) -> Self {
        self.detector = options;
        self
    }
    pub fn recognizer_options(mut self, options: RecognizerOptions) -> Self {
        self.recognizer = options;
        self
    }
    pub fn line_options(mut self, options: LineOptions) -> Self {
        self.lines = options;
        self
    }
    pub fn ocr_options(mut self, options: OcrOptions) -> Self {
        self.ocr = options;
        self
    }
    pub fn detector_limits(mut self, limits: ExecutionLimits) -> Self {
        self.detector_limits = limits;
        self
    }
    pub fn recognizer_limits(mut self, limits: ExecutionLimits) -> Self {
        self.recognizer_limits = limits;
        self
    }
    pub fn direction_classification(mut self, enabled: bool) -> Self {
        self.classification = enabled;
        self
    }
    pub fn recognition_batch_size(mut self, count: usize) -> Self {
        self.batch_size = count;
        self
    }
    pub fn build(self) -> ImageResult<OcrPlan> {
        let cpu = match self.cpu {
            Some(cpu) => cpu,
            None => CpuContext::shared_default()?,
        };
        let compiler = self.compiler.unwrap_or_else(|| {
            Arc::new(crate::runtime::Compiler::new(
                crate::runtime::CompilerOptions {
                    #[cfg(feature = "gpu")]
                    gpu: self.gpu,
                    #[cfg(feature = "gpu")]
                    gpu_options: self.gpu_options,
                },
            ))
        });
        let compile = |model, stage: usize| -> ImageResult<CompiledModel> {
            Ok(compiler
                .compile(
                    model,
                    self.compile,
                    self.stage_targets[stage].unwrap_or(self.target),
                )
                .map_err(|e| {
                    e.at_stage(
                        [
                            crate::error::OcrStage::Detector,
                            crate::error::OcrStage::Recognizer,
                            crate::error::OcrStage::Classifier,
                        ][stage],
                    )
                })?)
        };
        let detection = compile(self.models.detection, 0)?;
        let recognition = compile(self.models.recognition, 1)?;
        let classification = if self.classification {
            self.models
                .classification
                .map(|model| compile(model, 2))
                .transpose()?
        } else {
            None
        };
        let detector = DetectorPlan::new(detection, self.detector, self.detector_limits)?;
        let extractor = OrientedLinePlan::new(detector, classification, self.lines)?;
        let recognizer = RecognizerPlan::new(
            recognition,
            self.models.dictionary,
            self.recognizer,
            self.recognizer_limits,
        )?;
        let plan = OcrPlan::new(
            extractor,
            recognizer,
            self.ocr,
            cpu,
            self.cpu_options,
            self.batch_size,
        )?;
        Ok(plan)
    }
    pub fn build_engine(self, limits: EngineLimits) -> ImageResult<OcrEngine> {
        OcrEngine::new(Arc::new(self.build()?), limits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{ImageView, PixelFormat};
    use std::io::{BufReader, Read};

    // A streaming source without Seek, returning short reads to exercise the
    // public reader contract rather than a byte-slice-only fast path.
    struct Chunks<'a>(&'a [u8]);
    impl Read for Chunks<'_> {
        fn read(&mut self, destination: &mut [u8]) -> std::io::Result<usize> {
            let count = self.0.len().min(destination.len()).min(7);
            destination[..count].copy_from_slice(&self.0[..count]);
            self.0 = &self.0[count..];
            Ok(count)
        }
    }

    #[test]
    fn streaming_loader_matches_low_level_pipeline_and_preserves_fork_settings() {
        let (det, rec) = crate::ocr::tests::synthetic_sources();
        let models = OcrModelSet::from_readers(
            BufReader::with_capacity(11, Chunks(&det)),
            BufReader::with_capacity(13, Chunks(&rec)),
            BufReader::new(Chunks(b"A\n")),
            None,
        )
        .unwrap();
        let cpu = CpuContext::serial();
        let mut request = models
            .builder()
            .cpu_context(cpu.clone())
            .detector_options(DetectorOptions {
                limit_side_length: 96,
                ..Default::default()
            })
            .recognition_batch_size(2)
            .build()
            .unwrap()
            .request()
            .unwrap();
        let mut low_level = crate::ocr::tests::synthetic_ocr(cpu.clone());
        let pixels = crate::ocr::tests::two_lines();
        let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let expected = low_level
            .run(
                image,
                crate::RunOptions {
                    capture_spans: true,
                },
            )
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert_eq!(
            request
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
        assert!(Arc::ptr_eq(request.cpu_context(), &cpu));
        assert_eq!(request.recognition_batch_size(), 2);
        let fork = request.fork().unwrap();
        assert!(Arc::ptr_eq(fork.cpu_context(), &cpu));
        assert_eq!(fork.recognition_batch_size(), 2);
        let plan = request.plan().clone();
        drop(request);
        let factory = move || plan.request();
        let mut recreated = factory().unwrap();
        assert!(Arc::ptr_eq(recreated.cpu_context(), &cpu));
        assert_eq!(recreated.recognition_batch_size(), 2);
        assert_eq!(
            recreated
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

    #[test]
    fn resize_and_width_options_survive_fork_and_factory() {
        let (det, rec) = crate::ocr::tests::synthetic_sources();
        let models = OcrModelSet::from_readers(&det[..], &rec[..], &b"A\n"[..], None).unwrap();
        let mut request = models
            .builder()
            .cpu_context(CpuContext::serial())
            .recognizer_options(RecognizerOptions {
                min_width: 48,
                score_mode: crate::recognizer::RecognitionScoreMode::Probability,
                ..Default::default()
            })
            .build()
            .unwrap()
            .request()
            .unwrap();
        request
            .set_detector_resize(crate::detector::DetResizePolicy::Min, 736)
            .unwrap();
        assert!(
            request
                .set_detector_resize(crate::detector::DetResizePolicy::Max, 0)
                .is_err()
        );
        for mut copy in [request.fork().unwrap(), request.plan().request().unwrap()] {
            assert_eq!(
                copy.recognizer_mut().resolved_options().score_mode,
                crate::recognizer::RecognitionScoreMode::Probability
            );
            let options = copy.extractor_mut().detector().resolved_options();
            assert_eq!(options.resize_policy, crate::detector::DetResizePolicy::Min);
            assert_eq!(options.limit_side_length, 736);
            assert_eq!(
                copy.recognizer_mut()
                    .select_width(10, 48)
                    .unwrap()
                    .tensor_width,
                64
            );
            assert_eq!(
                copy.recognizer_mut()
                    .select_width(321, 48)
                    .unwrap()
                    .tensor_width,
                352
            );
        }
        let (det, rec) = crate::ocr::tests::synthetic_sources();
        let mut accuracy = OcrModelSet::from_readers(&det[..], &rec[..], &b"A\n"[..], None)
            .unwrap()
            .builder()
            .cpu_context(CpuContext::serial())
            .recognizer_options(RecognizerOptions {
                min_width: 320,
                ..Default::default()
            })
            .build()
            .unwrap()
            .request()
            .unwrap();
        assert_eq!(
            accuracy
                .recognizer_mut()
                .select_width(10, 48)
                .unwrap()
                .tensor_width,
            320
        );
    }

    #[test]
    fn reader_errors_identify_the_failed_stage() {
        let (det, rec) = crate::ocr::tests::synthetic_sources();
        let error = OcrModelSet::from_readers(&b"invalid"[..], &rec[..], &b"A\n"[..], None)
            .err()
            .expect("invalid detector must fail");
        assert!(error.to_string().contains("detection model:"));
        let error = OcrModelSet::from_readers(&det[..], &rec[..], &b"\xff"[..], None)
            .err()
            .expect("invalid UTF-8 dictionary must fail");
        assert!(error.to_string().contains("UTF-8"));
        let error =
            OcrModelSet::from_readers(&det[..], &rec[..], &b"A\n"[..], Some(&mut &b"invalid"[..]))
                .err()
                .expect("invalid optional classifier must fail");
        assert!(error.to_string().contains("classification model:"));
    }
}
