//! Optional B-Scale paragraph grouping for original images and detected line boxes.

mod decode;
mod features;
pub(crate) mod scorer;

use crate::{
    cpu::CpuContext,
    error::{ErrorKind, FailureProvenance, ImportError, OcrStage, RuntimeError},
    image::{ImageError, ImageResult, ImageView},
    profile::Timer,
    runtime::{
        Backend, Compiler, CompilerOptions, ExecutionBackend, ExecutionConfig, ExecutionLimits,
        ExecutionStatistics, ExecutionTarget, KernelPolicy, MemorySnapshot,
    },
};
use features::PreparedPage;
use scorer::{CpuScorer, CpuWeights, Input, Weights};
use std::{fs::File, io::BufReader, path::Path, sync::Arc};

#[cfg(feature = "gpu")]
use crate::runtime::gpu::paragraph::{ParagraphGpuPlan, ParagraphGpuRequest};

/// Shared weights for the epoch 15 B-Scale EMA export.
#[derive(Clone)]
pub struct ParagraphPlan {
    cpu: Option<Arc<CpuWeights>>,
    backend: Option<Backend>,
    #[cfg(feature = "gpu")]
    gpu: Option<Arc<ParagraphGpuPlan>>,
    #[cfg(feature = "gpu")]
    auto: bool,
    fallback: Option<FailureProvenance>,
    limits: ExecutionLimits,
}

impl ParagraphPlan {
    /// Load the supplied ONNX directly. The default execution target is SIMD CPU.
    /// Paragraph scoring uses FP32 on either target.
    pub fn load_path(path: impl AsRef<Path>, execution: ExecutionConfig) -> ImageResult<Self> {
        let compiler = Compiler::new(CompilerOptions {
            #[cfg(feature = "gpu")]
            gpu: execution.gpu,
            #[cfg(feature = "gpu")]
            gpu_options: execution.gpu_options,
        });
        Self::load(
            path.as_ref(),
            &compiler,
            execution.target,
            KernelPolicy::Auto,
            ExecutionLimits::default(),
        )
    }

    pub(crate) fn load(
        path: &Path,
        compiler: &Compiler,
        target: ExecutionTarget,
        kernels: KernelPolicy,
        limits: ExecutionLimits,
    ) -> ImageResult<Self> {
        let file = File::open(path).map_err(|error| {
            ImageError::caused_by(
                ErrorKind::Io,
                format!("open paragraph model {}", path.display()),
                error,
            )
            .at_stage(OcrStage::Paragraph)
        })?;
        let weights = crate::onnx::parse_b_scale(BufReader::new(file))
            .and_then(Weights::load)
            .map_err(import_error)?;
        #[cfg(not(feature = "gpu"))]
        let _ = compiler;
        let mut fallback = None;
        #[cfg(feature = "gpu")]
        let mut gpu = None;
        if target != ExecutionTarget::Cpu {
            #[cfg(feature = "gpu")]
            let result = compiler
                .gpu()
                .and_then(|context| ParagraphGpuPlan::new(context, &weights));
            #[cfg(not(feature = "gpu"))]
            let result: Result<(), RuntimeError> = Err(RuntimeError::with_kind(
                ErrorKind::Unsupported,
                "paragraph GPU execution requires the gpu Cargo feature",
            ));
            match result {
                #[cfg(feature = "gpu")]
                Ok(plan) => gpu = Some(Arc::new(plan)),
                #[cfg(not(feature = "gpu"))]
                Ok(()) => unreachable!(),
                Err(error) if target == ExecutionTarget::Auto && error.allows_fallback() => {
                    fallback = Some(failure(&error));
                }
                Err(error) => return Err(stage_error(error.into())),
            }
        }
        let (cpu, backend) = if target == ExecutionTarget::Gpu {
            (None, None)
        } else {
            let backend = crate::runtime::simd::select_backend(kernels)
                .map_err(|msg| ImageError::with_kind(ErrorKind::Unsupported, msg))
                .map_err(stage_error)?;
            (
                Some(Arc::new(
                    CpuWeights::pack(weights)
                        .map_err(ImageError::from)
                        .map_err(stage_error)?,
                )),
                Some(backend),
            )
        };
        Ok(Self {
            cpu,
            backend,
            #[cfg(feature = "gpu")]
            gpu,
            #[cfg(feature = "gpu")]
            auto: target == ExecutionTarget::Auto,
            fallback,
            limits,
        })
    }

    /// Create an independent reusable workspace with one CPU inference thread.
    pub fn request(&self) -> ImageResult<ParagraphRequest> {
        self.request_in(CpuContext::serial(), 1)
    }

    pub(crate) fn request_in(
        &self,
        context: Arc<CpuContext>,
        threads: usize,
    ) -> ImageResult<ParagraphRequest> {
        #[allow(unused_mut)]
        let mut fallback = self.fallback.clone();
        #[cfg(feature = "gpu")]
        let gpu = match self
            .gpu
            .as_ref()
            .map(|plan| ParagraphGpuRequest::new(plan.clone()))
            .transpose()
        {
            Ok(gpu) => gpu,
            Err(error) if self.auto && error.allows_fallback() => {
                fallback = Some(failure(&error));
                None
            }
            Err(error) => return Err(stage_error(error.into())),
        };
        Ok(ParagraphRequest {
            plan: self.clone(),
            context,
            threads,
            cpu: CpuScorer::default(),
            #[cfg(feature = "gpu")]
            gpu,
            #[cfg(feature = "gpu")]
            previous_gpu_stats: None,
            fallback,
            last_backend: None,
            page: None,
            logits: Vec::new(),
            groups: Vec::new(),
        })
    }

    pub(crate) fn shared_weights(&self) -> MemorySnapshot {
        MemorySnapshot {
            host_bytes: self.cpu.as_ref().map_or(0, |weights| weights.bytes()),
            #[cfg(feature = "gpu")]
            device_bytes: self.gpu.as_ref().map_or(0, |plan| plan.weight_bytes()),
            #[cfg(not(feature = "gpu"))]
            device_bytes: 0,
        }
    }
}

/// Wall time for the three parts of a paragraph request.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct ParagraphProfile {
    pub preprocess_ns: u64,
    /// Includes any device transfer and synchronous wait for the logits.
    pub inference_ns: u64,
    pub decode_ns: u64,
    pub candidate_count: usize,
}

/// An exclusive workspace for repeated paragraph requests.
pub struct ParagraphRequest {
    plan: ParagraphPlan,
    context: Arc<CpuContext>,
    threads: usize,
    cpu: CpuScorer,
    #[cfg(feature = "gpu")]
    gpu: Option<ParagraphGpuRequest>,
    #[cfg(feature = "gpu")]
    previous_gpu_stats: Option<ExecutionStatistics>,
    fallback: Option<FailureProvenance>,
    last_backend: Option<ExecutionBackend>,
    page: Option<PreparedPage>,
    logits: Vec<f32>,
    groups: Vec<Vec<usize>>,
}

impl ParagraphRequest {
    /// Group original quadrilaterals using the original image. The two nesting
    /// levels are each in reading order and every original index appears once.
    pub fn group(
        &mut self,
        image: ImageView<'_>,
        quads: &[[[f32; 2]; 4]],
    ) -> ImageResult<&[Vec<usize>]> {
        self.run::<false>(image, quads)?;
        Ok(self.groups())
    }

    /// Group lines and measure the same synchronous execution path.
    #[cfg(feature = "profiling")]
    pub fn group_profiled(
        &mut self,
        image: ImageView<'_>,
        quads: &[[[f32; 2]; 4]],
    ) -> ImageResult<(&[Vec<usize>], ParagraphProfile)> {
        let profile = self.run::<true>(image, quads)?;
        Ok((self.groups(), profile))
    }

    /// Backend used by the latest invocation, plus cumulative GPU traffic.
    pub fn statistics(&self) -> ExecutionStatistics {
        #[cfg(not(feature = "gpu"))]
        let mut stats = ExecutionStatistics::default();
        #[cfg(feature = "gpu")]
        let mut stats = self
            .gpu
            .as_ref()
            .map(ParagraphGpuRequest::statistics)
            .or_else(|| self.previous_gpu_stats.clone())
            .unwrap_or_default();
        stats.execution_backend = self.last_backend;
        stats.fallback = self.fallback.clone();
        if let Some(ExecutionBackend::Cpu(backend)) = self.last_backend {
            stats.backend = format!("{backend:?}");
            stats.effective_precision = "F32".to_owned();
            stats.adapter = None;
            stats.packed_weight_bytes = self.plan.cpu.as_ref().map_or(0, |weights| weights.bytes());
            stats.activation_bytes = self.cpu.retained_bytes();
            stats.retained_activation_bytes = stats.activation_bytes;
        }
        stats
    }

    #[doc(hidden)]
    pub fn diagnostics(&self) -> ParagraphDiagnostics<'_> {
        ParagraphDiagnostics {
            nodes: self.page.as_ref().map_or(&[], |page| &page.node_inputs),
            edges: self.page.as_ref().map_or(&[], |page| &page.edge_inputs),
            pairs: self.page.as_ref().map_or(&[], |page| &page.pairs),
            logits: &self.logits,
        }
    }

    #[doc(hidden)]
    pub fn score_prepared(
        &mut self,
        nodes: &[f32],
        directed_edges: &[f32],
        pairs: &[[usize; 2]],
    ) -> ImageResult<&[f32]> {
        self.reset();
        let input = Input {
            nodes,
            edges: directed_edges,
            pairs,
        };
        input
            .validate()
            .map_err(ImageError::from)
            .map_err(stage_error)?;
        self.score(&input, self.plan.limits)
            .map_err(ImageError::from)
            .map_err(stage_error)?;
        Ok(&self.logits)
    }

    pub(crate) fn use_context(&mut self, cpu: Arc<CpuContext>, threads: usize) {
        self.context = cpu;
        self.threads = threads;
    }

    pub(crate) fn uses_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    pub(crate) fn reset(&mut self) {
        self.page = None;
        self.groups.clear();
        self.logits.clear();
        self.last_backend = None;
    }

    pub(crate) fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }

    pub(crate) fn group_bytes(&self) -> usize {
        self.groups.len() * size_of::<Vec<usize>>()
            + self
                .groups
                .iter()
                .map(|g| g.len() * size_of::<usize>())
                .sum::<usize>()
    }

    pub(crate) fn retained_device_bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        {
            self.gpu
                .as_ref()
                .map_or(0, ParagraphGpuRequest::retained_device_bytes)
        }
        #[cfg(not(feature = "gpu"))]
        {
            0
        }
    }

    pub(crate) fn retained_activation_bytes(&self) -> usize {
        self.cpu.retained_bytes()
            + self.retained_device_bytes()
            + self.logits.capacity() * size_of::<f32>()
    }

    pub(crate) fn retained_workspace_bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        let gpu = self
            .gpu
            .as_ref()
            .map_or(0, ParagraphGpuRequest::retained_host_bytes);
        #[cfg(not(feature = "gpu"))]
        let gpu = 0;
        gpu + self.page.as_ref().map_or(0, PreparedPage::retained_bytes)
            + self.groups.capacity() * size_of::<Vec<usize>>()
            + self
                .groups
                .iter()
                .map(|g| g.capacity() * size_of::<usize>())
                .sum::<usize>()
    }

    pub(crate) fn run<const PROFILE: bool>(
        &mut self,
        image: ImageView<'_>,
        quads: &[[[f32; 2]; 4]],
    ) -> ImageResult<ParagraphProfile> {
        self.reset();
        let timer = Timer::<PROFILE>::start();
        let page = features::prepare(image, quads, self.plan.limits).map_err(stage_error)?;
        let preprocess_ns = timer.elapsed();
        let input = Input {
            nodes: &page.node_inputs,
            edges: &page.edge_inputs,
            pairs: &page.pairs,
        };
        let limits = ExecutionLimits {
            max_activation_bytes: self
                .plan
                .limits
                .max_activation_bytes
                .checked_sub(page.retained_bytes())
                .ok_or_else(|| {
                    stage_error(ImageError::with_kind(
                        ErrorKind::ResourceLimit,
                        "paragraph features exceed execution limits",
                    ))
                })?,
        };
        let timer = Timer::<PROFILE>::start();
        self.score(&input, limits)
            .map_err(ImageError::from)
            .map_err(stage_error)?;
        let inference_ns = timer.elapsed();
        let timer = Timer::<PROFILE>::start();
        self.groups = decode::decode(&page, &self.logits).map_err(stage_error)?;
        let profile = ParagraphProfile {
            preprocess_ns,
            inference_ns,
            decode_ns: timer.elapsed(),
            candidate_count: page.pairs.len(),
        };
        self.page = Some(page);
        Ok(profile)
    }

    fn score(&mut self, input: &Input<'_>, limits: ExecutionLimits) -> Result<(), RuntimeError> {
        if input.pairs.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &mut self.gpu {
            match gpu.score(input, limits, &mut self.logits) {
                Ok(()) => {
                    self.last_backend = Some(ExecutionBackend::Gpu);
                    return check_logits(&self.logits);
                }
                Err(error) if self.plan.auto && error.allows_fallback() => {
                    self.fallback = Some(failure(&error));
                    self.previous_gpu_stats = Some(gpu.statistics());
                    self.gpu = None;
                }
                Err(error) => return Err(error),
            }
        }
        let (Some(weights), Some(backend)) = (&self.plan.cpu, self.plan.backend) else {
            return Err(RuntimeError::new("B-Scale scorer has no active backend"));
        };
        self.cpu.score(
            weights,
            input,
            limits,
            backend,
            &self.context,
            self.threads,
            &mut self.logits,
        )?;
        self.last_backend = Some(ExecutionBackend::Cpu(backend));
        check_logits(&self.logits)
    }
}

#[doc(hidden)]
pub struct ParagraphDiagnostics<'a> {
    pub nodes: &'a [f32],
    pub edges: &'a [f32],
    pub pairs: &'a [[usize; 2]],
    pub logits: &'a [f32],
}

fn stage_error(error: ImageError) -> ImageError {
    error.at_stage(OcrStage::Paragraph)
}

fn import_error(error: ImportError) -> ImageError {
    stage_error(ImageError::caused_by(
        error.kind(),
        "load epoch 15 B-Scale model",
        error,
    ))
}

fn failure(error: &RuntimeError) -> FailureProvenance {
    let mut result = FailureProvenance::from(error);
    result.stage = Some(OcrStage::Paragraph);
    result
}

fn check_logits(logits: &[f32]) -> Result<(), RuntimeError> {
    if logits.iter().any(|v| !v.is_finite()) {
        return Err(RuntimeError::new(
            "B-Scale scorer returned a nonfinite logit",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RunOptions, image::PixelFormat, ocr::OcrTrace};

    // Empty and singleton results must work without binding or executing
    // backend storage. The real checkpoint is exercised by the parity harness.
    fn no_scoring() -> ParagraphPlan {
        ParagraphPlan {
            cpu: None,
            backend: None,
            #[cfg(feature = "gpu")]
            gpu: None,
            #[cfg(feature = "gpu")]
            auto: false,
            fallback: None,
            limits: ExecutionLimits::default(),
        }
    }

    #[test]
    fn paragraph_request_recovers_after_nonfinite_quad() {
        let mut request = no_scoring().request().unwrap();
        let pixels = [0; 12];
        let image = ImageView::new(&pixels, 2, 2, 0, PixelFormat::Rgb24).unwrap();
        let valid = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];

        assert_eq!(request.group(image, &[valid]).unwrap(), [vec![0]]);
        let mut invalid = valid;
        invalid[2][0] = f32::INFINITY;
        let error = request.group(image, &[invalid]).unwrap_err();
        assert_eq!(error.stage(), Some(OcrStage::Paragraph));
        assert!(request.group(image, &[]).unwrap().is_empty());
        assert_eq!(request.statistics().execution_backend, None);
        assert_eq!(request.group(image, &[valid]).unwrap(), [vec![0]]);
    }

    #[test]
    fn paragraph_ocr_empty_trace_owned_and_json_agree() {
        let before = crate::ocr::tests::synthetic_ocr(CpuContext::serial());
        let pixels = vec![0; 96 * 96 * 3];
        let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
        let mut unconfigured = before.fork().unwrap();
        let plain = unconfigured.run(image, RunOptions::default()).unwrap();
        assert!(plain.is_empty());
        assert!(plain.paragraphs().is_none());
        assert!(
            serde_json::to_value(plain)
                .unwrap()
                .get("paragraphs")
                .is_none()
        );

        let mut configured = before
            .plan()
            .clone()
            .with_paragraph(no_scoring())
            .request()
            .unwrap();
        let mut traced = false;
        let grouped = configured
            .run_with_trace(image, false, |event| {
                if let OcrTrace::Result(result) = event {
                    assert_eq!(result.paragraphs(), Some(&[][..]));
                    traced = true;
                }
                Ok(())
            })
            .unwrap();
        assert!(traced);
        let owned = grouped.try_to_owned().unwrap();
        assert_eq!(owned.paragraphs, Some(Vec::new()));
        assert_eq!(
            serde_json::to_value(grouped).unwrap(),
            serde_json::to_value(&owned).unwrap()
        );
        assert_eq!(
            configured.latest_run().unwrap().stages[&OcrStage::Paragraph][0].execution_backend,
            None
        );
        assert_eq!(
            configured
                .fork()
                .unwrap()
                .run(image, RunOptions::default())
                .unwrap()
                .paragraphs(),
            Some(&[][..])
        );
    }
}
