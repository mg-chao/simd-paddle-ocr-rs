//! Explicit compilation ownership. Imported models never own execution caches.
use super::{
    CompileOptions, CompiledModel, ExecutionTarget, LogicalGraph, RuntimeResult,
    packing::WeightCache,
};
use crate::model::Model;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use std::sync::OnceLock;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[derive(Clone, Default)]
pub struct CompilerOptions {
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pub gpu: Option<Arc<super::gpu::GpuContext>>,
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pub gpu_options: super::gpu::GpuContextOptions,
}

pub(crate) struct ModelResources {
    pub graph: Arc<LogicalGraph>,
    pub packed: Arc<WeightCache>,
    cpu: Mutex<Vec<(super::CpuCompileOptions, bool, Arc<super::CpuCompiledModel>)>>,
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    gpu: Mutex<
        Vec<(
            super::gpu::GpuCompileOptions,
            bool,
            Arc<super::gpu::GpuCompiledModel>,
        )>,
    >,
}

impl ModelResources {
    pub(super) fn cpu(
        &self,
        options: CompileOptions,
    ) -> RuntimeResult<Arc<super::CpuCompiledModel>> {
        let mut cache = self
            .cpu
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((_, _, compiled)) = cache
            .iter()
            .find(|(cpu, fusion, _)| *cpu == options.cpu && *fusion == options.fusion)
        {
            return Ok(compiled.clone());
        }
        let compiled = Arc::new(super::CpuCompiledModel::from_graph_cached(
            self.graph.clone(),
            options,
            self.packed.clone(),
        )?);
        cache.push((options.cpu, options.fusion, compiled.clone()));
        Ok(compiled)
    }
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pub(super) fn gpu(
        &self,
        context: Arc<super::gpu::GpuContext>,
        options: CompileOptions,
    ) -> RuntimeResult<Arc<super::gpu::GpuCompiledModel>> {
        let mut cache = self
            .gpu
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((_, _, compiled)) = cache
            .iter()
            .find(|(gpu, fusion, _)| *gpu == options.gpu && *fusion == options.fusion)
        {
            return Ok(compiled.clone());
        }
        let compiled = Arc::new(super::gpu::GpuCompiledModel::from_graph(
            self.graph.clone(),
            context,
            options,
        )?);
        cache.push((options.gpu, options.fusion, compiled.clone()));
        Ok(compiled)
    }
}

/// Share this compiler across model sets to reuse graph validation and CPU packs.
/// Cached models live until the compiler is dropped; compiled artifacts retain
/// their resources independently. GPU initialization is lazy and attempted once.
pub struct Compiler {
    models: Mutex<HashMap<usize, Arc<ModelResources>>>,
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    options: CompilerOptions,
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    gpu: OnceLock<RuntimeResult<Arc<super::gpu::GpuContext>>>,
    #[cfg(all(test, feature = "gpu", not(target_os = "macos")))]
    gpu_initializations: std::sync::atomic::AtomicUsize,
    #[cfg(all(test, feature = "gpu", not(target_os = "macos")))]
    gpu_initializer:
        fn(super::gpu::GpuContextOptions) -> RuntimeResult<Arc<super::gpu::GpuContext>>,
}
impl Default for Compiler {
    fn default() -> Self {
        Self::new(CompilerOptions::default())
    }
}
impl Compiler {
    pub fn new(options: CompilerOptions) -> Self {
        #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
        let _ = options;
        Self {
            models: Mutex::new(HashMap::new()),
            #[cfg(all(feature = "gpu", not(target_os = "macos")))]
            options,
            #[cfg(all(feature = "gpu", not(target_os = "macos")))]
            gpu: OnceLock::new(),
            #[cfg(all(test, feature = "gpu", not(target_os = "macos")))]
            gpu_initializations: Default::default(),
            #[cfg(all(test, feature = "gpu", not(target_os = "macos")))]
            gpu_initializer: super::gpu::GpuContext::new_with_options,
        }
    }
    pub(crate) fn resources(&self, model: Arc<Model>) -> RuntimeResult<Arc<ModelResources>> {
        let key = Arc::as_ptr(&model) as usize;
        let mut models = self
            .models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(resources) = models.get(&key) {
            return Ok(resources.clone());
        }
        let resources = Arc::new(ModelResources {
            graph: Arc::new(LogicalGraph::compile(model)?),
            packed: Arc::default(),
            cpu: Mutex::new(Vec::new()),
            #[cfg(all(feature = "gpu", not(target_os = "macos")))]
            gpu: Mutex::new(Vec::new()),
        });
        // The graph retains the model, so its identity cannot be recycled while cached.
        models.insert(key, resources.clone());
        Ok(resources)
    }
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pub(crate) fn gpu(&self) -> RuntimeResult<Arc<super::gpu::GpuContext>> {
        self.gpu
            .get_or_init(|| {
                #[cfg(test)]
                self.gpu_initializations
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.options.gpu.clone().map(Ok).unwrap_or_else(|| {
                    #[cfg(test)]
                    let initialize = self.gpu_initializer;
                    #[cfg(not(test))]
                    let initialize = super::gpu::GpuContext::new_with_options;
                    initialize(self.options.gpu_options)
                })
            })
            .clone()
    }
    pub fn compile(
        &self,
        model: Arc<Model>,
        options: CompileOptions,
        target: ExecutionTarget,
    ) -> RuntimeResult<CompiledModel> {
        CompiledModel::compile_with(self, model, options, target)
    }
    /// CPU packing retained by this compiler, excluding imported coefficients.
    pub fn cached_packed_weight_bytes(&self) -> usize {
        self.models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .map(|r| r.packed.bytes())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn models() -> crate::OcrModelSet {
        let (det, rec) = crate::ocr::tests::synthetic_sources();
        crate::OcrModelSet::from_readers(&det[..], &rec[..], &b"A\n"[..], None).unwrap()
    }
    #[test]
    fn compiler_shares_lowered_plans_but_requests_allocate_independently() {
        let compiler = Arc::new(Compiler::default());
        let models = models();
        let plan = models
            .clone()
            .builder()
            .compiler(compiler.clone())
            .cpu_context(crate::cpu::CpuContext::serial())
            .build()
            .unwrap();
        let other = models
            .builder()
            .compiler(compiler)
            .cpu_context(crate::cpu::CpuContext::serial())
            .build()
            .unwrap();
        assert!(Arc::ptr_eq(
            plan.recognizer().compiled().as_cpu().unwrap(),
            other.recognizer().compiled().as_cpu().unwrap()
        ));
        let mut first = plan.request().unwrap();
        let second = plan.request().unwrap();
        assert_eq!(first.activation_bytes(), 0);
        assert_eq!(second.activation_bytes(), 0);
        let pixels = crate::ocr::tests::two_lines();
        let image =
            crate::image::ImageView::new(&pixels, 96, 96, 0, crate::image::PixelFormat::Bgr24)
                .unwrap();
        first.run(image, crate::RunOptions::default()).unwrap();
        assert!(first.activation_bytes() > 0);
        assert_eq!(second.activation_bytes(), 0);
    }
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    #[test]
    fn compiler_resolves_gpu_once_and_preserves_strict_auto_sources() {
        use crate::error::{ErrorKind, RuntimeError};
        use std::{error::Error, sync::atomic::Ordering};
        let compiler = Compiler {
            gpu_initializer: |_| {
                Err(RuntimeError::caused_by(
                    ErrorKind::Unsupported,
                    "injected initialization",
                    std::io::Error::other("original adapter error"),
                ))
            },
            ..Default::default()
        };
        let compiler = Arc::new(compiler);
        let models = models();
        let mut disabled = models.clone();
        disabled.classification = Some(models.detection.clone());
        disabled
            .builder()
            .compiler(compiler.clone())
            .execution_target(ExecutionTarget::Cpu)
            .classifier_target(ExecutionTarget::Gpu)
            .direction_classification(false)
            .build()
            .unwrap();
        assert_eq!(compiler.gpu_initializations.load(Ordering::Relaxed), 0);
        let auto = models
            .clone()
            .builder()
            .compiler(compiler.clone())
            .execution_target(ExecutionTarget::Auto)
            .build()
            .unwrap();
        let error = auto.recognizer().compiled().fallback_error().unwrap();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.source().unwrap().to_string(),
            "original adapter error"
        );
        let error = models
            .builder()
            .compiler(compiler.clone())
            .execution_target(ExecutionTarget::Gpu)
            .build()
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("original adapter error"));
        assert_eq!(compiler.gpu_initializations.load(Ordering::Relaxed), 1);
    }
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    #[test]
    fn compiler_does_not_fallback_on_invalid_gpu_configuration() {
        use crate::runtime::gpu::GpuContextOptions;
        use std::sync::atomic::Ordering;
        let compiler = Compiler::new(CompilerOptions {
            gpu_options: GpuContextOptions {
                max_in_flight: 0,
                ..Default::default()
            },
            ..Default::default()
        });
        let models = models();
        for target in [ExecutionTarget::Auto, ExecutionTarget::Gpu] {
            let error = compiler
                .compile(models.detection.clone(), CompileOptions::default(), target)
                .err()
                .unwrap();
            assert_eq!(error.kind(), crate::error::ErrorKind::InvalidInput);
        }
        assert_eq!(compiler.gpu_initializations.load(Ordering::Relaxed), 1);
    }
}
