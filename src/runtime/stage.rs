//! Shared model ownership and exclusive synchronous execution across backends.
use super::{
    Backend, CompileOptions, CpuCompiledModel, CpuContext, CpuSession, ExecutionLimits,
    ExecutionStatistics, NodeTrace, RuntimeError, RuntimeResult,
};
use crate::{
    error::ErrorKind,
    model::Model,
    tensor::{Shape, TensorView},
};
use std::sync::{Arc, OnceLock};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum ExecutionTarget {
    #[default]
    Cpu,
    Auto,
    Gpu,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ExecutionBackend {
    Cpu(Backend),
    Gpu,
}
#[derive(Clone, Default)]
pub struct ExecutionConfig {
    pub target: ExecutionTarget,
    #[cfg(feature = "gpu")]
    pub gpu: Option<Arc<super::gpu::GpuContext>>,
    #[cfg(feature = "gpu")]
    pub gpu_options: super::gpu::GpuContextOptions,
}
#[derive(Clone)]
enum ModelPlan {
    Cpu(Arc<CpuCompiledModel>),
    #[cfg(feature = "gpu")]
    Gpu(Arc<super::gpu::GpuCompiledModel>),
}
struct CpuFallback {
    graph: Arc<super::LogicalGraph>,
    options: CompileOptions,
    compiled: OnceLock<RuntimeResult<Arc<CpuCompiledModel>>>,
}
impl CpuFallback {
    fn get(&self) -> RuntimeResult<Arc<CpuCompiledModel>> {
        self.compiled
            .get_or_init(|| {
                CpuCompiledModel::from_graph(self.graph.clone(), self.options).map(Arc::new)
            })
            .clone()
    }
}
/// Immutable backend plan. Auto prepares full CPU resources only on fallback.
#[derive(Clone)]
pub struct CompiledModel {
    plan: ModelPlan,
    fallback: Option<Arc<CpuFallback>>,
    reason: Option<String>,
}
impl From<Arc<CpuCompiledModel>> for CompiledModel {
    fn from(value: Arc<CpuCompiledModel>) -> Self {
        Self {
            plan: ModelPlan::Cpu(value),
            fallback: None,
            reason: None,
        }
    }
}
#[cfg(feature = "gpu")]
impl From<Arc<super::gpu::GpuCompiledModel>> for CompiledModel {
    fn from(value: Arc<super::gpu::GpuCompiledModel>) -> Self {
        Self {
            plan: ModelPlan::Gpu(value),
            fallback: None,
            reason: None,
        }
    }
}
impl CompiledModel {
    pub fn compile(
        model: Arc<Model>,
        options: CompileOptions,
        execution: ExecutionConfig,
    ) -> RuntimeResult<Self> {
        let graph = Arc::new(super::LogicalGraph::compile(model)?);
        if execution.target == ExecutionTarget::Cpu {
            return CpuCompiledModel::from_graph(graph, options)
                .map(Arc::new)
                .map(Into::into);
        }
        #[cfg(feature = "gpu")]
        let gpu = execution
            .gpu
            .map(Ok)
            .unwrap_or_else(|| {
                if execution.gpu_options == super::gpu::GpuContextOptions::default() {
                    super::gpu::GpuContext::shared_default()
                } else {
                    super::gpu::GpuContext::new_with_options(execution.gpu_options)
                }
            })
            .and_then(|context| {
                super::gpu::GpuCompiledModel::from_graph(graph.clone(), context, options.fusion)
            })
            .map(Arc::new);
        #[cfg(not(feature = "gpu"))]
        let gpu: RuntimeResult<()> = Err(RuntimeError::with_kind(
            ErrorKind::Unsupported,
            "GPU execution requires the gpu Cargo feature",
        ));
        match gpu {
            #[cfg(feature = "gpu")]
            Ok(gpu) => {
                let mut result: Self = gpu.into();
                if execution.target == ExecutionTarget::Auto {
                    result.fallback = Some(Arc::new(CpuFallback {
                        graph,
                        options,
                        compiled: OnceLock::new(),
                    }));
                }
                Ok(result)
            }
            #[cfg(not(feature = "gpu"))]
            Ok(()) => unreachable!(),
            Err(error) if execution.target == ExecutionTarget::Auto && fallback_allowed(&error) => {
                let mut result: Self =
                    Arc::new(CpuCompiledModel::from_graph(graph, options)?).into();
                result.reason = Some(error.to_string());
                Ok(result)
            }
            Err(error) => Err(error),
        }
    }
    pub(crate) fn model_arc(&self) -> Arc<Model> {
        match &self.plan {
            ModelPlan::Cpu(c) => c.program.model.clone(),
            #[cfg(feature = "gpu")]
            ModelPlan::Gpu(c) => c.model_arc(),
        }
    }
    pub fn model(&self) -> &Model {
        match &self.plan {
            ModelPlan::Cpu(c) => c.model(),
            #[cfg(feature = "gpu")]
            ModelPlan::Gpu(c) => c.model(),
        }
    }
    pub fn execution_backend(&self) -> ExecutionBackend {
        match &self.plan {
            ModelPlan::Cpu(c) => ExecutionBackend::Cpu(c.backend()),
            #[cfg(feature = "gpu")]
            ModelPlan::Gpu(_) => ExecutionBackend::Gpu,
        }
    }
    pub fn as_cpu(&self) -> Option<&Arc<CpuCompiledModel>> {
        match &self.plan {
            ModelPlan::Cpu(c) => Some(c),
            #[cfg(feature = "gpu")]
            ModelPlan::Gpu(_) => None,
        }
    }
    #[cfg(feature = "gpu")]
    pub fn as_gpu(&self) -> Option<&Arc<super::gpu::GpuCompiledModel>> {
        match &self.plan {
            ModelPlan::Cpu(_) => None,
            ModelPlan::Gpu(c) => Some(c),
        }
    }
    pub fn is_gpu(&self) -> bool {
        self.as_cpu().is_none()
    }
    pub fn fallback_reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }
    #[cfg(feature = "gpu")]
    pub(crate) fn resident_fallback(&self, error: RuntimeError) -> RuntimeResult<Self> {
        if !fallback_allowed(&error) {
            return Err(error);
        }
        let Some(fallback) = &self.fallback else {
            return Err(error);
        };
        let mut model: Self = fallback.get()?.into();
        model.reason = Some(error.to_string());
        Ok(model)
    }
}
fn fallback_allowed(error: &RuntimeError) -> bool {
    matches!(error.kind(), ErrorKind::Unsupported | ErrorKind::DeviceLost)
}

#[derive(Clone, Copy)]
enum Mode {
    Full,
    Prefix(usize),
    Diagnostic,
}
enum SessionKind {
    Cpu(CpuSession),
    #[cfg(feature = "gpu")]
    Gpu(super::gpu::GpuSession),
}
/// Output borrows this request; GPU waits do not occupy the CPU executor.
pub struct Session {
    inner: SessionKind,
    compiled: CompiledModel,
    mode: Mode,
    reason: Option<String>,
    cpu: Arc<CpuContext>,
    threads: usize,
}
impl Session {
    pub fn new_serial(
        compiled: impl Into<CompiledModel>,
        shape: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        Self::create(compiled.into(), shape, limits, Mode::Full)
    }
    pub fn new_prefix(
        compiled: impl Into<CompiledModel>,
        shape: &[usize],
        limits: ExecutionLimits,
        last: usize,
    ) -> RuntimeResult<Self> {
        Self::create(compiled.into(), shape, limits, Mode::Prefix(last))
    }
    pub fn new_diagnostic(
        compiled: impl Into<CompiledModel>,
        shape: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        Self::create(compiled.into(), shape, limits, Mode::Diagnostic)
    }
    fn cpu_session(
        compiled: Arc<CpuCompiledModel>,
        shape: &[usize],
        limits: ExecutionLimits,
        mode: Mode,
    ) -> RuntimeResult<CpuSession> {
        match mode {
            Mode::Full => CpuSession::new_serial(compiled, shape, limits),
            Mode::Prefix(last) => CpuSession::new_prefix(compiled, shape, limits, last),
            Mode::Diagnostic => CpuSession::new_diagnostic(compiled, shape, limits),
        }
    }
    fn create(
        compiled: CompiledModel,
        shape: &[usize],
        limits: ExecutionLimits,
        mode: Mode,
    ) -> RuntimeResult<Self> {
        let mut reason = compiled.reason.clone();
        let candidate = match &compiled.plan {
            ModelPlan::Cpu(c) => {
                Self::cpu_session(c.clone(), shape, limits, mode).map(SessionKind::Cpu)
            }
            #[cfg(feature = "gpu")]
            ModelPlan::Gpu(c) => match mode {
                Mode::Full => super::gpu::GpuSession::new(c.clone(), shape, limits),
                Mode::Prefix(last) => {
                    super::gpu::GpuSession::new_prefix(c.clone(), shape, limits, last)
                }
                Mode::Diagnostic => {
                    super::gpu::GpuSession::new_diagnostic(c.clone(), shape, limits)
                }
            }
            .map(SessionKind::Gpu),
        };
        let inner = match candidate {
            Err(error) if fallback_allowed(&error) && compiled.fallback.is_some() => {
                let cpu = compiled.fallback.as_ref().unwrap().get()?;
                reason = Some(error.to_string());
                SessionKind::Cpu(Self::cpu_session(cpu, shape, limits, mode)?)
            }
            other => other?,
        };
        Ok(Self {
            inner,
            compiled,
            mode,
            reason,
            cpu: CpuContext::serial(),
            threads: 1,
        })
    }
    fn fallback(
        &mut self,
        error: RuntimeError,
        shape: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<()> {
        if !fallback_allowed(&error) {
            return Err(error);
        }
        let Some(fallback) = &self.compiled.fallback else {
            return Err(error);
        };
        if matches!(self.inner, SessionKind::Cpu(_)) {
            return Err(error);
        }
        let mut cpu = Self::cpu_session(fallback.get()?, shape, limits, self.mode)?;
        cpu.set_execution_context(self.cpu.clone(), self.threads)?;
        self.inner = SessionKind::Cpu(cpu);
        self.reason = Some(error.to_string());
        Ok(())
    }
    pub fn execution_backend(&self) -> ExecutionBackend {
        match &self.inner {
            SessionKind::Cpu(s) => ExecutionBackend::Cpu(s.compiled.backend()),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(_) => ExecutionBackend::Gpu,
        }
    }
    pub fn fallback_reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }
    pub fn cpu_context(&self) -> &Arc<CpuContext> {
        &self.cpu
    }
    pub fn intra_op_threads(&self) -> usize {
        self.threads
    }
    pub fn set_execution_context(
        &mut self,
        cpu: Arc<CpuContext>,
        threads: usize,
    ) -> RuntimeResult<()> {
        if threads == 0 || threads > cpu.capacity() {
            return Err(RuntimeError::new(
                "intra-op threads must be within CPU context capacity",
            ));
        }
        match &mut self.inner {
            SessionKind::Cpu(s) => s.set_execution_context(cpu.clone(), threads)?,
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(_) => {}
        }
        self.cpu = cpu;
        self.threads = threads.min(16);
        Ok(())
    }
    pub fn reshape(&mut self, shape: &[usize]) -> RuntimeResult<()> {
        self.reshape_with_limits(shape, self.execution_limits())
    }
    pub fn reshape_with_limits(
        &mut self,
        shape: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<()> {
        let result = match &mut self.inner {
            SessionKind::Cpu(s) => s.reshape_with_limits(shape, limits),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.reshape_with_limits(shape, limits),
        };
        result.or_else(|error| self.fallback(error, shape, limits))
    }
    pub fn run(&mut self, input: &[f32]) -> RuntimeResult<TensorView<'_, f32>> {
        self.run_with_input(|destination| {
            if input.len() != destination.len() {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "input length mismatch",
                ));
            }
            destination.copy_from_slice(input);
            Ok(())
        })
    }
    pub fn run_with_input(
        &mut self,
        fill: impl FnOnce(&mut [f32]) -> RuntimeResult<()>,
    ) -> RuntimeResult<TensorView<'_, f32>> {
        // Public inputs are always logical NCHW, including CPU fallback.
        #[allow(irrefutable_let_patterns)]
        if let SessionKind::Cpu(s) = &mut self.inner {
            s.run_with_input(fill)?;
            return self.output();
        }
        self.run_with_layout_input(|input, _| fill(input))
    }
    pub(crate) fn run_with_layout_input(
        &mut self,
        fill: impl FnOnce(&mut [f32], bool) -> RuntimeResult<()>,
    ) -> RuntimeResult<TensorView<'_, f32>> {
        self.run_with_layout_input_profiled::<false, _>(fill)
    }
    pub(crate) fn run_with_layout_input_profiled<const PROFILE: bool, F>(
        &mut self,
        fill: F,
    ) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: FnOnce(&mut [f32], bool) -> RuntimeResult<()>,
    {
        #[allow(unused_mut)]
        let mut filled = false;
        let result = match &mut self.inner {
            SessionKind::Cpu(s) => s.run_with_layout_input(fill).map(|_| ()),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s
                .run_with_input_profiled::<PROFILE>(|input| {
                    fill(input, false)?;
                    filled = true;
                    Ok(())
                })
                .map(|_| ()),
        };
        if let Err(error) = result {
            if !filled || !fallback_allowed(&error) {
                return Err(error);
            }
            #[cfg(feature = "gpu")]
            {
                let shape = self.input_shape().dimensions().to_vec();
                let input = match &self.inner {
                    SessionKind::Gpu(s) => s.input_data().to_vec(),
                    _ => return Err(error),
                };
                self.fallback(error, &shape, self.execution_limits())?;
                if let SessionKind::Cpu(s) = &mut self.inner {
                    s.run(&input)?;
                }
            }
            #[cfg(not(feature = "gpu"))]
            return Err(error);
        }
        self.output()
    }
    fn output(&self) -> RuntimeResult<TensorView<'_, f32>> {
        match &self.inner {
            SessionKind::Cpu(s) => s.output_view(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.output_view(),
        }
    }
    pub fn run_with_trace<F>(
        &mut self,
        input: &[f32],
        observer: F,
    ) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: for<'a> FnMut(NodeTrace<'a>) -> RuntimeResult<()>,
    {
        self.run_with_input_and_trace(
            |destination| {
                if input.len() != destination.len() {
                    return Err(RuntimeError::with_kind(
                        ErrorKind::InvalidInput,
                        "input length mismatch",
                    ));
                }
                destination.copy_from_slice(input);
                Ok(())
            },
            observer,
        )
    }
    pub fn run_with_input_and_trace<F, O>(
        &mut self,
        fill: F,
        observer: O,
    ) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: FnOnce(&mut [f32]) -> RuntimeResult<()>,
        O: for<'a> FnMut(NodeTrace<'a>) -> RuntimeResult<()>,
    {
        // Callbacks may have observable side effects: never replay on fallback.
        match &mut self.inner {
            SessionKind::Cpu(s) => s.run_with_input_and_trace(fill, observer),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.run_with_input_and_trace(fill, observer),
        }
    }
    pub fn retained_device_bytes(&self) -> usize {
        match &self.inner {
            SessionKind::Cpu(_) => 0,
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.retained_device_bytes(),
        }
    }
    pub fn statistics(&self) -> ExecutionStatistics {
        let mut stats = match &self.inner {
            SessionKind::Cpu(s) => s.statistics(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.statistics(),
        };
        stats.execution_backend = Some(self.execution_backend());
        stats.fallback_reason = self.reason.clone();
        stats
    }
    pub(crate) fn accumulate_gpu_profile<const PROFILE: bool>(
        &self,
        profile: &mut crate::profile::StageProfile,
    ) {
        let _ = profile;
        #[cfg(feature = "gpu")]
        if PROFILE && let SessionKind::Gpu(s) = &self.inner {
            let t = s.gpu_execution_timings();
            profile.gpu_admission_wait_ns += t.admission_wait_ns;
            profile.gpu_submission_ns += t.submission_ns;
            profile.gpu_completion_wait_ns += t.completion_wait_ns;
            profile.gpu_readback_ns += t.readback_ns;
            if let Some(ns) = s.last_gpu_profile().and_then(|p| p.compute_ns()) {
                profile.gpu_compute_ns = Some(profile.gpu_compute_ns.unwrap_or(0) + ns);
            }
        }
    }
}
macro_rules! delegate {
    ($($name:ident -> $ty:ty),* $(,)?) => { impl Session { $(pub fn $name(&self)->$ty { match &self.inner { SessionKind::Cpu(s)=>s.$name(), #[cfg(feature="gpu")] SessionKind::Gpu(s)=>s.$name() } })* } }
}
delegate!(input_shape->Shape,output_shape->Shape,execution_limits->ExecutionLimits,activation_bytes->usize,retained_activation_bytes->usize,retained_scratch_bytes->usize,retained_planning_bytes->usize);
