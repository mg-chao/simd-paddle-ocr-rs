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
    resources: Arc<super::compiler::ModelResources>,
    options: CompileOptions,
    compiled: OnceLock<RuntimeResult<Arc<CpuCompiledModel>>>,
}
impl CpuFallback {
    fn get(&self) -> RuntimeResult<Arc<CpuCompiledModel>> {
        self.compiled
            .get_or_init(|| self.resources.cpu(self.options))
            .clone()
    }
}
/// Immutable backend plan. Auto prepares full CPU resources only on fallback.
#[derive(Clone)]
pub struct CompiledModel {
    pub(crate) cpu_options: super::CpuCompileOptions,
    pub(crate) packed: Arc<super::packing::WeightCache>,
    plan: ModelPlan,
    fallback: Option<Arc<CpuFallback>>,
    reason: Option<RuntimeError>,
}
impl From<Arc<CpuCompiledModel>> for CompiledModel {
    fn from(value: Arc<CpuCompiledModel>) -> Self {
        Self {
            cpu_options: value.options().cpu,
            packed: value.packed_cache.clone(),
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
            cpu_options: Default::default(),
            packed: Arc::default(),
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
        let compiler = super::Compiler::new(super::CompilerOptions {
            #[cfg(feature = "gpu")]
            gpu: execution.gpu,
            #[cfg(feature = "gpu")]
            gpu_options: execution.gpu_options,
        });
        compiler.compile(model, options, execution.target)
    }
    pub(crate) fn compile_with(
        compiler: &super::Compiler,
        model: Arc<Model>,
        options: CompileOptions,
        target: ExecutionTarget,
    ) -> RuntimeResult<Self> {
        let resources = compiler.resources(model)?;
        if target == ExecutionTarget::Cpu {
            return resources.cpu(options).map(Into::into);
        }
        #[cfg(feature = "gpu")]
        let gpu = compiler
            .gpu()
            .and_then(|context| resources.gpu(context, options));
        #[cfg(not(feature = "gpu"))]
        let gpu: RuntimeResult<()> = Err(RuntimeError::with_kind(
            ErrorKind::Unsupported,
            "GPU execution requires the gpu Cargo feature",
        ));
        match gpu {
            #[cfg(feature = "gpu")]
            Ok(gpu) => {
                let mut result: Self = gpu.into();
                result.packed = resources.packed.clone();
                result.cpu_options = options.cpu;
                if target == ExecutionTarget::Auto {
                    result.fallback = Some(Arc::new(CpuFallback {
                        resources: resources.clone(),
                        options,
                        compiled: OnceLock::new(),
                    }));
                }
                Ok(result)
            }
            #[cfg(not(feature = "gpu"))]
            Ok(()) => unreachable!(),
            Err(error) if target == ExecutionTarget::Auto && fallback_allowed(&error) => {
                let mut result: Self = resources.cpu(options)?.into();
                result.reason = Some(error);
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
    /// Bind a stage contract without constructing an execution request.
    pub(crate) fn validate_shape(
        &self,
        shape: &[usize],
        limits: ExecutionLimits,
        prefix: Option<usize>,
    ) -> RuntimeResult<Shape> {
        Session::validate(self, shape, limits, prefix.map_or(Mode::Full, Mode::Prefix))
            .map(|bound| bound.output)
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
    pub fn fallback_error(&self) -> Option<&RuntimeError> {
        self.reason.as_ref()
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
        model.reason = Some(error);
        Ok(model)
    }
}
fn fallback_allowed(error: &RuntimeError) -> bool {
    error.allows_fallback()
}

#[derive(Clone)]
pub(crate) struct RequestSettings {
    pub cpu: Arc<CpuContext>,
    pub threads: usize,
}
impl Default for RequestSettings {
    fn default() -> Self {
        Self {
            cpu: CpuContext::serial(),
            threads: 1,
        }
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Full,
    Prefix(usize),
    Diagnostic,
}
struct PendingSession {
    input: Shape,
    output: Shape,
    limits: ExecutionLimits,
}
enum SessionKind {
    Pending(PendingSession),
    Cpu(Box<CpuSession>),
    #[cfg(feature = "gpu")]
    Gpu(Box<super::gpu::GpuSession>),
}
/// Output borrows this request; GPU waits do not occupy the CPU executor.
pub struct Session {
    inner: SessionKind,
    compiled: CompiledModel,
    mode: Mode,
    reason: Option<RuntimeError>,
    cpu: Arc<CpuContext>,
    threads: usize,
}
impl Session {
    /// Validate the stage contract without allocating execution storage.
    pub(crate) fn deferred(
        compiled: impl Into<CompiledModel>,
        shape: &[usize],
        limits: ExecutionLimits,
        prefix: Option<usize>,
    ) -> RuntimeResult<Self> {
        let compiled = compiled.into();
        let mode = prefix.map_or(Mode::Full, Mode::Prefix);
        let pending = Self::validate(&compiled, shape, limits, mode)?;
        Ok(Self {
            reason: compiled.reason.clone(),
            compiled,
            inner: SessionKind::Pending(pending),
            mode,
            cpu: CpuContext::serial(),
            threads: 1,
        })
    }
    fn validate(
        compiled: &CompiledModel,
        shape: &[usize],
        limits: ExecutionLimits,
        mode: Mode,
    ) -> RuntimeResult<PendingSession> {
        if limits.max_activation_bytes == 0 {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "max_activation_bytes must be positive",
            ));
        }
        let graph = match &compiled.plan {
            ModelPlan::Cpu(c) => &c.program,
            #[cfg(feature = "gpu")]
            ModelPlan::Gpu(c) => &c.program,
        };
        let (count, output) = match mode {
            Mode::Prefix(last) => (
                last + 1,
                graph
                    .instructions
                    .get(last)
                    .ok_or_else(|| RuntimeError::new("prefix node out of range"))?
                    .output,
            ),
            _ => (graph.instructions.len(), graph.output),
        };
        let mut plan = super::ShapePlan::empty(graph.model.graph().tensors.len(), count);
        super::bind_into(
            graph,
            shape,
            ExecutionLimits {
                max_activation_bytes: usize::MAX,
            },
            count,
            output,
            &mut plan,
            &mut super::BindScratch::new(graph),
        )?;
        Ok(PendingSession {
            input: plan.input_shape,
            output: plan.shapes[output.0],
            limits,
        })
    }
    fn prepare(&mut self) -> RuntimeResult<()> {
        if let SessionKind::Pending(p) = &self.inner {
            let mut ready = Self::create(
                self.compiled.clone(),
                p.input.dimensions(),
                p.limits,
                self.mode,
            )?;
            ready.set_execution_context(self.cpu.clone(), self.threads)?;
            *self = ready;
        }
        Ok(())
    }
    pub(crate) fn is_gpu(&self) -> bool {
        matches!(self.execution_backend(), ExecutionBackend::Gpu)
    }

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
            ModelPlan::Cpu(c) => Self::cpu_session(c.clone(), shape, limits, mode)
                .map(Box::new)
                .map(SessionKind::Cpu),
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
            .map(Box::new)
            .map(SessionKind::Gpu),
        };
        let inner = match candidate {
            Err(error) if fallback_allowed(&error) && compiled.fallback.is_some() => {
                let cpu = compiled.fallback.as_ref().unwrap().get()?;
                reason = Some(error);
                SessionKind::Cpu(Box::new(Self::cpu_session(cpu, shape, limits, mode)?))
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
        self.inner = SessionKind::Cpu(Box::new(cpu));
        self.reason = Some(error);
        Ok(())
    }
    pub fn execution_backend(&self) -> ExecutionBackend {
        match &self.inner {
            SessionKind::Pending(_) => self.compiled.execution_backend(),
            SessionKind::Cpu(s) => ExecutionBackend::Cpu(s.compiled.backend()),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(_) => ExecutionBackend::Gpu,
        }
    }
    pub fn fallback_error(&self) -> Option<&RuntimeError> {
        self.reason.as_ref()
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
            SessionKind::Pending(_) => {}
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
        if matches!(self.inner, SessionKind::Pending(_)) {
            self.inner =
                SessionKind::Pending(Self::validate(&self.compiled, shape, limits, self.mode)?);
            return Ok(());
        }
        let result = match &mut self.inner {
            SessionKind::Pending(_) => unreachable!(),
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
        self.prepare()?;
        // Public inputs are always logical NCHW, including CPU fallback.
        #[allow(irrefutable_let_patterns)]
        if let SessionKind::Cpu(s) = &mut self.inner {
            s.run_with_input(|input| fill(input).map_err(RuntimeError::callback))?;
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
        self.prepare()?;
        #[allow(unused_mut)]
        let mut filled = false;
        let result = match &mut self.inner {
            SessionKind::Pending(_) => unreachable!(),
            SessionKind::Cpu(s) => s
                .run_with_layout_input(|input, layout| {
                    fill(input, layout).map_err(RuntimeError::callback)
                })
                .map(|_| ()),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s
                .run_with_input_profiled::<PROFILE>(|input| {
                    fill(input, false).map_err(RuntimeError::callback)?;
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
            SessionKind::Pending(_) => Err(RuntimeError::new("session has not executed")),
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
        mut observer: O,
    ) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: FnOnce(&mut [f32]) -> RuntimeResult<()>,
        O: for<'a> FnMut(NodeTrace<'a>) -> RuntimeResult<()>,
    {
        self.prepare()?;
        // Callbacks may have observable side effects: never replay on fallback.
        match &mut self.inner {
            SessionKind::Pending(_) => unreachable!(),
            SessionKind::Cpu(s) => s.run_with_input_and_trace(
                |input| fill(input).map_err(RuntimeError::callback),
                |event| observer(event).map_err(RuntimeError::callback),
            ),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.run_with_input_and_trace(
                |input| fill(input).map_err(RuntimeError::callback),
                |event| observer(event).map_err(RuntimeError::callback),
            ),
        }
    }
    pub fn retained_device_bytes(&self) -> usize {
        match &self.inner {
            SessionKind::Pending(_) | SessionKind::Cpu(_) => 0,
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.retained_device_bytes(),
        }
    }
    pub(crate) fn counter_snapshot(&self) -> super::StatisticsCounters {
        match &self.inner {
            SessionKind::Pending(_) => super::StatisticsCounters::default(),
            SessionKind::Cpu(s) => s.counter_snapshot(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.counter_snapshot(),
        }
    }
    pub fn statistics(&self) -> ExecutionStatistics {
        let mut stats = match &self.inner {
            SessionKind::Pending(_) => ExecutionStatistics::default(),
            SessionKind::Cpu(s) => s.statistics(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.statistics(),
        };
        stats.execution_backend =
            (!matches!(self.inner, SessionKind::Pending(_))).then(|| self.execution_backend());
        stats.fallback = self
            .reason
            .as_ref()
            .map(crate::error::FailureProvenance::from);
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
            profile.gpu_encoding_ns += t.encoding_ns;
            profile.gpu_submission_ns += t.submission_ns;
            profile.gpu_completion_wait_ns += t.completion_wait_ns;
            profile.gpu_readback_ns += t.readback_ns;
            if let Some(ns) = s.last_gpu_profile().and_then(|p| p.compute_ns()) {
                profile.gpu_compute_ns = Some(profile.gpu_compute_ns.unwrap_or(0) + ns);
            }
        }
    }
}
impl Session {
    pub fn input_shape(&self) -> Shape {
        match &self.inner {
            SessionKind::Pending(p) => p.input,
            SessionKind::Cpu(s) => s.input_shape(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.input_shape(),
        }
    }
}
impl Session {
    pub fn output_shape(&self) -> Shape {
        match &self.inner {
            SessionKind::Pending(p) => p.output,
            SessionKind::Cpu(s) => s.output_shape(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.output_shape(),
        }
    }
}
impl Session {
    pub fn execution_limits(&self) -> ExecutionLimits {
        match &self.inner {
            SessionKind::Pending(p) => p.limits,
            SessionKind::Cpu(s) => s.execution_limits(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.execution_limits(),
        }
    }
}
impl Session {
    pub fn activation_bytes(&self) -> usize {
        match &self.inner {
            SessionKind::Pending(_) => 0,
            SessionKind::Cpu(s) => s.activation_bytes(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.activation_bytes(),
        }
    }
}
impl Session {
    pub fn retained_activation_bytes(&self) -> usize {
        match &self.inner {
            SessionKind::Pending(_) => 0,
            SessionKind::Cpu(s) => s.retained_activation_bytes(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.retained_activation_bytes(),
        }
    }
}
impl Session {
    pub fn retained_scratch_bytes(&self) -> usize {
        match &self.inner {
            SessionKind::Pending(_) => 0,
            SessionKind::Cpu(s) => s.retained_scratch_bytes(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => s.retained_scratch_bytes(),
        }
    }
}
impl Session {
    pub fn retained_planning_bytes(&self) -> usize {
        match &self.inner {
            SessionKind::Pending(_) => 0,
            SessionKind::Cpu(s) => size_of::<CpuSession>() + s.retained_planning_bytes(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(s) => {
                size_of::<super::gpu::GpuSession>() + s.retained_planning_bytes()
            }
        }
    }
}

#[cfg(all(test, not(feature = "gpu")))]
mod tests {
    use super::*;

    #[test]
    fn missing_gpu_feature_is_unsupported_and_auto_executes_on_cpu() {
        let (source, _) = crate::ocr::tests::synthetic_sources();
        let model = Arc::new(Model::load_bytes(&source).unwrap());
        let compiler = super::super::Compiler::default();
        let options = CompileOptions::default();
        let error = compiler
            .compile(model.clone(), options, ExecutionTarget::Gpu)
            .err()
            .expect("missing feature must reject explicit GPU execution");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.to_string(),
            "GPU execution requires the gpu Cargo feature"
        );

        let auto = compiler
            .compile(model.clone(), options, ExecutionTarget::Auto)
            .unwrap();
        assert!(matches!(auto.execution_backend(), ExecutionBackend::Cpu(_)));
        assert_eq!(
            auto.fallback_error().unwrap().kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(
            auto.fallback_error().unwrap().to_string(),
            error.to_string()
        );

        let cpu = compiler
            .compile(model, options, ExecutionTarget::Cpu)
            .unwrap();
        assert!(cpu.fallback_error().is_none());
        let shape = [1, 3, 32, 32];
        let input = vec![0.125; shape.iter().product()];
        let mut auto = Session::new_serial(auto, &shape, ExecutionLimits::default()).unwrap();
        let mut cpu = Session::new_serial(cpu, &shape, ExecutionLimits::default()).unwrap();
        assert_eq!(
            auto.run(&input).unwrap().data(),
            cpu.run(&input).unwrap().data()
        );
        assert_eq!(
            auto.fallback_error().unwrap().to_string(),
            error.to_string()
        );
        assert_eq!(auto.statistics().production_submissions, 0);
    }
}
