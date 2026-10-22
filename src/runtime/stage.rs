//! Backend-neutral ownership used by the OCR stages.
use super::{
    Backend, CompiledModel, CpuContext, ExecutionLimits, ExecutionStatistics, NodeTrace,
    RuntimeError, RuntimeResult, Session,
};
use crate::{
    model::Model,
    tensor::{Shape, TensorView},
};
use std::sync::Arc;

/// Explicit inference placement. GPU initialization and compilation never fall
/// back to CPU; CPU resources remain available for image and text processing.
#[derive(Clone, Default)]
pub enum ExecutionTarget {
    #[default]
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu(Arc<super::gpu::GpuContext>),
}

/// The inference backend, separate from a CPU SIMD kernel policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ExecutionBackend {
    Cpu(Backend),
    #[cfg(feature = "gpu")]
    Gpu,
}

/// Shareable compiled inference resources for an OCR stage.
#[derive(Clone)]
pub enum CompiledStage {
    Cpu(Arc<CompiledModel>),
    #[cfg(feature = "gpu")]
    Gpu(Arc<super::gpu::GpuCompiledModel>),
}
impl From<Arc<CompiledModel>> for CompiledStage {
    fn from(compiled: Arc<CompiledModel>) -> Self {
        Self::Cpu(compiled)
    }
}
#[cfg(feature = "gpu")]
impl From<Arc<super::gpu::GpuCompiledModel>> for CompiledStage {
    fn from(compiled: Arc<super::gpu::GpuCompiledModel>) -> Self {
        Self::Gpu(compiled)
    }
}
impl CompiledStage {
    pub fn model(&self) -> &Model {
        match self {
            Self::Cpu(compiled) => compiled.model(),
            #[cfg(feature = "gpu")]
            Self::Gpu(compiled) => compiled.model(),
        }
    }
    pub fn execution_backend(&self) -> ExecutionBackend {
        match self {
            Self::Cpu(compiled) => ExecutionBackend::Cpu(compiled.backend()),
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => ExecutionBackend::Gpu,
        }
    }
    pub fn as_cpu(&self) -> Option<&Arc<CompiledModel>> {
        match self {
            Self::Cpu(compiled) => Some(compiled),
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => None,
        }
    }
    #[cfg(feature = "gpu")]
    pub fn as_gpu(&self) -> Option<&Arc<super::gpu::GpuCompiledModel>> {
        match self {
            Self::Cpu(_) => None,
            Self::Gpu(compiled) => Some(compiled),
        }
    }
    pub fn is_gpu(&self) -> bool {
        self.as_cpu().is_none()
    }
}

enum SessionKind {
    Cpu(Session),
    #[cfg(feature = "gpu")]
    Gpu(super::gpu::GpuSession),
}

/// Image stages own CPU orchestration independently of inference placement.
pub(crate) struct StageSession {
    inner: SessionKind,
    cpu: Arc<CpuContext>,
    threads: usize,
}
impl StageSession {
    pub(crate) fn new_serial(
        compiled: CompiledStage,
        shape: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        let inner = match compiled {
            CompiledStage::Cpu(compiled) => {
                SessionKind::Cpu(Session::new_serial(compiled, shape, limits)?)
            }
            #[cfg(feature = "gpu")]
            CompiledStage::Gpu(compiled) => {
                SessionKind::Gpu(super::gpu::GpuSession::new(compiled, shape, limits)?)
            }
        };
        Ok(Self {
            inner,
            cpu: CpuContext::serial(),
            threads: 1,
        })
    }
    pub(crate) fn new_prefix(
        compiled: CompiledStage,
        shape: &[usize],
        limits: ExecutionLimits,
        last: usize,
    ) -> RuntimeResult<Self> {
        let inner = match compiled {
            CompiledStage::Cpu(compiled) => {
                SessionKind::Cpu(Session::new_prefix(compiled, shape, limits, last)?)
            }
            #[cfg(feature = "gpu")]
            CompiledStage::Gpu(compiled) => SessionKind::Gpu(super::gpu::GpuSession::new_prefix(
                compiled, shape, limits, last,
            )?),
        };
        Ok(Self {
            inner,
            cpu: CpuContext::serial(),
            threads: 1,
        })
    }
    pub(crate) fn new_diagnostic(
        compiled: CompiledStage,
        shape: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        let inner = match compiled {
            CompiledStage::Cpu(compiled) => {
                SessionKind::Cpu(Session::new_diagnostic(compiled, shape, limits)?)
            }
            #[cfg(feature = "gpu")]
            CompiledStage::Gpu(compiled) => SessionKind::Gpu(
                super::gpu::GpuSession::new_diagnostic(compiled, shape, limits)?,
            ),
        };
        Ok(Self {
            inner,
            cpu: CpuContext::serial(),
            threads: 1,
        })
    }
    pub(crate) fn new_recognition(
        compiled: CompiledStage,
        shape: &[usize],
        limits: ExecutionLimits,
        last: usize,
        _classes: usize,
        probability: bool,
    ) -> RuntimeResult<Self> {
        match compiled {
            CompiledStage::Cpu(compiled) => {
                if probability {
                    Self::new_serial(compiled.into(), shape, limits)
                } else {
                    Self::new_prefix(compiled.into(), shape, limits, last)
                }
            }
            #[cfg(feature = "gpu")]
            CompiledStage::Gpu(compiled) => Ok(Self {
                inner: SessionKind::Gpu(super::gpu::GpuSession::new_recognition(
                    compiled,
                    shape,
                    limits,
                    _classes,
                    probability,
                )?),
                cpu: CpuContext::serial(),
                threads: 1,
            }),
        }
    }
    pub(crate) fn new_classification(
        compiled: CompiledStage,
        shape: &[usize],
        limits: ExecutionLimits,
        last: usize,
    ) -> RuntimeResult<Self> {
        match compiled {
            CompiledStage::Cpu(compiled) => Self::new_prefix(compiled.into(), shape, limits, last),
            #[cfg(feature = "gpu")]
            CompiledStage::Gpu(compiled) => Ok(Self {
                inner: SessionKind::Gpu(super::gpu::GpuSession::new_classification(
                    compiled, shape, limits,
                )?),
                cpu: CpuContext::serial(),
                threads: 1,
            }),
        }
    }
    pub(crate) fn cpu_context(&self) -> &Arc<CpuContext> {
        &self.cpu
    }
    pub(crate) fn intra_op_threads(&self) -> usize {
        self.threads
    }
    pub(crate) fn set_execution_context(
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
            SessionKind::Cpu(session) => session.set_execution_context(cpu.clone(), threads)?,
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(_) => {}
        }
        self.cpu = cpu;
        self.threads = threads.min(16);
        Ok(())
    }
    pub(crate) fn execution_limits(&self) -> ExecutionLimits {
        match &self.inner {
            SessionKind::Cpu(session) => session.execution_limits(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(session) => session.execution_limits(),
        }
    }
    pub(crate) fn input_shape(&self) -> Shape {
        match &self.inner {
            SessionKind::Cpu(session) => session.input_shape(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(session) => session.input_shape(),
        }
    }
    pub(crate) fn output_shape(&self) -> Shape {
        match &self.inner {
            SessionKind::Cpu(session) => session.output_shape(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(session) => session.output_shape(),
        }
    }
    pub(crate) fn reshape(&mut self, shape: &[usize]) -> RuntimeResult<()> {
        match &mut self.inner {
            SessionKind::Cpu(session) => session.reshape(shape),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(session) => session.reshape(shape),
        }
    }
    pub(crate) fn reshape_with_limits(
        &mut self,
        shape: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<()> {
        match &mut self.inner {
            SessionKind::Cpu(session) => session.reshape_with_limits(shape, limits),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(session) => session.reshape_with_limits(shape, limits),
        }
    }
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_input_data(&self) -> RuntimeResult<&[f32]> {
        match &self.inner {
            SessionKind::Gpu(session) => Ok(session.input_data()),
            SessionKind::Cpu(_) => Err(RuntimeError::new("GPU input access requires a GPU stage")),
        }
    }
    pub(crate) fn activation_bytes(&self) -> usize {
        match &self.inner {
            SessionKind::Cpu(session) => session.activation_bytes(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(session) => session.activation_bytes(),
        }
    }
    pub(crate) fn retained_activation_bytes(&self) -> usize {
        match &self.inner {
            SessionKind::Cpu(session) => session.retained_activation_bytes(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(session) => session.retained_activation_bytes(),
        }
    }
    pub(crate) fn retained_scratch_bytes(&self) -> usize {
        match &self.inner {
            SessionKind::Cpu(session) => session.retained_scratch_bytes(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(session) => session.retained_scratch_bytes(),
        }
    }
    pub(crate) fn retained_planning_bytes(&self) -> usize {
        match &self.inner {
            SessionKind::Cpu(session) => session.retained_planning_bytes(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(session) => session.retained_planning_bytes(),
        }
    }
    pub(crate) fn retained_device_bytes(&self) -> usize {
        match &self.inner {
            SessionKind::Cpu(_) => 0,
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(session) => session.retained_device_bytes(),
        }
    }
    pub(crate) fn statistics(&self) -> ExecutionStatistics {
        match &self.inner {
            SessionKind::Cpu(session) => session.statistics(),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(session) => session.statistics(),
        }
    }
    pub(crate) fn run_with_layout_input<F>(&mut self, fill: F) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: FnOnce(&mut [f32], bool) -> RuntimeResult<()>,
    {
        self.run_with_layout_input_profiled::<false, F>(fill)
    }
    pub(crate) fn run_with_layout_input_profiled<const PROFILE: bool, F>(
        &mut self,
        fill: F,
    ) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: FnOnce(&mut [f32], bool) -> RuntimeResult<()>,
    {
        match &mut self.inner {
            SessionKind::Cpu(session) => session.run_with_layout_input(fill),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(session) => {
                session.run_with_input_profiled::<PROFILE>(|input| fill(input, false))
            }
        }
    }
    pub(crate) fn run_with_input_and_trace<F, O>(
        &mut self,
        fill: F,
        observer: O,
    ) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: FnOnce(&mut [f32]) -> RuntimeResult<()>,
        O: for<'a> FnMut(NodeTrace<'a>) -> RuntimeResult<()>,
    {
        match &mut self.inner {
            SessionKind::Cpu(session) => session.run_with_input_and_trace(fill, observer),
            #[cfg(feature = "gpu")]
            SessionKind::Gpu(session) => session.run_with_input_and_trace(fill, observer),
        }
    }
    #[cfg(feature = "gpu")]
    pub(crate) fn run_compact_with_input<F>(
        &mut self,
        fill: F,
    ) -> RuntimeResult<&[super::gpu::CompactRow]>
    where
        F: FnOnce(&mut [f32]) -> RuntimeResult<()>,
    {
        self.run_compact_with_input_profiled::<false, F>(fill)
    }
    #[cfg(feature = "gpu")]
    pub(crate) fn run_compact_with_input_profiled<const PROFILE: bool, F>(
        &mut self,
        fill: F,
    ) -> RuntimeResult<&[super::gpu::CompactRow]>
    where
        F: FnOnce(&mut [f32]) -> RuntimeResult<()>,
    {
        match &mut self.inner {
            SessionKind::Gpu(session) => {
                if PROFILE {
                    session.run_compact_with_input_profiled::<PROFILE>(fill)
                } else {
                    session.run_compact_with_input(fill)
                }
            }
            SessionKind::Cpu(_) => Err(RuntimeError::new(
                "compact GPU execution requires a GPU stage",
            )),
        }
    }
    /// Merge only explicitly profiled production execution, after its output borrow ends.
    pub(crate) fn accumulate_gpu_profile<const PROFILE: bool>(
        &self,
        _profile: &mut crate::profile::StageProfile,
    ) {
        if PROFILE {
            #[cfg(feature = "gpu")]
            if let SessionKind::Gpu(session) = &self.inner {
                let timings = session.gpu_execution_timings();
                _profile.gpu_admission_wait_ns = _profile
                    .gpu_admission_wait_ns
                    .saturating_add(timings.admission_wait_ns);
                _profile.gpu_submission_ns = _profile
                    .gpu_submission_ns
                    .saturating_add(timings.submission_ns);
                _profile.gpu_completion_wait_ns = _profile
                    .gpu_completion_wait_ns
                    .saturating_add(timings.completion_wait_ns);
                _profile.gpu_readback_ns =
                    _profile.gpu_readback_ns.saturating_add(timings.readback_ns);
            }
        }
    }
}

/// Remaining aggregate stage activation budget after retained sibling sessions.
pub(crate) fn remaining_limits(
    limits: ExecutionLimits,
    siblings: usize,
) -> RuntimeResult<ExecutionLimits> {
    Ok(ExecutionLimits {
        max_activation_bytes: limits
            .max_activation_bytes
            .checked_sub(siblings)
            .ok_or_else(|| {
                RuntimeError::with_kind(
                    crate::error::ErrorKind::ResourceLimit,
                    "retained sibling GPU activations exceed max_activation_bytes",
                )
            })?,
    })
}

#[cfg(all(test, feature = "gpu"))]
mod tests;
