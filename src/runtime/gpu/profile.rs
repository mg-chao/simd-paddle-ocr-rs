//! Bounded production timing records. No activation tracing is involved.
use std::mem::size_of;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum GpuTimingGranularity {
    Unavailable,
    Dispatch,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct GpuKernelProfile {
    pub stage: String,
    pub nodes: Vec<usize>,
    pub kernel: String,
    pub output_shape: Vec<usize>,
    pub dispatch_count: u64,
    pub device_ns: Option<u64>,
    pub granularity: GpuTimingGranularity,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct GpuExecutionProfile {
    pub kernels: Vec<GpuKernelProfile>,
}
impl GpuExecutionProfile {
    pub(crate) fn retained_bytes(&self) -> usize {
        self.kernels.capacity() * size_of::<GpuKernelProfile>()
            + self
                .kernels
                .iter()
                .map(|k| {
                    k.stage.capacity()
                        + k.kernel.capacity()
                        + (k.nodes.capacity() + k.output_shape.capacity()) * size_of::<usize>()
                })
                .sum::<usize>()
    }

    pub fn compute_ns(&self) -> Option<u64> {
        self.kernels
            .iter()
            .try_fold(0u64, |sum, entry| {
                entry.device_ns.map(|ns| sum.saturating_add(ns))
            })
            .filter(|_| !self.kernels.is_empty())
    }
}
