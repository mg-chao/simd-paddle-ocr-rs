//! Bounded production timing records. No activation tracing is involved.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum GpuTimingGranularity {
    Dispatch,
    Pass,
    Unavailable,
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
    pub fn compute_ns(&self) -> Option<u64> {
        self.kernels
            .iter()
            .try_fold(0u64, |sum, entry| {
                entry.device_ns.map(|ns| sum.saturating_add(ns))
            })
            .filter(|_| !self.kernels.is_empty())
    }
}

pub(super) struct TimestampStorage {
    query: wgpu::QuerySet,
    resolve: GpuBuffer,
    readback: GpuBuffer,
    capacity: u32,
}
impl TimestampStorage {
    pub(super) fn new(context: &GpuContext, count: u32) -> RuntimeResult<Self> {
        if count == 0 || count > wgpu::QUERY_SET_MAX_QUERIES {
            return Err(resource(
                "GPU timestamp query count exceeds device contract",
            ));
        }
        context.checked("GPU profiling allocation", || {
            let query = context.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("OCR production timestamps"),
                count,
                ty: wgpu::QueryType::Timestamp,
            });
            let size = u64::from(count) * 8;
            Ok(Self {
                query,
                resolve: context.create_buffer(
                    "GPU timestamp resolve",
                    size,
                    wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                )?,
                readback: context.create_buffer(
                    "GPU timestamp readback",
                    size,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                )?,
                capacity: count,
            })
        })
    }
    pub(super) fn bytes(&self) -> usize {
        self.resolve.bytes() + self.readback.bytes()
    }
    pub(super) fn capacity(&self) -> u32 {
        self.capacity
    }
    pub(super) fn writes(&self, index: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: Some(index + 1),
        }
    }
    pub(super) fn write(&self, pass: &mut wgpu::ComputePass<'_>, index: u32) {
        pass.write_timestamp(&self.query, index);
    }
    pub(super) fn resolve(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        if count == 0 {
            return;
        }
        encoder.resolve_query_set(&self.query, 0..count, &self.resolve, 0);
    }
    pub(super) fn copy_results(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        if count == 0 {
            return;
        }
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, u64::from(count) * 8);
    }
    pub(super) fn map(&self, count: u32) {
        if count == 0 {
            return;
        }
        self.readback
            .slice(..u64::from(count) * 8)
            .map_async(wgpu::MapMode::Read, |_| {});
    }
    pub(super) fn finish(
        &self,
        context: &GpuContext,
        report: &mut GpuExecutionProfile,
    ) -> RuntimeResult<()> {
        if report.kernels.is_empty() {
            return Ok(());
        }
        let result = (|| {
            let bytes = self
                .readback
                .slice(..report.kernels.len() as u64 * 16)
                .get_mapped_range()
                .map_err(|e| {
                    RuntimeError::caused_by(
                        ErrorKind::GpuMapping,
                        "GPU timestamps mapping failed",
                        e,
                    )
                })?;
            let period = f64::from(context.queue.get_timestamp_period());
            for (index, record) in report.kernels.iter_mut().enumerate() {
                let offset = index * 16;
                let start = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
                let end = u64::from_le_bytes(bytes[offset + 8..offset + 16].try_into().unwrap());
                record.device_ns = if start == 0 && end == 0 {
                    None
                } else {
                    end.checked_sub(start)
                        .map(|ticks| (ticks as f64 * period) as u64)
                };
                if record.device_ns.is_none() {
                    record.granularity = GpuTimingGranularity::Unavailable;
                }
            }
            Ok(())
        })();
        self.readback.unmap();
        result
    }
    pub(super) fn unmap(&self) {
        self.readback.unmap();
    }
}
