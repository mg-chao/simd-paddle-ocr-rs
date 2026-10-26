//! Backend-independent device information and the resource contract used by OCR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum GpuApi {
    Vulkan,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum GpuDeviceType {
    DiscreteGpu,
    IntegratedGpu,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct GpuAdapterInfo {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: GpuDeviceType,
    pub backend: GpuApi,
    pub driver: String,
    pub driver_info: String,
    pub api_version: u32,
    pub subgroup_min_size: u32,
    pub subgroup_max_size: u32,
}
/// Limits relevant to tensor storage, parameter packets and compute dispatch.
#[derive(Clone, Debug)]
pub struct GpuLimits {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_uniform_buffer_binding_size: u32,
    /// Conservative common bound; native validation also checks each axis.
    pub max_compute_workgroups_per_dimension: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub min_uniform_buffer_offset_alignment: u32,
    pub min_storage_buffer_offset_alignment: u32,
}
impl Default for GpuLimits {
    fn default() -> Self {
        Self {
            max_buffer_size: 256 << 20,
            max_storage_buffer_binding_size: 128 << 20,
            max_uniform_buffer_binding_size: 65536,
            max_compute_workgroups_per_dimension: 65535,
            max_compute_workgroup_storage_size: 16384,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
        }
    }
}
/// Actual roles of a buffer, checked before recording any GPU operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BufferUsage(u8);
impl BufferUsage {
    #[cfg(test)]
    pub const fn empty() -> Self {
        Self(0)
    }
    pub const STORAGE: Self = Self(1);
    pub const UNIFORM: Self = Self(2);
    pub const COPY_SRC: Self = Self(4);
    pub const COPY_DST: Self = Self(8);
    pub const MAP_READ: Self = Self(16);
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}
impl std::ops::BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}
