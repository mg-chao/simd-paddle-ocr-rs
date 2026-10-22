//! Storage precision is independent of shader arithmetic: all accumulators are F32.
use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub enum GpuPrecision {
    #[default]
    F32,
    MixedF16,
}
impl GpuPrecision {
    pub(super) fn bytes(self, elements: usize) -> usize {
        (elements * if self == Self::F32 { 4 } else { 2 }).div_ceil(4) * 4
    }
    pub(super) fn encode(self, values: &[f32]) -> Vec<u8> {
        let mut bytes = vec![0; self.bytes(values.len())];
        self.write(values, &mut bytes);
        bytes
    }
    pub(super) fn write(self, values: &[f32], bytes: &mut [u8]) {
        if self == Self::F32 {
            bytes[..values.len() * 4].copy_from_slice(bytemuck::cast_slice(values));
        } else {
            for (value, dest) in values.iter().zip(bytes.chunks_exact_mut(2)) {
                dest.copy_from_slice(&half::f16::from_f32(*value).to_le_bytes());
            }
        }
    }
    pub(super) fn decode(self, bytes: &[u8], values: &mut [f32]) -> RuntimeResult<()> {
        if bytes.len() != self.bytes(values.len()) {
            return Err(RuntimeError::with_kind(
                ErrorKind::GpuMapping,
                "GPU readback length mismatch",
            ));
        }
        if self == Self::F32 {
            return decode_f32(bytes, values);
        }
        for (dest, src) in values.iter_mut().zip(bytes.chunks_exact(2)) {
            *dest = half::f16::from_le_bytes([src[0], src[1]]).to_f32();
        }
        Ok(())
    }
}

/// Physical tensor descriptors used by schedules and diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum GpuLayout {
    Canonical,
    PaddedNhwc { channels: usize },
}
#[derive(Clone, Debug)]
pub struct GpuTensorDescriptor {
    pub shape: Shape,
    pub layout: GpuLayout,
    pub precision: GpuPrecision,
    pub strides: Vec<usize>,
    pub storage_offset: usize,
}
impl GpuTensorDescriptor {
    pub(super) fn canonical(shape: Shape, precision: GpuPrecision) -> Self {
        let mut stride = 1;
        let mut strides = vec![0; shape.rank()];
        for (axis, &dim) in shape.dimensions().iter().enumerate().rev() {
            strides[axis] = stride;
            stride *= dim;
        }
        Self {
            shape,
            layout: GpuLayout::Canonical,
            precision,
            strides,
            storage_offset: 0,
        }
    }
}
