//! Storage precision is independent of shader arithmetic: all accumulators are F32.
use super::*;

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
            for (value, dest) in values.iter().zip(bytes.as_chunks_mut::<2>().0.iter_mut()) {
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
        for (dest, src) in values.iter_mut().zip(bytes.as_chunks::<2>().0.iter()) {
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
    /// Required allocation range for this logical view, including its offset
    /// and noncontiguous physical strides. Arithmetic cannot wrap at binding.
    pub(super) fn required_bytes(&self) -> RuntimeResult<u64> {
        if self.strides.len() != self.shape.rank() {
            return Err(resource("GPU tensor stride rank differs"));
        }
        let last = self.shape.dimensions().iter().zip(&self.strides).try_fold(
            self.storage_offset as u64,
            |offset, (&dimension, &stride)| {
                (dimension as u64)
                    .checked_sub(1)?
                    .checked_mul(stride as u64)?
                    .checked_add(offset)
            },
        );
        last.and_then(|offset| offset.checked_add(1))
            .and_then(|elements| {
                elements.checked_mul(if self.precision == GpuPrecision::F32 {
                    4
                } else {
                    2
                })
            })
            .and_then(|bytes| bytes.checked_add(3))
            .map(|bytes| bytes / 4 * 4)
            .ok_or_else(|| resource("GPU tensor binding range overflow"))
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tensor_view_binding_ranges_include_padding_offsets_and_overflow() {
        let mut view =
            GpuTensorDescriptor::canonical(Shape::new(&[2, 3]).unwrap(), GpuPrecision::MixedF16);
        view.storage_offset = 3;
        assert_eq!(view.required_bytes().unwrap(), 20);
        view.strides = vec![8, 1];
        assert_eq!(view.required_bytes().unwrap(), 28);
        view.storage_offset = usize::MAX;
        assert!(view.required_bytes().is_err());
    }
}
