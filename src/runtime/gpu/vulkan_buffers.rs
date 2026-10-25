//! Device-local Vulkan storage; initialization is staged by wgpu.
use super::*;

#[derive(Clone, Debug)]
pub(super) struct NativeBuffer;

pub(super) fn allocate(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
    contents: Option<&[u8]>,
) -> RuntimeResult<(wgpu::Buffer, NativeBuffer)> {
    if size == 0 || size > device.limits().max_buffer_size {
        return Err(resource("Vulkan allocation exceeds device buffer limit"));
    }
    if contents.is_some_and(|bytes| bytes.len() as u64 > size) {
        return Err(RuntimeError::with_kind(
            ErrorKind::InvalidInput,
            "initial contents exceed buffer capacity",
        ));
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: contents.is_some(),
    });
    if let Some(contents) = contents {
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .map_err(|e| RuntimeError::new(e.to_string()))?
            .slice(..contents.len())
            .copy_from_slice(contents);
        buffer.unmap();
    }
    Ok((buffer, NativeBuffer))
}
