//! Shared, initialized Metal allocations imported into the owning wgpu device.
//!
//! Keeping an owned native handle avoids reading private wgpu buffer fields and
//! lets the native graph encoder bind the same allocation as transfer commands.
#![cfg(target_os = "macos")]

use super::{ErrorKind, RuntimeError, RuntimeResult, resource};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResource, MTLResourceOptions};

#[derive(Clone, Debug)]
pub(in crate::runtime::gpu) struct NativeBuffer {
    pub(in crate::runtime::gpu) raw: Retained<ProtocolObject<dyn MTLBuffer>>,
}

// SAFETY: Retained Metal buffer resources can be retained, released and bound
// from multiple threads, as also required by wgpu-hal's Metal Buffer wrapper.
// This wrapper never grants a mutable Rust slice or performs a CPU write after
// publication. GPU/CPU ordering and conflicting GPU accesses remain the owning
// executor's responsibility; retaining a resource is not a synchronization act.
unsafe impl Send for NativeBuffer {}
unsafe impl Sync for NativeBuffer {}

/// Allocate one initialized Metal buffer and expose it through both APIs.
///
/// The caller accounts for `size` in its normal resource reservation. Optional
/// contents may occupy a prefix; every remaining byte is initialized to zero.
/// Size is preserved exactly, so callers performing aligned copy operations
/// must request their usual aligned capacity.
pub(super) fn allocate(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
    contents: Option<&[u8]>,
) -> RuntimeResult<(wgpu::Buffer, NativeBuffer)> {
    objc2::rc::autoreleasepool(|_| allocate_inner(device, label, size, usage, contents))
}

fn allocate_inner(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
    contents: Option<&[u8]>,
) -> RuntimeResult<(wgpu::Buffer, NativeBuffer)> {
    if size == 0 || size > device.limits().max_buffer_size {
        return Err(resource(
            "native Metal allocation exceeds device buffer limit",
        ));
    }
    let length = usize::try_from(size)
        .ok()
        .filter(|&length| length <= isize::MAX as usize)
        .ok_or_else(|| resource("native Metal allocation exceeds host address range"))?;
    if contents.is_some_and(|bytes| bytes.len() > length) {
        return Err(RuntimeError::with_kind(
            ErrorKind::InvalidInput,
            "native Metal initial contents exceed buffer capacity",
        ));
    }

    // SAFETY: The guard is borrowed only to allocate from the exact underlying
    // device. No underlying device/resource is destroyed through this handle.
    let metal = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }.ok_or_else(|| {
        RuntimeError::with_kind(
            ErrorKind::Unsupported,
            "native Metal buffers require a Metal wgpu device",
        )
    })?;
    let raw = metal
        .raw_device()
        .newBufferWithLength_options(
            length,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked,
        )
        .ok_or_else(|| {
            RuntimeError::with_kind(
                ErrorKind::Allocation,
                "native Metal buffer allocation failed",
            )
        })?;
    // Release the HAL device guard before calling back into wgpu for import.
    drop(metal);
    raw.setLabel(Some(&NSString::from_str(label)));

    // SAFETY: Shared storage exposes a CPU-accessible allocation of `length`
    // bytes. The new allocation has no GPU users and has not been published.
    // Initializing the complete range satisfies create_buffer_from_hal's
    // initialized-memory contract, including unused alignment/tail bytes.
    unsafe {
        let destination = raw.contents().cast::<u8>().as_ptr();
        std::ptr::write_bytes(destination, 0, length);
        if let Some(bytes) = contents {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
        }
    }
    let native = NativeBuffer { raw };
    // SAFETY: This retained allocation was created on this wgpu device's Metal
    // device, has nonzero exact `size`, is fully initialized, and supports the
    // requested buffer usages. Shared storage also supports wgpu map/readback.
    // The HAL and native wrappers each retain the same Metal object.
    let buffer = unsafe {
        let hal = wgpu::hal::metal::Device::buffer_from_raw(native.raw.clone(), size);
        device.create_buffer_from_hal::<wgpu::hal::api::Metal>(
            hal,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            },
        )
    };
    Ok((buffer, native))
}
