//! Metal-only shader fixtures for its existing compiler and ABI tests.
use super::super::*;
use resident::{Batch, Kernel};

pub(in crate::runtime::gpu) struct MetalTestShader<'a> {
    pub source: &'a str,
    pub entry_point: &'a str,
    pub workgroup_size: [u32; 3],
}
pub(in crate::runtime::gpu) trait MetalTestBatch {
    fn dispatch_msl(
        &mut self,
        name: &str,
        shader: MetalTestShader<'_>,
        buffers: &[(&GpuBuffer, bool)],
        words: &[u32],
        groups: [u32; 3],
    ) -> RuntimeResult<()>;
    fn dispatch_wgsl(
        &mut self,
        name: &str,
        source: &str,
        buffers: &[(&GpuBuffer, bool)],
        words: &[u32],
        groups: [u32; 3],
    ) -> RuntimeResult<()>;
}
impl MetalTestBatch for Batch {
    fn dispatch_msl(
        &mut self,
        name: &str,
        shader: MetalTestShader<'_>,
        buffers: &[(&GpuBuffer, bool)],
        words: &[u32],
        groups: [u32; 3],
    ) -> RuntimeResult<()> {
        let pipeline = executor::compile_resident_msl(
            &self.context,
            shader.source,
            shader.entry_point,
            shader.workgroup_size,
            buffers.len() + 1,
        )?;
        let kernel = Kernel::test_pipeline(&self.context, pipeline, buffers);
        self.dispatch_prepared(name, &kernel, buffers, words, groups)
    }
    fn dispatch_wgsl(
        &mut self,
        name: &str,
        source: &str,
        buffers: &[(&GpuBuffer, bool)],
        words: &[u32],
        groups: [u32; 3],
    ) -> RuntimeResult<()> {
        let module = naga::front::wgsl::parse_str(source).map_err(|e| {
            RuntimeError::with_kind(ErrorKind::GpuValidation, e.emit_to_string(source))
        })?;
        for (_, global) in module.global_variables.iter() {
            let Some(binding) = global.binding else {
                continue;
            };
            let valid = binding.group == 0
                && match global.space {
                    naga::AddressSpace::Storage { access } => buffers
                        .get(binding.binding as usize)
                        .is_some_and(|(_, read)| {
                            *read == !access.contains(naga::StorageAccess::STORE)
                        }),
                    naga::AddressSpace::Uniform => binding.binding as usize == buffers.len(),
                    _ => false,
                };
            if !valid {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "Metal fixture binding layout differs",
                ));
            }
        }
        let pipeline = executor::compile_wgsl(&self.context, source, "main", [0; 3])?;
        let kernel = Kernel::test_pipeline(&self.context, pipeline, buffers);
        self.dispatch_prepared(name, &kernel, buffers, words, groups)
    }
}
