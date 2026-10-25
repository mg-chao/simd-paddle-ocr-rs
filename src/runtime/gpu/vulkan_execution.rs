//! Prepared Vulkan compute pipelines and bindings over the shared resident graph.
use super::*;

pub(super) const SEPARATE_TRANSFER_ENCODERS: bool = false;

struct Binding {
    slot: usize,
    minimum: u64,
    access: wgpu::BufferBindingType,
}

pub(super) struct NativePipeline {
    raw: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    owner: Arc<Ledger>,
    pub(super) workgroup_size: [u32; 3],
    bindings: Vec<Binding>,
}

impl std::fmt::Debug for NativePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanPipeline")
            .field("workgroup_size", &self.workgroup_size)
            .finish_non_exhaustive()
    }
}
impl PartialEq for NativePipeline {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}
impl Eq for NativePipeline {}

impl NativePipeline {
    pub(super) fn validate_binding_sizes(
        &self,
        count: usize,
        size: impl Fn(usize) -> u64,
    ) -> RuntimeResult<()> {
        for &Binding {
            slot: binding,
            minimum,
            ..
        } in &self.bindings
        {
            if binding >= count || size(binding) < minimum {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    format!("Vulkan binding {binding} needs {minimum} bytes"),
                ));
            }
        }
        Ok(())
    }
}

pub(super) struct NativeDispatch {
    pipeline: Arc<NativePipeline>,
    bindings: wgpu::BindGroup,
    owners: Vec<GpuBuffer>,
    groups: [u32; 3],
    hazards: Box<[(usize, bool)]>,
}
impl std::fmt::Debug for NativeDispatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDispatch")
            .field("groups", &self.groups)
            .finish_non_exhaustive()
    }
}
impl NativeDispatch {
    pub(super) fn prepare(
        pipeline: Arc<NativePipeline>,
        owners: Vec<GpuBuffer>,
        offsets: Vec<u64>,
        groups: [u32; 3],
        read_only: &[bool],
    ) -> RuntimeResult<Arc<Self>> {
        if owners.len() != offsets.len() || owners.len() != read_only.len() + 1 {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "Vulkan binding/offset count differs",
            ));
        }
        let limits = pipeline.owner.owner_device.limits();
        if groups
            .iter()
            .any(|&n| n > limits.max_compute_workgroups_per_dimension)
        {
            return Err(resource("Vulkan dispatch exceeds device limits"));
        }
        for (i, (owner, &offset)) in owners.iter().zip(&offsets).enumerate() {
            let uniform = i == read_only.len();
            let usage = if uniform {
                wgpu::BufferUsages::UNIFORM
            } else {
                wgpu::BufferUsages::STORAGE
            };
            let alignment = if uniform {
                limits.min_uniform_buffer_offset_alignment
            } else {
                limits.min_storage_buffer_offset_alignment
            };
            if !Arc::ptr_eq(&owner.reservation.ledger, &pipeline.owner)
                || !owner.usage().contains(usage)
            {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "Vulkan buffer context or usage differs",
                ));
            }
            if offset >= owner.size() || !offset.is_multiple_of(alignment as u64) {
                return Err(resource(
                    "Vulkan binding offset is outside allocation or unaligned",
                ));
            }
        }
        pipeline.validate_binding_sizes(owners.len(), |i| owners[i].size() - offsets[i])?;
        for binding in &pipeline.bindings {
            let expected = match binding.access {
                wgpu::BufferBindingType::Uniform => binding.slot == read_only.len(),
                wgpu::BufferBindingType::Storage { read_only: read } => {
                    read_only.get(binding.slot) == Some(&read)
                }
            };
            if !expected {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "Vulkan binding access differs from shader declaration",
                ));
            }
        }
        let entries = pipeline
            .bindings
            .iter()
            .map(
                |&Binding {
                     slot: i, minimum, ..
                 }| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &owners[i],
                        offset: offsets[i],
                        size: if i == read_only.len() {
                            std::num::NonZeroU64::new(minimum)
                        } else {
                            None
                        },
                    }),
                },
            )
            .collect::<Vec<_>>();
        let bindings = pipeline
            .owner
            .owner_device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prepared resident bindings"),
                layout: &pipeline.layout,
                entries: &entries,
            });
        let hazards = owners
            .iter()
            .zip(read_only)
            .map(|(b, &r)| (Arc::as_ptr(&b.reservation) as usize, r))
            .collect();
        Ok(Arc::new(Self {
            pipeline,
            bindings,
            owners,
            groups,
            hazards,
        }))
    }
    pub(super) fn groups(&self) -> [u32; 3] {
        self.groups
    }
    #[cfg(test)]
    pub(super) fn pipeline_for_test(&self) -> Arc<NativePipeline> {
        self.pipeline.clone()
    }
    pub(super) fn hazards(&self) -> &[(usize, bool)] {
        &self.hazards
    }
    pub(super) fn owners(&self) -> &[GpuBuffer] {
        &self.owners
    }
    pub(super) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.owners.capacity() * std::mem::size_of::<GpuBuffer>()
            + std::mem::size_of_val(self.hazards.as_ref())
    }
}

pub(super) fn compile_wgsl(
    context: &GpuContext,
    source: &str,
    entry: &str,
    _: [u32; 3],
) -> RuntimeResult<Arc<NativePipeline>> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| RuntimeError::with_kind(ErrorKind::GpuValidation, e.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| RuntimeError::with_kind(ErrorKind::GpuValidation, e.to_string()))?;
    let index = module
        .entry_points
        .iter()
        .position(|ep| ep.name == entry && ep.stage == naga::ShaderStage::Compute)
        .ok_or_else(|| unsupported("Vulkan compute entry is missing"))?;
    let workgroup_size = module.entry_points[index].workgroup_size;
    let ep = info.get_entry_point(index);
    let mut bindings = Vec::new();
    let mut entries = Vec::new();
    for (handle, global) in module.global_variables.iter() {
        let Some(binding) = global.binding else {
            continue;
        };
        if ep[handle].is_empty() {
            continue;
        }
        if binding.group != 0 {
            return Err(unsupported("Vulkan requires group-zero bindings"));
        }
        let ty = match global.space {
            naga::AddressSpace::Uniform => wgpu::BufferBindingType::Uniform,
            naga::AddressSpace::Storage { access } => wgpu::BufferBindingType::Storage {
                read_only: !access.contains(naga::StorageAccess::STORE),
            },
            _ => return Err(unsupported("Vulkan requires buffer bindings")),
        };
        let minimum = module.types[global.ty].inner.size(module.to_ctx()) as u64;
        bindings.push(Binding {
            slot: binding.binding as usize,
            minimum,
            access: ty,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: binding.binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(minimum),
            },
            count: None,
        });
    }
    context.checked("Vulkan shader compilation", || {
        let layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("resident buffers"),
                entries: &entries,
            });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                });
        let shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                // Reflection already parsed this module. Reuse its IR instead
                // of parsing the same WGSL again during first-use compilation.
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
            });
        let raw = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        Ok(Arc::new(NativePipeline {
            raw,
            layout,
            owner: context.ledger.clone(),
            workgroup_size,
            bindings,
        }))
    })
}

pub(super) fn encode(
    context: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    dispatches: &[(usize, &NativeDispatch)],
    _: bool,
) -> RuntimeResult<()> {
    for (_, dispatch) in dispatches {
        if !Arc::ptr_eq(&dispatch.pipeline.owner, &context.ledger) {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "foreign Vulkan dispatch",
            ));
        }
    }
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("resident graph"),
        timestamp_writes: None,
    });
    let mut bound = None;
    for (_, dispatch) in dispatches {
        let identity = Arc::as_ptr(&dispatch.pipeline);
        if bound != Some(identity) {
            pass.set_pipeline(&dispatch.pipeline.raw);
            bound = Some(identity);
        }
        pass.set_bind_group(0, &dispatch.bindings, &[]);
        let [x, y, z] = dispatch.groups();
        pass.dispatch_workgroups(x, y, z);
    }
    Ok(())
}

fn unsupported(message: impl Into<String>) -> RuntimeError {
    RuntimeError::with_kind(ErrorKind::Unsupported, message)
}
pub(super) fn compile_msl(
    _: &GpuContext,
    _: &str,
    _: &str,
    _: [u32; 3],
) -> RuntimeResult<Arc<NativePipeline>> {
    Err(unsupported("Metal specialization is unavailable on Vulkan"))
}
pub(super) fn compile_resident_msl(
    c: &GpuContext,
    s: &str,
    e: &str,
    w: [u32; 3],
    _: usize,
) -> RuntimeResult<Arc<NativePipeline>> {
    compile_msl(c, s, e, w)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn compile_msl_metadata(
    c: &GpuContext,
    s: &str,
    e: &str,
    w: [u32; 3],
    _: Vec<usize>,
    _: Vec<usize>,
    _: Vec<(usize, u64)>,
) -> RuntimeResult<Arc<NativePipeline>> {
    compile_msl(c, s, e, w)
}

#[cfg(test)]
#[path = "vulkan_execution_tests.rs"]
mod tests;
