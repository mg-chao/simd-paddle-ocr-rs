//! Immutable pipelines are cached by kernel and operand storage types, not shape.
use super::*;
use operators::ShaderKind;
use std::collections::HashMap;

pub(super) struct Pipelines {
    pub layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    variants: Mutex<HashMap<(ShaderKind, [GpuPrecision; 6]), wgpu::ComputePipeline>>,
}
impl Pipelines {
    pub fn shared(context: &GpuContext) -> RuntimeResult<Arc<Self>> {
        let mut cache = context
            .pipelines
            .lock()
            .map_err(|_| RuntimeError::new("GPU pipeline cache poisoned"))?;
        if let Some(value) = &*cache {
            return Ok(value.clone());
        }
        let result = context.checked("GPU pipeline layout", || Ok(Self::new(&context.device)))?;
        // Compile the portable foundation once. Unfused mixed-storage signatures
        // are added lazily during binding, and reused by all models and shapes.
        for &kind in ShaderKind::ALL {
            result.get(context, kind, [context.precision(); 6])?;
        }
        let result = Arc::new(result);
        *cache = Some(result.clone());
        Ok(result)
    }
    fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..7)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 6 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 5,
                        }
                    },
                    has_dynamic_offset: binding == 6,
                    min_binding_size: if binding == 6 {
                        std::num::NonZeroU64::new(512)
                    } else {
                        None
                    },
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU shared operands"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GPU shared pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        Self {
            layout,
            pipeline_layout,
            variants: Mutex::new(HashMap::new()),
        }
    }
    pub fn get(
        &self,
        context: &GpuContext,
        kind: ShaderKind,
        storage: [GpuPrecision; 6],
    ) -> RuntimeResult<wgpu::ComputePipeline> {
        let mut variants = self
            .variants
            .lock()
            .map_err(|_| RuntimeError::new("GPU pipeline variants poisoned"))?;
        if let Some(pipeline) = variants.get(&(kind, storage)) {
            return Ok(pipeline.clone());
        }
        let pipeline = context.checked("GPU pipeline compilation", || {
            let kernel = if context.options.subgroups
                && matches!(
                    kind,
                    ShaderKind::Reduce | ShaderKind::Softmax | ShaderKind::LayerNorm
                ) {
                include_str!("reduce_subgroup.wgsl").replace(
                    "$MODE",
                    match kind {
                        ShaderKind::Reduce => "0",
                        ShaderKind::Softmax => "1",
                        _ => "2",
                    },
                )
            } else {
                kind.source()
            };
            let mut source = format!("{}\n{}", include_str!("common.wgsl"), kernel).replace(
                "$ENABLE",
                if storage.contains(&GpuPrecision::MixedF16) {
                    "enable f16;"
                } else {
                    ""
                },
            );
            for (token, precision) in ["$A", "$B", "$C", "$D", "$E", "$O"]
                .into_iter()
                .zip(storage)
            {
                source = source.replace(
                    token,
                    if precision == GpuPrecision::F32 {
                        "f32"
                    } else {
                        "f16"
                    },
                );
            }
            let shader = context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(kind.name()),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            Ok(context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(kind.name()),
                    layout: Some(&self.pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }))
        })?;
        variants.insert((kind, storage), pipeline.clone());
        Ok(pipeline)
    }
}
