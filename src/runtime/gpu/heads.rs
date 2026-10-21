//! Compact public-stage heads. Features and dense probabilities remain on device.
use super::{GpuBuffer, GpuContext, RuntimeError, RuntimeResult, checked_storage_bytes, resource};
use crate::{
    error::ErrorKind,
    model::{Model, TensorId},
    projection::ProjectionHead,
    tensor::Shape,
};

const TILE: usize = 64;
const RECORD_BYTES: usize = 16;

/// One public recognition time step or orientation classification result.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CompactRow {
    pub index: u32,
    pub score: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Logit,
    Probability,
    Classification,
}

#[derive(Clone)]
pub(super) struct HeadSpec {
    projection: ProjectionHead,
    mode: Mode,
    node_count: usize,
    source: TensorId,
}

impl HeadSpec {
    pub(super) fn recognition(
        model: &Model,
        classes: usize,
        probability: bool,
    ) -> RuntimeResult<Self> {
        let projection = ProjectionHead::compile(model.graph(), classes, 3, "GPU recognizer")
            .map_err(|e| RuntimeError::new(e.to_string()))?;
        let (mode, node_count, source) = if probability {
            (
                Mode::Probability,
                model.graph().nodes.len(),
                model.graph().outputs[0],
            )
        } else {
            (Mode::Logit, projection.feature_node + 1, projection.feature)
        };
        Ok(Self {
            projection,
            mode,
            node_count,
            source,
        })
    }

    pub(super) fn classification(model: &Model) -> RuntimeResult<Self> {
        let projection = ProjectionHead::compile(model.graph(), 2, 2, "GPU classifier")
            .map_err(|e| RuntimeError::new(e.to_string()))?;
        Ok(Self {
            node_count: projection.feature_node + 1,
            source: projection.feature,
            projection,
            mode: Mode::Classification,
        })
    }

    pub(super) fn node_count(&self) -> usize {
        self.node_count
    }

    pub(super) fn selected_output(&self) -> TensorId {
        self.source
    }

    pub(super) fn rows(&self, shape: Shape) -> RuntimeResult<usize> {
        let rank = if self.mode == Mode::Classification {
            2
        } else {
            3
        };
        let columns = if self.mode == Mode::Probability {
            self.projection.classes
        } else {
            self.projection.inner
        };
        if shape.rank() != rank || shape.dimensions()[rank - 1] != columns || columns == 0 {
            return Err(RuntimeError::new(
                "GPU compact head source shape is incompatible",
            ));
        }
        let rows = shape.element_count() / columns;
        if rows == 0 {
            return Err(RuntimeError::new("GPU compact head requires nonempty rows"));
        }
        Ok(rows)
    }

    pub(super) fn memory_requirements(
        &self,
        shape: Shape,
        limits: &wgpu::Limits,
    ) -> RuntimeResult<(usize, usize)> {
        let geometry = self.geometry(shape, limits)?;
        Ok((geometry.output_bytes, geometry.workspace_bytes))
    }

    fn geometry(&self, source_shape: Shape, limits: &wgpu::Limits) -> RuntimeResult<HeadGeometry> {
        let rows = self.rows(source_shape)?;
        let tiles = self.projection.classes.div_ceil(TILE);
        if tiles == 0 {
            return Err(RuntimeError::new("GPU compact head requires classes"));
        }
        let partial_records = if self.mode == Mode::Classification {
            1
        } else {
            rows.checked_mul(tiles)
                .ok_or_else(|| resource("GPU compact partial count overflow"))?
        };
        let partial_bytes = record_bytes(partial_records, limits)?;
        let output_bytes = record_bytes(rows, limits)?;
        let project_grid = groups(partial_records, limits)?;
        let reduce_grid = super::dispatch_grid(rows, limits)?;
        let u32_value = |value| {
            u32::try_from(value).map_err(|_| resource("GPU compact head parameter exceeds U32"))
        };
        checked_storage_bytes(source_shape.element_count(), limits)?;
        let weight_elements = self
            .projection
            .inner
            .checked_mul(self.projection.classes)
            .ok_or_else(|| resource("GPU compact weight index overflow"))?;
        checked_storage_bytes(weight_elements, limits)?;
        let parameters = [
            u32_value(rows)?,
            u32_value(self.projection.classes)?,
            u32_value(self.projection.inner)?,
            u32_value(tiles)?,
            u32::from(self.projection.bias.is_some()),
            project_grid.0,
            reduce_grid
                .0
                .checked_mul(TILE as u32)
                .ok_or_else(|| resource("GPU compact dispatch stride overflow"))?,
            0,
        ];
        Ok(HeadGeometry {
            rows,
            partial_bytes,
            output_bytes,
            project_grid,
            reduce_grid,
            parameters,
            workspace_bytes: partial_bytes
                .checked_add(output_bytes)
                .ok_or_else(|| resource("GPU compact workspace overflow"))?,
        })
    }
}

struct HeadGeometry {
    rows: usize,
    partial_bytes: usize,
    output_bytes: usize,
    project_grid: (u32, u32),
    reduce_grid: (u32, u32),
    parameters: [u32; 8],
    workspace_bytes: usize,
}

pub(super) struct HeadPipelines {
    layout: wgpu::BindGroupLayout,
    project: wgpu::ComputePipeline,
    probability: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    classify: wgpu::ComputePipeline,
}

impl HeadPipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU compact head layout"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, false),
                storage(4, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GPU compact head pipelines"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU compact heads"),
            source: wgpu::ShaderSource::Wgsl(include_str!("heads.wgsl").into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            project: pipeline("project_tiles"),
            probability: pipeline("probability_tiles"),
            reduce: pipeline("reduce_tiles"),
            classify: pipeline("classify"),
            layout,
        }
    }
}

pub(super) struct HeadPlan {
    spec: HeadSpec,
    rows: usize,
    // Drop bindings before the buffers whose reservations account for them.
    bindings: wgpu::BindGroup,
    _partial: GpuBuffer,
    output: GpuBuffer,
    _parameters: GpuBuffer,
    project_grid: (u32, u32),
    reduce_grid: (u32, u32),
    workspace_bytes: usize,
}

impl HeadPlan {
    pub(super) fn new<'a>(
        spec: &HeadSpec,
        source_shape: Shape,
        context: &GpuContext,
        pipelines: &HeadPipelines,
        resolve: impl Fn(TensorId) -> &'a wgpu::Buffer,
    ) -> RuntimeResult<Self> {
        let HeadGeometry {
            rows,
            partial_bytes,
            output_bytes,
            project_grid,
            reduce_grid,
            parameters,
            workspace_bytes,
        } = spec.geometry(source_shape, &context.limits)?;
        let partial = context.create_buffer(
            "GPU compact partial maxima",
            partial_bytes as u64,
            wgpu::BufferUsages::STORAGE,
        )?;
        let output = context.create_buffer(
            "GPU compact output",
            output_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        let parameters = context.create_buffer_init(
            "GPU compact parameters",
            bytemuck::cast_slice(&parameters),
            wgpu::BufferUsages::UNIFORM,
        )?;
        let weights = resolve(spec.projection.weights);
        // The bias binding is never read when absent; an existing immutable
        // coefficient buffer supplies a valid readonly binding without allocation.
        let bias = spec.projection.bias.map_or(weights, &resolve);
        let bindings = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GPU compact operands"),
                layout: &pipelines.layout,
                entries: &[
                    super::bind_entry(0, resolve(spec.source)),
                    super::bind_entry(1, weights),
                    super::bind_entry(2, bias),
                    super::bind_entry(3, &partial),
                    super::bind_entry(4, &output),
                    super::bind_entry(5, &parameters),
                ],
            });
        Ok(Self {
            spec: spec.clone(),
            rows,
            _partial: partial,
            output,
            _parameters: parameters,
            bindings,
            project_grid,
            reduce_grid,
            workspace_bytes,
        })
    }

    pub(super) fn encode(&self, encoder: &mut wgpu::CommandEncoder, pipelines: &HeadPipelines) {
        let dispatch = |encoder: &mut wgpu::CommandEncoder,
                        pipeline: &wgpu::ComputePipeline,
                        grid: (u32, u32)| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GPU compact head"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bindings, &[]);
            pass.dispatch_workgroups(grid.0, grid.1, 1);
        };
        match self.spec.mode {
            Mode::Classification => dispatch(encoder, &pipelines.classify, self.reduce_grid),
            Mode::Logit | Mode::Probability => {
                dispatch(
                    encoder,
                    if self.spec.mode == Mode::Logit {
                        &pipelines.project
                    } else {
                        &pipelines.probability
                    },
                    self.project_grid,
                );
                dispatch(encoder, &pipelines.reduce, self.reduce_grid);
            }
        }
    }

    pub(super) fn output_buffer(&self) -> &wgpu::Buffer {
        &self.output
    }

    pub(super) fn output_bytes(&self) -> usize {
        self.rows * RECORD_BYTES
    }

    pub(super) fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    pub(super) fn parameter_bytes(&self) -> usize {
        8 * std::mem::size_of::<u32>()
    }

    pub(super) fn retained_device_bytes(&self) -> usize {
        self.workspace_bytes + self.parameter_bytes()
    }

    pub(super) fn decode(&self, bytes: &[u8], output: &mut Vec<CompactRow>) -> RuntimeResult<()> {
        decode_records(&self.spec, self.rows, bytes, output)
    }
}

fn decode_records(
    spec: &HeadSpec,
    rows: usize,
    bytes: &[u8],
    output: &mut Vec<CompactRow>,
) -> RuntimeResult<()> {
    if rows.checked_mul(RECORD_BYTES) != Some(bytes.len()) {
        return Err(RuntimeError::new("GPU compact readback size mismatch"));
    }
    let record = |bytes: &[u8]| {
        let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        (word(0), f32::from_bits(word(4)), word(8))
    };
    for bytes in bytes.chunks_exact(RECORD_BYTES) {
        let (index, score, valid) = record(bytes);
        if valid != 1
            || index as usize >= spec.projection.classes
            || !score.is_finite()
            || (spec.mode != Mode::Logit && !(0. ..=1.).contains(&score))
        {
            return Err(RuntimeError::new(
                "nonfinite or invalid GPU compact head candidate",
            ));
        }
    }
    if output.capacity() < rows {
        output
            .try_reserve_exact(rows.saturating_sub(output.len()))
            .map_err(|e| {
                RuntimeError::caused_by(ErrorKind::Allocation, "GPU compact host output", e)
            })?;
    }
    output.clear();
    for bytes in bytes.chunks_exact(RECORD_BYTES) {
        let (index, score, _) = record(bytes);
        output.push(CompactRow { index, score });
    }
    Ok(())
}

fn record_bytes(records: usize, limits: &wgpu::Limits) -> RuntimeResult<usize> {
    let words = records
        .checked_mul(RECORD_BYTES / 4)
        .ok_or_else(|| resource("GPU compact record size overflow"))?;
    usize::try_from(checked_storage_bytes(words, limits)?)
        .map_err(|_| resource("GPU compact byte count exceeds host address space"))
}

fn groups(count: usize, limits: &wgpu::Limits) -> RuntimeResult<(u32, u32)> {
    let maximum = limits.max_compute_workgroups_per_dimension as usize;
    if count == 0 || maximum == 0 || u32::try_from(count).is_err() {
        return Err(resource(
            "GPU compact workgroup count exceeds device limits",
        ));
    }
    let x = count.min(maximum);
    let y = count.div_ceil(x);
    if y > maximum || x.checked_mul(y).is_none_or(|n| n > u32::MAX as usize) {
        return Err(resource("GPU compact dispatch grid exceeds device limits"));
    }
    Ok((x as u32, y as u32))
}

#[cfg(test)]
#[path = "head_tests.rs"]
mod tests;
