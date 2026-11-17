//! Device-resident B-Scale scorer over host-prepared paragraph features.
use super::{resident::Batch, *};
use crate::paragraph::scorer::{
    EDGE_INPUT, EDGE_WIDTH, EPSILON, Input, Layer, Linear, NODE_INPUT, NODE_WIDTH, Norm,
    RELATION_WIDTH, Weights,
};

struct DeviceLinear {
    input: usize,
    output: usize,
    weight: GpuBuffer,
    bias: Option<GpuBuffer>,
}

struct DeviceNorm {
    weight: GpuBuffer,
    bias: GpuBuffer,
}

struct DeviceLayer {
    edge_update: [DeviceLinear; 2],
    edge_norm: DeviceNorm,
    attention_mixed: DeviceLinear,
    attention_final: DeviceLinear,
    message: [DeviceLinear; 2],
    update: [DeviceLinear; 2],
    node_norm: DeviceNorm,
}

struct WeightUpload<'a> {
    context: &'a GpuContext,
    bytes: usize,
}

impl WeightUpload<'_> {
    fn buffer(&mut self, values: &[f32]) -> RuntimeResult<GpuBuffer> {
        let bytes = bytemuck::cast_slice(values);
        let buffer =
            self.context
                .create_buffer_init("paragraph weights", bytes, BufferUsage::STORAGE)?;
        self.bytes += buffer.bytes();
        Ok(buffer)
    }

    fn linear(&mut self, layer: &Linear) -> RuntimeResult<DeviceLinear> {
        // Existing matrix shaders consume the right operand in [inner, columns] order.
        let mut transposed = vec![0.0; layer.weight.len()];
        for (column, weights) in layer.weight.chunks_exact(layer.input).enumerate() {
            for (inner, &weight) in weights.iter().enumerate() {
                transposed[inner * layer.output + column] = weight;
            }
        }
        Ok(DeviceLinear {
            input: layer.input,
            output: layer.output,
            weight: self.buffer(&transposed)?,
            bias: if layer.bias.is_empty() {
                None
            } else {
                Some(self.buffer(&layer.bias)?)
            },
        })
    }

    fn mlp(&mut self, layers: &[Linear; 2]) -> RuntimeResult<[DeviceLinear; 2]> {
        Ok([self.linear(&layers[0])?, self.linear(&layers[1])?])
    }

    fn norm(&mut self, norm: &Norm) -> RuntimeResult<DeviceNorm> {
        Ok(DeviceNorm {
            weight: self.buffer(&norm.weight)?,
            bias: self.buffer(&norm.bias)?,
        })
    }

    fn layer(&mut self, layer: &Layer) -> RuntimeResult<DeviceLayer> {
        Ok(DeviceLayer {
            edge_update: self.mlp(&layer.edge_update)?,
            edge_norm: self.norm(&layer.edge_norm)?,
            attention_mixed: self.linear(&layer.attention_mixed)?,
            attention_final: self.linear(&layer.attention_final)?,
            message: self.mlp(&layer.message)?,
            update: self.mlp(&layer.update)?,
            node_norm: self.norm(&layer.node_norm)?,
        })
    }
}

pub(crate) struct ParagraphGpuPlan {
    context: Arc<GpuContext>,
    dummy: GpuBuffer,
    node: [DeviceLinear; 2],
    edge: [DeviceLinear; 2],
    layers: [DeviceLayer; 4],
    classifier: [DeviceLinear; 2],
    weight_bytes: usize,
}

const NEURAL_READS: &[bool] = &[true, true, true, true, true, false];
const GRAPH_READS: &[bool] = &[true, true, true, true, false];

impl ParagraphGpuPlan {
    pub(crate) fn new(context: Arc<GpuContext>, weights: &Weights) -> RuntimeResult<Self> {
        for key in [
            KernelId::General,
            KernelId::Matmul16,
            KernelId::Matmul32,
            KernelId::LayerNorm,
        ] {
            resident::Kernel::get(&context, KernelKey::new(key), NEURAL_READS)?;
        }
        resident::Kernel::get(
            &context,
            KernelKey::new(KernelId::ParagraphGraph),
            GRAPH_READS,
        )?;
        let _gate = context.acquire_gate()?;
        let mut upload = WeightUpload {
            context: &context,
            bytes: 0,
        };
        let dummy = upload.buffer(&[0.0])?;
        let node = upload.mlp(&weights.node)?;
        let edge = upload.mlp(&weights.edge)?;
        let layers = [
            upload.layer(&weights.layers[0])?,
            upload.layer(&weights.layers[1])?,
            upload.layer(&weights.layers[2])?,
            upload.layer(&weights.layers[3])?,
        ];
        let classifier = upload.mlp(&weights.classifier)?;
        Ok(Self {
            context: context.clone(),
            dummy,
            node,
            edge,
            layers,
            classifier,
            weight_bytes: upload.bytes,
        })
    }

    pub(crate) fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }
}

#[derive(Clone, Copy)]
struct Dimensions {
    nodes: usize,
    pairs: usize,
    directions: usize,
    offsets: usize,
    incident: usize,
    index_words: usize,
    bytes: usize,
}

impl Dimensions {
    fn new(nodes: usize, pairs: usize) -> RuntimeResult<Self> {
        let directions = pairs
            .checked_mul(2)
            .filter(|&n| n <= u32::MAX as usize / RELATION_WIDTH)
            .ok_or_else(|| resource("paragraph GPU direction count exceeds shader indexing"))?;
        if nodes > u32::MAX as usize / (2 * NODE_WIDTH) {
            return Err(resource("paragraph GPU node count exceeds shader indexing"));
        }
        let offsets = 2 * directions;
        let incident = offsets + nodes + 1;
        let index_words = incident + directions;
        let bytes = (nodes
            .checked_mul(NODE_INPUT + 5 * NODE_WIDTH)
            .and_then(|n| {
                let floats = EDGE_INPUT + 3 * EDGE_WIDTH + RELATION_WIDTH + 2 * NODE_WIDTH + 2;
                n.checked_add(directions.checked_mul(floats)?)
            })
            .and_then(|n| n.checked_add(index_words))
            .and_then(|n| n.checked_add(pairs.checked_mul(2)?)))
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .ok_or_else(|| resource("paragraph GPU activation size overflow"))?;
        Ok(Self {
            nodes,
            pairs,
            directions,
            offsets,
            incident,
            index_words,
            bytes,
        })
    }

    fn contains(self, other: Self) -> bool {
        self.nodes >= other.nodes && self.pairs >= other.pairs
    }
}

struct Scratch {
    dimensions: Dimensions,
    node_features: GpuBuffer,
    edge_features: GpuBuffer,
    index_buffer: GpuBuffer,
    nodes: GpuBuffer,
    node_hidden: GpuBuffer,
    node_pre_norm: GpuBuffer,
    node_join: GpuBuffer,
    edges: GpuBuffer,
    edge_hidden: GpuBuffer,
    edge_pre_norm: GpuBuffer,
    relation: GpuBuffer,
    message_hidden: GpuBuffer,
    message: GpuBuffer,
    logits: GpuBuffer,
    probability: GpuBuffer,
    symmetric: GpuBuffer,
    readback: GpuBuffer,
}

impl Scratch {
    fn new(context: &GpuContext, dimensions: Dimensions) -> RuntimeResult<Self> {
        let nodes = dimensions.nodes;
        let edges = dimensions.directions;
        let buffer =
            |label, elements| resident::storage(context, label, elements * size_of::<f32>());
        Ok(Self {
            dimensions,
            node_features: buffer("paragraph node features", nodes * NODE_INPUT)?,
            edge_features: buffer("paragraph edge features", edges * EDGE_INPUT)?,
            index_buffer: buffer("paragraph indices", dimensions.index_words)?,
            nodes: buffer("paragraph nodes", nodes * NODE_WIDTH)?,
            node_hidden: buffer("paragraph node hidden", nodes * NODE_WIDTH)?,
            node_pre_norm: buffer("paragraph node pre-normalization", nodes * NODE_WIDTH)?,
            node_join: buffer("paragraph node and message", nodes * 2 * NODE_WIDTH)?,
            edges: buffer("paragraph edges", edges * EDGE_WIDTH)?,
            edge_hidden: buffer("paragraph edge hidden", edges * EDGE_WIDTH)?,
            edge_pre_norm: buffer("paragraph edge pre-normalization", edges * EDGE_WIDTH)?,
            relation: buffer("paragraph relations", edges * RELATION_WIDTH)?,
            message_hidden: buffer("paragraph message hidden", edges * NODE_WIDTH)?,
            message: buffer("paragraph messages", edges * NODE_WIDTH)?,
            logits: buffer("paragraph directional logits", edges)?,
            probability: buffer("paragraph probability", edges)?,
            symmetric: buffer("paragraph symmetric logits", dimensions.pairs)?,
            readback: context.create_buffer(
                "paragraph logit readback",
                (dimensions.pairs * size_of::<f32>()) as u64,
                BufferUsage::MAP_READ | BufferUsage::COPY_DST,
            )?,
        })
    }
}

pub(crate) struct ParagraphGpuRequest {
    plan: Arc<ParagraphGpuPlan>,
    scratch: Option<Scratch>,
    indices: Vec<u32>,
    cursors: Vec<u32>,
    statistics: ExecutionStatistics,
}

impl ParagraphGpuRequest {
    pub(crate) fn new(plan: Arc<ParagraphGpuPlan>) -> RuntimeResult<Self> {
        plan.context.check_alive()?;
        let adapter = plan.context.adapter_info();
        let statistics = ExecutionStatistics {
            execution_backend: Some(super::super::ExecutionBackend::Gpu),
            effective_precision: "f32".into(),
            adapter: Some(adapter.name.clone()),
            backend: format!("{:?}", adapter.backend),
            ..Default::default()
        };
        Ok(Self {
            plan,
            scratch: None,
            indices: Vec::new(),
            cursors: Vec::new(),
            statistics,
        })
    }

    pub(crate) fn retained_device_bytes(&self) -> usize {
        self.scratch.as_ref().map_or(0, |s| s.dimensions.bytes)
    }

    pub(crate) fn retained_host_bytes(&self) -> usize {
        (self.indices.capacity() + self.cursors.capacity()) * size_of::<u32>()
    }

    pub(crate) fn statistics(&self) -> ExecutionStatistics {
        let mut statistics = self.statistics.clone();
        statistics.retained_activation_bytes = self.retained_device_bytes();
        statistics.scratch_bytes = self.retained_host_bytes();
        statistics
    }

    fn prepare_indices(
        &mut self,
        dimensions: Dimensions,
        pairs: &[[usize; 2]],
    ) -> RuntimeResult<()> {
        reserve(&mut self.indices, dimensions.index_words)?;
        reserve(&mut self.cursors, dimensions.nodes)?;
        self.indices.resize(dimensions.index_words, 0);
        self.cursors.resize(dimensions.nodes, 0);
        let (endpoints, segments) = self.indices.split_at_mut(dimensions.offsets);
        let (offsets, incident) = segments.split_at_mut(dimensions.nodes + 1);
        offsets.fill(0);
        for (i, &[a, b]) in pairs.iter().enumerate() {
            if a >= dimensions.nodes || b >= dimensions.nodes {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "paragraph GPU pair endpoint is outside the node table",
                ));
            }
            endpoints[2 * i..][..2].copy_from_slice(&[a as u32, b as u32]);
            endpoints[2 * (i + dimensions.pairs)..][..2].copy_from_slice(&[b as u32, a as u32]);
            offsets[a + 1] += 1;
            offsets[b + 1] += 1;
        }
        for i in 0..dimensions.nodes {
            offsets[i + 1] += offsets[i];
        }
        self.cursors.copy_from_slice(&offsets[..dimensions.nodes]);
        // Preserve the ONNX ScatterElements input order within each receiver segment.
        for (row, pair) in endpoints.chunks_exact(2).enumerate() {
            let cursor = &mut self.cursors[pair[0] as usize];
            incident[*cursor as usize] = row as u32;
            *cursor += 1;
        }
        Ok(())
    }

    pub(crate) fn score(
        &mut self,
        input: &Input<'_>,
        limits: ExecutionLimits,
        output: &mut Vec<f32>,
    ) -> RuntimeResult<()> {
        if input.pairs.is_empty() {
            output.clear();
            return Ok(());
        }
        let dimensions = Dimensions::new(input.node_count(), input.pairs.len())?;
        if input.nodes.len() != dimensions.nodes * NODE_INPUT
            || input.edges.len() != dimensions.directions * EDGE_INPUT
            || dimensions.nodes == 0
        {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "invalid paragraph GPU feature shape",
            ));
        }
        let fit = |device_bytes: usize, host_elements: usize| {
            device_bytes
                .checked_add(host_elements.saturating_mul(size_of::<f32>()))
                .is_some_and(|bytes| bytes <= limits.max_activation_bytes)
        };
        if !fit(dimensions.bytes, output.capacity().max(dimensions.pairs)) {
            return Err(resource("paragraph GPU activation limit exceeded"));
        }
        reserve(output, dimensions.pairs)?;
        if !fit(dimensions.bytes, output.capacity()) {
            return Err(resource("paragraph GPU activation limit exceeded"));
        }
        self.prepare_indices(dimensions, input.pairs)?;

        let context = &self.plan.context;
        let _gate = context.acquire_gate()?;
        if self.scratch.as_ref().is_none_or(|scratch| {
            !scratch.dimensions.contains(dimensions)
                || !fit(scratch.dimensions.bytes, output.capacity())
        }) {
            self.scratch = Some(Scratch::new(context, dimensions)?);
            self.statistics.plan_cache_misses += 1;
        } else {
            self.statistics.plan_cache_hits += 1;
        }
        let scratch = self.scratch.as_ref().unwrap();
        let mut batch = Batch::new(context.clone())?;
        for (buffer, bytes) in [
            (&scratch.node_features, bytemuck::cast_slice(input.nodes)),
            (&scratch.edge_features, bytemuck::cast_slice(input.edges)),
            (&scratch.index_buffer, bytemuck::cast_slice(&self.indices)),
        ] {
            context.upload_resident(buffer, bytes)?;
            batch.statistics.uploaded_bytes += bytes.len() as u64;
        }
        Schedule {
            batch: &mut batch,
            plan: &self.plan,
            scratch,
            dimensions,
        }
        .encode()?;
        let (_, stats) = batch.read_into(
            &scratch.symmetric,
            &scratch.readback,
            dimensions.pairs * size_of::<f32>(),
            &_gate,
            |bytes| {
                output.clear();
                output.extend(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&bytes| f32::from_ne_bytes(bytes)),
                );
                Ok(())
            },
        )?;
        self.statistics.activation_bytes = dimensions.bytes;
        self.statistics.physical_steps = stats.dispatches as usize;
        self.statistics.production_submissions += stats.submissions;
        self.statistics.production_readbacks += 1;
        self.statistics.uploaded_bytes += stats.uploaded_bytes;
        self.statistics.output_readback_bytes += stats.output_readback_bytes;
        Ok(())
    }
}

fn reserve<T>(values: &mut Vec<T>, length: usize) -> RuntimeResult<()> {
    values
        .try_reserve_exact(length.saturating_sub(values.len()))
        .map_err(|error| {
            RuntimeError::caused_by(ErrorKind::Allocation, "paragraph GPU data", error)
        })
}

#[repr(u32)]
enum GraphOp {
    Relation = 0,
    Aggregate = 1,
    LeakyRelu = 2,
    Exp = 3,
    Symmetrize = 4,
}

struct Schedule<'a> {
    batch: &'a mut Batch,
    plan: &'a ParagraphGpuPlan,
    scratch: &'a Scratch,
    dimensions: Dimensions,
}

impl Schedule<'_> {
    fn encode(&mut self) -> RuntimeResult<()> {
        let s = self.scratch;
        let n = self.dimensions.nodes;
        let d = self.dimensions.directions;
        let dummy = &self.plan.dummy;
        self.mlp(
            &self.plan.node,
            n,
            [&s.node_features, &s.node_hidden, &s.nodes],
            None,
        )?;
        self.mlp(
            &self.plan.edge,
            d,
            [&s.edge_features, &s.edge_hidden, &s.edges],
            None,
        )?;
        self.relation()?;
        for layer in &self.plan.layers {
            self.mlp(
                &layer.edge_update,
                d,
                [&s.relation, &s.edge_hidden, &s.edge_pre_norm],
                Some(&s.edges),
            )?;
            self.norm(&layer.edge_norm, d, EDGE_WIDTH, &s.edge_pre_norm, &s.edges)?;
            self.relation()?;
            self.linear(
                &layer.attention_mixed,
                d,
                &s.relation,
                &s.message_hidden,
                false,
                None,
            )?;
            self.graph(
                GraphOp::LeakyRelu,
                d * NODE_WIDTH,
                [&s.message_hidden, dummy, dummy, &s.message],
            )?;
            self.linear(
                &layer.attention_final,
                d,
                &s.message,
                &s.logits,
                false,
                None,
            )?;
            self.graph(GraphOp::Exp, d, [&s.logits, dummy, dummy, &s.probability])?;
            self.mlp(
                &layer.message,
                d,
                [&s.relation, &s.message_hidden, &s.message],
                None,
            )?;
            self.graph(
                GraphOp::Aggregate,
                n,
                [&s.message, &s.probability, &s.nodes, &s.node_join],
            )?;
            self.mlp(
                &layer.update,
                n,
                [&s.node_join, &s.node_hidden, &s.node_pre_norm],
                Some(&s.nodes),
            )?;
            self.norm(&layer.node_norm, n, NODE_WIDTH, &s.node_pre_norm, &s.nodes)?;
            self.relation()?;
        }
        self.mlp(
            &self.plan.classifier,
            d,
            [&s.relation, &s.message_hidden, &s.logits],
            None,
        )?;
        self.graph(
            GraphOp::Symmetrize,
            self.dimensions.pairs,
            [&s.logits, dummy, dummy, &s.symmetric],
        )
    }

    fn relation(&mut self) -> RuntimeResult<()> {
        let s = self.scratch;
        self.graph(
            GraphOp::Relation,
            self.dimensions.directions * RELATION_WIDTH,
            [&s.nodes, &s.edges, &self.plan.dummy, &s.relation],
        )
    }

    fn mlp(
        &mut self,
        layers: &[DeviceLinear; 2],
        rows: usize,
        [input, hidden, output]: [&GpuBuffer; 3],
        residual: Option<&GpuBuffer>,
    ) -> RuntimeResult<()> {
        self.linear(&layers[0], rows, input, hidden, true, None)?;
        self.linear(&layers[1], rows, hidden, output, false, residual)
    }

    fn linear(
        &mut self,
        layer: &DeviceLinear,
        rows: usize,
        input: &GpuBuffer,
        output: &GpuBuffer,
        relu: bool,
        residual: Option<&GpuBuffer>,
    ) -> RuntimeResult<()> {
        let inner = layer.input;
        let columns = layer.output;
        let elements = rows * columns;
        let mut words = [0u32; 80];
        words[0] = elements as u32;
        words[2..6].copy_from_slice(&[20, rows as u32, inner as u32, columns as u32]);
        words[65] = columns as u32;
        words[66] = 1;
        for (active, code, broadcast) in [
            (layer.bias.is_some(), 1, 1),
            (residual.is_some(), 2, 2),
            (relu, 3, 0),
        ] {
            if active {
                let start = 68 + words[64] as usize * 3;
                words[start] = code;
                words[start + 1] = broadcast;
                words[64] += 1;
            }
        }
        let limits = self.plan.context.limits();
        let groups16 = rows.div_ceil(16) * columns.div_ceil(16);
        let groups32 = rows.div_ceil(32) * columns.div_ceil(32);
        let (kernel, name, x, y) = if inner >= 16 && columns >= 16 && groups16 >= 64 {
            let tile = if columns >= 64 && groups32 >= 64 {
                32
            } else {
                16
            };
            let (x, y) = dispatch_grid(rows.div_ceil(tile) * columns.div_ceil(tile) * 64, limits)?;
            words[32] = rows.div_ceil(tile) as u32;
            words[33] = columns.div_ceil(tile) as u32;
            words[34] = x;
            words[35] = 1;
            if tile == 32 {
                (KernelId::Matmul32, "paragraph_matmul32", x, y)
            } else {
                (KernelId::Matmul16, "paragraph_matmul16", x, y)
            }
        } else {
            let (x, y) = dispatch_grid(elements, limits)?;
            words[1] = x * 64;
            (KernelId::General, "paragraph_matmul", x, y)
        };
        let dummy = &self.plan.dummy;
        self.batch.dispatch(
            name,
            KernelKey::new(kernel),
            &[
                (input, true),
                (&layer.weight, true),
                (dummy, true),
                (layer.bias.as_ref().unwrap_or(dummy), true),
                (residual.unwrap_or(dummy), true),
                (output, false),
            ],
            &words,
            [x, y, 1],
        )
    }

    fn norm(
        &mut self,
        norm: &DeviceNorm,
        rows: usize,
        width: usize,
        input: &GpuBuffer,
        output: &GpuBuffer,
    ) -> RuntimeResult<()> {
        let dummy = &self.plan.dummy;
        let (x, y) = dispatch_grid(rows * 64, self.plan.context.limits())?;
        self.batch.dispatch(
            "paragraph_layer_norm",
            KernelKey::new(KernelId::LayerNorm),
            &[
                (input, true),
                (&norm.weight, true),
                (&norm.bias, true),
                (dummy, true),
                (dummy, true),
                (output, false),
            ],
            &[rows as u32, x, 0, width as u32, EPSILON.to_bits()],
            [x, y, 1],
        )
    }

    fn graph(
        &mut self,
        op: GraphOp,
        count: usize,
        [a, b, c, output]: [&GpuBuffer; 4],
    ) -> RuntimeResult<()> {
        let (name, dispatch_count) = match op {
            GraphOp::Relation => ("paragraph_gather_endpoints", count),
            GraphOp::Aggregate => ("paragraph_aggregate_messages", count * 64),
            GraphOp::LeakyRelu => ("paragraph_leaky_relu", count),
            GraphOp::Exp => ("paragraph_exp", count),
            GraphOp::Symmetrize => ("paragraph_symmetrize", count),
        };
        let (x, y) = dispatch_grid(dispatch_count, self.plan.context.limits())?;
        let stride = if matches!(op, GraphOp::Aggregate) {
            x
        } else {
            x * 64
        };
        self.batch.dispatch(
            name,
            KernelKey::new(KernelId::ParagraphGraph),
            &[
                (a, true),
                (b, true),
                (c, true),
                (&self.scratch.index_buffer, true),
                (output, false),
            ],
            &[
                op as u32,
                count as u32,
                stride,
                self.dimensions.offsets as u32,
                self.dimensions.incident as u32,
            ],
            [x, y, 1],
        )
    }
}
