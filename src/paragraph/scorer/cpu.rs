use super::{EDGE_WIDTH, EPSILON, Input, Layer, Linear, NODE_WIDTH, Norm, RELATION_WIDTH, Weights};
use crate::{
    cpu::CpuContext,
    error::{ErrorKind, RuntimeError},
    runtime::{
        Backend, ExecutionLimits, RuntimeResult,
        packing::PackedMatrix,
        parallel::aligned_bands,
        simd::{self, BinaryOp, IdentityStore, StoreTransform},
    },
};
use std::simd::{Simd, num::SimdFloat};

struct PackedLinear {
    input: usize,
    output: usize,
    panel: PackedMatrix,
    bias: Vec<f32>,
}

impl PackedLinear {
    fn pack(linear: Linear) -> RuntimeResult<Self> {
        Ok(Self {
            input: linear.input,
            output: linear.output,
            panel: PackedMatrix::from_columns(&linear.weight, linear.input, linear.output)
                .map_err(RuntimeError::new)?,
            bias: linear.bias,
        })
    }
    fn bytes(&self) -> usize {
        self.panel.bytes() + self.bias.capacity() * size_of::<f32>()
    }
}

struct PackedLayer {
    edge_update: (RelationProjection, PackedLinear),
    edge_norm: Norm,
    attention_mixed: RelationProjection,
    attention_final: PackedLinear,
    message: (RelationProjection, PackedLinear),
    update: [PackedLinear; 2],
    node_norm: Norm,
}

impl PackedLayer {
    fn pack(layer: Layer) -> RuntimeResult<Self> {
        Ok(Self {
            edge_update: pack_relation_mlp(layer.edge_update)?,
            edge_norm: layer.edge_norm,
            attention_mixed: RelationProjection::pack(layer.attention_mixed)?,
            attention_final: PackedLinear::pack(layer.attention_final)?,
            message: pack_relation_mlp(layer.message)?,
            update: pack_mlp(layer.update)?,
            node_norm: layer.node_norm,
        })
    }

    fn bytes(&self) -> usize {
        [&self.edge_update.1, &self.message.1, &self.attention_final]
            .into_iter()
            .chain(&self.update)
            .map(PackedLinear::bytes)
            .sum::<usize>()
            + self.edge_update.0.bytes()
            + self.message.0.bytes()
            + self.attention_mixed.bytes()
            + [&self.edge_norm, &self.node_norm]
                .into_iter()
                .map(|norm| (norm.weight.capacity() + norm.bias.capacity()) * size_of::<f32>())
                .sum::<usize>()
    }
}

/// Factor the affine projection of [receiver, sender, edge] into three
/// projections. A line may participate in many edges, so do its work once.
struct RelationProjection {
    receiver: PackedLinear,
    sender: PackedLinear,
    edge: PackedLinear,
}

impl RelationProjection {
    fn pack(linear: Linear) -> RuntimeResult<Self> {
        assert_eq!(linear.input, RELATION_WIDTH);
        let pack = |start, input, bias| {
            let panel = PackedMatrix::from_fn(input, linear.output, |k, column| {
                linear.weight[column * RELATION_WIDTH + start + k]
            })
            .map_err(RuntimeError::new)?;
            Ok::<_, RuntimeError>(PackedLinear {
                input,
                output: linear.output,
                panel,
                bias,
            })
        };
        Ok(Self {
            receiver: pack(0, NODE_WIDTH, Vec::new())?,
            sender: pack(NODE_WIDTH, NODE_WIDTH, Vec::new())?,
            edge: pack(2 * NODE_WIDTH, EDGE_WIDTH, linear.bias)?,
        })
    }

    fn bytes(&self) -> usize {
        self.receiver.bytes() + self.sender.bytes() + self.edge.bytes()
    }
}

pub(crate) struct CpuWeights {
    node: [PackedLinear; 2],
    edge: [PackedLinear; 2],
    layers: [PackedLayer; 4],
    classifier: (RelationProjection, PackedLinear),
}

fn pack_mlp([a, b]: [Linear; 2]) -> RuntimeResult<[PackedLinear; 2]> {
    Ok([PackedLinear::pack(a)?, PackedLinear::pack(b)?])
}

fn pack_relation_mlp([a, b]: [Linear; 2]) -> RuntimeResult<(RelationProjection, PackedLinear)> {
    Ok((RelationProjection::pack(a)?, PackedLinear::pack(b)?))
}

impl CpuWeights {
    pub(crate) fn pack(weights: Weights) -> RuntimeResult<Self> {
        let [a, b, c, d] = weights.layers;
        Ok(Self {
            node: pack_mlp(weights.node)?,
            edge: pack_mlp(weights.edge)?,
            layers: [
                PackedLayer::pack(a)?,
                PackedLayer::pack(b)?,
                PackedLayer::pack(c)?,
                PackedLayer::pack(d)?,
            ],
            classifier: pack_relation_mlp(weights.classifier)?,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        self.node
            .iter()
            .chain(&self.edge)
            .chain([&self.classifier.1])
            .map(PackedLinear::bytes)
            .sum::<usize>()
            + self.classifier.0.bytes()
            + self.layers.iter().map(PackedLayer::bytes).sum::<usize>()
    }
}

#[derive(Default)]
pub(crate) struct CpuScorer {
    nodes: Vec<f32>,
    edge: Vec<f32>,
    edge_work: Vec<f32>,
    endpoints: Vec<f32>,
    scratch: Vec<f32>,
    update: Vec<f32>,
    attention: Vec<f32>,
    mass: Vec<f32>,
}

impl CpuScorer {
    pub(crate) fn retained_bytes(&self) -> usize {
        [
            &self.nodes,
            &self.edge,
            &self.edge_work,
            &self.endpoints,
            &self.scratch,
            &self.update,
            &self.attention,
            &self.mass,
        ]
        .into_iter()
        .map(|v| v.capacity() * size_of::<f32>())
        .sum()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn score(
        &mut self,
        model: &CpuWeights,
        input: &Input<'_>,
        limits: ExecutionLimits,
        backend: Backend,
        cpu: &CpuContext,
        threads: usize,
        output: &mut Vec<f32>,
    ) -> RuntimeResult<()> {
        let count = input.node_count();
        let edges = input.pairs.len();
        let area = |rows: usize, width: usize| {
            rows.checked_mul(width).ok_or_else(|| {
                RuntimeError::with_kind(
                    ErrorKind::ResourceLimit,
                    "B-Scale CPU activation size overflow",
                )
            })
        };
        let rows = area(edges, 2)?;
        let node_size = area(count, NODE_WIDTH)?;
        let edge_size = area(rows, EDGE_WIDTH)?;
        // The shared scratch buffer serves one projection at a time.
        let sizes = [
            node_size,
            edge_size,
            edge_size,
            area(count, 2 * EDGE_WIDTH)?,
            edge_size.max(node_size),
            area(node_size, 2)?,
            rows,
            count,
        ];
        let capacity = [
            &self.nodes,
            &self.edge,
            &self.edge_work,
            &self.endpoints,
            &self.scratch,
            &self.update,
            &self.attention,
            &self.mass,
        ]
        .map(Vec::capacity);
        let total = |sizes: [usize; 8], additional: usize| {
            sizes
                .into_iter()
                .try_fold(additional, usize::checked_add)
                .and_then(|n| n.checked_mul(size_of::<f32>()))
                .is_none_or(|n| n > limits.max_activation_bytes)
        };
        if total(sizes, edges) {
            return Err(RuntimeError::with_kind(
                ErrorKind::ResourceLimit,
                "B-Scale CPU activation limit exceeded",
            ));
        }
        if total(
            std::array::from_fn(|i| sizes[i].max(capacity[i])),
            edges.max(output.capacity()),
        ) {
            *self = Self::default();
            *output = Vec::new();
        }
        for (buffer, len) in [
            &mut self.nodes,
            &mut self.edge,
            &mut self.edge_work,
            &mut self.endpoints,
            &mut self.scratch,
            &mut self.update,
            &mut self.attention,
            &mut self.mass,
        ]
        .into_iter()
        .zip(sizes)
        {
            resize(buffer, len)?;
        }
        resize(output, edges)?;

        let compute = Compute {
            backend,
            cpu,
            threads,
        };
        let (nodes, edge, edge_work, endpoints, work, update, attention, mass) = (
            &mut self.nodes,
            &mut self.edge,
            &mut self.edge_work,
            &mut self.endpoints,
            &mut self.scratch,
            &mut self.update,
            &mut self.attention,
            &mut self.mass,
        );
        compute.linear(
            &model.node[0],
            input.nodes,
            &mut work[..node_size],
            &ReluStore,
        );
        compute.linear(&model.node[1], &work[..node_size], nodes, &IdentityStore);
        compute.linear(
            &model.edge[0],
            input.edges,
            &mut work[..edge_size],
            &ReluStore,
        );
        compute.linear(&model.edge[1], &work[..edge_size], edge, &IdentityStore);
        for layer in &model.layers {
            compute.relation(
                &layer.edge_update.0,
                input.pairs,
                nodes,
                edge,
                endpoints,
                &mut work[..edge_size],
                &ReluStore,
            );
            compute.linear(
                &layer.edge_update.1,
                &work[..edge_size],
                edge_work,
                &IdentityStore,
            );
            compute.residual_norm(edge, edge_work, &layer.edge_norm);

            let message_size = rows * NODE_WIDTH;
            compute.relation(
                &layer.attention_mixed,
                input.pairs,
                nodes,
                edge,
                endpoints,
                &mut work[..message_size],
                &LeakyReluStore,
            );
            compute.linear(
                &layer.attention_final,
                &work[..message_size],
                attention,
                &WeightStore,
            );
            compute.relation(
                &layer.message.0,
                input.pairs,
                nodes,
                edge,
                endpoints,
                &mut work[..message_size],
                &ReluStore,
            );
            compute.linear(
                &layer.message.1,
                &work[..message_size],
                &mut edge_work[..message_size],
                &IdentityStore,
            );

            // Match the two ScatterElements sums in directed input order.
            mass.fill(0.0);
            for (index, &weight) in attention.iter().enumerate() {
                mass[receiver(input.pairs, index)] += weight;
            }
            for (dst, node) in update
                .chunks_exact_mut(2 * NODE_WIDTH)
                .zip(nodes.chunks_exact(NODE_WIDTH))
            {
                dst[..NODE_WIDTH].copy_from_slice(node);
                dst[NODE_WIDTH..].fill(0.0);
            }
            for (index, (message, &weight)) in edge_work[..message_size]
                .chunks_exact(NODE_WIDTH)
                .zip(attention.iter())
                .enumerate()
            {
                let node = receiver(input.pairs, index);
                let normalized = weight / mass[node].max(1e-12);
                let start = node * (2 * NODE_WIDTH) + NODE_WIDTH;
                for (dst, &value) in update[start..start + NODE_WIDTH].iter_mut().zip(message) {
                    *dst += value * normalized;
                }
            }
            compute.linear(&layer.update[0], update, &mut work[..node_size], &ReluStore);
            compute.linear(
                &layer.update[1],
                &work[..node_size],
                &mut update[..node_size],
                &IdentityStore,
            );
            compute.residual_norm(nodes, &update[..node_size], &layer.node_norm);
        }
        let hidden_size = rows * NODE_WIDTH;
        compute.relation(
            &model.classifier.0,
            input.pairs,
            nodes,
            edge,
            endpoints,
            &mut work[..hidden_size],
            &ReluStore,
        );
        compute.linear(
            &model.classifier.1,
            &work[..hidden_size],
            attention,
            &IdentityStore,
        );
        for (i, score) in output.iter_mut().enumerate() {
            *score = (attention[i] + attention[edges + i]) * 0.5;
        }
        Ok(())
    }
}

fn resize(buffer: &mut Vec<f32>, len: usize) -> RuntimeResult<()> {
    if len > buffer.capacity() {
        buffer
            .try_reserve_exact(len - buffer.len())
            .map_err(|error| {
                RuntimeError::caused_by(
                    ErrorKind::Allocation,
                    "B-Scale activation allocation",
                    error,
                )
            })?;
    }
    buffer.resize(len, 0.0);
    Ok(())
}

fn receiver(pairs: &[[usize; 2]], index: usize) -> usize {
    if index < pairs.len() {
        pairs[index][0]
    } else {
        pairs[index - pairs.len()][1]
    }
}

struct Compute<'a> {
    backend: Backend,
    cpu: &'a CpuContext,
    threads: usize,
}

impl Compute<'_> {
    #[allow(clippy::too_many_arguments)]
    fn relation(
        &self,
        weights: &RelationProjection,
        pairs: &[[usize; 2]],
        nodes: &[f32],
        edge: &[f32],
        endpoints: &mut [f32],
        out: &mut [f32],
        transform: &impl StoreTransform,
    ) {
        let width = weights.edge.output;
        let table_size = (nodes.len() / NODE_WIDTH) * width;
        let (left, right) = endpoints[..2 * table_size].split_at_mut(table_size);
        self.linear(&weights.receiver, nodes, left, &IdentityStore);
        self.linear(&weights.sender, nodes, right, &IdentityStore);
        self.linear(&weights.edge, edge, out, &IdentityStore);
        let (forward, reverse) = out.split_at_mut(pairs.len() * width);
        for (direction, rows) in [forward, reverse].into_iter().enumerate() {
            for (pair, row) in pairs.iter().zip(rows.chunks_exact_mut(width)) {
                let a = pair[direction] * width;
                let b = pair[1 - direction] * width;
                for ((dst, &receiver), &sender) in row
                    .as_chunks_mut::<8>()
                    .0
                    .iter_mut()
                    .zip(left[a..a + width].as_chunks::<8>().0)
                    .zip(right[b..b + width].as_chunks::<8>().0)
                {
                    let sum = Simd::from_array(receiver)
                        + Simd::from_array(sender)
                        + Simd::from_array(*dst);
                    *dst = transform.vector(sum, 0).to_array();
                }
            }
        }
    }

    fn linear(
        &self,
        layer: &PackedLinear,
        input: &[f32],
        out: &mut [f32],
        transform: &impl StoreTransform,
    ) {
        let rows = out.len() / layer.output;
        let apply = |band: &mut [f32], first| {
            let len = band.len() / layer.output;
            simd::matmul_bias_store(
                self.backend,
                &input[first * layer.input..(first + len) * layer.input],
                &[],
                Some(&layer.panel),
                &layer.bias,
                band,
                len,
                layer.input,
                layer.output,
                transform,
                0,
            );
        };
        if self.threads > 1 && rows >= 64 && layer.output >= 16 {
            aligned_bands(out, layer.output, rows, self.threads, 4, self.cpu, apply);
        } else {
            apply(out, 0);
        }
    }

    fn residual_norm(&self, target: &mut [f32], update: &[f32], norm: &Norm) {
        let width = norm.weight.len();
        simd::binary_inplace(self.backend, BinaryOp::Add, Some(update), false, target);
        for row in target.chunks_exact_mut(width) {
            let mean = row.iter().sum::<f32>() / width as f32;
            let variance = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / width as f32;
            simd::layer_norm_affine(
                self.backend,
                None,
                &norm.weight,
                &norm.bias,
                mean,
                (variance + EPSILON).sqrt(),
                row,
            );
        }
    }
}

struct ReluStore;
impl StoreTransform for ReluStore {
    fn scalar(&self, value: f32, _: usize) -> f32 {
        value.max(0.0)
    }
    fn vector<const LANES: usize>(&self, value: Simd<f32, LANES>, _: usize) -> Simd<f32, LANES> {
        value.simd_max(Simd::splat(0.0))
    }
}

struct LeakyReluStore;
impl StoreTransform for LeakyReluStore {
    fn scalar(&self, value: f32, _: usize) -> f32 {
        if value >= 0.0 { value } else { value * 0.2 }
    }
    fn vector<const LANES: usize>(&self, value: Simd<f32, LANES>, _: usize) -> Simd<f32, LANES> {
        value.simd_max(value * Simd::splat(0.2))
    }
}

struct WeightStore;
impl StoreTransform for WeightStore {
    fn scalar(&self, value: f32, _: usize) -> f32 {
        value.clamp(-10.0, 10.0).exp()
    }
    fn vector<const LANES: usize>(&self, value: Simd<f32, LANES>, _: usize) -> Simd<f32, LANES> {
        Simd::from_array(value.to_array().map(|x| self.scalar(x, 0)))
    }
}
