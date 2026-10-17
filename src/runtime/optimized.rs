//! Physical execution is derived from, and never rewrites, the normalized graph.
use super::kernels::{BinaryOp, BoundKind};
use super::*;
use crate::tensor::TensorView;

mod elementwise_alias;
mod layout;
mod semantic;
mod store;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Layout {
    Canonical,
    Nhwc,
}
#[derive(Clone)]
struct Value {
    shape: Shape,
    layout: Layout,
    constant: Option<TensorId>,
    offset: usize,
    alias: Option<(usize, usize)>,
}
#[derive(Clone, Copy)]
enum Activation {
    Relu,
    HardSwish,
    Gelu,
    Sigmoid,
}
#[derive(Clone)]
enum Epilogue {
    Add(usize),
    Activation(Activation),
}

/// A borrowed program over independent convolution accumulator lanes. The graph
/// planner has already checked every Add's shape and liveness. Looking up the
/// original operand at a global output index keeps worker bands and tiled
/// convolutions identical without materializing a residual or an intermediate.
struct EpilogueStore<'a, F> {
    epilogues: &'a [Epilogue],
    inputs: &'a [usize],
    data: &'a F,
    channels: usize,
    plane: usize,
    layout: Layout,
    math: simd::UnaryMath,
}

impl<'data, F> simd::StoreTransform for EpilogueStore<'_, F>
where
    F: Fn(usize) -> &'data [f32] + Sync,
{
    #[inline(always)]
    fn scalar(&self, mut value: f32, index: usize) -> f32 {
        for epilogue in self.epilogues {
            match epilogue {
                Epilogue::Add(slot) => {
                    let operand = (self.data)(self.inputs[*slot]);
                    value += operand[if operand.len() == 1 {
                        0
                    } else if operand.len() == self.channels {
                        self.channel(index)
                    } else {
                        index
                    }];
                }
                Epilogue::Activation(act) => {
                    value = simd::unary_value_with_math(activation_op(act), value, self.math)
                }
            }
        }
        value
    }

    #[inline(always)]
    fn vector<const LANES: usize>(
        &self,
        mut value: std::simd::Simd<f32, LANES>,
        index: usize,
    ) -> std::simd::Simd<f32, LANES> {
        use std::simd::Simd;
        for epilogue in self.epilogues {
            match epilogue {
                Epilogue::Add(slot) => {
                    let operand = (self.data)(self.inputs[*slot]);
                    value += if operand.len() == 1 {
                        Simd::splat(operand[0])
                    } else if operand.len() == self.channels && self.layout == Layout::Canonical {
                        // Spatial vectors normally stay inside one channel. Use
                        // each global lane index as well so a short plane or a
                        // different store tile cannot cross a channel silently.
                        Simd::from_array(std::array::from_fn(|lane| {
                            operand[self.channel(index + lane)]
                        }))
                    } else {
                        let start = if operand.len() == self.channels {
                            self.channel(index)
                        } else {
                            index
                        };
                        Simd::load_or_default(&operand[start..(start + LANES).min(operand.len())])
                    };
                }
                Epilogue::Activation(act) => {
                    value = simd::unary_vector_with_math(activation_op(act), value, self.math)
                }
            }
        }
        value
    }
}
impl<F> EpilogueStore<'_, F> {
    #[inline(always)]
    fn channel(&self, index: usize) -> usize {
        if self.layout == Layout::Nhwc {
            index % self.channels
        } else {
            index / self.plane % self.channels
        }
    }
}
#[derive(Clone)]
// Keep bound descriptors inline: no per-node allocation or hot pointer chase.
#[allow(clippy::large_enum_variant)]
enum Operation {
    Native {
        index: usize,
        kind: BoundKind,
        epilogues: Vec<Epilogue>,
    },
    Convert,
    Activation(Activation),
    LayerNorm {
        inner: usize,
        epsilon: f32,
    },
    View,
}
#[derive(Clone)]
struct Step {
    inputs: Vec<usize>,
    output: usize,
    op: Operation,
}
#[derive(Clone)]
struct Plan {
    values: Vec<Value>,
    steps: Vec<Step>,
    input: usize,
    output: usize,
    elements: usize,
    fused: usize,
    conversions: usize,
}
pub(super) struct State {
    plan: Arc<Plan>,
    arena: Vec<f32>,
    canonical: Vec<f32>,
    workspaces: Vec<Vec<f32>>,
    cache: VecDeque<Arc<Plan>>,
    hits: u64,
    misses: u64,
    // Opt-in local diagnosis of the real fused path; absent from library builds.
    #[cfg(test)]
    step_timings: Option<Vec<std::time::Duration>>,
}
fn add_value(
    values: &mut Vec<Value>,
    shape: Shape,
    layout: Layout,
    constant: Option<TensorId>,
) -> usize {
    let id = values.len();
    values.push(Value {
        shape,
        layout,
        constant,
        offset: 0,
        alias: None,
    });
    id
}
/// Storage order is identical for one channel or one spatial position.
/// The planner still binds an explicit layout-only alias so later consumers
/// see the requested tag without moving a byte or extending an arena borrow.
fn layout_neutral(shape: Shape) -> bool {
    shape.rank() == 4
        && (shape.dimensions()[1] == 1
            || (shape.dimensions()[2] == 1 && shape.dimensions()[3] == 1))
}

fn conv_nhwc_capable(compiled: &CompiledModel, index: usize, p: &kernels::Conv) -> bool {
    let instruction = &compiled.instructions[index];
    let constant_rank = |slot: usize, rank: usize| {
        instruction.inputs.get(slot).is_some_and(|id| {
            matches!(&compiled.model.graph().tensors[id.0].constant,
                Some(TensorData::F32(tensor)) if tensor.shape().rank() == rank)
        })
    };
    if !constant_rank(1, 4)
        || (instruction.inputs.len() > 2 && !constant_rank(2, 1))
        || p.dh != 1
        || p.dw != 1
    {
        return false;
    }
    if p.cpg == p.cin {
        p.cout >= 16 && p.cout.is_multiple_of(16)
    } else {
        p.cpg == 1 && p.opg == 1 && p.cout == p.cin && p.cout.is_multiple_of(8)
    }
}

fn convert(
    values: &mut Vec<Value>,
    steps: &mut Vec<Step>,
    conversions: &mut HashMap<(usize, Layout), usize>,
    id: usize,
    target: Layout,
) -> usize {
    if values[id].layout == target || values[id].shape.rank() != 4 {
        return id;
    }
    if let Some(&value) = conversions.get(&(id, target)) {
        return value;
    }
    let output = add_value(values, values[id].shape, target, None);
    if layout_neutral(values[id].shape) {
        values[output].alias = Some((id, 0));
    } else {
        steps.push(Step {
            inputs: vec![id],
            output,
            op: Operation::Convert,
        });
    }
    conversions.insert((id, target), output);
    output
}
fn constant(compiled: &CompiledModel, id: TensorId) -> &[f32] {
    match &compiled.model.graph().tensors[id.0].constant {
        Some(TensorData::F32(t)) => t.data(),
        _ => unreachable!("validated F32 constant"),
    }
}
fn scalar(compiled: &CompiledModel, id: TensorId) -> Option<f32> {
    match &compiled.model.graph().tensors[id.0].constant {
        Some(TensorData::F32(t)) if t.data().len() == 1 => Some(t.data()[0]),
        _ => None,
    }
}
fn other_binary(instruction: &Instruction, value: TensorId) -> Option<TensorId> {
    if instruction.inputs.first() == Some(&value) {
        instruction.inputs.get(1).copied()
    } else if instruction.inputs.get(1) == Some(&value) {
        instruction.inputs.first().copied()
    } else {
        None
    }
}
fn closed(
    compiled: &CompiledModel,
    first: usize,
    end: usize,
    count: usize,
    selected: TensorId,
) -> bool {
    for instruction in &compiled.instructions[first..end] {
        if instruction.output == selected {
            return false;
        }
        // Compilation validates source order: a produced tensor cannot have a
        // consumer before this group. Only later uses inside this execution
        // prefix can require an intermediate to remain materialized.
        if compiled
            .graph_uses
            .used_after(instruction.output, end, count)
        {
            return false;
        }
    }
    true
}
fn activation(
    compiled: &CompiledModel,
    start: usize,
    input: TensorId,
    count: usize,
) -> Option<(Activation, usize)> {
    let n = &compiled.instructions;
    if start >= count {
        return None;
    }
    if matches!(n[start].kind, Op::Relu) && n[start].inputs[0] == input {
        return Some((Activation::Relu, start));
    }
    if matches!(n[start].kind, Op::Sigmoid) && n[start].inputs[0] == input {
        return Some((Activation::Sigmoid, start));
    }
    if start + 1 < count
        && matches!(n[start].kind, Op::HardSigmoid { alpha, beta } if alpha == 0.16666667 && beta == 0.5)
        && n[start].inputs[0] == input
        && matches!(n[start + 1].kind, Op::Mul)
        && other_binary(&n[start + 1], n[start].output) == Some(input)
    {
        return Some((Activation::HardSwish, start + 1));
    }
    if start + 4 < count
        && matches!(n[start].kind, Op::Div)
        && n[start].inputs[0] == input
        && scalar(compiled, n[start].inputs[1]) == Some(std::f32::consts::SQRT_2)
        && matches!(n[start + 1].kind, Op::Erf)
        && n[start + 1].inputs[0] == n[start].output
        && matches!(n[start + 2].kind, Op::Add)
        && other_binary(&n[start + 2], n[start + 1].output).and_then(|id| scalar(compiled, id))
            == Some(1.)
        && matches!(n[start + 3].kind, Op::Mul)
        && other_binary(&n[start + 3], n[start + 2].output) == Some(input)
        && matches!(n[start + 4].kind, Op::Mul)
        && other_binary(&n[start + 4], n[start + 3].output).and_then(|id| scalar(compiled, id))
            == Some(0.5)
    {
        return Some((Activation::Gelu, start + 4));
    }
    None
}
fn layer_norm(
    compiled: &CompiledModel,
    plan: &ShapePlan,
    start: usize,
    count: usize,
) -> Option<(usize, f32, TensorId, TensorId)> {
    if start + 8 >= count {
        return None;
    }
    let n = &compiled.instructions[start..start + 9];
    let input = n[0].inputs[0];
    let shape = plan.shapes[input.0];
    let inner = *shape.dimensions().last()?;
    let trailing = |kind: &BoundKind| matches!(kind, BoundKind::Reduce(r) if r.contiguous == Some(inner) && r.reduction == inner && r.input == shape);
    let kept_axis = |output: TensorId| {
        let out = plan.shapes[output.0];
        out.rank() == shape.rank()
            && out.dimensions().last() == Some(&1)
            && out.dimensions()[..out.rank() - 1] == shape.dimensions()[..shape.rank() - 1]
    };
    if !kept_axis(n[0].output) || !kept_axis(n[3].output) {
        return None;
    }
    if !trailing(&plan.steps[start].kind)
        || !matches!(n[1].kind, Op::Sub)
        || n[1].inputs.as_ref() != [input, n[0].output]
        || !matches!(n[2].kind, Op::Square)
        || n[2].inputs[0] != n[1].output
        || !trailing(&plan.steps[start + 3].kind)
        || n[3].inputs[0] != n[2].output
        || !matches!(n[4].kind, Op::Add)
        || !matches!(n[5].kind, Op::Sqrt)
        || n[5].inputs[0] != n[4].output
        || !matches!(n[6].kind, Op::Div)
        || n[6].inputs.as_ref() != [n[1].output, n[5].output]
        || !matches!(n[7].kind, Op::Mul)
        || !matches!(n[8].kind, Op::Add)
    {
        return None;
    }
    let epsilon = scalar(compiled, other_binary(&n[4], n[3].output)?)?;
    let gamma = other_binary(&n[7], n[6].output)?;
    let beta = other_binary(&n[8], n[7].output)?;
    for id in [gamma, beta] {
        match &compiled.model.graph().tensors[id.0].constant {
            Some(TensorData::F32(t))
                if t.data().len() == inner
                    && t.dimensions().last() == Some(&inner)
                    && t.dimensions()[..t.dimensions().len() - 1]
                        .iter()
                        .all(|&d| d == 1) => {}
            _ => return None,
        }
    }
    Some((inner, epsilon, gamma, beta))
}
impl Plan {
    fn bind(
        compiled: &CompiledModel,
        logical: &ShapePlan,
        count: usize,
        selected: TensorId,
    ) -> RuntimeResult<Self> {
        let mut values = Vec::new();
        let mut steps = Vec::new();
        let mut map = vec![None; logical.shapes.len()];
        let mut conversions = HashMap::new();
        let mut fused = 0;
        let nhwc = compiled.options.layout == LayoutPolicy::Auto;
        let input_layout = if nhwc
            && logical.input_shape.rank() == 4
            && !layout_neutral(logical.input_shape)
            && logical.steps.first().is_some_and(|step| match &step.kind {
                BoundKind::Conv(p) => p.cpg == p.cin && conv_nhwc_capable(compiled, 0, p),
                _ => false,
            }) {
            Layout::Nhwc
        } else {
            Layout::Canonical
        };
        let input = add_value(&mut values, logical.input_shape, input_layout, None);
        map[compiled.input.0] = Some(input);
        for (i, tensor) in compiled.model.graph().tensors.iter().enumerate() {
            if let Some(TensorData::F32(t)) = &tensor.constant {
                map[i] = Some(add_value(
                    &mut values,
                    t.shape(),
                    Layout::Canonical,
                    Some(TensorId(i)),
                ));
            }
        }
        let mut semantic_members = vec![false; count];
        let mut semantic_ends = (0..count).map(|_| None).collect::<Vec<_>>();
        for group in semantic::groups(compiled, logical, count, selected) {
            for &member in &group.members {
                semantic_members[member] = true;
            }
            let end = group.end;
            semantic_ends[end] = Some(group);
        }
        // An adjacent matcher may consume ordinary gap instructions, but must
        // stop before any instruction reserved by a semantic group. Its use
        // checks still inspect the full graph prefix, including later groups.
        let mut adjacent_limits = vec![count; count];
        let mut next_reserved = count;
        for index in (0..count).rev() {
            adjacent_limits[index] = next_reserved;
            if semantic_members[index] {
                next_reserved = index;
            }
        }
        let mut i = 0;
        while i < count {
            if semantic_members[i] && semantic_ends[i].is_none() {
                i += 1;
                continue;
            }
            let (base, mut ids, epilogues, semantic_fused) =
                if let Some(group) = semantic_ends[i].take() {
                    (
                        group.start,
                        group.inputs,
                        group.epilogues,
                        group.members.len() - 1,
                    )
                } else {
                    (
                        i,
                        compiled.instructions[i].inputs.clone().into_vec(),
                        Vec::new(),
                        0,
                    )
                };
            let is_semantic = base != i;
            let instruction = &compiled.instructions[base];
            let mut end = i;
            let mut kind = Operation::Native {
                index: base,
                kind: logical.steps[base].kind.clone(),
                epilogues,
            };
            if compiled.options.fusion && !is_semantic {
                let adjacent_limit = adjacent_limits[i];
                if let Some((inner, epsilon, gamma, beta)) =
                    layer_norm(compiled, logical, i, adjacent_limit)
                {
                    if closed(compiled, i, i + 8, count, selected) {
                        kind = Operation::LayerNorm { inner, epsilon };
                        ids = vec![instruction.inputs[0], gamma, beta];
                        end = i + 8;
                    }
                } else if let Some((act, last)) =
                    activation(compiled, i, instruction.inputs[0], adjacent_limit)
                {
                    if last > i && closed(compiled, i, last, count, selected) {
                        kind = Operation::Activation(act);
                        ids = vec![instruction.inputs[0]];
                        end = last;
                    }
                } else if matches!(instruction.kind, Op::Conv(_) | Op::ConvTranspose) {
                    let mut epilogues = Vec::new();
                    let mut current = instruction.output;
                    while end + 1 < adjacent_limit {
                        let next = &compiled.instructions[end + 1];
                        if matches!(next.kind, Op::Add)
                            && logical.shapes[next.output.0] == logical.shapes[instruction.output.0]
                            && let Some(other) = other_binary(next, current)
                        {
                            if other == current {
                                break;
                            }
                            let shape = logical.shapes[other.0];
                            let output = logical.shapes[current.0];
                            if (shape == output
                                && (!nhwc
                                    || compiled.model.graph().tensors[other.0].constant.is_none()))
                                || shape.element_count() == 1
                                || (output.rank() == 4
                                    && shape.rank() == 4
                                    && shape.dimensions() == [1, output.dimensions()[1], 1, 1])
                            {
                                if !closed(compiled, i, end + 1, count, selected) {
                                    break;
                                }
                                ids.push(other);
                                epilogues.push(Epilogue::Add(ids.len() - 1));
                                current = next.output;
                                end += 1;
                                continue;
                            }
                        }
                        if let Some((act, last)) =
                            activation(compiled, end + 1, current, adjacent_limit)
                            && closed(compiled, i, last, count, selected)
                        {
                            epilogues.push(Epilogue::Activation(act));
                            end = last;
                            current = compiled.instructions[last].output;
                            continue;
                        }
                        break;
                    }
                    if let Operation::Native {
                        epilogues: ref mut e,
                        ..
                    } = kind
                    {
                        *e = epilogues;
                    }
                }
            }
            fused += if is_semantic { semantic_fused } else { end - i };
            let output_id = compiled.instructions[end].output;
            let output_shape = logical.shapes[output_id.0];
            let first =
                map[ids[0].0].ok_or_else(|| RuntimeError::new("physical input is undefined"))?;
            let capable = match &kind {
                Operation::Native { kind, .. } => match kind {
                    BoundKind::Conv(p) => conv_nhwc_capable(compiled, base, p),
                    BoundKind::ConvTranspose(_)
                    | BoundKind::MaxPool { .. }
                    | BoundKind::AveragePool { .. }
                    | BoundKind::BatchNormalization { .. }
                    | BoundKind::Resize { .. } => true,
                    BoundKind::Concat(p) => {
                        p.axis == 1
                            && output_shape.rank() == 4
                            && ids.iter().all(|id| {
                                logical.shapes[id.0].rank() == 4
                                    && compiled.model.graph().tensors[id.0].constant.is_none()
                            })
                    }
                    BoundKind::Binary(..)
                    | BoundKind::Relu
                    | BoundKind::HardSigmoid(..)
                    | BoundKind::Sigmoid
                    | BoundKind::Erf
                    | BoundKind::Square
                    | BoundKind::Sqrt => {
                        values[first].layout == Layout::Nhwc && output_shape.rank() == 4
                    }
                    BoundKind::Reduce(r) => {
                        values[first].layout == Layout::Nhwc
                            && r.input.rank() == 4
                            && r.contiguous
                                == Some(r.input.dimensions()[2] * r.input.dimensions()[3])
                    }
                    _ => false,
                },
                Operation::Activation(_) => {
                    values[first].layout == Layout::Nhwc && output_shape.rank() == 4
                }
                _ => false,
            };
            let capable = capable && values[first].constant.is_none();
            // Only non-neutral activations continue a segment. Dense group-one
            // convolutions (and transpose) may start one; depthwise/pool/unary
            // operations never start a segment from canonical storage.
            let mut any_nchw = false;
            let mut any_nhwc = false;
            // Fusion-appended residuals must not change the base node's
            // source-layout eligibility or start a segment on its behalf.
            for id in &instruction.inputs {
                let value = &values[map[id.0].expect("physical input is defined")];
                if value.constant.is_some()
                    || value.shape.rank() != 4
                    || layout_neutral(value.shape)
                {
                    continue;
                }
                any_nhwc |= value.layout == Layout::Nhwc;
                any_nchw |= value.layout == Layout::Canonical;
            }
            let starts_segment = matches!(&kind,
                Operation::Native { kind: BoundKind::Conv(p), .. } if p.cpg == p.cin)
                || matches!(
                    &kind,
                    Operation::Native {
                        kind: BoundKind::ConvTranspose(_),
                        ..
                    }
                );
            let mode = nhwc && capable && (any_nhwc || (any_nchw && starts_segment));
            let layout = if mode && output_shape.rank() == 4 && !layout_neutral(output_shape) {
                Layout::Nhwc
            } else {
                Layout::Canonical
            };
            let input_layout = if mode && values[first].shape.rank() == 4 {
                Layout::Nhwc
            } else {
                Layout::Canonical
            };
            let mut inputs = Vec::with_capacity(ids.len());
            for id in ids {
                let value =
                    map[id.0].ok_or_else(|| RuntimeError::new("fusion input is undefined"))?;
                // Constants keep ONNX storage; full-sized runtime residuals follow the sink.
                let desired = if values[value].constant.is_none() {
                    input_layout
                } else {
                    Layout::Canonical
                };
                inputs.push(convert(
                    &mut values,
                    &mut steps,
                    &mut conversions,
                    value,
                    desired,
                ));
            }
            let output = add_value(&mut values, output_shape, layout, None);
            map[output_id.0] = Some(output);
            if matches!(
                &kind,
                Operation::Native {
                    kind: BoundKind::Copy,
                    ..
                }
            ) && values[inputs[0]].constant.is_none()
                && values[inputs[0]].layout == layout
            {
                values[output].alias = Some((inputs[0], 0));
            } else {
                steps.push(Step {
                    inputs,
                    output,
                    op: kind,
                });
            }
            i = end + 1;
        }
        let output =
            map[selected.0].ok_or_else(|| RuntimeError::new("physical output is undefined"))?;
        let output = convert(
            &mut values,
            &mut steps,
            &mut conversions,
            output,
            Layout::Canonical,
        );
        let conversion_count = steps
            .iter()
            .filter(|step| matches!(step.op, Operation::Convert))
            .count();
        let mut plan = Self {
            values,
            steps,
            input,
            output,
            elements: 0,
            fused,
            conversions: conversion_count,
        };
        plan.place()?;
        Ok(plan)
    }
    fn root(&self, mut value: usize) -> (usize, usize) {
        let mut delta = 0;
        while let Some((parent, offset)) = self.values[value].alias {
            value = parent;
            delta += offset;
        }
        (value, delta)
    }
    fn place(&mut self) -> RuntimeResult<()> {
        // A concat with one outer group is a single contiguous allocation.
        // Only exclusive producer values may be assigned disjoint sink regions.
        for index in 0..self.steps.len() {
            let step = &self.steps[index];
            if !matches!(&step.op,Operation::Native{kind:BoundKind::Concat(p),epilogues,..} if p.outer==1 && epilogues.is_empty())
            {
                continue;
            }
            if self.values[step.output].layout != Layout::Canonical {
                continue;
            }
            let legal = step.inputs.iter().all(|&id| {
                id != self.input
                    && self.values[id].constant.is_none()
                    && self.values[id].alias.is_none()
                    && self.steps.iter().filter(|s| s.inputs.contains(&id)).count() == 1
                    && !self
                        .values
                        .iter()
                        .any(|v| v.alias.is_some_and(|(parent, _)| parent == id))
            }) && step
                .inputs
                .iter()
                .enumerate()
                .all(|(i, id)| !step.inputs[..i].contains(id));
            if !legal {
                continue;
            }
            let mut offset = 0;
            for &id in &step.inputs {
                self.values[id].alias = Some((step.output, offset));
                offset += self.values[id].shape.element_count();
            }
            self.steps[index].op = Operation::View;
            self.steps[index].inputs.clear();
        }
        let mut start = vec![usize::MAX; self.values.len()];
        let mut end = vec![0; self.values.len()];
        start[self.input] = 0;
        for (index, step) in self.steps.iter().enumerate() {
            start[step.output] = start[step.output].min(index + 1);
            end[step.output] = end[step.output].max(index + 1);
            for &v in &step.inputs {
                end[v] = end[v].max(index + 1);
            }
        }
        end[self.output] = self.steps.len() + 2;
        self.alias_elementwise(&end);
        // View-only values have no writer; the owning producer establishes
        // their start while every view still extends the group end.
        let mut groups: HashMap<usize, (usize, usize, usize)> = HashMap::new();
        for (v, value) in self.values.iter().enumerate() {
            if value.constant.is_some() {
                continue;
            }
            let (root, delta) = self.root(v);
            let g = groups.entry(root).or_insert((usize::MAX, 0, 0));
            g.0 = g.0.min(start[v]);
            g.1 = g.1.max(end[v]);
            g.2 = g.2.max(
                delta
                    .checked_add(value.shape.element_count())
                    .ok_or_else(|| RuntimeError::new("physical range overflow"))?,
            );
        }
        let mut groups: Vec<_> = groups.into_iter().collect();
        // Place the largest alias groups first, as in the reference planner.
        // Reusing a whole dead block without splitting it strands its unused
        // tail when several smaller tensors are simultaneously live later.
        // Conflicting lifetimes alone reserve address ranges, so those tensors
        // can instead occupy disjoint parts of the same former large region.
        groups.sort_unstable_by(|&(a, (sa, _, la)), &(b, (sb, _, lb))| {
            lb.cmp(&la)
                .then_with(|| sa.cmp(&sb))
                .then_with(|| a.cmp(&b))
        });
        let mut placed: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut conflicts = Vec::new();
        let mut offsets = HashMap::new();
        self.elements = 0;
        for (root, (s, e, len)) in groups {
            conflicts.clear();
            conflicts.extend(
                placed
                    .iter()
                    // A last read and first write at the same step still overlap.
                    .filter(|(_, _, ps, pe)| *ps <= e && s <= *pe)
                    .map(|&(offset, size, _, _)| (offset, size)),
            );
            conflicts.sort_unstable_by_key(|&(offset, _)| offset);
            let mut offset = 0;
            for &(other, size) in &conflicts {
                if other >= offset && other - offset >= len {
                    break;
                }
                let end = other
                    .checked_add(size)
                    .ok_or_else(|| RuntimeError::new("arena size overflow"))?;
                // Align starts, not tensor lengths: a sole three-element
                // tensor must still fit a twelve-byte activation budget.
                let next = end
                    .checked_add(15)
                    .ok_or_else(|| RuntimeError::new("arena alignment overflow"))?
                    / 16
                    * 16;
                offset = offset.max(next);
            }
            let extent = offset
                .checked_add(len)
                .ok_or_else(|| RuntimeError::new("arena size overflow"))?;
            self.elements = self.elements.max(extent);
            placed.push((offset, len, s, e));
            offsets.insert(root, offset);
        }
        for v in 0..self.values.len() {
            if self.values[v].constant.is_none() {
                let (root, delta) = self.root(v);
                self.values[v].offset = offsets[&root] + delta;
            }
        }
        // Validate every simultaneous read/write before any kernel can obtain slices.
        for step in &self.steps {
            let o = &self.values[step.output];
            for &id in &step.inputs {
                let v = &self.values[id];
                if v.constant.is_none()
                    && ranges_overlap(
                        o.offset,
                        o.shape.element_count(),
                        v.offset,
                        v.shape.element_count(),
                    )
                {
                    let exact =
                        o.offset == v.offset && o.shape.element_count() == v.shape.element_count();
                    let operands = if matches!(&step.op,
                        Operation::Native { kind: BoundKind::Binary(..), epilogues, .. }
                            if epilogues.is_empty())
                    {
                        2
                    } else {
                        1
                    };
                    let inplace = step.inputs.iter().take(operands).any(|&source| {
                        self.values[source].constant.is_none()
                            && self.values[source].offset == o.offset
                            && self.values[source].shape == o.shape
                            && self.values[source].layout == o.layout
                    });
                    if !exact || !inplace {
                        return Err(RuntimeError::new(
                            "physical planner produced overlapping input/output",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}
fn ranges_overlap(a: usize, alen: usize, b: usize, blen: usize) -> bool {
    a < b + blen && b < a + alen
}
impl State {
    pub(super) fn new(
        compiled: &CompiledModel,
        logical: &ShapePlan,
        count: usize,
        selected: TensorId,
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        let plan = Arc::new(Plan::bind(compiled, logical, count, selected)?);
        let bytes = plan
            .elements
            .checked_mul(4)
            .ok_or_else(|| RuntimeError::new("arena bytes overflow"))?;
        if bytes > limits.max_activation_bytes {
            return Err(RuntimeError::new(format!(
                "physical activation plan needs {bytes} bytes, limit is {}",
                limits.max_activation_bytes
            )));
        }
        let arena = allocate_buffer(plan.elements)?;
        check_retained_limit(std::slice::from_ref(&arena), limits)?;
        Ok(Self {
            plan,
            arena,
            canonical: Vec::new(),
            workspaces: Vec::new(),
            cache: VecDeque::new(),
            hits: 0,
            misses: 1,
            #[cfg(test)]
            step_timings: None,
        })
    }
    pub(super) fn reshape(
        &mut self,
        compiled: &CompiledModel,
        logical: &ShapePlan,
        count: usize,
        selected: TensorId,
        limits: ExecutionLimits,
    ) -> RuntimeResult<()> {
        let shape = logical.input_shape;
        let cached = self
            .cache
            .iter()
            .position(|p| p.values[p.input].shape == shape);
        let next = if let Some(i) = cached {
            self.hits += 1;
            self.cache.remove(i).unwrap()
        } else {
            self.misses += 1;
            Arc::new(Plan::bind(compiled, logical, count, selected)?)
        };
        let bytes = next
            .elements
            .checked_mul(4)
            .ok_or_else(|| RuntimeError::new("arena size overflow"))?;
        if bytes > limits.max_activation_bytes {
            return Err(RuntimeError::new(
                "physical activation plan exceeds max_activation_bytes",
            ));
        }
        if self.arena.capacity() < next.elements
            || self.retained_bytes() > limits.max_activation_bytes
        {
            let arena = allocate_buffer(next.elements)?;
            check_retained_limit(std::slice::from_ref(&arena), limits)?;
            self.arena = arena;
        } else if self.arena.len() < next.elements {
            // Keep the initialized high-water span across smaller bindings.
            // Every live value is written by fill or its producing step;
            // regrowing a retained shape must not zero the same tail again.
            self.arena.resize(next.elements, 0.);
        }
        let old = std::mem::replace(&mut self.plan, next);
        self.cache.push_front(old);
        self.cache.truncate(8);
        Ok(())
    }
    pub(super) fn fill<F: FnOnce(&mut [f32]) -> RuntimeResult<()>>(
        &mut self,
        fill: F,
        backend: Backend,
        cpu: &CpuContext,
        threads: usize,
    ) -> RuntimeResult<()> {
        let input = &self.plan.values[self.plan.input];
        let len = input.shape.element_count();
        if input.layout == Layout::Canonical {
            let slice = &mut self.arena[input.offset..input.offset + len];
            fill(slice)?;
            finite(slice)?;
        } else {
            if self.canonical.len() < len {
                self.canonical = allocate_buffer(len)?;
            }
            fill(&mut self.canonical[..len])?;
            finite(&self.canonical[..len])?;
            layout_convert(
                &self.canonical[..len],
                &mut self.arena[input.offset..input.offset + len],
                input.shape,
                Layout::Canonical,
                backend,
                cpu,
                threads,
            );
        }
        Ok(())
    }
    pub(super) fn fill_layout<F: FnOnce(&mut [f32], bool) -> RuntimeResult<()>>(
        &mut self,
        fill: F,
    ) -> RuntimeResult<()> {
        let input = &self.plan.values[self.plan.input];
        let slice = &mut self.arena[input.offset..input.offset + input.shape.element_count()];
        fill(slice, input.layout == Layout::Nhwc)?;
        finite(slice)
    }
    pub(super) fn execute(
        &mut self,
        compiled: &CompiledModel,
        cpu: &CpuContext,
        threads: usize,
    ) -> RuntimeResult<()> {
        // Reserve every worker's packing tile before writing any activation.
        // These buffers are request-local and survive reshape and warm runs.
        let mut tile_elements = 0usize;
        for step in &self.plan.steps {
            if step
                .inputs
                .first()
                .is_some_and(|&id| self.plan.values[id].layout == Layout::Nhwc)
                && let Operation::Native {
                    kind: BoundKind::BatchNormalization { channels, .. },
                    ..
                } = &step.op
            {
                tile_elements = tile_elements.max(*channels);
            }
            if step
                .inputs
                .first()
                .is_some_and(|&id| self.plan.values[id].layout == Layout::Nhwc)
                && let Operation::Native {
                    index,
                    kind: BoundKind::ConvTranspose(p),
                    ..
                } = &step.op
                && compiled.transpose_packed.contains_key(index)
            {
                tile_elements =
                    tile_elements.max(p.cout.checked_mul(32).ok_or_else(|| {
                        RuntimeError::new("transposed convolution scratch overflow")
                    })?);
            }
        }
        {
            let workers = threads.clamp(1, 16);
            self.workspaces
                .try_reserve(workers.saturating_sub(self.workspaces.len()))
                .map_err(|e| {
                    RuntimeError::caused_by(
                        crate::error::ErrorKind::Allocation,
                        "packing scratch allocation failed",
                        e,
                    )
                })?;
            while self.workspaces.len() < workers {
                self.workspaces.push(Vec::new());
            }
            for workspace in &mut self.workspaces[..workers] {
                if workspace.len() < tile_elements {
                    *workspace = allocate_buffer(tile_elements)?;
                }
            }
        }
        #[cfg(test)]
        if let Some(timings) = &mut self.step_timings {
            timings.clear();
        }
        for step in &self.plan.steps {
            #[cfg(test)]
            let timer = self
                .step_timings
                .as_ref()
                .map(|_| std::time::Instant::now());
            let o = &self.plan.values[step.output];
            let end = o.offset + o.shape.element_count();
            let (before, tail) = self.arena.split_at_mut(o.offset);
            let (out, after) = tail.split_at_mut(o.shape.element_count());
            let data = |id: usize| -> &[f32] {
                let v = &self.plan.values[id];
                if let Some(id) = v.constant {
                    constant(compiled, id)
                } else if v.offset < o.offset {
                    &before[v.offset..v.offset + v.shape.element_count()]
                } else {
                    &after[v.offset - end..v.offset - end + v.shape.element_count()]
                }
            };
            let operands = if matches!(&step.op,
                Operation::Native { kind: BoundKind::Binary(..), epilogues, .. }
                    if epilogues.is_empty())
            {
                2
            } else {
                1
            };
            if let Some(alias_slot) = step.inputs.iter().take(operands).position(|&id| {
                self.plan.values[id].constant.is_none() && self.plan.values[id].offset == o.offset
            }) {
                match &step.op {
                    Operation::Activation(act) => {
                        simd::parallel::unary_inplace(
                            compiled.backend,
                            activation_op(act),
                            out,
                            cpu,
                            threads,
                        );
                    }
                    Operation::LayerNorm { inner, epsilon } => {
                        let gamma = data(step.inputs[1]);
                        let beta = data(step.inputs[2]);
                        for row in out.chunks_exact_mut(*inner) {
                            let mean = row.iter().copied().fold(0., |a, b| a + b) / *inner as f32;
                            let variance = row
                                .iter()
                                .map(|&x| {
                                    let d = x - mean;
                                    d * d
                                })
                                .fold(0., |a, b| a + b)
                                / *inner as f32;
                            let sd = (variance + epsilon).sqrt();
                            simd::layer_norm_affine(
                                compiled.backend,
                                None,
                                gamma,
                                beta,
                                mean,
                                sd,
                                row,
                            );
                        }
                    }
                    Operation::Native {
                        kind: BoundKind::Binary(plan, op),
                        ..
                    } => {
                        let id = step.inputs[1 - alias_slot];
                        let v = &self.plan.values[id];
                        let same = v.constant.is_none() && v.offset == o.offset;
                        elementwise_alias::binary_inplace(
                            compiled.backend,
                            *op,
                            plan,
                            |slot| {
                                let value = &self.plan.values[step.inputs[slot]];
                                (value.shape, value.layout)
                            },
                            alias_slot,
                            if same { None } else { Some(data(id)) },
                            out,
                            cpu,
                            threads,
                        );
                    }
                    Operation::Native { kind, .. } => {
                        let op = match kind {
                            BoundKind::Square => simd::UnaryOp::Square,
                            BoundKind::Sqrt => simd::UnaryOp::Sqrt,
                            BoundKind::Relu => simd::UnaryOp::Relu,
                            BoundKind::Sigmoid => simd::UnaryOp::Sigmoid,
                            BoundKind::Erf => simd::UnaryOp::Erf,
                            BoundKind::HardSigmoid(alpha, beta) => simd::UnaryOp::HardSigmoid {
                                alpha: *alpha,
                                beta: *beta,
                            },
                            _ => unreachable!("validated exact in-place kernel"),
                        };
                        simd::parallel::unary_inplace(compiled.backend, op, out, cpu, threads);
                    }
                    _ => unreachable!("validated in-place operation"),
                }
                #[cfg(test)]
                if let (Some(timings), Some(timer)) = (&mut self.step_timings, timer) {
                    timings.push(timer.elapsed());
                }
                continue;
            }
            match &step.op {
                Operation::View => {}
                Operation::Convert => layout_convert(
                    data(step.inputs[0]),
                    out,
                    o.shape,
                    self.plan.values[step.inputs[0]].layout,
                    compiled.backend,
                    cpu,
                    threads,
                ),
                Operation::Activation(act) => {
                    simd::parallel::unary(
                        compiled.backend,
                        activation_op(act),
                        data(step.inputs[0]),
                        out,
                        cpu,
                        threads,
                    );
                }
                Operation::LayerNorm { inner, epsilon } => {
                    let x = data(step.inputs[0]);
                    let gamma = data(step.inputs[1]);
                    let beta = data(step.inputs[2]);
                    for (row, dst) in x.chunks_exact(*inner).zip(out.chunks_exact_mut(*inner)) {
                        let mean = row.iter().copied().fold(0., |a, b| a + b) / *inner as f32;
                        let variance = row
                            .iter()
                            .map(|&x| {
                                let d = x - mean;
                                d * d
                            })
                            .fold(0., |a, b| a + b)
                            / *inner as f32;
                        let sd = (variance + epsilon).sqrt();
                        simd::layer_norm_affine(
                            compiled.backend,
                            Some(row),
                            gamma,
                            beta,
                            mean,
                            sd,
                            dst,
                        );
                    }
                }
                Operation::Native {
                    index,
                    kind,
                    epilogues,
                } => {
                    let nhwc = self.plan.values[step.inputs[0]].layout == Layout::Nhwc;
                    let store_eligible = !epilogues.is_empty()
                        && matches!(kind, BoundKind::Conv(_) | BoundKind::ConvTranspose(_));
                    let mut store_fused = false;
                    if store_eligible {
                        let store = store::PreparedStore::new(EpilogueStore {
                            epilogues,
                            inputs: &step.inputs,
                            data: &data,
                            channels: o.shape.dimensions()[1],
                            plane: o.shape.dimensions()[2] * o.shape.dimensions()[3],
                            layout: o.layout,
                            math: simd::UnaryMath::for_backend(compiled.backend),
                        });
                        if nhwc {
                            execute_nhwc(
                                compiled,
                                *index,
                                kind,
                                &step.inputs,
                                &self.plan.values,
                                &data,
                                out,
                                cpu,
                                threads,
                                &mut self.workspaces,
                                &store,
                            )?;
                            store_fused = true;
                        } else {
                            store_fused = execute_native_store(
                                compiled,
                                *index,
                                kind,
                                &step.inputs,
                                &data,
                                out,
                                cpu,
                                threads,
                                &store,
                            );
                        }
                    }
                    if !store_fused && nhwc {
                        execute_nhwc(
                            compiled,
                            *index,
                            kind,
                            &step.inputs,
                            &self.plan.values,
                            &data,
                            out,
                            cpu,
                            threads,
                            &mut self.workspaces,
                            &simd::IdentityStore,
                        )?;
                    } else if !store_fused {
                        execute_native(
                            compiled,
                            *index,
                            kind,
                            &step.inputs,
                            &data,
                            out,
                            cpu,
                            threads,
                        );
                    }
                    for epilogue in if store_fused { &[][..] } else { epilogues } {
                        match epilogue {
                            Epilogue::Activation(act) => {
                                simd::parallel::unary_inplace(
                                    compiled.backend,
                                    activation_op(act),
                                    out,
                                    cpu,
                                    threads,
                                );
                            }
                            Epilogue::Add(slot) => {
                                let b = data(step.inputs[*slot]);
                                let channels = if o.shape.rank() == 4 {
                                    o.shape.dimensions()[1]
                                } else {
                                    o.shape.element_count()
                                };
                                let plane = if o.shape.rank() == 4 {
                                    o.shape.dimensions()[2] * o.shape.dimensions()[3]
                                } else {
                                    1
                                };
                                if b.len() > 1 && b.len() == channels {
                                    simd::binary_channel_inplace(
                                        compiled.backend,
                                        simd::BinaryOp::Add,
                                        b,
                                        simd::ChannelBroadcast {
                                            channels,
                                            plane,
                                            batched: false,
                                            gate_left: false,
                                        },
                                        o.layout == Layout::Nhwc,
                                        out,
                                    );
                                } else {
                                    simd::parallel::binary_inplace(
                                        compiled.backend,
                                        simd::BinaryOp::Add,
                                        Some(b),
                                        false,
                                        out,
                                        cpu,
                                        threads,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            #[cfg(test)]
            if let (Some(timings), Some(timer)) = (&mut self.step_timings, timer) {
                timings.push(timer.elapsed());
            }
        }
        finite(self.output()?.data())
    }
    pub(super) fn output(&self) -> RuntimeResult<TensorView<'_, f32>> {
        let v = &self.plan.values[self.plan.output];
        TensorView::from_slice(
            v.shape,
            &self.arena[v.offset..v.offset + v.shape.element_count()],
        )
        .map_err(Into::into)
    }
    pub(super) fn activation_bytes(&self) -> usize {
        self.plan.elements * 4
    }
    pub(super) fn retained_bytes(&self) -> usize {
        self.arena.capacity() * 4
    }
    pub(super) fn scratch_bytes(&self) -> usize {
        (self.canonical.capacity() + self.workspaces.iter().map(Vec::capacity).sum::<usize>()) * 4
    }
    pub(super) fn planning_bytes(&self) -> usize {
        let bytes = |p: &Plan| {
            p.values.capacity() * size_of::<Value>()
                + p.steps.capacity() * size_of::<Step>()
                + p.steps
                    .iter()
                    .map(|s| {
                        s.inputs.capacity() * size_of::<usize>()
                            + match &s.op {
                                Operation::Native { epilogues, .. } => {
                                    epilogues.capacity() * size_of::<Epilogue>()
                                }
                                _ => 0,
                            }
                    })
                    .sum::<usize>()
        };
        bytes(&self.plan)
            + self.cache.iter().map(|p| bytes(p)).sum::<usize>()
            + self.cache.capacity() * size_of::<Arc<Plan>>()
            + self.workspaces.capacity() * size_of::<Vec<f32>>()
    }
    pub(super) fn statistics(
        &self,
        compiled: &CompiledModel,
        threads: usize,
    ) -> ExecutionStatistics {
        ExecutionStatistics {
            specialized_kernels: specialization_counts(self.plan.steps.iter().map(|step| {
                match &step.op {
                    Operation::Native {
                        index,
                        kind,
                        epilogues,
                    } => {
                        let operator = compiled.model.graph().nodes[*index].op_type();
                        let nhwc = step
                            .inputs
                            .first()
                            .is_some_and(|&id| self.plan.values[id].layout == Layout::Nhwc);
                        let packed = if matches!(kind, BoundKind::Conv(_)) {
                            compiled.conv_packed.contains_key(index)
                        } else if matches!(kind, BoundKind::ConvTranspose(_)) {
                            compiled.transpose_packed.contains_key(index)
                        } else {
                            compiled.instructions[*index]
                                .inputs
                                .get(1)
                                .is_some_and(|id| compiled.packed.contains_key(id))
                        };
                        let (family, isa) = if nhwc {
                            let (family, backend) =
                                super::nhwc::selection(kind, compiled.backend, packed);
                            (family, format!("{backend:?}"))
                        } else {
                            native_selection(
                                kind,
                                compiled.backend,
                                packed,
                                threads,
                                compiled.options.layout == LayoutPolicy::Auto,
                            )
                        };
                        let fused = !epilogues.is_empty()
                            && (nhwc
                                && matches!(
                                    kind,
                                    BoundKind::Conv(_) | BoundKind::ConvTranspose(_)
                                )
                                || !nhwc
                                    && native_store_eligible(
                                        kind,
                                        compiled.backend,
                                        packed,
                                        threads,
                                    ));
                        (operator, family, isa, fused)
                    }
                    Operation::Convert => ("Convert", "transpose_8x8", "Portable4".into(), false),
                    Operation::View => ("View", "alias", "none".into(), false),
                    Operation::LayerNorm { .. } => (
                        "LayerNorm",
                        "sequential_stats_simd_affine",
                        format!("{:?}", compiled.backend),
                        false,
                    ),
                    Operation::Activation(act) => (
                        match act {
                            Activation::Relu => "Relu",
                            Activation::HardSwish => "HardSwish",
                            Activation::Gelu => "Gelu",
                            Activation::Sigmoid => "Sigmoid",
                        },
                        "unary",
                        format!("{:?}", compiled.backend),
                        false,
                    ),
                }
            })),
            selected_kernels: kernel_selections(self.plan.steps.iter().map(|s| {
                let layout = if self.plan.values[s.output].layout == Layout::Nhwc {
                    "nhwc"
                } else {
                    "canonical"
                };
                match &s.op {
                    Operation::Native { index, kind, .. } => (
                        compiled.model.graph().nodes[*index].op_type(),
                        layout,
                        (self.plan.values[s.inputs[0]].layout == Layout::Nhwc
                            && (compiled.conv_packed.contains_key(index)
                                || compiled.transpose_packed.contains_key(index)))
                            || native_consumes_packed(
                                kind,
                                compiled.backend,
                                compiled.instruction_packed(*index).is_some(),
                                threads,
                            ),
                    ),
                    Operation::Convert => ("Convert", layout, false),
                    Operation::View => ("View", layout, false),
                    Operation::LayerNorm { .. } => ("LayerNorm", layout, false),
                    Operation::Activation(a) => (
                        match a {
                            Activation::Relu => "Relu",
                            Activation::Sigmoid => "Sigmoid",
                            Activation::HardSwish => "HardSwish",
                            Activation::Gelu => "Gelu",
                        },
                        layout,
                        false,
                    ),
                }
            })),
            physical_steps: self.plan.steps.len(),
            fused_nodes: self.plan.fused,
            store_fused_steps: self
                .plan
                .steps
                .iter()
                .filter(|s| {
                    let Operation::Native {
                        index,
                        kind,
                        epilogues,
                    } = &s.op
                    else {
                        return false;
                    };
                    if epilogues.is_empty() {
                        return false;
                    }
                    let nhwc = s
                        .inputs
                        .first()
                        .is_some_and(|&input| self.plan.values[input].layout == Layout::Nhwc);
                    nhwc && matches!(kind, BoundKind::Conv(_) | BoundKind::ConvTranspose(_))
                        || !nhwc
                            && native_store_eligible(
                                kind,
                                compiled.backend,
                                compiled.instruction_packed(*index).is_some(),
                                threads,
                            )
                })
                .count(),
            nhwc_values: self
                .plan
                .values
                .iter()
                .filter(|v| v.layout == Layout::Nhwc)
                .count(),
            layout_conversions: self.plan.conversions,
            scratch_bytes: self.scratch_bytes(),
            plan_cache_hits: self.hits,
            plan_cache_misses: self.misses,
            ..Default::default()
        }
    }
}

fn conv_params(p: &kernels::Conv) -> simd::Conv2d {
    simd::Conv2d {
        batch: p.batch,
        input_channels: p.cin,
        output_channels: p.cout,
        height: p.height,
        width: p.width,
        output_height: p.oh,
        output_width: p.ow,
        kernel_height: p.kh,
        kernel_width: p.kw,
        stride_height: p.sh,
        stride_width: p.sw,
        dilation_height: p.dh,
        dilation_width: p.dw,
        pad_top: p.pt,
        pad_left: p.pl,
        groups: p.cin / p.cpg,
    }
}

fn native_store_eligible(kind: &BoundKind, backend: Backend, packed: bool, threads: usize) -> bool {
    match kind {
        BoundKind::Conv(p) => {
            super::nchw::family_for_threads(backend, &conv_params(p), packed, threads).is_some()
        }
        BoundKind::ConvTranspose(_) => backend != Backend::Scalar,
        _ => false,
    }
}

pub(super) fn native_selection(
    kind: &BoundKind,
    backend: Backend,
    packed: bool,
    threads: usize,
    allow_pointwise_matrix: bool,
) -> (&'static str, String) {
    if backend == Backend::Scalar {
        return ("scalar_native", "Scalar".into());
    }
    match kind {
        BoundKind::MatMul {
            rows,
            inner,
            columns,
        } => {
            if allow_pointwise_matrix
                && super::nhwc::pointwise_matrix_eligible(backend, packed, *rows, *inner, *columns)
            {
                return (
                    if backend == Backend::Portable4 {
                        "nhwc_constant_matmul_6x8_pair2"
                    } else {
                        "nhwc_constant_matmul_6x16"
                    },
                    format!("{backend:?}"),
                );
            }
            if simd::matrix::eligible(backend, *rows, *columns, packed) {
                return (
                    "dynamic_matmul_4x32",
                    format!("{:?}", simd::matrix::actual_backend(backend)),
                );
            }
            let wide = backend != Backend::NeonFma
                && simd::wide_projection_eligible(*rows, *inner, *columns, packed);
            let isa = if backend == Backend::Avx512 {
                #[cfg(feature = "parallel")]
                let sharded = threads > 1 && rows * inner * columns >= 262_144 && *rows >= 8;
                #[cfg(not(feature = "parallel"))]
                let sharded = false;
                let (full, last) = if sharded {
                    let chunk = rows.div_ceil(threads).max(4).next_multiple_of(4);
                    let last = rows % chunk;
                    (
                        simd::wide_projection_eligible(chunk, *inner, *columns, packed),
                        if last == 0 {
                            simd::wide_projection_eligible(chunk, *inner, *columns, packed)
                        } else {
                            simd::wide_projection_eligible(last, *inner, *columns, packed)
                        },
                    )
                } else {
                    (wide, wide)
                };
                match (full, last) {
                    (true, true) => "Avx512",
                    (false, false) => "Avx2Fma",
                    _ => "Avx512+Avx2Fma",
                }
                .into()
            } else {
                format!("{backend:?}")
            };
            (
                if wide {
                    "packed_projection_wide"
                } else {
                    "matmul_4x16"
                },
                isa,
            )
        }
        BoundKind::BatchedMatMul(p) => (
            if simd::matrix::eligible(backend, p.rows, p.columns, false) {
                "dynamic_matmul_4x32"
            } else {
                "dynamic_matmul_4x16"
            },
            format!("{:?}", simd::matrix::actual_backend(backend)),
        ),
        BoundKind::Conv(p) => {
            let params = conv_params(p);
            if let Some(family) = super::nchw::family_for_threads(backend, &params, packed, threads)
            {
                return (
                    family,
                    format!(
                        "{:?}",
                        super::nchw::selected_backend_for_threads(
                            backend, &params, packed, threads
                        )
                    ),
                );
            }
            if params.uses_original_convolution(backend) {
                (
                    if params.is_direct_pointwise() {
                        "nchw_scalar_pointwise"
                    } else {
                        "nchw_scalar_narrow"
                    },
                    "Scalar".into(),
                )
            } else {
                ("nchw_spatial_4oc", format!("{backend:?}"))
            }
        }
        BoundKind::ConvTranspose(_) => (
            "nchw_transpose2x2_spatial",
            format!(
                "{:?}",
                if backend == Backend::NeonFma {
                    Backend::Portable4
                } else {
                    backend
                }
            ),
        ),
        BoundKind::MaxPool { .. } | BoundKind::AveragePool { .. } => {
            ("nchw_spatial_pool", format!("{backend:?}"))
        }
        BoundKind::Reduce(p)
            if p.input.rank() == 4
                && p.contiguous == Some(p.input.dimensions()[2] * p.input.dimensions()[3]) =>
        {
            ("nchw_spatial_mean", format!("{backend:?}"))
        }
        BoundKind::Reduce(_) => ("canonical_sequential_mean", "Scalar".into()),
        _ => ("canonical_native", format!("{backend:?}")),
    }
}

pub(super) fn native_consumes_packed(
    kind: &BoundKind,
    backend: Backend,
    prepared: bool,
    threads: usize,
) -> bool {
    if !prepared || backend == Backend::Scalar {
        return false;
    }
    match kind {
        BoundKind::MatMul { .. } => true,
        BoundKind::Conv(p) => {
            super::nchw::uses_pack_for_threads(backend, &conv_params(p), true, threads)
        }
        _ => false,
    }
}
fn finite(values: &[f32]) -> RuntimeResult<()> {
    if values.iter().any(|v| !v.is_finite()) {
        Err(RuntimeError::new("nonfinite output F32"))
    } else {
        Ok(())
    }
}
fn activation_op(act: &Activation) -> simd::UnaryOp {
    match act {
        Activation::Relu => simd::UnaryOp::Relu,
        Activation::HardSwish => simd::UnaryOp::HardSwish,
        Activation::Gelu => simd::UnaryOp::Gelu,
        Activation::Sigmoid => simd::UnaryOp::Sigmoid,
    }
}
fn binary_op(op: &BinaryOp) -> simd::BinaryOp {
    match op {
        BinaryOp::Add => simd::BinaryOp::Add,
        BinaryOp::Mul => simd::BinaryOp::Mul,
        BinaryOp::Div => simd::BinaryOp::Div,
        BinaryOp::Sub => simd::BinaryOp::Sub,
    }
}
fn layout_convert(
    input: &[f32],
    output: &mut [f32],
    shape: Shape,
    source: Layout,
    backend: Backend,
    cpu: &CpuContext,
    threads: usize,
) {
    layout::convert(input, output, shape, source, backend, cpu, threads);
}

#[allow(clippy::too_many_arguments)]
fn execute_native<'a>(
    compiled: &CompiledModel,
    index: usize,
    kind: &BoundKind,
    inputs: &[usize],
    data: &impl Fn(usize) -> &'a [f32],
    out: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
) {
    if compiled.options.layout == LayoutPolicy::Auto
        && let BoundKind::MatMul {
            rows,
            inner,
            columns,
        } = kind
        && let Some(packed) = compiled.instruction_packed(index)
        && super::nhwc::try_pointwise_matrix(
            compiled.backend,
            data(inputs[0]),
            packed,
            out,
            *rows,
            *inner,
            *columns,
            cpu,
            threads,
        )
    {
        return;
    }
    let logical = &compiled.instructions[index].inputs;
    kernels::execute_optimized(
        kind,
        logical,
        |id| {
            data(
                inputs[logical
                    .iter()
                    .position(|&x| x == id)
                    .expect("physical operand")],
            )
        },
        out,
        compiled.backend,
        compiled.instruction_packed(index),
        cpu,
        threads,
    );
}

#[allow(clippy::too_many_arguments)]
fn execute_native_store<'a>(
    compiled: &CompiledModel,
    index: usize,
    kind: &BoundKind,
    inputs: &[usize],
    data: &(impl Fn(usize) -> &'a [f32] + Sync),
    out: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
    store: &impl simd::StoreTransform,
) -> bool {
    let input = data(inputs[0]);
    let weights = data(inputs[1]);
    // Appended physical operands belong to epilogues. Bias is present only
    // when the original logical convolution has a third operand.
    let bias = if compiled.instructions[index].inputs.len() > 2 {
        data(inputs[2])
    } else {
        &[]
    };
    match kind {
        BoundKind::Conv(p) => super::nchw::try_conv_store(
            compiled.backend,
            &conv_params(p),
            input,
            weights,
            bias,
            compiled.instruction_packed(index),
            out,
            cpu,
            threads,
            store,
        ),
        BoundKind::ConvTranspose(p) => super::nchw::try_transpose_store(
            compiled.backend,
            p,
            input,
            weights,
            bias,
            out,
            cpu,
            threads,
            store,
        ),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_nhwc<'a>(
    compiled: &CompiledModel,
    index: usize,
    kind: &BoundKind,
    inputs: &[usize],
    values: &[Value],
    data: &(impl Fn(usize) -> &'a [f32] + Sync),
    out: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
    workspaces: &mut [Vec<f32>],
    store: &impl simd::StoreTransform,
) -> RuntimeResult<()> {
    super::nhwc::execute(
        compiled,
        index,
        kind,
        |slot| (values[inputs[slot]].shape, values[inputs[slot]].layout),
        |slot| data(inputs[slot]),
        out,
        cpu,
        threads,
        workspaces,
        store,
    )
}

#[cfg(test)]
mod tests;
