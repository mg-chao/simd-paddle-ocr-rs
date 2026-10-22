//! Backend-independent semantic pattern matching.
use super::kernels::BoundKind;
use super::*;
#[derive(Clone, Copy)]
pub(super) enum Activation {
    Relu,
    HardSwish,
    Gelu,
    Sigmoid,
}
pub(super) fn scalar(program: &Program, id: TensorId) -> Option<f32> {
    match &program.model.graph().tensors[id.0].constant {
        Some(TensorData::F32(t)) if t.data().len() == 1 => Some(t.data()[0]),
        _ => None,
    }
}
pub(super) fn other_binary(instruction: &Instruction, value: TensorId) -> Option<TensorId> {
    if instruction.inputs.first() == Some(&value) {
        instruction.inputs.get(1).copied()
    } else if instruction.inputs.get(1) == Some(&value) {
        instruction.inputs.first().copied()
    } else {
        None
    }
}
pub(super) fn closed(
    program: &Program,
    first: usize,
    end: usize,
    count: usize,
    selected: TensorId,
) -> bool {
    for instruction in &program.instructions[first..end] {
        if instruction.output == selected {
            return false;
        }
        // Compilation validates source order: a produced tensor cannot have a
        // consumer before this group. Only later uses inside this execution
        // prefix can require an intermediate to remain materialized.
        if program
            .graph_uses
            .used_after(instruction.output, end, count)
        {
            return false;
        }
    }
    true
}
pub(super) fn activation(
    program: &Program,
    start: usize,
    input: TensorId,
    count: usize,
) -> Option<(Activation, usize)> {
    let n = &program.instructions;
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
        && scalar(program, n[start].inputs[1]) == Some(std::f32::consts::SQRT_2)
        && matches!(n[start + 1].kind, Op::Erf)
        && n[start + 1].inputs[0] == n[start].output
        && matches!(n[start + 2].kind, Op::Add)
        && other_binary(&n[start + 2], n[start + 1].output).and_then(|id| scalar(program, id))
            == Some(1.)
        && matches!(n[start + 3].kind, Op::Mul)
        && other_binary(&n[start + 3], n[start + 2].output) == Some(input)
        && matches!(n[start + 4].kind, Op::Mul)
        && other_binary(&n[start + 4], n[start + 3].output).and_then(|id| scalar(program, id))
            == Some(0.5)
    {
        return Some((Activation::Gelu, start + 4));
    }
    None
}
pub(super) fn layer_norm(
    program: &Program,
    plan: &ShapePlan,
    start: usize,
    count: usize,
) -> Option<(usize, f32, TensorId, TensorId)> {
    if start + 8 >= count {
        return None;
    }
    let n = &program.instructions[start..start + 9];
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
    let epsilon = scalar(program, other_binary(&n[4], n[3].output)?)?;
    let gamma = other_binary(&n[7], n[6].output)?;
    let beta = other_binary(&n[8], n[7].output)?;
    for id in [gamma, beta] {
        match &program.model.graph().tensors[id.0].constant {
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

#[cfg(feature = "gpu")]
pub(super) fn closed_group(
    program: &Program,
    members: &[usize],
    output: TensorId,
    selected: TensorId,
    count: usize,
) -> bool {
    members.iter().all(|&node| {
        let value = program.instructions[node].output;
        value == output
            || (value != selected
                && program.instructions[..count]
                    .iter()
                    .enumerate()
                    .all(|(consumer, op)| {
                        !op.inputs.contains(&value) || members.contains(&consumer)
                    }))
    })
}
