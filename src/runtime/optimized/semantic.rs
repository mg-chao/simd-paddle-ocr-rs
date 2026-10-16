//! Conv epilogues selected by tensor consumers rather than source adjacency.
//!
//! A group executes at its last member. Every earlier member's output has one
//! use, the following member, so delaying that work cannot hide a value needed
//! by an intervening instruction. Unrelated instructions keep their positions.
use super::*;

pub(super) struct SemanticGroup {
    pub start: usize,
    pub end: usize,
    /// Original instruction indices, in execution/dataflow order. Gap nodes
    /// are deliberately absent and remain ordinary physical instructions.
    pub members: Vec<usize>,
    /// Base Conv operands followed by each Add's external operand.
    pub inputs: Vec<TensorId>,
    pub epilogues: Vec<Epilogue>,
}

/// Return nonoverlapping groups containing at least one unrelated source node.
/// Adjacent groups stay with the existing activation/LayerNorm matcher.
pub(super) fn groups(
    compiled: &CompiledModel,
    logical: &ShapePlan,
    count: usize,
    selected: TensorId,
) -> Vec<SemanticGroup> {
    if !compiled.options.fusion
        || count > compiled.instructions.len()
        || count > logical.steps.len()
    {
        return Vec::new();
    }
    let nodes = &compiled.instructions[..count];
    let tensors = logical.shapes.len();
    let mut consumer = vec![None; tensors];
    let mut repeated = vec![false; tensors];
    let mut producer = vec![None; tensors];
    for (index, node) in nodes.iter().enumerate() {
        producer[node.output.0] = Some(index);
        for &input in &node.inputs {
            if consumer[input.0].replace(index).is_some() {
                // Two uses in one Add are a self-add, and also prevent fusion.
                repeated[input.0] = true;
            }
        }
    }
    let mut occupied = vec![false; count];
    let mut result = Vec::new();
    for (start, base) in nodes.iter().enumerate() {
        if occupied[start] || !matches!(base.kind, Op::Conv(_) | Op::ConvTranspose) {
            continue;
        }
        let output_shape = logical.shapes[base.output.0];
        let mut group = SemanticGroup {
            start,
            end: start,
            members: vec![start],
            inputs: base.inputs.to_vec(),
            epilogues: Vec::new(),
        };
        let mut current = base.output;
        while current != selected && !repeated[current.0] {
            let Some(next) = consumer[current.0] else {
                break;
            };
            if next <= group.end || occupied[next] {
                break;
            }
            let node = &nodes[next];
            if logical.shapes[node.output.0] != output_shape {
                // In particular, a leading singleton constant can otherwise
                // promote a four-dimensional activation to rank five.
                break;
            }
            let epilogue = match &node.kind {
                Op::Add if node.inputs.len() == 2 => {
                    let Some(other) = other_binary(node, current) else {
                        break;
                    };
                    if other == current
                        || producer[other.0]
                            .is_some_and(|p| p >= next || group.members.contains(&p))
                        || !compatible_add(compiled, logical, other, output_shape)
                    {
                        break;
                    }
                    group.inputs.push(other);
                    Epilogue::Add(group.inputs.len() - 1)
                }
                Op::Relu if node.inputs.as_ref() == [current] => {
                    Epilogue::Activation(Activation::Relu)
                }
                Op::Sigmoid if node.inputs.as_ref() == [current] => {
                    Epilogue::Activation(Activation::Sigmoid)
                }
                _ => break,
            };
            group.epilogues.push(epilogue);
            group.members.push(next);
            group.end = next;
            current = node.output;
        }
        if group.members.windows(2).any(|pair| pair[1] > pair[0] + 1) {
            for &member in &group.members {
                occupied[member] = true;
            }
            result.push(group);
        }
    }
    result
}

fn compatible_add(
    compiled: &CompiledModel,
    logical: &ShapePlan,
    other: TensorId,
    output: Shape,
) -> bool {
    let shape = logical.shapes[other.0];
    // Scalar and [1,C,1,1] broadcasts have the same meaning in either layout.
    // Full-sized constants retain canonical storage, so a NHWC epilogue must
    // leave them to the normal broadcast/layout path instead of reading b[i].
    shape.element_count() == 1
        || (output.rank() == 4
            && shape.rank() == 4
            && shape.dimensions() == [1, output.dimensions()[1], 1, 1])
        || (shape == output
            && (compiled.options.layout == LayoutPolicy::Nchw
                || compiled.model.graph().tensors[other.0].constant.is_none()))
}
