//! Dataflow matches: every internal value must be private to the fused group.
use super::*;
use crate::runtime::{Op, graph_patterns as patterns};
use operators::{Recipe, RecipeKind, ShaderKind};

pub(super) fn replacement(
    program: &Program,
    plan: &ShapePlan,
    start: usize,
    selected: TensorId,
    limits: &wgpu::Limits,
) -> RuntimeResult<Option<Recipe>> {
    let n = &program.instructions;
    let count = plan.steps.len();
    let next = |id| program.graph_uses.unique_consumer(id, count);
    let mut members = vec![start];
    let mut inputs = n[start].inputs.to_vec();
    let mut words = [0; 128];
    let kind;
    let groups;
    let end;
    match &plan.steps[start].kind {
        BoundKind::BatchedMatMul(_) => {
            let Some(mut soft) = next(n[start].output) else {
                return Ok(None);
            };
            let mut scale = 1.0;
            if matches!(n[soft].kind, Op::Mul) {
                let Some(value) = patterns::other_binary(&n[soft], n[start].output)
                    .and_then(|id| patterns::scalar(program, id))
                else {
                    return Ok(None);
                };
                scale = value;
                members.push(soft);
                let Some(s) = next(n[soft].output) else {
                    return Ok(None);
                };
                soft = s;
            }
            if !matches!(plan.steps[soft].kind, BoundKind::Softmax { .. }) {
                return Ok(None);
            };
            let Some(last) = next(n[soft].output) else {
                return Ok(None);
            };
            if !matches!(plan.steps[last].kind, BoundKind::BatchedMatMul(_))
                || n[last].inputs[0] != n[soft].output
            {
                return Ok(None);
            };
            let q = plan.shapes[inputs[0].0];
            let k = plan.shapes[inputs[1].0];
            let v = plan.shapes[n[last].inputs[1].0];
            let rank = q.rank();
            if rank < 3 || k.rank() != rank || v.rank() != rank {
                return Ok(None);
            };
            let q = q.dimensions();
            let k = k.dimensions();
            let v = v.dimensions();
            let d = q[rank - 1];
            let rows = q[rank - 2];
            let keys = k[rank - 1];
            let dv = v[rank - 1];
            if d > 32
                || dv > 32
                || k[rank - 2] != d
                || v[rank - 2] != keys
                || q[..rank - 2] != k[..rank - 2]
                || q[..rank - 2] != v[..rank - 2]
            {
                return Ok(None);
            };
            members.extend([soft, last]);
            inputs.push(n[last].inputs[1]);
            end = last;
            let batches: usize = q[..rank - 2].iter().product();
            words[3] = rows as u32;
            words[4] = keys as u32;
            words[5] = d as u32;
            words[6] = dv as u32;
            words[7] = scale.to_bits();
            words[8] = rows.div_ceil(4) as u32;
            groups = batches * rows.div_ceil(4);
            kind = ShaderKind::Attention;
        }
        BoundKind::Resize {
            height,
            width,
            factor,
        } => {
            let Some(last) = next(n[start].output) else {
                return Ok(None);
            };
            if !matches!(n[last].kind, Op::Add) {
                return Ok(None);
            };
            let Some(other) = patterns::other_binary(&n[last], n[start].output) else {
                return Ok(None);
            };
            if plan.shapes[other.0] != plan.shapes[n[start].output.0] {
                return Ok(None);
            };
            inputs.push(other);
            members.push(last);
            end = last;
            words[2] = 23;
            words[3] = *height as u32;
            words[4] = *width as u32;
            words[5] = *factor as u32;
            words[0] = plan.shapes[n[last].output.0].element_count() as u32;
            groups = (words[0] as usize).div_ceil(64);
            kind = ShaderKind::General;
        }
        BoundKind::Conv(g)
            if g.kh == 1
                && g.kw == 1
                && g.height == 1
                && g.width == 1
                && g.cpg == g.cin
                && g.oh == 1
                && g.ow == 1 =>
        {
            let Some(act) = next(n[start].output) else {
                return Ok(None);
            };
            match n[act].kind {
                Op::Sigmoid => words[8] = 0,
                Op::HardSigmoid { alpha, beta } => {
                    words[8] = 1;
                    words[9] = alpha.to_bits();
                    words[10] = beta.to_bits();
                }
                _ => return Ok(None),
            }
            let Some(last) = next(n[act].output) else {
                return Ok(None);
            };
            if !matches!(n[last].kind, Op::Mul) {
                return Ok(None);
            };
            let Some(feature) = patterns::other_binary(&n[last], n[act].output) else {
                return Ok(None);
            };
            let shape = plan.shapes[feature.0];
            if shape.rank() != 4
                || shape.dimensions()[..2] != [g.batch, g.cout]
                || plan.shapes[n[last].output.0] != shape
            {
                return Ok(None);
            };
            words[3] = g.cin as u32;
            words[4] = g.cout as u32;
            words[5] = (shape.element_count() / (g.batch * g.cout)) as u32;
            words[7] = u32::from(inputs.len() == 3);
            while inputs.len() < 3 {
                inputs.push(inputs[0]);
            }
            inputs.push(feature);
            groups = g.batch * g.cout;
            members.extend([act, last]);
            end = last;
            kind = ShaderKind::SqueezeExcite;
        }
        _ => return Ok(None),
    }
    let output = n[end].output;
    if !patterns::closed_group(program, &members, output, selected, count) {
        return Ok(None);
    };
    let (x, y) = dispatch_grid(groups * 64, limits)?;
    words[1] = if kind == ShaderKind::General {
        x * 64
    } else {
        x
    };
    if kind != ShaderKind::General {
        words[0] = groups as u32;
    }
    Ok(Some(Recipe {
        node: end,
        nodes: members,
        kind: RecipeKind::Compute(kind),
        inputs,
        output,
        words,
        x,
        y,
    }))
}
