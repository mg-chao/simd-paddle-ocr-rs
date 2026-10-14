use super::{RuntimeError, RuntimeResult, shapes};
use crate::{
    model::{ConvAttributes, TensorId},
    tensor::{MAX_RANK, Shape},
};

pub(super) struct BoundStep {
    pub a: TensorId,
    pub b: Option<TensorId>,
    pub bias: Option<TensorId>,
    pub output: TensorId,
    pub kind: BoundKind,
}
pub(super) enum BoundKind {
    Binary(Broadcast, bool),
    Relu,
    HardSigmoid(f32, f32),
    Conv(Conv),
    Reduce(Reduction),
    Copy,
    MatMul {
        rows: usize,
        inner: usize,
        columns: usize,
    },
    Softmax {
        rows: usize,
        columns: usize,
    },
}
pub(super) fn execute(
    step: &BoundStep,
    a: &[f32],
    b: Option<&[f32]>,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    match &step.kind {
        BoundKind::Binary(plan, multiply) => binary(plan, a, b.unwrap(), out, *multiply),
        BoundKind::Relu => {
            for (o, &x) in out.iter_mut().zip(a) {
                *o = if x > 0.0 { x } else { 0.0 };
            }
        }
        BoundKind::HardSigmoid(alpha, beta) => {
            for (o, &x) in out.iter_mut().zip(a) {
                *o = (alpha * x + beta).clamp(0.0, 1.0);
            }
        }
        BoundKind::Conv(plan) => conv(plan, a, b.unwrap(), bias.unwrap_or(&[]), out),
        BoundKind::Reduce(plan) => reduce(plan, a, out),
        BoundKind::Copy => out.copy_from_slice(a),
        BoundKind::MatMul {
            rows,
            inner,
            columns,
        } => {
            for row in 0..*rows {
                let input = &a[row * inner..(row + 1) * inner];
                for (column, value) in out[row * columns..(row + 1) * columns]
                    .iter_mut()
                    .enumerate()
                {
                    let mut sum = 0.0f32;
                    for (k, &x) in input.iter().enumerate() {
                        sum += x * b.unwrap()[k * columns + column];
                    }
                    *value = sum;
                }
            }
        }
        BoundKind::Softmax { rows, columns } => {
            for row in 0..*rows {
                let input = &a[row * columns..(row + 1) * columns];
                let output = &mut out[row * columns..(row + 1) * columns];
                let mut max = input[0];
                for &value in &input[1..] {
                    max = max.max(value);
                }
                let mut sum = 0.0f32;
                for (&x, o) in input.iter().zip(output.iter_mut()) {
                    let value = (x - max).exp();
                    *o = value;
                    sum += value;
                }
                for value in output {
                    *value /= sum;
                }
            }
        }
    }
}

enum BroadcastKind {
    Full,
    RightScalar,
    LeftScalar,
    Channel {
        scalar_left: bool,
        channels: usize,
        plane: usize,
        batched: bool,
    },
    LastAxis {
        scalar_left: bool,
        columns: usize,
    },
    Generic,
}
pub(super) struct Broadcast {
    kind: BroadcastKind,
    rank: usize,
    dimensions: [usize; MAX_RANK],
    a_strides: [usize; MAX_RANK],
    b_strides: [usize; MAX_RANK],
}
impl Broadcast {
    pub(super) fn new(a: Shape, b: Shape, out: Shape) -> Self {
        let channel = |full: Shape, gate: Shape, scalar_left| {
            let fd = full.dimensions();
            let gd = gate.dimensions();
            if full == out
                && fd.len() == 4
                && gd.len() == 4
                && (gd[0] == 1 || gd[0] == fd[0])
                && gd[1] == fd[1]
                && gd[2..] == [1, 1]
            {
                Some(BroadcastKind::Channel {
                    scalar_left,
                    channels: fd[1],
                    plane: fd[2] * fd[3],
                    batched: gd[0] != 1,
                })
            } else {
                None
            }
        };
        let last = |full: Shape, gate: Shape, scalar_left| {
            if full == out
                && gate.rank() == 1
                && full.rank() > 1
                && gate.dimensions()[0] == full.dimensions()[full.rank() - 1]
            {
                Some(BroadcastKind::LastAxis {
                    scalar_left,
                    columns: gate.dimensions()[0],
                })
            } else {
                None
            }
        };
        let kind = if b.element_count() == 1 {
            BroadcastKind::RightScalar
        } else if a.element_count() == 1 {
            BroadcastKind::LeftScalar
        } else if a.element_count() == out.element_count()
            && b.element_count() == out.element_count()
        {
            BroadcastKind::Full
        } else if let Some(kind) = channel(a, b, false)
            .or_else(|| channel(b, a, true))
            .or_else(|| last(a, b, false))
            .or_else(|| last(b, a, true))
        {
            kind
        } else {
            BroadcastKind::Generic
        };
        let mut dimensions = [1; MAX_RANK];
        dimensions[..out.rank()].copy_from_slice(out.dimensions());
        Self {
            kind,
            rank: out.rank(),
            dimensions,
            a_strides: aligned_strides(a, out.rank()),
            b_strides: aligned_strides(b, out.rank()),
        }
    }
}
fn aligned_strides(shape: Shape, rank: usize) -> [usize; MAX_RANK] {
    let mut strides = [0; MAX_RANK];
    let mut stride = 1;
    for axis in (0..shape.rank()).rev() {
        if shape.dimensions()[axis] != 1 {
            strides[rank - shape.rank() + axis] = stride;
        }
        stride *= shape.dimensions()[axis];
    }
    strides
}
#[inline]
fn arithmetic(a: f32, b: f32, multiply: bool) -> f32 {
    if multiply { a * b } else { a + b }
}
fn binary(plan: &Broadcast, a: &[f32], b: &[f32], out: &mut [f32], multiply: bool) {
    match plan.kind {
        BroadcastKind::Full => {
            for ((o, &a), &b) in out.iter_mut().zip(a).zip(b) {
                *o = arithmetic(a, b, multiply);
            }
        }
        BroadcastKind::RightScalar => {
            for (o, &a) in out.iter_mut().zip(a) {
                *o = arithmetic(a, b[0], multiply);
            }
        }
        BroadcastKind::LeftScalar => {
            for (o, &b) in out.iter_mut().zip(b) {
                *o = arithmetic(a[0], b, multiply);
            }
        }
        BroadcastKind::Channel {
            scalar_left,
            channels,
            plane,
            batched,
        } => {
            let (full, gate) = if scalar_left { (b, a) } else { (a, b) };
            for (block, (input, output)) in full
                .chunks_exact(plane)
                .zip(out.chunks_exact_mut(plane))
                .enumerate()
            {
                let scalar = gate[if batched { block } else { block % channels }];
                for (&value, output) in input.iter().zip(output) {
                    *output = if scalar_left {
                        arithmetic(scalar, value, multiply)
                    } else {
                        arithmetic(value, scalar, multiply)
                    };
                }
            }
        }
        BroadcastKind::LastAxis {
            scalar_left,
            columns,
        } => {
            let (full, gate) = if scalar_left { (b, a) } else { (a, b) };
            for (input, output) in full
                .chunks_exact(columns)
                .zip(out.chunks_exact_mut(columns))
            {
                for ((&value, &scalar), output) in input.iter().zip(gate).zip(output) {
                    *output = if scalar_left {
                        arithmetic(scalar, value, multiply)
                    } else {
                        arithmetic(value, scalar, multiply)
                    };
                }
            }
        }
        BroadcastKind::Generic => {
            let mut counters = [0; MAX_RANK];
            let mut ai = 0;
            let mut bi = 0;
            for o in out {
                *o = arithmetic(a[ai], b[bi], multiply);
                for axis in (0..plan.rank).rev() {
                    counters[axis] += 1;
                    if counters[axis] < plan.dimensions[axis] {
                        ai += plan.a_strides[axis];
                        bi += plan.b_strides[axis];
                        break;
                    }
                    ai -= (counters[axis] - 1) * plan.a_strides[axis];
                    bi -= (counters[axis] - 1) * plan.b_strides[axis];
                    counters[axis] = 0;
                }
            }
        }
    }
}

pub(super) struct Conv {
    batch: usize,
    cin: usize,
    cout: usize,
    height: usize,
    width: usize,
    oh: usize,
    ow: usize,
    cpg: usize,
    opg: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
    pt: usize,
    pl: usize,
}
pub(super) fn bind_conv(
    a: Shape,
    weights: Shape,
    bias: Option<Shape>,
    attrs: &ConvAttributes,
) -> RuntimeResult<(Shape, BoundKind)> {
    if a.rank() != 4 || weights.rank() != 4 {
        return Err(RuntimeError::new(
            "Conv requires rank-four input and weights",
        ));
    }
    let id = a.dimensions();
    let wd = weights.dimensions();
    let positive = |v: i64| {
        usize::try_from(v)
            .ok()
            .filter(|&v| v > 0)
            .ok_or_else(|| RuntimeError::new("Conv group/kernel/stride/dilation must be positive"))
    };
    let pad = |v: i64| {
        usize::try_from(v).map_err(|_| RuntimeError::new("Conv padding must be nonnegative"))
    };
    let group = positive(attrs.group)?;
    let kh = positive(attrs.kernel_shape[0])?;
    let kw = positive(attrs.kernel_shape[1])?;
    let sh = positive(attrs.strides[0])?;
    let sw = positive(attrs.strides[1])?;
    let dh = positive(attrs.dilations[0])?;
    let dw = positive(attrs.dilations[1])?;
    let pt = pad(attrs.pads[0])?;
    let pl = pad(attrs.pads[1])?;
    let pb = pad(attrs.pads[2])?;
    let pr = pad(attrs.pads[3])?;
    if !id[1].is_multiple_of(group)
        || !wd[0].is_multiple_of(group)
        || wd[1] != id[1] / group
        || wd[2] != kh
        || wd[3] != kw
    {
        return Err(RuntimeError::new(
            "Conv channels/groups/kernel do not match weights",
        ));
    }
    if bias.is_some_and(|shape| shape.dimensions() != [wd[0]]) {
        return Err(RuntimeError::new(
            "Conv bias must have shape [output_channels]",
        ));
    }
    let extent = |input: usize,
                  before: usize,
                  after: usize,
                  kernel: usize,
                  dilation: usize,
                  stride: usize| {
        let padded = input
            .checked_add(before)
            .and_then(|v| v.checked_add(after))
            .filter(|&v| v <= isize::MAX as usize)
            .ok_or_else(|| RuntimeError::new("Conv padded extent overflow"))?;
        let effective = (kernel - 1)
            .checked_mul(dilation)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| RuntimeError::new("Conv effective kernel overflow"))?;
        padded
            .checked_sub(effective)
            .map(|v| v / stride + 1)
            .ok_or_else(|| RuntimeError::new("Conv kernel exceeds padded input"))
    };
    let oh = extent(id[2], pt, pb, kh, dh, sh)?;
    let ow = extent(id[3], pl, pr, kw, dw, sw)?;
    let output = Shape::new(&[id[0], wd[0], oh, ow])?;
    Ok((
        output,
        BoundKind::Conv(Conv {
            batch: id[0],
            cin: id[1],
            cout: wd[0],
            height: id[2],
            width: id[3],
            oh,
            ow,
            cpg: id[1] / group,
            opg: wd[0] / group,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw,
            pt,
            pl,
        }),
    ))
}
fn conv(p: &Conv, input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32]) {
    if p.kh == 1
        && p.kw == 1
        && p.sh == 1
        && p.sw == 1
        && p.pt == 0
        && p.pl == 0
        && p.oh == p.height
        && p.ow == p.width
    {
        conv1x1(p, input, weights, bias, output);
        return;
    }
    if p.cpg == 1 && p.cin == p.cout && p.opg == 1 {
        depthwise(p, input, weights, bias, output);
        return;
    }
    let plane = p.height * p.width;
    let out_plane = p.oh * p.ow;
    for batch in 0..p.batch {
        for co in 0..p.cout {
            for y in 0..p.oh {
                for x in 0..p.ow {
                    let mut sum = if bias.is_empty() { 0.0 } else { bias[co] };
                    let group = co / p.opg;
                    for ci in 0..p.cpg {
                        for ky in 0..p.kh {
                            let iy = y * p.sh + ky * p.dh;
                            if iy < p.pt || iy - p.pt >= p.height {
                                continue;
                            }
                            let input_row = (batch * p.cin + group * p.cpg + ci) * plane
                                + (iy - p.pt) * p.width;
                            let weight_row = ((co * p.cpg + ci) * p.kh + ky) * p.kw;
                            for kx in 0..p.kw {
                                let ix = x * p.sw + kx * p.dw;
                                if ix >= p.pl && ix - p.pl < p.width {
                                    sum += input[input_row + ix - p.pl] * weights[weight_row + kx];
                                }
                            }
                        }
                    }
                    output[(batch * p.cout + co) * out_plane + y * p.ow + x] = sum;
                }
            }
        }
    }
}
fn conv1x1(p: &Conv, input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32]) {
    let plane = p.height * p.width;
    for batch in 0..p.batch {
        for group in 0..p.cout / p.opg {
            let input_base = (batch * p.cin + group * p.cpg) * plane;
            let first_channel = group * p.opg;
            let tiled = p.opg / 4 * 4;
            for channel in (0..tiled).step_by(4) {
                let co = first_channel + channel;
                let (w0, rest) = weights[co * p.cpg..(co + 4) * p.cpg].split_at(p.cpg);
                let (w1, rest) = rest.split_at(p.cpg);
                let (w2, w3) = rest.split_at(p.cpg);
                let base = (batch * p.cout + co) * plane;
                let (o0, rest) = output[base..base + 4 * plane].split_at_mut(plane);
                let (o1, rest) = rest.split_at_mut(plane);
                let (o2, o3) = rest.split_at_mut(plane);
                let initial = if bias.is_empty() {
                    [0.0; 4]
                } else {
                    [bias[co], bias[co + 1], bias[co + 2], bias[co + 3]]
                };
                // Four independent scalar chains reuse each input load. Each
                // chain still starts with bias and visits channels in order.
                for pixel in 0..plane {
                    let [mut s0, mut s1, mut s2, mut s3] = initial;
                    for ci in 0..p.cpg {
                        let value = input[input_base + ci * plane + pixel];
                        s0 += value * w0[ci];
                        s1 += value * w1[ci];
                        s2 += value * w2[ci];
                        s3 += value * w3[ci];
                    }
                    o0[pixel] = s0;
                    o1[pixel] = s1;
                    o2[pixel] = s2;
                    o3[pixel] = s3;
                }
            }
            for channel in tiled..p.opg {
                let co = first_channel + channel;
                let weight = &weights[co * p.cpg..(co + 1) * p.cpg];
                for (pixel, output) in output
                    [(batch * p.cout + co) * plane..(batch * p.cout + co + 1) * plane]
                    .iter_mut()
                    .enumerate()
                {
                    let mut sum = if bias.is_empty() { 0.0 } else { bias[co] };
                    for (ci, &weight) in weight.iter().enumerate() {
                        sum += input[input_base + ci * plane + pixel] * weight;
                    }
                    *output = sum;
                }
            }
        }
    }
}
fn depthwise(p: &Conv, input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32]) {
    let plane = p.height * p.width;
    let out_plane = p.oh * p.ow;
    for batch in 0..p.batch {
        for channel in 0..p.cout {
            let source =
                &input[(batch * p.cin + channel) * plane..(batch * p.cin + channel + 1) * plane];
            let destination = &mut output[(batch * p.cout + channel) * out_plane
                ..(batch * p.cout + channel + 1) * out_plane];
            destination.fill(if bias.is_empty() { 0.0 } else { bias[channel] });
            for ky in 0..p.kh {
                let offset = ky * p.dh;
                let begin = p.pt.saturating_sub(offset).div_ceil(p.sh).min(p.oh);
                let end = (p.height + p.pt)
                    .saturating_sub(offset)
                    .div_ceil(p.sh)
                    .min(p.oh);
                for kx in 0..p.kw {
                    let offset_x = kx * p.dw;
                    let left = p.pl.saturating_sub(offset_x).div_ceil(p.sw).min(p.ow);
                    let right = (p.width + p.pl)
                        .saturating_sub(offset_x)
                        .div_ceil(p.sw)
                        .min(p.ow);
                    let weight = weights[(channel * p.kh + ky) * p.kw + kx];
                    for y in begin..end {
                        let row = (y * p.sh + offset - p.pt) * p.width;
                        let output_row = &mut destination[y * p.ow..(y + 1) * p.ow];
                        for (x, output) in output_row.iter_mut().enumerate().take(right).skip(left)
                        {
                            *output += source[row + x * p.sw + offset_x - p.pl] * weight;
                        }
                    }
                }
            }
        }
    }
}

pub(super) struct Reduction {
    input: Shape,
    reduction: usize,
    output_strides: [usize; MAX_RANK],
    spatial: Option<usize>,
}
pub(super) fn bind_reduce(
    input: Shape,
    axes: &[i64],
    keep: bool,
    noop: bool,
) -> RuntimeResult<(Shape, BoundKind)> {
    if axes.is_empty() && noop {
        return Ok((input, BoundKind::Copy));
    }
    let mut reduced = [false; MAX_RANK];
    if axes.is_empty() {
        reduced[..input.rank()].fill(true);
    }
    for &value in axes {
        let axis = shapes::axis(value, input.rank())?;
        if std::mem::replace(&mut reduced[axis], true) {
            return Err(RuntimeError::new("duplicate ReduceMean axis"));
        }
    }
    let mut dimensions = [0; MAX_RANK];
    let mut rank = 0;
    let mut reduction = 1usize;
    for (axis, &dimension) in input.dimensions().iter().enumerate() {
        if reduced[axis] {
            reduction = reduction
                .checked_mul(dimension)
                .ok_or_else(|| RuntimeError::new("reduction count overflow"))?;
            if keep {
                dimensions[rank] = 1;
                rank += 1;
            }
        } else {
            dimensions[rank] = dimension;
            rank += 1;
        }
    }
    let output = Shape::new(&dimensions[..rank])?;
    let mut output_strides = [0; MAX_RANK];
    let mut stride = 1;
    for axis in (0..input.rank()).rev() {
        if !reduced[axis] {
            output_strides[axis] = stride;
            stride *= input.dimensions()[axis];
        }
    }
    let spatial = if input.rank() == 4 && reduced[..4] == [false, false, true, true] {
        Some(input.dimensions()[2] * input.dimensions()[3])
    } else {
        None
    };
    Ok((
        output,
        BoundKind::Reduce(Reduction {
            input,
            reduction,
            output_strides,
            spatial,
        }),
    ))
}
fn reduce(plan: &Reduction, input: &[f32], output: &mut [f32]) {
    if let Some(spatial) = plan.spatial {
        for (values, output) in input.chunks_exact(spatial).zip(output) {
            let mut sum = 0.0f32;
            for &value in values {
                sum += value;
            }
            *output = sum / plan.reduction as f32;
        }
        return;
    }
    output.fill(0.0);
    let mut counters = [0; MAX_RANK];
    let mut offset = 0;
    for &value in input {
        output[offset] += value;
        for axis in (0..plan.input.rank()).rev() {
            counters[axis] += 1;
            if counters[axis] < plan.input.dimensions()[axis] {
                offset += plan.output_strides[axis];
                break;
            }
            offset -= (counters[axis] - 1) * plan.output_strides[axis];
            counters[axis] = 0;
        }
    }
    for value in output {
        *value /= plan.reduction as f32;
    }
}
