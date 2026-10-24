//! Backend-neutral bound operations and checked shape geometry.
use super::{RuntimeError, RuntimeResult, shapes};
use crate::{
    model::{ConvAttributes, TensorId},
    tensor::{MAX_RANK, Shape},
};

#[derive(Clone)]
pub(super) struct BoundStep {
    pub output: TensorId,
    pub kind: BoundKind,
}

#[derive(Clone, Copy)]
pub(super) enum BinaryOp {
    Add,
    Mul,
    Div,
    Sub,
}

#[derive(Clone)]
pub(super) enum BoundKind {
    Binary(Broadcast, BinaryOp),
    Square,
    Sqrt,
    Relu,
    Erf,
    Sigmoid,
    HardSigmoid(f32, f32),
    Conv(Conv),
    ConvTranspose(ConvTranspose),
    MaxPool {
        height: usize,
        width: usize,
    },
    AveragePool {
        height: usize,
        width: usize,
        output_height: usize,
        output_width: usize,
    },
    BatchNormalization {
        channels: usize,
        plane: usize,
        epsilon: f32,
    },
    Transpose(Transpose),
    Concat(Concat),
    Resize {
        height: usize,
        width: usize,
        factor: usize,
    },
    Reduce(Reduction),
    Copy,
    MatMul {
        rows: usize,
        inner: usize,
        columns: usize,
    },
    BatchedMatMul(BatchedMatMul),
    Softmax {
        rows: usize,
        columns: usize,
    },
}

#[derive(Clone)]
pub(super) struct BatchedMatMul {
    pub(super) rows: usize,
    pub(super) inner: usize,
    pub(super) columns: usize,
    pub(super) batches: usize,
    pub(super) batch_rank: usize,
    pub(super) dimensions: [usize; MAX_RANK],
    pub(super) a_strides: [usize; MAX_RANK],
    pub(super) b_strides: [usize; MAX_RANK],
    pub(super) a_matrix: usize,
    pub(super) b_matrix: usize,
    pub(super) output_matrix: usize,
    pub(super) equal_batches: bool,
}

pub(super) fn bind_matmul(a: Shape, b: Shape) -> RuntimeResult<(Shape, BoundKind)> {
    if a.rank() < 2 || b.rank() < 2 {
        return Err(RuntimeError::new(
            "MatMul requires rank-at-least-two F32 inputs",
        ));
    }
    let ad = a.dimensions();
    let bd = b.dimensions();
    let rows = ad[a.rank() - 2];
    let inner = ad[a.rank() - 1];
    let columns = bd[b.rank() - 1];
    if inner != bd[b.rank() - 2] {
        return Err(RuntimeError::new("MatMul inner dimensions differ"));
    }
    let batch_rank = a.rank().max(b.rank()) - 2;
    let mut dimensions = [1; MAX_RANK];
    for (axis, dimension) in dimensions.iter_mut().enumerate().take(batch_rank) {
        let leading = |shape: Shape| {
            axis.checked_sub(batch_rank - (shape.rank() - 2))
                .map_or(1, |axis| shape.dimensions()[axis])
        };
        let av = leading(a);
        let bv = leading(b);
        if av != bv && av != 1 && bv != 1 {
            return Err(RuntimeError::new(
                "MatMul leading dimensions do not broadcast",
            ));
        }
        *dimension = av.max(bv);
    }
    dimensions[batch_rank] = rows;
    dimensions[batch_rank + 1] = columns;
    let output = Shape::new(&dimensions[..batch_rank + 2])?;
    let a_matrix = rows
        .checked_mul(inner)
        .ok_or_else(|| RuntimeError::new("MatMul left matrix size overflow"))?;
    let b_matrix = inner
        .checked_mul(columns)
        .ok_or_else(|| RuntimeError::new("MatMul right matrix size overflow"))?;
    let output_matrix = rows
        .checked_mul(columns)
        .ok_or_else(|| RuntimeError::new("MatMul output matrix size overflow"))?;
    if a_matrix == 0 || b_matrix == 0 || output_matrix == 0 {
        return Err(RuntimeError::new(
            "MatMul zero-sized inputs are unsupported",
        ));
    }
    // Keep the existing flattened shared-weight path and its accumulation order.
    // It also handles a reusable rank-two activation RHS.
    if b.rank() == 2 {
        return Ok((
            output,
            BoundKind::MatMul {
                rows: a.element_count() / inner,
                inner,
                columns,
            },
        ));
    }
    let strides = |shape: Shape, matrix: usize| -> RuntimeResult<[usize; MAX_RANK]> {
        let mut strides = [0; MAX_RANK];
        let mut stride = matrix;
        let offset = batch_rank - (shape.rank() - 2);
        for axis in (0..shape.rank() - 2).rev() {
            let dimension = shape.dimensions()[axis];
            if dimension != 1 {
                strides[offset + axis] = stride;
            }
            stride = stride
                .checked_mul(dimension)
                .ok_or_else(|| RuntimeError::new("MatMul batch stride overflow"))?;
        }
        Ok(strides)
    };
    Ok((
        output,
        BoundKind::BatchedMatMul(BatchedMatMul {
            rows,
            inner,
            columns,
            batches: output.element_count() / output_matrix,
            batch_rank,
            dimensions,
            a_strides: strides(a, a_matrix)?,
            b_strides: strides(b, b_matrix)?,
            a_matrix,
            b_matrix,
            output_matrix,
            equal_batches: ad[..a.rank() - 2] == bd[..b.rank() - 2],
        }),
    ))
}

pub(super) fn bind_average_pool(input: Shape) -> RuntimeResult<(Shape, BoundKind)> {
    if input.rank() != 4 {
        return Err(RuntimeError::new(
            "AveragePool requires rank-four NCHW input",
        ));
    }
    let d = input.dimensions();
    let height = d[2]
        .checked_sub(3)
        .map(|v| v / 3 + 1)
        .ok_or_else(|| RuntimeError::new("AveragePool kernel exceeds input height"))?;
    let width = d[3]
        .checked_sub(2)
        .map(|v| v / 2 + 1)
        .ok_or_else(|| RuntimeError::new("AveragePool kernel exceeds input width"))?;
    Ok((
        Shape::new(&[d[0], d[1], height, width])?,
        BoundKind::AveragePool {
            height: d[2],
            width: d[3],
            output_height: height,
            output_width: width,
        },
    ))
}

pub(super) fn bind_batch_normalization(
    input: Shape,
    parameters: &[Shape],
    epsilon: f32,
) -> RuntimeResult<(Shape, BoundKind)> {
    if input.rank() < 2
        || parameters.len() != 4
        || parameters
            .iter()
            .any(|shape| shape.dimensions() != [input.dimensions()[1]])
    {
        return Err(RuntimeError::new(
            "BatchNormalization requires [N,C,...] input and four [C] parameters",
        ));
    }
    let channels = input.dimensions()[1];
    Ok((
        input,
        BoundKind::BatchNormalization {
            channels,
            plane: input.element_count() / input.dimensions()[0] / channels,
            epsilon,
        },
    ))
}

#[derive(Clone)]
pub(super) struct Transpose {
    pub(super) rank: usize,
    pub(super) dimensions: [usize; MAX_RANK],
    pub(super) strides: [usize; MAX_RANK],
    pub(super) base: usize,
}

pub(super) fn bind_transpose(
    input: Shape,
    permutation: &[i64],
) -> RuntimeResult<(Shape, BoundKind)> {
    if input.rank() == 0 || permutation.len() != input.rank() {
        return Err(RuntimeError::new("Transpose permutation rank mismatch"));
    }
    let mut input_strides = [0; MAX_RANK];
    let mut stride = 1usize;
    for axis in (0..input.rank()).rev() {
        input_strides[axis] = stride;
        stride *= input.dimensions()[axis];
    }
    let mut seen = [false; MAX_RANK];
    let mut dimensions = [0; MAX_RANK];
    let mut strides = [0; MAX_RANK];
    for (output_axis, &axis) in permutation.iter().enumerate() {
        let axis = usize::try_from(axis)
            .ok()
            .filter(|&axis| axis < input.rank())
            .ok_or_else(|| RuntimeError::new("Transpose permutation axis is invalid"))?;
        if seen[axis] {
            return Err(RuntimeError::new(
                "Transpose permutation axis is duplicated",
            ));
        }
        seen[axis] = true;
        dimensions[output_axis] = input.dimensions()[axis];
        strides[output_axis] = input_strides[axis];
    }
    Ok((
        Shape::new(&dimensions[..input.rank()])?,
        BoundKind::Transpose(Transpose {
            rank: input.rank(),
            dimensions,
            strides,
            base: 0,
        }),
    ))
}

pub(super) fn bind_slice(
    input: Shape,
    parameters: &shapes::SliceParameters,
) -> RuntimeResult<(Shape, BoundKind)> {
    let (output, starts) = parameters.bind(input)?;
    let mut strides = [0; MAX_RANK];
    let mut stride = 1usize;
    let mut base = 0usize;
    for axis in (0..input.rank()).rev() {
        strides[axis] = stride;
        base = starts[axis]
            .checked_mul(stride)
            .and_then(|offset| base.checked_add(offset))
            .ok_or_else(|| RuntimeError::new("Slice starting offset overflow"))?;
        stride = stride
            .checked_mul(input.dimensions()[axis])
            .ok_or_else(|| RuntimeError::new("Slice stride overflow"))?;
    }
    let mut dimensions = [0; MAX_RANK];
    dimensions[..output.rank()].copy_from_slice(output.dimensions());
    Ok((
        output,
        BoundKind::Transpose(Transpose {
            rank: output.rank(),
            dimensions,
            strides,
            base,
        }),
    ))
}

#[derive(Clone)]
pub(super) enum BroadcastKind {
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
    RowScalar {
        scalar_left: bool,
        columns: usize,
    },
    Generic,
}

#[derive(Clone)]
pub(super) struct Broadcast {
    pub(super) kind: BroadcastKind,
    pub(super) rank: usize,
    pub(super) dimensions: [usize; MAX_RANK],
    pub(super) a_strides: [usize; MAX_RANK],
    pub(super) b_strides: [usize; MAX_RANK],
}

impl Broadcast {
    pub(super) fn row_scalar_columns(&self) -> Option<usize> {
        match self.kind {
            BroadcastKind::RowScalar { columns, .. } => Some(columns),
            _ => None,
        }
    }
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
        let row = |full: Shape, gate: Shape, scalar_left| {
            if full == out
                && full.rank() >= 2
                && gate.rank() == full.rank()
                && gate.dimensions()[gate.rank() - 1] == 1
                && gate.dimensions()[..gate.rank() - 1] == full.dimensions()[..full.rank() - 1]
            {
                Some(BroadcastKind::RowScalar {
                    scalar_left,
                    columns: full.dimensions()[full.rank() - 1],
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
            .or_else(|| row(a, b, false))
            .or_else(|| row(b, a, true))
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

#[derive(Clone)]
pub(super) struct Concat {
    pub(super) axis: usize,
    pub(super) outer: usize,
    pub(super) output_chunk: usize,
}

pub(super) fn bind_concat(
    inputs: &[TensorId],
    shapes_by_id: &[Option<Shape>],
    value: i64,
) -> RuntimeResult<(Shape, BoundKind)> {
    let first = inputs
        .first()
        .and_then(|id| shapes_by_id[id.0])
        .ok_or_else(|| RuntimeError::new("Concat requires resolved input shapes"))?;
    let axis = shapes::axis(value, first.rank())?;
    let checked_product = |dimensions: &[usize]| {
        dimensions
            .iter()
            .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
            .ok_or_else(|| RuntimeError::new("Concat extent overflow"))
    };
    let outer = checked_product(&first.dimensions()[..axis])?;
    let inner = checked_product(&first.dimensions()[axis + 1..])?;
    let mut dimensions = [0; MAX_RANK];
    dimensions[..first.rank()].copy_from_slice(first.dimensions());
    dimensions[axis] = 0;
    let mut output_chunk = 0usize;
    for &id in inputs {
        let input =
            shapes_by_id[id.0].ok_or_else(|| RuntimeError::new("unresolved Concat input shape"))?;
        if input.rank() != first.rank()
            || input
                .dimensions()
                .iter()
                .zip(first.dimensions())
                .enumerate()
                .any(|(index, (a, b))| index != axis && a != b)
        {
            return Err(RuntimeError::new(
                "Concat input ranks/non-axis dimensions differ",
            ));
        }
        dimensions[axis] = dimensions[axis]
            .checked_add(input.dimensions()[axis])
            .ok_or_else(|| RuntimeError::new("Concat axis sum overflow"))?;
        let chunk = input.dimensions()[axis]
            .checked_mul(inner)
            .ok_or_else(|| RuntimeError::new("Concat input chunk overflow"))?;
        output_chunk = output_chunk
            .checked_add(chunk)
            .ok_or_else(|| RuntimeError::new("Concat output chunk overflow"))?;
    }
    Ok((
        Shape::new(&dimensions[..first.rank()])?,
        BoundKind::Concat(Concat {
            axis,
            outer,
            output_chunk,
        }),
    ))
}

pub(super) fn bind_max_pool(input: Shape) -> RuntimeResult<(Shape, BoundKind)> {
    if input.rank() != 4 {
        return Err(RuntimeError::new("MaxPool requires rank-four NCHW input"));
    }
    Ok((
        input,
        BoundKind::MaxPool {
            height: input.dimensions()[2],
            width: input.dimensions()[3],
        },
    ))
}

pub(super) fn bind_resize(input: Shape, factor: usize) -> RuntimeResult<(Shape, BoundKind)> {
    if input.rank() != 4 {
        return Err(RuntimeError::new("Resize requires rank-four NCHW input"));
    }
    let dimensions = input.dimensions();
    let height = dimensions[2]
        .checked_mul(factor)
        .ok_or_else(|| RuntimeError::new("Resize height overflow"))?;
    let width = dimensions[3]
        .checked_mul(factor)
        .ok_or_else(|| RuntimeError::new("Resize width overflow"))?;
    Ok((
        Shape::new(&[dimensions[0], dimensions[1], height, width])?,
        BoundKind::Resize {
            height: dimensions[2],
            width: dimensions[3],
            factor,
        },
    ))
}

#[derive(Clone)]
pub(super) struct ConvTranspose {
    pub(super) batch: usize,
    pub(super) cin: usize,
    pub(super) cout: usize,
    pub(super) height: usize,
    pub(super) width: usize,
}

pub(super) fn bind_conv_transpose(
    input: Shape,
    weights: Shape,
    bias: Option<Shape>,
) -> RuntimeResult<(Shape, BoundKind)> {
    if input.rank() != 4 || weights.rank() != 4 {
        return Err(RuntimeError::new(
            "ConvTranspose requires rank-four input and weights",
        ));
    }
    let dimensions = input.dimensions();
    let wd = weights.dimensions();
    if wd[0] != dimensions[1] || wd[2..] != [2, 2] {
        return Err(RuntimeError::new(
            "ConvTranspose weights must have shape [input_channels, output_channels, 2, 2]",
        ));
    }
    if bias.is_some_and(|shape| shape.dimensions() != [wd[1]]) {
        return Err(RuntimeError::new(
            "ConvTranspose bias must have shape [output_channels]",
        ));
    }
    let height = dimensions[2]
        .checked_mul(2)
        .ok_or_else(|| RuntimeError::new("ConvTranspose height overflow"))?;
    let width = dimensions[3]
        .checked_mul(2)
        .ok_or_else(|| RuntimeError::new("ConvTranspose width overflow"))?;
    Ok((
        Shape::new(&[dimensions[0], wd[1], height, width])?,
        BoundKind::ConvTranspose(ConvTranspose {
            batch: dimensions[0],
            cin: dimensions[1],
            cout: wd[1],
            height: dimensions[2],
            width: dimensions[3],
        }),
    ))
}

#[derive(Clone)]
pub(super) struct Conv {
    pub(super) batch: usize,
    pub(super) cin: usize,
    pub(super) cout: usize,
    pub(super) height: usize,
    pub(super) width: usize,
    pub(super) oh: usize,
    pub(super) ow: usize,
    pub(super) cpg: usize,
    pub(super) opg: usize,
    pub(super) kh: usize,
    pub(super) kw: usize,
    pub(super) sh: usize,
    pub(super) sw: usize,
    pub(super) dh: usize,
    pub(super) dw: usize,
    pub(super) pt: usize,
    pub(super) pl: usize,
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

#[derive(Clone)]
pub(super) struct Reduction {
    pub(super) input: Shape,
    pub(super) reduction: usize,
    pub(super) output_strides: [usize; MAX_RANK],
    pub(super) contiguous: Option<usize>,
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
    let contiguous = if input.rank() == 4 && reduced[..4] == [false, false, true, true] {
        Some(input.dimensions()[2] * input.dimensions()[3])
    } else if input.rank() > 0
        && reduced[input.rank() - 1]
        && reduced[..input.rank() - 1].iter().all(|&axis| !axis)
    {
        Some(input.dimensions()[input.rank() - 1])
    } else {
        None
    };
    Ok((
        output,
        BoundKind::Reduce(Reduction {
            input,
            reduction,
            output_strides,
            contiguous,
        }),
    ))
}
