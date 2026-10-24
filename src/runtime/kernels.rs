use super::bound::*;
use crate::{math::exp_f32, model::TensorId, tensor::MAX_RANK};

impl Broadcast {
    pub(super) fn channel(&self) -> Option<super::simd::ChannelBroadcast> {
        if let BroadcastKind::Channel {
            scalar_left,
            channels,
            plane,
            batched,
        } = self.kind
        {
            Some(super::simd::ChannelBroadcast {
                channels,
                plane,
                batched,
                gate_left: scalar_left,
            })
        } else {
            None
        }
    }
}

pub(super) fn execute<'a>(
    kind: &BoundKind,
    inputs: &[TensorId],
    data: impl Fn(TensorId) -> &'a [f32],
    out: &mut [f32],
) {
    let a = data(inputs[0]);
    match kind {
        BoundKind::Binary(plan, operation) => {
            let b = data(inputs[1]);
            match operation {
                BinaryOp::Add => binary::<0>(plan, a, b, out),
                BinaryOp::Mul => binary::<1>(plan, a, b, out),
                BinaryOp::Div => binary::<2>(plan, a, b, out),
                BinaryOp::Sub => binary::<3>(plan, a, b, out),
            }
        }
        BoundKind::Square => {
            // Native MathF.Pow(x, 2) differs by occasional rounding/underflow
            // bits. The shipped subset uses this portable square under the
            // existing numerical tolerance, with native LayerNorm gates.
            for (o, &x) in out.iter_mut().zip(a) {
                *o = x * x;
            }
        }
        BoundKind::Sqrt => {
            for (o, &x) in out.iter_mut().zip(a) {
                *o = x.sqrt();
            }
        }
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
        BoundKind::Erf => {
            for (o, &value) in out.iter_mut().zip(a) {
                *o = erf(value);
            }
        }
        BoundKind::Sigmoid => {
            for (o, &value) in out.iter_mut().zip(a) {
                *o = if value >= 0.0 {
                    1.0 / (1.0 + exp_f32(-value))
                } else {
                    let exponent = exp_f32(value);
                    exponent / (1.0 + exponent)
                };
            }
        }
        BoundKind::Conv(plan) => conv(
            plan,
            a,
            data(inputs[1]),
            inputs.get(2).map_or(&[], |&id| data(id)),
            out,
        ),
        BoundKind::ConvTranspose(plan) => conv_transpose(
            plan,
            a,
            data(inputs[1]),
            inputs.get(2).map_or(&[], |&id| data(id)),
            out,
        ),
        BoundKind::MaxPool { height, width } => max_pool(a, out, *height, *width),
        BoundKind::AveragePool {
            height,
            width,
            output_height,
            output_width,
        } => average_pool(a, out, *height, *width, *output_height, *output_width),
        BoundKind::BatchNormalization {
            channels,
            plane,
            epsilon,
        } => batch_normalization(
            a,
            data(inputs[1]),
            data(inputs[2]),
            data(inputs[3]),
            data(inputs[4]),
            out,
            *channels,
            *plane,
            *epsilon,
        ),
        BoundKind::Transpose(plan) => transpose(plan, a, out),
        BoundKind::Concat(plan) => concat(plan, inputs, data, out),
        BoundKind::Resize {
            height,
            width,
            factor,
        } => resize(a, out, *height, *width, *factor),
        BoundKind::Reduce(plan) => reduce(plan, a, out),
        BoundKind::Copy => out.copy_from_slice(a),
        BoundKind::MatMul {
            rows,
            inner,
            columns,
        } => matmul(a, data(inputs[1]), out, *rows, *inner, *columns),
        BoundKind::BatchedMatMul(plan) => batched_matmul(plan, a, data(inputs[1]), out),
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
                    let value = exp_f32(x - max);
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

/// Shape binding selects this dispatch once; every scalar fallback remains available.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_optimized<'a>(
    kind: &BoundKind,
    inputs: &[TensorId],
    data: impl Fn(TensorId) -> &'a [f32],
    out: &mut [f32],
    backend: super::Backend,
    packed: Option<&super::packing::PackedMatrix>,
    cpu: &super::CpuContext,
    threads: usize,
) {
    use super::simd;
    #[cfg(not(feature = "parallel"))]
    let _ = (cpu, threads);
    if backend == super::Backend::Scalar {
        execute(kind, inputs, data, out);
        return;
    }
    let a = data(inputs[0]);
    match kind {
        BoundKind::MatMul {
            rows,
            inner,
            columns,
        } => {
            let b = data(inputs[1]);
            #[cfg(feature = "parallel")]
            if threads > 1 && rows * inner * columns >= 262_144 && *rows >= 8 {
                use rayon::prelude::*;
                let row_chunk = rows.div_ceil(threads).max(4).next_multiple_of(4);
                cpu.install(|| {
                    out.par_chunks_mut(row_chunk * columns)
                        .enumerate()
                        .for_each(|(i, dst)| {
                            let start = i * row_chunk;
                            simd::matmul(
                                backend,
                                &a[start * inner..start * inner + dst.len() / columns * inner],
                                b,
                                packed,
                                dst,
                                dst.len() / columns,
                                *inner,
                                *columns,
                            );
                        })
                });
                return;
            }
            simd::matmul(backend, a, b, packed, out, *rows, *inner, *columns);
        }
        BoundKind::BatchedMatMul(p) => {
            let b = data(inputs[1]);
            let mut counters = [0; MAX_RANK];
            let (mut ai, mut bi) = (0, 0);
            for batch in 0..p.batches {
                simd::matmul(
                    backend,
                    &a[ai..ai + p.a_matrix],
                    &b[bi..bi + p.b_matrix],
                    None,
                    &mut out[batch * p.output_matrix..(batch + 1) * p.output_matrix],
                    p.rows,
                    p.inner,
                    p.columns,
                );
                for axis in (0..p.batch_rank).rev() {
                    counters[axis] += 1;
                    if counters[axis] < p.dimensions[axis] {
                        ai += p.a_strides[axis];
                        bi += p.b_strides[axis];
                        break;
                    }
                    ai -= (counters[axis] - 1) * p.a_strides[axis];
                    bi -= (counters[axis] - 1) * p.b_strides[axis];
                    counters[axis] = 0;
                }
            }
        }
        BoundKind::Conv(p) => {
            let params = simd::Conv2d {
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
            };
            let weights = data(inputs[1]);
            let bias: &[f32] = inputs.get(2).map_or(&[], |&id| data(id));
            if super::nchw::try_conv(
                backend, &params, a, weights, bias, packed, out, cpu, threads,
            ) {
                return;
            }
            // Whole-output scalar APIs must be selected before channel bands.
            if params.uses_original_convolution(backend) {
                simd::conv2d(backend, &params, a, weights, bias, out);
                return;
            }
            #[cfg(feature = "parallel")]
            if threads > 1
                && p.batch * p.cout * p.oh * p.ow * p.cpg * p.kh * p.kw >= 262_144
                && p.cout >= 4
            {
                use rayon::prelude::*;
                let channels = p.cout.div_ceil(threads).max(4).next_multiple_of(4);
                let plane = p.oh * p.ow;
                for (batch, dst) in out.chunks_exact_mut(p.cout * plane).enumerate() {
                    cpu.install(|| {
                        dst.par_chunks_mut(channels * plane)
                            .enumerate()
                            .for_each(|(band, dst)| {
                                simd::conv2d_output_range(
                                    backend,
                                    &params,
                                    a,
                                    weights,
                                    bias,
                                    dst,
                                    batch,
                                    band * channels,
                                    dst.len() / plane,
                                )
                            })
                    });
                }
                return;
            }
            simd::conv2d(backend, &params, a, weights, bias, out);
        }
        BoundKind::ConvTranspose(p) => {
            let weights = data(inputs[1]);
            let bias: &[f32] = inputs.get(2).map_or(&[], |&id| data(id));
            if !super::nchw::try_transpose(backend, p, a, weights, bias, out, cpu, threads) {
                simd::conv_transpose2x2(
                    backend, a, weights, bias, out, p.batch, p.cin, p.cout, p.height, p.width,
                );
            }
        }
        BoundKind::Square => {
            simd::parallel::unary(backend, simd::UnaryOp::Square, a, out, cpu, threads)
        }
        BoundKind::Erf => simd::parallel::unary(backend, simd::UnaryOp::Erf, a, out, cpu, threads),
        BoundKind::Sigmoid => {
            simd::parallel::unary(backend, simd::UnaryOp::Sigmoid, a, out, cpu, threads)
        }
        BoundKind::Softmax { rows, columns } => simd::softmax(backend, a, out, *rows, *columns),
        BoundKind::MaxPool { height, width } => simd::max_pool(backend, a, out, *height, *width),
        BoundKind::AveragePool {
            height,
            width,
            output_height,
            output_width,
        } => simd::average_pool(
            backend,
            a,
            out,
            *height,
            *width,
            *output_height,
            *output_width,
        ),
        BoundKind::Reduce(p)
            if p.input.rank() == 4
                && p.contiguous == Some(p.input.dimensions()[2] * p.input.dimensions()[3])
                && out.len() == p.input.dimensions()[0] * p.input.dimensions()[1] =>
        {
            simd::spatial_mean(
                backend,
                a,
                out,
                p.input.dimensions()[2] * p.input.dimensions()[3],
            );
        }
        BoundKind::Relu => {
            simd::parallel::unary(backend, simd::UnaryOp::Relu, a, out, cpu, threads)
        }
        BoundKind::Sqrt => {
            simd::parallel::unary(backend, simd::UnaryOp::Sqrt, a, out, cpu, threads)
        }
        BoundKind::HardSigmoid(alpha, beta) => simd::unary(
            backend,
            simd::UnaryOp::HardSigmoid {
                alpha: *alpha,
                beta: *beta,
            },
            a,
            out,
        ),
        BoundKind::Binary(p, operation) => {
            let b = data(inputs[1]);
            let op = match operation {
                BinaryOp::Add => simd::BinaryOp::Add,
                BinaryOp::Mul => simd::BinaryOp::Mul,
                BinaryOp::Div => simd::BinaryOp::Div,
                BinaryOp::Sub => simd::BinaryOp::Sub,
            };
            match p.kind {
                BroadcastKind::Full => simd::parallel::binary(backend, op, a, b, out, cpu, threads),
                BroadcastKind::RightScalar => simd::binary_scalar(backend, op, a, b[0], false, out),
                BroadcastKind::LeftScalar => simd::binary_scalar(backend, op, b, a[0], true, out),
                _ => execute(kind, inputs, data, out),
            }
        }
        _ => execute(kind, inputs, data, out),
    }
}

pub(super) fn erf(x: f32) -> f32 {
    // Identical individual F32 operations to the C# scalar reference.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let magnitude = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * magnitude);
    let polynomial =
        ((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449672) * t + 0.2548296) * t;
    sign * (1.0 - polynomial * exp_f32(-magnitude * magnitude))
}

fn batched_matmul(plan: &BatchedMatMul, a: &[f32], b: &[f32], out: &mut [f32]) {
    if plan.equal_batches {
        for ((left, right), output) in a
            .chunks_exact(plan.a_matrix)
            .zip(b.chunks_exact(plan.b_matrix))
            .zip(out.chunks_exact_mut(plan.output_matrix))
        {
            matmul(left, right, output, plan.rows, plan.inner, plan.columns);
        }
        return;
    }
    let mut counters = [0; MAX_RANK];
    let (mut ai, mut bi) = (0, 0);
    for batch in 0..plan.batches {
        matmul(
            &a[ai..ai + plan.a_matrix],
            &b[bi..bi + plan.b_matrix],
            &mut out[batch * plan.output_matrix..(batch + 1) * plan.output_matrix],
            plan.rows,
            plan.inner,
            plan.columns,
        );
        for axis in (0..plan.batch_rank).rev() {
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

// Four independent rows share each contiguous four-column weight load. Each
// accumulator still adds products in ascending K order, without FMA or packing.
pub(super) fn matmul(
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    let full_rows = rows / 4 * 4;
    let full_columns = columns / 4 * 4;
    for row in (0..full_rows).step_by(4) {
        let f0 = &input[row * inner..(row + 1) * inner];
        let f1 = &input[(row + 1) * inner..(row + 2) * inner];
        let f2 = &input[(row + 2) * inner..(row + 3) * inner];
        let f3 = &input[(row + 3) * inner..(row + 4) * inner];
        for column in (0..full_columns).step_by(4) {
            let (mut a00, mut a01, mut a02, mut a03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let (mut a10, mut a11, mut a12, mut a13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let (mut a20, mut a21, mut a22, mut a23) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let (mut a30, mut a31, mut a32, mut a33) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for k in 0..inner {
                let w = &weights[k * columns + column..k * columns + column + 4];
                let (w0, w1, w2, w3) = (w[0], w[1], w[2], w[3]);
                let (v0, v1, v2, v3) = (f0[k], f1[k], f2[k], f3[k]);
                a00 += v0 * w0;
                a01 += v0 * w1;
                a02 += v0 * w2;
                a03 += v0 * w3;
                a10 += v1 * w0;
                a11 += v1 * w1;
                a12 += v1 * w2;
                a13 += v1 * w3;
                a20 += v2 * w0;
                a21 += v2 * w1;
                a22 += v2 * w2;
                a23 += v2 * w3;
                a30 += v3 * w0;
                a31 += v3 * w1;
                a32 += v3 * w2;
                a33 += v3 * w3;
            }
            output[row * columns + column..row * columns + column + 4]
                .copy_from_slice(&[a00, a01, a02, a03]);
            output[(row + 1) * columns + column..(row + 1) * columns + column + 4]
                .copy_from_slice(&[a10, a11, a12, a13]);
            output[(row + 2) * columns + column..(row + 2) * columns + column + 4]
                .copy_from_slice(&[a20, a21, a22, a23]);
            output[(row + 3) * columns + column..(row + 3) * columns + column + 4]
                .copy_from_slice(&[a30, a31, a32, a33]);
        }
        for column in full_columns..columns {
            let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for k in 0..inner {
                let weight = weights[k * columns + column];
                a0 += f0[k] * weight;
                a1 += f1[k] * weight;
                a2 += f2[k] * weight;
                a3 += f3[k] * weight;
            }
            output[row * columns + column] = a0;
            output[(row + 1) * columns + column] = a1;
            output[(row + 2) * columns + column] = a2;
            output[(row + 3) * columns + column] = a3;
        }
    }
    for row in full_rows..rows {
        let f = &input[row * inner..(row + 1) * inner];
        for (column, out) in output[row * columns..(row + 1) * columns]
            .iter_mut()
            .enumerate()
        {
            let mut sum = 0.0f32;
            for (k, &value) in f.iter().enumerate() {
                sum += value * weights[k * columns + column];
            }
            *out = sum;
        }
    }
}

fn average_pool(
    input: &[f32],
    output: &mut [f32],
    height: usize,
    width: usize,
    output_height: usize,
    output_width: usize,
) {
    for (source, destination) in input
        .chunks_exact(height * width)
        .zip(output.chunks_exact_mut(output_height * output_width))
    {
        for (y, row) in destination.chunks_exact_mut(output_width).enumerate() {
            for (x, value) in row.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for ky in 0..3 {
                    let start = (y * 3 + ky) * width + x * 2;
                    sum += source[start];
                    sum += source[start + 1];
                }
                *value = sum / 6.0;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn batch_normalization(
    input: &[f32],
    scale: &[f32],
    bias: &[f32],
    mean: &[f32],
    variance: &[f32],
    output: &mut [f32],
    channels: usize,
    plane: usize,
    epsilon: f32,
) {
    for (block, (source, destination)) in input
        .chunks_exact(plane)
        .zip(output.chunks_exact_mut(plane))
        .enumerate()
    {
        let channel = block % channels;
        let multiplier = scale[channel] / (variance[channel] + epsilon).sqrt();
        for (&value, out) in source.iter().zip(destination) {
            *out = (value - mean[channel]) * multiplier + bias[channel];
        }
    }
}

fn transpose(plan: &Transpose, input: &[f32], output: &mut [f32]) {
    let inner = plan.dimensions[plan.rank - 1];
    let inner_stride = plan.strides[plan.rank - 1];
    let mut counters = [0; MAX_RANK];
    let mut offset = plan.base;
    for destination in output.chunks_exact_mut(inner) {
        if inner_stride == 1 {
            destination.copy_from_slice(&input[offset..offset + inner]);
        } else {
            for (i, value) in destination.iter_mut().enumerate() {
                *value = input[offset + i * inner_stride];
            }
        }
        for axis in (0..plan.rank - 1).rev() {
            counters[axis] += 1;
            if counters[axis] < plan.dimensions[axis] {
                offset += plan.strides[axis];
                break;
            }
            offset -= (counters[axis] - 1) * plan.strides[axis];
            counters[axis] = 0;
        }
    }
}

#[inline]
fn arithmetic<const OP: u8>(a: f32, b: f32) -> f32 {
    match OP {
        0 => a + b,
        1 => a * b,
        2 => a / b,
        _ => a - b,
    }
}
fn binary<const OP: u8>(plan: &Broadcast, a: &[f32], b: &[f32], out: &mut [f32]) {
    match plan.kind {
        BroadcastKind::Full => {
            for ((o, &a), &b) in out.iter_mut().zip(a).zip(b) {
                *o = arithmetic::<OP>(a, b);
            }
        }
        BroadcastKind::RightScalar => {
            for (o, &a) in out.iter_mut().zip(a) {
                *o = arithmetic::<OP>(a, b[0]);
            }
        }
        BroadcastKind::LeftScalar => {
            for (o, &b) in out.iter_mut().zip(b) {
                *o = arithmetic::<OP>(a[0], b);
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
                        arithmetic::<OP>(scalar, value)
                    } else {
                        arithmetic::<OP>(value, scalar)
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
                        arithmetic::<OP>(scalar, value)
                    } else {
                        arithmetic::<OP>(value, scalar)
                    };
                }
            }
        }
        BroadcastKind::RowScalar {
            scalar_left,
            columns,
        } => {
            let (full, scalars) = if scalar_left { (b, a) } else { (a, b) };
            for ((input, output), &scalar) in full
                .chunks_exact(columns)
                .zip(out.chunks_exact_mut(columns))
                .zip(scalars)
            {
                for (&value, output) in input.iter().zip(output) {
                    *output = if scalar_left {
                        arithmetic::<OP>(scalar, value)
                    } else {
                        arithmetic::<OP>(value, scalar)
                    };
                }
            }
        }
        BroadcastKind::Generic => {
            let mut counters = [0; MAX_RANK];
            let mut ai = 0;
            let mut bi = 0;
            for o in out {
                *o = arithmetic::<OP>(a[ai], b[bi]);
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

fn concat<'a>(
    plan: &Concat,
    inputs: &[TensorId],
    data: impl Fn(TensorId) -> &'a [f32],
    output: &mut [f32],
) {
    for outer in 0..plan.outer {
        let mut destination = outer * plan.output_chunk;
        for &id in inputs {
            let source = data(id);
            let chunk = source.len() / plan.outer;
            output[destination..destination + chunk]
                .copy_from_slice(&source[outer * chunk..(outer + 1) * chunk]);
            destination += chunk;
        }
    }
}

fn max_pool(input: &[f32], output: &mut [f32], height: usize, width: usize) {
    let plane = height * width;
    for (source, destination) in input
        .chunks_exact(plane)
        .zip(output.chunks_exact_mut(plane))
    {
        for y in 0..height - 1 {
            let row = &source[y * width..(y + 1) * width];
            let next = &source[(y + 1) * width..(y + 2) * width];
            let out = &mut destination[y * width..(y + 1) * width];
            for x in 0..width - 1 {
                out[x] = row[x].max(row[x + 1]).max(next[x].max(next[x + 1]));
            }
            out[width - 1] = row[width - 1].max(next[width - 1]);
        }
        let last = (height - 1) * width;
        for x in 0..width - 1 {
            destination[last + x] = source[last + x].max(source[last + x + 1]);
        }
        destination[last + width - 1] = source[last + width - 1];
    }
}

fn resize(input: &[f32], output: &mut [f32], height: usize, width: usize, factor: usize) {
    let output_width = width * factor;
    let output_plane = height * factor * output_width;
    for (source, destination) in input
        .chunks_exact(height * width)
        .zip(output.chunks_exact_mut(output_plane))
    {
        for (row, values) in source.chunks_exact(width).enumerate() {
            let rows =
                &mut destination[row * factor * output_width..(row + 1) * factor * output_width];
            let (first, remaining) = rows.split_at_mut(output_width);
            for (&value, repeated) in values.iter().zip(first.chunks_exact_mut(factor)) {
                repeated.fill(value);
            }
            for repeated in remaining.chunks_exact_mut(output_width) {
                repeated.copy_from_slice(first);
            }
        }
    }
}

fn conv_transpose(
    plan: &ConvTranspose,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    let input_plane = plan.height * plan.width;
    let output_width = plan.width * 2;
    let output_plane = input_plane * 4;
    for batch in 0..plan.batch {
        for co in 0..plan.cout {
            let destination = &mut output[(batch * plan.cout + co) * output_plane
                ..(batch * plan.cout + co + 1) * output_plane];
            destination.fill(if bias.is_empty() { 0.0 } else { bias[co] });
            for ci in 0..plan.cin {
                let source = &input[(batch * plan.cin + ci) * input_plane
                    ..(batch * plan.cin + ci + 1) * input_plane];
                let weight = &weights[(ci * plan.cout + co) * 4..(ci * plan.cout + co + 1) * 4];
                for (row, values) in source.chunks_exact(plan.width).enumerate() {
                    let rows =
                        &mut destination[row * 2 * output_width..(row + 1) * 2 * output_width];
                    let (top, bottom) = rows.split_at_mut(output_width);
                    for ((&value, top), bottom) in values
                        .iter()
                        .zip(top.as_chunks_mut::<2>().0.iter_mut())
                        .zip(bottom.as_chunks_mut::<2>().0.iter_mut())
                    {
                        top[0] += value * weight[0];
                        top[1] += value * weight[1];
                        bottom[0] += value * weight[2];
                        bottom[1] += value * weight[3];
                    }
                }
            }
        }
    }
}

pub(super) fn conv(p: &Conv, input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32]) {
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
        for group in 0..p.cout / p.opg {
            let first_channel = group * p.opg;
            let tiled_end = first_channel + p.opg / 4 * 4;
            let input_group = &input[(batch * p.cin + group * p.cpg) * plane
                ..(batch * p.cin + (group + 1) * p.cpg) * plane];
            let weight_plane = p.cpg * p.kh * p.kw;
            for co in (first_channel..tiled_end).step_by(4) {
                let (w0, rest) =
                    weights[co * weight_plane..(co + 4) * weight_plane].split_at(weight_plane);
                let (w1, rest) = rest.split_at(weight_plane);
                let (w2, w3) = rest.split_at(weight_plane);
                let base = (batch * p.cout + co) * out_plane;
                let (o0, rest) = output[base..base + 4 * out_plane].split_at_mut(out_plane);
                let (o1, rest) = rest.split_at_mut(out_plane);
                let (o2, o3) = rest.split_at_mut(out_plane);
                let initial = if bias.is_empty() {
                    [0.0; 4]
                } else {
                    [bias[co], bias[co + 1], bias[co + 2], bias[co + 3]]
                };
                for y in 0..p.oh {
                    let rows = valid_taps(y * p.sh, p.pt, p.height, p.dh, p.kh);
                    let d0 = &mut o0[y * p.ow..(y + 1) * p.ow];
                    let d1 = &mut o1[y * p.ow..(y + 1) * p.ow];
                    let d2 = &mut o2[y * p.ow..(y + 1) * p.ow];
                    let d3 = &mut o3[y * p.ow..(y + 1) * p.ow];
                    for x in 0..p.ow {
                        let columns = valid_taps(x * p.sw, p.pl, p.width, p.dw, p.kw);
                        let [mut s0, mut s1, mut s2, mut s3] = initial;
                        // Four independent bias-first chains visit exactly the
                        // same ci/ky/kx order as the reference. Load each source
                        // tap once; padding ranges are independent of channels.
                        for ci in 0..p.cpg {
                            for ky in rows.clone() {
                                let source_row = &input_group[ci * plane
                                    + (y * p.sh + ky * p.dh - p.pt) * p.width
                                    ..ci * plane + (y * p.sh + ky * p.dh - p.pt + 1) * p.width];
                                let weight_row = (ci * p.kh + ky) * p.kw;
                                for kx in columns.clone() {
                                    let value = source_row[x * p.sw + kx * p.dw - p.pl];
                                    let tap = weight_row + kx;
                                    s0 += value * w0[tap];
                                    s1 += value * w1[tap];
                                    s2 += value * w2[tap];
                                    s3 += value * w3[tap];
                                }
                            }
                        }
                        d0[x] = s0;
                        d1[x] = s1;
                        d2[x] = s2;
                        d3[x] = s3;
                    }
                }
            }
            for co in tiled_end..first_channel + p.opg {
                let weight = &weights[co * weight_plane..(co + 1) * weight_plane];
                let destination = &mut output
                    [(batch * p.cout + co) * out_plane..(batch * p.cout + co + 1) * out_plane];
                for y in 0..p.oh {
                    let rows = valid_taps(y * p.sh, p.pt, p.height, p.dh, p.kh);
                    for (x, output) in destination[y * p.ow..(y + 1) * p.ow].iter_mut().enumerate()
                    {
                        let columns = valid_taps(x * p.sw, p.pl, p.width, p.dw, p.kw);
                        let mut sum = if bias.is_empty() { 0.0 } else { bias[co] };
                        for ci in 0..p.cpg {
                            for ky in rows.clone() {
                                let row = ci * plane + (y * p.sh + ky * p.dh - p.pt) * p.width;
                                let weight_row = (ci * p.kh + ky) * p.kw;
                                for kx in columns.clone() {
                                    sum += input_group[row + x * p.sw + kx * p.dw - p.pl]
                                        * weight[weight_row + kx];
                                }
                            }
                        }
                        *output = sum;
                    }
                }
            }
        }
    }
}

fn valid_taps(
    origin: usize,
    padding: usize,
    input: usize,
    dilation: usize,
    kernel: usize,
) -> std::ops::Range<usize> {
    let first = padding
        .saturating_sub(origin)
        .div_ceil(dilation)
        .min(kernel);
    let end = (input + padding)
        .saturating_sub(origin)
        .div_ceil(dilation)
        .min(kernel);
    first..end
}
pub(super) fn conv1x1(p: &Conv, input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32]) {
    let plane = p.height * p.width;
    let full_pixels = plane / 4 * 4;
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
                // Sixteen independent bias-first chains share four weights
                // and four adjacent NCHW pixels. Each sum still visits input
                // channels in ascending order without FMA or packing.
                for pixel in (0..full_pixels).step_by(4) {
                    let mut a00 = initial[0];
                    let mut a01 = initial[0];
                    let mut a02 = initial[0];
                    let mut a03 = initial[0];
                    let mut a10 = initial[1];
                    let mut a11 = initial[1];
                    let mut a12 = initial[1];
                    let mut a13 = initial[1];
                    let mut a20 = initial[2];
                    let mut a21 = initial[2];
                    let mut a22 = initial[2];
                    let mut a23 = initial[2];
                    let mut a30 = initial[3];
                    let mut a31 = initial[3];
                    let mut a32 = initial[3];
                    let mut a33 = initial[3];
                    for ci in 0..p.cpg {
                        let v = &input
                            [input_base + ci * plane + pixel..input_base + ci * plane + pixel + 4];
                        let (v0, v1, v2, v3) = (v[0], v[1], v[2], v[3]);
                        let (t0, t1, t2, t3) = (w0[ci], w1[ci], w2[ci], w3[ci]);
                        a00 += v0 * t0;
                        a01 += v1 * t0;
                        a02 += v2 * t0;
                        a03 += v3 * t0;
                        a10 += v0 * t1;
                        a11 += v1 * t1;
                        a12 += v2 * t1;
                        a13 += v3 * t1;
                        a20 += v0 * t2;
                        a21 += v1 * t2;
                        a22 += v2 * t2;
                        a23 += v3 * t2;
                        a30 += v0 * t3;
                        a31 += v1 * t3;
                        a32 += v2 * t3;
                        a33 += v3 * t3;
                    }
                    o0[pixel..pixel + 4].copy_from_slice(&[a00, a01, a02, a03]);
                    o1[pixel..pixel + 4].copy_from_slice(&[a10, a11, a12, a13]);
                    o2[pixel..pixel + 4].copy_from_slice(&[a20, a21, a22, a23]);
                    o3[pixel..pixel + 4].copy_from_slice(&[a30, a31, a32, a33]);
                }
                for pixel in full_pixels..plane {
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

fn reduce(plan: &Reduction, input: &[f32], output: &mut [f32]) {
    if let Some(contiguous) = plan.contiguous {
        for (values, output) in input.chunks_exact(contiguous).zip(output) {
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

#[cfg(test)]
mod scalar_refinement_tests {
    use super::*;
    use crate::{model::ConvAttributes, tensor::Shape};

    fn assert_bits(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "element {index}: {actual} versus {expected}"
            );
        }
    }
    fn distribution(count: usize) -> Vec<f32> {
        let values = [
            1e20f32, -1e20, 3., 1e-20, -0., 1., -1., 2.5, -2.5, 1.0000001, 0.9999999,
        ];
        (0..count)
            .map(|index| values[index % values.len()])
            .collect()
    }
    #[test]
    fn contiguous_trailing_reduction_matches_the_original_generic_accumulation_bits() {
        for dimensions in [
            vec![7],
            vec![2, 3, 7],
            vec![2, 5, 120],
            vec![1, 11, 192],
            vec![2, 3, 1],
            vec![2, 2, 3, 5],
        ] {
            let input_shape = Shape::new(&dimensions).unwrap();
            let values = distribution(input_shape.element_count());
            for keep in [false, true] {
                let (output_shape, kind) = bind_reduce(input_shape, &[-1], keep, false).unwrap();
                let BoundKind::Reduce(mut plan) = kind else {
                    panic!("expected reduction");
                };
                assert!(plan.contiguous.is_some());
                let mut specialized = vec![f32::NAN; output_shape.element_count()];
                let mut original = specialized.clone();
                reduce(&plan, &values, &mut specialized);
                plan.contiguous = None;
                reduce(&plan, &values, &mut original);
                assert_bits(&specialized, &original);
            }
        }
    }
    fn compare_row_scalar<const OP: u8>(a_shape: Shape, b_shape: Shape, a: &[f32], b: &[f32]) {
        let output_shape = a_shape.broadcast(b_shape).unwrap();
        let mut plan = Broadcast::new(a_shape, b_shape, output_shape);
        assert!(matches!(plan.kind, BroadcastKind::RowScalar { .. }));
        let mut specialized = vec![f32::NAN; output_shape.element_count()];
        let mut original = specialized.clone();
        binary::<OP>(&plan, a, b, &mut specialized);
        plan.kind = BroadcastKind::Generic;
        binary::<OP>(&plan, a, b, &mut original);
        assert_bits(&specialized, &original);
    }
    #[test]
    fn row_scalar_paths_preserve_operand_order_and_generic_arithmetic_bits() {
        for dimensions in [
            vec![5, 7],
            vec![2, 3, 7],
            vec![2, 5, 120],
            vec![1, 11, 192],
            vec![2, 3, 4, 5],
        ] {
            let full_shape = Shape::new(&dimensions).unwrap();
            let mut scalar_dimensions = dimensions.clone();
            *scalar_dimensions.last_mut().unwrap() = 1;
            let scalar_shape = Shape::new(&scalar_dimensions).unwrap();
            let full = distribution(full_shape.element_count());
            let scalars = (0..scalar_shape.element_count())
                .map(|index| (index as f32 + 1.) / 8.)
                .collect::<Vec<_>>();
            for scalar_left in [false, true] {
                let (a_shape, b_shape, a, b) = if scalar_left {
                    (
                        scalar_shape,
                        full_shape,
                        scalars.as_slice(),
                        full.as_slice(),
                    )
                } else {
                    (
                        full_shape,
                        scalar_shape,
                        full.as_slice(),
                        scalars.as_slice(),
                    )
                };
                compare_row_scalar::<0>(a_shape, b_shape, a, b);
                compare_row_scalar::<1>(a_shape, b_shape, a, b);
                compare_row_scalar::<2>(a_shape, b_shape, a, b);
                compare_row_scalar::<3>(a_shape, b_shape, a, b);
            }
        }
        // Broadcasting a single batch of row scalars is a different contract:
        // it must remain in the generic planner instead of truncating the zip.
        let full = Shape::new(&[2, 3, 7]).unwrap();
        let shared = Shape::new(&[1, 3, 1]).unwrap();
        assert!(matches!(
            Broadcast::new(full, shared, full).kind,
            BroadcastKind::Generic
        ));
    }
    fn reference_conv1x1(p: &Conv, input: &[f32], weights: &[f32], bias: &[f32]) -> Vec<f32> {
        let plane = p.height * p.width;
        let mut output = vec![0.0; p.batch * p.cout * plane];
        for batch in 0..p.batch {
            for channel in 0..p.cout {
                let group = channel / p.opg;
                for pixel in 0..plane {
                    let mut sum = if bias.is_empty() { 0.0 } else { bias[channel] };
                    for ci in 0..p.cpg {
                        sum += input[(batch * p.cin + group * p.cpg + ci) * plane + pixel]
                            * weights[channel * p.cpg + ci];
                    }
                    output[(batch * p.cout + channel) * plane + pixel] = sum;
                }
            }
        }
        output
    }
    fn compare_conv1x1(
        dimensions: [usize; 4],
        channels: usize,
        groups: usize,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
    ) {
        let (shape, kind) = bind_conv(
            Shape::new(&dimensions).unwrap(),
            Shape::new(&[channels, dimensions[1] / groups, 1, 1]).unwrap(),
            (!bias.is_empty()).then(|| Shape::new(&[channels]).unwrap()),
            &ConvAttributes {
                kernel_shape: [1, 1],
                strides: [1, 1],
                dilations: [1, 1],
                pads: [0, 0, 0, 0],
                group: groups as i64,
            },
        )
        .unwrap();
        let BoundKind::Conv(plan) = kind else {
            panic!("expected convolution");
        };
        let expected = reference_conv1x1(&plan, input, weights, bias);
        let mut actual = vec![f32::NAN; shape.element_count()];
        conv(&plan, input, weights, bias, &mut actual);
        assert_bits(&actual, &expected);
    }
    #[test]
    fn four_channel_four_pixel_conv_preserves_group_batch_and_tail_bits() {
        for (batch, groups, input_channels, output_channels) in [
            (1, 1, 1, 1),
            (2, 1, 3, 7),
            (1, 1, 48, 96),
            (1, 2, 10, 14),
            (2, 3, 15, 27),
            (1, 4, 4, 4),
        ] {
            for (height, width) in [(1, 1), (1, 3), (1, 4), (1, 5), (2, 7), (3, 11)] {
                let input = distribution(batch * input_channels * height * width);
                let weights = (0..output_channels * input_channels / groups)
                    .map(|index| (index as i32 % 19 - 9) as f32 / 16.0)
                    .collect::<Vec<_>>();
                let bias = (0..output_channels)
                    .map(|index| (index as i32 % 7 - 3) as f32 / 8.0)
                    .collect::<Vec<_>>();
                for bias in [bias.as_slice(), &[]] {
                    compare_conv1x1(
                        [batch, input_channels, height, width],
                        output_channels,
                        groups,
                        &input,
                        &weights,
                        bias,
                    );
                }
            }
        }
    }
    #[test]
    fn tiled_conv_retains_bias_first_cancellation_and_signed_zero() {
        let dimensions = [2, 6, 1, 7];
        let input = (0..dimensions.iter().product::<usize>())
            .map(|index| [1e20f32, -1e20, 3.0][index / 7 % 3])
            .collect::<Vec<_>>();
        compare_conv1x1(dimensions, 10, 2, &input, &[1.0; 30], &[1.0; 10]);
        compare_conv1x1(
            dimensions,
            10,
            2,
            &vec![-0.0; input.len()],
            &[1.0; 30],
            &[-0.0; 10],
        );
    }
}
