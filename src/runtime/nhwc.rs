//! Channels-last kernels. Logical shapes always retain ONNX NCHW dimensions.
use super::kernels::{BinaryOp, BoundKind, Conv};
use super::optimized::Layout;
#[cfg(test)]
use super::simd::IdentityStore;
use super::simd::StoreTransform;
use super::*;
use std::simd::{Simd, StdFloat};
mod dense;
mod misc;
pub(super) mod pointwise;
mod strip;
mod transpose;

/// The reference balances complete six-pixel tiles, rather than rounding each
/// worker's row count up to a larger cache group. Its last tile may be short.
#[cfg(any(feature = "parallel", test))]
#[derive(Clone, Copy)]
struct PointwiseBands {
    rows: usize,
    workers: usize,
}

#[cfg(any(feature = "parallel", test))]
impl PointwiseBands {
    fn new(rows: usize, inner: usize, columns: usize, budget: usize) -> Self {
        let work = rows.saturating_mul(inner).saturating_mul(columns);
        let workers = if work >= 2_000_000 {
            budget.max(1).min(rows.div_ceil(6).max(1))
        } else {
            1
        };
        Self { rows, workers }
    }

    fn boundary(self, worker: usize) -> usize {
        let tiles = self.rows.div_ceil(6);
        // The widened product also covers theoretical sizes near usize::MAX.
        let tile = (tiles as u128 * worker as u128 / self.workers as u128) as usize;
        tile.saturating_mul(6).min(self.rows)
    }
}

/// Split exclusive output slices recursively. Rayon join uses stack jobs;
/// no per-call Vec of bands or heap-allocated scoped job list is needed.
#[cfg(feature = "parallel")]
fn pointwise_bands(
    output: &mut [f32],
    columns: usize,
    bands: PointwiseBands,
    action: impl Fn(&mut [f32], usize) + Sync,
) {
    fn visit(
        output: &mut [f32],
        columns: usize,
        bands: PointwiseBands,
        first_worker: usize,
        workers: usize,
        action: &(impl Fn(&mut [f32], usize) + Sync),
    ) {
        let first = bands.boundary(first_worker);
        if workers == 1 {
            action(output, first);
            return;
        }
        let left_workers = workers / 2;
        let middle_worker = first_worker + left_workers;
        let split = (bands.boundary(middle_worker) - first) * columns;
        let (left, right) = output.split_at_mut(split);
        rayon::join(
            || visit(left, columns, bands, first_worker, left_workers, action),
            || {
                visit(
                    right,
                    columns,
                    bands,
                    middle_worker,
                    workers - left_workers,
                    action,
                )
            },
        );
    }
    visit(output, columns, bands, 0, bands.workers, &action);
}

#[allow(clippy::too_many_arguments)]
fn pointwise_execute(
    backend: Backend,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    inner: usize,
    columns: usize,
    cpu: &CpuContext,
    threads: usize,
    store: &impl StoreTransform,
) {
    #[cfg(feature = "parallel")]
    {
        let bands = PointwiseBands::new(
            output.len() / columns,
            inner,
            columns,
            threads.max(1).min(cpu.capacity()),
        );
        if bands.workers > 1 {
            cpu.install(|| {
                pointwise_bands(output, columns, bands, |out, first| {
                    pointwise::execute(
                        backend, input, bias, packed, out, first, inner, columns, store,
                    );
                });
            });
            return;
        }
    }
    #[cfg(not(feature = "parallel"))]
    let _ = (cpu, threads);
    pointwise::execute(
        backend, input, bias, packed, output, 0, inner, columns, store,
    );
}

fn packed_pointwise_eligible(_backend: Backend, p: &Conv, packed: bool) -> bool {
    // Scalar reaches the same existing pointwise math through dense::execute;
    // share the reference six-pixel worker partition as well.
    packed
        && p.cpg == p.cin
        // This one-channel case selects depthwise before packed dense.
        && !(p.cin == 1 && p.opg == 1)
        && p.kh == 1
        && p.kw == 1
        && p.sh == 1
        && p.sw == 1
        && p.pt == 0
        && p.pl == 0
        && p.oh == p.height
        && p.ow == p.width
}

/// The pinned reference's constant rank-two MatMul uses its NHWC pointwise
/// GEMM only for complete16-column panels below the wide CTC vocabulary route.
/// Matrix rows remain ordinary row-major storage; no layout conversion occurs.
pub(super) fn pointwise_matrix_eligible(
    backend: Backend,
    packed: bool,
    rows: usize,
    inner: usize,
    columns: usize,
) -> bool {
    backend != Backend::Scalar
        && packed
        && rows > 0
        && inner > 0
        && (16..1024).contains(&columns)
        && columns.is_multiple_of(16)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn try_pointwise_matrix(
    backend: Backend,
    input: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
    cpu: &CpuContext,
    threads: usize,
) -> bool {
    if !pointwise_matrix_eligible(backend, true, rows, inner, columns) {
        return false;
    }
    assert!(
        packed.matches(inner, columns),
        "constant MatMul panel shape mismatch"
    );
    assert_eq!(
        input.len(),
        rows.checked_mul(inner).expect("matrix input overflow")
    );
    assert_eq!(
        output.len(),
        rows.checked_mul(columns).expect("matrix output overflow")
    );
    let supported = match backend {
        Backend::Scalar | Backend::Portable4 => true,
        Backend::Avx => simd::avx_available(),
        Backend::AvxFma => simd::avx_fma_available(),
        Backend::Avx2Fma => simd::avx2_fma_available(),
        Backend::Avx512 => simd::avx512_available(),
    };
    assert!(
        supported,
        "constant MatMul backend {backend:?} is unavailable"
    );
    pointwise_execute(
        backend,
        input,
        &[],
        packed,
        output,
        inner,
        columns,
        cpu,
        threads,
        &simd::IdentityStore,
    );
    true
}

pub(super) fn selection(
    kind: &BoundKind,
    backend: Backend,
    packed: bool,
) -> (&'static str, Backend) {
    match kind {
        BoundKind::Conv(p) if p.cpg == 1 && p.cout == p.cin && p.opg == 1 => (
            if p.kh >= 6 {
                "nhwc_depthwise_strip"
            } else {
                "nhwc_depthwise_pixel8"
            },
            if backend == Backend::Avx512 {
                Backend::Avx2Fma
            } else {
                backend
            },
        ),
        BoundKind::Conv(p) if packed && p.cpg == p.cin => (
            if p.kh == 1
                && p.kw == 1
                && p.sh == 1
                && p.sw == 1
                && p.pt == 0
                && p.pl == 0
                && p.oh == p.height
                && p.ow == p.width
            {
                if backend == Backend::Portable4 {
                    "nhwc_pointwise_6x8_pair2_group8"
                } else {
                    "nhwc_pointwise_6x16_group8"
                }
            } else {
                if backend == Backend::Portable4 {
                    "nhwc_dense_6x8_pair2_k512"
                } else {
                    "nhwc_dense_6x16_k512"
                }
            },
            backend,
        ),
        BoundKind::Conv(_) => ("nhwc_grouped_fallback", backend),
        BoundKind::ConvTranspose(p) if p.cout == 1 => (
            "nhwc_transpose_single_spatial",
            if backend == Backend::Avx512 {
                Backend::Avx2Fma
            } else {
                backend
            },
        ),
        BoundKind::ConvTranspose(p) if packed && p.cout >= 16 => (
            "nhwc_transpose_4tap_matrix",
            if backend == Backend::Avx512 {
                Backend::Avx2Fma
            } else {
                backend
            },
        ),
        BoundKind::ConvTranspose(_) => ("nhwc_transpose_scalar", Backend::Scalar),
        BoundKind::Reduce(_) => (
            "nhwc_spatial_mean",
            if backend == Backend::Avx512 {
                Backend::Avx2Fma
            } else {
                backend
            },
        ),
        BoundKind::MaxPool { .. } | BoundKind::AveragePool { .. } => (
            "nhwc_pool_channels",
            if backend == Backend::Avx512 {
                Backend::Avx2Fma
            } else {
                backend
            },
        ),
        BoundKind::Resize { .. } => ("nhwc_nearest_copy", Backend::Portable4),
        BoundKind::BatchNormalization { .. } => (
            "nhwc_batch_norm_affine",
            if backend == Backend::Avx512 {
                Backend::Avx2Fma
            } else {
                backend
            },
        ),
        BoundKind::Binary(..) => ("nhwc_binary", backend),
        _ => ("nhwc_native_fallback", backend),
    }
}

/// Reference NHWC dense/depthwise dispatch thresholds. Complete output-row
/// bands avoid splitting a receptive-field row and need no heap job storage.
fn conv_row_workers(p: &Conv, budget: usize) -> usize {
    let work = [p.batch, p.oh, p.ow, p.cout, p.cpg, p.kh, p.kw]
        .into_iter()
        .fold(1u128, |total, value| total.saturating_mul(value as u128));
    // Group-one C1 is dense; the planner normally excludes that neutral
    // case, but keeping the predicate literal makes standalone policy queries
    // agree with the source group's dense-before-depthwise dispatch.
    let depthwise = p.cin > 1 && p.cpg == 1 && p.opg == 1 && p.cout == p.cin;
    let threshold = if depthwise { 1_000_000 } else { 2_000_000 };
    if work < threshold {
        1
    } else {
        budget.max(1).min(p.batch.saturating_mul(p.oh).max(1))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute<'a>(
    compiled: &CompiledModel,
    index: usize,
    kind: &BoundKind,
    info: impl Fn(usize) -> (Shape, Layout),
    data: impl Fn(usize) -> &'a [f32] + Sync,
    out: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
    workspaces: &mut [Vec<f32>],
    store: &impl StoreTransform,
) -> RuntimeResult<()> {
    let a = data(0);
    let shape = info(0).0;
    #[cfg(feature = "parallel")]
    let d = shape.dimensions();
    match kind {
        BoundKind::Conv(p) => {
            let weights = data(1);
            let bias = if compiled.instructions[index].inputs.len() > 2 {
                data(2)
            } else {
                &[]
            };
            let packed = compiled.conv_packed.get(&index).map(Arc::as_ref);
            if workspaces.is_empty() {
                return Err(RuntimeError::new(
                    "NHWC convolution requires a reserved worker workspace",
                ));
            }
            if let Some(panel) =
                packed.filter(|_| packed_pointwise_eligible(compiled.backend, p, true))
            {
                pointwise_execute(
                    compiled.backend,
                    a,
                    bias,
                    panel,
                    out,
                    p.cin,
                    p.cout,
                    cpu,
                    threads.max(1).min(workspaces.len()),
                    store,
                );
                return Ok(());
            }
            let workers = conv_row_workers(p, threads.min(cpu.capacity()).min(workspaces.len()));
            if workers > 1 {
                // These production convolution stores use fixed local tiles;
                // conv_dispatch_store's workspace parameter is currently unused.
                // A distinct empty slice per closure preserves exclusive borrows
                // while keeping the session's reserved workspace count unchanged.
                super::parallel::aligned_bands(
                    out,
                    p.ow * p.cout,
                    p.batch * p.oh,
                    workers,
                    1,
                    cpu,
                    |dst, row| {
                        conv_dispatch_store(
                            compiled.backend,
                            p,
                            a,
                            weights,
                            bias,
                            packed,
                            dst,
                            row * p.ow,
                            &mut [],
                            store,
                        );
                    },
                );
                return Ok(());
            }
            conv_dispatch_store(
                compiled.backend,
                p,
                a,
                weights,
                bias,
                packed,
                out,
                0,
                &mut workspaces[0],
                store,
            );
        }
        BoundKind::ConvTranspose(p) => {
            let weights = data(1);
            let bias = if compiled.instructions[index].inputs.len() > 2 {
                data(2)
            } else {
                &[]
            };
            if p.cout == 1 {
                #[cfg(feature = "parallel")]
                if threads > 1 && p.batch * p.height * p.width * p.cin >= 1_000_000 {
                    use rayon::prelude::*;
                    let rows = p.batch * p.height;
                    let workers = threads.min(cpu.capacity()).min(rows);
                    let chunk = rows.div_ceil(workers).max(1);
                    cpu.install(|| {
                        out.par_chunks_mut(chunk * 4 * p.width)
                            .enumerate()
                            .for_each(|(band, dst)| {
                                transpose::single(
                                    compiled.backend,
                                    p,
                                    a,
                                    weights,
                                    bias,
                                    dst,
                                    band * chunk,
                                    store,
                                );
                            })
                    });
                    return Ok(());
                }
                transpose::single(compiled.backend, p, a, weights, bias, out, 0, store);
                return Ok(());
            }
            if let Some(panels) = compiled
                .transpose_packed
                .get(&index)
                .filter(|_| p.cout >= 16)
            {
                let rows = p.batch * p.height;
                let workers = threads
                    .max(1)
                    .min(cpu.capacity())
                    .min(workspaces.len())
                    .min(rows);
                if workers == 0 || workspaces[..workers].iter().any(|v| v.len() < 32 * p.cout) {
                    return Err(RuntimeError::new(
                        "NHWC transpose convolution requires reserved 32-row output scratch",
                    ));
                }
                #[cfg(feature = "parallel")]
                if workers > 1 && rows * p.width * p.cin * p.cout * 4 >= 2_000_000 {
                    use rayon::prelude::*;
                    let row_chunk = rows.div_ceil(workers);
                    cpu.install(|| {
                        out.par_chunks_mut(row_chunk * 4 * p.width * p.cout)
                            .zip(workspaces[..workers].par_iter_mut())
                            .enumerate()
                            .for_each(|(band, (destination, workspace))| {
                                transpose_packed_rows_store(
                                    compiled.backend,
                                    p,
                                    a,
                                    bias,
                                    panels,
                                    destination,
                                    band * row_chunk,
                                    workspace,
                                    store,
                                );
                            });
                    });
                    return Ok(());
                }
                transpose_packed_rows_store(
                    compiled.backend,
                    p,
                    a,
                    bias,
                    panels,
                    out,
                    0,
                    &mut workspaces[0],
                    store,
                );
            } else {
                transpose_scalar_store(p, a, weights, bias, out, store);
            }
        }
        BoundKind::MaxPool { .. } | BoundKind::AveragePool { .. } | BoundKind::Resize { .. } => {
            #[cfg(feature = "parallel")]
            if threads > 1 && out.len() >= 262_144 {
                use rayon::prelude::*;
                let (height, width) = match kind {
                    BoundKind::MaxPool { height, width } => (*height, *width),
                    BoundKind::AveragePool {
                        output_height,
                        output_width,
                        ..
                    } => (*output_height, *output_width),
                    BoundKind::Resize {
                        height,
                        width,
                        factor,
                    } => (height * factor, width * factor),
                    _ => unreachable!(),
                };
                let rows = d[0] * height;
                let workers = threads.min(cpu.capacity()).min(rows);
                let chunk = rows.div_ceil(workers).max(1);
                cpu.install(|| {
                    out.par_chunks_mut(chunk * width * d[1])
                        .enumerate()
                        .for_each(|(band, dst)| {
                            misc::execute_range(
                                compiled.backend,
                                kind,
                                a,
                                shape,
                                dst,
                                band * chunk * width,
                            );
                        })
                });
                return Ok(());
            }
            misc::execute(compiled.backend, kind, a, shape, out);
        }
        BoundKind::Reduce(_) => {
            misc::execute(compiled.backend, kind, a, shape, out);
        }
        BoundKind::BatchNormalization {
            channels, epsilon, ..
        } => {
            if workspaces
                .first()
                .is_none_or(|workspace| workspace.len() < *channels)
            {
                return Err(RuntimeError::new(
                    "NHWC BatchNorm requires reserved channel multipliers",
                ));
            }
            misc::batch_norm(
                compiled.backend,
                a,
                out,
                *channels,
                data(1),
                data(2),
                data(3),
                data(4),
                *epsilon,
                &mut workspaces[0],
            );
        }
        BoundKind::Binary(plan, op) => {
            let b = data(1);
            let operation = match op {
                BinaryOp::Add => simd::BinaryOp::Add,
                BinaryOp::Mul => simd::BinaryOp::Mul,
                BinaryOp::Div => simd::BinaryOp::Div,
                BinaryOp::Sub => simd::BinaryOp::Sub,
            };
            if a.len() == out.len() && b.len() == out.len() && info(1).1 == Layout::Nhwc {
                simd::parallel::binary(compiled.backend, operation, a, b, out, cpu, threads);
            } else if b.len() == 1 {
                simd::binary_scalar(compiled.backend, operation, a, b[0], false, out);
            } else if a.len() == 1 {
                simd::binary_scalar(compiled.backend, operation, b, a[0], true, out);
            } else if let Some(channel) = plan.channel() {
                let (full, gate) = if channel.gate_left { (b, a) } else { (a, b) };
                simd::binary_channel(compiled.backend, operation, full, gate, channel, out);
            } else {
                binary_generic(plan, *op, &info, a, b, out);
            }
        }
        _ => {
            let logical = &compiled.instructions[index].inputs;
            kernels::execute_optimized(
                kind,
                logical,
                |id| data(logical.iter().position(|&v| v == id).unwrap()),
                out,
                compiled.backend,
                logical.get(1).and_then(|id| compiled.packed_matrix(*id)),
                cpu,
                threads,
            );
        }
    }
    Ok(())
}

/// General logical-coordinate fallback, also retained as the measured oracle
/// for the channel specialization. No activation or coefficient is copied.
pub(super) fn binary_generic(
    plan: &kernels::Broadcast,
    op: BinaryOp,
    info: impl Fn(usize) -> (Shape, Layout),
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
) {
    let od = plan.dimensions;
    let c = od[1];
    let plane = od[2] * od[3];
    for (physical, y) in out.iter_mut().enumerate() {
        let pixel = physical / c;
        let n = pixel / plane;
        let p = pixel % plane;
        let coords = [n, physical % c, p / od[3], p % od[3]];
        let (mut ai, mut bi) = (0, 0);
        for (axis, &coordinate) in coords.iter().enumerate().take(plan.rank) {
            ai += coordinate * plan.a_strides[axis];
            bi += coordinate * plan.b_strides[axis];
        }
        let physical_index = |slot: usize, logical: usize| {
            let (shape, layout) = info(slot);
            if layout == Layout::Canonical || shape.rank() != 4 {
                logical
            } else {
                let sd = shape.dimensions();
                let sp = sd[2] * sd[3];
                ((logical / (sd[1] * sp) * sp + logical % sp) * sd[1]) + (logical / sp % sd[1])
            }
        };
        let av = a[physical_index(0, ai)];
        let bv = b[physical_index(1, bi)];
        *y = match op {
            BinaryOp::Add => av + bv,
            BinaryOp::Mul => av * bv,
            BinaryOp::Div => av / bv,
            BinaryOp::Sub => av - bv,
        };
    }
}

#[cfg(test)]
pub(super) fn transpose_scalar(
    p: &kernels::ConvTranspose,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    transpose_scalar_store(p, input, weights, bias, out, &IdentityStore);
}

fn transpose_scalar_store(
    p: &kernels::ConvTranspose,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    out: &mut [f32],
    store: &impl StoreTransform,
) {
    for n in 0..p.batch {
        for y in 0..p.height * 2 {
            for x in 0..p.width * 2 {
                let tap = y % 2 * 2 + x % 2;
                let source = ((n * p.height + y / 2) * p.width + x / 2) * p.cin;
                let destination = ((n * p.height * 2 + y) * p.width * 2 + x) * p.cout;
                for co in 0..p.cout {
                    let mut sum = if bias.is_empty() { 0.0 } else { bias[co] };
                    for ci in 0..p.cin {
                        sum += input[source + ci] * weights[(ci * p.cout + co) * 4 + tap];
                    }
                    out[destination + co] = store.scalar(sum, destination + co);
                }
            }
        }
    }
}

/// A whole input-row band maps to an exclusive contiguous pair of output
/// rows. Only the 32-row result tile is materialized, in reserved storage.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn transpose_packed_rows(
    backend: Backend,
    p: &kernels::ConvTranspose,
    input: &[f32],
    bias: &[f32],
    panels: &[PackedMatrix; 4],
    out: &mut [f32],
    first_row: usize,
    workspace: &mut [f32],
) {
    transpose_packed_rows_store(
        backend,
        p,
        input,
        bias,
        panels,
        out,
        first_row,
        workspace,
        &IdentityStore,
    );
}

#[allow(clippy::too_many_arguments)]
fn transpose_packed_rows_store(
    backend: Backend,
    p: &kernels::ConvTranspose,
    input: &[f32],
    bias: &[f32],
    panels: &[PackedMatrix; 4],
    out: &mut [f32],
    first_row: usize,
    workspace: &mut [f32],
    store: &impl StoreTransform,
) {
    let backend = if backend == Backend::Avx512 {
        Backend::Avx2Fma
    } else {
        backend
    };
    assert!(out.len().is_multiple_of(4 * p.width * p.cout));
    let pixels = out.len() / (4 * p.cout);
    let first = first_row * p.width;
    assert!(first + pixels <= p.batch * p.height * p.width);
    assert!(workspace.len() >= 32 * p.cout);
    for (tile, start) in (0..pixels).step_by(32).enumerate() {
        let rows = 32.min(pixels - start);
        let source = &input[(first + start) * p.cin..(first + start + rows) * p.cin];
        let temporary = &mut workspace[..rows * p.cout];
        for (tap, panel) in panels.iter().enumerate() {
            simd::matmul_bias(
                backend,
                source,
                &[],
                Some(panel),
                bias,
                temporary,
                rows,
                p.cin,
                p.cout,
            );
            for (row, values) in temporary.chunks_exact(p.cout).enumerate() {
                let local = tile * 32 + row;
                let y = local / p.width;
                let x = local % p.width;
                let destination = ((y * 2 + tap / 2) * p.width * 2 + x * 2 + tap % 2) * p.cout;
                let global = first_row * 4 * p.width * p.cout + destination;
                for (co, &value) in values.iter().enumerate() {
                    out[destination + co] = store.scalar(value, global + co);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn conv_dispatch(
    backend: Backend,
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    first: usize,
    workspace: &mut [f32],
) {
    conv_dispatch_store(
        backend,
        p,
        input,
        weights,
        bias,
        packed,
        out,
        first,
        workspace,
        &IdentityStore,
    );
}

#[allow(clippy::too_many_arguments)]
fn conv_dispatch_store(
    backend: Backend,
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    first: usize,
    _workspace: &mut [f32],
    store: &impl StoreTransform,
) {
    let groups = p.cin / p.cpg;
    if groups == p.cin && p.opg == 1 {
        depthwise_dispatch(backend, p, input, weights, bias, packed, out, first, store);
        return;
    }
    if let Some(panel) = packed.filter(|_| groups == 1) {
        dense::execute(backend, p, input, bias, panel, out, first, store);
        return;
    }
    match backend {
        Backend::Avx2Fma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validates the selected backend's target features.
            unsafe {
                conv_avx2(p, input, weights, bias, packed, out, first, store);
            }
        }
        Backend::Avx => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validates the selected backend's target features.
            unsafe {
                conv_avx(p, input, weights, bias, packed, out, first, store);
            }
        }
        Backend::AvxFma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validates the selected backend's target features.
            unsafe {
                conv_avx_fma(p, input, weights, bias, packed, out, first, store);
            }
        }
        Backend::Avx512 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validates the selected backend's target features.
            unsafe {
                conv_avx512(p, input, weights, bias, packed, out, first, store);
            }
        }
        Backend::Portable4 => conv::<4, false>(p, input, weights, bias, packed, out, first, store),
        Backend::Scalar => conv::<1, false>(p, input, weights, bias, packed, out, first, store),
    }
}

/// Materialize only one small row panel in the session's reserved storage.
/// K remains `[input_channel][kernel_y][kernel_x]`, matching the original
/// coefficient order. Padding contributes zero coefficients in this optimized
/// path; the diagnostic/reference executor retains tap skipping.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn dense_im2row(
    backend: Backend,
    p: &Conv,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    out: &mut [f32],
    first: usize,
    workspace: &mut [f32],
    store: &impl StoreTransform,
) {
    let inner = p.cin * p.kh * p.kw;
    assert!(packed.matches(inner, p.cout));
    let tile_rows = 32.min(out.len() / p.cout).max(1);
    for (tile, destination) in out.chunks_mut(tile_rows * p.cout).enumerate() {
        let rows = destination.len() / p.cout;
        let panel = &mut workspace[..rows * inner];
        for (row, patch) in panel.chunks_exact_mut(inner).enumerate() {
            let pixel = first + tile * tile_rows + row;
            let n = pixel / (p.oh * p.ow);
            let y = pixel / p.ow % p.oh;
            let x = pixel % p.ow;
            for ci in 0..p.cin {
                for ky in 0..p.kh {
                    let sy = y * p.sh + ky * p.dh;
                    let values = &mut patch[(ci * p.kh + ky) * p.kw..(ci * p.kh + ky + 1) * p.kw];
                    if sy < p.pt || sy - p.pt >= p.height {
                        values.fill(0.0);
                        continue;
                    }
                    for (kx, value) in values.iter_mut().enumerate() {
                        let sx = x * p.sw + kx * p.dw;
                        *value = if sx < p.pl || sx - p.pl >= p.width {
                            0.0
                        } else {
                            input[((n * p.height + sy - p.pt) * p.width + sx - p.pl) * p.cin + ci]
                        };
                    }
                }
            }
        }
        simd::matmul_bias_store(
            backend,
            panel,
            &[],
            Some(packed),
            bias,
            destination,
            rows,
            inner,
            p.cout,
            store,
            (first + tile * tile_rows) * p.cout,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn depthwise_dispatch(
    backend: Backend,
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    if let Some(panel) = packed {
        assert!(panel.matches(p.kh * p.kw, p.cout));
    }
    match backend {
        Backend::Scalar => {
            depthwise::<1, false>(p, input, weights, bias, packed, out, first, store)
        }
        Backend::Portable4 => {
            depthwise::<4, false>(p, input, weights, bias, packed, out, first, store)
        }
        Backend::Avx2Fma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validates the selected backend's target features.
            unsafe {
                depthwise_avx2(p, input, weights, bias, packed, out, first, store)
            }
        }
        Backend::Avx => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validates Avx before this selection.
            unsafe {
                depthwise_avx(p, input, weights, bias, packed, out, first, store)
            }
        }
        Backend::AvxFma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validates AvxFma before this selection.
            unsafe {
                depthwise_avx_fma(p, input, weights, bias, packed, out, first, store)
            }
        }
        Backend::Avx512 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: CompiledModel validates Avx512 before this selection.
            unsafe {
                depthwise_avx512(p, input, weights, bias, packed, out, first, store)
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn depthwise_avx2(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    depthwise::<8, true>(p, input, weights, bias, packed, out, first, store);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn depthwise_avx(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    depthwise::<8, false>(p, input, weights, bias, packed, out, first, store);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn depthwise_avx_fma(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    depthwise::<8, true>(p, input, weights, bias, packed, out, first, store);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn depthwise_avx512(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    depthwise::<8, true>(p, input, weights, bias, packed, out, first, store);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn depthwise<const LANES: usize, const FUSED: bool>(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    if p.kh >= 6 {
        strip::execute::<LANES, FUSED>(p, input, weights, bias, packed, out, first, store);
        return;
    }
    let pixels = out.len() / p.cout;
    let vector_channels = p.cout / LANES * LANES;
    let mut local = 0;
    while local < pixels {
        let pixel = first + local;
        let batch = pixel / (p.oh * p.ow);
        let y = pixel / p.ow % p.oh;
        let x = pixel % p.ow;
        let tile = if (pixels - local).min(p.ow - x) >= 8 {
            8
        } else {
            1
        };
        for channel in (0..vector_channels).step_by(LANES) {
            if tile == 8 {
                depthwise_pixels::<LANES, FUSED, 8>(
                    p, input, weights, bias, packed, out, local, batch, y, x, channel, first, store,
                );
            } else {
                depthwise_pixels::<LANES, FUSED, 1>(
                    p, input, weights, bias, packed, out, local, batch, y, x, channel, first, store,
                );
            }
        }
        // The original optimized CPU kernels use scalar arithmetic for the
        // remaining channels; preserve that tail contract here as well.
        for channel in vector_channels..p.cout {
            if tile == 8 {
                depthwise_pixels::<1, false, 8>(
                    p, input, weights, bias, packed, out, local, batch, y, x, channel, first, store,
                );
            } else {
                depthwise_pixels::<1, false, 1>(
                    p, input, weights, bias, packed, out, local, batch, y, x, channel, first, store,
                );
            }
        }
        local += tile;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn depthwise_pixels<const LANES: usize, const FUSED: bool, const PIXELS: usize>(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    local: usize,
    batch: usize,
    y: usize,
    x: usize,
    channel: usize,
    first: usize,
    store: &impl StoreTransform,
) {
    let initial: Simd<f32, LANES> = if bias.is_empty() {
        Simd::splat(0.0)
    } else {
        Simd::from_slice(&bias[channel..channel + LANES])
    };
    let mut sums = [initial; PIXELS];
    for ky in 0..p.kh {
        let sy = y * p.sh + ky * p.dh;
        if sy < p.pt || sy - p.pt >= p.height {
            continue;
        }
        for kx in 0..p.kw {
            let tap = ky * p.kw + kx;
            let weight = if let Some(panel) = packed {
                let start = ((channel / 16) * p.kh * p.kw + tap) * 16 + channel % 16;
                Simd::from_slice(&panel.data()[start..start + LANES])
            } else {
                Simd::from_array(std::array::from_fn(|lane| {
                    weights[(channel + lane) * p.kh * p.kw + tap]
                }))
            };
            for (pixel, sum) in sums.iter_mut().enumerate() {
                let sx = (x + pixel) * p.sw + kx * p.dw;
                if sx < p.pl || sx - p.pl >= p.width {
                    continue;
                }
                let start =
                    ((batch * p.height + sy - p.pt) * p.width + sx - p.pl) * p.cin + channel;
                let value = Simd::from_slice(&input[start..start + LANES]);
                *sum = if FUSED {
                    value.mul_add(weight, *sum)
                } else {
                    value * weight + *sum
                };
            }
        }
    }
    for (pixel, sum) in sums.into_iter().enumerate() {
        let start = (local + pixel) * p.cout + channel;
        store
            .vector(sum, first * p.cout + start)
            .copy_to_slice(&mut out[start..start + LANES]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_avx2(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    conv::<8, true>(p, input, weights, bias, packed, out, first, store);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_avx(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    conv::<8, false>(p, input, weights, bias, packed, out, first, store);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_avx_fma(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    conv::<8, true>(p, input, weights, bias, packed, out, first, store);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_avx512(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    conv::<16, true>(p, input, weights, bias, packed, out, first, store);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn conv<const L: usize, const FMA: bool>(
    p: &Conv,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    packed: Option<&PackedMatrix>,
    out: &mut [f32],
    first: usize,
    store: &impl StoreTransform,
) {
    let groups = p.cin / p.cpg;
    let depthwise = groups == p.cin && p.opg == 1;
    for (local, destination) in out.chunks_exact_mut(p.cout).enumerate() {
        let pixel = first + local;
        let n = pixel / (p.oh * p.ow);
        let y = pixel / p.ow % p.oh;
        let x = pixel % p.ow;
        let mut co = 0;
        while co + L <= p.cout && (groups == 1 || depthwise) {
            let mut sum = if bias.is_empty() {
                Simd::<f32, L>::splat(0.)
            } else {
                Simd::from_slice(&bias[co..])
            };
            if depthwise {
                for ky in 0..p.kh {
                    let sy = y * p.sh + ky * p.dh;
                    if sy < p.pt || sy - p.pt >= p.height {
                        continue;
                    }
                    for kx in 0..p.kw {
                        let sx = x * p.sw + kx * p.dw;
                        if sx < p.pl || sx - p.pl >= p.width {
                            continue;
                        }
                        let at = ((n * p.height + sy - p.pt) * p.width + sx - p.pl) * p.cin + co;
                        let w = Simd::from_array(std::array::from_fn(|lane| {
                            weights[(co + lane) * p.kh * p.kw + ky * p.kw + kx]
                        }));
                        let a = Simd::from_slice(&input[at..]);
                        sum = if FMA { a.mul_add(w, sum) } else { a * w + sum };
                    }
                }
            } else {
                for ci in 0..p.cin {
                    for ky in 0..p.kh {
                        let sy = y * p.sh + ky * p.dh;
                        if sy < p.pt || sy - p.pt >= p.height {
                            continue;
                        }
                        for kx in 0..p.kw {
                            let sx = x * p.sw + kx * p.dw;
                            if sx < p.pl || sx - p.pl >= p.width {
                                continue;
                            }
                            let k = (ci * p.kh + ky) * p.kw + kx;
                            let w = if let Some(pack) = packed {
                                Simd::from_slice(
                                    &pack.data()
                                        [((co / 16) * p.cin * p.kh * p.kw + k) * 16 + co % 16..],
                                )
                            } else {
                                Simd::from_array(std::array::from_fn(|lane| {
                                    weights[(co + lane) * p.cin * p.kh * p.kw + k]
                                }))
                            };
                            let a = Simd::splat(
                                input[((n * p.height + sy - p.pt) * p.width + sx - p.pl) * p.cin
                                    + ci],
                            );
                            sum = if FMA { a.mul_add(w, sum) } else { a * w + sum };
                        }
                    }
                }
            }
            store
                .vector(sum, pixel * p.cout + co)
                .copy_to_slice(&mut destination[co..co + L]);
            co += L;
        }
        for co in co..p.cout {
            let group = co / p.opg;
            let mut sum = if bias.is_empty() { 0. } else { bias[co] };
            for ci in 0..p.cpg {
                for ky in 0..p.kh {
                    let sy = y * p.sh + ky * p.dh;
                    if sy < p.pt || sy - p.pt >= p.height {
                        continue;
                    }
                    for kx in 0..p.kw {
                        let sx = x * p.sw + kx * p.dw;
                        if sx < p.pl || sx - p.pl >= p.width {
                            continue;
                        }
                        sum += input[((n * p.height + sy - p.pt) * p.width + sx - p.pl) * p.cin
                            + group * p.cpg
                            + ci]
                            * weights[((co * p.cpg + ci) * p.kh + ky) * p.kw + kx];
                    }
                }
            }
            destination[co] = store.scalar(sum, pixel * p.cout + co);
        }
    }
}

#[cfg(test)]
mod parallel_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod row_policy_tests;
