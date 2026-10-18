//! Full-K AVX2 tiles with an inline, register-contained reduction loop.
use super::*;
use std::arch::{asm, x86_64::__m256};

// Explicit register operands keep loop geometry and epilogue state outside K
// without imposing a function/return-buffer boundary on every six-row tile.
macro_rules! tile {
    ($rows:expr, $panel:expr, $low:expr, $high:expr, $inner:expr) => {{
        let rows: [&[f32]; ROWS] = $rows;
        let panel: &[f32] = $panel;
        // SAFETY: both vector representations hold the same eight f32 lanes.
        let (low, high) = unsafe {
            (std::mem::transmute::<Simd<f32, 8>, __m256>($low),
             std::mem::transmute::<Simd<f32, 8>, __m256>($high))
        };
        let (mut a0, mut b0) = (low, high);
        let (mut a1, mut b1) = (low, high);
        let (mut a2, mut b2) = (low, high);
        let (mut a3, mut b3) = (low, high);
        let (mut a4, mut b4) = (low, high);
        let (mut a5, mut b5) = (low, high);
        // SAFETY: the AVX2/FMA entry checks inner > 0. Each row contains inner
        // floats; panel contains inner * 16. k starts at zero and advances to
        // inner, so every scalar/vector load stays within its checked span.
        // The assembly only reads memory, declares every modified register and
        // flags, and performs the same bias-first ascending-K FMA chain.
        unsafe {
            asm!(
                "xor {k:e}, {k:e}",
                ".p2align 4",
                "2:",
                "vmovups {w0}, [{weights}]",
                "vmovups {w1}, [{weights} + 32]",
                "vbroadcastss {v}, [{p0} + {k} * 4]",
                "vfmadd231ps {a0}, {v}, {w0}",
                "vfmadd231ps {b0}, {v}, {w1}",
                "vbroadcastss {v}, [{p1} + {k} * 4]",
                "vfmadd231ps {a1}, {v}, {w0}",
                "vfmadd231ps {b1}, {v}, {w1}",
                "vbroadcastss {v}, [{p2} + {k} * 4]",
                "vfmadd231ps {a2}, {v}, {w0}",
                "vfmadd231ps {b2}, {v}, {w1}",
                "vbroadcastss {v}, [{p3} + {k} * 4]",
                "vfmadd231ps {a3}, {v}, {w0}",
                "vfmadd231ps {b3}, {v}, {w1}",
                "vbroadcastss {v}, [{p4} + {k} * 4]",
                "vfmadd231ps {a4}, {v}, {w0}",
                "vfmadd231ps {b4}, {v}, {w1}",
                "vbroadcastss {v}, [{p5} + {k} * 4]",
                "vfmadd231ps {a5}, {v}, {w0}",
                "vfmadd231ps {b5}, {v}, {w1}",
                "inc {k}",
                "add {weights}, 64",
                "cmp {k}, {inner}",
                "jb 2b",
                a0 = inout(ymm_reg) a0, b0 = inout(ymm_reg) b0,
                a1 = inout(ymm_reg) a1, b1 = inout(ymm_reg) b1,
                a2 = inout(ymm_reg) a2, b2 = inout(ymm_reg) b2,
                a3 = inout(ymm_reg) a3, b3 = inout(ymm_reg) b3,
                a4 = inout(ymm_reg) a4, b4 = inout(ymm_reg) b4,
                a5 = inout(ymm_reg) a5, b5 = inout(ymm_reg) b5,
                w0 = out(ymm_reg) _, w1 = out(ymm_reg) _, v = out(ymm_reg) _,
                p0 = in(reg) rows[0].as_ptr(), p1 = in(reg) rows[1].as_ptr(),
                p2 = in(reg) rows[2].as_ptr(), p3 = in(reg) rows[3].as_ptr(),
                p4 = in(reg) rows[4].as_ptr(), p5 = in(reg) rows[5].as_ptr(),
                weights = inout(reg) panel.as_ptr() => _,
                inner = in(reg) $inner, k = out(reg) _,
                options(nostack, readonly),
            );
        }
        // SAFETY: arrays of the two eight-f32 vector types have identical layout.
        unsafe { std::mem::transmute::<[[__m256; 2]; ROWS], [[Simd<f32, 8>; 2]; ROWS]>(
            [[a0, b0], [a1, b1], [a2, b2], [a3, b3], [a4, b4], [a5, b5]]) }
    }};
}

/// Caller must establish AVX2/FMA; all data spans are checked before assembly.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn execute(
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    inner: usize,
    columns: usize,
    store: &impl StoreTransform,
) {
    assert!(inner > 0 && columns > 0 && packed.matches(inner, columns));
    assert_eq!(output.len() % columns, 0);
    assert!(bias.is_empty() || bias.len() == columns);
    let rows = output.len() / columns;
    assert_eq!(
        input.len(),
        rows.checked_mul(inner)
            .expect("pointwise input size overflow")
    );
    let group_rows = avx_group_rows(inner, columns);
    for group in (0..rows).step_by(group_rows) {
        let count = (rows - group).min(group_rows);
        let input = &input[group * inner..(group + count) * inner];
        for (block, panel) in packed.data().chunks_exact(inner * 16).enumerate() {
            let column = block * 16;
            let low = initial::<8>(bias, column);
            let high = initial::<8>(bias, column + 8);
            for tile in (0..count).step_by(ROWS) {
                let [[a0, b0], [a1, b1], [a2, b2], [a3, b3], [a4, b4], [a5, b5]] = tile!(
                    row_inputs(input, tile, count, inner),
                    panel,
                    low,
                    high,
                    inner
                );
                // Named rows allow direct stores without an addressable tile
                // array. Tail checks never enter the reduction loop.
                macro_rules! store {
                    ($r:literal, $a:ident, $b:ident) => {
                        if tile + $r < count {
                            store_row($a, output, group + tile + $r, column, columns, first, store);
                            store_row(
                                $b,
                                output,
                                group + tile + $r,
                                column + 8,
                                columns,
                                first,
                                store,
                            );
                        }
                    };
                }
                store!(0, a0, b0);
                store!(1, a1, b1);
                store!(2, a2, b2);
                store!(3, a3, b3);
                store!(4, a4, b4);
                store!(5, a5, b5);
            }
        }
    }
}

#[cfg(test)]
mod tests;
