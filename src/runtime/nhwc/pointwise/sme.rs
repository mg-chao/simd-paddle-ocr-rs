//! macOS SME 32x32 pointwise blocks with reusable input packing.
//!
//! The same immutable 16-column weight panels and bias-first ascending FMA
//! order are used by NEON. Four independent 16x16 accumulators reuse each
//! input/weight vector twice; short row or channel tails retain a single tile.
//! Small shapes and unsupported thread state fall back.
#[cfg(feature = "parallel")]
use super::super::PointwiseBands;
use super::*;
use std::arch::asm;

pub(super) const GROUP: usize = 64;

pub(super) fn available() -> bool {
    std::arch::is_aarch64_feature_detected!("sme")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_execute(
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    inner: usize,
    columns: usize,
    scratch: &mut [f32],
    store: &impl StoreTransform,
) -> bool {
    if !available() || inner == 0 || columns == 0 {
        return false;
    }
    let Some(required) = GROUP.checked_mul(inner) else {
        return false;
    };
    if scratch.len() < required {
        return false;
    }
    assert!(packed.matches(inner, columns));
    assert_eq!(output.len() % columns, 0);
    assert!(bias.is_empty() || bias.len() == columns);
    let rows = output.len() / columns;
    let start = first.checked_mul(inner).expect("SME input offset overflow");
    let length = rows.checked_mul(inner).expect("SME input size overflow");
    let input = &input[start..start.checked_add(length).expect("SME input end overflow")];
    for group in (0..rows).step_by(GROUP) {
        let count = (rows - group).min(GROUP);
        for t in (0..count).step_by(16) {
            let destination = &mut scratch[t * inner..(t + 16) * inner];
            for k in 0..inner {
                for r in 0..16 {
                    destination[k * 16 + r] = input[(group + (t + r).min(count - 1)) * inner + k];
                }
            }
        }
        for (block, panels) in packed.data().chunks(inner * 32).enumerate() {
            let column = block * 32;
            let seed: [f32; 32] =
                std::array::from_fn(|v| bias.get(column + v).copied().unwrap_or(0.));
            let mut t = 0;
            while t < count {
                if count - t > 16 && panels.len() == inner * 32 {
                    let mut sums = [0f32; 1024];
                    // SAFETY: two full packed input/weight panels and fixed
                    // arrays cover all four tiles. Each panel has inner*16
                    // values, including duplicated input rows at the tail.
                    if !unsafe {
                        tile32(
                            scratch[t * inner..].as_ptr(),
                            panels.as_ptr(),
                            seed.as_ptr(),
                            sums.as_mut_ptr(),
                            inner,
                        )
                    } {
                        return false;
                    }
                    for r in 0..(count - t).min(32) {
                        for v in 0..8 {
                            let tile = r / 16 * 2 + v / 4;
                            let offset = tile * 256 + r % 16 * 16 + v % 4 * 4;
                            store_row(
                                Simd::<f32, 4>::from_slice(&sums[offset..]),
                                output,
                                group + t + r,
                                column + v * 4,
                                columns,
                                first,
                                store,
                            );
                        }
                    }
                    t += 32;
                    continue;
                }
                for (panel_index, panel) in panels.chunks_exact(inner * 16).enumerate() {
                    let mut sums = [0f32; 256];
                    // SAFETY: checked full input/weight panels and fixed local
                    // arrays cover all loads and stores; available() detected SME.
                    if !unsafe {
                        tile(
                            scratch[t * inner..].as_ptr(),
                            panel.as_ptr(),
                            seed[panel_index * 16..].as_ptr(),
                            sums.as_mut_ptr(),
                            inner,
                        )
                    } {
                        return false;
                    }
                    for r in 0..(count - t).min(16) {
                        for v in 0..4 {
                            store_row(
                                Simd::<f32, 4>::from_slice(&sums[r * 16 + v * 4..]),
                                output,
                                group + t + r,
                                column + panel_index * 16 + v * 4,
                                columns,
                                first,
                                store,
                            );
                        }
                    }
                }
                t += 16;
            }
        }
    }
    true
}

/// Compute one full 16-by-16 tile in ascending K order, seeded with bias.
///
/// SAFETY: the caller detected SME; a/b contain k*16 readable floats, bias
/// contains 16 and c contains 256 writable floats, with k > 0. No aliases.
/// The entire streaming region is a single asm block, with no calls or unwind.
/// AAPCS64 private-ZA callers can have dormant state: skip both live ZA/SM and
/// any pending TPIDR2 save block, without touching either. A non-default FPCR
/// also falls back to the existing NEON semantics. Fixed loads/stores require
/// exactly 64-byte streaming vectors, checked on the executing thread.
/// SMSTART/SMSTOP invalidate vector/predicate registers, so all are clobbered;
/// Rust consequently preserves the ABI's callee-saved vector portions.
#[inline(never)]
unsafe fn tile(a: *const f32, b: *const f32, bias: *const f32, c: *mut f32, k: usize) -> bool {
    let succeeded: usize;
    unsafe {
        asm!(
            ".arch_extension sme",
            "mrs {state}, SVCR",
            "cbnz {state}, 9f",
            "mrs {state}, TPIDR2_EL0",
            "cbnz {state}, 9f",
            "rdsvl {state}, #1",
            "cmp {state}, #64",
            "b.ne 9f",
            "mrs {state}, FPCR",
            "cbnz {state}, 9f",
            "smstart",
            "ptrue p0.s",
            "zero {{za}}",
            "ld1w {{z2.s}}, p0/z, [{bias}]",
            "mov w12, #0",
            "2:",
            "mova za0h.s[w12, 0], p0/m, z2.s",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 2b",
            "3:",
            "ld1w {{z0.s}}, p0/z, [{a}]",
            "ld1w {{z1.s}}, p0/z, [{b}]",
            "fmopa za0.s, p0/m, p0/m, z0.s, z1.s",
            "add {a}, {a}, #64",
            "add {b}, {b}, #64",
            "subs {k}, {k}, #1",
            "b.ne 3b",
            "mov w12, #0",
            "4:",
            "st1w {{za0h.s[w12, 0]}}, p0, [{c}]",
            "add {c}, {c}, #64",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 4b",
            "smstop",
            "mov {state}, #1",
            "b 10f",
            "9:",
            "mov {state}, #0",
            "10:",
            state = out(reg) succeeded,
            a = inout(reg) a => _, b = inout(reg) b => _,
            bias = in(reg) bias, c = inout(reg) c => _, k = inout(reg) k => _,
            out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _,
            out("p4") _, out("p5") _, out("p6") _, out("p7") _,
            out("p8") _, out("p9") _, out("p10") _, out("p11") _,
            out("p12") _, out("p13") _, out("p14") _, out("p15") _,
            options(nostack)
        );
    }
    succeeded != 0
}

/// Compute a 32-by-32 block with four independent 16-by-16 accumulators.
/// Two input and weight panels are reused in ascending K order, seeded with bias.
///
/// SAFETY: the caller detected SME; a/b contain 2*k*16 readable floats, bias
/// contains 32 and c contains 1024 writable floats, with k > 0. No aliases.
/// The entire streaming region is a single asm block, with no calls or unwind.
/// AAPCS64 private-ZA callers can have dormant state: skip both live ZA/SM and
/// any pending TPIDR2 save block, without touching either. A non-default FPCR
/// also falls back to the existing NEON semantics. Fixed loads/stores require
/// exactly 64-byte streaming vectors, checked on the executing thread.
/// SMSTART/SMSTOP invalidate vector/predicate registers, so all are clobbered;
/// Rust consequently preserves the ABI's callee-saved vector portions.
#[inline(never)]
unsafe fn tile32(a: *const f32, b: *const f32, bias: *const f32, c: *mut f32, k: usize) -> bool {
    let succeeded: usize;
    // SAFETY: the caller provides two full panels.
    let a1 = unsafe { a.add(k * 16) };
    let b1 = unsafe { b.add(k * 16) };
    let bias1 = unsafe { bias.add(16) };
    let c1 = unsafe { c.add(256) };
    let c2 = unsafe { c.add(512) };
    let c3 = unsafe { c.add(768) };
    unsafe {
        asm!(
            ".arch_extension sme",
            "mrs {state}, SVCR",
            "cbnz {state}, 9f",
            "mrs {state}, TPIDR2_EL0",
            "cbnz {state}, 9f",
            "rdsvl {state}, #1",
            "cmp {state}, #64",
            "b.ne 9f",
            "mrs {state}, FPCR",
            "cbnz {state}, 9f",
            "smstart",
            "ptrue p0.s",
            "zero {{za}}",
            "ld1w {{z2.s}}, p0/z, [{bias}]",
            "ld1w {{z3.s}}, p0/z, [{bias1}]",
            "mov w12, #0",
            "2:",
            "mova za0h.s[w12, 0], p0/m, z2.s",
            "mova za1h.s[w12, 0], p0/m, z3.s",
            "mova za2h.s[w12, 0], p0/m, z2.s",
            "mova za3h.s[w12, 0], p0/m, z3.s",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 2b",
            "3:",
            "ld1w {{z0.s}}, p0/z, [{a}]",
            "ld1w {{z1.s}}, p0/z, [{a1}]",
            "ld1w {{z2.s}}, p0/z, [{b}]",
            "ld1w {{z3.s}}, p0/z, [{b1}]",
            "fmopa za0.s, p0/m, p0/m, z0.s, z2.s",
            "fmopa za1.s, p0/m, p0/m, z0.s, z3.s",
            "fmopa za2.s, p0/m, p0/m, z1.s, z2.s",
            "fmopa za3.s, p0/m, p0/m, z1.s, z3.s",
            "add {a}, {a}, #64",
            "add {b}, {b}, #64",
            "add {a1}, {a1}, #64",
            "add {b1}, {b1}, #64",
            "subs {k}, {k}, #1",
            "b.ne 3b",
            "mov w12, #0",
            "4:",
            "st1w {{za0h.s[w12, 0]}}, p0, [{c}]",
            "st1w {{za1h.s[w12, 0]}}, p0, [{c1}]",
            "st1w {{za2h.s[w12, 0]}}, p0, [{c2}]",
            "st1w {{za3h.s[w12, 0]}}, p0, [{c3}]",
            "add {c}, {c}, #64",
            "add {c1}, {c1}, #64",
            "add {c2}, {c2}, #64",
            "add {c3}, {c3}, #64",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 4b",
            "smstop",
            "mov {state}, #1",
            "b 10f",
            "9:",
            "mov {state}, #0",
            "10:",
            state = out(reg) succeeded,
            a = inout(reg) a => _, b = inout(reg) b => _,
            a1 = inout(reg) a1 => _, b1 = inout(reg) b1 => _,
            bias = in(reg) bias, bias1 = in(reg) bias1,
            c = inout(reg) c => _, c1 = inout(reg) c1 => _,
            c2 = inout(reg) c2 => _, c3 = inout(reg) c3 => _,
            k = inout(reg) k => _,
            out("x12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _,
            out("p4") _, out("p5") _, out("p6") _, out("p7") _,
            out("p8") _, out("p9") _, out("p10") _, out("p11") _,
            out("p12") _, out("p13") _, out("p14") _, out("p15") _,
            options(nostack)
        );
    }
    succeeded != 0
}

/// Split exclusive output slices recursively. Rayon join uses stack jobs;
/// no per-call Vec of bands or heap-allocated scoped job list is needed.
#[cfg(feature = "parallel")]
fn sme_bands(
    output: &mut [f32],
    columns: usize,
    bands: PointwiseBands,
    workspaces: &mut [Vec<f32>],
    action: impl Fn(&mut [f32], usize, &mut [f32]) + Sync,
) {
    fn visit(
        output: &mut [f32],
        columns: usize,
        bands: PointwiseBands,
        first_worker: usize,
        workers: usize,
        workspaces: &mut [Vec<f32>],
        action: &(impl Fn(&mut [f32], usize, &mut [f32]) + Sync),
    ) {
        let first = bands.boundary(first_worker);
        if workers == 1 {
            let scratch = workspaces
                .first_mut()
                .map(Vec::as_mut_slice)
                .unwrap_or(&mut []);
            action(output, first, scratch);
            return;
        }
        let left_workers = workers / 2;
        let middle_worker = first_worker + left_workers;
        let split = (bands.boundary(middle_worker) - first) * columns;
        let (left, right) = output.split_at_mut(split);
        let (left_scratch, right_scratch) =
            workspaces.split_at_mut(left_workers.min(workspaces.len()));
        rayon::join(
            || {
                visit(
                    left,
                    columns,
                    bands,
                    first_worker,
                    left_workers,
                    left_scratch,
                    action,
                )
            },
            || {
                visit(
                    right,
                    columns,
                    bands,
                    middle_worker,
                    workers - left_workers,
                    right_scratch,
                    action,
                )
            },
        );
    }
    visit(
        output,
        columns,
        bands,
        0,
        bands.workers,
        workspaces,
        &action,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_bands(
    backend: Backend,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    inner: usize,
    columns: usize,
    cpu: &CpuContext,
    threads: usize,
    workspaces: &mut [Vec<f32>],
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
                sme_bands(output, columns, bands, workspaces, |out, first, scratch| {
                    execute_with_scratch(
                        backend, input, bias, packed, out, first, inner, columns, scratch, store,
                    );
                });
            });
            return;
        }
    }
    #[cfg(not(feature = "parallel"))]
    let _ = (cpu, threads);
    let scratch = workspaces
        .first_mut()
        .map(Vec::as_mut_slice)
        .unwrap_or(&mut []);
    execute_with_scratch(
        backend, input, bias, packed, output, 0, inner, columns, scratch, store,
    );
}

#[cfg(test)]
mod benchmark;
#[cfg(test)]
mod tests;
