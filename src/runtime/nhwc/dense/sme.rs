//! Pack only sixty-four receptive fields, in the existing IC/KY/KX order.
//! Each field is consumed by the same SME tiles and fused stores as pointwise.
use super::*;
use crate::runtime::nhwc::pointwise::sme::{GROUP, packed_group};
use std::arch::aarch64::{
    vcombine_f32, vdupq_n_f32, vget_high_f32, vget_low_f32, vld1q_f32, vst1q_f32, vtrn1q_f32,
    vtrn2q_f32,
};

fn pack_fields(p: &Conv, input: &[f32], first: usize, count: usize, scratch: &mut [f32]) {
    let taps = p.kh * p.kw;
    let inner = p.cin * taps;
    assert!((1..=GROUP).contains(&count));
    assert!(scratch.len() >= count.div_ceil(16) * 16 * inner);
    assert!(first + count <= p.batch * p.oh * p.ow);
    for panel_first in (0..count).step_by(16) {
        let panel = &mut scratch[panel_first * inner..(panel_first + 16) * inner];
        for tap in 0..taps {
            // Bind complete channel spans once per tap. Padding supplies
            // positive zero, including FMA with zero-valued kernel taps.
            let fields: [Option<&[f32]>; 16] = std::array::from_fn(|r| {
                let pixel = first + (panel_first + r).min(count - 1);
                let n = pixel / (p.oh * p.ow);
                let y = pixel / p.ow % p.oh;
                let x = pixel % p.ow;
                let sy = y * p.sh + tap / p.kw * p.dh;
                let sx = x * p.sw + tap % p.kw * p.dw;
                if sy < p.pt || sy - p.pt >= p.height || sx < p.pl || sx - p.pl >= p.width {
                    None
                } else {
                    let base = ((n * p.height + sy - p.pt) * p.width + sx - p.pl) * p.cin;
                    Some(&input[base..base + p.cin])
                }
            });
            for row in (0..16).step_by(4) {
                for ci in (0..p.cin / 4 * 4).step_by(4) {
                    // SAFETY: NEON is baseline on this target. Bound fields
                    // contain cin floats and ci+4 <= cin. The final store is
                    // within the checked sixteen-row, inner-column panel.
                    unsafe {
                        let load = |r: usize| {
                            fields[r]
                                .map_or_else(|| vdupq_n_f32(0.), |s| vld1q_f32(s.as_ptr().add(ci)))
                        };
                        let (a, b, c, d) = (load(row), load(row + 1), load(row + 2), load(row + 3));
                        let ab0 = vtrn1q_f32(a, b);
                        let ab1 = vtrn2q_f32(a, b);
                        let cd0 = vtrn1q_f32(c, d);
                        let cd1 = vtrn2q_f32(c, d);
                        let destination = panel.as_mut_ptr().add((ci * taps + tap) * 16 + row);
                        vst1q_f32(
                            destination,
                            vcombine_f32(vget_low_f32(ab0), vget_low_f32(cd0)),
                        );
                        vst1q_f32(
                            destination.add(taps * 16),
                            vcombine_f32(vget_low_f32(ab1), vget_low_f32(cd1)),
                        );
                        vst1q_f32(
                            destination.add(taps * 32),
                            vcombine_f32(vget_high_f32(ab0), vget_high_f32(cd0)),
                        );
                        vst1q_f32(
                            destination.add(taps * 48),
                            vcombine_f32(vget_high_f32(ab1), vget_high_f32(cd1)),
                        );
                    }
                }
            }
            for ci in p.cin / 4 * 4..p.cin {
                for (row, field) in fields.iter().enumerate() {
                    panel[(ci * taps + tap) * 16 + row] = field.map_or(0., |s| s[ci]);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_band(
    p: &Conv,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    first: usize,
    scratch: &mut [f32],
    store: &impl StoreTransform,
) {
    let inner = p.cin * p.kh * p.kw;
    if scratch.len() >= GROUP * inner {
        let rows = output.len() / p.cout;
        for group in (0..rows).step_by(GROUP) {
            let count = (rows - group).min(GROUP);
            pack_fields(p, input, first + group, count, scratch);
            if !packed_group(
                scratch,
                bias,
                packed,
                &mut output[group * p.cout..(group + count) * p.cout],
                first + group,
                inner,
                p.cout,
                count,
                store,
            ) {
                // The guard declines before changing SME state. Recompute the
                // full band with the existing kernel, overwriting any prefix.
                super::execute(
                    Backend::NeonFma,
                    p,
                    input,
                    bias,
                    packed,
                    output,
                    first,
                    store,
                );
                return;
            }
        }
    } else {
        super::execute(
            Backend::NeonFma,
            p,
            input,
            bias,
            packed,
            output,
            first,
            store,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_bands(
    p: &Conv,
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
    workspaces: &mut [Vec<f32>],
    store: &impl StoreTransform,
) {
    #[cfg(feature = "parallel")]
    {
        let bands = crate::runtime::nhwc::PointwiseBands::new(
            output.len() / p.cout,
            p.cin * p.kh * p.kw,
            p.cout,
            threads
                .max(1)
                .min(cpu.capacity())
                .min(workspaces.len().max(1)),
        );
        if bands.workers > 1 {
            cpu.install(|| {
                crate::runtime::nhwc::pointwise::sme::sme_bands(
                    output,
                    p.cout,
                    bands,
                    workspaces,
                    |out, first, scratch| {
                        execute_band(p, input, bias, packed, out, first, scratch, store)
                    },
                )
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
    execute_band(p, input, bias, packed, output, 0, scratch, store);
}

#[cfg(test)]
mod tests;
