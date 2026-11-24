use super::*;
use crate::runtime::nhwc::dense::neon_tests::{
    assert_bits, oracle, ordered_oracle, pack, plan, values,
};

#[test]
fn spatial_sme_pack_preserves_taps_padding_batches_and_channel_tails() {
    for kernel in [2, 3, 5, 7] {
        for stride in [1usize, 2] {
            let mut p = plan(49, 33, kernel, 2);
            p.sh = stride;
            p.sw = stride;
            p.oh = p.height.div_ceil(stride);
            p.ow = p.width.div_ceil(stride);
            let taps = kernel * kernel;
            let inner = p.cin * taps;
            let input: Vec<_> = (0..p.batch * p.height * p.width * p.cin)
                .map(|i| match i % 5 {
                    0 => -0.,
                    1 => f32::from_bits(0x7fc0_1234),
                    2 => f32::INFINITY,
                    _ => i as f32,
                })
                .collect();
            for count in [1usize, 17, 64] {
                let first = 7;
                let size = count.div_ceil(16) * 16 * inner;
                let sentinel = f32::from_bits(0x7fc0_5678);
                let mut scratch = vec![sentinel; size + 2];
                pack_fields::<false>(&p, &input, first, count, &mut scratch[1..=size]);
                for panel in (0..count).step_by(16) {
                    for k in 0..inner {
                        for row in 0..16 {
                            let pixel = first + (panel + row).min(count - 1);
                            let n = pixel / (p.oh * p.ow);
                            let sy = (pixel / p.ow % p.oh * stride + k % taps / kernel * p.dh)
                                as isize
                                - p.pt as isize;
                            let sx = (pixel % p.ow * stride + k % kernel * p.dw) as isize
                                - p.pl as isize;
                            let expected = if sy < 0
                                || sx < 0
                                || sy >= p.height as isize
                                || sx >= p.width as isize
                            {
                                0.
                            } else {
                                input[((n * p.height + sy as usize) * p.width + sx as usize)
                                    * p.cin
                                    + k / taps]
                            };
                            assert_eq!(
                                scratch[1 + panel * inner + k * 16 + row].to_bits(),
                                expected.to_bits()
                            );
                        }
                    }
                }
                assert_eq!(scratch[0].to_bits(), sentinel.to_bits());
                assert_eq!(scratch[size + 1].to_bits(), sentinel.to_bits());
            }
        }
    }
}

#[test]
fn tap_major_interior_panels_match_reordered_fields_across_row_boundaries() {
    let mut p = plan(32, 16, 3, 2);
    p.width = 79;
    p.sh = 2;
    p.sw = 2;
    p.oh = p.height.div_ceil(p.sh);
    p.ow = p.width.div_ceil(p.sw);
    let input: Vec<_> = (0..p.batch * p.height * p.width * p.cin)
        .map(|i| f32::from_bits(0x8000_0000 | i as u32))
        .collect();
    // Start in the second image. The third panel crosses a row boundary
    // and the final partial panel repeats its last output position.
    let first = p.oh * p.ow + p.ow + 1;
    let count = 57;
    let taps = p.kh * p.kw;
    let inner = taps * p.cin;
    let mut reference = vec![0.; GROUP * inner];
    let mut actual = vec![0.; reference.len()];
    pack_fields::<false>(&p, &input, first, count, &mut reference);
    pack_fields::<true>(&p, &input, first, count, &mut actual);
    for panel in (0..count).step_by(16) {
        for tap in 0..taps {
            for ci in 0..p.cin {
                let src = panel * inner + (ci * taps + tap) * 16;
                let dst = panel * inner + (tap * p.cin + ci) * 16;
                for row in 0..16 {
                    assert_eq!(actual[dst + row].to_bits(), reference[src + row].to_bits());
                }
            }
        }
    }
}

struct IndexStore;
impl StoreTransform for IndexStore {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        value + (index % 97) as f32 / 127.
    }
    fn vector<const N: usize>(&self, value: Simd<f32, N>, index: usize) -> Simd<f32, N> {
        value + Simd::from_array(std::array::from_fn(|i| ((index + i) % 97) as f32 / 127.))
    }
}

#[test]
fn spatial_sme_matches_ordered_oracle_with_borders_offsets_and_fused_stores() {
    if !crate::runtime::nhwc::pointwise::sme::available() {
        return;
    }
    for (cin, cout, kernel, stride, dilation) in [
        (49, 33, 3, 1, 1),
        (64, 64, 3, 2, 1),
        (65, 48, 3, 1, 2),
        (48, 24, 2, 1, 1),
        (33, 32, 5, 2, 1),
        (32, 32, 7, 1, 2),
    ] {
        let mut p = plan(cin, cout, kernel, dilation);
        p.sh = stride;
        p.sw = stride;
        p.oh = p.height.div_ceil(stride);
        p.ow = p.width.div_ceil(stride);
        let input = values(p.batch * p.height * p.width * cin, 3);
        let weights = values(cin * cout * kernel * kernel, 5);
        let bias = values(cout, 7);
        let tap_major = kernel == 3;
        let packed = if tap_major {
            PackedMatrix::from_spatial_3x3(&weights, p.cin, p.cout).unwrap()
        } else {
            pack(&p, &weights)
        };
        for bias in [bias.as_slice(), &[]] {
            let mut expected = ordered_oracle(&p, &input, &weights, bias, tap_major);
            for (index, v) in expected.iter_mut().enumerate() {
                *v = IndexStore.scalar(*v, index);
            }
            let first = 3;
            let mut actual = vec![f32::NAN; expected.len() - first * cout];
            let mut scratch = vec![0.; GROUP * cin * kernel * kernel];
            execute_band(
                &p,
                &input,
                bias,
                &packed,
                &mut actual,
                first,
                &mut scratch,
                &IndexStore,
            );
            assert_bits(&actual, &expected[first * cout..]);
            actual.fill(f32::NAN);
            execute_band(
                &p,
                &input,
                bias,
                &packed,
                &mut actual,
                first,
                &mut [],
                &IndexStore,
            );
            assert_bits(&actual, &expected[first * cout..]);
        }
    }
}

#[test]
fn spatial_sme_parallel_bands_reuse_workspace_and_preserve_output_order() {
    if !crate::runtime::nhwc::pointwise::sme::available() {
        return;
    }
    let p = plan(65, 33, 3, 1);
    let input = values(p.batch * p.height * p.width * p.cin, 3);
    let weights = values(p.cin * p.cout * 9, 5);
    let bias = values(p.cout, 7);
    let packed = pack(&p, &weights);
    let mut expected = oracle(&p, &input, &weights, &bias);
    for (i, v) in expected.iter_mut().enumerate() {
        *v = IndexStore.scalar(*v, i);
    }
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 4 } else { 1 }).unwrap();
    let mut workspace = vec![vec![0.; GROUP * p.cin * 9]; cpu.capacity()];
    let addresses: Vec<_> = workspace.iter().map(Vec::as_ptr).collect();
    for _ in 0..2 {
        let mut actual = vec![f32::NAN; expected.len()];
        execute_bands(
            &p,
            &input,
            &bias,
            &packed,
            &mut actual,
            &cpu,
            cpu.capacity(),
            &mut workspace,
            &IndexStore,
        );
        assert_bits(&actual, &expected);
        assert_eq!(
            addresses,
            workspace.iter().map(Vec::as_ptr).collect::<Vec<_>>()
        );
    }
}
