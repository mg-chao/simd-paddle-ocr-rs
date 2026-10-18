use super::super::tests::{assert_bits, oracle, values};
use super::*;

#[test]
fn sme_packing_preserves_bits_panel_layout_and_short_rows() {
    for count in [1usize, 3, 4, 15, 16, 17, 31, 32, 33, 48, 63, 64] {
        for inner in [1, 3, 4, 5, 7, 16, 95, 96, 97, 128, 513] {
            let mut input: Vec<_> = (0..count * inner)
                .map(|i| f32::from_bits((i as u32).wrapping_mul(0x9e3779b9)))
                .collect();
            // Include signed zeros, subnormals, infinities, and both quiet and
            // signaling NaN payloads explicitly: packing must never compute.
            for (value, bits) in input.iter_mut().zip([
                0,
                0x8000_0000,
                1,
                0x8000_0001,
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_1234,
                0x7f80_1234,
                0xff80_1234,
            ]) {
                *value = f32::from_bits(bits);
            }
            let sentinel = f32::from_bits(0x7fc0_1234);
            let size = count.div_ceil(16) * 16 * inner;
            let mut packed = vec![sentinel; size + 8];
            pack_rows(&input, inner, count, &mut packed[4..4 + size]);
            for first in (0..count).step_by(16) {
                for k in 0..inner {
                    for row in 0..16 {
                        assert_eq!(
                            packed[4 + first * inner + k * 16 + row].to_bits(),
                            input[(first + row).min(count - 1) * inner + k].to_bits(),
                            "count={count} inner={inner} first={first} k={k} row={row}",
                        );
                    }
                }
            }
            assert!(
                packed[..4]
                    .iter()
                    .chain(&packed[4 + size..])
                    .all(|x| x.to_bits() == sentinel.to_bits())
            );
        }
    }
}

#[test]
fn sme_rows_preserve_order_tails_and_offsets() {
    if !available() {
        return;
    }
    for (rows, inner, columns) in [
        (1, 1, 1),
        (15, 7, 17),
        (16, 17, 16),
        (17, 65, 33),
        (17, 2, 32),
        (31, 191, 31),
        (32, 192, 32),
        (33, 193, 63),
        (33, 96, 192),
        (63, 192, 96),
        (64, 128, 256),
        (49, 130, 65),
        (65, 256, 128),
        (63, 129, 65),
        (64, 384, 97),
        (65, 513, 17),
        (79, 385, 129),
    ] {
        let input = values((rows + 3) * inner, 3);
        let weights = values(inner * columns, 5);
        let bias = values(columns, 7);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let mut scratch = vec![0.; 64 * inner];
        for bias in [bias.as_slice(), &[]] {
            let mut expected = oracle(Backend::NeonFma, &input, &weights, bias, inner, columns);
            for (i, v) in expected.iter_mut().enumerate() {
                *v = IndexStore.scalar(*v, i);
            }
            let sentinel = f32::from_bits(0x7fc0_1234);
            let mut actual = vec![sentinel; rows * columns + 6];
            assert!(try_execute(
                &input,
                bias,
                &packed,
                &mut actual[3..3 + rows * columns],
                3,
                inner,
                columns,
                &mut scratch,
                &IndexStore
            ));
            assert_bits(
                &actual[3..3 + rows * columns],
                &expected[3 * columns..],
                "packed rows offset",
            );
            assert!(
                actual[..3]
                    .iter()
                    .chain(&actual[3 + rows * columns..])
                    .all(|v| v.to_bits() == sentinel.to_bits())
            );
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
fn sme_fused_reductions_preserve_extremes_subnormals_and_bias_order() {
    if !available() {
        return;
    }
    for (rows, inner, columns) in [(17, 515, 17), (64, 515, 32)] {
        let cases = [
            0.,
            -0.,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1e-20,
            1e20,
            f32::MAX / 1024.,
        ];
        let input: Vec<_> = (0..rows * inner)
            .map(|i| cases[(i / inner + i % inner) % cases.len()])
            .collect();
        let weights: Vec<_> = (0..inner * columns)
            .map(|i| ((i % 7) as f32 - 3.) / 8192.)
            .collect();
        let mut scratch = vec![0.; GROUP * inner];
        for bias in [
            vec![0.; columns],
            vec![-0.; columns],
            vec![f32::MIN_POSITIVE; columns],
        ] {
            let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
            let expected = oracle(Backend::NeonFma, &input, &weights, &bias, inner, columns);
            let mut actual = vec![f32::NAN; rows * columns];
            assert!(try_execute(
                &input,
                &bias,
                &packed,
                &mut actual,
                0,
                inner,
                columns,
                &mut scratch,
                &IdentityStore
            ));
            assert_bits(&actual, &expected, "extremes and subnormals");
        }
        let input = vec![1.; rows * inner];
        let mut weights = vec![0.; inner * columns];
        weights[511 * columns..512 * columns].fill(1e20);
        weights[512 * columns..513 * columns].fill(-1e20);
        weights[513 * columns..514 * columns].fill(0.5);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let mut actual = vec![f32::NAN; rows * columns];
        assert!(try_execute(
            &input,
            &vec![5.; columns],
            &packed,
            &mut actual,
            0,
            inner,
            columns,
            &mut scratch,
            &IdentityStore
        ));
        assert_bits(&actual, &vec![0.5; actual.len()], "bias first, ascending K");
    }
}

#[test]
fn sme_parallel_bands_reuse_exclusive_scratch_and_preserve_fused_stores() {
    if !available() {
        return;
    }
    let (rows, inner, columns) = (433, 385, 389);
    let input = values(rows * inner, 3);
    let weights = values(inner * columns, 5);
    let bias = values(columns, 7);
    let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
    let mut expected = vec![0.; rows * columns];
    execute(
        Backend::NeonFma,
        &input,
        &bias,
        &packed,
        &mut expected,
        0,
        inner,
        columns,
        &IndexStore,
    );
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 8 } else { 1 }).unwrap();
    for budget in [1, 2, 4, 8] {
        let mut workspaces = vec![vec![0.; GROUP * inner]; budget];
        let addresses: Vec<_> = workspaces.iter().map(|w| w.as_ptr()).collect();
        let mut actual = vec![f32::NAN; rows * columns];
        assert!(try_sme(
            Backend::NeonFma,
            &input,
            &bias,
            &packed,
            &mut actual,
            inner,
            columns,
            &cpu,
            budget,
            &mut workspaces,
            &IndexStore,
        ));
        assert_bits(&actual, &expected, "parallel SME fused indexed stores");
        assert!(
            workspaces
                .iter()
                .zip(addresses)
                .all(|(w, address)| w.as_ptr() == address)
        );
    }
    let mut fallback = vec![f32::NAN; expected.len()];
    execute_with_scratch(
        Backend::NeonFma,
        &input,
        &bias,
        &packed,
        &mut fallback,
        0,
        inner,
        columns,
        &mut [],
        &IndexStore,
    );
    assert_bits(&fallback, &expected, "missing scratch falls back");
}

#[test]
fn sme_guard_preserves_za_state_and_nondefault_fpcr() {
    if !available() {
        return;
    }
    // ZA-only state does not change non-streaming vector semantics. Never
    // assert or unwind before restoring it. The tile must reject live ZA.
    type Tile = unsafe fn(*const f32, *const f32, *const f32, *mut f32, usize) -> bool;
    for tile in [tile as Tile, tile32 as Tile, tile32_repeated::<2> as Tile] {
        let a = [0.25f32; 64];
        let bias = [0f32; 32];
        let mut out = [f32::NAN; 2048];
        let before: u64;
        let after: u64;
        let mut skipped: bool;
        unsafe {
            asm!(".arch_extension sme", "mrs {v}, SVCR", v = out(reg) before, options(nomem, nostack));
        }
        if before != 0 {
            return;
        }
        unsafe {
            asm!(".arch_extension sme", "smstart za", options(nostack));
            skipped = !tile(a.as_ptr(), a.as_ptr(), bias.as_ptr(), out.as_mut_ptr(), 1);
            asm!(".arch_extension sme", "mrs {v}, SVCR", "smstop za", v = out(reg) after, options(nostack));
        }
        assert!(skipped);
        assert_eq!(after, 2);
        assert!(out.iter().all(|x| x.is_nan()));
        let fpcr: u64;
        let restored: u64;
        unsafe {
            asm!("mrs {v}, FPCR", v = out(reg) fpcr, options(nomem, nostack));
            asm!("msr FPCR, {v}", v = in(reg) (fpcr | (1 << 24)), options(nostack));
            skipped = !tile(a.as_ptr(), a.as_ptr(), bias.as_ptr(), out.as_mut_ptr(), 1);
            asm!("mrs {v}, FPCR", "msr FPCR, {old}", v = out(reg) restored, old = in(reg) fpcr, options(nostack));
        }
        assert!(skipped);
        assert_eq!(restored, fpcr | (1 << 24));
        assert!(out.iter().all(|x| x.is_nan()));
    }
}
