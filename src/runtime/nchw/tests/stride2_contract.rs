//! Independent pinned Conv3x3Stride2 packed contracts. These tests call the
//! existing entry point and never consult its selector, geometry or oracle.
use super::*;

fn literal_packets(p: &Conv2d, lanes: usize, dual: bool) -> Vec<bool> {
    let mut packets = vec![false; p.output_height * p.output_width];
    for oy in 0..p.output_height {
        let mut ox = 0;
        while ox < p.output_width {
            let full = |n| {
                oy > 0
                    && oy * 2 + 1 < p.height
                    && ox > 0
                    && ox + n <= p.output_width
                    && (ox + n - 1) * 2 + 1 < p.width
            };
            let count = if dual && full(lanes * 2) {
                lanes * 2
            } else if full(lanes) {
                lanes
            } else {
                0
            };
            if count == 0 {
                ox += 1;
            } else {
                packets[oy * p.output_width + ox..oy * p.output_width + ox + count].fill(true);
                ox += count;
            }
        }
    }
    packets
}

#[test]
fn stride2_literal_spatial_envelope_and_cancellation() {
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 4 } else { 1 }).unwrap();
    // FMA policies run first so a RED identifies the arithmetic boundary, not
    // merely a callback packet-size mismatch. Available non-FMA paths follow.
    let ordered = backends()
        .filter(|b| matches!(b, Backend::AvxFma | Backend::Avx2Fma | Backend::Avx512))
        .chain(backends().filter(|b| matches!(b, Backend::Avx | Backend::Portable4)));
    for backend in ordered {
        let lanes = match backend {
            Backend::Portable4 => 4,
            Backend::Avx512 => 16,
            _ => 8,
        };
        for height in [1, 2, 3, 4, 5, 6] {
            for width in [7, 8, 9, 10, 15, 16, 17, 18, 19, 31, 32, 33, 34, 35] {
                let mut p = parameters(3, 24, height, width, [3, 3], [2, 2], [1, 1], 1);
                p.batch = 2;
                let plane = p.output_height * p.output_width;
                let length = p.batch * p.output_channels * plane;
                let packets = literal_packets(&p, lanes, backend == Backend::Portable4);
                let input = vec![1. + f32::EPSILON; p.batch * 3 * height * width];
                let mut weights = vec![0.; 24 * 3 * 9];
                for co in 0..24 {
                    weights[co * 3 * 9 + 4] = 1. + f32::EPSILON;
                }
                let bias = vec![-(1. + 2. * f32::EPSILON); 24];
                let packed = panels(&weights, &p);
                for threads in [1, cpu.capacity()] {
                    let store = PacketVisits::new(length, 0..length, plane);
                    let guard = f32::from_bits(0x7fc0_62aa);
                    let mut output = vec![guard; length + 8];
                    assert!(try_conv_store(
                        backend,
                        &p,
                        &input,
                        &weights,
                        &bias,
                        Some(&packed),
                        &mut output[4..4 + length],
                        &cpu,
                        threads,
                        &store
                    ));
                    assert!(
                        output[..4]
                            .iter()
                            .chain(&output[4 + length..])
                            .all(|v| v.to_bits() == guard.to_bits())
                    );
                    for (index, value) in output[4..4 + length].iter().enumerate() {
                        // The single nonzero center tap is always in bounds.
                        // Fused (1+2^-23)^2-(1+2^-22)=2^-46; separate ops=+0.
                        let fused = matches!(
                            backend,
                            Backend::AvxFma | Backend::Avx2Fma | Backend::Avx512
                        ) && packets[index % plane];
                        assert_eq!(
                            value.to_bits(),
                            if fused { 0x2880_0000 } else { 0 },
                            "pinned stride2 rounding {backend:?}, input={height}x{width}, threads={threads}, index={index}"
                        );
                    }
                    store.assert_coverage(|index| packets[index % plane]);
                }
            }
        }
    }
}

struct StrideOrder {
    visits: PacketVisits,
    clock: std::sync::atomic::AtomicUsize,
    stamps: Vec<std::sync::atomic::AtomicUsize>,
}
impl StrideOrder {
    fn new(length: usize, plane: usize) -> Self {
        Self {
            visits: PacketVisits::new(length, 0..length, plane),
            clock: std::sync::atomic::AtomicUsize::new(0),
            stamps: (0..length)
                .map(|_| std::sync::atomic::AtomicUsize::new(0))
                .collect(),
        }
    }
    fn record(&self, index: usize, count: usize) {
        use std::sync::atomic::Ordering::SeqCst;
        let stamp = self.clock.fetch_add(1, SeqCst) + 1;
        for item in &self.stamps[index..index + count] {
            assert_eq!(
                item.swap(stamp, SeqCst),
                0,
                "duplicate stride2 store at {index}"
            );
        }
    }
    fn at(&self, index: usize) -> usize {
        let value = self.stamps[index].load(std::sync::atomic::Ordering::SeqCst);
        assert_ne!(value, 0, "missing stride2 store at {index}");
        value
    }
}
impl simd::StoreTransform for StrideOrder {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        self.record(index, 1);
        self.visits.scalar(value, index) * 2. + index as f32 / 2048.
    }
    fn vector<const N: usize>(&self, value: Simd<f32, N>, index: usize) -> Simd<f32, N> {
        self.record(index, N);
        self.visits.vector(value, index) * Simd::splat(2.)
            + Simd::from_array(std::array::from_fn(|i| (index + i) as f32 / 2048.))
    }
}

fn stride_order_case(
    backend: Backend,
    outputs: usize,
    batch: usize,
    threads: usize,
    width: usize,
) -> (Conv2d, StrideOrder) {
    let mut p = parameters(3, outputs, 4, width, [3, 3], [2, 2], [1, 1], 1);
    p.batch = batch;
    let input_plane = p.height * p.width;
    let plane = p.output_height * p.output_width;
    let length = batch * outputs * plane;
    let input: Vec<_> = (0..batch * 3 * input_plane)
        .map(|i| (i % 31 + 1) as f32 / 32.)
        .collect();
    let mut weights = vec![0.; outputs * 3 * 9];
    let bias: Vec<_> = (0..outputs).map(|co| co as f32 / 32.).collect();
    for co in 0..outputs {
        weights[(co * 3 + co % 3) * 9 + 4] = (co + 1) as f32 / 256.;
    }
    let packed = panels(&weights, &p);
    let store = StrideOrder::new(length, plane);
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 4 } else { 1 }).unwrap();
    let guard = f32::from_bits(0x7fc0_63aa);
    let mut output = vec![guard; length + 8];
    assert!(try_conv_store(
        backend,
        &p,
        &input,
        &weights,
        &bias,
        Some(&packed),
        &mut output[4..4 + length],
        &cpu,
        threads,
        &store
    ));
    assert!(
        output[..4]
            .iter()
            .chain(&output[4 + length..])
            .all(|v| v.to_bits() == guard.to_bits())
    );
    for (index, value) in output[4..4 + length].iter().enumerate() {
        let b = index / (outputs * plane);
        let co = index / plane % outputs;
        let spatial = index % plane;
        let y = spatial / p.output_width * 2;
        let x = spatial % p.output_width * 2;
        let expected = (bias[co]
            + input[(b * 3 + co % 3) * input_plane + y * width + x] * (co + 1) as f32 / 256.)
            * 2.
            + index as f32 / 2048.;
        assert_eq!(
            value.to_bits(),
            expected.to_bits(),
            "stride2 independent delta {backend:?}, cout={outputs}, batch={batch}, threads={threads}, index={index}"
        );
    }
    (p, store)
}

#[test]
fn stride2_native_pair16_and_small_work_recursive_store_order() {
    for backend in
        backends().filter(|b| matches!(b, Backend::Avx | Backend::AvxFma | Backend::Avx2Fma))
    {
        for (outputs, batch, threads) in [(16, 1, 1), (16, 2, 4), (32, 1, 2), (80, 1, 4)] {
            if !cfg!(feature = "parallel") && batch == 1 && threads > 1 {
                continue;
            }
            let (p, store) = stride_order_case(backend, outputs, batch, threads, 18);
            let plane = p.output_height * p.output_width;
            let bands: &[(usize, usize, bool)] = match (outputs, batch) {
                (16, _) => &[(0, 16, true)],
                (32, _) => &[(0, 16, true), (16, 16, true)],
                _ => &[
                    (0, 16, true),
                    (16, 24, false),
                    (40, 16, true),
                    (56, 24, false),
                ],
            };
            for b in 0..batch {
                for &(first, count, paired) in bands {
                    let begin = (b * outputs + first) * plane;
                    // Source scalar top-row stores expose actual OC16 sharing,
                    // even when every numeric result equals the OC8 alternative.
                    assert_eq!(
                        store.at(begin + 8 * plane) < store.at(begin + 1),
                        paired,
                        "stride2 child family {backend:?}, cout={outputs}, batch={batch}, threads={threads}, first={first}, count={count}"
                    );
                }
            }
            let packets = literal_packets(&p, 8, false);
            store.visits.assert_coverage(|index| packets[index % plane]);
        }
    }
}

#[test]
fn stride2_portable4_dual_tiles_share_weight_panel_before_next_channel() {
    let (p, store) = stride_order_case(Backend::Portable4, 24, 2, 1, 26);
    let plane = p.output_height * p.output_width;
    for b in 0..2 {
        for first in [0, 8, 16] {
            let begin = (b * 24 + first) * plane + p.output_width;
            // At oy1/ox1 the reference prefers TWO adjacent four-lane tiles.
            // It stores co0/ox1,co0/ox5,co1/ox1 before advancing channels.
            assert!(
                store.at(begin + 5) < store.at(begin + plane + 1),
                "pinned Vector4 dual tile order batch={b}, first={first}"
            );
            // The final single packet ox9 starts only after all dual channels.
            assert!(
                store.at(begin + 7 * plane + 5) < store.at(begin + 9),
                "pinned Vector4 dual-to-single order batch={b}, first={first}"
            );
        }
    }
    let packets = literal_packets(&p, 4, true);
    store.visits.assert_coverage(|index| packets[index % plane]);
}
