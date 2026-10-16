//! Actual callback order distinguishes the pinned recursive OC16 child from
//! OC8 without a production observation hook or a numeric-rounding difference.
#![cfg(feature = "parallel")]
use super::*;

struct OrderedStore {
    visits: PacketVisits,
    clock: std::sync::atomic::AtomicUsize,
    stamps: Vec<std::sync::atomic::AtomicUsize>,
}
impl OrderedStore {
    fn new(length: usize, plane: usize) -> Self {
        Self {
            visits: PacketVisits::new(length, 0..length, plane),
            clock: std::sync::atomic::AtomicUsize::new(0),
            stamps: (0..length)
                .map(|_| std::sync::atomic::AtomicUsize::new(0))
                .collect(),
        }
    }
    fn stamp(&self, index: usize, count: usize) {
        use std::sync::atomic::Ordering::SeqCst;
        let stamp = self.clock.fetch_add(1, SeqCst) + 1;
        for value in &self.stamps[index..index + count] {
            assert_eq!(
                value.swap(stamp, SeqCst),
                0,
                "duplicate final store at {index}"
            );
        }
    }
    fn at(&self, index: usize) -> usize {
        let value = self.stamps[index].load(std::sync::atomic::Ordering::SeqCst);
        assert_ne!(value, 0, "missing final store at {index}");
        value
    }
}
impl simd::StoreTransform for OrderedStore {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        self.stamp(index, 1);
        self.visits.scalar(value, index) * 2. + index as f32 / 2048.
    }
    fn vector<const N: usize>(&self, value: Simd<f32, N>, index: usize) -> Simd<f32, N> {
        assert_eq!(N, 8, "native packed3 spatial octet");
        self.stamp(index, N);
        self.visits.vector(value, index) * Simd::splat(2.)
            + Simd::from_array(std::array::from_fn(|lane| (index + lane) as f32 / 2048.))
    }
}

#[cfg(feature = "parallel")]
#[test]
fn audit_packed3_recursive_child_store_order_matches_literal_floor_bands() {
    let cpu = CpuContext::new(4).unwrap();
    // Literal pinned OC8 floor bands. Pair16/OC8 arithmetic is deliberately
    // identical here; output-order observations must prove dispatch instead.
    for (outputs, inputs, height, width, threads) in [(128, 256, 8, 8, 2), (272, 64, 5, 18, 4)] {
        let bands: &[(usize, usize, bool)] = if outputs == 128 {
            &[(0, 64, true), (64, 64, true)]
        } else {
            &[
                (0, 64, true),
                (64, 72, false),
                (136, 64, true),
                (200, 72, false),
            ]
        };
        let mut p = parameters(inputs, outputs, height, width, [3, 3], [1, 1], [1, 1], 1);
        p.batch = 1;
        let plane = height * width;
        let length = outputs * plane;
        let input: Vec<_> = (0..inputs * plane)
            .map(|i| (i % 31 + 1) as f32 / 32.)
            .collect();
        let mut weights = vec![0.; outputs * inputs * 9];
        let bias: Vec<_> = (0..outputs).map(|co| co as f32 / 32.).collect();
        for co in 0..outputs {
            weights[(co * inputs + co % inputs) * 9 + 4] = (co + 1) as f32 / 256.;
        }
        let packed = panels(&weights, &p);
        for backend in backends()
            .filter(|backend| matches!(backend, Backend::Avx | Backend::AvxFma | Backend::Avx2Fma))
        {
            assert!(channel_shard_eligible(backend, Kind::Packed3, &p, threads));
            let store = OrderedStore::new(length, plane);
            let guard = f32::from_bits(0x7fc0_6baa);
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
            for (index, actual) in output[4..4 + length].iter().enumerate() {
                let co = index / plane;
                // All terms and the indexed epilogue are exact dyadics. This
                // independent lookup exposes incorrect co136 panel selection.
                let expected = (bias[co]
                    + input[(co % inputs) * plane + index % plane] * (co + 1) as f32 / 256.)
                    * 2.
                    + index as f32 / 2048.;
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "band values {backend:?}, cout={outputs}, index={index}"
                );
            }
            store.visits.assert_coverage(|index| {
                let y = index % plane / width;
                let x = index % width;
                let end = 1 + width.saturating_sub(2) / 8 * 8;
                y > 0 && y + 1 < height && x > 0 && x < end
            });
            for &(first, count, paired) in bands {
                assert!(first + count <= outputs && count >= 16);
                // Both probes belong to the SAME expected worker band. Other
                // workers may interleave timestamps but cannot invert its order.
                let second_octet_x0 = store.at((first + 8) * plane);
                let first_octet_x1 = store.at(first * plane + 1);
                assert_eq!(
                    second_octet_x0 < first_octet_x1,
                    paired,
                    "pinned recursive child family {backend:?}, cout={outputs}, threads={threads}, first={first}, count={count}"
                );
            }
        }
    }
}
