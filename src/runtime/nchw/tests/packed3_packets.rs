//! Separate packet/store contracts for reference Vector4 and reachable AVX512
//! eight-output packed3. Native512 executes only on a capable host; selection
//! assertions below are source/dispatch simulation and make no hardware claim.
use super::*;

struct PacketStore {
    visits: PacketVisits,
    lanes: usize,
}
impl simd::StoreTransform for PacketStore {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        self.visits.scalar(value, index) * 2. + index as f32 / 1024.
    }
    fn vector<const N: usize>(&self, value: Simd<f32, N>, index: usize) -> Simd<f32, N> {
        assert_eq!(N, self.lanes, "pinned spatial packet width");
        self.visits.vector(value, index) * Simd::splat(2.)
            + Simd::from_array(std::array::from_fn(|lane| (index + lane) as f32 / 1024.))
    }
}

#[test]
fn packed3_portable4_and_available512_store_packets() {
    for backend in
        backends().filter(|backend| matches!(backend, Backend::Portable4 | Backend::Avx512))
    {
        let lanes = if backend == Backend::Portable4 { 4 } else { 16 };
        let widths = if lanes == 4 {
            [1, 5, 6, 9, 10, 13]
        } else {
            [1, 16, 17, 18, 33, 34]
        };
        for height in [1, 2, 3, 5] {
            for width in widths {
                let p = parameters(3, 24, height, width, [3, 3], [1, 1], [1, 1], 1);
                for threads in [1, 2] {
                    assert_eq!(select(backend, &p, true, threads), Some(Kind::Packed3));
                    assert_eq!(
                        selected_backend_for_threads(backend, &p, true, threads),
                        backend
                    );
                }
                let plane = height * width;
                let length = p.batch * p.output_channels * plane;
                let input: Vec<_> = (0..p.batch * p.input_channels * plane)
                    .map(|index| (index % 31 + 1) as f32 / 32.)
                    .collect();
                let mut weights = vec![0.; p.output_channels * p.input_channels * 9];
                let bias: Vec<_> = (0..p.output_channels).map(|co| co as f32 / 16.).collect();
                for co in 0..p.output_channels {
                    // Distinct adjacent OC8 groups expose an omitted co%16.
                    weights[(co * p.input_channels + co % 3) * 9 + 4] = (co + 1) as f32 / 16.;
                    weights[(co * p.input_channels + (co + 1) % 3) * 9] = 0.125;
                }
                let packed = panels(&weights, &p);
                // The middle case starts at the upper half of a shared16 panel;
                // the last case uses its padded final lower half. Nonzero batch
                // verifies epilogue indices independently of local output bands.
                for (batch, first, count) in [(0, 0, 24), (1, 8, 8), (1, 16, 8)] {
                    let begin = (batch * p.output_channels + first) * plane;
                    let end = begin + count * plane;
                    let store = PacketStore {
                        visits: PacketVisits::new(length, begin..end, plane),
                        lanes,
                    };
                    let guard = f32::from_bits(0x7fc0_57aa);
                    let mut output = vec![guard; count * plane + 8];
                    run(
                        backend,
                        Kind::Packed3,
                        &p,
                        &input,
                        &weights,
                        &bias,
                        Some(&packed),
                        &mut output[4..4 + count * plane],
                        batch,
                        first,
                        count,
                        &store,
                    );
                    assert!(
                        output[..4]
                            .iter()
                            .chain(&output[4 + count * plane..])
                            .all(|value| value.to_bits() == guard.to_bits())
                    );
                    for (offset, actual) in output[4..4 + count * plane].iter().enumerate() {
                        let index = begin + offset;
                        let co = index / plane % p.output_channels;
                        let spatial = index % plane;
                        let (y, x) = (spatial / width, spatial % width);
                        let mut expected = bias[co];
                        // Literal two-tap dyadic convolution, no production
                        // edge/interior/madd/oracle helper supplies its result.
                        for ci in 0..3 {
                            if ci == (co + 1) % 3 && y > 0 && x > 0 {
                                expected += input
                                    [(batch * 3 + ci) * plane + (y - 1) * width + x - 1]
                                    * 0.125;
                            }
                            if ci == co % 3 {
                                expected += input[(batch * 3 + ci) * plane + spatial]
                                    * (co + 1) as f32
                                    / 16.;
                            }
                        }
                        expected = expected * 2. + index as f32 / 1024.;
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "spatial/store contract {backend:?}, {height}x{width}, batch={batch}, first={first}, index={index}"
                        );
                    }
                    store.visits.assert_coverage(|index| {
                        let (y, x) = (index % plane / width, index % width);
                        // Reference x starts0; exactly one scalar precedes the
                        // maximal complete N-wide packets, with both neighbours.
                        let packet_end = 1 + width.saturating_sub(2) / lanes * lanes;
                        y > 0 && y + 1 < height && x > 0 && x < packet_end
                    });
                }
            }
        }
    }
}

#[test]
fn packed3_simulated512_keeps_reachable_eight_output_family() {
    // No native entry is called here. The pinned sixteen-output AVX512 helper
    // has no callsites: Try selects AVX2 paired16 for eligible single-thread
    // widths, otherwise its reachable AVX512 body owns eight output channels.
    for outputs in [8, 24, 56, 64, 72, 80] {
        let p = parameters(3, outputs, 3, 34, [3, 3], [1, 1], [1, 1], 1);
        for threads in [1, 2, 4] {
            let paired = threads == 1 && outputs >= 64 && outputs % 16 == 0;
            assert_eq!(
                family_for_threads(Backend::Avx512, &p, true, threads),
                Some(if paired {
                    "nchw_packed_3x3_pair16"
                } else {
                    "nchw_packed_3x3"
                })
            );
            assert_eq!(
                selected_backend_for_threads(Backend::Avx512, &p, true, threads),
                if paired {
                    Backend::Avx2Fma
                } else {
                    Backend::Avx512
                }
            );
        }
    }
}
