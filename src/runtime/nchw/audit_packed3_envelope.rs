//! Independent cancellation fixture for pinned Conv3x3Packed.Avx.cs old8/new16.
//! Both native paths use spatial octets only when y>0, y+1<h, x>0, x+8<w.
use super::*;

#[test]
fn audit_packed3_native_octet_envelope_is_thread_stable() {
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 8 } else { 1 }).unwrap();
    for (height, width) in [(1, 7), (2, 8), (3, 9), (3, 10), (5, 17), (5, 18), (8, 8)] {
        for outputs in [8, 56, 64, 72, 80] {
            // The 64x256x8x8 case exceeds the channel-worker work threshold;
            // narrower fixtures still exercise the requested-thread selector.
            let inputs = if height == 8 && outputs == 64 { 256 } else { 3 };
            let mut p = parameters(inputs, outputs, height, width, [3, 3], [1, 1], [1, 1], 1);
            p.batch = 1;
            let plane = height * width;
            let input = vec![1. + f32::EPSILON; inputs * plane];
            let mut weights = vec![0.; outputs * inputs * 9];
            for co in 0..outputs {
                weights[(co * inputs) * 9 + 4] = 1. + f32::EPSILON;
            }
            let bias = vec![-(1. + 2. * f32::EPSILON); outputs];
            let packed = panels(&weights, &p);
            for backend in backends().filter(|backend| {
                matches!(backend, Backend::Avx | Backend::AvxFma | Backend::Avx2Fma)
            }) {
                for threads in [1, 2, 4, 8] {
                    if inputs == 256 && threads > 1 {
                        assert!(channel_shard_eligible(backend, Kind::Packed3, &p, threads));
                    }
                    let mut output = vec![f32::NAN; outputs * plane];
                    assert!(try_conv(
                        backend,
                        &p,
                        &input,
                        &weights,
                        &bias,
                        Some(&packed),
                        &mut output,
                        &cpu,
                        threads
                    ));
                    for (index, actual) in output.iter().enumerate() {
                        let y = index % plane / width;
                        let x = index % width;
                        // Literal transcription of the reference while-loop,
                        // not the production interior() or any kernel oracle.
                        let packet_end = 1 + width.saturating_sub(2) / 8 * 8;
                        let vector = y > 0 && y + 1 < height && x > 0 && x < packet_end;
                        // (1+2^-23)^2 - (1+2^-22) is exactly 2^-46 with
                        // one fused operation, but exactly +0 after rounding
                        // the product separately. Zero coefficients do not
                        // change either result in subsequent K iterations.
                        let expected = if backend != Backend::Avx && vector {
                            0x2880_0000
                        } else {
                            0
                        };
                        assert_eq!(
                            actual.to_bits(),
                            expected,
                            "pinned native octet envelope {backend:?}, threads={threads}, \
                             cout={outputs}, cin={inputs}, {height}x{width}, index={index}"
                        );
                    }
                }
            }
        }
    }
}
