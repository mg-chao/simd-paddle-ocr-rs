use super::*;

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i.wrapping_mul(37).wrapping_add(seed * 17)) % 101) as f32 / 101. - 0.5)
        .collect()
}

fn plan(
    dimensions: [usize; 5],
    kernel: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    pad: [usize; 4],
    groups: usize,
) -> Conv {
    let [batch, cin, cout, height, width] = dimensions;
    let [kh, kw] = kernel;
    let [sh, sw] = stride;
    let [dh, dw] = dilation;
    Conv {
        batch,
        cin,
        cout,
        height,
        width,
        oh: (height + pad[0] + pad[2] - (kh - 1) * dh - 1) / sh + 1,
        ow: (width + pad[1] + pad[3] - (kw - 1) * dw - 1) / sw + 1,
        cpg: cin / groups,
        opg: cout / groups,
        kh,
        kw,
        sh,
        sw,
        dh,
        dw,
        pt: pad[0],
        pl: pad[1],
    }
}

fn pack(p: &Conv, weights: &[f32]) -> PackedMatrix {
    let taps = p.kh * p.kw;
    let mut transposed = vec![0.; taps * p.cout];
    for co in 0..p.cout {
        for tap in 0..taps {
            transposed[tap * p.cout + co] = weights[co * taps + tap];
        }
    }
    PackedMatrix::new(&transposed, taps, p.cout).unwrap()
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "index={index}");
    }
}

struct ChannelResidual<'a> {
    channel: &'a [f32],
    residual: &'a [f32],
}
impl StoreTransform for ChannelResidual<'_> {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        ((value + self.channel[index % self.channel.len()]).max(0.)) + self.residual[index]
    }
    fn vector<const N: usize>(&self, value: Simd<f32, N>, index: usize) -> Simd<f32, N> {
        self.vector_channel(value, index, index % self.channel.len())
    }
    fn vector_channel<const N: usize>(
        &self,
        value: Simd<f32, N>,
        index: usize,
        channel: usize,
    ) -> Simd<f32, N> {
        use std::simd::num::SimdFloat;
        assert_eq!(channel, index % self.channel.len());
        (value + Simd::from_slice(&self.channel[channel..channel + N])).simd_max(Simd::splat(0.))
            + Simd::from_slice(&self.residual[index..index + N])
    }
}

#[test]
fn short_depthwise_preserves_checked_bits_bands_tails_and_fused_operands() {
    if !simd::avx2_fma_available() {
        return;
    }
    for p in [
        plan([2, 35, 35, 11, 37], [5, 5], [1, 1], [1, 1], [2; 4], 35),
        plan(
            [2, 17, 17, 13, 37],
            [3, 3],
            [2, 1],
            [2, 2],
            [3, 2, 1, 4],
            17,
        ),
        plan([2, 8, 8, 9, 1], [3, 3], [1, 1], [1, 1], [1; 4], 8),
        plan([1, 64, 64, 8, 41], [3, 3], [1, 1], [1, 1], [1; 4], 64),
        plan(
            [2, 24, 24, 13, 73],
            [3, 5],
            [1, 2],
            [2, 1],
            [2, 3, 2, 1],
            24,
        ),
    ] {
        let pixels = p.batch * p.oh * p.ow;
        // Both source/destination starts deliberately miss vector alignment.
        let mut input_storage = vec![0.; 3];
        input_storage.extend(values(p.batch * p.height * p.width * p.cin, 83));
        let input = &input_storage[3..];
        let weights = values(p.cout * p.kh * p.kw, 89);
        let packed = pack(&p, &weights);
        let bias = values(p.cout, 97);
        let channel = values(p.cout, 101);
        let residual = values(pixels * p.cout, 103);
        let fused = ChannelResidual {
            channel: &channel,
            residual: &residual,
        };
        fn check(
            p: &Conv,
            input: &[f32],
            weights: &[f32],
            bias: &[f32],
            packed: &PackedMatrix,
            store: &impl StoreTransform,
        ) {
            let pixels = p.batch * p.oh * p.ow;
            let mut expected = vec![0.; pixels * p.cout];
            execute_control::<false>(p, input, weights, bias, packed, &mut expected, 0, store);
            for band in [1, 7, 8, 9, 31, pixels] {
                let guard = f32::from_bits(0x7fc0_1234);
                let mut storage = vec![guard; expected.len() + 6];
                for (part, output) in storage[3..3 + expected.len()]
                    .chunks_mut(band * p.cout)
                    .enumerate()
                {
                    execute_control::<true>(
                        p,
                        input,
                        weights,
                        bias,
                        packed,
                        output,
                        part * band,
                        store,
                    );
                }
                assert_bits(&storage[3..3 + expected.len()], &expected);
                assert!(
                    storage[..3]
                        .iter()
                        .chain(&storage[3 + expected.len()..])
                        .all(|value| value.to_bits() == guard.to_bits())
                );
            }
        }
        for bias in [bias.as_slice(), &[]] {
            check(&p, input, &weights, bias, &packed, &IdentityStore);
            check(&p, input, &weights, bias, &packed, &fused);
        }
    }
}

#[test]
fn short_depthwise_declines_invalid_storage_and_ineligible_geometry_without_writes() {
    if !simd::avx2_fma_available() {
        return;
    }
    let p = plan([1, 16, 16, 9, 17], [3, 3], [1, 1], [1, 1], [1; 4], 16);
    let input = values(p.cin * p.height * p.width, 107);
    let weights = values(p.cout * p.kh * p.kw, 109);
    let packed = pack(&p, &weights);
    let guard = f32::from_bits(0x7fc0_5678);
    let mut output = vec![guard; p.oh * p.ow * p.cout];
    let call = |p: &Conv, input: &[f32], packed, output: &mut [f32], first| {
        // SAFETY: the test feature check supplies AVX2/FMA; validation rejects
        // these malformed storage/geometry arguments before entering a leaf.
        unsafe {
            try_execute(
                p,
                input,
                &weights,
                &[],
                packed,
                output,
                first,
                &IdentityStore,
            )
        }
    };
    assert!(!call(&p, &input, None, &mut output, 0));
    assert!(!call(
        &p,
        &input[..input.len() - 1],
        Some(&packed),
        &mut output,
        0
    ));
    assert!(!call(&p, &input, Some(&packed), &mut output, 1));
    let len = output.len();
    assert!(!call(&p, &input, Some(&packed), &mut output[..len - 1], 0));
    for altered in [
        Conv { kh: 6, ..p.clone() },
        Conv { kw: 6, ..p.clone() },
        Conv { sw: 0, ..p.clone() },
        Conv {
            sw: usize::MAX,
            ..p.clone()
        },
    ] {
        assert!(!call(&altered, &input, Some(&packed), &mut output, 0));
    }
    assert!(
        output
            .iter()
            .all(|value| value.to_bits() == guard.to_bits())
    );
}

#[test]
fn short_depthwise_eligibility_requires_a_complete_receptive_field_tile() {
    for p in [
        plan([1, 16, 16, 1, 17], [3, 3], [1; 2], [1; 2], [1; 4], 16),
        plan([1, 16, 16, 3, 8], [3, 3], [1; 2], [1; 2], [1; 4], 16),
        plan([1, 16, 16, 3, 10], [3, 3], [1; 2], [1; 2], [1; 4], 16),
        plan([1, 16, 16, 5, 17], [5, 5], [1; 2], [1; 2], [2; 4], 16),
        plan(
            [2, 24, 24, 13, 73],
            [3, 5],
            [1, 2],
            [2, 1],
            [2, 3, 2, 1],
            24,
        ),
    ] {
        let input = values(p.batch * p.cin * p.height * p.width, 137);
        let weights = values(p.cout * p.kh * p.kw, 139);
        let packed = pack(&p, &weights);
        let vertical = (0..p.oh).any(|y| {
            (0..p.kh).all(|ky| {
                (y * p.sh + ky * p.dh)
                    .checked_sub(p.pt)
                    .is_some_and(|sy| sy < p.height)
            })
        });
        let horizontal = (0..p.ow).any(|x| {
            x + PIXELS <= p.ow
                && (0..PIXELS).all(|pixel| {
                    (0..p.kw).all(|kx| {
                        ((x + pixel) * p.sw + kx * p.dw)
                            .checked_sub(p.pl)
                            .is_some_and(|sx| sx < p.width)
                    })
                })
        });
        assert_eq!(
            eligible(
                &p,
                &input,
                &[],
                Some(&packed),
                p.batch * p.oh * p.ow * p.cout,
                0
            ),
            vertical && horizontal,
            "height={} width={} kernel={}x{}",
            p.height,
            p.width,
            p.kh,
            p.kw,
        );
        // A worker band shorter than eight pixels cannot enter the native tile.
        assert!(!eligible(
            &p,
            &input,
            &[],
            Some(&packed),
            (PIXELS - 1) * p.cout,
            0
        ));
    }
}

#[test]
#[ignore = "local release AVX2 short-depthwise probe: two actual shapes, one warmup and three single-call pairs"]
fn short_depthwise_interior_comparison() {
    assert!(!cfg!(debug_assertions), "use a release build");
    assert!(simd::avx2_fma_available());
    #[cfg(windows)]
    // SAFETY: valid process pseudo-handle and writable mask addresses.
    unsafe {
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn GetProcessAffinityMask(
                process: *mut std::ffi::c_void,
                allowed: *mut usize,
                system: *mut usize,
            ) -> i32;
            fn SetProcessAffinityMask(process: *mut std::ffi::c_void, mask: usize) -> i32;
        }
        let process = GetCurrentProcess();
        let (mut allowed, mut system) = (0, 0);
        assert_ne!(
            GetProcessAffinityMask(process, &mut allowed, &mut system),
            0
        );
        assert_ne!(allowed, 0);
        assert_ne!(
            SetProcessAffinityMask(process, allowed.isolate_lowest_one()),
            0
        );
    }
    let deadline = std::time::Instant::now();
    for (channels, height, width, kernel) in [(64, 240, 152, 5), (32, 240, 152, 3)] {
        let p = plan(
            [1, channels, channels, height, width],
            [kernel; 2],
            [1; 2],
            [1; 2],
            [kernel / 2; 4],
            channels,
        );
        let input = values(channels * height * width, 113);
        let weights = values(channels * kernel * kernel, 127);
        let bias = values(channels, 131);
        let packed = pack(&p, &weights);
        let mut checked = vec![0.; channels * height * width];
        let mut interior = checked.clone();
        let run = |fast, output: &mut [f32]| {
            if fast {
                execute_control::<true>(
                    &p,
                    &input,
                    &weights,
                    &bias,
                    &packed,
                    output,
                    0,
                    &IdentityStore,
                );
            } else {
                execute_control::<false>(
                    &p,
                    &input,
                    &weights,
                    &bias,
                    &packed,
                    output,
                    0,
                    &IdentityStore,
                );
            }
        };
        run(false, &mut checked);
        run(true, &mut interior);
        assert_bits(&interior, &checked);
        assert!(
            deadline.elapsed().as_secs_f64() < 10.,
            "probe exceeded ten seconds"
        );
        let mut samples = [[0.; 3]; 2];
        for sample in 0..3 {
            for variant in if sample == 1 { [1, 0] } else { [0, 1] } {
                let output = if variant == 0 {
                    &mut checked
                } else {
                    &mut interior
                };
                let start = std::time::Instant::now();
                run(variant == 1, std::hint::black_box(output));
                samples[variant][sample] = start.elapsed().as_secs_f64() * 1000.;
                assert!(
                    deadline.elapsed().as_secs_f64() < 10.,
                    "probe exceeded ten seconds"
                );
            }
            assert_bits(&interior, &checked);
        }
        println!(
            "short_depthwise_interior {}",
            serde_json::json!({
                "channels":channels,"height":height,"width":width,"kernel":kernel,
                "warmups_per_variant":1,"calls_per_sample":1,"samples_per_variant":3,
                "checked_ms":samples[0],"interior_ms":samples[1],
                "order":[[0,1],[1,0],[0,1]],"outputs_equal_bits":true,
                "scope":"one logical CPU, cached synthetic inputs and packed weights, identity final store"
            })
        );
    }
}
