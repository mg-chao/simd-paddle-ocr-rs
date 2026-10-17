//! Bounded, paired native depthwise diagnostic. All setup and comparison stays
//! outside timed calls. Run in release with an external sixty-second timeout.
use super::*;
use std::time::{Duration, Instant};

#[test]
#[ignore = "local ARM64 release diagnostic: three shapes, one warm-up and three single calls per policy; external timeout 60s"]
#[allow(clippy::assertions_on_constants)]
fn arm64_depthwise_strip_fixed_diagnostic() {
    assert!(
        !cfg!(debug_assertions),
        "run this diagnostic in release mode"
    );
    assert!(simd::neon_available());
    let budget = Instant::now();
    for p in [
        plan([1, 96, 152, 240], [7, 7], [1, 1], [1, 1], [3; 4]),
        plan([1, 256, 240, 152], [9, 9], [1, 1], [1, 1], [4; 4]),
        plan([1, 256, 120, 76], [9, 9], [1, 1], [1, 1], [4; 4]),
    ] {
        let input = values(p.batch * p.height * p.width * p.cin, 3);
        let weights = values(p.cout * p.kh * p.kw, 5);
        let bias = values(p.cout, 7);
        let packed = pack(&p, &weights);
        let length = p.batch * p.oh * p.ow * p.cout;
        let mut outputs = [vec![0.; length], vec![0.; length], vec![0.; length]];
        let call = |policy: usize, output: &mut [f32]| {
            let accepted = match policy {
                0 => try_execute_policy::<false, 0>(
                    &p,
                    &input,
                    &weights,
                    &bias,
                    Some(&packed),
                    output,
                    0,
                    &IdentityStore,
                ),
                1 => try_execute_policy::<true, 0>(
                    &p,
                    &input,
                    &weights,
                    &bias,
                    Some(&packed),
                    output,
                    0,
                    &IdentityStore,
                ),
                2 => try_execute_policy::<true, 64>(
                    &p,
                    &input,
                    &weights,
                    &bias,
                    Some(&packed),
                    output,
                    0,
                    &IdentityStore,
                ),
                _ => unreachable!(),
            };
            assert!(accepted);
        };
        for (policy, output) in outputs.iter_mut().enumerate() {
            call(policy, output);
        }
        assert_bits(&outputs[1], &outputs[0]);
        assert_bits(&outputs[2], &outputs[0]);
        let mut samples_ms = [[0.; 3]; 3];
        for sample in 0..3 {
            for policy in if sample == 1 { [2, 1, 0] } else { [0, 1, 2] } {
                assert!(budget.elapsed() < Duration::from_secs(60));
                let timer = Instant::now();
                call(policy, std::hint::black_box(&mut outputs[policy]));
                samples_ms[policy][sample] = timer.elapsed().as_secs_f64() * 1000.;
                assert_bits(&outputs[1], &outputs[0]);
                assert_bits(&outputs[2], &outputs[0]);
            }
        }
        println!(
            "{}",
            serde_json::json!({
                "diagnostic": "arm64_depthwise_strip_fixed",
                "shape_nchw": [p.batch, p.cin, p.height, p.width],
                "kernel": [p.kh, p.kw],
                "threads": 1,
                "warmups_per_policy": 1,
                "calls_per_sample": 1,
                "order": "ABC,CBA,ABC",
                "current_ms": samples_ms[0],
                "strip_full_row_ms": samples_ms[1],
                "strip_64px_ms": samples_ms[2],
                "exact_output_bits": true,
            })
        );
        assert!(budget.elapsed() < Duration::from_secs(60));
    }
}
