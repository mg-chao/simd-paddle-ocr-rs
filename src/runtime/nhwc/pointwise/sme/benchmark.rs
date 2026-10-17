//! Fixed local crossover diagnostic; never part of the ordinary test suite.
use super::super::tests::{assert_bits, values};
use super::*;

#[test]
#[ignore = "local release diagnostic: three shapes, one warm-up and three single-call samples per kernel; external timeout 10s"]
#[allow(clippy::assertions_on_constants)] // Keep the ignored test buildable in debug.
fn sme_crossover_fixed() {
    assert!(!cfg!(debug_assertions), "use a release build");
    assert!(available(), "requires SME hardware");
    let started = std::time::Instant::now();
    for (shape, (rows, inner, columns)) in [(192, 192, 384), (1920, 256, 512), (960, 512, 1024)]
        .into_iter()
        .enumerate()
    {
        let input = values(rows * inner, 3);
        let weights = values(inner * columns, 5);
        let bias = values(columns, 7);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let mut scratch = vec![0.; GROUP * inner];
        let mut outputs = [vec![0.; rows * columns], vec![0.; rows * columns]];
        let mut run = |variant, output: &mut [f32]| {
            if variant == 0 {
                execute(
                    Backend::NeonFma,
                    &input,
                    &bias,
                    &packed,
                    output,
                    0,
                    inner,
                    columns,
                    &IdentityStore,
                );
            } else {
                assert!(try_execute(
                    &input,
                    &bias,
                    &packed,
                    output,
                    0,
                    inner,
                    columns,
                    &mut scratch,
                    &IdentityStore,
                ));
            }
        };
        for (variant, output) in outputs.iter_mut().enumerate() {
            run(variant, output);
        }
        assert_bits(&outputs[1], &outputs[0], "crossover warm-up");
        let mut samples = [[0.; 2]; 3];
        for (sample, timings) in samples.iter_mut().enumerate() {
            for variant in if (shape + sample) % 2 == 0 {
                [0, 1]
            } else {
                [1, 0]
            } {
                let timer = std::time::Instant::now();
                run(variant, std::hint::black_box(&mut outputs[variant]));
                timings[variant] = timer.elapsed().as_secs_f64() * 1000.;
            }
            assert_bits(&outputs[1], &outputs[0], "crossover measured");
            assert!(started.elapsed().as_secs_f64() < 10.);
        }
        println!(
            "sme_crossover {}",
            serde_json::json!({
                "rows": rows, "inner": inner, "columns": columns,
                "neon_ms": samples.map(|s| s[0]), "sme_ms": samples.map(|s| s[1]), "threads": 1,
                "warmups_per_kernel": 1, "measured_calls_per_kernel": 3,
                "outputs_equal_bits": true, "packing_included": true,
            })
        );
    }
}
