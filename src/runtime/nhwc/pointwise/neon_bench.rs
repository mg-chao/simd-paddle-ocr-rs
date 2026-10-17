//! Fixed local diagnostic for ARM64 pointwise attribution. Run only in release
//! mode, with an external ten-second process timeout as well as this internal
//! deadline. One warm-up and three single calls per backend/shape; no growth.
use super::*;
use std::time::{Duration, Instant};

fn check_budget(started: Instant) {
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "fixed ARM64 pointwise diagnostic exceeded ten seconds"
    );
}

#[allow(clippy::too_many_arguments)]
fn spot_oracle(
    backend: Backend,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    inner: usize,
    columns: usize,
    row: usize,
    column: usize,
) -> f32 {
    let mut expected = bias[column];
    for k in 0..inner {
        let value = input[row * inner + k];
        let weight = weights[k * columns + column];
        expected = if backend == Backend::NeonFma {
            value.mul_add(weight, expected)
        } else {
            value * weight + expected
        };
    }
    expected
}

#[test]
#[ignore = "local ARM64 release diagnostic: two shapes, one warm-up and three single calls per policy; external timeout 10s"]
#[allow(clippy::assertions_on_constants)] // Runtime guard keeps ignored tests buildable in debug.
fn arm64_pointwise_fixed_diagnostic() {
    assert!(
        !cfg!(debug_assertions),
        "run this diagnostic in release mode"
    );
    let started = Instant::now();
    for (shape_index, (rows, inner, columns)) in
        [(96, 384, 768), (36_480, 96, 24)].into_iter().enumerate()
    {
        check_budget(started);
        let input = values(rows * inner, 3);
        let weights = values(inner * columns, 5);
        let bias = values(columns, 7);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let mut output = vec![0.; rows * columns];
        let order = if shape_index == 0 {
            [Backend::Portable4, Backend::NeonFma]
        } else {
            [Backend::NeonFma, Backend::Portable4]
        };
        for (order_in_shape, backend) in order.into_iter().enumerate() {
            let mut spots = Vec::new();
            for row in [0, 5, rows - 1] {
                for column in [0, 7, columns - 1] {
                    spots.push((
                        row * columns + column,
                        spot_oracle(
                            backend, &input, &weights, &bias, inner, columns, row, column,
                        )
                        .to_bits(),
                    ));
                }
            }
            let check_spots = |output: &[f32]| {
                for &(index, expected) in &spots {
                    assert_eq!(
                        output[index].to_bits(),
                        expected,
                        "{backend:?} shape={rows}x{inner}x{columns} index={index}"
                    );
                }
            };
            let call = |output: &mut [f32]| {
                execute(
                    backend,
                    &input,
                    &bias,
                    &packed,
                    output,
                    0,
                    inner,
                    columns,
                    &IdentityStore,
                );
            };
            check_budget(started);
            call(std::hint::black_box(&mut output));
            check_budget(started);
            check_spots(&output);
            let mut samples_ms = [0.; 3];
            for sample in &mut samples_ms {
                check_budget(started);
                let timer = Instant::now();
                call(std::hint::black_box(&mut output));
                *sample = timer.elapsed().as_secs_f64() * 1000.;
                check_budget(started);
                check_spots(&output);
            }
            println!(
                "{}",
                serde_json::json!({
                    "diagnostic": "arm64_pointwise_fixed",
                    "shape_index": shape_index,
                    "rows": rows,
                    "inner": inner,
                    "columns": columns,
                    "backend": backend,
                    "order_in_shape": order_in_shape,
                    "threads": 1,
                    "warmups": 1,
                    "calls_per_sample": 1,
                    "samples_ms": samples_ms,
                    "exact_oracle_spots": spots.len(),
                    "oracle_fma": backend == Backend::NeonFma,
                    "budget_seconds": 10,
                })
            );
        }
    }
    check_budget(started);
}
