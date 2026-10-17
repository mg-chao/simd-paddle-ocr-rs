use super::*;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};

fn native_backends() -> Vec<Backend> {
    let mut result = vec![Backend::Portable4];
    for (available, backend) in [
        (simd::neon_available(), Backend::NeonFma),
        (simd::avx_available(), Backend::Avx),
        (simd::avx_fma_available(), Backend::AvxFma),
        (simd::avx2_fma_available(), Backend::Avx2Fma),
        (simd::avx512_available(), Backend::Avx512),
    ] {
        if available {
            result.push(backend);
        }
    }
    result
}

fn cpu() -> CpuContext {
    #[cfg(feature = "parallel")]
    let capacity = 8;
    #[cfg(not(feature = "parallel"))]
    let capacity = 1;
    CpuContext::new(capacity).unwrap()
}

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 17 + seed) % 31) as f32 / 32. - 0.5)
        .collect()
}

#[test]
fn six_row_bands_cover_exactly_once_and_balance_whole_reference_tiles() {
    for rows in [1, 5, 6, 7, 32, 96, 432, 433] {
        for budget in [1, 2, 4, 8, 16] {
            let bands = PointwiseBands::new(rows, 2_000_000, 1, budget);
            assert_eq!(bands.workers, budget.min(rows.div_ceil(6)));
            let mut visited = vec![0; rows];
            let mut smallest = usize::MAX;
            let mut largest = 0;
            for worker in 0..bands.workers {
                let first = bands.boundary(worker);
                let end = bands.boundary(worker + 1);
                assert_eq!(first % 6, 0);
                assert!(end > first);
                assert!(end == rows || end.is_multiple_of(6));
                for visits in &mut visited[first..end] {
                    *visits += 1;
                }
                let tiles = (end - first).div_ceil(6);
                smallest = smallest.min(tiles);
                largest = largest.max(tiles);
            }
            assert!(visited.into_iter().all(|visits| visits == 1));
            assert!(largest - smallest <= 1);
        }
    }
    let ninety_six = PointwiseBands::new(96, 256, 256, 8);
    assert_eq!(ninety_six.workers, 8);
    assert_eq!(ninety_six.boundary(1), 12);
    let four_thirty_two = PointwiseBands::new(432, 256, 256, 8);
    for worker in 0..8 {
        assert_eq!(
            four_thirty_two.boundary(worker + 1) - four_thirty_two.boundary(worker),
            54
        );
    }
}

#[test]
fn pointwise_parallel_threshold_is_two_million_and_never_exceeds_tiles_or_budget() {
    assert_eq!(PointwiseBands::new(32, 124, 500, 8).workers, 1);
    assert_eq!(PointwiseBands::new(32, 125, 500, 8).workers, 6);
    assert_eq!(PointwiseBands::new(96, 256, 256, 2).workers, 2);
    assert_eq!(PointwiseBands::new(96, 256, 256, 0).workers, 1);
    assert_eq!(PointwiseBands::new(5, 2_000_000, 1, 8).workers, 1);
    // Work overflow is certainly above the threshold, while the band geometry
    // itself remains valid even at theoretical sizes near usize::MAX.
    let huge = PointwiseBands::new(usize::MAX, usize::MAX, 16, 8);
    assert_eq!(huge.boundary(0), 0);
    assert_eq!(huge.boundary(8), usize::MAX);
}

#[cfg(feature = "parallel")]
#[test]
fn recursive_pointwise_jobs_borrow_disjoint_balanced_bands_on_the_shared_pool() {
    let cpu = cpu();
    for rows in [32, 96, 432, 433] {
        for budget in [1, 2, 4, 8] {
            let bands = PointwiseBands::new(rows, 256, 256, budget);
            let visits: Vec<_> = (0..rows).map(|_| AtomicUsize::new(0)).collect();
            let jobs = AtomicUsize::new(0);
            let mut output = vec![f32::NAN; rows * 3];
            cpu.install(|| {
                pointwise_bands(&mut output, 3, bands, |out, first| {
                    assert!(cpu.is_worker_thread());
                    jobs.fetch_add(1, Ordering::Relaxed);
                    for (row, values) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                        let global = first + row;
                        visits[global].fetch_add(1, Ordering::Relaxed);
                        values.fill(global as f32);
                    }
                });
            });
            assert_eq!(jobs.load(Ordering::Relaxed), bands.workers);
            assert!(
                visits
                    .iter()
                    .all(|count| count.load(Ordering::Relaxed) == 1)
            );
            for (row, values) in output.as_chunks::<3>().0.iter().enumerate() {
                assert_eq!(*values, [row as f32; 3]);
            }
        }
    }
}

struct IndexedStore<'a> {
    residual: &'a [f32],
}
impl StoreTransform for IndexedStore<'_> {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        value + self.residual[index]
    }
    fn vector<const LANES: usize>(
        &self,
        value: Simd<f32, LANES>,
        index: usize,
    ) -> Simd<f32, LANES> {
        value
            + Simd::from_array(std::array::from_fn(|lane| {
                self.residual.get(index + lane).copied().unwrap_or(0.)
            }))
    }
}

#[test]
fn pointwise_one_two_four_eight_budgets_preserve_ordered_math_and_global_store_indices() {
    let cpu = cpu();
    for (rows, inner, columns) in [
        (32, 512, 128),
        (96, 192, 128),
        (432, 64, 80),
        (433, 273, 17),
        (97, 191, 129),
    ] {
        let input = values(rows * inner, 3);
        let weights = values(inner * columns, 5);
        let bias = values(columns, 7);
        let residual = values(rows * columns, 11);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let store = IndexedStore {
            residual: &residual,
        };
        for backend in native_backends() {
            let mut expected = vec![0.; rows * columns];
            for (row, output) in expected.chunks_exact_mut(columns).enumerate() {
                for (column, value) in output.iter_mut().enumerate() {
                    let mut sum = bias[column];
                    for k in 0..inner {
                        let x = input[row * inner + k];
                        let w = weights[k * columns + column];
                        sum = if backend.uses_fma() {
                            x.mul_add(w, sum)
                        } else {
                            x * w + sum
                        };
                    }
                    *value = store.scalar(sum, row * columns + column);
                }
            }
            for threads in [1, 2, 4, 8] {
                let mut actual = vec![f32::NAN; expected.len()];
                pointwise_execute(
                    backend,
                    &input,
                    &bias,
                    &packed,
                    &mut actual,
                    inner,
                    columns,
                    &cpu,
                    threads,
                    &store,
                );
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "backend={backend:?} rows={rows} columns={columns} budget={threads}"
                );
            }
        }
    }
}

#[test]
fn constant_matrix_entry_preserves_whole_outputs_at_each_thread_budget() {
    let (rows, inner, columns) = (96, 192, 128);
    let input = values(rows * inner, 13);
    let weights = values(inner * columns, 17);
    let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
    let cpu = cpu();
    for backend in native_backends() {
        let mut expected = vec![0.; rows * columns];
        pointwise::execute(
            backend,
            &input,
            &[],
            &packed,
            &mut expected,
            0,
            inner,
            columns,
            &simd::IdentityStore,
        );
        for threads in [1, 2, 4, 8] {
            let mut actual = vec![f32::NAN; expected.len()];
            assert!(try_pointwise_matrix(
                backend,
                &input,
                &packed,
                &mut actual,
                rows,
                inner,
                columns,
                &cpu,
                threads
            ));
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
}

#[test]
fn specialized_bands_decline_scalar_unpacked_dense_stride_padding_and_depthwise() {
    let p = Conv {
        batch: 1,
        cin: 16,
        cout: 32,
        height: 3,
        width: 32,
        oh: 3,
        ow: 32,
        cpg: 16,
        opg: 32,
        kh: 1,
        kw: 1,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
        pt: 0,
        pl: 0,
    };
    assert!(packed_pointwise_eligible(Backend::Portable4, &p, true));
    assert!(packed_pointwise_eligible(Backend::Scalar, &p, true));
    assert!(!packed_pointwise_eligible(Backend::Portable4, &p, false));
    for other in [
        Conv { kh: 3, kw: 3, ..p },
        Conv { sw: 2, ..p },
        Conv { pt: 1, ..p },
        Conv {
            cpg: 1,
            opg: 1,
            ..p
        },
        Conv {
            cin: 1,
            cout: 1,
            cpg: 1,
            opg: 1,
            ..p
        },
    ] {
        assert!(!packed_pointwise_eligible(Backend::Portable4, &other, true));
    }
}
