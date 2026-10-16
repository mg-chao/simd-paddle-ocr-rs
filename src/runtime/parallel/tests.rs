use super::*;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[test]
fn aligned_output_bands_stay_exclusive_on_the_requested_cpu_context() {
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 8 } else { 1 }).unwrap();
    for alignment in [1, 8, 16] {
        for rows in [1, 7, 8, 15, 16, 17, 145, 262_145] {
            for budget in [1, 2, 4, 8, 32] {
                let mut output = vec![usize::MAX; rows * 3];
                let visits = (0..rows).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
                let threads = Mutex::new(std::collections::HashSet::new());
                aligned_bands(
                    &mut output,
                    3,
                    rows,
                    budget,
                    alignment,
                    &cpu,
                    |band, first| {
                        cpu.assert_production_thread("aligned output band");
                        threads.lock().unwrap().insert(std::thread::current().id());
                        assert_eq!(first % alignment, 0);
                        for (row, values) in band.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                            visits[first + row].fetch_add(1, Ordering::Relaxed);
                            for (column, value) in values.iter_mut().enumerate() {
                                *value = (first + row) * 3 + column;
                            }
                        }
                    },
                );
                assert!(threads.lock().unwrap().len() <= budget.min(cpu.capacity()));
                assert!(visits.iter().all(|n| n.load(Ordering::Relaxed) == 1));
                assert!(output.iter().copied().eq(0..rows * 3));
            }
        }
    }
}

#[test]
fn aligned_output_boundaries_match_reference_floor_then_alignment() {
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 4 } else { 1 }).unwrap();
    for (alignment, expected) in [(8, [0, 32, 72, 104]), (16, [0, 32, 64, 96])] {
        let mut output = [0u8; 145];
        let starts = Mutex::new(Vec::new());
        aligned_bands(&mut output, 1, 145, 4, alignment, &cpu, |band, first| {
            starts.lock().unwrap().push(first);
            band.fill(1);
        });
        let mut starts = starts.into_inner().unwrap();
        starts.sort_unstable();
        if cfg!(feature = "parallel") {
            assert_eq!(starts, expected);
        } else {
            assert_eq!(starts, [0]);
        }
        assert_eq!(output, [1; 145]);
    }
    let called = AtomicUsize::new(0);
    aligned_bands(&mut [] as &mut [u8], 1, 0, 4, 16, &cpu, |_, _| {
        called.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(called.load(Ordering::Relaxed), 0);
}
