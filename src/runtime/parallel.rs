//! Exclusive aligned output bands on the request's bounded CPU context.
use crate::cpu::CpuContext;

/// Every band owns complete rows and the final band owns any alignment tail.
/// Recursive stack jobs avoid a temporary Vec of mutable slices or closures.
#[allow(clippy::too_many_arguments)]
pub(crate) fn aligned_bands<T: Send>(
    output: &mut [T],
    row_length: usize,
    rows: usize,
    workers: usize,
    alignment: usize,
    cpu: &CpuContext,
    action: impl Fn(&mut [T], usize) + Sync,
) {
    assert!(alignment.is_power_of_two());
    assert_eq!(
        output.len(),
        rows.checked_mul(row_length).expect("band size overflow")
    );
    if rows == 0 || row_length == 0 {
        return;
    }
    let workers = workers
        .max(1)
        .min(cpu.capacity())
        .min((rows / alignment).max(1));
    #[cfg(not(feature = "parallel"))]
    let workers = {
        let _ = workers;
        1
    };
    cpu.install(|| {
        #[cfg(feature = "parallel")]
        visit(
            output, row_length, rows, workers, alignment, 0, workers, &action,
        );
        #[cfg(not(feature = "parallel"))]
        {
            let _ = workers;
            action(output, 0);
        }
    });
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn visit<T: Send>(
    output: &mut [T],
    row_length: usize,
    rows: usize,
    workers: usize,
    alignment: usize,
    first_worker: usize,
    worker_count: usize,
    action: &(impl Fn(&mut [T], usize) + Sync),
) {
    let boundary = |worker| {
        if worker == workers {
            rows
        } else {
            (rows as u128 * worker as u128 / workers as u128) as usize & !(alignment - 1)
        }
    };
    let first = boundary(first_worker);
    if worker_count == 1 {
        action(output, first);
        return;
    }
    let left_count = worker_count / 2;
    let middle = first_worker + left_count;
    let split = (boundary(middle) - first) * row_length;
    let (left, right) = output.split_at_mut(split);
    rayon::join(
        || {
            visit(
                left,
                row_length,
                rows,
                workers,
                alignment,
                first_worker,
                left_count,
                action,
            )
        },
        || {
            visit(
                right,
                row_length,
                rows,
                workers,
                alignment,
                middle,
                worker_count - left_count,
                action,
            )
        },
    );
}

#[cfg(test)]
mod tests;
