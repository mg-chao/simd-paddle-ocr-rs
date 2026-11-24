//! Index permutation used by the reference NumPy float64 argsort on AVX2.
//! A tie at a batch boundary changes the canvas and therefore the model input.
//! Scalar implementation of the x86-simd-sort network/partition order. See
//! licenses/x86-simd-sort.txt. Ratios are finite and strictly positive.
const NETWORK: usize = 256;
const BLOCK: usize = 16;
type Entry = (f64, usize);

pub(super) fn sort(entries: &mut [Entry]) {
    if !entries.is_empty() {
        partition_sort(entries, 2 * entries.len().ilog2());
    }
}

fn partition_sort(entries: &mut [Entry], depth: u32) {
    if entries.len() <= NETWORK {
        let length = entries.len().next_power_of_two();
        let mut padded = [(f64::INFINITY, 0); NETWORK];
        padded[..entries.len()].copy_from_slice(entries);
        let mut width = 2;
        while width <= length {
            for group in padded[..length].chunks_exact_mut(width) {
                // Merge two ascending runs. A compare/exchange preserves ties
                // in their current lanes, including comparisons across runs.
                for i in 0..width / 2 {
                    exchange(group, i, width - 1 - i);
                }
                let mut stride = width / 4;
                while stride > 0 {
                    for part in group.chunks_exact_mut(2 * stride) {
                        for i in 0..stride {
                            exchange(part, i, i + stride);
                        }
                    }
                    stride /= 2;
                }
            }
            width *= 2;
        }
        entries.copy_from_slice(&padded[..entries.len()]);
        return;
    }
    if depth == 0 {
        entries.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        return;
    }
    let step = (entries.len() - 1) / 4;
    let mut sample = [1, 2, 3, 4].map(|i| entries[i * step].0);
    sample.sort_unstable_by(f64::total_cmp);
    let pivot = sample[2];
    let middle = partition(entries, pivot);
    let (lower, upper) = entries.split_at_mut(middle);
    if lower.iter().any(|e| e.0 != pivot) {
        partition_sort(lower, depth - 1);
    }
    if upper.iter().any(|e| e.0 != pivot) {
        partition_sort(upper, depth - 1);
    }
}

fn exchange(entries: &mut [Entry], a: usize, b: usize) {
    if entries[a].0 > entries[b].0 {
        entries.swap(a, b);
    }
}

fn partition(entries: &mut [Entry], pivot: f64) -> usize {
    let (mut left, mut right) = (0, entries.len());
    for _ in 0..entries.len() % BLOCK {
        if entries[left].0 < pivot {
            left += 1;
        } else {
            right -= 1;
            entries.swap(left, right);
        }
    }
    let (mut write_left, mut write_right) = (left, right);
    let first: [Entry; BLOCK] = entries[left..left + BLOCK].try_into().unwrap();
    let last: [Entry; BLOCK] = entries[right - BLOCK..right].try_into().unwrap();
    left += BLOCK;
    right -= BLOCK;
    while left < right {
        let start = if write_right - right < left - write_left {
            right -= BLOCK;
            right
        } else {
            left += BLOCK;
            left - BLOCK
        };
        let block: [Entry; BLOCK] = entries[start..start + BLOCK].try_into().unwrap();
        distribute(entries, block, pivot, &mut write_left, &mut write_right);
    }
    for block in [first, last] {
        distribute(entries, block, pivot, &mut write_left, &mut write_right);
    }
    write_left
}

fn distribute(
    entries: &mut [Entry],
    block: [Entry; BLOCK],
    pivot: f64,
    left: &mut usize,
    right: &mut usize,
) {
    for e in block {
        if e.0 < pivot {
            entries[*left] = e;
            *left += 1;
        } else {
            *right -= 1;
            entries[*right] = e;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_aspect_crops_keep_reference_batch_membership() {
        let order = |n| {
            let mut entries: Vec<_> = (0..n).map(|i| (((i * 13 + 7) % 11) as f64, i)).collect();
            sort(&mut entries);
            assert!(entries.windows(2).all(|p| p[0].0 <= p[1].0));
            entries.into_iter().map(|e| e.1).collect::<Vec<_>>()
        };
        assert_eq!(order(13), [2, 8, 3, 9, 4, 10, 5, 0, 11, 6, 12, 1, 7]);
        let long = order(273);
        assert_eq!(
            &long[..13],
            &[24, 244, 233, 255, 222, 211, 46, 35, 200, 68, 145, 79, 156]
        );
        assert_eq!(&long[266..], &[183, 249, 227, 40, 238, 29, 18]);
    }
}
