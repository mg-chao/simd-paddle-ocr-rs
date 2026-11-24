//! NumPy's scalar float64 argsort, used by its AArch64 build. Equal aspect
//! ratios can straddle a six-line group and receive different canvas widths.
//! Adapted from NumPy 2.2.6 aquicksort (see licenses/numpy.txt).
type Entry = (f64, usize);

pub(super) fn sort(entries: &mut [Entry]) {
    if let Some(depth) = entries.len().checked_ilog2() {
        quicksort(entries, (2 * depth) as i32);
    }
}

fn quicksort(mut entries: &mut [Entry], mut depth: i32) {
    while entries.len() > 16 {
        let last = entries.len() - 1;
        let middle = last / 2;
        for (a, b) in [(0, middle), (middle, last), (0, middle)] {
            if entries[b].0 < entries[a].0 {
                entries.swap(a, b);
            }
        }
        let pivot = entries[middle].0;
        entries.swap(middle, last - 1);
        let (mut left, mut right) = (0, last - 1);
        loop {
            left += 1;
            while entries[left].0 < pivot {
                left += 1;
            }
            right -= 1;
            while pivot < entries[right].0 {
                right -= 1;
            }
            if left >= right {
                break;
            }
            entries.swap(left, right);
        }
        entries.swap(left, last - 1);
        let (lo, rest) = entries.split_at_mut(left);
        let hi = &mut rest[1..];
        // NumPy tests the depth limit when popping the larger partition.
        depth -= 1;
        let (small, large) = if lo.len() < hi.len() {
            (lo, hi)
        } else {
            (hi, lo)
        };
        quicksort(small, depth);
        if depth < 0 {
            heapsort(large);
            return;
        }
        entries = large;
    }
    for i in 1..entries.len() {
        let mut j = i;
        while j > 0 && entries[j].0 < entries[j - 1].0 {
            entries.swap(j, j - 1);
            j -= 1;
        }
    }
}

fn heapsort(entries: &mut [Entry]) {
    fn sift(heap: &mut [Entry], mut root: usize) {
        while 2 * root + 1 < heap.len() {
            let mut child = root * 2 + 1;
            if child + 1 < heap.len() && heap[child].0 < heap[child + 1].0 {
                child += 1;
            }
            if heap[root].0 >= heap[child].0 {
                break;
            }
            heap.swap(root, child);
            root = child;
        }
    }
    for root in (0..entries.len() / 2).rev() {
        sift(entries, root);
    }
    for last in (1..entries.len()).rev() {
        entries.swap(0, last);
        sift(&mut entries[..last], 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tied_ratios_follow_numpy_scalar_partition_order() {
        let mut entries: Vec<_> = (0..273).map(|i| (((i * 13 + 7) % 11) as f64, i)).collect();
        sort(&mut entries);
        assert!(entries.windows(2).all(|p| p[0].0 <= p[1].0));
        let order: Vec<_> = entries.into_iter().map(|e| e.1).collect();
        assert_eq!(
            &order[..13],
            &[112, 46, 222, 35, 200, 57, 189, 233, 178, 68, 24, 244, 211]
        );
        assert_eq!(&order[266..], &[95, 18, 216, 172, 139, 128, 271]);
    }
}
