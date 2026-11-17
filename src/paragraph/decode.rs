//! Sparse evidence decoder and geometric reading order.

use super::features::{PreparedPage, Quad};
use crate::image::{ImageError, ImageResult};
use crate::onnx::BScalePartition;
use std::{cmp::Ordering, collections::HashSet};

const THRESHOLD: f32 = -1.0;
const SUPPORT: f64 = -0.65;
const VETO: f32 = -3.0;
const TOLERANCE: f64 = 1e-6;

pub(super) fn decode(
    page: &PreparedPage,
    logits: &[f32],
    partition: BScalePartition,
) -> ImageResult<Vec<Vec<usize>>> {
    if logits.len() != page.pairs.len() || logits.iter().any(|x| !x.is_finite()) {
        return Err(ImageError::new(
            "expected one finite paragraph logit per candidate",
        ));
    }
    let groups = match partition {
        BScalePartition::Evidence => cluster(page.quads.len(), &page.pairs, logits),
        BScalePartition::Connected => connected(page.quads.len(), &page.pairs, logits),
    };
    let mut result = Vec::with_capacity(groups.len());
    for vertical in [false, true] {
        let mut ordered: Vec<_> = groups
            .iter()
            .filter(|group| page.vertical[group[0]] == vertical)
            .map(|group| block(group, &page.quads, vertical))
            .collect();
        reading_sort(&mut ordered, |(y, x, _)| (*y, *x));
        result.extend(ordered.into_iter().map(|(_, _, group)| group));
    }
    Ok(result)
}

fn connected(n: usize, pairs: &[[usize; 2]], logits: &[f32]) -> Vec<Vec<usize>> {
    let mut parent: Vec<_> = (0..n).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for (&[a, b], &score) in pairs.iter().zip(logits) {
        if score >= 0.0 {
            let a = root(&mut parent, a);
            let b = root(&mut parent, b);
            parent[a] = b;
        }
    }
    let mut members = vec![Vec::new(); n];
    for i in 0..n {
        let component = root(&mut parent, i);
        members[component].push(i);
    }
    members
        .into_iter()
        .filter(|group| !group.is_empty())
        .collect()
}

fn cluster(n: usize, pairs: &[[usize; 2]], logits: &[f32]) -> Vec<Vec<usize>> {
    let mut members: Vec<_> = (0..n).map(|i| vec![i]).collect();
    if pairs.is_empty() {
        return members;
    }
    let mut proposals = Vec::new();
    let mut adjacent = vec![Vec::new(); n];
    for (i, (&[a, b], &score)) in pairs.iter().zip(logits).enumerate() {
        adjacent[a].push((b, score));
        adjacent[b].push((a, score));
        if score >= THRESHOLD {
            proposals.push(i);
        }
    }
    proposals.sort_by(|&i, &j| logits[j].partial_cmp(&logits[i]).unwrap());
    let mut component: Vec<_> = (0..n).collect();
    let mut failed = HashSet::new();
    for i in proposals {
        let [first, second] = pairs[i];
        let (mut a, mut b) = (component[first], component[second]);
        if a == b {
            continue;
        }
        if (members[a].len(), a) > (members[b].len(), b) {
            std::mem::swap(&mut a, &mut b);
        }
        let small = members[a].len();
        let key = (a, b, small, members[b].len());
        if failed.contains(&key) {
            continue;
        }

        let (mut total, mut minimum) = (0.0, 0.0_f32);
        let (mut count, mut agrees) = (0, 0);
        for &source in &members[a] {
            for &(destination, score) in &adjacent[source] {
                if component[destination] == b {
                    total += f64::from(score.clamp(-8.0, 8.0));
                    minimum = minimum.min(score);
                    count += 1;
                    agrees += usize::from(score >= THRESHOLD);
                }
            }
        }
        if total < SUPPORT * f64::from(count)
            || (minimum < VETO && agrees < 2 + usize::from(minimum < VETO - 3.0))
            || (count == 1 && small > 1 && logits[i] < THRESHOLD + 0.75)
        {
            failed.insert(key);
            continue;
        }
        let absorbed = std::mem::take(&mut members[a]);
        for &node in &absorbed {
            component[node] = b;
        }
        members[b].extend(absorbed);
    }
    members
        .into_iter()
        .filter(|group| !group.is_empty())
        .collect()
}

fn reading_sort<T>(items: &mut [T], position: impl Fn(&T) -> (f64, f64)) {
    // Treat near-equal rows as ties, just as the reference comparator does.
    // Its tolerance means the comparator is not necessarily transitive.
    for i in 1..items.len() {
        let value = position(&items[i]);
        let mut j = i;
        while j > 0 {
            let other = position(&items[j - 1]);
            let before = if (value.0 - other.0).abs() > TOLERANCE {
                value.0 < other.0
            } else {
                value.1 < other.1
            };
            if !before {
                break;
            }
            j -= 1;
        }
        items[j..=i].rotate_right(1);
    }
}

struct Bounds {
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
}

impl Bounds {
    fn center_y(&self) -> f64 {
        (self.top + self.bottom) * 0.5
    }
    fn center_x(&self) -> f64 {
        (self.left + self.right) * 0.5
    }

    fn overlaps_row(&self, other: &Self) -> bool {
        let overlap = self.bottom.min(other.bottom) - self.top.max(other.top);
        let height = (self.bottom - self.top).min(other.bottom - other.top);
        overlap >= 0.5 * height.max(TOLERANCE)
    }
}

fn block(group: &[usize], quads: &[Quad], vertical: bool) -> (f64, f64, Vec<usize>) {
    let mut group = group.to_vec();
    group.sort_unstable();
    let orient = |[x, y]: [f32; 2]| {
        let (x, y) = (f64::from(x), f64::from(y));
        if vertical { [y, -x] } else { [x, y] }
    };
    let points: Vec<_> = group.iter().map(|&i| quads[i].map(orient)).collect();
    let (mut sum_x, mut sum_y) = (0.0, 0.0);
    for q in &points {
        let mut heading = [0.0; 2];
        let (mut longest, mut shortest) = (-1.0, f64::INFINITY);
        for i in 0..4 {
            let a = q[i];
            let b = q[(i + 1) % 4];
            let [x, y] = [b[0] - a[0], b[1] - a[1]];
            let length = (x * x + y * y).sqrt();
            shortest = shortest.min(length);
            if length > longest {
                longest = length;
                heading = [x, y];
            }
        }
        if shortest > TOLERANCE {
            let [x, y] = heading;
            let sign = if x < 0.0 || (x.abs() <= TOLERANCE && y < 0.0) {
                -1.0
            } else {
                1.0
            };
            sum_x += sign * x;
            sum_y += sign * y;
        }
    }
    let norm = (sum_x * sum_x + sum_y * sum_y).sqrt();
    let (c, s) = if norm > TOLERANCE {
        (sum_x / norm, sum_y / norm)
    } else {
        (1.0, 0.0)
    };
    let bounds: Vec<_> = points
        .iter()
        .map(|q| {
            let mut out = Bounds {
                left: f64::INFINITY,
                right: f64::NEG_INFINITY,
                top: f64::INFINITY,
                bottom: f64::NEG_INFINITY,
            };
            for [x, y] in q {
                let (x, y) = (y.mul_add(s, x * c), y.mul_add(c, -x * s));
                out.left = out.left.min(x);
                out.right = out.right.max(x);
                out.top = out.top.min(y);
                out.bottom = out.bottom.max(y);
            }
            out
        })
        .collect();
    let mut order: Vec<_> = (0..group.len()).collect();
    reading_sort(&mut order, |&i| (bounds[i].center_y(), bounds[i].left));
    let mut rows = Vec::<Vec<usize>>::new();
    for i in order {
        if let Some(row) = rows
            .iter_mut()
            .find(|row| row.iter().all(|&j| bounds[i].overlaps_row(&bounds[j])))
        {
            row.push(i);
        } else {
            rows.push(vec![i]);
        }
    }
    let mut lines = Vec::with_capacity(group.len());
    for mut row in rows {
        row.sort_by(|&i, &j| {
            let (a, b) = (&bounds[i], &bounds[j]);
            a.left
                .partial_cmp(&b.left)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.top.partial_cmp(&b.top).unwrap_or(Ordering::Equal))
                .then_with(|| {
                    a.center_y()
                        .partial_cmp(&b.center_y())
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| {
                    a.center_x()
                        .partial_cmp(&b.center_x())
                        .unwrap_or(Ordering::Equal)
                })
        });
        lines.extend(row.into_iter().map(|i| group[i]));
    }
    let l = bounds.iter().map(|b| b.left).fold(f64::INFINITY, f64::min);
    let r = bounds
        .iter()
        .map(|b| b.right)
        .fold(f64::NEG_INFINITY, f64::max);
    let t = bounds.iter().map(|b| b.top).fold(f64::INFINITY, f64::min);
    let b = bounds
        .iter()
        .map(|b| b.bottom)
        .fold(f64::NEG_INFINITY, f64::max);
    let (x, y) = ((l + r) * 0.5, (t + b) * 0.5);
    (y.mul_add(c, x * s), y.mul_add(-s, x * c), lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(x: f32, y: f32, w: f32, h: f32) -> Quad {
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    }

    #[test]
    fn veto_prevents_chaining_and_order_keeps_original_indices() {
        let page = PreparedPage {
            quads: vec![
                quad(0., 35., 50., 10.),
                quad(0., 5., 60., 10.),
                quad(0., 20., 50., 10.),
            ],
            vertical: vec![false; 3],
            pairs: vec![[1, 2], [2, 0], [1, 0]],
            node_inputs: Vec::new(),
            edge_inputs: Vec::new(),
        };
        assert_eq!(
            decode(&page, &[2., 1., -4.], BScalePartition::Evidence).unwrap(),
            vec![vec![1, 2], vec![0]]
        );
        assert!(decode(&page, &[f32::NAN, 1., 2.], BScalePartition::Evidence).is_err());
        assert_eq!(
            decode(&page, &[2., 1., -4.], BScalePartition::Connected).unwrap(),
            vec![vec![1, 2, 0]],
        );
        assert_eq!(
            connected(4, &[[0, 1], [1, 2]], &[0., -0.1]),
            vec![vec![0, 1], vec![2], vec![3]]
        );
    }

    #[test]
    fn vertical_columns_read_top_to_bottom_from_the_right() {
        let page = PreparedPage {
            quads: vec![
                quad(12., 4., 5., 20.),
                quad(25., 30., 5., 20.),
                quad(25., 4., 5., 20.),
            ],
            vertical: vec![true; 3],
            pairs: vec![[1, 2]],
            node_inputs: Vec::new(),
            edge_inputs: Vec::new(),
        };
        // The reference orders paragraph centers, then orders the members in
        // their own rotated frame.
        assert_eq!(
            decode(&page, &[2.], BScalePartition::Evidence).unwrap(),
            vec![vec![2, 1], vec![0]]
        );
    }
}
