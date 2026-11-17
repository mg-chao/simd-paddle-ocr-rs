//! Input features for the epoch 15 B-Scale graph.

mod appearance;
mod layout;

use crate::{
    error::ErrorKind,
    image::{ImageError, ImageResult, ImageView, push, resize_vec},
    runtime::ExecutionLimits,
};
use std::{cmp::Ordering, collections::VecDeque};

pub(super) type Quad = [[f32; 2]; 4];
type Point = [f32; 2];
type Point64 = [f64; 2];
type Bounds = [f32; 4];

pub(super) struct PreparedPage {
    pub(super) quads: Vec<Quad>,
    pub(super) vertical: Vec<bool>,
    pub(super) pairs: Vec<[usize; 2]>,
    pub(super) node_inputs: Vec<f32>,
    pub(super) edge_inputs: Vec<f32>,
}

impl PreparedPage {
    pub(super) fn retained_bytes(&self) -> usize {
        self.quads.capacity() * size_of::<Quad>()
            + self.vertical.capacity() * size_of::<bool>()
            + self.pairs.capacity() * size_of::<[usize; 2]>()
            + (self.node_inputs.capacity() + self.edge_inputs.capacity()) * size_of::<f32>()
    }
}

fn too_large() -> ImageError {
    ImageError::with_kind(
        ErrorKind::ResourceLimit,
        "paragraph features exceed execution limits",
    )
}

pub(super) fn prepare(
    image: ImageView<'_>,
    quads: &[Quad],
    limits: ExecutionLimits,
) -> ImageResult<PreparedPage> {
    const NODE_BYTES: usize = size_of::<Quad>() + size_of::<bool>() + 3 * size_of::<f32>();
    const PAIR_BYTES: usize = size_of::<[usize; 2]>() + 2 * 26 * size_of::<f32>();
    let remaining = quads
        .len()
        .checked_mul(NODE_BYTES)
        .and_then(|bytes| limits.max_activation_bytes.checked_sub(bytes))
        .ok_or_else(too_large)?;
    if quads.iter().flatten().flatten().any(|x| !x.is_finite()) {
        return Err(ImageError::new(
            "paragraph quads must have finite coordinates",
        ));
    }
    let mut points = Vec::new();
    resize_vec(&mut points, quads.len())?;
    for (out, &quad) in points.iter_mut().zip(quads) {
        *out = canonical(quad);
    }
    let vertical = directions(&points);
    let info: Vec<_> = points.iter().copied().map(Measure::new).collect();
    let pairs = candidates(&points, &vertical, remaining / PAIR_BYTES)?;
    let mut page = PreparedPage {
        quads: points,
        vertical,
        pairs,
        node_inputs: Vec::new(),
        edge_inputs: Vec::new(),
    };
    resize_vec(&mut page.node_inputs, info.len() * 3)?;
    if page.pairs.is_empty() {
        return Ok(page);
    }

    let appearance: Vec<_> = page
        .quads
        .iter()
        .map(|quad| appearance::line(image, quad))
        .collect();
    for direction in [false, true] {
        let mut thicknesses: Vec<_> = info
            .iter()
            .zip(&page.vertical)
            .filter_map(|(m, &v)| (v == direction && m.scale_valid()).then_some(m.short))
            .collect();
        let scale = median(&mut thicknesses, 1.0);
        for (i, ((m, a), &v)) in info.iter().zip(&appearance).zip(&page.vertical).enumerate() {
            if direction == v && m.valid {
                let row = &mut page.node_inputs[i * 3..][..3];
                if m.scale_valid() {
                    row[0] = (m.long / scale).ln_1p() as f32;
                    row[1] = (m.short / m.long) as f32;
                }
                row[2] = a[2];
            }
        }
    }
    let layouts = layout::pair_features(&page.quads, &page.vertical, &page.pairs);
    let edges = page.pairs.len();
    resize_vec(&mut page.edge_inputs, edges * 2 * 26)?;
    for (index, &[a, b]) in page.pairs.iter().enumerate() {
        if !info[a].valid || !info[b].valid {
            return Err(ImageError::new(
                "paragraph candidate contains a degenerate quad",
            ));
        }
        let style = appearance::pair(&appearance[a], &appearance[b]);
        for (row, left, right) in [(index, a, b), (index + edges, b, a)] {
            let out = &mut page.edge_inputs[row * 26..][..26];
            out[..12].copy_from_slice(&info[left].geometry(&info[right]));
            out[12..18].copy_from_slice(&style);
            out[18..].copy_from_slice(&layouts[index]);
        }
    }
    if page
        .node_inputs
        .iter()
        .chain(&page.edge_inputs)
        .any(|value| !value.is_finite())
    {
        return Err(ImageError::new("nonfinite paragraph measurements"));
    }
    Ok(page)
}

fn sub<T: core::ops::Sub<Output = T> + Copy>(a: [T; 2], b: [T; 2]) -> [T; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot(a: Point64, b: Point64) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn dot32(a: Point, b: Point) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross<T: core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + Copy>(
    a: [T; 2],
    b: [T; 2],
) -> T {
    a[0] * b[1] - a[1] * b[0]
}

fn cmp<T: PartialOrd>(a: T, b: T) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

fn edges<T: core::ops::Sub<Output = T> + Copy>(q: &[[T; 2]; 4]) -> [[T; 2]; 4] {
    std::array::from_fn(|i| sub(q[(i + 1) % 4], q[i]))
}

fn twice_area<T>(q: &[[T; 2]; 4]) -> T
where
    T: core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::iter::Sum + Copy,
{
    (0..4).map(|i| cross(q[i], q[(i + 1) % 4])).sum()
}

fn canonical(mut q: Quad) -> Quad {
    let center: Point = std::array::from_fn(|axis| q.iter().map(|p| p[axis]).sum::<f32>() * 0.25);
    q.sort_by(|&a, &b| {
        let a = sub(a, center);
        let b = sub(b, center);
        cmp(a[1].atan2(a[0]), b[1].atan2(b[0]))
    });
    let start = (0..4)
        .min_by(|&a, &b| cmp(q[a][1], q[b][1]).then(cmp(q[a][0], q[b][0])))
        .unwrap();
    q.rotate_left(start);
    q
}

fn longest<T: PartialOrd + Copy>(lengths: &[T; 4]) -> usize {
    let mut largest = 0;
    for i in 1..4 {
        if lengths[i] > lengths[largest] {
            largest = i;
        }
    }
    largest
}

fn bounds(q: &Quad) -> Bounds {
    let mut b = [q[0][0], q[0][1], q[0][0], q[0][1]];
    for &p in &q[1..] {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    b
}

fn top_left_distance(corners: &[Point64], a: usize, b: usize) -> f64 {
    let d = sub(corners[a], corners[b]);
    dot(d, d)
}

fn directions(quads: &[Quad]) -> Vec<bool> {
    let mut result = Vec::with_capacity(quads.len());
    let mut anchors = Vec::new();
    let mut strongest_ratio = -1.0;
    let mut strongest_direction = false;
    for (i, quad) in quads.iter().enumerate() {
        let sides = edges(&quad.map(|p| p.map(f64::from)));
        let lengths = sides.map(|p| p[0].hypot(p[1]));
        let major = longest(&lengths);
        let short = lengths.into_iter().fold(f64::INFINITY, f64::min);
        let ratio = if short > 1e-6 {
            lengths[major] / short
        } else {
            0.0
        };
        let vertical = ratio > 0.0 && sides[major][1].abs() > sides[major][0].abs();
        result.push(vertical);
        if ratio >= 1.2 {
            anchors.push(i);
        }
        if ratio > strongest_ratio {
            strongest_ratio = ratio;
            strongest_direction = vertical;
        }
    }
    if anchors.is_empty() || anchors.iter().all(|&i| result[i] == strongest_direction) {
        result.fill(strongest_direction);
        return result;
    }

    // Resolve ambiguous lines by following their nearest neighbor. A cycle
    // inherits the direction of the closest assigned line.
    let corners: Vec<_> = quads
        .iter()
        .map(|q| {
            let [x, y, _, _] = bounds(q);
            [f64::from(x), f64::from(y)]
        })
        .collect();
    let mut assigned = vec![false; quads.len()];
    let mut children = vec![Vec::new(); quads.len()];
    for &i in &anchors {
        assigned[i] = true;
    }
    for i in (0..quads.len()).filter(|&i| !assigned[i]) {
        let parent = (0..quads.len())
            .filter(|&j| j != i)
            .min_by(|&a, &b| {
                cmp(
                    top_left_distance(&corners, i, a),
                    top_left_distance(&corners, i, b),
                )
            })
            .unwrap();
        children[parent].push(i);
    }
    let mut nearest = vec![(f64::INFINITY, usize::MAX); quads.len()];
    let update = |source: usize, assigned: &[bool], nearest: &mut [(f64, usize)]| {
        for (i, (best, parent)) in nearest
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| !assigned[*i])
        {
            let distance = top_left_distance(&corners, source, i);
            let tied = (distance - *best).abs() * 1e12 <= distance.min(*best);
            if distance < *best || (tied && source < *parent) {
                *best = distance;
                *parent = source;
            }
        }
    };
    let adopt = |source: usize,
                 assigned: &mut [bool],
                 nearest: &mut [(f64, usize)],
                 result: &mut [bool]| {
        let mut queue = VecDeque::from([source]);
        while let Some(parent) = queue.pop_front() {
            for &i in &children[parent] {
                if !assigned[i] {
                    result[i] = result[parent];
                    assigned[i] = true;
                    update(i, assigned, nearest);
                    queue.push_back(i);
                }
            }
        }
    };
    for &i in &anchors {
        update(i, &assigned, &mut nearest);
    }
    for i in anchors {
        adopt(i, &mut assigned, &mut nearest, &mut result);
    }
    while let Some(i) = (0..quads.len())
        .filter(|&i| !assigned[i])
        .min_by(|&a, &b| cmp(nearest[a].0, nearest[b].0))
    {
        result[i] = result[nearest[i].1];
        assigned[i] = true;
        update(i, &assigned, &mut nearest);
        adopt(i, &mut assigned, &mut nearest, &mut result);
    }
    result
}

fn aligned_frame(quads: &[Quad], indices: &[usize], valid: &[bool], vertical: bool) -> Vec<Bounds> {
    let orient = |p: Point| if vertical { [p[1], p[0]] } else { p };
    let mut axis = [0.0_f32; 2];
    for (&i, &valid) in indices.iter().zip(valid) {
        if !valid {
            continue;
        }
        let sides = edges(&quads[i].map(orient));
        let lengths = sides.map(|p| dot32(p, p).sqrt());
        let major = longest(&lengths);
        let short = lengths.into_iter().fold(f32::INFINITY, f32::min);
        let [mut x, mut y] = sides[major];
        if x < 0.0 {
            x = -x;
            y = -y;
        }
        if lengths[major] >= 1.2 * short && x.abs() >= y.abs() {
            axis[0] += x;
            axis[1] += y;
        }
    }
    let norm = dot32(axis, axis).sqrt();
    let [c, s] = if norm > 1e-6 {
        [axis[0] / norm, axis[1] / norm]
    } else {
        [1.0, 0.0]
    };
    indices
        .iter()
        .map(|&i| {
            bounds(&quads[i].map(|p| {
                let [x, y] = orient(p);
                [y.mul_add(s, x * c), y.mul_add(c, -x * s)]
            }))
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Neighbor {
    metric: f32,
    index: usize,
}

fn nominate<const K: usize>(best: &mut [Neighbor; K], metric: f32, index: usize) {
    // The traversal is in input order. Equality preserves NumPy's stable argsort.
    if let Some(pos) = best.iter().position(|n| metric < n.metric) {
        best.copy_within(pos..K - 1, pos + 1);
        best[pos] = Neighbor { metric, index };
    }
}

fn candidates(quads: &[Quad], vertical: &[bool], max_pairs: usize) -> ImageResult<Vec<[usize; 2]>> {
    const EMPTY: Neighbor = Neighbor {
        metric: f32::INFINITY,
        index: usize::MAX,
    };
    let mut pairs = Vec::new();
    for direction in [false, true] {
        let indices: Vec<_> = quads
            .iter()
            .zip(vertical)
            .enumerate()
            .filter_map(|(i, (q, &v))| {
                (v == direction
                    && twice_area(q).abs() > 1e-3
                    && edges(q).iter().all(|&p| dot32(p, p) > 1e-6))
                .then_some(i)
            })
            .collect();
        if indices.len() < 2 {
            continue;
        }
        let bounds = aligned_frame(quads, &indices, &vec![true; indices.len()], direction);
        let sizes: Vec<_> = bounds
            .iter()
            .map(|&[l, t, r, b]| [(r - l).max(0.25), (b - t).max(0.25)])
            .collect();
        let centers: Vec<_> = bounds
            .iter()
            .map(|&[l, t, r, b]| [(r + l) * 0.5, (b + t) * 0.5])
            .collect();
        let mut order: Vec<_> = (0..indices.len()).collect();
        order.sort_by(|&i, &j| {
            cmp(centers[i][1], centers[j][1]).then(cmp(centers[i][0], centers[j][0]))
        });
        let mut rank = vec![0; order.len()];
        for (position, index) in order.into_iter().enumerate() {
            rank[index] = position;
        }
        let mut picks = Vec::new();
        let max_picks = max_pairs.saturating_sub(pairs.len()).saturating_mul(2);
        for (i, &a) in bounds.iter().enumerate() {
            let (mut near_text, mut near_center) = ([EMPTY; 6], [EMPTY; 5]);
            let (mut above, mut below) = ([EMPTY; 4], [EMPTY; 4]);
            let (mut left, mut right) = ([EMPTY; 2], [EMPTY; 2]);
            for (j, &b) in bounds.iter().enumerate() {
                if i == j {
                    continue;
                }
                let [dx, dy] = sub(centers[j], centers[i]);
                let scale = (sizes[i][1] + sizes[j][1]) * 0.5;
                let x_overlap = a[2].min(b[2]) - a[0].max(b[0]);
                let y_overlap = a[3].min(b[3]) - a[1].max(b[1]);
                let align_gap = (b[0] - a[0]).abs().min((b[2] - a[2]).abs());
                if dy.abs() < 15.0 * scale || dx.abs() < 15.0 * scale {
                    nominate(
                        &mut near_text,
                        ((-x_overlap).max(0.0) + 0.18 * dx.abs()).hypot(dy),
                        j,
                    );
                    nominate(&mut near_center, dx.hypot(dy), j);
                }
                if (x_overlap.max(0.0) / sizes[i][0].min(sizes[j][0]) > 0.16
                    || align_gap < 0.9 * scale)
                    && dy.abs() > 0.35 * scale
                {
                    let score = (dy.abs() + 0.25 * align_gap) / scale;
                    nominate(if dy < 0.0 { &mut above } else { &mut below }, score, j);
                }
                if y_overlap.max(0.0) / sizes[i][1].min(sizes[j][1]) > 0.50
                    && dy.abs() < 0.75 * scale
                {
                    let score = (x_overlap.abs() + dy.abs()) / scale;
                    if dx < 0.0 {
                        nominate(&mut left, score, j);
                    }
                    if dx > 0.0 {
                        nominate(&mut right, score, j);
                    }
                }
            }
            // Deduplicate each endpoint's nominations before writing the sparse
            // union. Every unique pair can appear at most twice globally.
            let mut selected = [usize::MAX; 23];
            for (out, n) in selected.iter_mut().zip(
                near_text
                    .into_iter()
                    .chain(near_center)
                    .chain(above)
                    .chain(below)
                    .chain(left)
                    .chain(right),
            ) {
                *out = n.index;
            }
            selected.sort_unstable();
            let mut previous = usize::MAX;
            for j in selected {
                if j == usize::MAX || j == previous {
                    continue;
                }
                previous = j;
                if picks.len() >= max_picks {
                    return Err(too_large());
                }
                push(
                    &mut picks,
                    if rank[i] < rank[j] {
                        [indices[i], indices[j]]
                    } else {
                        [indices[j], indices[i]]
                    },
                )?;
            }
        }
        // NumPy nonzero enumerates row then column in the original input order.
        picks.sort_unstable();
        picks.dedup();
        if picks.len() > max_pairs.saturating_sub(pairs.len()) {
            return Err(too_large());
        }
        for pair in picks {
            push(&mut pairs, pair)?;
        }
    }
    Ok(pairs)
}

fn median(values: &mut [f64], default: f64) -> f64 {
    if values.is_empty() {
        return default;
    }
    let lo = (values.len() - 1) / 2;
    let even = values.len().is_multiple_of(2);
    let (_, mid, hi) = values.select_nth_unstable_by(lo, f64::total_cmp);
    if even {
        (*mid + hi.iter().copied().fold(f64::INFINITY, f64::min)) * 0.5
    } else {
        *mid
    }
}

fn projection(q: &[[f64; 2]; 4], axis: Point64) -> [f64; 2] {
    let values = q.map(|p| dot(p, axis));
    [
        values.into_iter().fold(f64::INFINITY, f64::min),
        values.into_iter().fold(f64::NEG_INFINITY, f64::max),
    ]
}

struct Measure {
    q: [[f64; 2]; 4],
    center: Point64,
    axis: Point64,
    normal: Point64,
    long: f64,
    short: f64,
    area: f64,
    length: f64,
    thickness: f64,
    valid: bool,
}

impl Measure {
    fn new(quad: Quad) -> Self {
        let q = quad.map(|p| p.map(f64::from));
        let sides = edges(&q);
        let spans = sides.map(|p| dot(p, p).sqrt());
        let major = longest(&spans);
        let long = spans[major];
        let short = spans.into_iter().fold(f64::INFINITY, f64::min);
        let mut axis = sides[major].map(|v| v / long.max(1e-12));
        if axis[0] < -1e-10 || (axis[0].abs() <= 1e-10 && axis[1] < 0.0) {
            axis = axis.map(|v| -v);
        }
        let normal = [-axis[1], axis[0]];
        let [a, b] = projection(&q, axis);
        let [c, d] = projection(&q, normal);
        let area = twice_area(&q).abs();
        let center = std::array::from_fn(|axis| q.iter().map(|p| p[axis]).sum::<f64>() * 0.25);
        Self {
            q,
            center,
            axis,
            normal,
            long,
            short,
            area,
            length: (b - a).max(1e-6),
            thickness: (d - c).max(1e-6),
            valid: short > 1e-6 && area * 0.5 > 1e-6,
        }
    }

    fn scale_valid(&self) -> bool {
        self.short * self.short > 1e-6 && self.area > 1e-3
    }

    fn geometry(&self, other: &Self) -> [f32; 12] {
        let scale = (self.thickness * other.thickness).sqrt();
        let signed_log = |x: f64| x.signum() * x.abs().ln_1p();
        let delta = sub(other.center, self.center);
        let measure = |axis| {
            let [a, b] = projection(&self.q, axis);
            let [c, d] = projection(&other.q, axis);
            let overlap = b.min(d) - a.max(c);
            (
                signed_log(-overlap / scale),
                overlap.max(0.0) / (b - a).min(d - c).max(1e-6),
                signed_log((c - a) / scale),
                signed_log((d - b) / scale),
            )
        };
        let (along_gap, along_overlap, start, end) = measure(self.axis);
        let (across_gap, across_overlap, _, _) = measure(self.normal);
        let angle_dot = dot(self.axis, other.axis);
        [
            signed_log(dot(delta, self.axis) / scale),
            signed_log(dot(delta, self.normal) / scale),
            along_gap,
            across_gap,
            along_overlap,
            across_overlap,
            start,
            end,
            (other.length / self.length).ln(),
            (other.thickness / self.thickness).ln(),
            2.0 * angle_dot * angle_dot - 1.0,
            2.0 * angle_dot * cross(self.axis, other.axis),
        ]
        .map(|v| v as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::PixelFormat;

    fn quad(y: f32) -> Quad {
        [[40.0, y + 5.0], [40.0, y], [10.0, y], [10.0, y + 5.0]]
    }

    #[test]
    fn candidate_rows_are_directed_without_reordering_input_lines() {
        let pixels = vec![255; 60 * 40 * 3];
        let image = ImageView::new(&pixels, 60, 40, 0, PixelFormat::Bgr24).unwrap();
        let degenerate = [[0.0, 0.0]; 4];
        let page = prepare(
            image,
            &[quad(20.0), degenerate, quad(3.0)],
            ExecutionLimits::default(),
        )
        .unwrap();
        assert_eq!(page.pairs, [[2, 0]]);
        assert_eq!(page.edge_inputs.len(), 2 * 26);
        assert_eq!(page.edge_inputs[1], -page.edge_inputs[26 + 1]);
        assert!(page.edge_inputs[1] > 0.0);
        assert_eq!(&page.node_inputs[3..6], &[0.0; 3]);
        assert_eq!(
            super::super::decode::decode(&page, &[0.0], crate::onnx::BScalePartition::Evidence)
                .unwrap(),
            [vec![1], vec![2, 0]]
        );
    }

    #[test]
    fn no_candidates_skip_visual_features_and_bad_coordinates_fail() {
        let image = ImageView::new(&[0; 3], 1, 1, 0, PixelFormat::Bgr24).unwrap();
        let single = prepare(image, &[quad(50.0)], ExecutionLimits::default()).unwrap();
        assert!(single.pairs.is_empty());
        assert!(single.edge_inputs.is_empty());
        assert!(prepare(image, &[[[f32::NAN; 2]; 4]], ExecutionLimits::default()).is_err());
        assert!(
            prepare(
                image,
                &[quad(50.0)],
                ExecutionLimits {
                    max_activation_bytes: 1
                }
            )
            .is_err()
        );
    }
}
