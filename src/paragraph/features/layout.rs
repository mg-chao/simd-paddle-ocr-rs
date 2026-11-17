use super::{Quad, aligned_frame, dot32, edges, median, twice_area};

#[derive(Clone, Copy)]
struct Line {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
    width: f64,
    height: f64,
}

impl Line {
    fn new(bounds: [f32; 4]) -> Self {
        let [left, top, right, bottom] = bounds.map(f64::from);
        Self {
            left,
            top,
            right,
            bottom,
            width: (right - left).max(1e-6),
            height: (bottom - top).max(1e-6),
        }
    }

    fn relation(self, other: Self) -> Relation {
        let dx = (other.left + other.right) * 0.5 - (self.left + self.right) * 0.5;
        let dy = (other.top + other.bottom) * 0.5 - (self.top + self.bottom) * 0.5;
        let scale = (self.height + other.height) * 0.5;
        let gap = if dy >= 0.0 {
            other.top - self.bottom
        } else {
            other.bottom - self.top
        };
        let overlap = (self.right.min(other.right) - self.left.max(other.left)).max(0.0)
            / self.width.min(other.width);
        Relation {
            dx,
            dy,
            scale,
            gap: gap.abs(),
            overlap,
        }
    }
}

struct Relation {
    dx: f64,
    dy: f64,
    scale: f64,
    gap: f64,
    overlap: f64,
}

#[derive(Clone, Copy, Default)]
struct Context {
    nearest: Option<usize>,
    indent_left: f64,
    indent_right: f64,
    rank: f64,
    count: usize,
    asymmetry: f64,
}

fn lower_median(values: &mut [f64]) -> f64 {
    let middle = (values.len() - 1) / 2;
    *values.select_nth_unstable_by(middle, f64::total_cmp).1
}

fn branch(lines: &[Line], valid: &[bool]) -> (Vec<Context>, f64, f64) {
    let mut context = vec![Context::default(); lines.len()];
    let mut schedule = Vec::new();
    for (i, &line) in lines.iter().enumerate() {
        if !valid[i] {
            continue;
        }
        let mut best = None::<(bool, f64, f64, usize, f64)>;
        for (j, &other) in lines.iter().enumerate() {
            if i == j || !valid[j] {
                continue;
            }
            let rel = line.relation(other);
            let candidate = (rel.overlap <= 0.1, rel.dy.abs(), rel.dx.abs(), j, rel.gap);
            if best.is_none_or(|prev| candidate < prev) {
                best = Some(candidate);
            }
        }
        if let Some((_, _, _, j, gap)) = best {
            context[i].nearest = Some(j);
            if gap > 1e-6 {
                schedule.push(gap);
            }
        }
    }
    let mut heights: Vec<_> = lines
        .iter()
        .zip(valid)
        .filter_map(|(line, &valid)| valid.then_some(line.height))
        .collect();
    let typical = median(&mut heights, 1.0);
    let (center, spread) = if schedule.is_empty() {
        (typical, typical)
    } else {
        let center = lower_median(&mut schedule);
        schedule
            .iter_mut()
            .for_each(|gap| *gap = (*gap - center).abs());
        (center, lower_median(&mut schedule))
    };
    let spread = spread.max(0.25 * typical).max(1e-6);

    let mut lefts = Vec::new();
    let mut rights = Vec::new();
    for (i, &line) in lines.iter().enumerate() {
        if !valid[i] {
            continue;
        }
        lefts.clear();
        rights.clear();
        lefts.push(line.left);
        rights.push(line.right);
        let mut narrower = 0;
        let mut equal = 0;
        let mut side = [(f64::INFINITY, center); 2];
        for (j, &other) in lines.iter().enumerate() {
            if i == j || !valid[j] {
                continue;
            }
            let rel = line.relation(other);
            if rel.overlap < 0.15 || rel.dy.abs() > 7.0 * rel.scale {
                continue;
            }
            lefts.push(other.left);
            rights.push(other.right);
            narrower += usize::from(other.width < line.width);
            equal += usize::from(other.width == line.width);
            if rel.gap / rel.scale <= 1.75 && (rel.gap - center).abs() / spread <= 2.5 {
                context[i].count += 1;
            }
            if rel.dy != 0.0 {
                let k = usize::from(rel.dy > 0.0);
                if rel.dy.abs() < side[k].0 {
                    side[k] = (rel.dy.abs(), rel.gap);
                }
            }
        }
        let other_count = lefts.len() - 1;
        context[i].rank = if other_count > 0 {
            (narrower as f64 + 0.5 * equal as f64) / other_count as f64
        } else {
            0.5
        };
        context[i].indent_left = (line.left - lower_median(&mut lefts)) / line.height;
        context[i].indent_right = (line.right - lower_median(&mut rights)) / line.height;
        context[i].asymmetry = side[1].1 / line.height - side[0].1 / line.height;
    }
    (context, center, spread)
}

pub(super) fn pair_features(
    quads: &[Quad],
    vertical: &[bool],
    pairs: &[[usize; 2]],
) -> Vec<[f32; 8]> {
    let mut result = vec![[0.0; 8]; pairs.len()];
    for direction in [false, true] {
        let indices: Vec<_> = (0..quads.len())
            .filter(|&i| vertical[i] == direction)
            .collect();
        if !pairs.iter().any(|&[a, _]| vertical[a] == direction) {
            continue;
        }
        let valid: Vec<_> = indices
            .iter()
            .map(|&i| {
                twice_area(&quads[i]).abs() > 1e-3
                    && edges(&quads[i]).iter().all(|&p| dot32(p, p).sqrt() > 1e-6)
            })
            .collect();
        let lines: Vec<_> = aligned_frame(quads, &indices, &valid, direction)
            .into_iter()
            .map(Line::new)
            .collect();
        let (context, center, spread) = branch(&lines, &valid);
        let mut inverse = vec![0; quads.len()];
        for (i, &original) in indices.iter().enumerate() {
            inverse[original] = i;
        }
        for (&[a, b], out) in pairs.iter().zip(&mut result) {
            if vertical[a] != direction {
                continue;
            }
            let (i, j) = (inverse[a], inverse[b]);
            if !valid[i] || !valid[j] {
                continue;
            }
            let (a, b) = (context[i], context[j]);
            let gap = lines[i].relation(lines[j]).gap;
            *out = [
                ((gap - center).abs() / spread).ln_1p() as f32,
                u8::from(a.nearest == Some(j) && b.nearest == Some(i)) as f32,
                (a.indent_left - b.indent_left).abs().ln_1p() as f32,
                (a.indent_right - b.indent_right).abs().ln_1p() as f32,
                (a.rank - b.rank).abs() as f32,
                a.count.min(b.count) as f32 / a.count.max(b.count).max(1) as f32,
                (a.asymmetry - b.asymmetry).abs().ln_1p() as f32,
                1.0,
            ];
        }
    }
    result
}
