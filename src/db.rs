//! Canonical DB quad postprocessing. All request scratch is reusable.
use crate::{
    detector::{DetectionBox, DetectorOptions},
    image::{ImageError, ImageResult, push, resize_vec},
    preprocess::{DetSourceGeometry, valid_map_size},
};
#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Point {
    x: f32,
    y: f32,
}
impl Point {
    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}
#[derive(Clone, Copy)]
struct Run {
    y: u32,
    left: u32,
    right: u32,
}
#[derive(Default)]
pub(crate) struct Workspace {
    bitmap: Vec<u8>,
    pending: Vec<u8>,
    blocked: Vec<u8>,
    dilated: Vec<u8>,
    interior: Vec<u8>,
    interior_ready: bool,
    stack: Vec<Run>,
    boundary: Vec<Point>,
    hull: Vec<Point>,
    expanded: Vec<Point>,
    candidates: std::collections::VecDeque<(usize, Contour)>,
    row_min: Vec<i32>,
    row_max: Vec<i32>,
    pub(crate) boxes: Vec<DetectionBox>,
}
impl Workspace {
    pub(crate) fn retained_bytes(&self) -> usize {
        self.candidates.capacity() * size_of::<(usize, Contour)>()
            + self.bitmap.capacity()
            + self.pending.capacity()
            + self.blocked.capacity()
            + self.dilated.capacity()
            + self.interior.capacity()
            + self.stack.capacity() * 12
            + (self.boundary.capacity() + self.hull.capacity() + self.expanded.capacity()) * 8
            + (self.row_min.capacity() + self.row_max.capacity()) * 4
            + self.boxes.capacity() * std::mem::size_of::<DetectionBox>()
    }
}
pub(crate) fn run(
    prediction: &[f32],
    width: usize,
    height: usize,
    options: &DetectorOptions,
    source: DetSourceGeometry,
    s: &mut Workspace,
) -> ImageResult<()> {
    run_with_kernel(
        prediction,
        width,
        height,
        options,
        source,
        s,
        crate::pixel_kernels::PixelKernel::selected(),
    )
}
#[allow(clippy::too_many_arguments)]
fn run_with_kernel(
    prediction: &[f32],
    width: usize,
    height: usize,
    options: &DetectorOptions,
    source: DetSourceGeometry,
    s: &mut Workspace,
    kernel: crate::pixel_kernels::PixelKernel,
) -> ImageResult<()> {
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| ImageError::new("DB map size overflow"))?;
    if width == 0 || height == 0 || pixels != prediction.len() || !valid_map_size(width, height) {
        return Err(ImageError::new("invalid DB probability map size"));
    }
    resize_vec(&mut s.bitmap, pixels)?;
    s.interior_ready = false;
    let foreground = crate::pixel_kernels::binarize(
        kernel,
        prediction,
        &mut s.bitmap[..pixels],
        options.bitmap_threshold,
    )?;
    s.boxes.clear();
    if !foreground {
        return Ok(());
    }
    if options.use_dilation {
        resize_vec(&mut s.dilated, pixels)?;
        for (y, dest) in s.dilated[..pixels].chunks_exact_mut(width).enumerate() {
            let row = &s.bitmap[y * width..(y + 1) * width];
            let above = &s.bitmap[y.saturating_sub(1) * width..(y.saturating_sub(1) + 1) * width];
            dest[0] = row[0] | above[0];
            for x in 1..width {
                dest[x] = row[x] | row[x - 1] | above[x] | above[x - 1];
            }
        }
        std::mem::swap(&mut s.bitmap, &mut s.dilated);
    }
    if kernel != crate::pixel_kernels::PixelKernel::Scalar && width >= 3 && height >= 3 {
        resize_vec(&mut s.interior, pixels)?;
        crate::pixel_kernels::interior(
            kernel,
            &s.bitmap[..pixels],
            width,
            height,
            &mut s.interior[..pixels],
        );
        s.interior_ready = true;
    }
    resize_vec(&mut s.pending, pixels)?;
    resize_vec(&mut s.blocked, pixels)?;
    s.pending[..pixels].copy_from_slice(&s.bitmap[..pixels]);
    s.blocked[..pixels].copy_from_slice(&s.bitmap[..pixels]);
    // Exclude exterior background. Each remaining four-connected background
    // component supplies a hole contour in the same raster scan as the outer
    // borders of eight-connected foreground components.
    for x in 0..width {
        fill_background(x, width, height, pixels, false, s)?;
        if height > 1 {
            fill_background((height - 1) * width + x, width, height, pixels, false, s)?;
        }
    }
    for y in 1..height.saturating_sub(1) {
        fill_background(y * width, width, height, pixels, false, s)?;
        if width > 1 {
            fill_background(y * width + width - 1, width, height, pixels, false, s)?;
        }
    }
    let mut foreground = next(&s.pending, 0, pixels, true);
    let mut hole = next(&s.blocked, 0, pixels, false);
    s.candidates.clear();
    while foreground < pixels || hole < pixels {
        let candidate = if foreground < hole {
            let seed = foreground;
            fill_foreground(seed, width, height, 0, s)?;
            foreground = next(&s.pending, seed + 1, pixels, true);
            (seed, Contour::Outer)
        } else {
            let seed = hole;
            fill_background(seed, width, height, pixels, false, s)?;
            hole = next(&s.blocked, seed + 1, pixels, false);
            (seed, Contour::Hole)
        };
        // RETR_LIST enumerates in reverse discovery order. Keep even the
        // seeds of contours that may fail size or confidence checks.
        if s.candidates.len() == options.max_candidates {
            s.candidates.pop_front();
        }
        s.candidates
            .try_reserve(1)
            .map_err(|e| ImageError::new(e.to_string()))?;
        s.candidates.push_back(candidate);
    }
    // Components of the same kind are disjoint. A retained hole cannot
    // reach exterior background, so only the selected seeds need replay.
    s.pending[..pixels].copy_from_slice(&s.bitmap[..pixels]);
    s.blocked[..pixels].copy_from_slice(&s.bitmap[..pixels]);
    while let Some((seed, kind)) = s.candidates.pop_back() {
        match kind {
            Contour::Outer => fill_foreground(seed, width, height, pixels, s)?,
            Contour::Hole => fill_background(seed, width, height, pixels, true, s)?,
            Contour::Offset => unreachable!(),
        }
        if s.boundary.len() >= 3
            && let Some(b) = build(prediction, width, height, source, options, kind, s)?
        {
            push(&mut s.boxes, b)?;
        }
    }
    sort_reading_order(&mut s.boxes);
    Ok(())
}
fn next(a: &[u8], from: usize, to: usize, nonzero: bool) -> usize {
    from + crate::pixel_kernels::scan(
        crate::pixel_kernels::PixelKernel::selected(),
        &a[from..to],
        nonzero,
        false,
    )
}
fn run_start(a: &[u8], lower: usize, end: usize, nonzero: bool) -> usize {
    let opposite = crate::pixel_kernels::scan(
        crate::pixel_kernels::PixelKernel::selected(),
        &a[lower..end],
        !nonzero,
        true,
    );
    if opposite == end - lower {
        lower
    } else {
        lower + opposite + 1
    }
}
fn boundary(s: &mut Workspace, x: usize, y: usize, cap: usize) -> ImageResult<()> {
    if s.boundary.len() < cap {
        push(&mut s.boundary, Point::new(x as f32, y as f32))?;
    }
    Ok(())
}
fn is_boundary(bitmap: &[u8], w: usize, h: usize, x: usize, y: usize) -> bool {
    if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
        return true;
    }
    for dy in -1..=1 {
        for dx in -1..=1 {
            if (dx != 0 || dy != 0)
                && bitmap[(y as isize + dy) as usize * w + (x as isize + dx) as usize] == 0
            {
                return true;
            }
        }
    }
    false
}
fn fill_foreground(
    start: usize,
    w: usize,
    h: usize,
    cap: usize,
    s: &mut Workspace,
) -> ImageResult<()> {
    s.boundary.clear();
    s.stack.clear();
    let y = start / w;
    let row = y * w;
    let left = run_start(&s.pending, row, start, true) - row;
    let right = next(&s.pending, start + 1, row + w, false) - row - 1;
    s.pending[row + left..=row + right].fill(0);
    push(
        &mut s.stack,
        Run {
            y: y as u32,
            left: left as u32,
            right: right as u32,
        },
    )?;
    while let Some(run) = s.stack.pop() {
        let y = run.y as usize;
        let left = run.left as usize;
        let right = run.right as usize;
        if cap != 0 {
            for x in left..=right {
                if if s.interior_ready {
                    s.interior[y * w + x] == 0
                } else {
                    is_boundary(&s.bitmap, w, h, x, y)
                } {
                    boundary(s, x, y, cap)?;
                }
            }
        }
        let from = left.saturating_sub(1);
        let to = (right + 1).min(w - 1);
        for direction in [-1isize, 1] {
            let ny = y as isize + direction;
            if ny < 0 || ny >= h as isize {
                continue;
            }
            let row = ny as usize * w;
            let mut x = from;
            while x <= to {
                x = next(&s.pending, row + x, row + to + 1, true) - row;
                if x > to {
                    break;
                }
                let l = run_start(&s.pending, row, row + x, true) - row;
                let r = next(&s.pending, row + x + 1, row + w, false) - row - 1;
                s.pending[row + l..=row + r].fill(0);
                push(
                    &mut s.stack,
                    Run {
                        y: ny as u32,
                        left: l as u32,
                        right: r as u32,
                    },
                )?;
                x = r + 2;
            }
        }
    }
    Ok(())
}
fn fill_background(
    seed: usize,
    w: usize,
    h: usize,
    cap: usize,
    collect: bool,
    s: &mut Workspace,
) -> ImageResult<()> {
    if collect {
        s.boundary.clear();
    }
    if s.blocked[seed] != 0 {
        return Ok(());
    }
    s.stack.clear();
    let y = seed / w;
    let row = y * w;
    let left = run_start(&s.blocked, row, seed, false) - row;
    let right = next(&s.blocked, seed + 1, row + w, true) - row - 1;
    s.blocked[row + left..=row + right].fill(1);
    push(
        &mut s.stack,
        Run {
            y: y as u32,
            left: left as u32,
            right: right as u32,
        },
    )?;
    while let Some(run) = s.stack.pop() {
        let y = run.y as usize;
        let left = run.left as usize;
        let right = run.right as usize;
        if collect {
            let row = y * w;
            if left > 0 && s.bitmap[row + left - 1] != 0 {
                boundary(s, left - 1, y, cap)?;
            }
            if right + 1 < w && s.bitmap[row + right + 1] != 0 {
                boundary(s, right + 1, y, cap)?;
            }
        }
        for direction in [-1isize, 1] {
            let ny = y as isize + direction;
            if ny < 0 || ny >= h as isize {
                continue;
            }
            let row = ny as usize * w;
            let mut x = left;
            while x <= right {
                if !collect {
                    x = next(&s.blocked, row + x, row + right + 1, false) - row;
                    if x > right {
                        break;
                    }
                } else if s.bitmap[row + x] != 0 {
                    // A hole contour follows the foreground pixels facing the hole.
                    boundary(s, x, ny as usize, cap)?;
                    x += 1;
                    continue;
                } else if s.blocked[row + x] != 0 {
                    x += 1;
                    continue;
                }
                let l = run_start(&s.blocked, row, row + x, false) - row;
                let r = next(&s.blocked, row + x + 1, row + w, true) - row - 1;
                s.blocked[row + l..=row + r].fill(1);
                push(
                    &mut s.stack,
                    Run {
                        y: ny as u32,
                        left: l as u32,
                        right: r as u32,
                    },
                )?;
                x = r + 1;
            }
        }
    }
    Ok(())
}
fn row_extremes(s: &mut Workspace, height: usize) -> ImageResult<usize> {
    resize_vec(&mut s.row_min, height)?;
    resize_vec(&mut s.row_max, height)?;
    let min = s.boundary.iter().map(|p| p.y as usize).min().unwrap();
    let max = s.boundary.iter().map(|p| p.y as usize).max().unwrap();
    s.row_min[min..=max].fill(i32::MAX);
    s.row_max[min..=max].fill(i32::MIN);
    for p in &s.boundary {
        let y = p.y as usize;
        let x = p.x as i32;
        s.row_min[y] = s.row_min[y].min(x);
        s.row_max[y] = s.row_max[y].max(x);
    }
    let mut count = 0;
    for y in min..=max {
        if s.row_min[y] > s.row_max[y] {
            continue;
        }
        s.boundary[count] = Point::new(s.row_min[y] as f32, y as f32);
        count += 1;
        if s.row_min[y] != s.row_max[y] {
            s.boundary[count] = Point::new(s.row_max[y] as f32, y as f32);
            count += 1;
        }
    }
    Ok(count)
}
fn cross(o: Point, a: Point, b: Point) -> f32 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}
fn distance(a: Point, b: Point) -> f32 {
    ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)).sqrt()
}
fn hull(points: &mut [Point], scratch: &mut Vec<Point>) -> ImageResult<usize> {
    resize_vec(
        scratch,
        points
            .len()
            .checked_mul(2)
            .ok_or_else(|| ImageError::new("hull size overflow"))?,
    )?;
    points.sort_unstable_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap()
            .then_with(|| a.y.partial_cmp(&b.y).unwrap())
    });
    let mut unique = 0;
    for i in 0..points.len() {
        if unique == 0 || points[i] != points[unique - 1] {
            points[unique] = points[i];
            unique += 1;
        }
    }
    if unique < 3 {
        return Ok(0);
    }
    let mut count = 0;
    for &p in &points[..unique] {
        while count >= 2 && cross(scratch[count - 2], scratch[count - 1], p) <= 0. {
            count -= 1;
        }
        scratch[count] = p;
        count += 1;
    }
    let lower = count;
    for i in (0..unique - 1).rev() {
        while count > lower && cross(scratch[count - 2], scratch[count - 1], points[i]) <= 0. {
            count -= 1;
        }
        scratch[count] = points[i];
        count += 1;
    }
    Ok(count.saturating_sub(1))
}
#[derive(Clone, Copy)]
enum Contour {
    Outer,
    Hole,
    Offset,
}

fn rectangle(
    kind: Contour,
    points: &mut [Point],
    scratch: &mut Vec<Point>,
) -> ImageResult<Option<(f32, [Point; 4])>> {
    let n = hull(points, scratch)?;
    if n < 3 {
        return Ok(None);
    }
    // OpenCV preserves the cyclic contour order in the hull. At tied areas
    // the last caliper direction wins, affecting float-to-integer offsets.
    let hull = &mut scratch[..n];
    let top = hull
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.y.total_cmp(&b.y).then_with(|| match kind {
                Contour::Offset => b.x.total_cmp(&a.x),
                _ => a.x.total_cmp(&b.x),
            })
        })
        .unwrap()
        .0;
    let start = match kind {
        Contour::Outer | Contour::Offset => (top + 1) % n,
        Contour::Hole => (top + n - 1) % n,
    };
    hull.rotate_left(start);
    Ok(Some(calipers::bounding_box(hull)))
}

mod calipers;

fn order_clockwise(points: &mut [Point; 4]) {
    points.sort_by(|a, b| a.x.total_cmp(&b.x));
    let [mut tl, mut bl, mut tr, mut br] = *points;
    if tl.y > bl.y {
        std::mem::swap(&mut tl, &mut bl);
    }
    if tr.y > br.y {
        std::mem::swap(&mut tr, &mut br);
    }
    *points = [tl, tr, br, bl];
}
fn build(
    prediction: &[f32],
    w: usize,
    h: usize,
    source: DetSourceGeometry,
    o: &DetectorOptions,
    kind: Contour,
    s: &mut Workspace,
) -> ImageResult<Option<DetectionBox>> {
    let count = row_extremes(s, h)?;
    let Some((short_side, mini)) = rectangle(kind, &mut s.boundary[..count], &mut s.hull)? else {
        return Ok(None);
    };
    if short_side < 3. {
        return Ok(None);
    }
    let score = polygon_score(prediction, w, h, &mini, &mut s.row_min, &mut s.row_max)?;
    if !score.is_finite() || score < o.box_threshold.unwrap() {
        return Ok(None);
    }
    unclip(&mini, o.offset_ratio(), &mut s.expanded)?;
    let Some((short_side, mut mapped)) = rectangle(Contour::Offset, &mut s.expanded, &mut s.hull)?
    else {
        return Ok(None);
    };
    if short_side < 5. {
        return Ok(None);
    }
    for p in &mut mapped {
        [p.x, p.y] = source.map_point([p.x, p.y], w, h);
    }
    let mut corners = mapped;
    order_clockwise(&mut corners);
    if distance(corners[0], corners[1]) < 4. || distance(corners[0], corners[3]) < 4. {
        return Ok(None);
    }
    Ok(Some(DetectionBox {
        points: corners.map(|p| [p.x, p.y]),
        score,
    }))
}
/// The fast DB score uses the filled, integer-rasterized minimum-area quad.
/// A convex polygon intersects each scanline in one run, including the outer
/// eight-connected boundary drawn by fillPoly.
fn polygon_score(
    prediction: &[f32],
    w: usize,
    h: usize,
    p: &[Point; 4],
    lo: &mut Vec<i32>,
    hi: &mut Vec<i32>,
) -> ImageResult<f32> {
    let left = p
        .iter()
        .map(|p| p.x.floor() as i32)
        .min()
        .unwrap()
        .clamp(0, w as i32 - 1);
    let top = p
        .iter()
        .map(|p| p.y.floor() as i32)
        .min()
        .unwrap()
        .clamp(0, h as i32 - 1);
    let bottom = p
        .iter()
        .map(|p| p.y.ceil() as i32)
        .max()
        .unwrap()
        .clamp(0, h as i32 - 1);
    let rows = (bottom - top + 1) as usize;
    resize_vec(lo, rows)?;
    resize_vec(hi, rows)?;
    lo[..rows].fill(i32::MAX);
    hi[..rows].fill(i32::MIN);
    let pts = p.map(|p| [(p.x - left as f32) as i32 + left, (p.y - top as f32) as i32]);
    for i in 0..4 {
        let (mut a, mut b) = (pts[i], pts[(i + 1) % 4]);
        if a[0] > b[0] {
            std::mem::swap(&mut a, &mut b);
        }
        let dx = b[0] - a[0];
        let dy = (b[1] - a[1]).abs();
        let dir = (b[1] - a[1]).signum();
        let steep = dy > dx;
        let (major, minor) = if steep { (dy, dx) } else { (dx, dy) };
        let mut error = major - 2 * minor;
        for _ in 0..=major {
            if let Ok(y) = usize::try_from(a[1])
                && y < rows
            {
                lo[y] = lo[y].min(a[0]);
                hi[y] = hi[y].max(a[0]);
            }
            let diagonal = error < 0;
            if diagonal {
                error += 2 * major;
            }
            error -= 2 * minor;
            if !steep || diagonal {
                a[0] += 1;
            }
            if steep || diagonal {
                a[1] += dir;
            }
        }
    }
    let mut sum = 0.;
    let mut count = 0;
    for y in 0..rows {
        let (x0, x1) = (lo[y].max(0), hi[y].min(w as i32 - 1));
        if x0 <= x1 {
            let row = (y + top as usize) * w;
            let run = &prediction[row + x0 as usize..=row + x1 as usize];
            sum += run.iter().map(|&p| f64::from(p)).sum::<f64>();
            count += run.len();
        }
    }
    Ok(if count == 0 {
        0.
    } else {
        (sum / count as f64) as f32
    })
}
fn rounded(out: &mut Vec<Point>, x: f64, y: f64) -> ImageResult<()> {
    let round = |v: f64| {
        if v < 0. {
            (v - 0.5) as i64 as f32
        } else {
            (v + 0.5) as i64 as f32
        }
    };
    push(out, Point::new(round(x), round(y)))
}
fn unclip(p: &[Point; 4], ratio: f64, out: &mut Vec<Point>) -> ImageResult<()> {
    out.clear();
    let (mut area, mut perimeter) = (0f64, 0f64);
    for i in 0..4 {
        let a = p[i];
        let b = p[(i + 1) % 4];
        area += a.x as f64 * b.y as f64 - b.x as f64 * a.y as f64;
        perimeter += (f64::from(b.x) - f64::from(a.x)).hypot(f64::from(b.y) - f64::from(a.y));
    }
    let delta = (area * 0.5).abs() * ratio / perimeter;
    if !delta.is_finite() || delta <= 0. {
        return Ok(());
    }
    let mut pts = [[0i64; 2]; 4];
    let mut n = 0;
    for q in p {
        let q = [q.x as i64, q.y as i64];
        if n > 0 && pts[n - 1] == q {
            continue;
        }
        pts[n] = q;
        n += 1;
    }
    if n >= 2 && pts[0] == pts[n - 1] {
        n -= 1;
    }
    if n < 3 {
        return Ok(());
    }
    let mut winding = 0.;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        winding += a[0] as f64 * b[1] as f64 - b[0] as f64 * a[1] as f64;
    }
    if winding < 0. {
        pts[..n].reverse();
    }
    let mut normals = [[0f64; 2]; 4];
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        let (mut dx, mut dy) = ((b[0] - a[0]) as f64, (b[1] - a[1]) as f64);
        let length = (dx * dx + dy * dy).sqrt();
        if length > 0. {
            dx /= length;
            dy /= length;
        }
        normals[i] = [dy, -dx];
    }
    // Round joins advance by a fixed angular step for the entire offset path.
    let tolerance = 0.25_f64.min(delta * 0.25);
    let steps =
        (std::f64::consts::PI / (1. - tolerance / delta).acos()).min(delta * std::f64::consts::PI);
    let angle_step = std::f64::consts::TAU / steps;
    let (sin, cos) = angle_step.sin_cos();
    let mut prev = n - 1;
    for i in 0..n {
        let [x, y] = pts[i].map(|v| v as f64);
        let [mut nx, mut ny] = normals[prev];
        let [ex, ey] = normals[i];
        let cross = nx * ey - ny * ex;
        let dot = nx * ex + ny * ey;
        if (cross * delta).abs() < 1. && dot > 0. {
            rounded(out, x + delta * nx, y + delta * ny)?;
            continue;
        }
        if cross < 0. {
            rounded(out, x + delta * nx, y + delta * ny)?;
            rounded(out, x, y)?;
        } else {
            let count = (cross.atan2(dot).abs() / angle_step).round().max(1.) as usize;
            for _ in 0..count {
                rounded(out, x + delta * nx, y + delta * ny)?;
                (nx, ny) = (nx * cos - ny * sin, nx * sin + ny * cos);
            }
        }
        rounded(out, x + delta * ex, y + delta * ey)?;
        prev = i;
    }
    Ok(())
}
pub(crate) fn sort_reading_order(boxes: &mut [DetectionBox]) {
    boxes.sort_by(|a, b| a.points[0][1].partial_cmp(&b.points[0][1]).unwrap());
    // RapidOCR groups by adjacent y differences before sorting each row by x.
    for row in boxes.chunk_by_mut(|a, b| b.points[0][1] - a.points[0][1] < 10.) {
        row.sort_by(|a, b| a.points[0][0].partial_cmp(&b.points[0][0]).unwrap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calipers_keep_integer_extents_before_clipper_conversion() {
        let mut hull = [
            [1431, 408],
            [1432, 409],
            [1432, 422],
            [1431, 425],
            [1430, 426],
            [1388, 428],
            [1386, 428],
            [1378, 427],
            [1377, 426],
            [1376, 420],
            [1376, 408],
        ]
        .map(|[x, y]| Point::new(x as f32, y as f32));
        let (_, points) = rectangle(Contour::Outer, &mut hull, &mut Vec::new())
            .unwrap()
            .unwrap();
        assert_eq!(
            points.map(|p| [p.x, p.y]),
            [[1376., 408.], [1432., 408.], [1432., 428.], [1376., 428.]]
        );
        let mut expanded = Vec::new();
        unclip(&points, 1.6, &mut expanded).unwrap();
        let (_, points) = rectangle(Contour::Offset, &mut expanded, &mut Vec::new())
            .unwrap()
            .unwrap();
        assert_eq!(
            points.map(|p| [p.x, p.y]),
            [[1364., 396.], [1444., 396.], [1444., 440.], [1364., 440.]]
        );
    }

    #[test]
    fn default_offset_preserves_half_pixel_rounding() {
        let points = [[906., 1091.], [928., 1091.], [928., 1101.], [906., 1101.]]
            .map(|[x, y]| Point::new(x, y));
        let mut expanded = Vec::new();
        unclip(
            &points,
            DetectorOptions::default().offset_ratio(),
            &mut expanded,
        )
        .unwrap();
        let (_, quad) = rectangle(Contour::Offset, &mut expanded, &mut Vec::new())
            .unwrap()
            .unwrap();
        assert_eq!(
            quad.map(|p| [p.x, p.y]),
            [[901., 1086.], [934., 1086.], [934., 1107.], [901., 1107.]]
        );
    }

    #[test]
    fn slanted_quad_score_includes_integer_edge_pixels() {
        let points = [[1., 2.], [7., 0.], [8., 4.], [2., 6.]].map(|[x, y]| Point::new(x, y));
        let map: Vec<_> = (0..63).map(|i| (i * i) as f32).collect();
        let score = polygon_score(&map, 9, 7, &points, &mut Vec::new(), &mut Vec::new()).unwrap();
        assert_eq!(score, 1190.1351351351352_f64 as f32);
    }

    #[test]
    fn chained_rows_sort_by_x_and_preserve_equal_coordinates() {
        let make = |x, y, score| DetectionBox {
            points: [[x, y]; 4],
            score,
        };
        let mut boxes = [
            make(9., 1., 0.),
            make(0., 6., 1.),
            make(-0., 12., 2.),
            make(1., 22., 3.),
        ];
        sort_reading_order(&mut boxes);
        assert_eq!(boxes.map(|b| b.score), [1., 2., 0., 3.]);
    }

    #[test]
    fn default_unclip_preserves_recognition_mask_and_expands_text_coverage() {
        let (width, height) = (160, 64);
        let mut prediction = vec![0.; width * height];
        for y in 26..37 {
            prediction[y * width + 20..y * width + 141].fill(0.9);
        }
        let source = DetSourceGeometry::new(width, height).unwrap();
        let options = DetectorOptions {
            box_threshold: Some(0.45),
            ..Default::default()
        };
        let mut actual = Workspace::default();
        run(&prediction, width, height, &options, source, &mut actual).unwrap();
        let mut previous = Workspace::default();
        run(
            &prediction,
            width,
            height,
            &DetectorOptions {
                unclip_ratio: 1.4,
                ..options
            },
            source,
            &mut previous,
        )
        .unwrap();
        assert_eq!(options.unclip_ratio, 1.6);
        assert_eq!(actual.boxes.len(), 1);
        assert_eq!(previous.boxes.len(), 1);
        assert_eq!(actual.bitmap, previous.bitmap);
        let a = actual.boxes[0];
        let b = previous.boxes[0];
        assert_eq!(a.score, b.score);
        assert!(a.points[0][1] < b.points[0][1]);
        assert!(a.points[3][1] > b.points[3][1]);
    }

    #[test]
    fn candidate_limit_counts_rejected_contours_in_reverse_scan_order() {
        let (w, h) = (40, 48);
        let mut map = vec![0.; w * h];
        for top in [3, 25] {
            for y in top..top + 8 {
                map[y * w + 5..y * w + 18].fill(0.9);
            }
        }
        map[45 * w + 35] = 0.9;
        let source = DetSourceGeometry::new(w, h).unwrap();
        let mut workspace = Workspace::default();
        let mut options = DetectorOptions {
            max_candidates: 1,
            box_threshold: Some(0.5),
            ..Default::default()
        };
        run(&map, w, h, &options, source, &mut workspace).unwrap();
        assert!(workspace.boxes.is_empty());
        options.max_candidates = 2;
        run(&map, w, h, &options, source, &mut workspace).unwrap();
        assert_eq!(workspace.boxes.len(), 1);
        assert!(workspace.boxes[0].points[0][1] > 15.);
    }

    #[test]
    fn bad_maps_are_errors_and_reused_workspace_can_recover() {
        let mut s = Workspace::default();
        let o = DetectorOptions {
            box_threshold: Some(0.4),
            ..Default::default()
        };
        let source = DetSourceGeometry::new(2, 2).unwrap();
        assert!(run(&[f32::NAN], 1, 1, &o, source, &mut s).is_err());
        assert!(run(&[0.; 4], 1, 1, &o, source, &mut s).is_err());
        run(&[0.1; 4], 2, 2, &o, source, &mut s).unwrap();
        assert!(s.boxes.is_empty());
    }
    #[test]
    fn cached_interior_preserves_contours_holes_dilation_thresholds_and_order() {
        let (width, height) = (65, 39);
        let mut prediction = vec![0.3; width * height];
        for y in 3..19 {
            for x in 4..31 {
                prediction[y * width + x] = 0.9;
            }
        }
        for y in 8..14 {
            for x in 11..23 {
                prediction[y * width + x] = 0.1;
            }
        }
        for y in 25..36 {
            for x in 37..62 {
                prediction[y * width + x] = 0.8;
            }
        }
        for i in 0..8 {
            prediction[(21 + i) * width + 3 + i] = f32::from_bits(0.3f32.to_bits() + 1);
        }
        for use_dilation in [false, true] {
            let options = DetectorOptions {
                bitmap_threshold: 0.3,
                box_threshold: Some(0.4),
                use_dilation,
                ..Default::default()
            };
            let mut original = Workspace::default();
            let source = DetSourceGeometry::new(130, 78).unwrap();
            run_with_kernel(
                &prediction,
                width,
                height,
                &options,
                source,
                &mut original,
                crate::pixel_kernels::PixelKernel::Scalar,
            )
            .unwrap();
            assert!(!original.boxes.is_empty());
            assert!(!original.interior_ready);
            let expected = serde_json::to_value(&original.boxes).unwrap();
            for kernel in crate::pixel_kernels::PixelKernel::test_kernels() {
                let mut actual = Workspace::default();
                run_with_kernel(
                    &prediction,
                    width,
                    height,
                    &options,
                    source,
                    &mut actual,
                    kernel,
                )
                .unwrap();
                assert_eq!(
                    serde_json::to_value(&actual.boxes).unwrap(),
                    expected,
                    "{kernel:?}/{use_dilation}"
                );
                let retained = actual.retained_bytes();
                run_with_kernel(
                    &prediction,
                    width,
                    height,
                    &options,
                    source,
                    &mut actual,
                    kernel,
                )
                .unwrap();
                assert_eq!(actual.retained_bytes(), retained);
                let mut invalid = prediction.clone();
                invalid[width * height - 1] = f32::NAN;
                assert!(
                    run_with_kernel(
                        &invalid,
                        width,
                        height,
                        &options,
                        source,
                        &mut actual,
                        kernel
                    )
                    .is_err()
                );
                run_with_kernel(
                    &prediction,
                    width,
                    height,
                    &options,
                    source,
                    &mut actual,
                    kernel,
                )
                .unwrap();
                assert_eq!(serde_json::to_value(&actual.boxes).unwrap(), expected);
                run_with_kernel(&[0.; 6], 2, 3, &options, source, &mut actual, kernel).unwrap();
                assert!(actual.boxes.is_empty());
                assert!(!actual.interior_ready);
            }
        }
    }
}
