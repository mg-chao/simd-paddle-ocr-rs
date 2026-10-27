//! Canonical DB quad postprocessing. All request scratch is reusable.
use crate::{
    detector::{DetectionBox, DetectorOptions},
    image::{ImageError, ImageResult, push, resize_vec},
    preprocess::{DET_MAX_SIDE, DetSourceGeometry},
};
use std::cmp::Ordering;
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
#[derive(Clone, Copy, Default)]
struct Rectangle {
    ux: f32,
    uy: f32,
    vx: f32,
    vy: f32,
    min_u: f32,
    max_u: f32,
    min_v: f32,
    max_v: f32,
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
    row_min: Vec<i32>,
    row_max: Vec<i32>,
    pub(crate) boxes: Vec<DetectionBox>,
}
impl Workspace {
    pub(crate) fn retained_bytes(&self) -> usize {
        self.bitmap.capacity()
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
    if width == 0
        || height == 0
        || pixels != prediction.len()
        || width > DET_MAX_SIDE
        || height > DET_MAX_SIDE
    {
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
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                s.dilated[i] = u8::from(
                    s.bitmap[i] != 0
                        || (x + 1 < width && s.bitmap[i + 1] != 0)
                        || (y + 1 < height && s.bitmap[i + width] != 0)
                        || (x + 1 < width && y + 1 < height && s.bitmap[i + width + 1] != 0),
                );
            }
        }
        s.bitmap[..pixels].copy_from_slice(&s.dilated[..pixels]);
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
    let mut candidates = 0;
    let mut start = 0;
    while start < pixels && candidates < options.max_candidates {
        start = next(&s.pending, start, pixels, true);
        if start >= pixels {
            break;
        }
        fill_foreground(start, width, height, pixels, s)?;
        if s.boundary.len() >= 3 {
            candidates += 1;
            if let Some(b) = build(prediction, width, height, source, options, s)? {
                push(&mut s.boxes, b)?;
            }
        }
        start += 1;
    }
    if candidates < options.max_candidates {
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
        start = 0;
        while start < pixels && candidates < options.max_candidates {
            start = next(&s.blocked, start, pixels, false);
            if start >= pixels {
                break;
            }
            fill_background(start, width, height, pixels, true, s)?;
            if s.boundary.len() >= 3 {
                candidates += 1;
                if let Some(b) = build(prediction, width, height, source, options, s)? {
                    push(&mut s.boxes, b)?;
                }
            }
            start += 1;
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
        for x in left..=right {
            if if s.interior_ready {
                s.interior[y * w + x] == 0
            } else {
                is_boundary(&s.bitmap, w, h, x, y)
            } {
                boundary(s, x, y, cap)?;
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
                boundary(s, left, y, cap)?;
            }
            if right + 1 < w && s.bitmap[row + right + 1] != 0 {
                boundary(s, right, y, cap)?;
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
                    // Emit the background pixel's coordinate, not its foreground neighbour.
                    boundary(s, x, y, cap)?;
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
fn rectangle(
    points: &mut [Point],
    scratch: &mut Vec<Point>,
) -> ImageResult<Option<(Rectangle, [Point; 4])>> {
    let n = hull(points, scratch)?;
    if n < 3 {
        return Ok(None);
    }
    let mut best = f32::INFINITY;
    let mut r = Rectangle::default();
    for edge in 0..n {
        let a = scratch[edge];
        let b = scratch[(edge + 1) % n];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let length = (dx * dx + dy * dy).sqrt();
        if length <= 0. {
            continue;
        }
        let ux = dx / length;
        let uy = dy / length;
        let vx = -uy;
        let vy = ux;
        let (mut min_u, mut max_u, mut min_v, mut max_v) = (
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        );
        for p in &scratch[..n] {
            let u = p.x * ux + p.y * uy;
            let v = p.x * vx + p.y * vy;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let area = (max_u - min_u) * (max_v - min_v);
        if area < best {
            best = area;
            r = Rectangle {
                ux,
                uy,
                vx,
                vy,
                min_u,
                max_u,
                min_v,
                max_v,
            };
        }
    }
    if !best.is_finite() || best <= 0. {
        return Ok(None);
    }
    let mut p = rectangle_points(r);
    order_clockwise(&mut p);
    Ok(Some((r, p)))
}
fn rectangle_points(r: Rectangle) -> [Point; 4] {
    let from = |u: f32, v: f32| Point::new(u * r.ux + v * r.vx, u * r.uy + v * r.vy);
    [
        from(r.min_u, r.min_v),
        from(r.max_u, r.min_v),
        from(r.max_u, r.max_v),
        from(r.min_u, r.max_v),
    ]
}
fn order_clockwise(p: &mut [Point; 4]) {
    let input = *p;
    let (mut mini, mut maxi) = (0, 0);
    let (mut min, mut max) = (input[0].x + input[0].y, input[0].x + input[0].y);
    for (i, q) in input.iter().enumerate().skip(1) {
        let sum = q.x + q.y;
        if sum < min {
            min = sum;
            mini = i;
        }
        if sum > max {
            max = sum;
            maxi = i;
        }
    }
    let mut remain = [0; 2];
    let mut n = 0;
    for i in 0..4 {
        if i != mini && i != maxi && n < 2 {
            remain[n] = i;
            n += 1;
        }
    }
    if n == 2 && mini != maxi {
        let a = input[remain[0]];
        let b = input[remain[1]];
        let (tr, bl) = if a.y - a.x < b.y - b.x {
            (a, b)
        } else {
            (b, a)
        };
        *p = [input[mini], tr, input[maxi], bl];
        if (0..4).all(|i| cross(p[i], p[(i + 1) & 3], p[(i + 2) & 3]) > 0.) {
            return;
        }
    }
    let (mut cx, mut cy) = (0., 0.);
    for q in input {
        cx += q.x;
        cy += q.y;
    }
    cx /= 4.;
    cy /= 4.;
    let angle = input.map(|q| (q.y - cy).atan2(q.x - cx));
    let mut order = [0, 1, 2, 3];
    for i in 1..4 {
        let k = order[i];
        let mut j = i;
        while j > 0 && angle[order[j - 1]] > angle[k] {
            order[j] = order[j - 1];
            j -= 1;
        }
        order[j] = k;
    }
    let mut start = 0;
    for i in 1..4 {
        let a = input[order[i]];
        let b = input[order[start]];
        let sa = a.x + a.y;
        let sb = b.x + b.y;
        if sa < sb || (sa == sb && (a.y < b.y || (a.y == b.y && a.x < b.x))) {
            start = i;
        }
    }
    for i in 0..4 {
        p[i] = input[order[(start + i) & 3]];
    }
}
fn build(
    prediction: &[f32],
    w: usize,
    h: usize,
    source: DetSourceGeometry,
    o: &DetectorOptions,
    s: &mut Workspace,
) -> ImageResult<Option<DetectionBox>> {
    let count = row_extremes(s, h)?;
    let Some((r, mini)) = rectangle(&mut s.boundary[..count], &mut s.hull)? else {
        return Ok(None);
    };
    if (r.max_u - r.min_u).min(r.max_v - r.min_v) < 3. {
        return Ok(None);
    }
    let score = polygon_score(prediction, w, h, &mini);
    if !score.is_finite() || score < o.box_threshold.unwrap() {
        return Ok(None);
    }
    unclip(&mini, o.unclip_ratio, &mut s.expanded)?;
    let Some((r, _)) = rectangle(&mut s.expanded, &mut s.hull)? else {
        return Ok(None);
    };
    if (r.max_u - r.min_u).min(r.max_v - r.min_v) < 5. {
        return Ok(None);
    }
    let mut mapped = rectangle_points(r);
    order_clockwise(&mut mapped);
    for p in &mut mapped {
        [p.x, p.y] = source.map_point([p.x, p.y], w, h);
    }
    let Some((_, corners)) = rectangle(&mut mapped, &mut s.hull)? else {
        return Ok(None);
    };
    if distance(corners[0], corners[1]) <= 4. || distance(corners[0], corners[3]) <= 4. {
        return Ok(None);
    }
    Ok(Some(DetectionBox {
        points: corners.map(|p| [p.x, p.y]),
        score,
    }))
}
fn polygon_score(prediction: &[f32], w: usize, h: usize, p: &[Point; 4]) -> f32 {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for q in p {
        min_x = min_x.min(q.x);
        max_x = max_x.max(q.x);
        min_y = min_y.min(q.y);
        max_y = max_y.max(q.y);
    }
    let left = (min_x.floor() as i32).max(0);
    let right = (max_x.ceil() as i32).min(w as i32 - 1);
    let top = (min_y.floor() as i32).max(0);
    let bottom = (max_y.ceil() as i32).min(h as i32 - 1);
    if left > right || top > bottom {
        return 0.;
    }
    let px = p.map(|p| p.x as i32 - left);
    let py = p.map(|p| p.y as i32 - top);
    let (mut sum, mut count) = (0f64, 0u64);
    for y in top..=bottom {
        let cy = y - top;
        let mut thresholds = [0f32; 4];
        let mut hit_low = [0; 4];
        let mut hit_high = [0; 4];
        let (mut nt, mut nh) = (0, 0);
        for i in 0..4 {
            let j = (i + 3) % 4;
            let (ax, ay, bx, by) = (px[i], py[i], px[j], py[j]);
            let dx = bx - ax;
            let dy = by - ay;
            if (ay > cy) != (by > cy) {
                thresholds[nt] = dx as f32 * (cy - ay) as f32 / dy as f32 + ax as f32;
                nt += 1;
            }
            if cy < ay.min(by) || cy > ay.max(by) {
                continue;
            }
            if dy == 0 {
                hit_low[nh] = ax.min(bx);
                hit_high[nh] = ax.max(bx);
                nh += 1;
            } else {
                let numerator = (cy - ay) as i64 * dx as i64;
                if numerator % dy as i64 == 0 {
                    let hx = ax + (numerator / dy as i64) as i32;
                    if hx >= ax.min(bx) && hx <= ax.max(bx) {
                        hit_low[nh] = hx;
                        hit_high[nh] = hx;
                        nh += 1;
                    }
                }
            }
        }
        for i in 1..nt {
            let v = thresholds[i];
            let mut j = i;
            while j > 0 && thresholds[j - 1] > v {
                thresholds[j] = thresholds[j - 1];
                j -= 1;
            }
            thresholds[j] = v;
        }
        let mut t = 0;
        for x in left..=right {
            let cx = x - left;
            while t < nt && thresholds[t] <= cx as f32 {
                t += 1;
            }
            let mut inside = (nt - t) & 1 != 0;
            if !inside {
                for k in 0..nh {
                    if cx >= hit_low[k] && cx <= hit_high[k] {
                        inside = true;
                        break;
                    }
                }
            }
            if inside {
                sum += prediction[y as usize * w + x as usize] as f64;
                count += 1;
            }
        }
    }
    if count == 0 {
        0.
    } else {
        (sum / count as f64) as f32
    }
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
fn unclip(p: &[Point; 4], ratio: f32, out: &mut Vec<Point>) -> ImageResult<()> {
    out.clear();
    let (mut area, mut perimeter) = (0f64, 0f64);
    for i in 0..4 {
        let a = p[i];
        let b = p[(i + 1) % 4];
        area += a.x as f64 * b.y as f64 - b.x as f64 * a.y as f64;
        perimeter += distance(a, b) as f64;
    }
    let delta = (area * 0.5).abs() * ratio as f64 / perimeter;
    if !delta.is_finite() || delta <= 0. {
        return Ok(());
    }
    let mut pts = [[0i64; 2]; 4];
    let mut n = 0;
    for q in p {
        let q = [q.x.round_ties_even() as i64, q.y.round_ties_even() as i64];
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
    let mut prev = n - 1;
    for i in 0..n {
        let [x, y] = pts[i];
        let incoming = normals[prev];
        let outgoing = normals[i];
        rounded(
            out,
            x as f64 + incoming[0] * delta,
            y as f64 + incoming[1] * delta,
        )?;
        if (incoming[0] * outgoing[1] - outgoing[0] * incoming[1]) * delta >= 0. {
            if outgoing[0] * incoming[0] + outgoing[1] * incoming[1] < 0.985 {
                let start = incoming[1].atan2(incoming[0]);
                let mut end = outgoing[1].atan2(outgoing[0]);
                if end < start {
                    end += std::f64::consts::PI * 2.;
                }
                arc(out, x as f64, y as f64, start, end, delta)?;
            }
        } else {
            rounded(out, x as f64, y as f64)?;
        }
        rounded(
            out,
            x as f64 + outgoing[0] * delta,
            y as f64 + outgoing[1] * delta,
        )?;
        prev = i;
    }
    Ok(())
}
fn arc(
    out: &mut Vec<Point>,
    cx: f64,
    cy: f64,
    start: f64,
    end: f64,
    radius: f64,
) -> ImageResult<()> {
    let radius_abs = radius.abs();
    let chord = 0.25f64.min(radius_abs);
    let fraction = (end - start).abs() / (2. * std::f64::consts::PI);
    if fraction <= 0. || radius_abs <= 0. {
        return Ok(());
    }
    let mut steps = (fraction * std::f64::consts::PI / (1. - chord / radius_abs).acos()) as usize;
    steps = steps.max(2);
    let cap = (222. * fraction) as usize;
    if steps > cap {
        steps = cap.max(2);
    }
    let (mut x, mut y) = (start.cos(), start.sin());
    let step = (end - start) / steps as f64;
    let (cos, sin) = (step.cos(), step.sin());
    for _ in 0..=steps {
        rounded(out, cx + x * radius, cy + y * radius)?;
        let nx = x * cos - y * sin;
        y = x * sin + y * cos;
        x = nx;
    }
    Ok(())
}
pub(crate) fn sort_reading_order(boxes: &mut [DetectionBox]) {
    dotnet_sort(boxes, |a, b| {
        a.points[0][1]
            .partial_cmp(&b.points[0][1])
            .unwrap()
            .then_with(|| a.points[0][0].partial_cmp(&b.points[0][0]).unwrap())
    });
    for i in 0..boxes.len().saturating_sub(1) {
        for j in (0..=i).rev() {
            let a = boxes[j];
            let b = boxes[j + 1];
            if (b.points[0][1] - a.points[0][1]).abs() >= 10. || b.points[0][0] >= a.points[0][0] {
                break;
            }
            boxes.swap(j, j + 1);
        }
    }
}
// Adapted from .NET Foundation ArraySortHelper<T> introsort, MIT licensed.
// https://github.com/dotnet/runtime/blob/main/src/libraries/System.Private.CoreLib/src/System/Collections/Generic/ArraySortHelper.cs
// Reproduces equal-key permutations of List<T>.Sort used by the reference.
fn dotnet_sort<T: Copy, F: Fn(&T, &T) -> Ordering>(a: &mut [T], compare: F) {
    if a.len() > 1 {
        intro(a, 2 * (a.len().ilog2() + 1), &compare);
    }
}
fn swap_greater<T: Copy, F: Fn(&T, &T) -> Ordering>(a: &mut [T], i: usize, j: usize, c: &F) {
    if c(&a[i], &a[j]) == Ordering::Greater {
        a.swap(i, j);
    }
}
fn intro<T: Copy, F: Fn(&T, &T) -> Ordering>(mut a: &mut [T], mut depth: u32, c: &F) {
    while a.len() > 1 {
        let n = a.len();
        if n <= 16 {
            if n == 2 {
                swap_greater(a, 0, 1, c);
            } else if n == 3 {
                swap_greater(a, 0, 1, c);
                swap_greater(a, 0, 2, c);
                swap_greater(a, 1, 2, c);
            } else {
                for i in 1..n {
                    let value = a[i];
                    let mut j = i;
                    while j > 0 && c(&value, &a[j - 1]) == Ordering::Less {
                        a[j] = a[j - 1];
                        j -= 1;
                    }
                    a[j] = value;
                }
            }
            return;
        }
        if depth == 0 {
            heap(a, c);
            return;
        }
        depth -= 1;
        let hi = n - 1;
        let middle = hi >> 1;
        swap_greater(a, 0, middle, c);
        swap_greater(a, 0, hi, c);
        swap_greater(a, middle, hi, c);
        let pivot = a[middle];
        a.swap(middle, hi - 1);
        let (mut left, mut right) = (0, hi - 1);
        loop {
            left += 1;
            while c(&a[left], &pivot) == Ordering::Less {
                left += 1;
            }
            right -= 1;
            while c(&pivot, &a[right]) == Ordering::Less {
                right -= 1;
            }
            if left >= right {
                break;
            }
            a.swap(left, right);
        }
        if left != hi - 1 {
            a.swap(left, hi - 1);
        }
        let (lo, rest) = a.split_at_mut(left);
        intro(&mut rest[1..], depth, c);
        a = lo;
    }
}
fn heap<T: Copy, F: Fn(&T, &T) -> Ordering>(a: &mut [T], c: &F) {
    let n = a.len();
    for i in (1..=n / 2).rev() {
        down_heap(a, i, n, c);
    }
    for i in (2..=n).rev() {
        a.swap(0, i - 1);
        down_heap(a, 1, i - 1, c);
    }
}
fn down_heap<T: Copy, F: Fn(&T, &T) -> Ordering>(a: &mut [T], mut i: usize, n: usize, c: &F) {
    let value = a[i - 1];
    while i <= n / 2 {
        let mut child = 2 * i;
        if child < n && c(&a[child - 1], &a[child]) == Ordering::Less {
            child += 1;
        }
        if c(&value, &a[child - 1]) != Ordering::Less {
            break;
        }
        a[i - 1] = a[child - 1];
        i = child;
    }
    a[i - 1] = value;
}

#[cfg(test)]
mod tests {
    use super::*;
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
