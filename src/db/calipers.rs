//! Minimum-area rectangle with OpenCV 4.12 rotated-rectangle rounding.
//! Adapted from modules/imgproc/src/rotcalipers.cpp. See licenses/opencv-calipers.txt.
use super::{Point, order_clockwise};

fn rotate(p: Point, side: usize) -> Point {
    match side {
        0 => p,
        1 => Point::new(p.y, -p.x),
        2 => Point::new(-p.x, -p.y),
        _ => Point::new(-p.y, p.x),
    }
}

pub(super) fn bounding_box(hull: &[Point]) -> (f32, [Point; 4]) {
    let mut support = [0; 4];
    for (i, p) in hull.iter().enumerate() {
        if p.y < hull[support[0]].y {
            support[0] = i;
        }
        if p.x > hull[support[1]].x {
            support[1] = i;
        }
        if p.y > hull[support[2]].y {
            support[2] = i;
        }
        if p.x < hull[support[3]].x {
            support[3] = i;
        }
    }
    let edge = |i: usize| {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        Point::new(b.x - a.x, b.y - a.y)
    };
    let mut best = f32::INFINITY;
    let mut result = (Point::default(), Point::default(), Point::default());
    for _ in hull {
        let dirs: [Point; 4] = std::array::from_fn(|k| rotate(edge(support[k]), k));
        let mut side = 0;
        for k in 1..4 {
            if dirs[k].y * dirs[side].x - dirs[k].x * dirs[side].y < 0. {
                side = k;
            }
        }
        let v = dirs[side];
        let inv = (1. / (f64::from(v.x).hypot(f64::from(v.y)))) as f32;
        let (a, b) = (v.x * inv, v.y * inv);
        support[side] = (support[side] + 1) % hull.len();
        let [bottom, right, top, left] = support.map(|i| hull[i]);
        let w = (right.x - left.x) * a + (right.y - left.y) * b;
        let h = -(top.x - bottom.x) * b + (top.y - bottom.y) * a;
        if w * h <= best {
            best = w * h;
            let u = a * left.x + b * left.y;
            let v = -b * bottom.x + a * bottom.y;
            let inv_det = 1. / (a * a + b * b);
            result = (
                Point::new((u * a - v * b) * inv_det, (a * v + b * u) * inv_det),
                Point::new(a * w, b * w),
                Point::new(-b * h, a * h),
            );
        }
    }
    let (origin, u, v) = result;
    let center = Point::new(origin.x + (u.x + v.x) * 0.5, origin.y + (u.y + v.y) * 0.5);
    let w = f64::from(u.x).hypot(f64::from(u.y)) as f32;
    let h = f64::from(v.x).hypot(f64::from(v.y)) as f32;
    let angle = f64::from(u.y).atan2(f64::from(u.x)) as f32;
    let degrees = (f64::from(angle * 180.) / std::f64::consts::PI) as f32;
    let (sin, cos) = f64::from(degrees).to_radians().sin_cos();
    let (a, b) = (sin as f32 * 0.5, cos as f32 * 0.5);
    let p = Point::new(center.x - a * h - b * w, center.y + b * h - a * w);
    let q = Point::new(center.x + a * h - b * w, center.y - b * h - a * w);
    let opposite = |p: Point| Point::new(2. * center.x - p.x, 2. * center.y - p.y);
    let mut points = [p, q, opposite(p), opposite(q)];
    order_clockwise(&mut points);
    (w.min(h), points)
}
