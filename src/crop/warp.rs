//! Invert the forward perspective transform using the same operation order as
//! getPerspectiveTransform/warpPerspective. Round-off matters at half phases.
//! See licenses/opencv-calipers.txt for the OpenCV attribution.
pub(super) fn inverse(points: [[f32; 2]; 4], w: usize, h: usize) -> Option<[f64; 9]> {
    let dst = [
        [0., 0.],
        [w as f32, 0.],
        [w as f32, h as f32],
        [0., h as f32],
    ];
    let mut rows = [[0.; 9]; 8];
    for (i, ([x, y], [u, v])) in points.into_iter().zip(dst).enumerate() {
        rows[i] = [x, y, 1., 0., 0., 0., -x * u, -y * u, u].map(f64::from);
        rows[i + 4] = [0., 0., 0., x, y, 1., -x * v, -y * v, v].map(f64::from);
    }
    for i in 0..8 {
        let mut pivot = i;
        for j in i + 1..8 {
            if rows[j][i].abs() > rows[pivot][i].abs() {
                pivot = j;
            }
        }
        if rows[pivot][i].abs() < 100. * f64::EPSILON {
            return None;
        }
        rows.swap(i, pivot);
        let row = rows[i];
        let scale = -1. / row[i];
        for target in &mut rows[i + 1..] {
            let factor = target[i] * scale;
            for k in i + 1..9 {
                target[k] += factor * row[k];
            }
        }
    }
    let mut forward = [0.; 8];
    for i in (0..8).rev() {
        let mut value = rows[i][8];
        for k in i + 1..8 {
            value -= rows[i][k] * forward[k];
        }
        forward[i] = value / rows[i][i];
    }
    let [a, b, c, d, e, f, g, h] = forward;
    let det = a * (e - f * h) - b * (d - f * g) + c * (d * h - e * g);
    if !det.is_finite() || det == 0. {
        return None;
    }
    let inv = 1. / det;
    Some(
        [
            e - f * h,
            c * h - b,
            b * f - c * e,
            f * g - d,
            a - c * g,
            c * d - a * f,
            d * h - e * g,
            b * g - a * h,
            a * e - b * d,
        ]
        .map(|k| k * inv),
    )
}
