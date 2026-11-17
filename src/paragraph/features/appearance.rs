use super::{Point, Quad, dot32, edges, twice_area};
use crate::image::ImageView;

const W: usize = 64;
const H: usize = 32;
const WINDOWS: usize = 3;
const WINDOW: usize = W * H;
const STROKE_SAMPLES: usize = WINDOW * WINDOWS;
const COLOR_INTERIOR: usize = 32 * 8;
const COLOR_SAMPLES: usize = COLOR_INTERIOR + 4 * 16;
const COLOR_WEIGHTS: [[f32; 4]; COLOR_SAMPLES] = color_weights();

const fn color_weights() -> [[f32; 4]; COLOR_SAMPLES] {
    let mut weights = [[0.0; 4]; COLOR_SAMPLES];
    let mut i = 0;
    while i < COLOR_SAMPLES {
        let (u, v) = if i < COLOR_INTERIOR {
            let x = (i % 32) as f32;
            let y = (i / 32) as f32;
            ((x + (y + 0.5) / 8.0) / 32.0, (y + (x + 0.5) / 32.0) / 8.0)
        } else {
            let edge = i - COLOR_INTERIOR;
            let t = ((edge % 16) as f32 + 0.5) / 16.0;
            match edge / 16 {
                0 => (t, 0.0),
                1 => (1.0, t),
                2 => (t, 1.0),
                _ => (0.0, t),
            }
        };
        weights[i] = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
        i += 1;
    }
    weights
}

fn gray([b, g, r]: [u8; 3]) -> u8 {
    // OpenCV's 15-bit integer BGR2GRAY transform.
    ((3735 * u32::from(b) + 19235 * u32::from(g) + 9798 * u32::from(r) + (1 << 14)) >> 15) as u8
}

fn otsu(histogram: &[u32], n: u32) -> u8 {
    let total: u32 = histogram
        .iter()
        .enumerate()
        .map(|(i, &count)| i as u32 * count)
        .sum();
    let mut best = (0, 0.0);
    let (mut left, mut weight) = (0, 0);
    for (i, &count) in histogram.iter().enumerate() {
        left += i as u32 * count;
        weight += count;
        let delta = f64::from(left) * f64::from(n) - f64::from(total) * f64::from(weight);
        let score = delta * delta / f64::from((weight * (n - weight)).max(1));
        if score > best.1 {
            best = (i as u8, score);
        }
    }
    best.0
}

fn visible(image: ImageView<'_>, [x, y]: Point) -> bool {
    x >= 0.0 && x <= (image.width() - 1) as f32 && y >= 0.0 && y <= (image.height() - 1) as f32
}

fn cyclic_hsv([b, g, r]: [f32; 3]) -> [f32; 3] {
    let value = r.max(g).max(b);
    let chroma = value - r.min(g).min(b);
    let saturation = chroma / (value.abs() + f32::EPSILON);
    let factor = 60.0 / (chroma + f32::EPSILON);
    let mut hue = if value == r {
        (g - b) * factor
    } else if value == g {
        (b - r) * factor + 120.0
    } else {
        (r - g) * factor + 240.0
    };
    if hue < 0.0 {
        hue += 360.0;
    }
    let (sine, cosine) = hue.to_radians().sin_cos();
    [saturation * cosine, saturation * sine, value]
}

fn colors(image: ImageView<'_>, q: &Quad, output: &mut [f32; 10]) {
    let mut samples = [None; COLOR_SAMPLES];
    let mut histogram = [0u32; 32];
    for (weight, sample) in COLOR_WEIGHTS.iter().zip(&mut samples) {
        let p = std::array::from_fn(|d| {
            let first = weight[0] * q[0][d];
            let second = weight[1].mul_add(q[1][d], first);
            let third = weight[2].mul_add(q[2][d], second);
            weight[3].mul_add(q[3][d], third)
        });
        if visible(image, p) {
            let [x, y] = p.map(|v| v.round_ties_even() as usize);
            let pixel = image.bgr(x.min(image.width() - 1), y.min(image.height() - 1));
            let bin = gray(pixel) >> 3;
            histogram[usize::from(bin)] += 1;
            *sample = Some((pixel, bin));
        }
    }
    let count = histogram.iter().sum();
    let threshold = otsu(&histogram, count);
    let mut sums = [[0u32; 3]; 2];
    let (mut totals, mut border) = ([0u32; 2], [0u32; 2]);
    for (i, sample) in samples.into_iter().enumerate() {
        if let Some((pixel, bin)) = sample {
            let light = usize::from(bin > threshold);
            totals[light] += 1;
            if i >= COLOR_INTERIOR {
                border[light] += 1;
            }
            for (sum, channel) in sums[light].iter_mut().zip(pixel) {
                *sum += u32::from(channel);
            }
        }
    }
    let voters = if border == [0, 0] { totals } else { border };
    let background = usize::from(voters[1] >= voters[0]);
    for (region, tone) in [background ^ 1, background].into_iter().enumerate() {
        if totals[tone] != 0 {
            let bgr = sums[tone].map(|sum| sum as f32 / (totals[tone] * 255) as f32);
            output[region * 3..][..3].copy_from_slice(&cyclic_hsv(bgr));
            output[7 + region] = 1.0;
        }
    }
}

fn quantized_gray(image: ImageView<'_>, [x, y]: Point) -> u8 {
    // INTER_LINEAR rounds the coordinate to a 1/32-pixel grid, then uses
    // 15-bit fixed-point bilinear weights. Here each weight contains 5+5
    // significant bits, so the common factor of 32 can be dropped.
    let quantize = |value: f32, size: usize| {
        let fixed = (value * 32.0).round_ties_even() as usize;
        let index = (fixed >> 5).min(size - 1);
        (index, (index + 1).min(size - 1), (fixed & 31) as u32)
    };
    let (x0, x1, fx) = quantize(x, image.width());
    let (y0, y1, fy) = quantize(y, image.height());
    let p = [
        image.bgr(x0, y0),
        image.bgr(x1, y0),
        image.bgr(x0, y1),
        image.bgr(x1, y1),
    ];
    let weights = [
        (32 - fx) * (32 - fy),
        fx * (32 - fy),
        (32 - fx) * fy,
        fx * fy,
    ];
    gray(std::array::from_fn(|d| {
        let sum: u32 = (0..4).map(|i| weights[i] * u32::from(p[i][d])).sum();
        ((sum + 512) >> 10) as u8
    }))
}

fn lerp(a: Point, b: Point, t: f32) -> Point {
    std::array::from_fn(|i| (1.0 - t) * a[i] + t * b[i])
}

// Squared Euclidean transform on an ink window padded by one background pixel.
// Find each pixel's horizontal distance to background, then transform only
// the runs of ink in each column.
fn distances(window: &[bool]) -> [u16; WINDOW] {
    let mut horizontal = [[0u16; W]; H + 2];
    for (row, result) in window.chunks_exact(W).zip(&mut horizontal[1..=H]) {
        let mut run = 0;
        for (&ink, out) in row.iter().zip(result.iter_mut()) {
            run = if ink { run + 1 } else { 0 };
            *out = run * run;
        }
        run = 0;
        for (&ink, out) in row.iter().rev().zip(result.iter_mut().rev()) {
            run = if ink { run + 1 } else { 0 };
            *out = (*out).min(run * run);
        }
    }
    let mut output = [0; WINDOW];
    for x in 0..W {
        let mut start = 1;
        while start <= H {
            if horizontal[start][x] == 0 {
                start += 1;
                continue;
            }
            let mut stop = start + 1;
            while horizontal[stop][x] != 0 {
                stop += 1;
            }
            if stop - start <= 2 {
                for y in start..stop {
                    output[(y - 1) * W + x] = 1;
                }
            } else {
                transform_run(&horizontal, x, start - 1, stop, &mut output);
            }
            start = stop + 1;
        }
    }
    output
}

fn transform_run(
    horizontal: &[[u16; W]; H + 2],
    x: usize,
    before: usize,
    after: usize,
    output: &mut [u16; WINDOW],
) {
    // The background pixels immediately above and below this run are
    // closer than any site outside it. Only the parabolas between those
    // two sites can contribute to its distance transform.
    let mut vertices = [0usize; H + 2];
    let mut limits = [0.0; H + 3];
    vertices[0] = before;
    limits[0] = f64::NEG_INFINITY;
    let mut k = 0;
    for q in before + 1..=after {
        let a = i32::from(horizontal[q][x]) + (q * q) as i32;
        let intersect = |p: usize| {
            let b = i32::from(horizontal[p][x]) + (p * p) as i32;
            f64::from(a - b) / (2 * (q - p)) as f64
        };
        let mut s = intersect(vertices[k]);
        while s <= limits[k] {
            k -= 1;
            s = intersect(vertices[k]);
        }
        k += 1;
        vertices[k] = q;
        limits[k] = s;
    }
    limits[k + 1] = f64::INFINITY;
    k = 0;
    for q in before + 1..after {
        while limits[k + 1] < q as f64 {
            k += 1;
        }
        let p = vertices[k];
        let dy = q.abs_diff(p) as u16;
        output[(q - 1) * W + x] = horizontal[p][x] + dy * dy;
    }
}

fn sample_stroke(
    image: ImageView<'_>,
    q: &Quad,
    center: f32,
    fraction: f32,
    samples: &mut [Option<u8>],
    histogram: &mut [u32; 256],
) {
    let columns: [(Point, Point); W] = std::array::from_fn(|x| {
        let u = center + fraction * ((x as f32 + 0.5) / W as f32 - 0.5);
        (lerp(q[0], q[1], u), lerp(q[3], q[2], u))
    });
    for (y, row) in samples.chunks_exact_mut(W).enumerate() {
        let v = (y as f32 + 0.5) / H as f32;
        for (sample, &(top, bottom)) in row.iter_mut().zip(&columns) {
            let p = lerp(top, bottom, v);
            if visible(image, p) {
                let g = quantized_gray(image, p);
                histogram[usize::from(g)] += 1;
                *sample = Some(g);
            }
        }
    }
}

fn light_stroke_background(samples: &[Option<u8>; STROKE_SAMPLES], threshold: u8) -> bool {
    let mut count = [0; 2];
    let mut visit = |g: Option<u8>| {
        if let Some(g) = g {
            count[usize::from(g > threshold)] += 1;
        }
    };
    for patch in samples.chunks_exact(WINDOW) {
        for &sample in patch[..W].iter().chain(&patch[WINDOW - W..]) {
            visit(sample);
        }
        for row in patch[W..WINDOW - W].chunks_exact(W) {
            visit(row[0]);
            visit(row[W - 1]);
        }
    }
    count[1] >= count[0]
}

fn stroke(image: ImageView<'_>, q: &Quad, length: f32, thickness: f32, output: &mut [f32; 10]) {
    let fraction = (2.0 * thickness / length.max(1e-6)).min(1.0);
    let mut samples = [None; STROKE_SAMPLES];
    let mut histogram = [0u32; 256];
    for (patch, center) in samples
        .chunks_exact_mut(WINDOW)
        .zip([1.0_f32 / 6.0, 0.5, 5.0 / 6.0])
    {
        let center = center.clamp(fraction * 0.5, 1.0 - fraction * 0.5);
        sample_stroke(image, q, center, fraction, patch, &mut histogram);
    }
    let count = histogram.iter().sum();
    if count < 32 {
        return;
    }
    let threshold = otsu(&histogram, count);
    let light_background = light_stroke_background(&samples, threshold);
    let mut widths = Vec::new();
    for patch in samples.chunks_exact(WINDOW) {
        let foreground: [bool; WINDOW] =
            std::array::from_fn(|i| patch[i].is_some_and(|g| (g > threshold) != light_background));
        let distance = distances(&foreground);
        for y in 1..H - 1 {
            for x in 1..W - 1 {
                let radius_squared = distance[y * W + x];
                if radius_squared != 0
                    && (y - 1..=y + 1).all(|row| {
                        distance[row * W + x - 1..=row * W + x + 1]
                            .iter()
                            .all(|&n| n <= radius_squared)
                    })
                {
                    widths.push(radius_squared);
                }
            }
        }
    }
    if widths.is_empty() {
        return;
    }
    let lo = (widths.len() - 1) / 2;
    let even = widths.len().is_multiple_of(2);
    let (_, mid, upper) = widths.select_nth_unstable(lo);
    let width = |s| f64::from(f32::from(s).sqrt() * 2.0);
    let typical = if even {
        (width(*mid) + width(*upper.iter().min().unwrap())) / 2.0
    } else {
        width(*mid)
    };
    output[6] = (typical / H as f64) as f32;
    if widths.len() >= 8 {
        let mean = widths.iter().map(|&s| width(s)).sum::<f64>() / widths.len() as f64;
        let variance = widths
            .iter()
            .map(|&s| (width(s) - mean).powi(2))
            .sum::<f64>()
            / widths.len() as f64;
        let consistency = (-variance.sqrt() / mean.max(1e-6)).exp();
        output[9] = (f64::from(output[7]) * f64::from(count) / STROKE_SAMPLES as f64
            * (f64::from(thickness) / 16.0).min(1.0)
            * consistency) as f32;
    }
}

pub(super) fn line(image: ImageView<'_>, quad: &Quad) -> [f32; 10] {
    let mut q = *quad;
    let mut output = [0.0; 10];
    let lengths = edges(&q).map(|p| dot32(p, p).sqrt());
    let area = twice_area(&q).abs();
    if area <= 1e-6 || lengths.iter().any(|&v| v <= 1e-6) {
        return output;
    }
    let (mut length, mut thickness) = (
        (lengths[0] + lengths[2]) / 2.0,
        (lengths[1] + lengths[3]) / 2.0,
    );
    if length < thickness {
        q.rotate_left(1);
        std::mem::swap(&mut length, &mut thickness);
    }
    colors(image, &q, &mut output);
    if area * 0.5 > 1e-6 {
        stroke(image, &q, length, thickness, &mut output);
    }
    output
}

pub(super) fn pair(a: &[f32; 10], b: &[f32; 10]) -> [f32; 6] {
    let difference = |start: usize| {
        let norm = (start..start + 3)
            .map(|i| (a[i] - b[i]).powi(2))
            .sum::<f32>()
            .sqrt();
        (f64::from(norm) / 5.0_f64.sqrt()) as f32
    };
    let mut values = [
        difference(0),
        difference(3),
        (a[6].max(1e-6) / b[6].max(1e-6)).ln().abs(),
        a[7].min(b[7]),
        a[8].min(b[8]),
        a[9].min(b[9]),
    ];
    for i in 0..3 {
        if values[3 + i] <= 0.0 {
            values[i] = 0.0;
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_window_includes_diagonal_background_and_padding() {
        let mut window = [true; WINDOW];
        window[12 * W + 15] = false;
        let distance = distances(&window);
        assert_eq!(distance[12 * W + 15], 0);
        assert_eq!(distance[14 * W + 18], 13);
        assert_eq!(distance[1], 1);
        assert_eq!(distance[2 * W - 2], 4);
    }

    #[test]
    fn independent_column_runs_match_nearest_background() {
        let window: [bool; WINDOW] = std::array::from_fn(|i| {
            let (x, y) = (i % W, i / W);
            x % 13 > 3 && y % 11 != 4 && (x + 3 * y) % 17 != 0
        });
        let background: Vec<_> = (-1..=H as i32)
            .flat_map(|y| (-1..=W as i32).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                x < 0
                    || y < 0
                    || x == W as i32
                    || y == H as i32
                    || !window[y as usize * W + x as usize]
            })
            .collect();
        for (i, actual) in distances(&window).into_iter().enumerate() {
            let (x, y) = ((i % W) as i32, (i / W) as i32);
            let expected = background
                .iter()
                .map(|&(bx, by)| (x - bx).pow(2) + (y - by).pow(2))
                .min()
                .unwrap();
            assert_eq!(i32::from(actual), expected, "pixel ({x}, {y})");
        }
    }
}
