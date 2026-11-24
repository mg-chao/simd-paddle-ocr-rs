//! Canonical perspective crops and in-place line orientation.
use crate::{
    detector::DetectionBox,
    image::{ImageError, ImageResult, ImageView},
};
mod warp;

const EPSILON: f64 = 1e-12;
#[derive(Clone, Copy, Debug)]
pub struct CropTransform {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    width: usize,
    height: usize,
    vertical: bool,
    warp: [f64; 9],
}
impl CropTransform {
    pub fn new(box_: &DetectionBox) -> ImageResult<Self> {
        let p = box_.points.map(|p| [p[0] as f64, p[1] as f64]);
        if p.iter().flatten().any(|x| !x.is_finite()) {
            return Err(ImageError::new("invalid detection quadrilateral"));
        }
        let distance = |a: [f64; 2], b: [f64; 2]| {
            ((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])).sqrt()
        };
        let w = distance(p[0], p[1]).max(distance(p[2], p[3]));
        let ht = distance(p[0], p[3]).max(distance(p[1], p[2]));
        if !w.is_finite()
            || !ht.is_finite()
            || w < 1.
            || ht < 1.
            || w > i32::MAX as f64
            || ht > i32::MAX as f64
        {
            return Err(ImageError::new("invalid crop dimensions"));
        }
        let width = w.floor() as usize;
        let height = ht.floor() as usize;
        let vertical = height as f64 >= width as f64 * 1.5;
        let dx1 = p[1][0] - p[2][0];
        let dx2 = p[3][0] - p[2][0];
        let dx3 = p[0][0] - p[1][0] + p[2][0] - p[3][0];
        let dy1 = p[1][1] - p[2][1];
        let dy2 = p[3][1] - p[2][1];
        let dy3 = p[0][1] - p[1][1] + p[2][1] - p[3][1];
        let (mut g, mut h) = (0., 0.);
        if dx3.abs() > EPSILON || dy3.abs() > EPSILON {
            let denominator = dx1 * dy2 - dx2 * dy1;
            if !denominator.is_finite() || denominator.abs() <= EPSILON {
                return Err(ImageError::new("invalid perspective transform"));
            }
            g = (dx3 * dy2 - dx2 * dy3) / denominator;
            h = (dx1 * dy3 - dx3 * dy1) / denominator;
        }
        let mut t = Self {
            a: p[1][0] - p[0][0] + g * p[1][0],
            b: p[3][0] - p[0][0] + h * p[3][0],
            c: p[0][0],
            d: p[1][1] - p[0][1] + g * p[1][1],
            e: p[3][1] - p[0][1] + h * p[3][1],
            f: p[0][1],
            g,
            h,
            width,
            height,
            vertical,
            warp: [0.; 9],
        };
        let (cw, ch) = (width as f64, height as f64);
        let analytic = [
            t.a / cw,
            t.b / ch,
            t.c,
            t.d / cw,
            t.e / ch,
            t.f,
            t.g / cw,
            t.h / ch,
            1.,
        ];
        t.warp = if analytic[0] == 1.
            && analytic[1] == 0.
            && analytic[3] == 0.
            && analytic[4] == 1.
            && analytic[6] == 0.
            && analytic[7] == 0.
            && t.c.fract() == 0.
            && t.f.fract() == 0.
        {
            analytic
        } else {
            warp::inverse(box_.points, width, height).unwrap_or(analytic)
        };
        if [t.a, t.b, t.c, t.d, t.e, t.f, t.g, t.h]
            .iter()
            .any(|x| !x.is_finite())
        {
            return Err(ImageError::new("invalid perspective transform"));
        }
        // A denominator is affine on the sampling rectangle. Reject poles before writing.
        let u = (width - 1) as f64 / width as f64;
        let v = (height - 1) as f64 / height as f64;
        let den = [1., g * u + 1., h * v + 1., g * u + h * v + 1.];
        if den.iter().any(|&x| !x.is_finite() || x <= EPSILON) {
            return Err(ImageError::new("perspective transform has a pole in crop"));
        }
        t.byte_count()?;
        Ok(t)
    }
    pub fn width(self) -> usize {
        if self.vertical {
            self.height
        } else {
            self.width
        }
    }
    pub fn height(self) -> usize {
        if self.vertical {
            self.width
        } else {
            self.height
        }
    }
    pub fn unrotated_width(self) -> usize {
        self.width
    }
    pub fn unrotated_height(self) -> usize {
        self.height
    }
    /// Tall crops are rotated 90 degrees counterclockwise before classification
    /// or recognition, so source top-to-bottom becomes crop left-to-right.
    pub fn vertical_rotation(self) -> bool {
        self.vertical
    }
    /// Map a normalized interval in the oriented crop back onto the source quad.
    pub(crate) fn map_span(
        self,
        left: f64,
        right: f64,
        vertical: bool,
    ) -> ImageResult<[[f32; 2]; 4]> {
        let coordinates = if vertical {
            // Undo the CCW rotation in crop-corner order (TL, TR, BR, BL).
            [[1., left], [1., right], [0., right], [0., left]]
        } else {
            [[left, 0.], [right, 0.], [right, 1.], [left, 1.]]
        };
        let mut result = [[0.; 2]; 4];
        for (point, [u, v]) in result.iter_mut().zip(coordinates) {
            let denominator = self.g * u + self.h * v + 1.;
            if !denominator.is_finite() || denominator.abs() <= EPSILON {
                return Err(ImageError::new("invalid perspective transform"));
            }
            let x = (self.a * u + self.b * v + self.c) / denominator;
            let y = (self.d * u + self.e * v + self.f) / denominator;
            if !x.is_finite()
                || !y.is_finite()
                || !((x as f32).is_finite() && (y as f32).is_finite())
            {
                return Err(ImageError::new("nonfinite character coordinates"));
            }
            *point = [x as f32, y as f32];
        }
        Ok(result)
    }
    pub fn byte_count(self) -> ImageResult<usize> {
        self.width
            .checked_mul(self.height)
            .and_then(|x| x.checked_mul(3))
            .filter(|&x| x <= isize::MAX as usize)
            .ok_or_else(|| ImageError::new("crop byte count overflow"))
    }
    pub fn extract_into(self, image: ImageView<'_>, destination: &mut [u8]) -> ImageResult<()> {
        let bytes = self.byte_count()?;
        if destination.len() < bytes {
            return Err(ImageError::new("crop destination is too small"));
        }
        self.extract_rows_into(image, 0, &mut destination[..bytes])
    }
    /// Extract a contiguous band of final output rows. Rotated crops use the
    /// same source-grid coordinates, so bands remain disjoint without unsafe
    /// aliases even when the source sampling grid is rotated ninety degrees.
    pub fn extract_rows_into(
        self,
        image: ImageView<'_>,
        first_row: usize,
        destination: &mut [u8],
    ) -> ImageResult<()> {
        self.extract_rows_with_kernel(
            image,
            first_row,
            destination,
            crate::pixel_kernels::PixelKernel::selected(),
        )
    }
    fn extract_rows_with_kernel(
        self,
        image: ImageView<'_>,
        first_row: usize,
        destination: &mut [u8],
        kernel: crate::pixel_kernels::PixelKernel,
    ) -> ImageResult<()> {
        let row_bytes = self
            .width()
            .checked_mul(3)
            .ok_or_else(|| ImageError::new("crop row size overflow"))?;
        if !destination.len().is_multiple_of(row_bytes)
            || first_row
                .checked_add(destination.len() / row_bytes)
                .is_none_or(|end| end > self.height())
        {
            return Err(ImageError::new("invalid crop output row band"));
        }
        self.map_rows(
            first_row,
            destination.len() / row_bytes,
            kernel,
            |i, sx, sy| {
                destination[i * 3..i * 3 + 3].copy_from_slice(&sample(image, sx, sy));
            },
        )
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn is_integer_translation(self) -> bool {
        let [a, b, c, d, e, f, g, h, i] = self.warp;
        [a, b, d, e, g, h, i] == [1., 0., 0., 1., 0., 0., 1.] && c.fract() == 0. && f.fract() == 0.
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn fixed_coordinates(
        self,
        source: [usize; 2],
        output: &mut [[i32; 2]],
    ) -> ImageResult<()> {
        if self.width().checked_mul(self.height()) != Some(output.len()) {
            return Err(ImageError::new("invalid GPU crop coordinate grid size"));
        }
        let max_source = i32::MAX as usize / 32 - 1;
        if source.into_iter().any(|side| side > max_source) {
            return Err(ImageError::new("source exceeds GPU crop coordinate range"));
        }
        self.map_rows(
            0,
            self.height(),
            crate::pixel_kernels::PixelKernel::selected(),
            |i, x, y| {
                // Beyond this range all four taps already replicate an edge.
                output[i] = [
                    fixed_coordinate(x.clamp(-2., source[0] as f64 + 1.)) as i32,
                    fixed_coordinate(y.clamp(-2., source[1] as f64 + 1.)) as i32,
                ];
            },
        )
    }

    fn map_rows(
        self,
        first_row: usize,
        rows: usize,
        kernel: crate::pixel_kernels::PixelKernel,
        mut store: impl FnMut(usize, f64, f64),
    ) -> ImageResult<()> {
        let width = self.width();
        for local_y in 0..rows {
            let dy = first_row + local_y;
            for first in (0..width).step_by(kernel.map_lanes()) {
                let count = kernel.map_lanes().min(width - first);
                // Inverse of a 90-degree CCW rotation: (dst_x, dst_y) maps
                // to (unrotated_width - 1 - dst_y, dst_x) before perspective.
                let x = std::array::from_fn(|lane| {
                    if self.vertical {
                        (self.width - 1 - dy) as f64
                    } else {
                        (first + lane.min(count - 1)) as f64
                    }
                });
                let y = std::array::from_fn(|lane| {
                    if self.vertical {
                        (first + lane.min(count - 1)) as f64
                    } else {
                        dy as f64
                    }
                });
                let mapped = crate::pixel_kernels::map_coordinates(
                    kernel,
                    self.warp,
                    x,
                    y,
                    self.width,
                    self.height,
                    count,
                )?;
                for lane in 0..count {
                    store(
                        local_y * width + first + lane,
                        mapped[0][lane],
                        mapped[1][lane],
                    );
                }
            }
        }
        Ok(())
    }
}
// warpPerspective rounds source coordinates to 1/32 pixel before bicubic
// interpolation. The four-by-four signed coefficient table sums to 2^15.
fn fixed_coordinate(value: f64) -> i64 {
    (value * 32.).round_ties_even() as i64
}
fn sample(image: ImageView<'_>, x: f64, y: f64) -> [u8; 3] {
    let fx = fixed_coordinate(x);
    let fy = fixed_coordinate(y);
    let (x, y) = (fx >> 5, fy >> 5);
    let clip_x = |x: i64| x.clamp(0, image.width() as i64 - 1) as usize;
    let clip_y = |y: i64| y.clamp(0, image.height() as i64 - 1) as usize;
    let phase = ((fy & 31) * 32 + (fx & 31)) as usize;
    if phase == 0 {
        return image.bgr(clip_x(x), clip_y(y));
    }
    static WEIGHTS: std::sync::OnceLock<Vec<[i16; 16]>> = std::sync::OnceLock::new();
    let table = WEIGHTS.get_or_init(|| (0..1024).map(cubic_weights).collect());
    let weights = &table[phase];
    let xs = [x - 1, x, x + 1, x + 2].map(clip_x);
    let mut sum = [1 << 14; 3];
    for (ky, row) in weights.chunks_exact(4).enumerate() {
        let sy = clip_y(y + ky as i64 - 1);
        for (&sx, &w) in xs.iter().zip(row) {
            let pixel = image.bgr(sx, sy);
            for c in 0..3 {
                sum[c] += i32::from(pixel[c]) * i32::from(w);
            }
        }
    }
    sum.map(|v| (v >> 15).clamp(0, 255) as u8)
}

fn cubic_weights(phase: usize) -> [i16; 16] {
    let axis = |k: usize| {
        let x = k as f32 / 32.;
        let y = 1. - x;
        let a = -0.75;
        let first = ((a * (x + 1.) - 5. * a) * (x + 1.) + 8. * a) * (x + 1.) - 4. * a;
        let second = ((a + 2.) * x - (a + 3.)) * x * x + 1.;
        let third = ((a + 2.) * y - (a + 3.)) * y * y + 1.;
        [first, second, third, 1. - first - second - third]
    };
    let (a, b) = (axis(phase % 32), axis(phase / 32));
    let mut table: [i16; 16] =
        std::array::from_fn(|i| (a[i % 4] * b[i / 4] * 32768.).round_ties_even() as i16);
    let delta = 32768 - table.iter().map(|&v| i32::from(v)).sum::<i32>();
    if delta != 0 {
        let mut index = 10;
        for i in [11, 14, 15] {
            if (delta > 0 && table[i] > table[index]) || (delta < 0 && table[i] < table[index]) {
                index = i;
            }
        }
        table[index] = (i32::from(table[index]) + delta) as i16;
    }
    table
}
pub fn rotate180(pixels: &mut [u8], width: usize, height: usize) -> ImageResult<()> {
    let count = width
        .checked_mul(height)
        .ok_or_else(|| ImageError::new("rotation size overflow"))?;
    if width == 0 || height == 0 || count.checked_mul(3) != Some(pixels.len()) {
        return Err(ImageError::new("invalid rotation image size"));
    }
    for left in 0..count / 2 {
        let right = count - 1 - left;
        for c in 0..3 {
            pixels.swap(left * 3 + c, right * 3 + c);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::PixelFormat;
    #[test]
    fn perspective_half_phase_uses_forward_solve_and_inverse() {
        let transform = CropTransform::new(&DetectionBox {
            points: [[20., 369.], [340., 370.], [340., 384.], [20., 383.]],
            score: 1.,
        })
        .unwrap();
        let mapped = crate::pixel_kernels::map_coordinates(
            crate::pixel_kernels::PixelKernel::Scalar,
            transform.warp,
            [45.; 8],
            [0.; 8],
            320,
            14,
            1,
        )
        .unwrap();
        assert_eq!((mapped[1][0] * 32.).round_ties_even(), 11813.);
    }
    #[test]
    fn cubic_phases_and_replicated_edges_match_fixed_pixel_values() {
        let pixels: Vec<_> = (0..90).map(|i| (i * 19) as u8).collect();
        let image = ImageView::new(&pixels, 6, 5, 0, PixelFormat::Bgr24).unwrap();
        assert_eq!(sample(image, 2.31, 1.78), [35, 54, 97]);
        assert_eq!(sample(image, -0.4, 2.15), [149, 168, 213]);
        assert_eq!(sample(image, 4.99, 3.97), [115, 134, 153]);
    }

    #[test]
    fn vector_mapping_and_interior_samples_preserve_scalar_bytes_and_row_bands() {
        let (width, height) = (37, 41);
        for format in [
            PixelFormat::Bgr24,
            PixelFormat::Rgb24,
            PixelFormat::Bgra32,
            PixelFormat::Rgba32,
        ] {
            let bpp = format.bytes_per_pixel();
            let stride = width * bpp + 7;
            let mut bytes = vec![153; (height - 1) * stride + width * bpp];
            for y in 0..height {
                for x in 0..width {
                    let pixel = [
                        (x * 17 + y * 29) as u8,
                        (x * 31 + y * 3) as u8,
                        (x * 7 + y * 11) as u8,
                    ];
                    let pixel = if format.swap_red_blue() {
                        [pixel[2], pixel[1], pixel[0]]
                    } else {
                        pixel
                    };
                    let at = y * stride + x * bpp;
                    bytes[at..at + 3].copy_from_slice(&pixel);
                }
            }
            let image = ImageView::new(&bytes, width, height, stride, format).unwrap();
            for points in [
                [[2.3, 3.1], [31.4, 2.5], [30.2, 27.7], [3.5, 29.2]],
                [[-2.3, -1.7], [16.1, 0.2], [16.8, 14.1], [-1.2, 13.6]],
                [[0., 0.], [7., 0.], [7., 19.], [0., 19.]],
                [[10., 2.], [17., 3.], [20., 32.], [5., 29.]],
                [[5., 5.], [14., 5.], [14., 14.], [5., 14.]],
            ] {
                let transform = CropTransform::new(&DetectionBox { points, score: 1. }).unwrap();
                let mut expected = vec![0; transform.byte_count().unwrap()];
                transform
                    .extract_rows_with_kernel(
                        image,
                        0,
                        &mut expected,
                        crate::pixel_kernels::PixelKernel::Scalar,
                    )
                    .unwrap();
                for kernel in crate::pixel_kernels::PixelKernel::test_kernels() {
                    let mut actual = vec![211; expected.len() + 3];
                    let row_bytes = transform.width() * 3;
                    for (band, output) in actual[..expected.len()]
                        .chunks_mut(row_bytes * 2)
                        .enumerate()
                    {
                        transform
                            .extract_rows_with_kernel(image, band * 2, output, kernel)
                            .unwrap();
                    }
                    assert_eq!(
                        &actual[..expected.len()],
                        expected,
                        "{kernel:?}/{format:?}/{points:?}"
                    );
                    assert_eq!(&actual[expected.len()..], [211; 3]);
                }
            }
        }
    }
    #[test]
    fn output_row_bands_preserve_vertical_counterclockwise_rotation_and_perspective_coordinates() {
        let image: Vec<u8> = (0..100)
            .flat_map(|i| [i as u8, (i * 2) as u8, (i + 3) as u8])
            .collect();
        let source = ImageView::new(&image, 10, 10, 0, PixelFormat::Bgr24).unwrap();
        let vertical = CropTransform::new(&DetectionBox {
            points: [[0., 0.], [3., 0.], [3., 7.], [0., 7.]],
            score: 1.,
        })
        .unwrap();
        assert_eq!((vertical.width(), vertical.height()), (7, 3));
        // The rightmost source column becomes the top output row, read from
        // top to bottom, matching numpy.rot90 / a 90-degree CCW rotation.
        let expected: Vec<u8> = [
            2, 12, 22, 32, 42, 52, 62, 1, 11, 21, 31, 41, 51, 61, 0, 10, 20, 30, 40, 50, 60,
        ]
        .into_iter()
        .flat_map(|value| [value, value * 2, value + 3])
        .collect();
        for kernel in crate::pixel_kernels::PixelKernel::test_kernels() {
            for rows_per_band in [1, 2, 3] {
                let mut output = vec![0; vertical.byte_count().unwrap()];
                for (band, pixels) in output.chunks_mut(7 * 3 * rows_per_band).enumerate() {
                    vertical
                        .extract_rows_with_kernel(source, band * rows_per_band, pixels, kernel)
                        .unwrap();
                }
                assert_eq!(output, expected, "{kernel:?}/{rows_per_band}");
            }
        }
        let skew = CropTransform::new(&DetectionBox {
            points: [[1., 1.], [8., 2.], [7., 8.], [2., 7.]],
            score: 1.,
        })
        .unwrap();
        let mut whole = vec![0; skew.byte_count().unwrap()];
        skew.extract_into(source, &mut whole).unwrap();
        let mut bands = vec![0; whole.len()];
        for (band, pixels) in bands.chunks_mut(skew.width() * 3 * 2).enumerate() {
            skew.extract_rows_into(source, band * 2, pixels).unwrap();
        }
        assert_eq!(whole, bands);
        assert!(
            skew.extract_rows_into(source, skew.height(), &mut [0; 3])
                .is_err()
        );
    }
}
