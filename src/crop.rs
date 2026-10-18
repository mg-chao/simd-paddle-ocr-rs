//! Canonical perspective crops and in-place line orientation.
use crate::{
    detector::DetectionBox,
    image::{ImageError, ImageResult, ImageView},
};
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
        let t = Self {
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
    pub(crate) fn map_span(self, left: f64, right: f64) -> ImageResult<[[f32; 2]; 4]> {
        let coordinates = if self.vertical {
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
        for (local_y, row) in destination.chunks_exact_mut(row_bytes).enumerate() {
            let dy = first_row + local_y;
            for first in (0..self.width()).step_by(kernel.map_lanes()) {
                let count = kernel.map_lanes().min(self.width() - first);
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
                    [
                        self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h,
                    ],
                    x,
                    y,
                    self.width,
                    self.height,
                    count,
                )?;
                for lane in 0..count {
                    let (sx, sy) = (mapped[0][lane], mapped[1][lane]);
                    let pixel = crate::pixel_kernels::sample_interior(kernel, image, sx, sy)
                        .unwrap_or_else(|| sample_clamped(image, sx, sy));
                    let offset = (first + lane) * 3;
                    row[offset..offset + 3].copy_from_slice(&pixel);
                }
            }
        }
        Ok(())
    }
}
fn sample_clamped(image: ImageView<'_>, sx: f64, sy: f64) -> [u8; 3] {
    let xb = sx.floor() as i64;
    let yb = sy.floor() as i64;
    let wx = std::array::from_fn::<_, 4, _>(|k| cubic(sx - (xb + k as i64 - 1) as f64));
    let wy = std::array::from_fn::<_, 4, _>(|k| cubic(sy - (yb + k as i64 - 1) as f64));
    let mut values = [0f64; 3];
    for (ky, &wy) in wy.iter().enumerate() {
        let py = (yb + ky as i64 - 1).clamp(0, image.height() as i64 - 1) as usize;
        for (kx, &wx) in wx.iter().enumerate() {
            let px = (xb + kx as i64 - 1).clamp(0, image.width() as i64 - 1) as usize;
            let pixel = image.bgr(px, py);
            for c in 0..3 {
                values[c] += pixel[c] as f64 * wx * wy;
            }
        }
    }
    values.map(crate::pixel_kernels::clamp_byte)
}
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    let a = -0.75;
    if x <= 1. {
        (a + 2.) * x * x * x - (a + 3.) * x * x + 1.
    } else if x < 2. {
        a * x * x * x - 5. * a * x * x + 8. * a * x - 4. * a
    } else {
        0.
    }
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
