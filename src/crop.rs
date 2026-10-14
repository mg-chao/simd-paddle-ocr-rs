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
    pub fn vertical_rotation(self) -> bool {
        self.vertical
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
        for y in 0..self.height {
            let v = y as f64 / self.height as f64;
            for x in 0..self.width {
                let u = x as f64 / self.width as f64;
                let denominator = self.g * u + self.h * v + 1.;
                if !denominator.is_finite() || denominator.abs() <= EPSILON {
                    return Err(ImageError::new("invalid perspective denominator"));
                }
                let sx = (self.a * u + self.b * v + self.c) / denominator;
                let sy = (self.d * u + self.e * v + self.f) / denominator;
                if !sx.is_finite()
                    || !sy.is_finite()
                    || sx < i32::MIN as f64
                    || sx > i32::MAX as f64 - 2.
                    || sy < i32::MIN as f64
                    || sy > i32::MAX as f64 - 2.
                {
                    return Err(ImageError::new("perspective coordinate is out of range"));
                }
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
                let (dx, dy) = if self.vertical {
                    (self.height - 1 - y, x)
                } else {
                    (x, y)
                };
                let offset = (dy * self.width() + dx) * 3;
                for c in 0..3 {
                    destination[offset + c] = if values[c] <= 0. {
                        0
                    } else if values[c] >= 255. {
                        255
                    } else {
                        (values[c] + 0.5).floor() as u8
                    };
                }
            }
        }
        Ok(())
    }
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
