//! Byte-domain OpenCV-compatible resize followed by canonical normalization.
use crate::image::{ImageError, ImageResult, ImageView, resize_vec};
use std::sync::OnceLock;

#[derive(Default, Debug)]
pub struct ResizeWorkspace {
    x_offsets: Vec<usize>,
    x_coefficients: Vec<[i32; 2]>,
    row0: Vec<[i32; 3]>,
    row1: Vec<[i32; 3]>,
    key: Option<(usize, usize)>,
}
impl ResizeWorkspace {
    pub fn retained_bytes(&self) -> usize {
        self.x_offsets.capacity() * std::mem::size_of::<usize>()
            + self.x_coefficients.capacity() * 8
            + (self.row0.capacity() + self.row1.capacity()) * 12
    }
    fn prepare(&mut self, source: usize, destination: usize) -> ImageResult<()> {
        resize_vec(&mut self.x_offsets, destination)?;
        resize_vec(&mut self.x_coefficients, destination)?;
        resize_vec(&mut self.row0, destination)?;
        resize_vec(&mut self.row1, destination)?;
        if self.key != Some((source, destination)) {
            for x in 0..destination {
                let coordinate = (x as f64 + 0.5) * source as f64 / destination as f64 - 0.5;
                let mut sx = coordinate.floor() as i64;
                let mut fraction = coordinate - sx as f64;
                if sx < 0 {
                    sx = 0;
                    fraction = 0.;
                }
                if sx >= source as i64 - 1 {
                    sx = source as i64 - 1;
                    fraction = 0.;
                }
                self.x_offsets[x] = sx as usize;
                self.x_coefficients[x] = [
                    ((1. - fraction) * 2048.).round_ties_even() as i32,
                    (fraction * 2048.).round_ties_even() as i32,
                ];
            }
            self.key = Some((source, destination));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetSize {
    pub width: usize,
    pub height: usize,
    pub width_ratio: f32,
    pub height_ratio: f32,
}
pub fn compute_det_size(width: usize, height: usize, limit: usize) -> ImageResult<DetSize> {
    if width == 0
        || height == 0
        || width > i32::MAX as usize
        || height > i32::MAX as usize
        || !(32..=4096).contains(&limit)
    {
        return Err(ImageError::new("invalid detector image size or limit"));
    }
    let (width, height) = if width + height < 64 {
        (width.max(32), height.max(32))
    } else {
        (width, height)
    };
    let maximum = width.max(height);
    let ratio = if maximum > limit {
        limit as f64 / maximum as f64
    } else {
        1.
    };
    let scaled_width = (width as f64 * ratio) as usize;
    let scaled_height = (height as f64 * ratio) as usize;
    let w = (((scaled_width as f64 / 32. + 0.5).floor() as usize) * 32).max(32);
    let h = (((scaled_height as f64 / 32. + 0.5).floor() as usize) * 32).max(32);
    Ok(DetSize {
        width: w,
        height: h,
        width_ratio: (w as f64 / width as f64) as f32,
        height_ratio: (h as f64 / height as f64) as f32,
    })
}
fn det_lut() -> &'static [[f32; 256]; 3] {
    static LUT: OnceLock<[[f32; 256]; 3]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mean = [0.485f64, 0.456, 0.406];
        let inverse = [1. / 0.229f64, 1. / 0.224, 1. / 0.225];
        let mut lut = [[0.; 256]; 3];
        for (c, channel) in lut.iter_mut().enumerate() {
            for (x, value) in channel.iter_mut().enumerate() {
                *value = ((x as f64 / 255. - mean[c]) * inverse[c]) as f32;
            }
        }
        lut
    })
}
fn cls_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0.; 256];
        for (x, v) in lut.iter_mut().enumerate() {
            *v = x as f32 * (2.0f32 / 255.0f32) - 1.;
        }
        lut
    })
}
pub fn det(
    image: ImageView<'_>,
    width: usize,
    height: usize,
    output: &mut [f32],
    workspace: &mut ResizeWorkspace,
) -> ImageResult<()> {
    resize::<true>(
        image,
        image.width(),
        width,
        height,
        width,
        output,
        workspace,
    )
}
pub fn cls(
    image: ImageView<'_>,
    output: &mut [f32],
    workspace: &mut ResizeWorkspace,
) -> ImageResult<usize> {
    if output.len() != 3 * 80 * 160 {
        return Err(ImageError::new("invalid CLS output size"));
    }
    let window = if image.width() as u64 > 4 * image.height() as u64 {
        4 * image.height()
    } else {
        image.width()
    };
    let actual = (80u64 * window as u64)
        .div_ceil(image.height() as u64)
        .min(160) as usize;
    if actual < 160 {
        output.fill(-1.);
    }
    resize::<false>(image, window, actual, 80, 160, output, workspace)?;
    Ok(actual)
}
fn horizontal(
    image: ImageView<'_>,
    source_width: usize,
    y: usize,
    width: usize,
    offsets: &[usize],
    coeff: &[[i32; 2]],
    out: &mut [[i32; 3]],
) {
    for x in 0..width {
        let sx = offsets[x];
        let a = image.bgr(sx, y);
        let b = image.bgr((sx + 1).min(source_width - 1), y);
        for c in 0..3 {
            out[x][c] = a[c] as i32 * coeff[x][0] + b[c] as i32 * coeff[x][1];
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn resize<const DET: bool>(
    image: ImageView<'_>,
    source_width: usize,
    width: usize,
    height: usize,
    output_width: usize,
    output: &mut [f32],
    workspace: &mut ResizeWorkspace,
) -> ImageResult<()> {
    let plane = height
        .checked_mul(output_width)
        .ok_or_else(|| ImageError::new("resize output size overflow"))?;
    if width == 0
        || height == 0
        || width > output_width
        || plane.checked_mul(3) != Some(output.len())
    {
        return Err(ImageError::new("invalid resize output size"));
    }
    workspace.prepare(source_width, width)?;
    // Resolve the once-initialized tables once per resize, outside pixel loops.
    let det_values = if DET { Some(det_lut()) } else { None };
    let cls_values = if DET { None } else { Some(cls_lut()) };
    // Cached rows are scoped to this call: pixels can change while dimensions remain equal.
    let mut y0 = None;
    let mut y1 = None;
    for y in 0..height {
        let coordinate = (y as f64 + 0.5) * image.height() as f64 / height as f64 - 0.5;
        let sy = coordinate.floor() as i64;
        let fraction = coordinate - sy as f64;
        let beta0 = ((1. - fraction) * 2048.).round_ties_even() as i32;
        let beta1 = (fraction * 2048.).round_ties_even() as i32;
        let sy0 = sy.clamp(0, image.height() as i64 - 1) as usize;
        let sy1 = (sy + 1).clamp(0, image.height() as i64 - 1) as usize;
        if y0 != Some(sy0) {
            if y1 == Some(sy0) {
                std::mem::swap(&mut workspace.row0, &mut workspace.row1);
                std::mem::swap(&mut y0, &mut y1);
            } else {
                horizontal(
                    image,
                    source_width,
                    sy0,
                    width,
                    &workspace.x_offsets,
                    &workspace.x_coefficients,
                    &mut workspace.row0,
                );
                y0 = Some(sy0);
            }
        }
        if sy1 != sy0 && y1 != Some(sy1) {
            horizontal(
                image,
                source_width,
                sy1,
                width,
                &workspace.x_offsets,
                &workspace.x_coefficients,
                &mut workspace.row1,
            );
            y1 = Some(sy1);
        }
        for x in 0..width {
            for c in 0..3 {
                let h0 = workspace.row0[x][c];
                let h1 = if sy0 == sy1 { h0 } else { workspace.row1[x][c] };
                let value = (((((h0 >> 4) * beta0) >> 16) + (((h1 >> 4) * beta1) >> 16) + 2) >> 2)
                    .clamp(0, 255) as usize;
                output[c * plane + y * output_width + x] = if DET {
                    det_values.unwrap()[c][value]
                } else {
                    cls_values.unwrap()[value]
                };
            }
        }
    }
    Ok(())
}
