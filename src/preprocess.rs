//! Byte-domain OpenCV-compatible resize followed by canonical normalization.
use crate::image::{ImageError, ImageResult, ImageView, resize_vec};
use std::sync::OnceLock;

mod detector;
pub use detector::DET_MAX_SIDE;
pub(crate) use detector::DetSourceGeometry;

#[derive(Default, Debug)]
pub struct ResizeWorkspace {
    x_offsets: Vec<usize>,
    x_coefficients: Vec<[i32; 2]>,
    row0: Vec<[i32; 3]>,
    row1: Vec<[i32; 3]>,
    key: Option<(usize, usize)>,
    workers: Vec<ResizeWorkspace>,
}
impl ResizeWorkspace {
    pub fn retained_bytes(&self) -> usize {
        self.x_offsets.capacity() * std::mem::size_of::<usize>()
            + self.x_coefficients.capacity() * 8
            + (self.row0.capacity() + self.row1.capacity()) * 12
            + self.workers.capacity() * std::mem::size_of::<ResizeWorkspace>()
            + self.workers.iter().map(Self::retained_bytes).sum::<usize>()
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum DetResizePolicy {
    #[default]
    Max,
    /// Enlarge the short side up to the requested limit, subject to DET_MAX_SIDE.
    Min,
}

pub fn compute_det_size(width: usize, height: usize, limit: usize) -> ImageResult<DetSize> {
    compute_det_size_with_policy(width, height, limit, DetResizePolicy::Max)
}

pub fn compute_det_size_with_policy(
    width: usize,
    height: usize,
    limit: usize,
    policy: DetResizePolicy,
) -> ImageResult<DetSize> {
    DetSourceGeometry::new(width, height)?.resize(limit, policy)
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
/// The same byte-domain DET resize with disjoint output row bands. Scratch is
/// retained per worker; nested graph work uses the caller's same CPU context.
/// Positive budgets are capped to the context capacity and 16; zero stays serial.
#[allow(clippy::too_many_arguments)]
pub fn det_with_context(
    image: ImageView<'_>,
    width: usize,
    height: usize,
    output: &mut [f32],
    workspace: &mut ResizeWorkspace,
    cpu: &crate::cpu::CpuContext,
    budget: usize,
) -> ImageResult<()> {
    det_with_context_layout(image, width, height, output, workspace, cpu, budget, false)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn det_with_context_layout(
    image: ImageView<'_>,
    width: usize,
    height: usize,
    output: &mut [f32],
    workspace: &mut ResizeWorkspace,
    cpu: &crate::cpu::CpuContext,
    budget: usize,
    nhwc: bool,
) -> ImageResult<()> {
    let workers = crate::cpu::cap_stage_threads(budget, cpu.capacity()).min((height / 24).max(1));
    if workers <= 1 {
        return resize_with_layout::<true>(
            image,
            image.width(),
            width,
            height,
            width,
            output,
            workspace,
            nhwc,
        );
    }
    let plane = width
        .checked_mul(height)
        .ok_or_else(|| ImageError::new("DET plane size overflow"))?;
    if width == 0 || height == 0 || plane.checked_mul(3) != Some(output.len()) {
        return Err(ImageError::new("invalid DET output size"));
    }
    workspace
        .workers
        .try_reserve(workers.saturating_sub(workspace.workers.len()))
        .map_err(|e| {
            ImageError::caused_by(
                crate::error::ErrorKind::Allocation,
                "resize workers allocation",
                e,
            )
        })?;
    while workspace.workers.len() < workers {
        workspace.workers.push(ResizeWorkspace::default());
    }
    for scratch in &mut workspace.workers[..workers] {
        scratch.prepare(image.width(), width)?;
    }
    let (a, tail) = output.split_at_mut(if nhwc { output.len() } else { plane });
    let (b, c) = tail.split_at_mut(if nhwc { 0 } else { plane });
    let (mut a, mut b, mut c) = (a, b, c);
    let mut first = 0;
    let mut jobs = Vec::new();
    jobs.try_reserve_exact(workers).map_err(|e| {
        ImageError::caused_by(
            crate::error::ErrorKind::Allocation,
            "resize jobs allocation",
            e,
        )
    })?;
    for (index, scratch) in workspace.workers[..workers].iter_mut().enumerate() {
        let end = height * (index + 1) / workers;
        let bytes = (end - first) * width;
        let (pa, ta) = a.split_at_mut(bytes * if nhwc { 3 } else { 1 });
        a = ta;
        let (pb, tb) = b.split_at_mut(if nhwc { 0 } else { bytes });
        b = tb;
        let (pc, tc) = c.split_at_mut(if nhwc { 0 } else { bytes });
        c = tc;
        jobs.push(ResizeJob {
            planes: [pa, pb, pc],
            first,
            rows: end - first,
            scratch,
            error: None,
        });
        first = end;
    }
    cpu.for_each_mut(&mut jobs, workers, |_, job| {
        job.error = resize_planes::<true>(
            image,
            image.width(),
            width,
            height,
            width,
            job.first,
            job.rows,
            &mut job.planes,
            job.scratch,
            nhwc,
        )
        .err();
    });
    for job in jobs {
        if let Some(error) = job.error {
            return Err(error);
        }
    }
    Ok(())
}
struct ResizeJob<'a> {
    planes: [&'a mut [f32]; 3],
    first: usize,
    rows: usize,
    scratch: &'a mut ResizeWorkspace,
    error: Option<ImageError>,
}
pub fn cls(
    image: ImageView<'_>,
    output: &mut [f32],
    workspace: &mut ResizeWorkspace,
) -> ImageResult<usize> {
    cls_with_layout(image, output, workspace, false)
}
pub(crate) fn cls_with_layout(
    image: ImageView<'_>,
    output: &mut [f32],
    workspace: &mut ResizeWorkspace,
    nhwc: bool,
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
    resize_with_layout::<false>(image, window, actual, 80, 160, output, workspace, nhwc)?;
    Ok(actual)
}
/// Height-48 recognition resize. Right padding is normalized zero, rather
/// than CLS's normalized black; the complete source width is retained.
pub fn rec(
    image: ImageView<'_>,
    tensor_width: usize,
    output: &mut [f32],
    workspace: &mut ResizeWorkspace,
) -> ImageResult<usize> {
    rec_with_layout(image, tensor_width, output, workspace, false)
}
pub(crate) fn rec_with_layout(
    image: ImageView<'_>,
    tensor_width: usize,
    output: &mut [f32],
    workspace: &mut ResizeWorkspace,
    nhwc: bool,
) -> ImageResult<usize> {
    let elements = tensor_width.checked_mul(3 * 48);
    if tensor_width == 0 || elements != Some(output.len()) {
        return Err(ImageError::new("invalid REC output size"));
    }
    let natural = (48u64 * image.width() as u64).div_ceil(image.height() as u64);
    let actual = natural.min(tensor_width as u64) as usize;
    if actual < tensor_width {
        output.fill(0.);
    }
    resize_with_layout::<false>(
        image,
        image.width(),
        actual,
        48,
        tensor_width,
        output,
        workspace,
        nhwc,
    )?;
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
    crate::pixel_kernels::horizontal(
        crate::pixel_kernels::PixelKernel::selected(),
        image,
        source_width,
        y,
        width,
        offsets,
        coeff,
        out,
    );
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
    resize_with_layout::<DET>(
        image,
        source_width,
        width,
        height,
        output_width,
        output,
        workspace,
        false,
    )
}

#[cfg(all(test, feature = "gpu", not(target_os = "macos")))]
pub(crate) fn resize_for_gpu_test(
    image: ImageView<'_>,
    width: usize,
    height: usize,
    output_width: usize,
    output: &mut [f32],
    workspace: &mut ResizeWorkspace,
) -> ImageResult<()> {
    resize::<false>(
        image,
        image.width(),
        width,
        height,
        output_width,
        output,
        workspace,
    )
}
#[allow(clippy::too_many_arguments)]
fn resize_with_layout<const DET: bool>(
    image: ImageView<'_>,
    source_width: usize,
    width: usize,
    height: usize,
    output_width: usize,
    output: &mut [f32],
    workspace: &mut ResizeWorkspace,
    nhwc: bool,
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
    let (a, tail) = output.split_at_mut(if nhwc { output.len() } else { plane });
    let (b, c) = tail.split_at_mut(if nhwc { 0 } else { plane });
    resize_planes::<DET>(
        image,
        source_width,
        width,
        height,
        output_width,
        0,
        height,
        &mut [a, b, c],
        workspace,
        nhwc,
    )
}
#[allow(clippy::too_many_arguments)]
fn resize_planes<const DET: bool>(
    image: ImageView<'_>,
    source_width: usize,
    width: usize,
    height: usize,
    output_width: usize,
    first_row: usize,
    rows: usize,
    output: &mut [&mut [f32]; 3],
    workspace: &mut ResizeWorkspace,
    nhwc: bool,
) -> ImageResult<()> {
    workspace.prepare(source_width, width)?;
    // Resolve the once-initialized tables once per resize, outside pixel loops.
    let det_values = if DET { Some(det_lut()) } else { None };
    let cls_values = if DET { None } else { Some(cls_lut()) };
    // Cached rows are scoped to this call: pixels can change while dimensions remain equal.
    let mut y0 = None;
    let mut y1 = None;
    for local_y in 0..rows {
        let y = first_row + local_y;
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
                let value = if DET {
                    det_values.unwrap()[c][value]
                } else {
                    cls_values.unwrap()[value]
                };
                if nhwc {
                    output[0][(local_y * output_width + x) * 3 + c] = value;
                } else {
                    output[c][local_y * output_width + x] = value;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cpu::CpuContext, image::PixelFormat};
    #[test]
    fn detector_resize_bounds_all_policies_before_inference() {
        for (width, height) in [
            (192, 32),
            (32, 192),
            (4096, 736),
            (4097, 736),
            (5000, 1000),
            (1, i32::MAX as usize),
            (i32::MAX as usize, 1),
        ] {
            for limit in [32, 736, 960, 4096] {
                for policy in [DetResizePolicy::Max, DetResizePolicy::Min] {
                    let size = compute_det_size_with_policy(width, height, limit, policy).unwrap();
                    assert!(
                        (32..=4096).contains(&size.width),
                        "{width}x{height}/{policy:?}"
                    );
                    assert!(
                        (32..=4096).contains(&size.height),
                        "{width}x{height}/{policy:?}"
                    );
                    assert_eq!(size.width % 32, 0);
                    assert_eq!(size.height % 32, 0);
                }
            }
        }
    }

    #[test]
    fn detector_resize_policies_preserve_defaults_and_bound_dimensions() {
        assert_eq!(
            compute_det_size(640, 480, 960).unwrap(),
            compute_det_size_with_policy(640, 480, 960, DetResizePolicy::Max).unwrap()
        );
        let size = compute_det_size_with_policy(640, 320, 736, DetResizePolicy::Min).unwrap();
        assert_eq!((size.width, size.height), (1472, 736));
        let size = compute_det_size_with_policy(1920, 1080, 736, DetResizePolicy::Min).unwrap();
        assert_eq!((size.width, size.height), (1920, 1088));
        let size = compute_det_size_with_policy(1920, 1080, 960, DetResizePolicy::Max).unwrap();
        assert_eq!((size.width, size.height), (960, 544));
        assert!(compute_det_size_with_policy(0, 320, 736, DetResizePolicy::Min).is_err());
        assert!(compute_det_size_with_policy(640, 320, 0, DetResizePolicy::Min).is_err());
        assert!(
            compute_det_size_with_policy(640, 320, DET_MAX_SIDE + 1, DetResizePolicy::Max).is_err()
        );
    }
    #[test]
    fn explicit_stage_budget_caps_det_resize_workers_and_preserves_serial_pixels() {
        #[cfg(feature = "parallel")]
        let capacity = 32;
        #[cfg(not(feature = "parallel"))]
        let capacity = 1;
        let cpu = CpuContext::new(capacity).unwrap();
        let image = ImageView::new(&[3, 17, 251], 1, 1, 0, PixelFormat::Bgr24).unwrap();
        let (width, height) = (32, 768);
        let mut serial = vec![0.; width * height * 3];
        det(
            image,
            width,
            height,
            &mut serial,
            &mut ResizeWorkspace::default(),
        )
        .unwrap();
        let mut observed_workers = Vec::new();
        for requested in [0, 1, 17, usize::MAX] {
            let mut workspace = ResizeWorkspace::default();
            let mut output = vec![f32::NAN; serial.len()];
            det_with_context(
                image,
                width,
                height,
                &mut output,
                &mut workspace,
                &cpu,
                requested,
            )
            .unwrap();
            assert_eq!(output, serial);
            observed_workers.push(workspace.workers.len());
        }
        assert_eq!(
            observed_workers,
            if capacity > 1 {
                [0, 0, 16, 16]
            } else {
                [0, 0, 0, 0]
            }
        );
    }
    #[test]
    fn parallel_det_rows_match_fixedpoint_serial_with_stride_and_alpha() {
        let width = 37;
        let height = 53;
        let stride = width * 4 + 7;
        let mut bytes = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                let at = y * stride + x * 4;
                bytes[at..at + 4].copy_from_slice(&[
                    (x * 11 + y) as u8,
                    (x + y * 7) as u8,
                    (y * 13 + x * 3) as u8,
                    255,
                ]);
            }
        }
        let image = ImageView::new(&bytes, width, height, stride, PixelFormat::Rgba32).unwrap();
        let mut serial = vec![0.; 3 * 96 * 128];
        det(image, 96, 128, &mut serial, &mut ResizeWorkspace::default()).unwrap();
        #[cfg(feature = "parallel")]
        let cpu = CpuContext::new(3).unwrap();
        #[cfg(not(feature = "parallel"))]
        let cpu = CpuContext::new(1).unwrap();
        let mut parallel = vec![0.; serial.len()];
        let mut workspace = ResizeWorkspace::default();
        det_with_context(image, 96, 128, &mut parallel, &mut workspace, &cpu, 3).unwrap();
        assert_eq!(serial, parallel);
        let retained = workspace.retained_bytes();
        det_with_context(image, 96, 128, &mut parallel, &mut workspace, &cpu, 3).unwrap();
        assert_eq!(retained, workspace.retained_bytes());
        let mut nhwc = vec![0.; serial.len()];
        det_with_context_layout(image, 96, 128, &mut nhwc, &mut workspace, &cpu, 3, true).unwrap();
        for pixel in 0..96 * 128 {
            for channel in 0..3 {
                assert_eq!(
                    nhwc[pixel * 3 + channel],
                    serial[channel * 96 * 128 + pixel]
                );
            }
        }
        for rec in [false, true] {
            let (width, height) = if rec { (64, 48) } else { (160, 80) };
            let mut nchw = vec![0.; 3 * width * height];
            let mut nhwc = vec![0.; nchw.len()];
            if rec {
                rec_with_layout(image, width, &mut nchw, &mut workspace, false).unwrap();
                rec_with_layout(image, width, &mut nhwc, &mut workspace, true).unwrap();
            } else {
                cls_with_layout(image, &mut nchw, &mut workspace, false).unwrap();
                cls_with_layout(image, &mut nhwc, &mut workspace, true).unwrap();
            }
            for pixel in 0..width * height {
                for channel in 0..3 {
                    assert_eq!(
                        nhwc[pixel * 3 + channel],
                        nchw[channel * width * height + pixel]
                    );
                }
            }
        }
    }
}
