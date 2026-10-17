//! Format-aware pixel and DB kernels. Every load stays inside a checked slice.
use crate::image::{ImageError, ImageResult, ImageView};
use std::simd::{
    Select, Simd,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::SimdFloat,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PixelKernel {
    Scalar,
    Portable,
    Avx,
    Avx2,
    Avx512,
}
impl PixelKernel {
    pub(crate) fn selected() -> Self {
        static SELECTED: std::sync::OnceLock<PixelKernel> = std::sync::OnceLock::new();
        *SELECTED.get_or_init(Self::detect)
    }
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx2")
            {
                return Self::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
            if std::arch::is_x86_feature_detected!("avx") {
                return Self::Avx;
            }
        }
        Self::Portable
    }
    pub(crate) fn map_lanes(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Portable => 2,
            Self::Avx | Self::Avx2 => 4,
            Self::Avx512 => 8,
        }
    }
    fn available(self) -> bool {
        match self {
            Self::Scalar | Self::Portable => true,
            #[cfg(target_arch = "x86_64")]
            Self::Avx => std::arch::is_x86_feature_detected!("avx"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx2")
            }
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }
    fn require(self) {
        assert!(
            self.available(),
            "pixel kernel {self:?} is unavailable on this CPU"
        );
    }
    #[cfg(test)]
    pub(crate) fn test_kernels() -> impl Iterator<Item = Self> {
        [
            Self::Scalar,
            Self::Portable,
            Self::Avx,
            Self::Avx2,
            Self::Avx512,
        ]
        .into_iter()
        .filter(|kernel| kernel.available())
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn horizontal(
    kernel: PixelKernel,
    image: ImageView<'_>,
    source_width: usize,
    y: usize,
    width: usize,
    offsets: &[usize],
    coefficients: &[[i32; 2]],
    destination: &mut [[i32; 3]],
) {
    kernel.require();
    assert!(source_width > 0 && source_width <= image.width());
    assert!(offsets.len() >= width && coefficients.len() >= width && destination.len() >= width);
    #[cfg(target_arch = "x86_64")]
    if matches!(kernel, PixelKernel::Avx2 | PixelKernel::Avx512) {
        let bpp = image.format().bytes_per_pixel();
        let row = &image.pixels()[y * image.stride()..y * image.stride() + source_width * bpp];
        // Gather indices are signed byte offsets. Very large rows use the safe
        // portable path rather than truncating an address to I32.
        if row.len() <= i32::MAX as usize && row.len() >= 4 {
            // SAFETY: selected/tested kernel implies the required ISA. Row
            // accesses are independently checked/masked by gather_pixels().
            unsafe {
                if kernel == PixelKernel::Avx512 {
                    horizontal_avx512(
                        image,
                        source_width,
                        y,
                        width,
                        offsets,
                        coefficients,
                        destination,
                    );
                } else {
                    horizontal_avx2(
                        image,
                        source_width,
                        y,
                        width,
                        offsets,
                        coefficients,
                        destination,
                    );
                }
            }
            return;
        }
    }
    if kernel == PixelKernel::Scalar {
        horizontal_scalar(
            image,
            source_width,
            y,
            width,
            offsets,
            coefficients,
            destination,
        );
    } else {
        horizontal_portable::<4>(
            image,
            source_width,
            y,
            width,
            offsets,
            coefficients,
            destination,
            false,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn horizontal_scalar(
    image: ImageView<'_>,
    source_width: usize,
    y: usize,
    width: usize,
    offsets: &[usize],
    coefficients: &[[i32; 2]],
    destination: &mut [[i32; 3]],
) {
    for x in 0..width {
        let first = image.bgr(offsets[x], y);
        let second = image.bgr((offsets[x] + 1).min(source_width - 1), y);
        for channel in 0..3 {
            destination[x][channel] = i32::from(first[channel]) * coefficients[x][0]
                + i32::from(second[channel]) * coefficients[x][1];
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn horizontal_portable<const N: usize>(
    image: ImageView<'_>,
    source_width: usize,
    y: usize,
    width: usize,
    offsets: &[usize],
    coefficients: &[[i32; 2]],
    destination: &mut [[i32; 3]],
    gather: bool,
) {
    let bpp = image.format().bytes_per_pixel();
    let row = &image.pixels()[y * image.stride()..y * image.stride() + source_width * bpp];
    let end = width / N * N;
    for x in (0..end).step_by(N) {
        let left: [usize; N] = std::array::from_fn(|lane| offsets[x + lane]);
        let right: [usize; N] = left.map(|sx| (sx + 1).min(source_width - 1));
        let pixels = |positions: [usize; N]| -> [[i32; N]; 3] {
            #[cfg(target_arch = "x86_64")]
            if gather {
                let mut packed = [0u32; N];
                for first in (0..N).step_by(8) {
                    let positions = std::array::from_fn(|lane| positions[first + lane] * bpp);
                    // SAFETY: gather is enabled only in AVX2/AVX512 wrappers;
                    // N is 8 or 16, and gather_pixels masks every short load.
                    let words = unsafe { gather_pixels(row, positions) };
                    packed[first..first + 8].copy_from_slice(&words);
                }
                let shifts = if image.format().swap_red_blue() {
                    [16, 8, 0]
                } else {
                    [0, 8, 16]
                };
                return shifts.map(|shift| packed.map(|word| ((word >> shift) & 255) as i32));
            }
            std::array::from_fn(|channel| positions.map(|sx| i32::from(image.bgr(sx, y)[channel])))
        };
        let a = pixels(left).map(Simd::from_array);
        let b = pixels(right).map(Simd::from_array);
        let c0 = Simd::<i32, N>::from_array(std::array::from_fn(|lane| coefficients[x + lane][0]));
        let c1 = Simd::<i32, N>::from_array(std::array::from_fn(|lane| coefficients[x + lane][1]));
        let values: [[i32; N]; 3] =
            std::array::from_fn(|channel| (a[channel] * c0 + b[channel] * c1).to_array());
        for lane in 0..N {
            destination[x + lane] = values.map(|channel| channel[lane]);
        }
    }
    horizontal_scalar(
        image,
        source_width,
        y,
        width - end,
        &offsets[end..],
        &coefficients[end..],
        &mut destination[end..],
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gather_pixels(row: &[u8], positions: [usize; 8]) -> [u32; 8] {
    use std::arch::x86_64::*;
    let indices = positions.map(|offset| {
        // Validate the three meaningful bytes even for the masked last RGB24
        // pixel. Such a pixel must never issue a four-byte gather past the row.
        let _ = &row[offset..offset + 3];
        if offset + 4 <= row.len() {
            offset as i32
        } else {
            0
        }
    });
    let mut result = [0u32; 8];
    // SAFETY: row has >=4 bytes; every gathered index has four bytes within row.
    // The unaligned intrinsic accesses no padding outside that checked slice.
    unsafe {
        let indices = _mm256_loadu_si256(indices.as_ptr().cast());
        let pixels = _mm256_i32gather_epi32::<1>(row.as_ptr().cast(), indices);
        _mm256_storeu_si256(result.as_mut_ptr().cast(), pixels);
    }
    for (value, offset) in result.iter_mut().zip(positions) {
        if offset + 4 > row.len() {
            *value = u32::from(row[offset])
                | (u32::from(row[offset + 1]) << 8)
                | (u32::from(row[offset + 2]) << 16);
        }
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn horizontal_avx2(
    image: ImageView<'_>,
    source: usize,
    y: usize,
    width: usize,
    offsets: &[usize],
    coefficients: &[[i32; 2]],
    out: &mut [[i32; 3]],
) {
    horizontal_portable::<8>(image, source, y, width, offsets, coefficients, out, true);
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn horizontal_avx512(
    image: ImageView<'_>,
    source: usize,
    y: usize,
    width: usize,
    offsets: &[usize],
    coefficients: &[[i32; 2]],
    out: &mut [[i32; 3]],
) {
    horizontal_portable::<16>(image, source, y, width, offsets, coefficients, out, true);
}

pub(crate) fn map_coordinates(
    kernel: PixelKernel,
    coefficients: [f64; 8],
    mut x: [f64; 8],
    mut y: [f64; 8],
    width: usize,
    height: usize,
    count: usize,
) -> ImageResult<[[f64; 8]; 2]> {
    kernel.require();
    assert!(count <= kernel.map_lanes());
    let mut out = [[0.; 8]; 2];
    if count == 0 {
        return Ok(out);
    }
    // Inactive tail lanes repeat an actual point so their validation cannot
    // create an error absent from the requested output band.
    for lane in count..8 {
        x[lane] = x[0];
        y[lane] = y[0];
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: kernel selected or validated by the caller before dispatch.
    unsafe {
        if kernel == PixelKernel::Avx512 {
            map_avx512(coefficients, x, y, width, height, &mut out)?;
            return Ok(out);
        }
        if matches!(kernel, PixelKernel::Avx | PixelKernel::Avx2) {
            map_avx(coefficients, x, y, width, height, &mut out)?;
            return Ok(out);
        }
    }
    if kernel == PixelKernel::Scalar {
        map_scalar(coefficients, x, y, width, height, count, &mut out)?;
    } else {
        map_vector::<2>(coefficients, x, y, width, height, &mut out)?;
    }
    Ok(out)
}

fn map_scalar(
    c: [f64; 8],
    x: [f64; 8],
    y: [f64; 8],
    width: usize,
    height: usize,
    count: usize,
    out: &mut [[f64; 8]; 2],
) -> ImageResult<()> {
    for lane in 0..count {
        let u = x[lane] / width as f64;
        let v = y[lane] / height as f64;
        let den = c[6] * u + c[7] * v + 1.;
        let sx = (c[0] * u + c[1] * v + c[2]) / den;
        let sy = (c[3] * u + c[4] * v + c[5]) / den;
        validate_coordinate(den, sx, sy)?;
        out[0][lane] = sx;
        out[1][lane] = sy;
    }
    Ok(())
}
fn validate_coordinate(den: f64, x: f64, y: f64) -> ImageResult<()> {
    if !den.is_finite() || den.abs() <= 1e-12 {
        return Err(ImageError::new("invalid perspective denominator"));
    }
    if !x.is_finite()
        || !y.is_finite()
        || x < i32::MIN as f64
        || x > i32::MAX as f64 - 2.
        || y < i32::MIN as f64
        || y > i32::MAX as f64 - 2.
    {
        return Err(ImageError::new("perspective coordinate is out of range"));
    }
    Ok(())
}
#[inline(always)]
fn map_vector<const N: usize>(
    c: [f64; 8],
    x: [f64; 8],
    y: [f64; 8],
    width: usize,
    height: usize,
    out: &mut [[f64; 8]; 2],
) -> ImageResult<()> {
    let u = Simd::<f64, N>::from_slice(&x[..N]) / Simd::splat(width as f64);
    let v = Simd::<f64, N>::from_slice(&y[..N]) / Simd::splat(height as f64);
    let den = Simd::splat(c[6]) * u + Simd::splat(c[7]) * v + Simd::splat(1.);
    let sx = (Simd::splat(c[0]) * u + Simd::splat(c[1]) * v + Simd::splat(c[2])) / den;
    let sy = (Simd::splat(c[3]) * u + Simd::splat(c[4]) * v + Simd::splat(c[5])) / den;
    let (den, sx, sy) = (den.to_array(), sx.to_array(), sy.to_array());
    for lane in 0..N {
        validate_coordinate(den[lane], sx[lane], sy[lane])?;
        out[0][lane] = sx[lane];
        out[1][lane] = sy[lane];
    }
    Ok(())
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn map_avx(
    c: [f64; 8],
    x: [f64; 8],
    y: [f64; 8],
    width: usize,
    height: usize,
    out: &mut [[f64; 8]; 2],
) -> ImageResult<()> {
    map_vector::<4>(c, x, y, width, height, out)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn map_avx512(
    c: [f64; 8],
    x: [f64; 8],
    y: [f64; 8],
    width: usize,
    height: usize,
    out: &mut [[f64; 8]; 2],
) -> ImageResult<()> {
    map_vector::<8>(c, x, y, width, height, out)
}

pub(crate) fn sample_interior(
    kernel: PixelKernel,
    image: ImageView<'_>,
    x: f64,
    y: f64,
) -> Option<[u8; 3]> {
    kernel.require();
    let xb = x.floor() as i64;
    let yb = y.floor() as i64;
    if kernel == PixelKernel::Scalar
        || xb < 1
        || xb >= image.width() as i64 - 3
        || yb < 1
        || yb >= image.height() as i64 - 2
    {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: dispatcher has checked the required ISA and all 4x4 pixels are in bounds.
    unsafe {
        if kernel == PixelKernel::Avx512 {
            return Some(sample_avx512(image, x, y, xb, yb));
        }
        if matches!(kernel, PixelKernel::Avx | PixelKernel::Avx2) {
            return Some(sample_avx(image, x, y, xb, yb));
        }
    }
    Some(sample_vector::<4>(image, x, y, xb, yb))
}
#[inline(always)]
fn cubic_vector<const N: usize>(value: Simd<f64, N>) -> Simd<f64, N> {
    let t = value.abs();
    let inner = Simd::splat(1.25) * t * t * t - Simd::splat(2.25) * t * t + Simd::splat(1.);
    let outer = Simd::splat(-0.75) * t * t * t - Simd::splat(-3.75) * t * t + Simd::splat(-6.) * t
        - Simd::splat(-3.);
    t.simd_le(Simd::splat(1.)).select(
        inner,
        t.simd_lt(Simd::splat(2.)).select(outer, Simd::splat(0.)),
    )
}
#[inline(always)]
fn sample_vector<const N: usize>(
    image: ImageView<'_>,
    x: f64,
    y: f64,
    xb: i64,
    yb: i64,
) -> [u8; 3] {
    let base =
        Simd::<f64, N>::from_array(std::array::from_fn(|lane| (xb + lane as i64 - 1) as f64));
    let wx = cubic_vector(Simd::splat(x) - base).to_array();
    let base =
        Simd::<f64, N>::from_array(std::array::from_fn(|lane| (yb + lane as i64 - 1) as f64));
    let wy = cubic_vector(Simd::splat(y) - base).to_array();
    let mut acc = Simd::<f64, N>::splat(0.);
    for (ky, &wy) in wy[..4].iter().enumerate() {
        for (kx, &wx) in wx[..4].iter().enumerate() {
            let pixel = image.bgr((xb + kx as i64 - 1) as usize, (yb + ky as i64 - 1) as usize);
            let pixel = Simd::<f64, N>::from_array(std::array::from_fn(|lane| {
                pixel.get(lane).map_or(0., |&v| f64::from(v))
            }));
            acc += pixel * Simd::splat(wx) * Simd::splat(wy);
        }
    }
    std::array::from_fn(|channel| clamp_byte(acc[channel]))
}
pub(crate) fn clamp_byte(value: f64) -> u8 {
    if value <= 0. {
        0
    } else if value >= 255. {
        255
    } else {
        (value + 0.5).floor() as u8
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sample_avx(image: ImageView<'_>, x: f64, y: f64, xb: i64, yb: i64) -> [u8; 3] {
    sample_vector::<4>(image, x, y, xb, yb)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sample_avx512(image: ImageView<'_>, x: f64, y: f64, xb: i64, yb: i64) -> [u8; 3] {
    sample_vector::<8>(image, x, y, xb, yb)
}

pub(crate) fn binarize(
    kernel: PixelKernel,
    prediction: &[f32],
    bitmap: &mut [u8],
    threshold: f32,
) -> ImageResult<bool> {
    kernel.require();
    assert!(bitmap.len() >= prediction.len());
    #[cfg(target_arch = "x86_64")]
    // SAFETY: each variant is selected only on hardware supporting its ISA.
    unsafe {
        if kernel == PixelKernel::Avx512 {
            return binarize_avx512(prediction, bitmap, threshold);
        }
        if matches!(kernel, PixelKernel::Avx | PixelKernel::Avx2) {
            return binarize_avx(prediction, bitmap, threshold);
        }
    }
    if kernel == PixelKernel::Scalar {
        binarize_scalar(prediction, bitmap, threshold)
    } else {
        #[cfg(target_arch = "aarch64")]
        {
            binarize_portable(prediction, bitmap, threshold)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            binarize_vector::<4>(prediction, bitmap, threshold)
        }
    }
}
fn binarize_scalar(prediction: &[f32], bitmap: &mut [u8], threshold: f32) -> ImageResult<bool> {
    let mut any = false;
    for (&value, output) in prediction.iter().zip(bitmap) {
        if !value.is_finite() {
            return Err(ImageError::new(
                "detector output contains a nonfinite value",
            ));
        }
        *output = u8::from(value > threshold);
        any |= *output != 0;
    }
    Ok(any)
}
// Each index expands eight comparison bits into eight 0/1 bytes. Byte arrays
// keep the lane order independent of the machine's integer endianness.
#[cfg(test)]
const BITMAP_MASK_EXPAND: [[u8; 8]; 256] = {
    let mut table = [[0; 8]; 256];
    let mut mask = 0;
    while mask < table.len() {
        let mut lane = 0;
        while lane < 8 {
            table[mask][lane] = ((mask >> lane) & 1) as u8;
            lane += 1;
        }
        mask += 1;
    }
    table
};

#[cfg(test)]
#[inline(always)]
fn store_bitmap_mask<const N: usize>(bitmap: &mut [u8], mask: u64) {
    // The concrete callers use 4, 8, or 16 lanes. The four-lane form is also
    // useful for comparing this lookup path with portable byte-mask stores.
    if N == 4 {
        bitmap[..4].copy_from_slice(&BITMAP_MASK_EXPAND[(mask & 15) as usize][..4]);
    } else {
        for offset in (0..N).step_by(8) {
            let bytes = &BITMAP_MASK_EXPAND[((mask >> offset) & 255) as usize];
            bitmap[offset..offset + 8].copy_from_slice(bytes);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn binarize_portable(prediction: &[f32], bitmap: &mut [u8], threshold: f32) -> ImageResult<bool> {
    let end = prediction.len() / 4 * 4;
    let mut any = std::simd::Mask::<i32, 4>::splat(false);
    let mut invalid = std::simd::Mask::<i32, 4>::splat(false);
    for offset in (0..end).step_by(4) {
        let values = Simd::<f32, 4>::from_slice(&prediction[offset..offset + 4]);
        invalid |= !values.is_finite();
        let mask = values.simd_gt(Simd::splat(threshold));
        any |= mask;
        // NEON can narrow four comparison lanes directly to bytes. Retaining
        // masks in SIMD registers also avoids a scalar bitmask extraction on
        // every group of four detector pixels.
        mask.cast::<i8>()
            .select(Simd::<u8, 4>::splat(1), Simd::splat(0))
            .copy_to_slice(&mut bitmap[offset..offset + 4]);
    }
    if invalid.any() {
        return Err(ImageError::new(
            "detector output contains a nonfinite value",
        ));
    }
    Ok(binarize_scalar(&prediction[end..], &mut bitmap[end..], threshold)? || any.any())
}

#[cfg(test)]
#[inline(always)]
fn binarize_packed_vector<const N: usize>(
    prediction: &[f32],
    bitmap: &mut [u8],
    threshold: f32,
) -> ImageResult<bool> {
    let end = prediction.len() / N * N;
    let mut any = false;
    let mut invalid = std::simd::Mask::<i32, N>::splat(false);
    for offset in (0..end).step_by(N) {
        let values = Simd::<f32, N>::from_slice(&prediction[offset..offset + N]);
        invalid |= !values.is_finite();
        let mask = values.simd_gt(Simd::splat(threshold)).to_bitmask();
        any |= mask != 0;
        store_bitmap_mask::<N>(&mut bitmap[offset..offset + N], mask);
    }
    // Keep validity checks in SIMD registers throughout the map traversal.
    if invalid.any() {
        return Err(ImageError::new(
            "detector output contains a nonfinite value",
        ));
    }
    Ok(binarize_scalar(&prediction[end..], &mut bitmap[end..], threshold)? || any)
}
// Preserve the established implementations on CPUs not measured by the
// ARM64 optimization diagnostic.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn binarize_vector<const N: usize>(
    prediction: &[f32],
    bitmap: &mut [u8],
    threshold: f32,
) -> ImageResult<bool> {
    let end = prediction.len() / N * N;
    let mut any = false;
    let mut invalid = false;
    for offset in (0..end).step_by(N) {
        let values = Simd::<f32, N>::from_slice(&prediction[offset..offset + N]);
        invalid |= (!values.is_finite()).any();
        let mask = values.simd_gt(Simd::splat(threshold)).to_bitmask();
        any |= mask != 0;
        for lane in 0..N {
            bitmap[offset + lane] = ((mask >> lane) & 1) as u8;
        }
    }
    if invalid {
        return Err(ImageError::new(
            "detector output contains a nonfinite value",
        ));
    }
    Ok(binarize_scalar(&prediction[end..], &mut bitmap[end..], threshold)? || any)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn binarize_avx(p: &[f32], b: &mut [u8], t: f32) -> ImageResult<bool> {
    binarize_vector::<8>(p, b, t)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn binarize_avx512(p: &[f32], b: &mut [u8], t: f32) -> ImageResult<bool> {
    binarize_vector::<16>(p, b, t)
}

pub(crate) fn interior(
    kernel: PixelKernel,
    bitmap: &[u8],
    width: usize,
    height: usize,
    output: &mut [u8],
) {
    kernel.require();
    assert_eq!(bitmap.len(), width * height);
    assert!(output.len() >= bitmap.len());
    output[..bitmap.len()].fill(0);
    if width < 3 || height < 3 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if matches!(kernel, PixelKernel::Avx2 | PixelKernel::Avx512) {
        // SAFETY: both policies require AVX2; loads are bounded to row interiors.
        unsafe {
            interior_avx2(bitmap, width, height, output);
        }
        return;
    }
    interior_vector::<16>(bitmap, width, height, output);
}
#[inline(always)]
fn interior_vector<const N: usize>(bitmap: &[u8], width: usize, height: usize, output: &mut [u8]) {
    for y in 1..height - 1 {
        let row = y * width;
        let mut x = 1;
        while x + N < width {
            let mut values = Simd::<u8, N>::splat(1);
            for dy in [-1isize, 0, 1] {
                for dx in [-1isize, 0, 1] {
                    if dx != 0 || dy != 0 {
                        let start = ((row + x) as isize + dy * width as isize + dx) as usize;
                        values &= Simd::from_slice(&bitmap[start..start + N]);
                    }
                }
            }
            output[row + x..row + x + N].copy_from_slice(&values.to_array());
            x += N;
        }
        for x in x..width - 1 {
            output[row + x] = u8::from([-1isize, 0, 1].into_iter().all(|dy| {
                [-1isize, 0, 1].into_iter().all(|dx| {
                    (dx == 0 && dy == 0)
                        || bitmap[((row + x) as isize + dy * width as isize + dx) as usize] != 0
                })
            }));
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interior_avx2(b: &[u8], w: usize, h: usize, o: &mut [u8]) {
    interior_vector::<32>(b, w, h, o)
}

pub(crate) fn scan(kernel: PixelKernel, bytes: &[u8], nonzero: bool, reverse: bool) -> usize {
    kernel.require();
    #[cfg(target_arch = "x86_64")]
    if matches!(kernel, PixelKernel::Avx2 | PixelKernel::Avx512) {
        // SAFETY: both policies require AVX2; every vector is a complete slice.
        return unsafe { scan_avx2(bytes, nonzero, reverse) };
    }
    if kernel == PixelKernel::Scalar {
        scan_scalar(bytes, nonzero, reverse)
    } else {
        scan_vector::<16>(bytes, nonzero, reverse)
    }
}
fn scan_scalar(bytes: &[u8], nonzero: bool, reverse: bool) -> usize {
    if reverse {
        bytes
            .iter()
            .rposition(|&v| (v != 0) == nonzero)
            .unwrap_or(bytes.len())
    } else {
        bytes
            .iter()
            .position(|&v| (v != 0) == nonzero)
            .unwrap_or(bytes.len())
    }
}

#[inline(always)]
fn scan_vector<const N: usize>(bytes: &[u8], nonzero: bool, reverse: bool) -> usize {
    let end = bytes.len() / N * N;
    let mask = |offset: usize| {
        let values = Simd::<u8, N>::from_slice(&bytes[offset..offset + N]);
        if nonzero {
            values.simd_ne(Simd::splat(0)).to_bitmask()
        } else {
            values.simd_eq(Simd::splat(0)).to_bitmask()
        }
    };
    if reverse {
        let tail = scan_scalar(&bytes[end..], nonzero, true);
        if tail < bytes.len() - end {
            return end + tail;
        }
        for offset in (0..end).step_by(N).rev() {
            let bits = mask(offset);
            if bits != 0 {
                return offset + (63 - bits.leading_zeros()) as usize;
            }
        }
    } else {
        for offset in (0..end).step_by(N) {
            let bits = mask(offset);
            if bits != 0 {
                return offset + bits.trailing_zeros() as usize;
            }
        }
        let tail = scan_scalar(&bytes[end..], nonzero, false);
        if tail < bytes.len() - end {
            return end + tail;
        }
    }
    bytes.len()
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2(b: &[u8], nonzero: bool, reverse: bool) -> usize {
    scan_vector::<32>(b, nonzero, reverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::PixelFormat;

    fn image_bytes(format: PixelFormat, width: usize, height: usize) -> (Vec<u8>, usize) {
        let bpp = format.bytes_per_pixel();
        let stride = width * bpp + 5;
        // The final row has no accessible stride padding.
        let mut bytes = vec![197; (height - 1) * stride + width * bpp];
        for y in 0..height {
            for x in 0..width {
                let pixel = [
                    (x * 17 + y * 11) as u8,
                    (x * 3 + y * 29) as u8,
                    (x * 43 + y * 7) as u8,
                ];
                let at = y * stride + x * bpp;
                let pixel = if format.swap_red_blue() {
                    [pixel[2], pixel[1], pixel[0]]
                } else {
                    pixel
                };
                bytes[at..at + 3].copy_from_slice(&pixel);
                if bpp == 4 {
                    bytes[at + 3] = (x + y * 3) as u8;
                }
            }
        }
        (bytes, stride)
    }

    #[test]
    fn horizontal_gathers_match_scalar_for_all_formats_tails_and_exact_row_ends() {
        for format in [
            PixelFormat::Bgr24,
            PixelFormat::Rgb24,
            PixelFormat::Bgra32,
            PixelFormat::Rgba32,
        ] {
            for source in [1, 2, 7, 8, 9, 17] {
                let (pixels, stride) = image_bytes(format, source, 3);
                let image = ImageView::new(&pixels, source, 3, stride, format).unwrap();
                for width in [1, 3, 4, 7, 8, 15, 16, 17, 31] {
                    let offsets: Vec<_> = (0..width)
                        .map(|x| {
                            if x % 3 == 0 {
                                source - 1
                            } else {
                                x * source / width
                            }
                        })
                        .collect();
                    let coefficients: Vec<_> = (0..width)
                        .map(|x| {
                            let first = (x * 137 % 2049) as i32;
                            [first, 2048 - first]
                        })
                        .collect();
                    let mut expected = vec![[0; 3]; width];
                    horizontal(
                        PixelKernel::Scalar,
                        image,
                        source,
                        2,
                        width,
                        &offsets,
                        &coefficients,
                        &mut expected,
                    );
                    for kernel in PixelKernel::test_kernels() {
                        let mut actual = vec![[991; 3]; width + 1];
                        horizontal(
                            kernel,
                            image,
                            source,
                            2,
                            width,
                            &offsets,
                            &coefficients,
                            &mut actual,
                        );
                        assert_eq!(
                            &actual[..width],
                            expected,
                            "{kernel:?}/{format:?}/{source}/{width}"
                        );
                        assert_eq!(actual[width], [991; 3]);
                    }
                }
            }
        }
    }

    #[test]
    fn mapping_preserves_separate_f64_operations_and_rejects_poles() {
        let c = [7.25, -0.125, 1.75, 0.25, 11.5, -2.125, 0.0625, -0.03125];
        let xs = [0., 1., 2., 3., 4., 5., 6., 7.];
        let ys = [9., 8., 7., 6., 5., 4., 3., 2.];
        for kernel in PixelKernel::test_kernels() {
            for count in 1..=kernel.map_lanes() {
                let actual = map_coordinates(kernel, c, xs, ys, 11, 13, count).unwrap();
                for lane in 0..count {
                    let u = xs[lane] / 11.;
                    let v = ys[lane] / 13.;
                    let denominator = c[6] * u + c[7] * v + 1.;
                    let expected = [
                        (c[0] * u + c[1] * v + c[2]) / denominator,
                        (c[3] * u + c[4] * v + c[5]) / denominator,
                    ];
                    assert_eq!(
                        actual[0][lane].to_bits(),
                        expected[0].to_bits(),
                        "{kernel:?}"
                    );
                    assert_eq!(
                        actual[1][lane].to_bits(),
                        expected[1].to_bits(),
                        "{kernel:?}"
                    );
                }
            }
            let pole = [1., 0., 0., 0., 1., 0., -1., 0.];
            assert!(map_coordinates(kernel, pole, [1.; 8], [0.; 8], 1, 1, 1).is_err());
            let identity = [8., 0., 0.25, 0., 8., 0.5, 0., 0.];
            let output = map_coordinates(kernel, identity, xs, ys, 8, 8, 1).unwrap();
            assert_eq!([output[0][0], output[1][0]], [0.25, 9.5]);
        }
    }

    #[test]
    fn interior_cubic_matches_independent_ordered_scalar_samples() {
        let cubic = |value: f64| {
            let t = value.abs();
            let a = -0.75;
            if t <= 1. {
                (a + 2.) * t * t * t - (a + 3.) * t * t + 1.
            } else if t < 2. {
                a * t * t * t - 5. * a * t * t + 8. * a * t - 4. * a
            } else {
                0.
            }
        };
        for format in [
            PixelFormat::Bgr24,
            PixelFormat::Rgb24,
            PixelFormat::Bgra32,
            PixelFormat::Rgba32,
        ] {
            let (pixels, stride) = image_bytes(format, 17, 13);
            let image = ImageView::new(&pixels, 17, 13, stride, format).unwrap();
            for kernel in
                PixelKernel::test_kernels().filter(|&kernel| kernel != PixelKernel::Scalar)
            {
                for i in 0..97 {
                    let x = 1. + (i * 37 % 1199) as f64 / 100.;
                    let y = 1. + (i * 17 % 899) as f64 / 100.;
                    let xb = x.floor() as i64;
                    let yb = y.floor() as i64;
                    let mut expected = [0.; 3];
                    for ky in 0..4 {
                        for kx in 0..4 {
                            let px = (xb + kx - 1) as usize;
                            let py = (yb + ky - 1) as usize;
                            let wx = cubic(x - (xb + kx - 1) as f64);
                            let wy = cubic(y - (yb + ky - 1) as f64);
                            let pixel = image.bgr(px, py);
                            for channel in 0..3 {
                                expected[channel] += f64::from(pixel[channel]) * wx * wy;
                            }
                        }
                    }
                    let expected =
                        expected.map(|value| (value + 0.5).floor().clamp(0., 255.) as u8);
                    assert_eq!(
                        sample_interior(kernel, image, x, y),
                        Some(expected),
                        "{kernel:?}/{format:?}/{x}/{y}"
                    );
                }
                assert!(sample_interior(kernel, image, 0.9, 2.).is_none());
                assert!(sample_interior(kernel, image, 14., 2.).is_none());
            }
        }
    }

    #[test]
    fn threshold_is_strict_finite_and_tail_bounded() {
        for kernel in PixelKernel::test_kernels() {
            for length in 0..70 {
                let choices = [
                    -1.,
                    -0.,
                    0.,
                    f32::from_bits(1),
                    0.3,
                    f32::from_bits(0.3f32.to_bits() + 1),
                    1.,
                ];
                let mut values: Vec<_> = (0..length).map(|i| choices[i % choices.len()]).collect();
                for threshold in [-0., 0.3] {
                    let expected: Vec<_> =
                        values.iter().map(|&v| u8::from(v > threshold)).collect();
                    let mut actual = vec![199; length + 3];
                    assert_eq!(
                        binarize(kernel, &values, &mut actual, threshold).unwrap(),
                        expected.contains(&1)
                    );
                    assert_eq!(&actual[..length], expected);
                    assert_eq!(&actual[length..], [199; 3]);
                }
                for index in 0..length {
                    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                        let previous = values[index];
                        values[index] = invalid;
                        let mut bitmap = vec![0; length];
                        assert!(
                            binarize(kernel, &values, &mut bitmap, 0.3).is_err(),
                            "{kernel:?}/{length}/{index}"
                        );
                        values[index] = previous;
                        assert!(binarize(kernel, &values, &mut bitmap, 0.3).is_ok());
                    }
                }
            }
        }
    }

    #[test]
    fn interior_cache_and_bidirectional_run_scans_match_neighbor_and_iterator_oracles() {
        for width in [1, 2, 3, 17, 31, 32, 33, 34, 63, 64, 65] {
            for height in [1, 2, 3, 5] {
                let bitmap: Vec<_> = (0..width * height)
                    .map(|i| u8::from(i % 17 != 3 && i % 29 != 5))
                    .collect();
                let mut expected = vec![0; bitmap.len()];
                if width >= 3 && height >= 3 {
                    for y in 1..height - 1 {
                        for x in 1..width - 1 {
                            let mut surrounded = true;
                            for ny in y - 1..=y + 1 {
                                for nx in x - 1..=x + 1 {
                                    if nx != x || ny != y {
                                        surrounded &= bitmap[ny * width + nx] != 0;
                                    }
                                }
                            }
                            expected[y * width + x] = u8::from(surrounded);
                        }
                    }
                }
                for kernel in PixelKernel::test_kernels() {
                    let mut actual = vec![77; bitmap.len() + 3];
                    interior(kernel, &bitmap, width, height, &mut actual);
                    assert_eq!(
                        &actual[..bitmap.len()],
                        expected,
                        "{kernel:?}/{width}/{height}"
                    );
                    assert_eq!(&actual[bitmap.len()..], [77; 3]);
                }
            }
        }
        let bytes: Vec<_> = (0..145)
            .map(|i| {
                if i % 17 == 5 {
                    255
                } else if i % 13 == 3 {
                    1
                } else {
                    0
                }
            })
            .collect();
        for kernel in PixelKernel::test_kernels() {
            for offset in 0..7 {
                for length in 0..98 {
                    for nonzero in [false, true] {
                        let values = &bytes[offset..offset + length];
                        assert_eq!(
                            scan(kernel, values, nonzero, false),
                            values
                                .iter()
                                .position(|&v| (v != 0) == nonzero)
                                .unwrap_or(length)
                        );
                        assert_eq!(
                            scan(kernel, values, nonzero, true),
                            values
                                .iter()
                                .rposition(|&v| (v != 0) == nonzero)
                                .unwrap_or(length)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn forged_unsupported_policies_are_rejected_before_native_execution() {
        for kernel in [PixelKernel::Avx, PixelKernel::Avx2, PixelKernel::Avx512]
            .into_iter()
            .filter(|kernel| !kernel.available())
        {
            assert!(
                std::panic::catch_unwind(|| {
                    let mut out = [0; 16];
                    let _ = binarize(kernel, &[0.; 16], &mut out, 0.3);
                })
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(|| {
                    let _ = map_coordinates(kernel, [0.; 8], [0.; 8], [0.; 8], 1, 1, 1);
                })
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(|| {
                    let _ = scan(kernel, &[0; 32], true, false);
                })
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(|| {
                    let mut out = [0; 9];
                    interior(kernel, &[0; 9], 3, 3, &mut out);
                })
                .is_err()
            );
        }
    }
}

#[cfg(test)]
#[path = "pixel_kernels/threshold_tests.rs"]
mod threshold_tests;
