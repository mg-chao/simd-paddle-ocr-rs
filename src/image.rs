//! Checked borrowed interleaved images. Alpha is ignored by OCR processing.

crate::error::stage_error!(ImageError, InvalidInput);

impl From<crate::runtime::RuntimeError> for ImageError {
    fn from(value: crate::runtime::RuntimeError) -> Self {
        Self::caused_by(value.kind(), "runtime", value)
    }
}
pub type ImageResult<T> = Result<T, ImageError>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PixelFormat {
    #[default]
    Bgr24,
    Rgb24,
    Bgra32,
    Rgba32,
}
impl PixelFormat {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Bgr24 | Self::Rgb24 => 3,
            Self::Bgra32 | Self::Rgba32 => 4,
        }
    }
    pub(crate) fn swap_red_blue(self) -> bool {
        matches!(self, Self::Rgb24 | Self::Rgba32)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
    pixels: &'a [u8],
    width: usize,
    height: usize,
    stride: usize,
    format: PixelFormat,
}
impl<'a> ImageView<'a> {
    pub fn new(
        pixels: &'a [u8],
        width: usize,
        height: usize,
        stride: usize,
        format: PixelFormat,
    ) -> ImageResult<Self> {
        if width == 0 || height == 0 || width > i32::MAX as usize || height > i32::MAX as usize {
            return Err(ImageError::new(
                "image dimensions must be positive and fit I32",
            ));
        }
        let row = width
            .checked_mul(format.bytes_per_pixel())
            .ok_or_else(|| ImageError::new("image row byte count overflow"))?;
        let stride = if stride == 0 { row } else { stride };
        if stride < row {
            return Err(ImageError::new("image stride is too small"));
        }
        let required = (height - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(row))
            .ok_or_else(|| ImageError::new("image byte count overflow"))?;
        if required > pixels.len() {
            return Err(ImageError::new("image buffer is too small"));
        }
        Ok(Self {
            pixels,
            width,
            height,
            stride,
            format,
        })
    }
    pub fn pixels(self) -> &'a [u8] {
        self.pixels
    }
    pub fn width(self) -> usize {
        self.width
    }
    pub fn height(self) -> usize {
        self.height
    }
    pub fn stride(self) -> usize {
        self.stride
    }
    pub fn format(self) -> PixelFormat {
        self.format
    }
    pub fn pixel_count(self) -> ImageResult<usize> {
        self.width
            .checked_mul(self.height)
            .ok_or_else(|| ImageError::new("image pixel count overflow"))
    }
    pub(crate) fn bgr(self, x: usize, y: usize) -> [u8; 3] {
        let offset = y * self.stride + x * self.format.bytes_per_pixel();
        let p = &self.pixels[offset..offset + 3];
        if self.format.swap_red_blue() {
            [p[2], p[1], p[0]]
        } else {
            [p[0], p[1], p[2]]
        }
    }
}

pub(crate) fn resize_vec<T: Default + Clone>(
    values: &mut Vec<T>,
    length: usize,
) -> ImageResult<()> {
    if length
        .checked_mul(std::mem::size_of::<T>())
        .is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(ImageError::new(
            "workspace allocation exceeds addressable size",
        ));
    }
    if values.len() < length {
        values.try_reserve(length - values.len()).map_err(|e| {
            ImageError::caused_by(
                crate::error::ErrorKind::Allocation,
                "workspace allocation failed",
                e,
            )
        })?;
        values.resize(length, T::default());
    }
    Ok(())
}
pub(crate) fn push<T>(values: &mut Vec<T>, value: T) -> ImageResult<()> {
    if values.len() == values.capacity() {
        values.try_reserve(1).map_err(|e| {
            ImageError::caused_by(
                crate::error::ErrorKind::Allocation,
                "workspace allocation failed",
                e,
            )
        })?;
    }
    values.push(value);
    Ok(())
}
