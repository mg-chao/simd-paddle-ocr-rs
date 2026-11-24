//! One source geometry for detector padding, resize bounds and box mapping.
use super::{DetResizePolicy, DetSize};
use crate::image::{ImageError, ImageResult};

const ALIGNMENT: usize = 32;
/// Maximum configurable resize limit.
pub const DET_MAX_SIDE: usize = 4096;
const MAX_LONG_SIDE: usize = 8192;
const MAX_MAP_PIXELS: usize = DET_MAX_SIDE * DET_MAX_SIDE;

pub(crate) fn valid_map_size(w: usize, h: usize) -> bool {
    w.max(h) <= MAX_LONG_SIDE && w.checked_mul(h).is_some_and(|n| n <= MAX_MAP_PIXELS)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DetSourceGeometry {
    pub original_width: usize,
    pub original_height: usize,
    pub padded_width: usize,
    pub padded_height: usize,
}

impl DetSourceGeometry {
    pub fn new(width: usize, height: usize) -> ImageResult<Self> {
        if width == 0 || height == 0 || width > i32::MAX as usize || height > i32::MAX as usize {
            return Err(ImageError::new("invalid detector image size"));
        }
        let (padded_width, padded_height) = if width + height < ALIGNMENT * 2 {
            (width.max(ALIGNMENT), height.max(ALIGNMENT))
        } else {
            (width, height)
        };
        Ok(Self {
            original_width: width,
            original_height: height,
            padded_width,
            padded_height,
        })
    }

    pub fn requires_padding(self) -> bool {
        self.original_width != self.padded_width || self.original_height != self.padded_height
    }

    pub fn resize(self, limit: usize, policy: DetResizePolicy) -> ImageResult<DetSize> {
        if !(ALIGNMENT..=DET_MAX_SIDE).contains(&limit) {
            return Err(ImageError::new("invalid detector resize limit"));
        }
        let width = self.padded_width;
        let height = self.padded_height;
        let side = match policy {
            DetResizePolicy::Max => width.max(height),
            DetResizePolicy::Min => width.min(height),
        };
        let requested_ratio = match policy {
            DetResizePolicy::Max if side > limit => limit as f64 / side as f64,
            DetResizePolicy::Min if side < limit => limit as f64 / side as f64,
            _ => 1.,
        };
        // The shared map limit takes precedence over enlarging the short side.
        // Bound the common scale before alignment, allocation and inference.
        let ratio = requested_ratio
            .min(MAX_LONG_SIDE as f64 / width.max(height) as f64)
            .min((MAX_MAP_PIXELS as f64 / (width as f64 * height as f64)).sqrt());
        let align = |side: usize| {
            let scaled = (side as f64 * ratio) as usize;
            (((scaled as f64 / ALIGNMENT as f64).round_ties_even() as usize) * ALIGNMENT)
                .max(ALIGNMENT)
        };
        let (mut resized_width, mut resized_height) = (align(width), align(height));
        if !valid_map_size(resized_width, resized_height) {
            let (long, short) = if width >= height {
                (&mut resized_width, resized_height)
            } else {
                (&mut resized_height, resized_width)
            };
            *long = MAX_MAP_PIXELS / short / ALIGNMENT * ALIGNMENT;
        }
        Ok(DetSize {
            width: resized_width,
            height: resized_height,
            width_ratio: (resized_width as f64 / width as f64) as f32,
            height_ratio: (resized_height as f64 / height as f64) as f32,
        })
    }

    pub fn map_point(self, point: [f32; 2], map_width: usize, map_height: usize) -> [f32; 2] {
        // Undo resizing in padded-source coordinates. Padding adds no scale or
        // translation; discard its excess only when clipping to the real image.
        [
            (point[0] / map_width as f32 * self.padded_width as f32)
                .round_ties_even()
                .clamp(0., (self.original_width - 1) as f32),
            (point[1] / map_height as f32 * self.padded_height as f32)
                .round_ties_even()
                .clamp(0., (self.original_height - 1) as f32),
        ]
    }
}
