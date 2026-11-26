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
    pub native_content: Option<[usize; 2]>,
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
            native_content: None,
        })
    }

    pub fn plan(mut self, limit: usize, policy: DetResizePolicy) -> ImageResult<(Self, DetSize)> {
        if !(ALIGNMENT..=DET_MAX_SIDE).contains(&limit) {
            return Err(ImageError::new("invalid detector resize limit"));
        }
        if policy != DetResizePolicy::Native {
            return Ok((self, self.legacy_size(limit, policy)));
        }

        let (width, height) = (self.original_width as u64, self.original_height as u64);
        let align = |side: u64| (side.div_ceil(ALIGNMENT as u64) * ALIGNMENT as u64) as usize;
        let (w, h) = (align(width), align(height));
        let [content_width, content_height] = if valid_map_size(w, h) {
            [self.original_width, self.original_height]
        } else {
            // Each aligned width has one largest feasible aligned height.
            // Compare the resulting rational scales exactly at padding boundaries.
            let mut best = (0, 1);
            for w in (ALIGNMENT..=MAX_LONG_SIDE).step_by(ALIGNMENT) {
                let h = MAX_LONG_SIDE.min(MAX_MAP_PIXELS / w) / ALIGNMENT * ALIGNMENT;
                let candidate = if w as u64 * height <= h as u64 * width {
                    (w as u64, width)
                } else {
                    (h as u64, height)
                };
                if candidate.0 * best.1 > best.0 * candidate.1 {
                    best = candidate;
                }
            }
            [
                (width * best.0).div_ceil(best.1) as usize,
                (height * best.0).div_ceil(best.1) as usize,
            ]
        };
        self.native_content = Some([content_width, content_height]);
        let size = DetSize {
            width: align(content_width as u64),
            height: align(content_height as u64),
            width_ratio: content_width as f32 / width as f32,
            height_ratio: content_height as f32 / height as f32,
        };
        debug_assert!(valid_map_size(size.width, size.height));
        Ok((self, size))
    }

    pub fn requires_padding(self) -> bool {
        self.original_width != self.padded_width || self.original_height != self.padded_height
    }

    pub fn resize(self, limit: usize, policy: DetResizePolicy) -> ImageResult<DetSize> {
        Ok(self.plan(limit, policy)?.1)
    }

    fn legacy_size(self, limit: usize, policy: DetResizePolicy) -> DetSize {
        let width = self.padded_width;
        let height = self.padded_height;
        let side = match policy {
            DetResizePolicy::Max => width.max(height),
            DetResizePolicy::Min => width.min(height),
            DetResizePolicy::Native => unreachable!(),
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
        DetSize {
            width: resized_width,
            height: resized_height,
            width_ratio: (resized_width as f64 / width as f64) as f32,
            height_ratio: (resized_height as f64 / height as f64) as f32,
        }
    }

    pub fn map_point(self, point: [f32; 2], map_width: usize, map_height: usize) -> [f32; 2] {
        let (denominator, numerator) = self.mapping(map_width, map_height);
        [
            (point[0] / denominator[0] as f32 * numerator[0] as f32)
                .round_ties_even()
                .clamp(0., (self.original_width - 1) as f32),
            (point[1] / denominator[1] as f32 * numerator[1] as f32)
                .round_ties_even()
                .clamp(0., (self.original_height - 1) as f32),
        ]
    }

    pub fn mapping(self, width: usize, height: usize) -> ([usize; 2], [usize; 2]) {
        match self.native_content {
            Some(content) => (content, [self.original_width, self.original_height]),
            None => ([width, height], [self.padded_width, self.padded_height]),
        }
    }
}
