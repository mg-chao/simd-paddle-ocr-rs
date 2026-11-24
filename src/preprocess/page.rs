//! Working image used for both detection and perspective crops.
use super::{ResizeWorkspace, resized_bytes};
use crate::image::{ImageError, ImageResult, ImageView, PixelFormat, resize_vec};

pub(crate) struct Page {
    pixels: Vec<u8>,
    pub width: usize,
    pub height: usize,
    top: usize,
}

impl Page {
    pub fn prepare(image: ImageView<'_>, max_pixels: usize) -> ImageResult<Self> {
        if image.pixel_count()? > max_pixels {
            return Err(ImageError::new("source image exceeds max_image_pixels"));
        }
        let (w, h) = (image.width(), image.height());
        let aligned = |n: usize, ratio: f64| {
            (((n as f64 * ratio).floor() / 32.).round_ties_even() as usize).max(1) * 32
        };
        let ratio = if w.max(h) > 2000 {
            2000. / w.max(h) as f64
        } else if w.min(h) < 30 {
            (30. / w.min(h) as f64).min(2000. / w.max(h) as f64)
        } else {
            1.
        };
        let (width, height) = if ratio == 1. {
            (w, h)
        } else {
            (aligned(w, ratio), aligned(h, ratio))
        };
        let top = if height <= 30 || width as u64 > 8 * height as u64 {
            (2 * (width / 8).max(30)).abs_diff(height) / 2
        } else {
            0
        };
        let mut page = Self {
            pixels: Vec::new(),
            width,
            height,
            top,
        };
        if (width, height) == (w, h) && top == 0 {
            return Ok(page);
        }
        let pixels = width
            .checked_mul(height + 2 * top)
            .filter(|n| *n <= max_pixels)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| ImageError::new("resized page exceeds max_image_pixels"))?;
        resize_vec(&mut page.pixels, pixels)?;
        resized_bytes(
            image,
            w,
            width,
            height,
            0,
            height,
            &mut ResizeWorkspace::default(),
            |y, x, c, value| page.pixels[((y + top) * width + x) * 3 + c] = value,
        )?;
        Ok(page)
    }

    pub fn transformed(&self) -> bool {
        !self.pixels.is_empty()
    }

    pub fn view<'a>(&'a self, original: ImageView<'a>) -> ImageResult<ImageView<'a>> {
        if !self.transformed() {
            return Ok(original);
        }
        ImageView::new(
            &self.pixels,
            self.width,
            self.height + 2 * self.top,
            0,
            PixelFormat::Bgr24,
        )
    }

    pub fn restore(&self, p: [f32; 2], original: ImageView<'_>) -> [f32; 2] {
        let sx = original.width() as f32 / self.width as f32;
        let sy = original.height() as f32 / self.height as f32;
        [
            (p[0] * sx).clamp(0., original.width() as f32),
            ((p[1] - self.top as f32) * sy).clamp(0., original.height() as f32),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thin_pages_stay_within_resize_and_pixel_limits() {
        for ([w, h], expected) in [([3000, 20], [1984, 32]), ([1, 1900], [32, 1984])] {
            let pixels = vec![255; w * h * 3];
            let original = ImageView::new(&pixels, w, h, 0, PixelFormat::Rgb24).unwrap();
            let page = Page::prepare(original, 1_000_000).unwrap();
            assert_eq!([page.width, page.height], expected);
            assert!(page.view(original).unwrap().pixel_count().unwrap() <= 1_000_000);
        }
    }

    #[test]
    fn letterbox_is_centered_and_boxes_are_restored_to_original_pixels() {
        let rgb = vec![255; 360 * 40 * 3];
        let img = ImageView::new(&rgb, 360, 40, 0, PixelFormat::Rgb24).unwrap();
        let page = Page::prepare(img, 100_000).unwrap();
        let work = page.view(img).unwrap();
        assert_eq!(work.height(), 90);
        assert_eq!(work.bgr(0, 24), [0; 3]);
        assert_eq!(work.bgr(0, 25), [255; 3]);
        assert_eq!(page.restore([17., 32.], img), [17., 7.]);
    }
}
