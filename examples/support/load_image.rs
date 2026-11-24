use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, RgbImage};
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

/// Decode a file with the same pixel and EXIF conventions as RapidOCR's path input.
pub fn read(path: &Path) -> Result<RgbImage, Box<dyn Error>> {
    let reader = ImageReader::open(path)?.with_guessed_format()?;
    let jpeg = reader.format() == Some(ImageFormat::Jpeg);
    if reader.format() == Some(ImageFormat::Png) {
        // Pillow exposes P images to NumPy as a single plane of palette indices.
        let mut stream = BufReader::new(File::open(path)?);
        let header = stream.fill_buf()?;
        if header.get(25) == Some(&3) {
            let mut png = png::Decoder::new(stream).read_info()?;
            let size = png.output_buffer_size().ok_or("PNG size overflow")?;
            let mut limits = image::Limits::default();
            limits.reserve(size as u64)?;
            limits.reserve(u64::from(png.info().width) * u64::from(png.info().height) * 4)?;
            let mut buf = vec![0; size];
            let frame = png.next_frame(&mut buf)?;
            let bits = frame.bit_depth as usize;
            let mask = (1u16 << bits) - 1;
            let mut gray = image::GrayImage::new(frame.width, frame.height);
            for (dst, src) in gray
                .chunks_exact_mut(frame.width as usize)
                .zip(buf.chunks_exact(frame.line_size))
            {
                for (x, sample) in dst.iter_mut().enumerate() {
                    let bit = x * bits;
                    *sample = (src[bit / 8] >> (8 - bits - bit % 8)) & mask as u8;
                }
            }
            let orientation = reader.into_decoder()?.orientation()?;
            let mut img = DynamicImage::ImageLuma8(gray);
            img.apply_orientation(orientation);
            return Ok(img.to_rgb8());
        }
    }
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut img = if jpeg
        && matches!(
            decoder.original_color_type(),
            image::ExtendedColorType::Rgb8 | image::ExtendedColorType::L8
        ) {
        let (w, h) = decoder.dimensions();
        let mut limits = image::Limits::default();
        limits.reserve(u64::from(w) * u64::from(h) * 3)?;
        let compressed = std::fs::read(path)?;
        let decoded = turbojpeg::decompress(&compressed, turbojpeg::PixelFormat::RGB)?;
        DynamicImage::ImageRgb8(
            RgbImage::from_raw(w, h, decoded.pixels).ok_or("invalid decoded JPEG size")?,
        )
    } else {
        DynamicImage::from_decoder(decoder)?
    };
    img.apply_orientation(orientation);
    match img {
        DynamicImage::ImageRgba8(rgba) if rgba.pixels().any(|p| p[3] < 255) => {
            let (mut luminance, mut visible) = (0., 0usize);
            for p in rgba.pixels().filter(|p| p[3] != 0) {
                luminance +=
                    0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]);
                visible += 1;
            }
            let bg = if visible == 0 || luminance < 128. * visible as f64 {
                255.
            } else {
                0.
            };
            Ok(RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
                let p = rgba.get_pixel(x, y);
                let alpha = f32::from(p[3]) / 255.;
                image::Rgb(std::array::from_fn(|c| {
                    (f32::from(p[c]) * alpha + bg * (1. - alpha)) as u8
                }))
            }))
        }
        DynamicImage::ImageLumaA8(gray) => {
            Ok(RgbImage::from_fn(gray.width(), gray.height(), |x, y| {
                let [g, a] = gray.get_pixel(x, y).0;
                image::Rgb([if a == 0 { 255 } else { g.saturating_add(!a) }; 3])
            }))
        }
        _ => Ok(img.to_rgb8()),
    }
}
