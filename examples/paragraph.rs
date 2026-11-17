//! Print recognized paragraphs with the original OCR line indices.
use simd_paddle_ocr_rs::{
    ExecutionTarget, ModelPreset, OcrModelSet, RunOptions,
    image::{ImageView, PixelFormat},
};
use std::{error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: paragraph OCR_MODELS PARAGRAPH_ONNX IMAGE [cpu|gpu]";
    let models_dir = PathBuf::from(args.next().ok_or(usage)?);
    let paragraph_model = PathBuf::from(args.next().ok_or(usage)?);
    let image_path = PathBuf::from(args.next().ok_or(usage)?);
    let target = match args.next() {
        None => ExecutionTarget::Cpu,
        Some(value) if value == "cpu" => ExecutionTarget::Cpu,
        Some(value) if value == "gpu" => ExecutionTarget::Gpu,
        _ => return Err(usage.into()),
    };
    if args.next().is_some() {
        return Err(usage.into());
    }

    let pixels = image::ImageReader::open(image_path)?
        .with_guessed_format()?
        .decode()?
        .to_rgb8();
    let image = ImageView::new(
        pixels.as_raw(),
        pixels.width() as usize,
        pixels.height() as usize,
        0,
        PixelFormat::Rgb24,
    )?;
    let models = OcrModelSet::from_directory(models_dir, ModelPreset::ChineseV6Tiny, true)?;
    let plan = models
        .builder()
        .paragraph_model(paragraph_model)
        .paragraph_target(target)
        .build()?;
    let mut request = plan.request()?;
    let ocr = request.run(image, RunOptions::default())?;
    let groups = ocr.paragraphs().ok_or("paragraph grouping was disabled")?;

    for (paragraph, indices) in groups.iter().enumerate() {
        println!("Paragraph {}: {indices:?}", paragraph + 1);
        for &index in indices {
            let line = ocr
                .get(index)
                .ok_or("paragraph contains an invalid line index")?;
            println!("  {index}: {}", line.text);
        }
    }
    Ok(())
}
