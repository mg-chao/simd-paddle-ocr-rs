//! OCR from an already decoded, tightly packed BGR24 file.
use simd_paddle_ocr_rs::{
    ModelPreset, OcrModelSet,
    cpu::CpuContext,
    image::{ImageView, PixelFormat},
};
use std::{error::Error, fs, sync::Arc};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(5..=8).contains(&args.len()) {
        return Err("usage: ocr <models-dir> <tiny|small|medium> <pixels.bgr> <width> <height> [threads=1] [batch-size=1] [no-cls]".into());
    }
    let preset = match args[1].as_str() {
        "tiny" => ModelPreset::ChineseV6Tiny,
        "small" => ModelPreset::ChineseV6Small,
        "medium" => ModelPreset::ChineseV6Medium,
        _ => return Err("preset must be tiny, small or medium".into()),
    };
    let width = args[3].parse()?;
    let height = args[4].parse()?;
    let threads = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(1);
    let batch_size = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(1);
    let classify = match args.get(7).map(String::as_str) {
        None => true,
        Some("no-cls") => false,
        Some(_) => return Err("last argument must be no-cls when supplied".into()),
    };
    let pixels = fs::read(&args[2])?;
    let image = ImageView::new(&pixels, width, height, 0, PixelFormat::Bgr24)?;
    let models = OcrModelSet::from_directory(&args[0], preset, classify)?;
    let mut ocr = models
        .builder()
        .cpu_context(Arc::new(CpuContext::new(threads)?))
        .recognition_batch_size(batch_size)
        .build()?;
    let result = ocr.run(image, true)?;
    let characters = result
        .iter()
        .map(|line| line.estimate_character_boxes())
        .collect::<Result<Vec<_>, _>>()?;
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &serde_json::json!({
            "result": result, "characters": characters, "packed_text_hash": result.packed_text_hash(),
        }),
    )?;
    Ok(())
}
