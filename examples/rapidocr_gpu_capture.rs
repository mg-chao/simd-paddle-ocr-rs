//! Collect public GPU OCR results while reusing the model and Vulkan/Metal pipelines.
#[path = "support/load_image.rs"]
mod load_image;

use simd_paddle_ocr_rs::{
    ExecutionTarget, ModelPreset, OcrModelSet, RunOptions,
    cpu::CpuContext,
    image::{ImageView, PixelFormat},
    runtime::gpu::GpuContextOptions,
};
use std::{
    error::Error,
    fs,
    io::{self, BufRead},
    path::Path,
    sync::Arc,
    time::Instant,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let preset = match args.next().as_deref() {
        Some("tiny") => ModelPreset::ChineseV6Tiny,
        Some("small") => ModelPreset::ChineseV6Small,
        Some("medium") => ModelPreset::ChineseV6Medium,
        _ => return Err("expected tiny, small or medium".into()),
    };
    let dest = args.next().ok_or("expected output directory")?;
    fs::create_dir_all(&dest)?;
    let timing = args.next().as_deref() == Some("--timing");
    let mut request = OcrModelSet::from_directory("../SimdPaddleOCR/models", preset, true)?
        .builder()
        .execution_target(ExecutionTarget::Gpu)
        .cpu_context(Arc::new(CpuContext::new(8)?))
        .gpu_options(GpuContextOptions {
            recognition_wave_size: 1,
            classification_batch_size: 4,
            ..Default::default()
        })
        .build()?
        .request()?;
    for (i, row) in io::stdin().lock().lines().enumerate() {
        let row: serde_json::Value = serde_json::from_str(&row?)?;
        let name = row["path"].as_str().ok_or("missing relative image path")?;
        let input = row["input"].as_str().ok_or("missing input")?;
        let capture = (|| -> Result<_, Box<dyn Error>> {
            let rgb = load_image::read(Path::new(input))?;
            let image = ImageView::new(
                rgb.as_raw(),
                rgb.width() as usize,
                rgb.height() as usize,
                0,
                PixelFormat::Rgb24,
            )?;
            if timing && i == 0 && preset == ModelPreset::ChineseV6Tiny {
                request.run(image, RunOptions::default())?;
            }
            let start = Instant::now();
            let result = request.run(image, RunOptions::default())?.try_to_owned()?;
            if timing {
                println!(
                    "{preset:?}\t{:.3}\t{}",
                    start.elapsed().as_secs_f64() * 1000.,
                    result.lines.len()
                );
            }
            let raw = serde_json::to_value(result)?;
            let lines: Vec<_> = raw["lines"].as_array().ok_or("missing OCR lines")?.iter().map(|v|
                serde_json::json!({"text":v["text"],"box":v["detection"]["points"],"score":v["recognition_score"]})).collect();
            if i == 0 {
                fs::write(
                    Path::new(&dest).join("placement.json"),
                    serde_json::to_vec_pretty(&request.latest_run())?,
                )?;
            }
            Ok(serde_json::json!({"status":"ok","lines":lines,"raw":raw}))
        })();
        let mut result =
            capture.unwrap_or_else(|e| serde_json::json!({"status":"error","error":e.to_string()}));
        result["path"] = row["path"].clone();
        result["source_sha256"] = row["sha256"].clone();
        let path = Path::new(&dest).join(format!("{name}.json"));
        fs::create_dir_all(path.parent().unwrap())?;
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, serde_json::to_vec(&result)?)?;
        fs::rename(tmp, path)?;
    }
    Ok(())
}
