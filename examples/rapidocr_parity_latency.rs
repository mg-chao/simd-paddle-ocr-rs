//! One measured public OCR call per preset, sharing decoded pixels and eight CPU workers.
use simd_paddle_ocr_rs::{
    ModelPreset, OcrModelSet, RunOptions,
    cpu::CpuContext,
    image::{ImageView, PixelFormat},
};
use std::{error::Error, sync::Arc, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let file = args.next().ok_or("expected one image")?;
    let stage_profile = args.next().is_some_and(|arg| arg == "--profile-small");
    #[cfg(not(feature = "profiling"))]
    if stage_profile {
        return Err("stage measurement requires --features profiling".into());
    }
    let rgb = image::ImageReader::open(file)?
        .with_guessed_format()?
        .decode()?
        .to_rgb8();
    let image = ImageView::new(
        &rgb,
        rgb.width() as usize,
        rgb.height() as usize,
        0,
        PixelFormat::Rgb24,
    )?;
    let cpu = Arc::new(CpuContext::new(8)?);
    for preset in [
        ModelPreset::ChineseV6Tiny,
        ModelPreset::ChineseV6Small,
        ModelPreset::ChineseV6Medium,
    ] {
        if stage_profile && preset != ModelPreset::ChineseV6Small {
            continue;
        }
        let models = OcrModelSet::from_directory("../SimdPaddleOCR/models", preset, true)?;
        let mut request = models
            .builder()
            .cpu_context(cpu.clone())
            .build()?
            .request()?;
        #[cfg(feature = "profiling")]
        if stage_profile {
            let (output, stages) = request.run_profiled(image, false)?;
            let _owned = output.try_to_owned()?;
            println!("{}", serde_json::to_string(&stages)?);
            continue;
        }
        if preset == ModelPreset::ChineseV6Tiny {
            request.run(image, RunOptions::default())?;
        }
        let start = Instant::now();
        let output = request.run(image, RunOptions::default())?.try_to_owned()?;
        println!(
            "{preset:?}\t{:.3}\t{}",
            start.elapsed().as_secs_f64() * 1000.,
            output.lines.len()
        );
    }
    Ok(())
}
