use simd_paddle_ocr_rs::{
    ExecutionTarget, ModelPreset, OcrModelSet, RunOptions,
    cpu::CpuContext,
    image::{ImageView, PixelFormat},
};
use std::{error::Error, sync::Arc, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let imgs = std::env::args_os().skip(1).map(|path| {
        Ok(image::ImageReader::open(path)?.decode()?.to_rgb8())
    }).collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    assert_eq!(imgs.len(), 3);
    let models = OcrModelSet::from_directory("../SimdPaddleOCR/models", ModelPreset::ChineseV6Tiny, true)?;
    let mut req = models.builder().cpu_context(Arc::new(CpuContext::new(8)?))
        .execution_target(ExecutionTarget::Cpu).build()?.request()?;
    // One warm-up total, then one timed call on each of three decoded inputs.
    for (turn, &index) in [0, 0, 1, 2].iter().enumerate() {
        let img = &imgs[index];
        let view = ImageView::new(img, img.width() as usize, img.height() as usize, 0, PixelFormat::Rgb24)?;
        let start = Instant::now();
        let result = req.run(view, RunOptions::default())?.try_to_owned()?;
        println!("{} {} {:.3} {}", turn, index, start.elapsed().as_secs_f64()*1000., result.lines.len());
    }
    Ok(())
}
