//! Dump stage inputs on one local diagnostic image. Never used by production inference.
#[path = "support/load_image.rs"]
mod load_image;
use simd_paddle_ocr_rs::{
    ModelPreset, OcrModelSet,
    cpu::CpuContext,
    detector::DetectionTrace,
    image::{ImageError, ImageView, PixelFormat},
    lines::LineTrace,
    ocr::OcrTrace,
    recognizer::RecognitionTrace,
};
use std::{
    error::Error,
    fs,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

fn floats(path: &Path, values: &[f32]) -> io::Result<()> {
    let mut out = BufWriter::new(fs::File::create(path)?);
    for v in values {
        out.write_all(&v.to_le_bytes())?;
    }
    out.flush()
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let preset = match args.next().as_deref().and_then(|s| s.to_str()) {
        Some("tiny") => ModelPreset::ChineseV6Tiny,
        Some("small") => ModelPreset::ChineseV6Small,
        Some("medium") => ModelPreset::ChineseV6Medium,
        _ => return Err("expected PRESET IMAGE OUT_DIR".into()),
    };
    let pixels = load_image::read(Path::new(&args.next().ok_or("missing image")?))?;
    let dir = PathBuf::from(args.next().ok_or("missing output directory")?);
    fs::create_dir_all(&dir)?;
    fs::write(dir.join("source.rgb"), pixels.as_raw())?;
    let image = ImageView::new(
        &pixels,
        pixels.width() as usize,
        pixels.height() as usize,
        0,
        PixelFormat::Rgb24,
    )?;
    let mut request = OcrModelSet::from_directory("../SimdPaddleOCR/models", preset, true)?
        .builder()
        .cpu_context(Arc::new(CpuContext::new(8)?))
        .build()?
        .request()?;
    let mode = args.next();
    if mode.as_deref() == Some(std::ffi::OsStr::new("--replay-rec")) {
        let line: usize = args
            .next()
            .and_then(|v| v.to_str()?.parse().ok())
            .ok_or("missing line index")?;
        let bytes = fs::read(dir.join(format!("rec-{line}.f32")))?;
        if bytes.is_empty() || !bytes.len().is_multiple_of(4 * 3 * 48) {
            return Err("invalid recognition tensor".into());
        }
        let input: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect();
        let compiled = request
            .recognizer()
            .compiled()
            .as_cpu()
            .ok_or("expected CPU model")?
            .clone();
        let mut session = simd_paddle_ocr_rs::runtime::CpuSession::new(
            compiled,
            Arc::new(CpuContext::new(8)?),
            &[1, 3, 48, input.len() / (3 * 48)],
            simd_paddle_ocr_rs::runtime::ExecutionOptions {
                intra_op_threads: 8,
                ..Default::default()
            },
        )?;
        floats(
            &dir.join(format!("optimized-rec-{line}.f32")),
            session.run(&input)?.data(),
        )?;
        return Ok(());
    }
    if mode.as_deref() != Some(std::ffi::OsStr::new("--replay-det")) {
        request.run_with_trace(image, false, |event| {
            let saved = match event {
                OcrTrace::Detection(DetectionTrace::Input { values, .. }) => {
                    floats(&dir.join("det-input.f32"), values)
                }
                OcrTrace::Detection(DetectionTrace::Probabilities { values, .. }) => {
                    floats(&dir.join("det-map.f32"), values)
                }
                OcrTrace::Detection(DetectionTrace::Boxes(b)) => fs::write(
                    dir.join("boxes.json"),
                    serde_json::to_vec(&b.try_to_owned()?)
                        .map_err(|e| ImageError::new(e.to_string()))?,
                ),
                OcrTrace::Line(LineTrace::OrientedCrop { index, image, .. }) => fs::write(
                    dir.join(format!(
                        "crop-{index}-{}x{}.bgr",
                        image.width(),
                        image.height()
                    )),
                    image.pixels(),
                ),
                OcrTrace::Recognition {
                    index,
                    event: RecognitionTrace::Input { values, .. },
                } => floats(&dir.join(format!("rec-{index}.f32")), values),
                OcrTrace::Recognition {
                    index,
                    event: RecognitionTrace::Public(line),
                } => fs::write(
                    dir.join(format!("decoded-{index}.json")),
                    serde_json::json!({"text": line.text, "score": line.score}).to_string(),
                ),
                _ => Ok(()),
            };
            saved.map_err(|e| ImageError::new(e.to_string()))
        })?;
    }
    // Node tracing uses the diagnostic graph. Replay the exact same NCHW input
    // in the production CPU session to distinguish graph arithmetic from DB.
    let boxes: serde_json::Value = serde_json::from_slice(&fs::read(dir.join("boxes.json"))?)?;
    let dim = |name: &str| {
        boxes[name]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("invalid trace dimensions")
    };
    let shape = [1, 3, dim("resized_height")?, dim("resized_width")?];
    let compiled = request
        .extractor()
        .detector()
        .compiled()
        .as_cpu()
        .ok_or("expected CPU model")?
        .clone();
    let mut session = simd_paddle_ocr_rs::runtime::CpuSession::new(
        compiled,
        Arc::new(CpuContext::new(8)?),
        &shape,
        simd_paddle_ocr_rs::runtime::ExecutionOptions {
            intra_op_threads: 8,
            ..Default::default()
        },
    )?;
    let bytes = fs::read(dir.join("det-input.f32"))?;
    let input: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect();
    floats(&dir.join("optimized-map.f32"), session.run(&input)?.data())?;
    Ok(())
}
