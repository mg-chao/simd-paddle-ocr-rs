//! Decode a JPEG/PNG in application code, then run the Rust OCR pipeline.
use serde::Serialize;
use simd_paddle_ocr_rs::{
    ModelPreset, OcrModelSet, OwnedCharacterBox,
    cpu::CpuContext,
    image::{ImageView, PixelFormat},
    ocr::OwnedOcr,
};
use std::{error::Error, ffi::OsString, io::Write, path::PathBuf, sync::Arc};

const USAGE: &str = "Usage: ocr [OPTIONS] IMAGE\n\
    --models-dir PATH   Model directory (default: ../SimdPaddleOCR/models)\n\
    --preset NAME       tiny, small, or medium (default: tiny)\n\
    --no-cls            Do not load or run cls.onnx\n\
    --threads N         Shared CPU worker capacity (default: automatic)\n\
    --device NAME       cpu or gpu (default: cpu; GPU requires the gpu feature)\n\
    --gpu-backend NAME  auto, vulkan, dx12, or metal (default: auto)\n\
    --json              Print owned OCR output as JSON instead of joined text\n\
    --alignment         Capture CTC spans and token boxes; implies --json\n\
    -h, --help          Show this help\n\n\
JPEG and PNG are decoded by this example's development-only image dependency.\n\
No model files are downloaded and no .NET runtime is required.";

struct Args {
    image: PathBuf,
    models: PathBuf,
    preset: ModelPreset,
    classification: bool,
    threads: Option<usize>,
    gpu: bool,
    gpu_backend: String,
    json: bool,
    alignment: bool,
}

#[derive(Serialize)]
struct Output {
    #[serde(flatten)]
    ocr: OwnedOcr,
    #[serde(skip_serializing_if = "Option::is_none")]
    character_boxes: Option<Vec<Vec<OwnedCharacterBox>>>,
}

fn parse_args() -> Result<Option<Args>, Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let mut models = PathBuf::from("../SimdPaddleOCR/models");
    let mut preset = ModelPreset::ChineseV6Tiny;
    let mut classification = true;
    let mut threads = None;
    let mut gpu = false;
    let mut gpu_backend = String::from("auto");
    let mut json = false;
    let mut alignment = false;
    let mut image = None;
    let value = |args: &mut std::iter::Skip<std::env::ArgsOs>, flag: &str| {
        args.next()
            .ok_or_else(|| format!("{flag} requires a value"))
    };
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("-h" | "--help") => return Ok(None),
            Some("--models-dir") => models = value(&mut args, "--models-dir")?.into(),
            Some("--preset") => {
                preset = match value(&mut args, "--preset")?.to_str() {
                    Some("tiny") => ModelPreset::ChineseV6Tiny,
                    Some("small") => ModelPreset::ChineseV6Small,
                    Some("medium") => ModelPreset::ChineseV6Medium,
                    _ => return Err("--preset must be tiny, small, or medium".into()),
                };
            }
            Some("--no-cls") => classification = false,
            Some("--device") => {
                gpu = match value(&mut args, "--device")?.to_str() {
                    Some("cpu") => false,
                    Some("gpu") => true,
                    _ => return Err("--device must be cpu or gpu".into()),
                };
            }
            Some("--gpu-backend") => {
                gpu_backend = value(&mut args, "--gpu-backend")?
                    .into_string()
                    .map_err(|_| "--gpu-backend must be auto, vulkan, dx12, or metal")?;
                if !matches!(gpu_backend.as_str(), "auto" | "vulkan" | "dx12" | "metal") {
                    return Err("--gpu-backend must be auto, vulkan, dx12, or metal".into());
                }
            }
            Some("--threads") => {
                let count: usize = value(&mut args, "--threads")?
                    .to_str()
                    .ok_or("--threads must be a positive integer")?
                    .parse()?;
                if !(1..=1024).contains(&count) {
                    return Err("--threads must be within 1..=1024".into());
                }
                threads = Some(count);
            }
            Some("--json") => json = true,
            Some("--alignment") => {
                alignment = true;
                json = true;
            }
            Some("--") => {
                for path in args.by_ref() {
                    set_image(&mut image, path)?;
                }
                break;
            }
            Some(s) if s.starts_with('-') => return Err(format!("unknown option: {s}").into()),
            _ => set_image(&mut image, arg)?,
        }
    }
    if !gpu && gpu_backend != "auto" {
        return Err("--gpu-backend requires --device gpu".into());
    }
    #[cfg(not(feature = "gpu"))]
    if gpu {
        return Err("GPU execution requires rebuilding with --features gpu".into());
    }
    Ok(Some(Args {
        image: image.ok_or("an image path is required; use --help for usage")?,
        models,
        preset,
        classification,
        threads,
        gpu,
        gpu_backend,
        json,
        alignment,
    }))
}

fn set_image(image: &mut Option<PathBuf>, path: OsString) -> Result<(), Box<dyn Error>> {
    if image.replace(path.into()).is_some() {
        return Err("exactly one image path is accepted".into());
    }
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        println!("{USAGE}");
        return Ok(());
    };
    // Guess the format from bytes: some public PaddleOCR JPEG filenames contain PNGs.
    // Decode without implicit EXIF rotation; callers own image orientation policy.
    let pixels = image::ImageReader::open(&args.image)
        .map_err(|e| format!("open {}: {e}", args.image.display()))?
        .with_guessed_format()?
        .decode()
        .map_err(|e| format!("decode {}: {e}", args.image.display()))?
        .to_rgb8();
    let image = ImageView::new(
        pixels.as_raw(),
        pixels.width() as usize,
        pixels.height() as usize,
        0,
        PixelFormat::Rgb24,
    )?;
    let models = OcrModelSet::from_directory(args.models, args.preset, args.classification)?;
    let mut builder = models
        .builder()
        .direction_classification(args.classification);
    if let Some(threads) = args.threads {
        builder = builder.cpu_context(Arc::new(CpuContext::new(threads)?));
    }
    #[cfg(feature = "gpu")]
    if args.gpu {
        use simd_paddle_ocr_rs::runtime::{
            ExecutionTarget,
            gpu::{GpuBackendPreference, GpuContext, GpuContextOptions},
        };
        let backend = match args.gpu_backend.as_str() {
            "vulkan" => GpuBackendPreference::Vulkan,
            "dx12" => GpuBackendPreference::Dx12,
            "metal" => GpuBackendPreference::Metal,
            _ => GpuBackendPreference::Auto,
        };
        let context = GpuContext::new_with_options(GpuContextOptions {
            backend,
            ..Default::default()
        })?;
        eprintln!("GPU: {:?}", context.adapter_info());
        builder = builder.execution_target(ExecutionTarget::Gpu(context));
    }
    #[cfg(not(feature = "gpu"))]
    let _ = (args.gpu, &args.gpu_backend);
    let mut request = builder.build()?;
    let result = request.run(image, args.alignment)?.try_to_owned()?;
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    if args.json {
        let character_boxes = args
            .alignment
            .then(|| {
                result
                    .lines
                    .iter()
                    .map(|line| {
                        line.estimate_character_boxes()?
                            .into_iter()
                            .map(|token| token.try_to_owned())
                            .collect()
                    })
                    .collect::<simd_paddle_ocr_rs::image::ImageResult<Vec<_>>>()
            })
            .transpose()?;
        serde_json::to_writer_pretty(
            &mut output,
            &Output {
                ocr: result,
                character_boxes,
            },
        )?;
        writeln!(output)?;
    } else {
        writeln!(output, "{}", result.text)?;
    }
    output.flush()?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ocr: {error}");
        std::process::exit(1);
    }
}
