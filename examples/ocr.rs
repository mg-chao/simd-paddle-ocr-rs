#![recursion_limit = "256"]
//! Decode a JPEG/PNG in application code, then run CPU or native GPU inference.
use serde::Serialize;
use simd_paddle_ocr_rs::{
    ExecutionTarget, ModelPreset, OcrModelSet, OwnedCharacterBox,
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
    --execution TARGET cpu, auto, or gpu (default: cpu)\n\
    --det-execution TARGET / --rec-execution TARGET / --cls-execution TARGET\n\
    --gpu-precision P   f32 (default), or mixed-f16\n\
    --gpu-matrix P      f32 (default), or f16 (native Metal fast mode)\n\
    --statistics       Print actual stage backends, kernels, and memory to stderr\n\
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
    json: bool,
    alignment: bool,
    target: ExecutionTarget,
    stages: [Option<ExecutionTarget>; 3],
    gpu_precision: String,
    gpu_matrix: String,
    statistics: bool,
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
    let mut json = false;
    let mut alignment = false;
    let mut image = None;
    let mut target = ExecutionTarget::Cpu;
    let mut stages = [None; 3];
    let mut gpu_precision = "f32".to_string();
    let mut gpu_matrix = "f32".to_string();
    let mut statistics = false;
    let value = |args: &mut std::iter::Skip<std::env::ArgsOs>, flag: &str| {
        args.next()
            .ok_or_else(|| format!("{flag} requires a value"))
    };
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--statistics") => statistics = true,
            Some(
                flag @ ("--execution" | "--det-execution" | "--rec-execution" | "--cls-execution"),
            ) => {
                let selected = match value(&mut args, flag)?.to_str() {
                    Some("cpu") => ExecutionTarget::Cpu,
                    Some("auto") => ExecutionTarget::Auto,
                    Some("gpu") => ExecutionTarget::Gpu,
                    _ => return Err("execution target must be cpu, auto, or gpu".into()),
                };
                match flag {
                    "--det-execution" => stages[0] = Some(selected),
                    "--rec-execution" => stages[1] = Some(selected),
                    "--cls-execution" => stages[2] = Some(selected),
                    _ => target = selected,
                };
            }
            Some("--gpu-precision") => {
                gpu_precision = value(&mut args, "--gpu-precision")?
                    .into_string()
                    .map_err(|_| "invalid precision")?
            }
            Some("--gpu-matrix") => {
                gpu_matrix = value(&mut args, "--gpu-matrix")?
                    .into_string()
                    .map_err(|_| "invalid GPU matrix precision")?;
            }
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
    Ok(Some(Args {
        image: image.ok_or("an image path is required; use --help for usage")?,
        models,
        preset,
        classification,
        threads,
        json,
        alignment,
        target,
        stages,
        gpu_precision,
        gpu_matrix,
        statistics,
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
        .execution_target(args.target)
        .direction_classification(args.classification);
    if let Some(target) = args.stages[0] {
        builder = builder.detector_target(target);
    }
    if let Some(target) = args.stages[1] {
        builder = builder.recognizer_target(target);
    }
    if let Some(target) = args.stages[2] {
        builder = builder.classifier_target(target);
    }
    #[cfg(feature = "gpu")]
    {
        use simd_paddle_ocr_rs::runtime::gpu::*;
        let precision = match args.gpu_precision.as_str() {
            "f32" => GpuPrecision::F32,
            "mixed-f16" => GpuPrecision::MixedF16,
            _ => return Err("invalid GPU precision".into()),
        };
        let matrix_precision = match args.gpu_matrix.as_str() {
            "f32" => GpuMatrixPrecision::F32,
            "f16" => GpuMatrixPrecision::F16,
            _ => return Err("GPU matrix precision must be f32 or f16".into()),
        };
        builder = builder.compile_options(simd_paddle_ocr_rs::runtime::CompileOptions {
            gpu: GpuCompileOptions {
                precision,
                matrix_precision,
            },
            ..Default::default()
        });
        builder = builder.gpu_options(GpuContextOptions {
            recognition_wave_size: if matrix_precision == GpuMatrixPrecision::F16 {
                4
            } else {
                1
            },
            classification_batch_size: if matrix_precision == GpuMatrixPrecision::F16 {
                16
            } else {
                4
            },
            ..Default::default()
        });
    }
    #[cfg(not(feature = "gpu"))]
    if args.gpu_precision != "f32" || args.gpu_matrix != "f32" {
        return Err("GPU options require --features gpu".into());
    }
    if let Some(threads) = args.threads {
        builder = builder.cpu_context(Arc::new(CpuContext::new(threads)?));
    }
    let mut request = builder.build()?.request()?;
    let result = request
        .run(
            image,
            simd_paddle_ocr_rs::RunOptions {
                capture_spans: args.alignment,
            },
        )?
        .try_to_owned()?;
    if args.statistics {
        eprintln!("{}", serde_json::to_string_pretty(&request.latest_run())?);
        #[cfg(feature = "gpu")]
        if let Some(statistics) = request.gpu_pipeline_statistics() {
            eprintln!("{}", serde_json::to_string_pretty(&statistics)?);
        }
    }
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
