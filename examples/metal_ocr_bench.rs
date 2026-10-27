//! Paired with tools/reference/MetalBench.cs and tools/metal_benchmark.py.
//! One fresh request, no untimed inference or implicit warm-up.
use simd_paddle_ocr_rs::{
    ExecutionTarget, ModelPreset, OcrModelSet, RunOptions,
    cpu::{CpuContext, CpuOptions},
    image::{ImageView, PixelFormat},
    ocr::{OcrExecution, OcrRequest, OwnedOcr},
    runtime::{
        CompileOptions, ExecutionBackend, GpuCompileOptions, GpuMatrixPrecision, GpuPrecision,
        gpu::{GpuApi, GpuContext, GpuContextOptions, GpuPlacement},
    },
};
use std::{
    collections::HashMap,
    error::Error,
    io::{self, Write},
    time::Instant,
};

#[derive(serde::Deserialize)]
struct Input {
    id: String,
    rgb: String,
    width: usize,
    height: usize,
}
impl Input {
    fn pixels(&self) -> Result<Vec<u8>, Box<dyn Error>> {
        let pixels = std::fs::read(&self.rgb)?;
        if self
            .width
            .checked_mul(self.height)
            .and_then(|n| n.checked_mul(3))
            != Some(pixels.len())
        {
            return Err("RGB byte length mismatch".into());
        }
        Ok(pixels)
    }
}

fn verify(request: &OcrRequest) -> Result<(), Box<dyn Error>> {
    let run = request.latest_run().ok_or("missing run statistics")?;
    if !run.succeeded || run.execution != OcrExecution::Resident {
        return Err("Rust OCR did not complete on the resident GPU path".into());
    }
    let gpu = run.gpu.as_ref().ok_or("missing resident statistics")?;
    if gpu.backend != "Metal"
        || !gpu.cpu_compute_stages.is_empty()
        || gpu.native_graph_dispatches == 0
        || [
            gpu.image_processing,
            gpu.geometry,
            gpu.projection,
            gpu.decoding,
        ]
        .iter()
        .any(|p| matches!(p, GpuPlacement::Cpu | GpuPlacement::Hybrid))
    {
        return Err(format!("Rust Metal fallback: {gpu:?}").into());
    }
    if run.stages.values().flatten().any(|s| {
        s.execution_backend
            .is_some_and(|b| b != ExecutionBackend::Gpu)
    }) {
        return Err("Rust neural stage used CPU".into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) || !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Err("release build on physical Apple Silicon required".into());
    }
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("metal_ocr_bench MODELS PRESET MANIFEST cold|sample|corpus INDEX".into());
    }
    let preset = match args[2].as_str() {
        "tiny" => ModelPreset::ChineseV6Tiny,
        "small" => ModelPreset::ChineseV6Small,
        "medium" => ModelPreset::ChineseV6Medium,
        _ => return Err("invalid preset".into()),
    };
    let inputs: Vec<Input> = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let mode = args[4].as_str();
    let selected: usize = args[5].parse()?;
    if selected >= inputs.len() || !matches!(mode, "cold" | "sample" | "corpus") {
        return Err("invalid mode or input index".into());
    }
    let initial_pixels = inputs[selected].pixels()?;
    let setup = Instant::now();
    // Explicit per-model numerical policy; fixture and corpus qualification
    // use these exact settings. Tiny's thin-text geometry needs FP32.
    let detector_precision = if args[2] == "tiny" {
        GpuPrecision::F32
    } else {
        GpuPrecision::MixedF16
    };
    let context = GpuContext::new_with_options(GpuContextOptions {
        recognition_wave_size: 16,
        classification_batch_size: 16,
        #[cfg(target_os = "macos")]
        metal_detector_precision: detector_precision,
        ..Default::default()
    })?;
    if context.adapter_info().backend != GpuApi::Metal {
        return Err("native Metal required".into());
    }
    let models = OcrModelSet::from_directory(&args[1], preset, true)?;
    let ocr = models
        .builder()
        .execution_target(ExecutionTarget::Gpu)
        .gpu_context(context.clone())
        .compile_options(CompileOptions {
            gpu: GpuCompileOptions {
                precision: GpuPrecision::MixedF16,
                matrix_precision: GpuMatrixPrecision::F16,
            },
            ..Default::default()
        })
        .cpu_context(CpuContext::shared_default()?)
        .cpu_options(CpuOptions {
            line_workers: 4,
            detector_threads: 8,
            preprocess_workers: 16,
        })
        .recognition_batch_size(2)
        .build()?;
    let mut request = ocr.request()?;
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.;
    let calls = match mode {
        "cold" => 1,
        "sample" => 2,
        _ => inputs.len() * 2,
    };
    let mut saved = HashMap::<&str, OwnedOcr>::new();
    let mut stdout = io::stdout().lock();
    for call in 0..calls {
        let i = if mode == "corpus" {
            call % inputs.len()
        } else {
            selected
        };
        let input = &inputs[i];
        let phase = if mode == "corpus" {
            if call < inputs.len() {
                "first_shape"
            } else {
                "warm"
            }
        } else if call == 0 {
            "cold"
        } else {
            "warm"
        };
        let loaded;
        let pixels = if i == selected && (mode != "corpus" || call == 0) {
            &initial_pixels
        } else {
            loaded = input.pixels()?;
            &loaded
        };
        let image = ImageView::new(pixels, input.width, input.height, 0, PixelFormat::Rgb24)?;
        let start = Instant::now();
        let output = request
            .run(
                image,
                RunOptions {
                    capture_spans: false,
                },
            )?
            .try_to_owned()?;
        let ms = start.elapsed().as_secs_f64() * 1000.;
        writeln!(stdout, "RESULT_READY {} {phase} {ms}", input.id)?;
        stdout.flush()?;
        verify(&request)?;
        if saved
            .get(input.id.as_str())
            .is_some_and(|old| old != &output)
        {
            return Err(format!("nondeterministic OCR output: {}", input.id).into());
        }
        serde_json::to_writer(
            &mut stdout,
            &serde_json::json!({
                "kind":"image", "id":input.id, "phase":phase, "ms":ms, "output":output,
                "statistics":request.latest_run()
            }),
        )?;
        writeln!(stdout)?;
        saved.insert(&input.id, output);
    }
    serde_json::to_writer(
        &mut stdout,
        &serde_json::json!({
            "kind":"summary", "implementation":"rust", "preset":args[2], "mode":mode, "calls":calls,
            "backend":"Metal", "adapter":context.adapter_info().name, "backend_verified":true,
            "precision":"mixed_f16", "detector_storage_precision":detector_precision, "setup_ms":setup_ms,
            "timing_contract":"RGB to owned result; parent launch-to-first-RESULT_READY includes process, raw input, models, Metal and inference; validation and serialization excluded"
        }),
    )?;
    writeln!(stdout)?;
    Ok(())
}
