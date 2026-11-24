//! Run an extracted ONNX subgraph on a captured tensor using the native Rust executor.
use simd_paddle_ocr_rs::{
    cpu::CpuContext,
    model::Model,
    runtime::{CpuCompiledModel, CpuSession, ExecutionOptions},
};
use std::{
    error::Error,
    fs,
    io::{BufWriter, Write},
    sync::Arc,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or("expected MODEL INPUT OUTPUT DIMS")?;
    let input = fs::read(args.next().ok_or("missing input")?)?;
    let output = args.next().ok_or("missing output")?;
    let dims = args
        .next()
        .ok_or("missing dimensions")?
        .split('x')
        .map(str::parse)
        .collect::<Result<Vec<usize>, _>>()?;
    let model = Arc::new(Model::load_path(model)?);
    let (words, tail) = input.as_chunks::<4>();
    if !tail.is_empty() {
        return Err("incomplete f32 input".into());
    }
    let input: Vec<_> = words.iter().map(|&b| f32::from_le_bytes(b)).collect();
    if let Some(backend) = args.next() {
        #[cfg(feature = "gpu")]
        if matches!(backend.as_str(), "--gpu-db" | "--gpu-mixed-db") {
            use simd_paddle_ocr_rs::{
                detector::{Detector, DetectorOptions},
                image::{ImageView, PixelFormat},
                runtime::gpu::GpuContext,
            };
            let [height, width, source_height, source_width] = dims.as_slice() else {
                return Err("expected MAP_HxMAP_WxPAGE_HxPAGE_W".into());
            };
            let pixels = vec![0; source_width * source_height * 3];
            let source = ImageView::new(
                &pixels,
                *source_width,
                *source_height,
                0,
                PixelFormat::Bgr24,
            )?;
            let model = gpu_model(model, GpuContext::new()?, backend == "--gpu-mixed-db")?;
            let options = DetectorOptions {
                use_dilation: true,
                max_candidates: 1000,
                box_threshold: Some(0.5),
                ..Default::default()
            };
            let mut session = Detector::new(model, options, Default::default())?;
            let result = session.postprocess(&input, *width, *height, source, 1., 1.)?;
            serde_json::to_writer(fs::File::create(output)?, result.boxes)?;
            return Ok(());
        }
        #[cfg(feature = "gpu")]
        if matches!(backend.as_str(), "--gpu" | "--gpu-mixed") {
            use simd_paddle_ocr_rs::runtime::gpu::{GpuContext, GpuSession};
            let compiled = gpu_model(model, GpuContext::new()?, backend == "--gpu-mixed")?;
            let mut session = GpuSession::new(compiled, &dims, Default::default())?;
            return write_output(output, session.run(&input)?.data());
        }
        return Err(format!("unsupported probe backend {backend}").into());
    }
    let compiled = CpuCompiledModel::new(model)?;
    let cpu = Arc::new(CpuContext::new(8)?);
    let mut session = CpuSession::new(
        Arc::new(compiled),
        cpu,
        &dims,
        ExecutionOptions {
            intra_op_threads: 8,
            ..Default::default()
        },
    )?;
    write_output(output, session.run(&input)?.data())
}

#[cfg(feature = "gpu")]
fn gpu_model(
    model: Arc<Model>,
    context: Arc<simd_paddle_ocr_rs::runtime::gpu::GpuContext>,
    mixed: bool,
) -> Result<Arc<simd_paddle_ocr_rs::runtime::gpu::GpuCompiledModel>, Box<dyn Error>> {
    use simd_paddle_ocr_rs::runtime::{CompileOptions, gpu::*};
    let mut options = CompileOptions::default();
    if mixed {
        options.gpu = GpuCompileOptions {
            precision: GpuPrecision::MixedF16,
            matrix_precision: GpuMatrixPrecision::F16,
        };
    }
    Ok(Arc::new(GpuCompiledModel::compile_with_options(model, context, options)?))
}

fn write_output(output: String, values: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut file = BufWriter::new(fs::File::create(output)?);
    for v in values {
        file.write_all(&v.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}
