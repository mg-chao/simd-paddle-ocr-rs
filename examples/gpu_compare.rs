#![recursion_limit = "256"]
//! Developer-only reproducible tensor comparison; no image preprocessing.
use simd_paddle_ocr_rs::{
    model::Model,
    runtime::{CompileOptions, CpuCompiledModel, CpuSession, ExecutionLimits, gpu::*},
};
use std::{error::Error, path::PathBuf, sync::Arc};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let measure = args.iter().any(|s| s == "measure");
    if measure && !cfg!(feature = "profiling") {
        return Err("measure requires the profiling Cargo feature".into());
    }
    if args.len() < 5 {
        return Err("usage: gpu_compare MODEL N C H W [mixed] [measure]".into());
    }
    if args[5..]
        .iter()
        .any(|arg| !matches!(arg.as_str(), "mixed" | "measure"))
    {
        return Err("unsupported comparison option".into());
    }
    let path = PathBuf::from(&args[0]);
    let shape = args[1..5]
        .iter()
        .map(|s| s.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    let precision = if args.iter().any(|s| s == "mixed") {
        GpuPrecision::MixedF16
    } else {
        GpuPrecision::F32
    };
    let context = GpuContext::new()?;
    let model = Arc::new(Model::load_path(&path)?);
    let input: Vec<f32> = if let Ok(path) = std::env::var("SIMD_OCR_INPUT") {
        let data = std::fs::read(path)?;
        if data.len() % 4 != 0 {
            return Err("invalid F32 input bytes".into());
        }
        data.as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    } else {
        (0..shape.iter().product())
            .map(|i| ((i * 37 + 19) % 127) as f32 / 63.0 - 1.0)
            .collect()
    };
    let compiled = Arc::new(GpuCompiledModel::compile_with_options(
        model.clone(),
        context.clone(),
        CompileOptions {
            gpu: GpuCompileOptions {
                precision,
                ..Default::default()
            },
            ..Default::default()
        },
    )?);
    let mut gpu = GpuSession::new(compiled, &shape, ExecutionLimits::default())?;
    let cpu = Arc::new(CpuCompiledModel::compile(
        model.clone(),
        CompileOptions {
            fusion: false,
            ..Default::default()
        },
    )?);
    let mut cpu = CpuSession::new_serial(cpu, &shape, ExecutionLimits::default())?;
    let expected = cpu.run(&input)?;
    // In measurement mode this correctness invocation is the only warmup.
    #[cfg(feature = "profiling")]
    let actual = if measure {
        gpu.run_profiled(&input)?
    } else {
        gpu.run(&input)?
    };
    #[cfg(not(feature = "profiling"))]
    let actual = gpu.run(&input)?;
    if let Ok(directory) = std::env::var("SIMD_OCR_COMPARE_DIR") {
        let directory = PathBuf::from(directory);
        std::fs::create_dir_all(&directory)?;
        for (name, values) in [
            ("input", input.as_slice()),
            ("rust-cpu", expected.data()),
            ("rust-gpu", actual.data()),
        ] {
            std::fs::write(
                directory.join(format!("{name}.f32")),
                bytemuck::cast_slice(values),
            )?;
        }
    }
    assert_eq!(actual.shape(), expected.shape());
    let mut max_abs = 0.0f32;
    let mut failures = 0;
    let (absolute_tolerance, relative_tolerance) = if precision == GpuPrecision::F32 {
        (1e-3, 1e-3)
    } else {
        (5e-2, 1e-2)
    };
    for (a, b) in actual.data().iter().zip(expected.data()) {
        max_abs = max_abs.max((a - b).abs());
        if !a.is_finite()
            || !b.is_finite()
            || (a - b).abs() > absolute_tolerance + relative_tolerance * b.abs()
        {
            failures += 1;
        }
    }
    let output_shape = actual.shape().dimensions().to_vec();
    let statistics = gpu.statistics();
    assert_eq!(
        statistics.execution_backend,
        Some(simd_paddle_ocr_rs::ExecutionBackend::Gpu)
    );
    assert_eq!(statistics.production_submissions, 1);
    assert_eq!(statistics.production_readbacks, 1);
    assert_eq!(statistics.diagnostic_readbacks, 0);
    if let Ok(path) = std::env::var("SIMD_OCR_TRACE_COMPARE") {
        let mut reference = std::collections::HashMap::new();
        cpu.run_with_trace(&input, |node| {
            reference.insert(
                model.graph().tensors[node.output_id.0].name.clone(),
                node.values.to_vec(),
            );
            Ok(())
        })?;
        let mut boundaries = Vec::new();
        gpu.run_with_trace(&input, |node| {
            let boundary = &model.graph().tensors[node.output_id.0].name;
            let expected = &reference[boundary];
            let maximum = node.values.iter().zip(expected).map(|(a,b)| (a-b).abs()).fold(0.0f32,f32::max);
            let outside = node.values.iter().zip(expected).filter(|(a,b)| (*a-*b).abs() > absolute_tolerance + relative_tolerance*b.abs()).count();
            boundaries.push(serde_json::json!({"boundary":boundary,"name":node.name,"operator":node.op_type,"shape":node.shape.dimensions(),"max_abs":maximum,"failures":outside}));
            Ok(())
        })?;
        std::fs::write(path, serde_json::to_vec_pretty(&boundaries)?)?;
    }
    let mut inventory = std::collections::BTreeMap::new();
    for n in &model.graph().nodes {
        *inventory.entry(n.op_type()).or_insert(0usize) += 1;
    }
    #[allow(unused_mut)]
    let mut measurements: Vec<serde_json::Value> = Vec::new();
    #[cfg(feature = "profiling")]
    if measure && failures == 0 {
        for _ in 0..3 {
            let start = std::time::Instant::now();
            gpu.run_profiled(&input)?;
            measurements.push(serde_json::json!({"wall_ms":start.elapsed().as_secs_f64()*1000.,"compute_ns":gpu.last_gpu_profile().and_then(|p|p.compute_ns())}));
        }
    }
    println!(
        "{}",
        serde_json::to_string(
            &serde_json::json!({"model":path,"shape":shape,"output_shape":output_shape,"adapter":context.adapter_info().name,"backend":format!("{:?}",context.adapter_info().backend),"precision":precision,"seed":1,"absolute_tolerance":absolute_tolerance,"relative_tolerance":relative_tolerance,"max_abs":max_abs,"failures":failures,"operators":inventory,"statistics":gpu.statistics(),"measurements":measurements})
        )?
    );
    if failures > 0 {
        return Err(format!("{failures} elements outside {precision:?} numerical gate").into());
    }
    Ok(())
}
