//! Inspect selected kernels without running inference or collecting timings.
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{
    cpu::CpuContext,
    model::{Model, Operator},
    runtime::{CompiledModel, ExecutionOptions, Session},
};
use std::{path::PathBuf, sync::Arc};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "--help" || a == "-h") {
        println!(
            "cpu_plan_probe MODEL.onnx N,C,H,W [THREADS=1]\nCompile/bind only; report selected kernels and normalized 2x2 Conv inputs. No inference or performance samples."
        );
        return Ok(());
    }
    if !(2..=3).contains(&args.len()) {
        return Err("expected MODEL.onnx N,C,H,W [THREADS=1]".into());
    }
    let path = PathBuf::from(&args[0]).canonicalize()?;
    let shape = args[1]
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    let threads = args.get(2).map_or("1", String::as_str).parse::<usize>()?;
    let compiled = Arc::new(CompiledModel::new(Arc::new(Model::load_path(&path)?))?);
    let graph = compiled.model().graph();
    let conv_2x2: Vec<_> = graph.nodes.iter().enumerate().filter_map(|(index,node)| {
        let Operator::Conv(attributes) = &node.operator else { return None };
        if attributes.kernel_shape != [2,2] { return None; }
        let inputs: Vec<_> = node.inputs.iter().flatten().map(|id| {
            let tensor = &graph.tensors[id.0];
            serde_json::json!({"name":tensor.name,"constant_shape":tensor.constant.as_ref().map(|t|t.dimensions())})
        }).collect();
        Some(serde_json::json!({"node":index,"inputs":inputs,"pads":attributes.pads,"strides":attributes.strides}))
    }).collect();
    let session = Session::new(
        compiled,
        Arc::new(CpuContext::new(threads)?),
        &shape,
        ExecutionOptions {
            intra_op_threads: threads,
            ..Default::default()
        },
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "model_path":path,"model_sha256":format!("{:x}",Sha256::digest(std::fs::read(&path)?)),
            "input_shape":shape,"threads":threads,"inference_calls":0,
            "scope":"Complete graph compile/bind only, default automatic kernel/layout/fusion policies. Recognition production uses a prefix of this graph.",
            "normalized_conv_2x2":conv_2x2,"statistics":session.statistics(),
        }))?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Plan probe: {error}");
        std::process::exit(1);
    }
}
