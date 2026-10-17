//! Bounded fused-step diagnostics, not OCR latency measurements. The detector
//! probe isolates steps; the recognition probe times a complete backbone pass.
use super::*;

#[test]
#[ignore = "local release diagnostic: two recognition widths, one warm-up and one instrumented pass"]
fn recognition_backbone_physical_steps() {
    #[cfg(windows)]
    unsafe {
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn GetProcessAffinityMask(
                process: *mut std::ffi::c_void,
                process_mask: *mut usize,
                system_mask: *mut usize,
            ) -> i32;
            fn SetProcessAffinityMask(process: *mut std::ffi::c_void, mask: usize) -> i32;
        }
        let (mut allowed, mut system) = (0, 0);
        let process = GetCurrentProcess();
        assert_ne!(
            GetProcessAffinityMask(process, &mut allowed, &mut system),
            0
        );
        assert_ne!(allowed, 0);
        assert_ne!(
            SetProcessAffinityMask(process, allowed.isolate_lowest_one()),
            0
        );
    }
    let deadline = std::time::Instant::now();
    for (model, width) in [("small", 128), ("medium", 640)] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../SimdPaddleOCR/models")
            .join(format!("{model}_rec.onnx"));
        let compiled =
            Arc::new(CompiledModel::new(Arc::new(Model::load_path(path).unwrap())).unwrap());
        let graph = compiled.model.graph();
        assert_eq!(graph.nodes.last().unwrap().op_type, "Softmax");
        let mut head = graph.nodes.len() - 2;
        if graph.nodes[head].op_type == "Add" {
            head -= 1;
        }
        assert_eq!(graph.nodes[head].op_type, "MatMul");
        let feature = graph.nodes[head].inputs[0].unwrap();
        let last = graph.nodes[..head]
            .iter()
            .position(|n| n.outputs.contains(&Some(feature)))
            .unwrap();
        let mut session = Session::new_prefix(
            compiled.clone(),
            &[1, 3, 48, width],
            ExecutionLimits::default(),
            last,
        )
        .unwrap();
        session
            .set_execution_context(CpuContext::serial(), 1)
            .unwrap();
        let input: Vec<_> = (0..3 * 48 * width)
            .map(|i| ((i * 37 % 251) as f32 - 125.) / 128.)
            .collect();
        let expected = session.run(&input).unwrap().data().to_vec();
        let state = session.optimized.as_mut().unwrap();
        state.step_timings = Some(Vec::with_capacity(state.plan.steps.len()));
        let actual = session.run(&input).unwrap().data();
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let state = session.optimized.as_ref().unwrap();
        let timings = state.step_timings.as_ref().unwrap();
        assert_eq!(timings.len(), state.plan.steps.len());
        let mut rows = Vec::new();
        let mut groups = std::collections::BTreeMap::<String, f64>::new();
        for (step, time) in state.plan.steps.iter().zip(timings) {
            let (family, description) = match &step.op {
                Operation::Native {
                    index,
                    kind: BoundKind::Conv(p),
                    ..
                } => (
                    format!("Conv {}x{} groups={}", p.kh, p.kw, p.cin / p.cpg),
                    format!(
                        "node={index} Conv {}->{} k{}x{} groups={}",
                        p.cin,
                        p.cout,
                        p.kh,
                        p.kw,
                        p.cin / p.cpg
                    ),
                ),
                Operation::Native { index, .. } => (
                    graph.nodes[*index].op_type.clone(),
                    format!("node={index} {}", graph.nodes[*index].op_type),
                ),
                Operation::Convert => ("Convert".into(), "Convert".into()),
                Operation::Activation(_) => ("Activation".into(), "Activation".into()),
                Operation::LayerNorm { .. } => ("LayerNorm".into(), "LayerNorm".into()),
                Operation::View => ("View".into(), "View".into()),
            };
            let ms = time.as_secs_f64() * 1000.;
            *groups.entry(family).or_default() += ms;
            let inputs: Vec<_> = step
                .inputs
                .iter()
                .map(|&id| state.plan.values[id].shape.dimensions().to_vec())
                .collect();
            rows.push((
                ms,
                format!(
                    "{description} inputs={inputs:?} output={:?}",
                    state.plan.values[step.output].shape.dimensions()
                ),
            ));
        }
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!(
            "{model} REC width={width} serial fused backbone; one instrumented pass; total_step_ms={:.6}; groups={groups:?}",
            rows.iter().map(|row| row.0).sum::<f64>()
        );
        for (ms, description) in rows.iter().take(25) {
            println!("{ms:.6}ms {description}");
        }
        assert!(
            deadline.elapsed().as_secs() < 30,
            "fixed diagnostic exceeded thirty seconds"
        );
    }
}

#[test]
#[ignore = "local release diagnostic: two shipped detector shapes, one warm-up and one call per physical step"]
fn sparse_detector_physical_steps() {
    #[cfg(windows)]
    unsafe {
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn GetProcessAffinityMask(
                process: *mut std::ffi::c_void,
                process_mask: *mut usize,
                system_mask: *mut usize,
            ) -> i32;
            fn SetProcessAffinityMask(process: *mut std::ffi::c_void, mask: usize) -> i32;
        }
        let (mut allowed, mut system) = (0, 0);
        let process = GetCurrentProcess();
        assert_ne!(
            GetProcessAffinityMask(process, &mut allowed, &mut system),
            0
        );
        assert_ne!(allowed, 0);
        assert_ne!(
            SetProcessAffinityMask(process, allowed.isolate_lowest_one()),
            0
        );
    }
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/baseline/manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let fixture = manifest["fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|value| value["id"] == "imgs_1")
        .unwrap();
    let size = crate::preprocess::compute_det_size(
        fixture["width"].as_u64().unwrap() as usize,
        fixture["height"].as_u64().unwrap() as usize,
        manifest["settings"]["detector_limit"].as_u64().unwrap() as usize,
    )
    .unwrap();
    println!(
        "imgs_1 resized tensor shape [1, 3, {}, {}]; neutral activations",
        size.height, size.width
    );
    let deadline = std::time::Instant::now();
    for model in ["small", "medium"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../SimdPaddleOCR/models")
            .join(format!("{model}_det.onnx"));
        let compiled =
            Arc::new(CompiledModel::new(Arc::new(Model::load_path(path).unwrap())).unwrap());
        let mut session = Session::new_serial(
            compiled.clone(),
            &[1, 3, size.height, size.width],
            ExecutionLimits::default(),
        )
        .unwrap();
        let cpu = session.cpu_context().clone();
        let state = session.optimized.as_mut().unwrap();
        let whole = state.plan.clone();
        let mut timings = Vec::new();
        for step in &whole.steps {
            // Keep physical shapes, fusion, alias offsets and all immutable
            // panels; only the step list and reported output are narrowed.
            let mut plan = (*whole).clone();
            plan.steps = vec![step.clone()];
            plan.output = step.output;
            state.plan = Arc::new(plan);
            let description = match &step.op {
                Operation::Native {
                    index,
                    kind: BoundKind::Conv(p),
                    ..
                } => format!(
                    "node={index} Conv {}->{} k{}x{} output={:?}",
                    p.cin,
                    p.cout,
                    p.kh,
                    p.kw,
                    whole.values[step.output].shape.dimensions()
                ),
                Operation::Native { index, .. } => format!(
                    "node={index} {}",
                    compiled.model.graph().nodes[*index].op_type
                ),
                _ => format!("output={:?}", whole.values[step.output].shape.dimensions()),
            };
            state.arena.fill(0.125);
            state.execute(&compiled, &cpu, 1).unwrap();
            state.arena.fill(0.125);
            let started = std::time::Instant::now();
            state.execute(&compiled, &cpu, 1).unwrap();
            let ms = started.elapsed().as_secs_f64() * 1000.;
            let families = state.statistics(&compiled, 1).specialized_kernels;
            timings.push((ms, description, format!("{families:?}")));
            assert!(
                deadline.elapsed().as_secs() < 60,
                "fixed diagnostic exceeded sixty seconds"
            );
        }
        state.plan = whole;
        timings.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!(
            "{model} isolated fused steps (neutral activations), one warm-up plus one sample; sum={:.3}ms",
            timings.iter().map(|v| v.0).sum::<f64>()
        );
        for (ms, description, family) in timings.iter().take(30) {
            println!("{ms:.6}ms {description} {family}");
        }
    }
}
