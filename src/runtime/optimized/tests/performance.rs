//! Isolated fused-step diagnosis. These samples are not OCR latency measurements:
//! each physical step receives neutral activation data in its original layout.
use super::*;

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
            SetProcessAffinityMask(process, allowed & allowed.wrapping_neg()),
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
