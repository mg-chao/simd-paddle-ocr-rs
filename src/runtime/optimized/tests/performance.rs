//! Bounded fused-step diagnostics, not OCR latency measurements. The detector
//! probe isolates steps; the recognition probe times a complete backbone pass.
use super::*;

#[test]
#[ignore = "local release diagnostic: three detectors, one warm-up and one instrumented production pass"]
fn sparse_detector_production_steps() {
    #[cfg(windows)]
    unsafe {
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn GetProcessAffinityMask(
                process: *mut std::ffi::c_void,
                allowed: *mut usize,
                system: *mut usize,
            ) -> i32;
            fn SetProcessAffinityMask(process: *mut std::ffi::c_void, mask: usize) -> i32;
        }
        let (mut allowed, mut system) = (0usize, 0usize);
        let process = GetCurrentProcess();
        assert_ne!(
            GetProcessAffinityMask(process, &mut allowed, &mut system),
            0
        );
        assert!(allowed.count_ones() >= 8);
        let mut mask = 0;
        for _ in 0..8 {
            let bit = allowed.isolate_lowest_one();
            mask |= bit;
            allowed &= !bit;
        }
        assert_ne!(SetProcessAffinityMask(process, mask), 0);
    }
    let deadline = std::time::Instant::now();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let pixels = ::image::open(root.join("tests/baseline/corpus/imgs_1.jpg"))
        .unwrap()
        .to_rgb8();
    let image = crate::image::ImageView::new(
        pixels.as_raw(),
        pixels.width() as usize,
        pixels.height() as usize,
        0,
        crate::image::PixelFormat::Rgb24,
    )
    .unwrap();
    let size = crate::preprocess::compute_det_size(image.width(), image.height(), 960).unwrap();
    let mut input = vec![0.; 3 * size.width * size.height];
    crate::preprocess::det(
        image,
        size.width,
        size.height,
        &mut input,
        &mut crate::preprocess::ResizeWorkspace::default(),
    )
    .unwrap();
    let cpu = Arc::new(CpuContext::new(8).unwrap());
    for (model, file) in [
        ("tiny", "det.onnx"),
        ("small", "small_det.onnx"),
        ("medium", "medium_det.onnx"),
    ] {
        let compiled = Arc::new(
            CompiledModel::new(Arc::new(
                Model::load_path(root.join("../SimdPaddleOCR/models").join(file)).unwrap(),
            ))
            .unwrap(),
        );
        let mut session = Session::new(
            compiled.clone(),
            cpu.clone(),
            &[1, 3, size.height, size.width],
            ExecutionOptions {
                intra_op_threads: 8,
                ..Default::default()
            },
        )
        .unwrap();
        let expected = session.run(&input).unwrap().data().to_vec();
        let state = session.optimized.as_mut().unwrap();
        state.step_timings = Some(Vec::with_capacity(state.plan.steps.len()));
        let actual = session.run(&input).unwrap();
        assert_eq!(actual.data().len(), expected.len());
        assert!(
            actual
                .data()
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
                        "node={index} Conv {}->{} k{}x{} stride={}x{} groups={}",
                        p.cin,
                        p.cout,
                        p.kh,
                        p.kw,
                        p.sh,
                        p.sw,
                        p.cin / p.cpg
                    ),
                ),
                Operation::Native { index, .. } => (
                    compiled.model.graph().nodes[*index].op_type().to_owned(),
                    format!(
                        "node={index} {}",
                        compiled.model.graph().nodes[*index].op_type()
                    ),
                ),
                Operation::Convert => ("Convert".into(), "Convert".into()),
                Operation::Activation(_) => ("Activation".into(), "Activation".into()),
                Operation::LayerNorm { .. } => ("LayerNorm".into(), "LayerNorm".into()),
                Operation::View => ("View".into(), "View".into()),
            };
            let ms = time.as_secs_f64() * 1000.;
            *groups.entry(family).or_default() += ms;
            rows.push(serde_json::json!({ "ms": ms, "description": description,
                "inputs": step.inputs.iter().map(|&id| state.plan.values[id].shape.dimensions().to_vec()).collect::<Vec<_>>(),
                "output": state.plan.values[step.output].shape.dimensions(),
                "layout": format!("{:?}", state.plan.values[step.output].layout),
            }));
        }
        rows.sort_by(|a, b| {
            b["ms"]
                .as_f64()
                .unwrap()
                .total_cmp(&a["ms"].as_f64().unwrap())
        });
        println!(
            "detector_production_steps {}",
            serde_json::json!({
                "model": model, "fixture": "imgs_1", "shape": [1, 3, size.height, size.width], "threads": 8,
                "backend": format!("{:?}", compiled.backend), "total_step_ms": timings.iter().map(|v| v.as_secs_f64()*1000.).sum::<f64>(),
                "groups_ms": groups, "steps": rows, "outputs_exactly_equal": true,
            })
        );
        assert!(
            deadline.elapsed().as_secs() < 20,
            "fixed detector diagnostic exceeded twenty seconds"
        );
    }
}

#[test]
#[ignore = "local release diagnostic: two recognition shapes, one warm-up and three fixed samples"]
fn cold_physical_plan_bind() {
    const CALLS: usize = 16;
    let deadline = std::time::Instant::now();
    for (model, width) in [("small", 128), ("medium", 640)] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../SimdPaddleOCR/models")
            .join(format!("{model}_rec.onnx"));
        let compiled = CompiledModel::new(Arc::new(Model::load_path(&path).unwrap())).unwrap();
        let logical = bind(&compiled, &[1, 3, 48, width], ExecutionLimits::default()).unwrap();
        let count = compiled.instructions.len();
        // Only physical planning is timed. Compilation, weight packing, shape
        // propagation and arena allocation are outside every measured sample.
        let warmup = Plan::bind(&compiled, &logical, count, compiled.output).unwrap();
        let expected = (warmup.steps.len(), warmup.fused, warmup.elements);
        std::hint::black_box(warmup);
        let mut sample_ns = [0u128; 3];
        for elapsed in &mut sample_ns {
            let started = std::time::Instant::now();
            for _ in 0..CALLS {
                let plan = Plan::bind(&compiled, &logical, count, compiled.output).unwrap();
                assert_eq!((plan.steps.len(), plan.fused, plan.elements), expected);
                std::hint::black_box(plan);
            }
            *elapsed = started.elapsed().as_nanos();
            assert!(
                deadline.elapsed() < std::time::Duration::from_secs(30),
                "fixed physical-planning diagnostic exceeded thirty seconds"
            );
        }
        println!(
            "cold_physical_plan_bind {}",
            serde_json::json!({
                "model": model,
                "width": width,
                "backend": format!("{:?}", compiled.backend),
                "nodes": count,
                "physical_steps": expected.0,
                "fused_nodes": expected.1,
                "warmup_calls": 1,
                "measured_samples": 3,
                "calls_per_sample": CALLS,
                "sample_ns": sample_ns,
                "ns_per_bind": sample_ns.map(|ns| ns as f64 / CALLS as f64),
                "measured_inference_runs": 0,
            })
        );
    }
}

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
        assert_eq!(graph.nodes.last().unwrap().op_type(), "Softmax");
        let mut head = graph.nodes.len() - 2;
        if graph.nodes[head].op_type() == "Add" {
            head -= 1;
        }
        assert_eq!(graph.nodes[head].op_type(), "MatMul");
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
                    graph.nodes[*index].op_type().to_owned(),
                    format!("node={index} {}", graph.nodes[*index].op_type()),
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
                    compiled.model.graph().nodes[*index].op_type()
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

#[test]
#[ignore = "local release diagnostic: tiny REC cached reshape, one warm-up cycle and three fixed measured cycles"]
fn recognition_cached_arena_rebind() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../SimdPaddleOCR/models/rec.onnx");
    let compiled =
        Arc::new(CompiledModel::new(Arc::new(Model::load_path(&path).unwrap())).unwrap());
    let graph = compiled.model.graph();
    assert_eq!(graph.nodes.last().unwrap().op_type(), "Softmax");
    let mut head = graph.nodes.len() - 2;
    if graph.nodes[head].op_type() == "Add" {
        head -= 1;
    }
    assert_eq!(graph.nodes[head].op_type(), "MatMul");
    let feature = graph.nodes[head].inputs[0].unwrap();
    let last = graph.nodes[..head]
        .iter()
        .position(|node| node.outputs.contains(&Some(feature)))
        .unwrap();
    // Construct the real backbone prefix before any clock. Its output is the
    // feature preceding the vocabulary head, as in production recognition.
    let mut request =
        Session::new_prefix(compiled, &[1, 3, 48, 32], ExecutionLimits::default(), last).unwrap();
    request
        .set_execution_context(CpuContext::serial(), 1)
        .unwrap();
    let deadline = std::time::Instant::now();
    let widths = [320, 32, 320, 32];
    // One warm-up cycle binds both shapes and primes their cached metadata.
    // No inference, input fill, or additional warm-up is performed.
    for width in widths {
        request.reshape(&[1, 3, 48, width]).unwrap();
    }
    let before = request.statistics();
    let physical_misses = request.optimized.as_ref().unwrap().misses;
    let retained = request.retained_activation_bytes();
    let mut samples_ns = [[0u128; 4]; 3];
    let mut transitions = Vec::with_capacity(12);
    for (cycle, sample) in samples_ns.iter_mut().enumerate() {
        for (transition, (width, elapsed_ns)) in widths.into_iter().zip(sample).enumerate() {
            let initialized_before = request.optimized.as_ref().unwrap().arena.len();
            let started = std::time::Instant::now();
            request.reshape(&[1, 3, 48, width]).unwrap();
            *elapsed_ns = started.elapsed().as_nanos();
            let state = request.optimized.as_ref().unwrap();
            assert_eq!(request.retained_activation_bytes(), retained);
            assert_eq!(state.misses, physical_misses);
            transitions.push(serde_json::json!({
                "cycle": cycle,
                "transition": transition,
                "width": width,
                "elapsed_ns": *elapsed_ns,
                "active_bytes": request.activation_bytes(),
                "retained_bytes": request.retained_activation_bytes(),
                "initialized_bytes": state.arena.len() * size_of::<f32>(),
                "initialized_growth_bytes": state.arena.len().saturating_sub(initialized_before) * size_of::<f32>(),
                "logical_cache_hits": request.cache_hits,
                "logical_cache_misses": request.cache_misses,
                "physical_cache_hits": state.hits,
                "physical_cache_misses": state.misses,
            }));
            assert!(
                deadline.elapsed() < std::time::Duration::from_secs(5),
                "fixed rebind diagnostic exceeded five seconds"
            );
        }
    }
    let after = request.statistics();
    assert_eq!(after.plan_cache_misses, before.plan_cache_misses);
    assert_eq!(after.plan_cache_hits - before.plan_cache_hits, 12);
    let cycle_ns = samples_ns.map(|sample| sample.into_iter().sum::<u128>());
    let cycle_ms = cycle_ns.map(|ns| ns as f64 / 1_000_000.);
    println!(
        "arena_rebind_probe {}",
        serde_json::json!({
            "model": "tiny",
            "model_path": path,
            "prefix_last_node": last,
            "backend": before.backend,
            "shape": [1, 3, 48],
            "width_transitions": widths,
            "warmup_cycles": 1,
            "measured_cycles": 3,
            "transitions_per_cycle": 4,
            "cycle_ns": cycle_ns,
            "cycle_ms": cycle_ms,
            "transition_ns": samples_ns,
            "transitions": transitions,
            "retained_bytes": retained,
            "logical_cache_hits_before": before.plan_cache_hits,
            "logical_cache_hits_after": after.plan_cache_hits,
            "logical_cache_misses_before": before.plan_cache_misses,
            "logical_cache_misses_after": after.plan_cache_misses,
            "physical_cache_misses": physical_misses,
            "measured_inference_runs": 0,
        })
    );
}
