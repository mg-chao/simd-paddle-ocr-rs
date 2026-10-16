//! Producer for the separately invoked, paired Windows CPU acceptance gate.
use super::{ALLOCATED, ALLOCATION_COUNT, REALLOCATION_COUNT, TRACK_ALLOCATIONS};
use serde_json::{Value, json};
use simd_paddle_ocr_rs::{
    ModelPreset, OcrModelSet,
    cpu::{CpuContext, CpuOptions},
    image::{ImageView, PixelFormat},
    ocr::{Ocr, OwnedOcr},
};
use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Barrier, atomic::Ordering, mpsc},
    thread,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone)]
struct Image {
    id: String,
    width: usize,
    height: usize,
    bytes: Arc<Vec<u8>>,
}
impl Image {
    fn view(&self) -> simd_paddle_ocr_rs::image::ImageResult<ImageView<'_>> {
        ImageView::new(&self.bytes, self.width, self.height, 0, PixelFormat::Bgr24)
    }
}
struct Command {
    image: Image,
    iterations: usize,
    profile: bool,
}
struct Response {
    result: std::result::Result<OwnedOcr, String>,
    borrowed_ns: u64,
    ownership_ns: u64,
    profile: Value,
    activations: usize,
    workspace: usize,
    kernel_scratch: usize,
}
struct Workers {
    commands: Vec<mpsc::Sender<Option<Command>>>,
    responses: mpsc::Receiver<Response>,
    ready: Arc<Barrier>,
    start: Arc<Barrier>,
    finish: Arc<Barrier>,
    handles: Vec<thread::JoinHandle<()>>,
}
impl Workers {
    fn new(requests: Vec<Ocr>) -> Self {
        let count = requests.len();
        let ready = Arc::new(Barrier::new(count + 1));
        let start = Arc::new(Barrier::new(count + 1));
        let finish = Arc::new(Barrier::new(count + 1));
        let (responses_tx, responses) = mpsc::channel();
        let mut commands = Vec::new();
        let mut handles = Vec::new();
        for mut request in requests {
            let (tx, rx) = mpsc::channel::<Option<Command>>();
            commands.push(tx);
            let (ready, start, finish, response_tx) = (
                ready.clone(),
                start.clone(),
                finish.clone(),
                responses_tx.clone(),
            );
            handles.push(thread::spawn(move || {
                while let Ok(Some(command)) = rx.recv() {
                    ready.wait();
                    start.wait();
                    let mut borrowed_ns = 0;
                    let mut ownership_ns = 0;
                    #[cfg(feature = "profiling")]
                    let mut raw_profile = None;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || -> std::result::Result<OwnedOcr, String> {
                            let mut output = None;
                            for _ in 0..command.iterations {
                                let view = command.image.view().map_err(|e| e.to_string())?;
                                let before = Instant::now();
                                #[cfg(feature = "profiling")]
                                let result = if command.profile {
                                    let (result, p) = request
                                        .run_profiled(view, true)
                                        .map_err(|e| e.to_string())?;
                                    raw_profile = Some(p);
                                    result
                                } else {
                                    request.run(view, true).map_err(|e| e.to_string())?
                                };
                                #[cfg(not(feature = "profiling"))]
                                let result = request.run(view, true).map_err(|e| e.to_string())?;
                                borrowed_ns += before.elapsed().as_nanos() as u64;
                                let before = Instant::now();
                                let owned = result.try_to_owned().map_err(|e| e.to_string())?;
                                ownership_ns += before.elapsed().as_nanos() as u64;
                                output = Some(owned);
                            }
                            output
                                .ok_or_else(|| "acceptance iteration count must be positive".into())
                        },
                    ))
                    .unwrap_or_else(|_| Err("OCR worker panicked".into()));
                    finish.wait();
                    #[cfg(feature = "profiling")]
                    let profile = raw_profile.map_or(Value::Null, |p| {
                        serde_json::to_value(p).expect("integer stage counters serialize")
                    });
                    #[cfg(not(feature = "profiling"))]
                    let profile = if command.profile {
                        json!({"available":false})
                    } else {
                        Value::Null
                    };
                    let response = Response {
                        result,
                        borrowed_ns,
                        ownership_ns,
                        profile,
                        activations: request.retained_activation_bytes(),
                        workspace: request.retained_workspace_bytes(),
                        kernel_scratch: request.retained_kernel_scratch_bytes(),
                    };
                    if response_tx.send(response).is_err() {
                        break;
                    }
                }
            }));
        }
        Self {
            commands,
            responses,
            ready,
            start,
            finish,
            handles,
        }
    }
    fn round(
        &self,
        image: &Image,
        iterations: usize,
        profile: bool,
    ) -> Result<(u64, Vec<Response>)> {
        for tx in &self.commands {
            tx.send(Some(Command {
                image: image.clone(),
                iterations,
                profile,
            }))?;
        }
        self.ready.wait();
        let before = Instant::now();
        self.start.wait();
        self.finish.wait();
        let wall = before.elapsed().as_nanos() as u64;
        let mut results = Vec::new();
        for _ in &self.commands {
            results.push(self.responses.recv()?);
        }
        for response in &results {
            if let Err(error) = &response.result {
                return Err(error.clone().into());
            }
        }
        Ok((wall, results))
    }
}
impl Drop for Workers {
    fn drop(&mut self) {
        for tx in &self.commands {
            let _ = tx.send(None);
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}
fn result_json(result: &OwnedOcr) -> Value {
    json!({"text":result.text,"detected_count":result.detected_count,"detector_resized_width":result.detector_resized_width,"detector_resized_height":result.detector_resized_height,"lines":result.as_view().iter().collect::<Vec<_>>()})
}
fn memory(responses: &[Response]) -> Value {
    let workspace = responses.iter().map(|r| r.workspace).sum::<usize>();
    let scratch = responses.iter().map(|r| r.kernel_scratch).sum::<usize>();
    let other = workspace
        .checked_sub(scratch)
        .expect("kernel scratch is a workspace subset");
    json!({"activation_bytes":responses.iter().map(|r|r.activations).sum::<usize>(),"workspace_bytes":workspace,
        "kernel_scratch_bytes":scratch,"nonkernel_workspace_bytes":other,
        "kernel_scratch_contract":"retained request kernel-buffer capacities, included in workspace_bytes; fixed stack tiles are excluded and not measured separately"})
}
fn check(responses: &[Response], expected: &OwnedOcr) -> Result<()> {
    if responses
        .iter()
        .any(|r| r.result.as_ref().ok() != Some(expected))
    {
        return Err(
            "a repeated/concurrent result changed within the same kernel configuration".into(),
        );
    }
    Ok(())
}
fn write(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

#[cfg(target_os = "windows")]
fn peak_working_set() -> Result<usize> {
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        faults: u32,
        peak: usize,
        current: usize,
        peak_paged: usize,
        paged: usize,
        peak_nonpaged: usize,
        nonpaged: usize,
        pagefile: usize,
        peak_pagefile: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut Counters, size: u32) -> i32;
    }
    let mut counters = Counters {
        cb: std::mem::size_of::<Counters>() as u32,
        ..Default::default()
    };
    // SAFETY: Counters follows PROCESS_MEMORY_COUNTERS, stays live for the call,
    // and the pseudo handle belongs to this current process.
    let success = unsafe {
        K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            std::mem::size_of::<Counters>() as u32,
        )
    };
    if success == 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(counters.peak)
}
#[cfg(not(target_os = "windows"))]
fn peak_working_set() -> Result<usize> {
    Err("CPU acceptance requires Windows x64".into())
}

pub fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("CPU acceptance requires a release build".into());
    }
    TRACK_ALLOCATIONS.store(false, Ordering::Relaxed);
    let mut options = BTreeMap::new();
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--acceptance" {
            continue;
        }
        if ![
            "--models-dir",
            "--corpus",
            "--output",
            "--mode",
            "--threads",
            "--warmups",
            "--samples",
            "--iterations",
        ]
        .contains(&key.as_str())
        {
            return Err(format!("unknown acceptance option {key}").into());
        }
        options.insert(key, args.next().ok_or("missing acceptance option value")?);
    }
    let required = |key: &str| -> Result<PathBuf> {
        Ok(PathBuf::from(
            options.get(key).ok_or_else(|| format!("missing {key}"))?,
        ))
    };
    let number = |key: &str, default: usize| -> Result<usize> {
        Ok(options.get(key).map_or(Ok(default), |s| s.parse())?)
    };
    let mode = options.get("--mode").map_or("validate", String::as_str);
    let budget = number("--threads", 1)?;
    if !["validate", "latency", "throughput"].contains(&mode) || ![1, 2, 4, 8].contains(&budget) {
        return Err("acceptance mode/budget is invalid".into());
    }
    let warmups = number("--warmups", 2)?;
    let samples = number("--samples", 5)?;
    let iterations = number("--iterations", 1)?;
    if warmups < 2 || samples < 5 || iterations == 0 {
        return Err(
            "acceptance needs at least two warmups, five samples and positive iterations".into(),
        );
    }
    let root = required("--corpus")?;
    let corpus: Value = serde_json::from_slice(&fs::read(root.join("corpus.json"))?)?;
    let mut images = Vec::new();
    for record in corpus["images"]
        .as_array()
        .ok_or("invalid acceptance corpus")?
    {
        let bytes = fs::read(root.join(record["file"].as_str().ok_or("missing image file")?))?;
        use sha2::Digest;
        if format!("{:x}", sha2::Sha256::digest(&bytes))
            != record["sha256"].as_str().ok_or("missing raw hash")?
        {
            return Err("acceptance raw image hash changed".into());
        }
        images.push(Image {
            id: record["id"].as_str().ok_or("missing image id")?.into(),
            width: record["width"].as_u64().ok_or("missing width")? as usize,
            height: record["height"].as_u64().ok_or("missing height")? as usize,
            bytes: Arc::new(bytes),
        });
    }
    if images.iter().map(|i| i.id.as_str()).collect::<Vec<_>>()
        != ["sample", "dense-multilingual", "text-long"]
    {
        return Err("all three fixed acceptance workloads are required".into());
    }
    let output = required("--output")?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let workers = if mode == "throughput" { budget } else { 1 };
    let capacity = if mode == "throughput" { 1 } else { budget };
    let mut cases = Vec::new();
    for (name, preset) in [
        ("rec", ModelPreset::ChineseV6Tiny),
        ("small_rec", ModelPreset::ChineseV6Small),
        ("medium_rec", ModelPreset::ChineseV6Medium),
    ] {
        let before = Instant::now();
        let models = OcrModelSet::from_directory(required("--models-dir")?, preset, true)?;
        let load_ns = before.elapsed().as_nanos() as u64;
        let before = Instant::now();
        let template = models
            .clone()
            .builder()
            .cpu_context(Arc::new(CpuContext::new(capacity)?))
            .cpu_options(CpuOptions {
                line_workers: capacity.min(4),
                preprocess_workers: capacity,
                detector_threads: capacity,
            })
            .build()?;
        let mut requests = Vec::new();
        for _ in 0..workers {
            let mut request = template.fork()?;
            if mode == "throughput" {
                request.set_cpu_context(CpuContext::serial())?;
            }
            requests.push(request);
        }
        drop(template);
        let pool = Workers::new(requests);
        let compile_requests_ns = before.elapsed().as_nanos() as u64;
        for image in &images {
            let (cold_ns, cold) = pool.round(image, 1, mode != "validate")?;
            let expected = cold[0].result.as_ref().map_err(Clone::clone)?.clone();
            check(&cold, &expected)?;
            let mut row = json!({"id":format!("{name}/{}",image.id),"result":result_json(&expected),"max_recognition_tensor_width":expected.lines.iter().map(|line|line.recognition_tensor_width).max().unwrap_or(0),"line_count":expected.lines.len(),"model_load_ns":load_ns,"compile_requests_ns":compile_requests_ns,
                "cold":if mode=="validate" {Value::Null}else{json!({"wall_ns":cold_ns,"profiles":cold.iter().map(|r|&r.profile).collect::<Vec<_>>(),"instrumented":cfg!(feature="profiling"),"includes_binding_and_reshape":true})}});
            if mode != "validate" {
                let mut warmed = Vec::new();
                for _ in 0..warmups {
                    let (_, responses) = pool.round(image, iterations, false)?;
                    check(&responses, &expected)?;
                    warmed = responses;
                }
                let warmed_memory = memory(&warmed);
                let mut timings = Vec::new();
                let mut final_responses = warmed;
                for _ in 0..samples {
                    let (wall, responses) = pool.round(image, iterations, false)?;
                    check(&responses, &expected)?;
                    timings.push(json!({"wall_ns":wall,"requests":workers*iterations,"borrowed_pipeline_ns_sum":responses.iter().map(|r|r.borrowed_ns).sum::<u64>(),"owned_copy_ns_sum":responses.iter().map(|r|r.ownership_ns).sum::<u64>()}));
                    final_responses = responses;
                }
                let after_samples = memory(&final_responses);
                let before = (
                    ALLOCATED.load(Ordering::Relaxed),
                    ALLOCATION_COUNT.load(Ordering::Relaxed),
                    REALLOCATION_COUNT.load(Ordering::Relaxed),
                );
                TRACK_ALLOCATIONS.store(true, Ordering::Relaxed);
                let allocation_probe = pool.round(image, 1, false);
                TRACK_ALLOCATIONS.store(false, Ordering::Relaxed);
                let (_, reused) = allocation_probe?;
                check(&reused, &expected)?;
                let allocations = json!({"allocated_bytes":ALLOCATED.load(Ordering::Relaxed)-before.0,"allocation_count":ALLOCATION_COUNT.load(Ordering::Relaxed)-before.1,"reallocation_count":REALLOCATION_COUNT.load(Ordering::Relaxed)-before.2,"requests":workers,"scope":"one separate owned-output worker batch, including runner/channel allocations"});
                row["samples"] = json!(timings);
                row["memory"] = json!({"warmed":warmed_memory,"after_samples":after_samples,"repeated_shape":memory(&reused),"allocation_probe":allocations,"peak_process_working_set_bytes":peak_working_set()?});
            }
            row["shared"] = json!({"normalized_weight_bytes":models.detection.normalized_weight_bytes()+models.recognition.normalized_weight_bytes()+models.classification.as_ref().map_or(0,|m|m.normalized_weight_bytes()),"packed_weight_bytes":models.detection.cached_packed_weight_bytes()+models.recognition.cached_packed_weight_bytes()+models.classification.as_ref().map_or(0,|m|m.cached_packed_weight_bytes())});
            cases.push(row);
            write(
                &output,
                &json!({"schema_version":1,"complete":false,"engine":"rust","mode":mode,"budget":budget,"workers":workers,"capacity_per_request":capacity,"cases":cases}),
            )?;
            println!("acceptance rust/{mode}/{budget}: {name}/{}", image.id);
        }
    }
    write(
        &output,
        &json!({"schema_version":1,"complete":true,"engine":"rust","mode":mode,"budget":budget,"workers":workers,"capacity_per_request":capacity,"warmups":warmups,"sample_count":samples,"iterations":iterations,"owned_outputs":true,"timed_allocation_tracking":false,"cases":cases}),
    )?;
    Ok(())
}
