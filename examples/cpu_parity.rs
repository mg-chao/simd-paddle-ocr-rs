//! Maintained raw-pixel parity and benchmark executable. See tools/cpu_parity.py.
#[path = "support/cpu_acceptance.rs"]
mod cpu_acceptance;
#[path = "support/import_contract.rs"]
mod import_contract;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{
    ModelPreset, OcrModelSet,
    cpu::CpuOptions,
    crop::CropTransform,
    detector::{DetectionBox, Detector, DetectorOptions},
    image::{ImageView, PixelFormat},
    model::Model,
    orientation::Classifier,
    preprocess::{self, ResizeWorkspace},
    recognizer::{Dictionary, Recognizer, RecognizerOptions},
    runtime::{
        CompileOptions, CompiledModel, CpuContext, ExecutionLimits, ExecutionOptions, KernelPolicy,
        LayoutPolicy, Session,
    },
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::{self, File},
    hint::black_box,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
static ALLOCATED: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static REALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static TRACK_ALLOCATIONS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
struct CountingAllocator;
// SAFETY: every allocation/deallocation delegates unchanged layouts and pointers
// to System. The counter is observational and never controls ownership.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(ptr, layout, size) };
        if !pointer.is_null() && TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(size as u64, Ordering::Relaxed);
            REALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const MODELS: [&str; 7] = [
    "cls",
    "det",
    "small_det",
    "medium_det",
    "rec",
    "small_rec",
    "medium_rec",
];

struct Image {
    id: String,
    width: usize,
    height: usize,
    pixels: Vec<u8>,
    public: bool,
    e2e: bool,
}
impl Image {
    fn view(&self) -> Result<ImageView<'_>> {
        Ok(ImageView::new(
            &self.pixels,
            self.width,
            self.height,
            self.width * 3,
            PixelFormat::Bgr24,
        )?)
    }
}

struct Capture {
    output: PathBuf,
    models: PathBuf,
    options: CompileOptions,
    cpu: Arc<CpuContext>,
    iterations: usize,
    warmups: usize,
    ocr_batch_size: usize,
    cases: Vec<Value>,
}
impl Capture {
    fn batches(&mut self, images: &[Image]) -> Result<()> {
        let ids = ["sample", "tiny", "blank", "sample"];
        let inputs = ids
            .iter()
            .map(|id| {
                images
                    .iter()
                    .find(|image| image.id == *id)
                    .ok_or("missing required batch image")?
                    .view()
            })
            .collect::<Result<Vec<_>>>()?;
        for (name, keys) in [
            ("rec", "ppocr_keys.txt"),
            ("small_rec", "rec_keys.txt"),
            ("medium_rec", "rec_keys.txt"),
        ] {
            let mut recognizer = Recognizer::new(
                self.compile(name)?,
                Arc::new(Dictionary::from_utf8(&fs::read(self.models.join(keys))?)?),
                RecognizerOptions::default(),
                ExecutionLimits::default(),
            )?;
            recognizer.set_execution_context(self.cpu.clone(), self.cpu.capacity())?;
            let standalone = inputs
                .iter()
                .map(|&image| recognizer.recognize(image, true)?.try_to_owned())
                .collect::<simd_paddle_ocr_rs::image::ImageResult<Vec<_>>>()?;
            let results = recognizer.recognize_batch(&inputs, true)?.try_to_owned()?;
            self.cases.push(json!({"id":format!("batch/{name}"),"image_ids":ids,"results":results,"standalone":standalone}));
            println!("batch {name}: four same-width inputs");
        }
        Ok(())
    }
    fn tensor(&self, relative: &str, shape: &[usize], data: &[f32]) -> Result<Value> {
        if data.len() != shape.iter().product::<usize>()
            || data.iter().any(|value| !value.is_finite())
        {
            return Err(format!("invalid captured tensor: {relative}").into());
        }
        let path = self.output.join(relative);
        fs::create_dir_all(path.parent().ok_or("tensor path has no parent")?)?;
        let mut file = BufWriter::new(File::create(path)?);
        let mut digest = Sha256::new();
        for value in data {
            let bytes = value.to_le_bytes();
            file.write_all(&bytes)?;
            digest.update(bytes);
        }
        file.flush()?;
        Ok(json!({"file":relative,"shape":shape,"sha256":format!("{:x}", digest.finalize())}))
    }
    fn compile(&self, name: &str) -> Result<Arc<CompiledModel>> {
        Ok(Arc::new(CompiledModel::compile(
            Arc::new(Model::load_path(self.models.join(format!("{name}.onnx")))?),
            self.options,
        )?))
    }
    fn graph(&mut self, name: &str) -> Result<()> {
        let load = Instant::now();
        let compiled = self.compile(name)?;
        let load_ns = load.elapsed().as_nanos() as u64;
        let shape = if name == "cls" {
            vec![1, 3, 80, 160]
        } else if name.ends_with("det") {
            vec![1, 3, 32, 32]
        } else {
            vec![1, 3, 48, 32]
        };
        let mut session = Session::new(
            compiled,
            self.cpu.clone(),
            &shape,
            ExecutionOptions {
                intra_op_threads: self.cpu.capacity(),
                ..Default::default()
            },
        )?;
        for index in 0..3 {
            let mut actual = shape.clone();
            if index == 1 && name != "cls" {
                actual[3] = 96;
            }
            let reshape = Instant::now();
            session.reshape(&actual)?;
            let reshape_ns = reshape.elapsed().as_nanos() as u64;
            let input: Vec<f32> = (0..actual.iter().product())
                .map(|i| ((i % 97) as f32 - 48.0) / 64.0)
                .collect();
            let benchmark = measure(self.iterations, self.warmups, || {
                black_box(session.run(&input)?.data());
                Ok(())
            })?;
            let id = format!("graph/{name}/{index}");
            let output = session.run(&input)?;
            let shape = output.shape();
            let tensor = self.tensor(
                &format!("{id}/output.f32"),
                shape.dimensions(),
                output.data(),
            )?;
            let mut nodes = Vec::new();
            let mut inventory: BTreeMap<(String, Vec<usize>), (u64, u64)> = BTreeMap::new();
            if index == 0 {
                session.run_with_trace(&input, |node| {
                    let tensor = self.tensor(&format!("{id}/node-{:04}.f32", node.index), node.shape.dimensions(), node.values)
                        .map_err(|error| simd_paddle_ocr_rs::runtime::RuntimeError::new(error.to_string()))?;
                    let entry = inventory.entry((node.op_type.to_owned(), node.shape.dimensions().to_vec())).or_default();
                    entry.0 += 1;
                    entry.1 = entry.1.saturating_add(node.elapsed_ns);
                    nodes.push(json!({"index":node.index,"name":node.name,"op_type":node.op_type,"lineage_key":node.key,
                        "logical_layout":node.layout,"elapsed_ns":node.elapsed_ns,"tensor":tensor}));
                    Ok(())
                })?;
            }
            let inventory: Vec<_> = inventory.into_iter().map(|((op_type, output_shape), (node_count, diagnostic_ns))|
                json!({"op_type":op_type,"output_shape":output_shape,"node_count":node_count,"diagnostic_ns":diagnostic_ns})).collect();
            self.cases.push(json!({"id":id,"input_shape":actual,"tensor":tensor,"nodes":nodes,"benchmark":benchmark,
                "diagnostic_operator_inventory":inventory,
                "load_compile_ns":load_ns,"reshape_ns":reshape_ns,"retained_activation_bytes":session.retained_activation_bytes(),
                "retained_planning_bytes":session.retained_planning_bytes(),"statistics":session.statistics()}));
            println!(
                "graph {name}/{index} {actual:?} -> {:?}",
                shape.dimensions()
            );
        }
        Ok(())
    }
    fn preprocess(&mut self, image: &Image) -> Result<()> {
        let view = image.view()?;
        let size = preprocess::compute_det_size(image.width, image.height, 320)?;
        let mut workspace = ResizeWorkspace::default();
        let mut det = vec![0.0; 3 * size.width * size.height];
        preprocess::det(view, size.width, size.height, &mut det, &mut workspace)?;
        let tensor = self.tensor(
            &format!("preprocess/det/{}.f32", image.id),
            &[1, 3, size.height, size.width],
            &det,
        )?;
        self.cases
            .push(json!({"id":format!("preprocess/det/{}",image.id),"tensor":tensor}));
        let mut cls = vec![0.0; 3 * 80 * 160];
        preprocess::cls(view, &mut cls, &mut workspace)?;
        let tensor = self.tensor(
            &format!("preprocess/cls/{}.f32", image.id),
            &[1, 3, 80, 160],
            &cls,
        )?;
        self.cases
            .push(json!({"id":format!("preprocess/cls/{}",image.id),"tensor":tensor}));
        let mut rec = vec![0.0; 3 * 48 * 320];
        preprocess::rec(view, 320, &mut rec, &mut workspace)?;
        let tensor = self.tensor(
            &format!("preprocess/rec/{}.f32", image.id),
            &[1, 3, 48, 320],
            &rec,
        )?;
        self.cases
            .push(json!({"id":format!("preprocess/rec/{}",image.id),"tensor":tensor}));
        let (w, h) = (image.width as f32, image.height as f32);
        let points = if w >= 9.0 && h >= 9.0 {
            [
                [1.25, 2.0],
                [w - 2.0, 1.5],
                [w - 1.0, h - 2.0],
                [0.5, h - 1.0],
            ]
        } else {
            [
                [0.0, 0.0],
                [w - 1.0, 0.0],
                [w - 1.0, h - 1.0],
                [0.0, h - 1.0],
            ]
        };
        let crop = CropTransform::new(&DetectionBox { points, score: 1.0 })?;
        let mut pixels = vec![0; crop.byte_count()?];
        crop.extract_into(view, &mut pixels)?;
        let floats: Vec<_> = pixels.into_iter().map(f32::from).collect();
        let tensor = self.tensor(
            &format!("crop/{}.f32", image.id),
            &[crop.height(), crop.width(), 3],
            &floats,
        )?;
        self.cases
            .push(json!({"id":format!("crop/{}",image.id),"tensor":tensor}));
        Ok(())
    }
    fn public(&mut self, images: &[Image]) -> Result<()> {
        let cls = self.compile("cls")?;
        let mut classifier = Classifier::new(cls.clone(), ExecutionLimits::default())?;
        classifier.set_execution_context(self.cpu.clone(), 1)?;
        for image in images.iter().filter(|image| image.public) {
            let view = image.view()?;
            let benchmark = measure(self.iterations, self.warmups, || {
                black_box(classifier.classify(view)?);
                Ok(())
            })?;
            let result = classifier.classify(view)?;
            self.cases.push(json!({"id":format!("classify/{}",image.id),"result":{"label":result.label,"score":result.score,"resized_width":result.resized_width},"benchmark":benchmark}));
        }
        for (det_name, rec_name, keys) in [
            ("det", "rec", "ppocr_keys.txt"),
            ("small_det", "small_rec", "rec_keys.txt"),
            ("medium_det", "medium_rec", "rec_keys.txt"),
        ] {
            let det = self.compile(det_name)?;
            let rec = self.compile(rec_name)?;
            let dictionary = Arc::new(Dictionary::from_utf8(&fs::read(self.models.join(keys))?)?);
            let detector_options = DetectorOptions {
                limit_side_length: 320,
                ..Default::default()
            };
            let mut detector =
                Detector::new(det.clone(), detector_options, ExecutionLimits::default())?;
            detector.set_execution_context(self.cpu.clone(), self.cpu.capacity())?;
            let mut recognizer = Recognizer::new(
                rec.clone(),
                dictionary.clone(),
                RecognizerOptions::default(),
                ExecutionLimits::default(),
            )?;
            recognizer.set_execution_context(self.cpu.clone(), self.cpu.capacity())?;
            let preset = match rec_name {
                "rec" => ModelPreset::ChineseV6Tiny,
                "small_rec" => ModelPreset::ChineseV6Small,
                _ => ModelPreset::ChineseV6Medium,
            };
            let mut ocr = OcrModelSet::from_directory(&self.models, preset, true)?
                .builder()
                .compile_options(self.options)
                .detector_options(detector_options)
                .cpu_context(self.cpu.clone())
                .recognition_batch_size(self.ocr_batch_size)
                .cpu_options(CpuOptions {
                    line_workers: self.cpu.capacity().min(4),
                    preprocess_workers: self.cpu.capacity(),
                    detector_threads: self.cpu.capacity(),
                })
                .build()?;
            for image in images.iter().filter(|image| image.public) {
                let view = image.view()?;
                let benchmark = measure(self.iterations, self.warmups, || {
                    black_box(recognizer.recognize(view, true)?);
                    Ok(())
                })?;
                let result = recognizer.recognize(view, true)?;
                self.cases.push(json!({"id":format!("recognize/{rec_name}/{}",image.id),"result":serde_json::to_value(result)?,"benchmark":benchmark,
                    "retained_activation_bytes":recognizer.retained_activation_bytes(),"retained_workspace_bytes":recognizer.retained_workspace_bytes()}));
                let benchmark = measure(self.iterations, self.warmups, || {
                    black_box(detector.detect(view)?);
                    Ok(())
                })?;
                let result = detector.detect(view)?.try_to_owned()?;
                self.cases.push(json!({"id":format!("detect/{det_name}/{}",image.id),"result":serde_json::to_value(result)?,"benchmark":benchmark,
                    "retained_activation_bytes":detector.retained_activation_bytes(),"retained_workspace_bytes":detector.retained_workspace_bytes()}));
                if !image.e2e {
                    continue;
                }
                let benchmark = measure(self.iterations, self.warmups, || {
                    black_box(ocr.run(view, true)?);
                    Ok(())
                })?;
                let result = ocr.run(view, true)?;
                let serialized = json!({"text":result.text,"detected_count":result.detected_count,
                    "detector_resized_width":result.detector_resized_width,"detector_resized_height":result.detector_resized_height,
                    "lines":result.iter().collect::<Vec<_>>()});
                #[cfg(feature = "profiling")]
                let stage_profile = if self.iterations == 0 {
                    Value::Null
                } else {
                    let (_, profile) = ocr.run_profiled(view, true)?;
                    serde_json::to_value(profile)?
                };
                #[cfg(not(feature = "profiling"))]
                let stage_profile = Value::Null;
                self.cases.push(json!({"id":format!("ocr/{rec_name}/{}",image.id),"result":serialized,"benchmark":benchmark,
                    "diagnostic_production_stage_profile":stage_profile,
                    "retained_activation_bytes":ocr.retained_activation_bytes(),"retained_workspace_bytes":ocr.retained_workspace_bytes()}));
            }
            println!(
                "public {rec_name}: {} images",
                images.iter().filter(|image| image.public).count()
            );
        }
        Ok(())
    }
}

fn measure(
    iterations: usize,
    warmups: usize,
    mut operation: impl FnMut() -> Result<()>,
) -> Result<Value> {
    // Every caller captures its output immediately afterward. That is the only
    // production inference in this mode; logical traces remain as diagnostics.
    if iterations == 0 {
        return Ok(json!({"measured":false,"reason":"correctness-only"}));
    }
    let cold_bytes_before = ALLOCATED.load(Ordering::Relaxed);
    let cold_allocations_before = ALLOCATION_COUNT.load(Ordering::Relaxed);
    let cold_reallocations_before = REALLOCATION_COUNT.load(Ordering::Relaxed);
    let start = Instant::now();
    operation()?;
    let cold = start.elapsed().as_nanos() as u64;
    let cold_allocated_bytes = ALLOCATED.load(Ordering::Relaxed) - cold_bytes_before;
    let cold_allocation_count = ALLOCATION_COUNT.load(Ordering::Relaxed) - cold_allocations_before;
    let cold_reallocation_count =
        REALLOCATION_COUNT.load(Ordering::Relaxed) - cold_reallocations_before;
    for _ in 0..warmups {
        operation()?;
    }
    let mut warm = Vec::with_capacity(iterations);
    let mut allocations = Vec::with_capacity(iterations);
    let mut allocation_counts = Vec::with_capacity(iterations);
    let mut reallocation_counts = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let before = ALLOCATED.load(Ordering::Relaxed);
        let allocation_count_before = ALLOCATION_COUNT.load(Ordering::Relaxed);
        let reallocation_count_before = REALLOCATION_COUNT.load(Ordering::Relaxed);
        let start = Instant::now();
        operation()?;
        let elapsed = start.elapsed().as_nanos() as u64;
        let allocated = ALLOCATED.load(Ordering::Relaxed) - before;
        let allocation_count = ALLOCATION_COUNT.load(Ordering::Relaxed) - allocation_count_before;
        let reallocation_count =
            REALLOCATION_COUNT.load(Ordering::Relaxed) - reallocation_count_before;
        warm.push(elapsed);
        allocations.push(allocated);
        allocation_counts.push(allocation_count);
        reallocation_counts.push(reallocation_count);
    }
    Ok(
        json!({"cold_ns":cold,"cold_allocated_bytes":cold_allocated_bytes,
        "cold_allocation_count":cold_allocation_count,"cold_reallocation_count":cold_reallocation_count,
        "warmup_runs":warmups,"warm_ns":warm,"allocated_bytes":allocations,
        "allocation_counts":allocation_counts,"reallocation_counts":reallocation_counts}),
    )
}

fn load_corpus(path: &Path, quick: bool) -> Result<Vec<Image>> {
    let manifest: Value = serde_json::from_slice(&fs::read(path.join("corpus.json"))?)?;
    if manifest["schema_version"] != 1 {
        return Err("unsupported corpus schema".into());
    }
    let mut images = Vec::new();
    for entry in manifest["images"].as_array().ok_or("missing images")? {
        if quick && entry["quick"] != true {
            continue;
        }
        let pixels = fs::read(path.join(entry["file"].as_str().ok_or("missing file")?))?;
        if format!("{:x}", Sha256::digest(&pixels))
            != entry["sha256"].as_str().ok_or("missing corpus checksum")?
        {
            return Err("corpus checksum mismatch".into());
        }
        let width = entry["width"].as_u64().ok_or("missing width")? as usize;
        let height = entry["height"].as_u64().ok_or("missing height")? as usize;
        if pixels.len() != width * height * 3 {
            return Err("corpus length mismatch".into());
        }
        images.push(Image {
            id: entry["id"].as_str().ok_or("missing id")?.into(),
            width,
            height,
            pixels,
            public: entry["public_case"] == true,
            e2e: entry["e2e_case"] == true,
        });
    }
    if images.is_empty() {
        return Err("empty corpus".into());
    }
    Ok(images)
}

fn main() -> Result<()> {
    if std::env::args().any(|arg| arg == "--acceptance") {
        return cpu_acceptance::main();
    }
    let mut options = BTreeMap::new();
    let mut flags = BTreeSet::new();
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--quick"
            || key == "--imports-only"
            || key == "--batch-only"
            || key == "--correctness-only"
        {
            flags.insert(key);
        } else if [
            "--models-dir",
            "--corpus",
            "--output",
            "--mode",
            "--iterations",
            "--threads",
            "--backend",
            "--ocr-batch-size",
        ]
        .contains(&key.as_str())
        {
            options.insert(key, args.next().ok_or("missing option value")?);
        } else {
            return Err(format!("unknown argument: {key}").into());
        }
    }
    let required = |key: &str| -> Result<PathBuf> {
        let value = options.get(key).ok_or_else(|| format!("missing {key}"))?;
        Ok(PathBuf::from(value))
    };
    if flags.contains("--imports-only") {
        for model in MODELS {
            import_contract::write(
                &required("--models-dir")?.join(format!("{model}.onnx")),
                &required("--output")?,
                model,
            )?;
        }
        return Ok(());
    }
    let mode = options.get("--mode").map_or("auto", String::as_str);
    let mut compile_options = match mode {
        "reference" => CompileOptions::reference(),
        "simd" => CompileOptions {
            layout: LayoutPolicy::Nchw,
            fusion: false,
            ..Default::default()
        },
        "nhwc" => CompileOptions {
            fusion: false,
            ..Default::default()
        },
        "auto" => CompileOptions::default(),
        _ => return Err("mode must be reference, simd, nhwc or auto".into()),
    };
    let backend = options.get("--backend").map_or("auto", String::as_str);
    let kernel_policy = match backend {
        "auto" => KernelPolicy::Auto,
        "scalar" => KernelPolicy::Scalar,
        "portable4" => KernelPolicy::Portable4,
        "avx" => KernelPolicy::Avx,
        "avx-fma" => KernelPolicy::AvxFma,
        "avx2-fma" => KernelPolicy::Avx2Fma,
        "avx512" => KernelPolicy::Avx512,
        _ => {
            return Err(
                "backend must be auto, scalar, portable4, avx, avx-fma, avx2-fma or avx512".into(),
            );
        }
    };
    if mode != "reference" {
        compile_options.kernels = kernel_policy;
    }
    let iterations = options
        .get("--iterations")
        .map_or(Ok(5), |n| n.parse::<usize>())?;
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }
    let threads = options
        .get("--threads")
        .map_or(Ok(1), |n| n.parse::<usize>())?;
    if !(1..=16).contains(&threads) {
        return Err("threads must be within 1..=16".into());
    }
    let ocr_batch_size = options
        .get("--ocr-batch-size")
        .map_or(Ok(1), |n| n.parse::<usize>())?;
    if !(1..=i32::MAX as usize).contains(&ocr_batch_size) {
        return Err("ocr-batch-size must be within 1..=i32::MAX".into());
    }
    let images = load_corpus(&required("--corpus")?, flags.contains("--quick"))?;
    let mut capture = Capture {
        output: required("--output")?,
        models: required("--models-dir")?,
        options: compile_options,
        cpu: Arc::new(CpuContext::new(threads)?),
        iterations: if flags.contains("--correctness-only") {
            0
        } else {
            iterations
        },
        warmups: if mode == "reference" { 0 } else { 2 },
        ocr_batch_size,
        cases: Vec::new(),
    };
    fs::create_dir_all(&capture.output)?;
    if flags.contains("--batch-only") {
        capture.batches(&images)?;
    } else {
        for model in MODELS {
            capture.graph(model)?;
        }
        for image in &images {
            capture.preprocess(image)?;
        }
        capture.public(&images)?;
    }
    let data = json!({"schema_version":1,"engine":"rust","mode":mode,"metadata":{"architecture":std::env::consts::ARCH,"os":std::env::consts::OS,
        "processor_count":std::thread::available_parallelism().map_or(1,usize::from),"threads":threads,
        "requested_backend":backend,"compile_options":compile_options,
        "profiling_enabled":cfg!(feature="profiling"),
        "correctness_only":flags.contains("--correctness-only"),
        "ocr_batch_size":ocr_batch_size,
        "corpus_sha256":format!("{:x}",Sha256::digest(fs::read(required("--corpus")?.join("corpus.json"))?)),
        "images_decoded_inside_timing":false},"cases":capture.cases});
    fs::write(
        capture.output.join("capture.json"),
        serde_json::to_vec_pretty(&data)?,
    )?;
    println!(
        "Rust {mode}: {} cases -> {}",
        data["cases"].as_array().unwrap().len(),
        capture.output.display()
    );
    Ok(())
}
