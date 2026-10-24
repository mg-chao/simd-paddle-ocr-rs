# Using the CPU implementation

The supported presets use the three Chinese PP-OCRv6 model bundles from the
pinned upstream checkout. Inference is entirely Rust CPU code. Consumers need
no .NET runtime, ONNX Runtime, Python, GPU runtime, or image decoder in the library.

Native Apple Silicon setup and local checks are documented in
[macOS CPU support](macos-cpu-validation.md).

## Toolchain and model files

Build with `nightly-2026-09-08`, pinned in `rust-toolchain.toml`, because the
library uses Rust's portable SIMD feature. Cargo honors this pin inside this
repository. A consuming project must also select that nightly (for example,
`cargo +nightly-2026-09-08 build --release`). No custom `RUSTFLAGS` are needed.
The default `parallel` feature uses a shared Rayon CPU pool. Disable default
features for serial execution without Rayon; `profiling` adds diagnostic APIs.

`CompileOptions::default()` selects the native CPU backend. On AArch64 this is
`KernelPolicy::NeonFma` (`neon_fma` in serialized policy values), with ordered
NEON FMA convolution and matrix kernels. `KernelPolicy::Portable4` retains
separate multiply/add for numerical and performance comparisons; forcing
`NeonFma` on an unsupported CPU fails during model compilation. x86 automatic
dispatch selects supported AVX/FMA backends. FMA can change F32 rounding, so
compare text and numeric output when selecting a policy.

For deployment builds, the application's root `Cargo.toml` can opt into:

```toml
[profile.release]
lto = "thin"
codegen-units = 1
```

These settings can increase build time; compare them on your deployment
workload. A library dependency cannot select the consuming application's
release profile.

Models are external assets. One reproducible way to acquire the supported
files is to clone the pinned reference repository from the Rust project root:

```powershell
git clone https://github.com/mg-chao/SimdPaddleOCR.git ../SimdPaddleOCR
git -C ../SimdPaddleOCR checkout 33ff8a81672452041d110e8ba52dcd2639e54228
```

Use its `models/` directory directly, or copy the needed files into your own
asset directory. Cloning and using these files does not require building C# or
installing .NET. Runtime loading never downloads models. The model assets
and dictionaries have upstream terms described by the reference
repository's [third-party notices](https://github.com/mg-chao/SimdPaddleOCR/blob/33ff8a81672452041d110e8ba52dcd2639e54228/THIRD-PARTY-NOTICES.md).

| Preset | Detector | Recognizer | Dictionary |
| --- | --- | --- | --- |
| `ChineseV6Tiny` / CLI `tiny` | `det.onnx` | `rec.onnx` | `ppocr_keys.txt` |
| `ChineseV6Small` / CLI `small` | `small_det.onnx` | `small_rec.onnx` | `rec_keys.txt` |
| `ChineseV6Medium` / CLI `medium` | `medium_det.onnx` | `medium_rec.onnx` | `rec_keys.txt` |

All presets use `cls.onnx` when classification is requested.
`OcrModelSet::from_directory(path, preset, true)` requires that file: absence or
invalid content returns an error. Pass `false` to omit it deliberately, or
construct `OcrModelSet` with `classification: None`. Building from such a model
set succeeds with classification omitted even though the builder defaults to
classification enabled. `.direction_classification(false)` also skips a
classifier that was already loaded. With no classifier, OCR line metadata has
classification score and label zero, and classifier rotation zero. Perspective
cropping still rotates tall lines 90 degrees counterclockwise, so top-to-bottom
text reaches recognition in left-to-right order even without classification.

`OcrModelSet::from_paths` accepts explicit model filenames, and `from_readers`
accepts non-seekable buffered streams. Parsing releases the raw ONNX input and
shares immutable models/dictionaries through `Arc`. The optional classifier is
`Option<&Path>` for paths and `Option<&mut dyn BufRead>` for readers, so `None`
needs no type annotation. For example:

```rust,no_run
use simd_paddle_ocr_rs::OcrModelSet;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let models = OcrModelSet::from_paths("det.onnx", "rec.onnx", "keys.txt", None)?;
# Ok(())
# }
```

### Model asset hashes

These SHA-256 hashes identify the supported files at the pinned upstream
revision. Dictionary hashes cover LF line endings; CRLF alternatives are
listed separately because checkouts can convert text files.

| File | SHA-256 |
| --- | --- |
| `cls.onnx` | `dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2` |
| `det.onnx` | `193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8` |
| `medium_det.onnx` | `eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1` |
| `medium_rec.onnx` | `9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba` |
| `ppocr_keys.txt` | `46e1b34ef45684cb46d75ac76d355341fe7f0a2c38d6ee02e63ae6b3878019fc` |
| `ppocr_keys.txt` (CRLF) | `5911341b8d8bdef3924e1ab7c85094f09cf05107d9983476a10c0ee57dac1d82` |
| `rec_keys.txt` | `b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d` |
| `rec_keys.txt` (CRLF) | `769e7fa79bb297b5f18d8dbd149e364a45bc61f2b3f574e5ea836f0b261c23a6` |
| `rec.onnx` | `9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6` |
| `small_det.onnx` | `d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e` |
| `small_rec.onnx` | `5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634` |

## Run an image

The `ocr` example decodes JPEG/PNG with the development-only `image`
dependency. It sniffs the file format and does not apply EXIF orientation.
Library users choose their own decoder and orientation policy.

```powershell
cargo run --release --locked --example ocr -- --models-dir ../SimdPaddleOCR/models --preset tiny input.jpg
cargo run --release --locked --example ocr -- --models-dir ../SimdPaddleOCR/models --preset small --alignment input.jpg
cargo run --release --locked --example ocr -- --models-dir ../SimdPaddleOCR/models --preset medium --no-cls --threads 1 --json input.png
```

Default output is joined text in detector reading order. `--json` prints owned
line metadata, quadrilaterals, scores and text. `--alignment` implies JSON and
adds CTC spans plus `character_boxes`, an array of token-box arrays in line
order. These boxes are estimates from CTC time intervals, not independently
detected glyph boundaries; dictionary tokens may contain multiple characters.
Missing files, decode errors, invalid settings, and inference failures return
a nonzero process exit code. Use `--help` for all flags.

## Borrowed pixels and results

Detector resizing aligns both tensor dimensions to 32 pixels and bounds each
side by `preprocess::DET_MAX_SIDE` (4096) before allocating or running the model.
Minimum-side scaling enlarges the short side toward its requested limit; the
4096-pixel long-side bound takes precedence for narrow images. Maximum-side
scaling keeps its configured limit within the same bound. Very small images
receive right/bottom padding before resizing. Detection boxes and recognition
crops remain in original-image coordinates; padding is clipped rather than
treated as an image scale. Frozen detector-map replay applies the same mapping.

```rust,no_run
use simd_paddle_ocr_rs::{ModelPreset, OcrModelSet, RunOptions};
use simd_paddle_ocr_rs::image::{ImageView, PixelFormat};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let models = OcrModelSet::from_directory("models", ModelPreset::ChineseV6Tiny, true)?;
let plan = models.builder().build()?;
let mut request = plan.request()?;
let pixels = vec![255u8; 640 * 480 * 3]; // Application-owned decoded pixels.
let image = ImageView::new(&pixels, 640, 480, 0, PixelFormat::Rgb24)?;
let result = request.run(image, RunOptions::default())?;
println!("{}", result.text);
let saved = result.try_to_owned()?; // May outlive the next request.run().
let _next = request.run(image, RunOptions::default())?;
println!("{}", saved.text);
# Ok(())
# }
```

`ImageView` borrows interleaved, top-to-bottom rows. It accepts `Bgr24`, `Rgb24`,
`Bgra32`, and `Rgba32`; alpha is ignored rather than composited. Width/height
must be positive. Stride is bytes between row starts; zero means tightly
packed. A nonzero stride may include padding, but must be at least
`width * bytes_per_pixel`. The slice must contain at least
`(height - 1) * stride + width * bytes_per_pixel` bytes. For a subimage, supply
a slice starting at its first pixel with the parent row stride. Negative
strides and planar pixel layouts are not accepted.

`OcrRequest::run` reuses mutable activation, crop and result buffers. Its `OcrView`
borrows that request, so finish reading it before the next mutable call.
`try_to_owned` copies output into an independent `OwnedOcr` and can fail on
allocation. Pass `RunOptions { capture_spans: true }` to retain CTC spans; then call
`line.estimate_character_boxes()`. Without capture, that method returns an
error, including on an empty recognized line. Errors return no partial OCR
result; a later valid call can reuse the request.

## CPU capacity and concurrent requests

The builder defaults to `CpuContext::shared_default()`: requests share one
process-wide pool sized from available parallelism. Use
`.cpu_context(CpuContext::serial())` for deterministic serial admission, or
`.cpu_context(Arc::new(CpuContext::new(8)?))` for an explicit shared capacity.
Reuse the same `Arc` across pipelines to share the worker cap. Independent
contexts have independent capacities.

`CpuOptions` controls maximum line workers, detector threads and preprocessing
workers. Zero selects automatic stage budgets; positive budgets are capped by
the context capacity and 16. Automatic OCR starts with up to four recognition
line workers. With AVX2/FMA or the native NEON backend, a context of at least
eight workers and at least eight independent recognition units selects eight line workers
and correspondingly less nested kernel parallelism. Windows x64 AVX2/FMA contexts
with 24–31 workers select sixteen single-threaded recognition owners when there
are at least 32 units.
A same-width batch counts
as one unit. Selection depends on the page and configuration, so concurrent
request timing does not change line ownership. `OcrRequest::cpu_budgets()` reports the
selected budgets after a run. More line workers retain additional activation
and workspace buffers; choose a capacity that fits the deployment memory budget.
On NEON, automatic scheduling with eight owners and a base kernel budget of
one can give one uniquely dominant, single-crop owner two kernel workers when
the request has its full CPU share. The shared pool still bounds execution;
jobs and recognizer buffers keep their owners. Explicit positive line-worker
settings retain their selected policy.
`cpu_budgets()` reports the base budget; profiling owner records expose the
individual kernel budgets.
Automatic classifiers can use up to sixteen single-threaded workers, bounded
by the request's available CPU share. A positive `line_workers` bounds both stages.
Standalone line extractors retain the default four-classifier maximum. Nested
recognizer work shares the context. `OcrRequest::fork()` creates another mutable
request sharing compiled models, packed weights, dictionary and CPU capacity,
while keeping its own activations and buffers. Recognition batching is opt-in
with `.recognition_batch_size(n)` and groups crops with the same selected
width. Larger batches and more simultaneous requests increase memory use.

With classification enabled, OCR completes classification and any 180-degree
rotation before the recognition wave. Recognition keeps stable width-based
line-worker ownership and supports same-width batches. Standalone extraction
completes orientation for every returned crop, and trace callbacks retain their
deterministic ordering on the calling thread.

OCR calls register their CPU demand before entering the shared executor. At
stage boundaries, each call limits detector/preprocessing workers, classifiers
and nested recognition threads to its share of the currently requested CPU
capacity.
Line-worker assignments remain stable, so concurrent traffic does not create
extra recognition buffers. A single call retains the configured budgets;
`cpu_budgets()` reports recognition, detector and preprocessing maxima;
automatic classifiers use the current request share. The fixed pool enforces the overall cap
even when demand changes after a stage begins. Standalone detector/classifier/
recognizer calls retain their explicit budgets.

For a service, configure an `OcrEngine` before sharing it:

```rust,no_run
use simd_paddle_ocr_rs::{ModelPreset, OcrModelSet, RunOptions};
use simd_paddle_ocr_rs::engine::EngineLimits;
use simd_paddle_ocr_rs::cpu::{CpuContext, CpuOptions};
use std::sync::Arc;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let models = OcrModelSet::from_directory("models", ModelPreset::ChineseV6Tiny, true)?;
let engine = models.builder()
    .cpu_context(Arc::new(CpuContext::new(8)?))
    .cpu_options(CpuOptions {
        line_workers: 4, detector_threads: 8, preprocess_workers: 8,
    })
    .build_engine(EngineLimits { max_requests: 2, max_idle_bytes: 512 << 20 })?;
// Share Arc::new(engine) with callers. engine.run_owned(image, RunOptions::default()) copies
// a result before releasing its request; engine.acquire() offers borrowed use.
println!("{:?}", engine.stats());
# Ok(())
# }
```

`acquire()` waits outside the executor for a request slot; `try_acquire()`
returns `None` immediately when full. A lease returns its request on drop and
ties borrowed results to that lease. Blocking acquisition from an executing
context or trace callback is rejected to avoid deadlock; use nonblocking
acquisition there. Configure options on the builder/plan before creating
the engine.

## Resource limits and supported model shapes

Set limits for your input policy rather than treating defaults as a total
process memory budget:

| API | Limit |
| --- | --- |
| `DetectorOptions` / `RecognizerOptions` | Source image pixel limits; defaults are 40 million per stage |
| `LineOptions` | Per-crop pixels, classifier image pixels, total crop bytes, classifier execution limits |
| `ExecutionLimits` via builder detector/recognizer limits | F32 activation capacity per session, default 1 GiB; excludes weights, plans and scratch |
| `RecognizerOptions::max_result_bytes` / `OcrOptions::max_total_result_bytes` | Active text, alignment and OCR metadata |
| `EngineLimits` | Maximum admitted requests and aggregate idle request capacities |

`max_idle_bytes` excludes shared weights, is checked when a lease returns, and
zero disables request caching. It does not limit peak memory of active
requests. Transactional shape changes may temporarily hold old and new
allocations. Kernels retain reusable scratch and requests retain high-water
capacities; the `retained_*_bytes()` APIs expose these capacities. Kernel
scratch is already included in retained workspace bytes, so do not add it
twice. Low-level `Model::load_*_with_limits` accepts `onnx::LoadLimits` for
input-byte, protobuf nesting, tensor rank, graph-node and shape-program limits.
Defaults are 1 GiB of input, nesting depth 64, rank 8, 100,000 source nodes and
64 combined shape-program nodes. The shape-node limit may be tightened below 64.
`max_constant_bytes` caps the normalization constant store (default 1 GiB), and
`max_generated_bytes` caps cumulative generated coefficients (default 1 GiB).
Generated arrays are reserved fallibly before rewriting the graph. An optional
rewrite that exceeds either budget is skipped; oversized source constants are
rejected. Branch merging also requires the merged dense kernel to contain no
more coefficients than its source kernels. These limits are separate from
process memory: protobuf metadata, small shape-program copies, allocator
capacity and concurrent imports also use memory.

Recognition result caches retain one UTF-8 buffer and compact alignment offsets
per line. Shorter text, capture toggles and smaller pages retain the high-water
capacity for later reuse. Inactive line caches are included in retained
workspace accounting and the engine's idle-cache budget.

This importer/runtime supports the OCR subset used by the shipped bundles,
with declared standard-domain source opsets 7 through 18. Domains, opsets,
attribute names and source topology are checked before nodes can be erased.
Resize supports spatial scales 2, 4 and 8 with nearest/asymmetric/floor
sampling; opset 11+ defaults are half_pixel/round_prefer_floor, which this
runtime rejects unless the model explicitly selects the supported modes.
Resize 10 uses its legacy asymmetric/floor defaults. GlobalAveragePool reduces
all spatial axes of a rank-three-through-eight input.
Graphs require one F32 input and output and supported normalized operators;
arbitrary ONNX graphs are not guaranteed to load. Detection requires a
`[1, 1, H, W]` probability map matching input spatial dimensions. Recognition
requires the supported MatMul, optional bias, last-axis Softmax CTC head,
with its class count matching the dictionary. The classifier requires the
shipped two-class MatMul/optional bias/Softmax head. An unsupported head or
operator returns an error. Dictionary files are UTF-8; their labels are kept
without trimming and blank/space classes are implicit. Keep each dictionary
paired with its model.

Default recognition is height 48 with adaptive width rounded up to 32 and no
320-pixel upper cap. Use `RecognizerOptions { adaptive_width: false,
target_width: 320, ..Default::default() }` for fixed width. The default detector
limit is 960 and its inferred box threshold is 0.4 for tiny, 0.45 for the other
supported bundles. Validate accuracy and performance with your deployment
models and images on each target CPU and operating system. See
[macOS CPU support](macos-cpu-validation.md) for Apple Silicon setup.

Applications can set `RecognizerOptions::min_width` to impose an adaptive
width floor before 32-pixel alignment (for example, 320 or 48). Its default is
zero, preserving natural-width selection; fixed-width mode ignores it.
`DetectorOptions::resize_policy` defaults to `DetResizePolicy::Max`, which
reduces the longest side to `limit_side_length` only when necessary. `Min`
enlarges the shortest side to that limit only when necessary. Both align the
result to 32 pixels. `Detector::set_resize_options` validates and updates these
two controls without recompiling weights. Requests, forks, and factories retain
the configured controls. The integer shape evaluator also accepts scalar/vector
Reshape metadata, including the singleton-to-scalar shape programs in Snow Shot's
original Small and Medium PP-OCRv6 recognition models.

## Import policy and API details

Import performs optional BN folding and convolution-branch merging by default.
These rewrites can change floating-point rounding. For an independent arithmetic
reference, disable both import rewrites and execution optimizations:

```rust,no_run
use simd_paddle_ocr_rs::model::{LoadOptions, Model, OptimizationPolicy};
use simd_paddle_ocr_rs::runtime::{CompileOptions, CpuCompiledModel};
use std::sync::Arc;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let model = Model::load_path_with_options("det.onnx", &LoadOptions {
    optimization: OptimizationPolicy::PreserveArithmetic,
    ..Default::default()
})?;
let compiled = CpuCompiledModel::compile(Arc::new(model), CompileOptions::reference())?;
# Ok(())
# }
```

`CompileOptions::reference()` alone preserves the arithmetic of an already
imported model. Structural alias removal and shape lowering still occur in
both import modes.

The normalized graph now stores `Node::operator: Operator`; use `op_type()`
for its name and match the enum for its parameters. This replaces the separate
`op_type` and `EffectiveAttributes` fields. Classifier and recognizer heads
share the same structural validator.

Construct `CpuBudgets` with `CpuOptions::resolve`, and read values through
accessors such as `line_workers()`. `OrientedLines` and `OwnedOrientedLines`
provide checked `new` constructors, `metadata()` and `pixels()` accessors, and
an immutable borrowed view. Iteration includes every validated line; only an
out-of-range `get` returns `None`. Their storage fields are private.

`Tensor::defaulted` replaces the generic `Tensor::zeros`: it fills elements
with `T::default()`, which is not necessarily numeric zero. The public error
types expose `kind()` and retain concrete causes through `Error::source()`;
inspect kinds and sources rather than comparing error strings. Errors remain
cloneable but no longer implement equality. The shared CPU context caches
only successful initialization, allowing a later call to retry a transient
pool-creation failure.

### Confidence probabilities

`RecognizerOptions::score_mode` defaults to `RecognitionScoreMode::Logit`,
which preserves the compact projection path and its uncalibrated maxima. Set
`RecognitionScoreMode::Probability` when applying confidence thresholds. This
executes the model's complete Softmax head and averages emitted-token
probabilities in `[0, 1]`; dense activations count toward the execution memory
limit. Single, batched, traced, forked and factory-created requests preserve
this setting. `projection_statistics()` reports compact projection selection
only, so it returns `None` in probability mode.
