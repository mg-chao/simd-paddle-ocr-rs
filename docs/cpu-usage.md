# Using the CPU implementation

The initial qualification target is Windows x64 and the three shipped Chinese
PP-OCRv6 model bundles. Inference is entirely Rust CPU code. Consumers need no
.NET runtime, ONNX Runtime, Python, GPU runtime, or image decoder in the library.
Correctness qualification and performance qualification are separate; consult
the [baseline guide](cpu-baseline.md) for recorded validation and reproducible
comparison commands.

## Toolchain and model files

Build with `nightly-2026-09-08`, pinned in `rust-toolchain.toml`, because the
library uses Rust's portable SIMD feature. Cargo honors this pin inside this
repository. A consuming project must also select that nightly (for example,
`cargo +nightly-2026-09-08 build --release`). No custom `RUSTFLAGS` are needed.
The default `parallel` feature uses a shared Rayon CPU pool. Disable default
features for serial execution without Rayon; `profiling` adds diagnostic APIs.

Models are external assets. One reproducible way to acquire the supported
files is to clone the pinned reference repository from the Rust project root:

```powershell
git clone https://github.com/mg-chao/SimdPaddleOCR.git ../SimdPaddleOCR
git -C ../SimdPaddleOCR checkout 33ff8a81672452041d110e8ba52dcd2639e54228
```

Use its `models/` directory directly, or copy the needed files into your own
asset directory. Cloning and using these files does not require building C# or
installing .NET. Runtime loading never downloads models. Exact supported
model/dictionary SHA-256 hashes are recorded in
[`tests/baseline/manifest.json`](../tests/baseline/manifest.json). The model
assets and dictionaries have upstream terms described by the reference
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
cropping's vertical-line rotation still applies.

`OcrModelSet::from_paths` accepts explicit model filenames, and `from_readers`
accepts non-seekable buffered streams. Parsing releases the raw ONNX input and
shares immutable models/dictionaries through `Arc`.

## Run an image

The `ocr` example decodes JPEG/PNG with the existing development-only `image`
dependency. It sniffs the file format and does not apply EXIF orientation.
Library users choose their own decoder and orientation policy.

```powershell
cargo run --release --locked --example ocr -- --models-dir ../SimdPaddleOCR/models --preset tiny tests/baseline/corpus/imgs_1.jpg
cargo run --release --locked --example ocr -- --models-dir ../SimdPaddleOCR/models --preset small --alignment tests/baseline/corpus/imgs_00006737.jpg
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

```rust,no_run
use simd_paddle_ocr_rs::{ModelPreset, OcrModelSet};
use simd_paddle_ocr_rs::image::{ImageView, PixelFormat};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let models = OcrModelSet::from_directory("models", ModelPreset::ChineseV6Tiny, true)?;
let mut request = models.builder().build()?;
let pixels = vec![255u8; 640 * 480 * 3]; // Application-owned decoded pixels.
let image = ImageView::new(&pixels, 640, 480, 0, PixelFormat::Rgb24)?;
let result = request.run(image, false)?;
println!("{}", result.text);
let saved = result.try_to_owned()?; // May outlive the next request.run().
let _next = request.run(image, false)?;
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

`Ocr::run` reuses mutable activation, crop and result buffers. Its `OcrView`
borrows that request, so finish reading it before the next mutable call.
`try_to_owned` copies output into an independent `OwnedOcr` and can fail on
allocation. Pass `true` as the capture argument to retain CTC spans; then call
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
the context capacity and 16. Defaults permit up to four line workers; nested
recognizer work shares the context. `Ocr::fork()` creates another mutable
request sharing compiled models, packed weights, dictionary and CPU capacity,
while keeping its own activations and buffers. Recognition batching is opt-in
with `.recognition_batch_size(n)` and groups crops with the same selected
width. Larger batches and more simultaneous requests increase memory use.

For a service, configure an `OcrEngine` before sharing it:

```rust,no_run
use simd_paddle_ocr_rs::{ModelPreset, OcrModelSet};
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
// Share Arc::new(engine) with callers. engine.run_owned(image, false) copies
// a result before releasing its request; engine.acquire() offers borrowed use.
println!("{:?}", engine.stats());
# Ok(())
# }
```

`acquire()` waits outside the executor for a request slot; `try_acquire()`
returns `None` immediately when full. A lease returns its request on drop and
ties borrowed results to that lease. Blocking acquisition from an executing
context or trace callback is rejected to avoid deadlock; use nonblocking
acquisition there. Configure options on the builder/template before creating
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
input-byte, nesting and rank limits.

This importer/runtime supports the OCR subset used by the shipped bundles.
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
shipped bundles. AVX-512 and additional platforms require their own physical
machine qualification; the Windows x64 gate does not establish those results.
