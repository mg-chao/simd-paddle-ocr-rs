# Native GPU OCR

The optional `gpu` feature provides synchronous, headless FP32 inference through
wgpu 30.0.1 on native Vulkan, D3D12, and Metal. Detection, optional orientation
classification, and recognition can use the GPU in all three model presets.
Image decoding, resize/normalization, perspective crops, DB postprocessing, and
CTC text assembly remain on the CPU. Browser execution is outside this feature.

CPU execution is the default. GPU initialization, unsupported model variants,
and resource failures return errors; there is no automatic CPU fallback.
Use the pinned nightly toolchain and enable the feature in the consuming crate:

```toml
[dependencies]
simd-paddle-ocr-rs = { path = "../simd-paddle-ocr-rs", features = ["gpu"] }
```

## Select GPU OCR

```rust,no_run
use simd_paddle_ocr_rs::{ExecutionTarget, ModelPreset, OcrModelSet};
use simd_paddle_ocr_rs::image::{ImageView, PixelFormat};
use simd_paddle_ocr_rs::runtime::gpu::{
    GpuBackendPreference, GpuContext, GpuContextOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = GpuContext::new_with_options(GpuContextOptions {
        backend: GpuBackendPreference::Vulkan,
        max_in_flight: 1,
        max_device_bytes: Some(2 * 1024 * 1024 * 1024),
    })?;
    println!("GPU: {:?}", gpu.adapter_info());
    let models = OcrModelSet::from_directory("models", ModelPreset::ChineseV6Tiny, true)?;
    let mut request = models.builder()
        .execution_target(ExecutionTarget::Gpu(gpu.clone()))
        .recognition_batch_size(2)
        .build()?;
    let pixels = vec![255u8; 640 * 480 * 3];
    let image = ImageView::new(&pixels, 640, 480, 0, PixelFormat::Rgb24)?;
    println!("{}", request.run(image, true)?.text);
    println!("Tracked device buffers: {:?}", gpu.memory_statistics());
    Ok(())
}
```

`GpuContext::new()` uses automatic native backend selection, one in-flight
submission, and no context-wide byte limit. Explicit preferences are `Auto`,
`Vulkan`, `Dx12`, and `Metal`. Initialization requires a physical hardware adapter.
No window, Vulkan SDK, external shader compiler, or async runtime is required.

The example accepts `--device cpu|gpu` and
`--gpu-backend auto|vulkan|dx12|metal`. Adapter diagnostics go to stderr, preserving
JSON output. `--threads` continues to configure host CPU work.

## Ownership, shapes, and service requests

Parsed models, uploaded weights, and pipelines are immutable shared resources.
Forked requests and engine leases share them while owning independent mutable
buffers. Each request uses one GPU classifier and one GPU recognizer; CPU line
worker counts do not multiply GPU arenas. Recognition groups exact selected
widths up to `recognition_batch_size` and restores detector reading order.

Low-level callers can still construct `GpuCompiledModel` and `GpuSession`.
`GpuSession::reshape` changes the bound input shape transactionally: invalid
shapes or failed allocations preserve the previous working binding. An
eight-entry LRU caches host plans; one active device-buffer set reuses capacity
across shapes. Complete activation arenas are never cached per width.

`run` uploads the input once and reads only the selected output. Normal OCR
keeps graph intermediates and projection features on the GPU. Logit recognition
reads class/score records without materializing dense vocabulary outputs;
probability recognition retains dense device logits/probabilities but reads back
only class/score records. Classification reads its label and probability.

`OcrEngine` retains its request-count and idle-cache limits. Idle statistics expose
host and device bytes; eviction releases request-owned buffers. GPU submission
admission is shared by the context and configured independently of request count.
GPU waits occur on the calling thread, outside Rayon workers. CPU contexts still
bound the host image work. Observers run without a GPU admission permit held.

## Numerical and operator contracts

The normalized model must have one FP32 input and output. GPU lowering uses the
same semantic compiler and shape binder as the CPU. Supported operators are
`Conv`, `Add`, `Mul`, `Div`, `Sub`, `Pow`, `Sqrt`, `Erf`, `Relu`, `Sigmoid`,
`HardSigmoid`, `MatMul`, `ConvTranspose`, `MaxPool`, `AveragePool`,
`BatchNormalization`, `GlobalAveragePool`, `ReduceMean`, `Softmax`, `Reshape`,
`Squeeze`, `Unsqueeze`, `Transpose`, `Slice`, `Concat`, and `Resize`.

This is the existing OCR operator subset, not general ONNX execution:

- Convolution uses NCHW activations and OIHW weights, including grouped/depthwise
  convolution. Binary arithmetic uses trailing-axis broadcasting through rank eight.
- MatMul supports shared rank-two RHS and broadcasted batched activation operands.
- Softmax operates on the last axis; reductions use the validated axes/keepdims.
- Pow accepts exponent two. Slice uses the compiler's unit-step contract.
- ConvTranspose uses unpadded 2x2 stride-two group-one geometry. MaxPool uses
  trailing-padded 2x2 stride-one geometry; AveragePool uses unpadded 3x2 kernels
  and strides.
- Resize uses spatial nearest/asymmetric/floor scales two, four, or eight.
- Shape metadata executes on the host without reading activations back.

Default recognition preserves raw-logit scores and bias-first projection.
Probability mode executes graph MatMul, optional Add, and Softmax and chooses
the maximum **rounded probability**. Both recognition modes prefer the lowest
class index on ties. Classification chooses label one only when its logit is
strictly larger, then computes that label's two-class probability. CTC keeps
the existing blank, duplicate, first-frame score, and alignment semantics.

All arithmetic/storage is FP32; no optional FP16/F64 or cooperative-matrix feature
is required. GPU exponential evaluation and shader arithmetic can round
differently from CPU arithmetic. Universal bitwise equality is not promised.
Synthetic arithmetic tests use `1e-4 + 1e-4 * abs(reference)`; real-model checks
use `1e-3 + 1e-3 * abs(reference)`, with separate public OCR acceptance checks.

## Memory and failures

Stage activation limits include retained host input/output, device activations,
readback buffers, head scratch, and diagnostic activation storage. Actual retained
capacities matter when a smaller shape reuses a larger allocation. Shared weights
and per-dispatch parameters/planning are reported separately. Transactional
growth can temporarily retain old and candidate buffers; its peak is separate
from post-commit retained capacity.

`GpuContextOptions.max_device_bytes` optionally caps tracked application-owned
device buffers, including shared weights once, parameters, readback, scratch,
and old/candidate overlap. Reservations fail immediately and release on rollback
or drop. These counters do not include driver allocations or all internal wgpu
staging, and are not a process-wide or total-VRAM measurement.

Invalid input, ordinary validation/allocation failure, or observer cancellation
returns no partial OCR result; a later valid call can reuse the request. Device
loss or an uncertain completion timeout fails the shared context. Recreate the
context and dependent resources to recover. Blocking recursive execution that
would deadlock returns an error instead of silently changing devices.

## Tracing, profiling, and API migration

Tracing lazily creates a separate GPU diagnostic path. It observes actual GPU
nodes before slot reuse, reads one intermediate at a time, and calls observers
on the calling thread. Dense graph probabilities remain distinct from public
compact results. Cancelling an observer stops subsequent diagnostic submission.
Production profiling does not allocate this diagnostic path; its wall counters
include submission/synchronization where applicable, not fabricated device time.
`gpu_admission_wait_ns` measures context admission; `gpu_submission_ns` measures
host upload, command encoding and submission; `gpu_completion_wait_ns` measures
the synchronous completion/map wait; `gpu_readback_ns` measures mapped result
decoding and unmapping. These calling-thread counters are included in the stage
wall durations and must not be added to them. Ordinary execution keeps these
profiling timers inactive.

`Detector::compiled()`, `Recognizer::compiled()`, and `Classifier::compiled()` now
return `&CompiledStage`. Common callers retain `.model()` and `.clone()`.
CPU-specific inspection uses `.as_cpu().unwrap().backend()` or `.options()` on
that CPU handle; GPU-specific inspection uses `.as_gpu()`. Use
`.execution_backend()` for general placement reporting. Existing SIMD `Backend`
and `KernelPolicy` still describe CPU kernel selection.

Stage constructors accept CPU or GPU compiled handles through `Into<CompiledStage>`.
`OrientedLineExtractor::new` accepts `Option<CompiledStage>` for classification;
wrap an existing CPU classifier handle with `.into()`. CPU `CompileOptions`
continue to control CPU compilation, independently of GPU device selection.

## Physical-machine validation

Run focused correctness checks locally. Hardware tests are ignored by default;
explicit invocation requires working hardware and fails rather than skipping it.

```powershell
cargo test --release --locked --lib
cargo test --release --locked --no-default-features --lib
cargo test --release --locked --features gpu --lib
$env:SIMD_OCR_GPU_BACKEND = 'vulkan'
cargo test --release --locked --features gpu --lib gpu_hardware -- --ignored --nocapture --test-threads=1
cargo test --release --locked --features gpu --lib gpu_ocr_context_quota -- --ignored --nocapture --test-threads=1
cargo test --release --locked --features gpu --lib gpu_ocr_synthetic -- --ignored --nocapture --test-threads=1
cargo test --release --locked --features gpu --lib gpu_ocr_real_models -- --ignored --nocapture --test-threads=1
cargo test --release --locked --features gpu --lib gpu_ocr_public -- --ignored --nocapture --test-threads=1
```

The last two commands require local reference models and the local OCR screenshot.
Override their paths with `SIMD_OCR_MODELS_DIR` and `SIMD_OCR_TEST_IMAGE`.
Set `SIMD_OCR_GPU_BACKEND=dx12` for explicit D3D12 hardware validation.
Linux and Metal validation require their own physical machines.

### Recorded acceptance: 2026-10-21

All checks ran locally on the physical Windows machine with an AMD Radeon
RX 6750 XT. The release all-features unit suite passed 364 tests; 29 opt-in tests
were excluded from that default invocation. Explicit hardware invocation then
passed all 21 GPU tests once with Vulkan and once with D3D12, with no warm-ups
or benchmarks. These tests include the seven local models, the same two-line
screenshot across three presets and two score modes, synthetic batching and
tracing, request forks, factories, concurrent engine leases, allocation rollback,
and failed-context admission wakeups.
The profiled public OCR cases also checked the separate host GPU timing fields;
synthetic profiling checked that no diagnostic device storage was allocated.

Dense model comparisons used classifier input `[1,3,80,160]`, detector input
`[1,3,128,128]`, and recognition input `[2,3,48,128]`. All outputs were finite and
matched CPU shapes and the stated elementwise tolerance. Maximum absolute errors:

| Model | Vulkan | D3D12 |
| --- | ---: | ---: |
| cls | 0 | 0 |
| det | 0.0000130 | 0.0000136 |
| small_det | 0.0000063 | 0.0000053 |
| medium_det | 0.0000044 | 0.0000048 |
| rec | 0.0000303 | 0.0000308 |
| small_rec | 0.0002120 | 0.0002124 |
| medium_rec | 0.0001212 | 0.0001221 |

Public OCR matched CPU text, detection count/order, orientation, and CTC spans.
Box vertices stayed within one source pixel; probability confidence differences
stayed within `1e-3`. The GPU CLI also produced parseable JSON with alignment and
kept adapter information on stderr.
The CPU-only example rejected `--device gpu` with guidance to rebuild using
`--features gpu`, before attempting to open the image or models.

The dynamic ReLU memory fixture measured the following retained activation
capacities on both backends. These totals include host input/output and device
activations/readback, excluding separately reported parameters and shared weights:

| Shape transition | Retained bytes |
| --- | ---: |
| Initial `[1,64]` | 1,280 |
| Grow to `[1,512]` | 10,240 |
| Return to `[1,64]`, reuse device capacity | 6,656 |
| Compact `[1,64]` to its exact activation budget | 1,280 |

The recorded transactional peak was 11,520 bytes. Failed growth and quota checks
preserved the previous binding and released candidate reservations; request and
engine eviction checks returned tracked device usage to the expected baseline.
These are application-owned buffer measurements, not total driver/VRAM usage.

Release build checks covered CPU default, serial, profiling, serial profiling,
GPU parallel, serial, parallel profiling, and serial profiling, including tests
and the example. Metal, Linux GPU drivers, and other GPU vendors remain untested
on this Windows hardware. Numerical agreement is bounded by the tested FP32
tolerances; it is not a guarantee of bitwise identity or every possible input.

This iteration establishes model coverage, ownership, and correctness. It makes
no speedup claim.
