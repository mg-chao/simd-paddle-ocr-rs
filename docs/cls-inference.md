# Scalar CLS inference

The Rust runtime executes the shipped shared orientation classifier from a
concrete normalized input tensor. It uses serial F32 kernels and contiguous
NCHW storage. Image preprocessing and complete OCR inference remain outside
this milestone. Import parity and execution parity are separate gates.

## Ownership and execution

`CompiledModel::new(Arc<Model>)` compiles supported operations and shape
metadata without cloning model weights. Multiple mutable `Session` instances
can share one `Arc<CompiledModel>`. A session binds its own shapes, activation
buffers, and execution parameters.

`Session::new(compiled, &[N, 3, 80, 160], ExecutionLimits::default())` binds the
shipped CLS network; `N` must be positive. `reshape` rebinds transactionally,
and an unchanged shape is a no-op. `run` returns a `TensorView<f32>` borrowing
the session. Its output must be released or copied before another mutable
session operation. `input_shape`, `output_shape`, and `activation_bytes`
provide the current binding and requested activation storage.

```rust
use std::sync::Arc;
use simd_paddle_ocr_rs::{
    model::Model,
    runtime::{CompiledModel, ExecutionLimits, Session},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(Model::load_path("models/cls.onnx")?);
    let compiled = Arc::new(CompiledModel::new(model)?);
    let mut session = Session::new(compiled, &[1, 3, 80, 160], ExecutionLimits::default())?;
    let input = vec![0.0; 3 * 80 * 160];
    let probabilities = session.run(&input)?;
    assert_eq!(probabilities.shape().dimensions(), &[1, 2]);
    Ok(())
}
```

Successful untraced inference reuses prepared buffers. Tensor lifetime
intervals overlap at a consuming/producing node, so live inputs never share
the destination slot. Reshape copies into a separate reusable slot. Every
kernel overwrites its complete active output. Failed observers restore buffer
ownership before returning an error; the session can be run again.

`ExecutionLimits.max_activation_bytes` defaults to 1 GiB of planned reusable
F32 slot storage, excluding immutable shared weights. Dimensions, strides,
element counts, and allocation sizes are checked before allocating. Invalid
rebindings leave the previous successful binding available.

## Supported operations and shape programs

The execution subset is Conv, HardSigmoid, Add, Mul, ReduceMean, Relu,
Reshape, rank-two MatMul, and the CLS last-axis Softmax. Unsupported
operations or parameter combinations return errors rather than a partial
network result. This is not a general ONNX backend.

Shapes are inferred in graph order. The retained integer Shape/Slice/Concat
program is evaluated when its source activation shape is known. For CLS it
produces `[N, -1]`, which resolves to `[N, 1280]`. Symbol strings remain opaque;
the runtime does not infer dimensions by parsing their names. Reshape checks
zero-copy-dimension rules, `allowzero`, inferred dimensions, and element-count
preservation. Zero-sized activation bindings are unsupported.
The initial integer subset uses rank-one I32/I64 vectors of at most eight
elements, attribute-form Slice with unit step, and rank-one Concat. Shape
programs have at most 64 metadata instructions, bounding evaluation depth.

Scalar arithmetic preserves the reference's individual F32 operations and
accumulation order. Convolution begins with bias and skips padded taps;
matrix multiplication and reductions accumulate in source order. HardSigmoid
uses each node's original parameter bits. Stable Softmax subtracts the maximum,
computes F32 exponentials, sums sequentially, and divides. Runtime-dependent
exponential rounding is covered by the stated float tolerance.
Pointwise convolution computes four independent output channels together,
reusing input loads while preserving each channel's accumulation order.
Depthwise kernels precompute valid border ranges and reuse each tap over its
output region; common broadcasts and spatial means use contiguous loops.

## Tracing and partial bundles

`run_with_trace` synchronously borrows each completed node's output before
later nodes can reuse its storage. Trace records contain the normalized
execution index, original source key, name, operator, output tensor ID,
shape, layout, and values. Callback values are valid only during that callback;
copy them to retain them. Rank-four outputs use `nchw`; other outputs use
`row_major`.

Artifact IDs use normalized execution indexes. For example, CLS
`node.0105` is execution ordinal 105, not the original ONNX source index.
The candidate generator checks the reference graph inventory before execution.

```powershell
cargo run --release --offline -- run-cls reference-bundles/tiny-canonical-allnodes-v1 --out reference-bundles/rust-cls-v1
cargo run --release --offline -- compare-cls reference-bundles/tiny-canonical-allnodes-v1 reference-bundles/rust-cls-v1 --atol 1e-5 --rtol 1e-4 --json
# Inspect one particular stage with the existing comparator.
cargo run --release --offline -- compare-stage reference-bundles/tiny-canonical-allnodes-v1 reference-bundles/rust-cls-v1 --case sample --artifact line.000.cls.node.0000
```

The writer keeps the reference contract, model identities, and cases. Frozen
CLS input tensors are copied, while node/probability tensors are generated by
Rust. It omits public preprocessing metadata and unimplemented stages. A
completed manifest is published last. `compare-cls` uses the reference's
complete classifier selection and validates every declared model, artifact,
and auxiliary file on each side in a single traversal. A successful comparison
establishes classifier coverage only.

## Verification and measurements

Normal tests use small independent fixtures. Real-model gates require explicit
local references and fail if a requested reference is absent:

```powershell
$env:SIMDOCR_IMPORT_REFERENCE = (Resolve-Path reference-bundles/import-reference-v1).Path
$env:SIMDOCR_CLS_REFERENCE = (Resolve-Path reference-bundles/tiny-canonical-allnodes-v1).Path
cargo test --release --offline --test model_import all_shipped_models_match_authoritative_imports -- --ignored
cargo test --release --offline --test cls_harness -- --ignored
cargo fmt --all -- --check
cargo test --locked --offline --all-features
cargo clippy --locked --offline --all-targets --all-features -- -D warnings

./tools/benchmark-cls.ps1 -Reference reference-bundles/tiny-canonical-allnodes-v1 -Out target/cls-measurements-v1 -Warmup 30 -Repeat 100 -Rounds 3
```

The benchmark compares the same frozen tensors and borrowed full-graph output
in fresh C# and Rust processes, alternating engine order. Model loading,
session preparation, and first inference are reported separately. Warm
measurements exclude tracing, hashing, serialization, and file I/O. Allocation
instrumentation runs separately from latency and must observe zero allocations
during successful untraced warm inference. Activation bytes, requested heap
bytes, and process memory have different accounting and are reported separately.
First-sweep inputs share the prepared session; only the first graph call in a
process is process-cold. Model loading and preparation may include runtime/JIT
startup work and are not comparable with the previous warm import measurements.
C# allocation volume covers the current thread's managed allocations, while
Rust instrumentation counts allocations routed through its System allocator;
neither number is a process memory peak.

Measurements establish a reproducible scalar baseline, not an inference speed
claim for other machines, models, or optimized backends. There is no fixed
speed target for this milestone.

## Recorded milestone on 2026-10-14

The final all-node reference covers 32 inputs and 3,392 node tensors. All 3,456
selected classifier artifacts match at `atol=1e-5`, `rtol=1e-4`; all 32 labels
match exactly. The reference was archived and validated after extraction; its
identity is in `baselines.json`. The final Rust candidate is available locally
as `reference-bundles/rust-cls-scalar-v1`. See [the parity report](cls-parity.json).

Formatting, Clippy with warnings denied, all 82 normal tests, the explicit
seven-model import gate, and the explicit canonical CLS gate passed. Regression
fixtures include distinct batched inputs, shared compiled models, invalid
empty constants, graph-input overwrites, and grouped pointwise tile tails.

Measurements used Windows x86_64, an AMD Ryzen 9 5950X with 24 exposed logical
processors, Rust 1.97.1, and .NET SDK 10.0.303. C# used one effective processor,
disabled hardware intrinsics/fusions, and NCHW storage. Both engines processed
the same 32 frozen inputs in each corpus sweep.

| Fresh-process round | Rust corpus median / p95 ms | C# corpus median / p95 ms |
|---|---:|---:|
| 1 | 367.23 / 429.88 | 439.60 / 583.38 |
| 2 | 359.68 / 422.03 | 441.99 / 745.66 |
| 3 | 253.54 / 327.29 | 408.27 / 564.78 |

Each timing process performed 30 warmup sweeps and 100 measured sweeps. Rust's
median was lower in every timing pair, with substantial variation between
processes; retain the individual rounds rather than assuming a stable speedup.
The separate three profiling processes each performed 3,200 measured graph
calls with **zero allocation calls, zero allocated bytes, and zero warm heap
growth**. Their timings are excluded from the latency comparison.

The session plans **399,360 activation bytes**, compared with **8,262,744 bytes**
for retaining all activation tensors. These counts exclude shared weights,
the preloaded benchmark corpus, and process/runtime overhead. C# recorded
56,601,600 current-thread managed allocation bytes per measured corpus run;
this is allocation volume, not retained heap or native memory.

[Machine-readable measurements](cls-performance.json) preserve all raw samples,
process orders, corpus/executable identities, source hashes, and memory counters.
Original local executables and reports remain in
`target/measurements/cls-scalar-v1`. Remaining CPU work should be guided by
profiling convolution and channel traversal; SIMD, packing, layout changes,
and fusion require their own parity profiles.
