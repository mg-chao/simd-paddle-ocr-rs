# CPU API migration

Compilation policy, CPU resources and request-local storage are separate concepts.

```rust,no_run
use std::sync::Arc;
use simd_paddle_ocr_rs::{model::Model, runtime::{
    CompileOptions, CompiledModel, CpuContext, ExecutionOptions, Session,
}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(Model::load_path("models/det.onnx")?);
    let compiled = Arc::new(CompiledModel::compile(model, CompileOptions::default())?);
    let cpu = Arc::new(CpuContext::new(4)?);
    let mut request = Session::new(compiled.clone(), cpu.clone(), &[1,3,32,32],
        ExecutionOptions { intra_op_threads: 4, max_activation_bytes: 1 << 30 })?;
    request.reshape(&[1,3,64,96])?;
    let output = request.run(&vec![0.; 3*64*96])?;
    println!("{:?}", output.shape());
    Ok(())
}
```

`CompiledModel::new(model)` remains an optimized-default convenience alias. Use `CompiledModel::compile(model, CompileOptions::reference())` to request the scalar NCHW unfused oracle. The previous three-argument `Session::new(compiled, shape, limits)` is now `Session::new_serial(compiled, shape, limits)`; the primary constructor takes an explicit shared `Arc<CpuContext>` and `ExecutionOptions`.

One compiled model shares validated graph state and packing across mutable requests. Packed formats are also cached on the parsed `Model`, so separate compilation policies over the same `Arc<Model>` share immutable panels. Formats are created only for eligible compiled consumers. `Model::cached_packed_weight_bytes()` and `Session::statistics().shared_cached_weight_bytes` report all formats retained by that shared model; `packed_weight_bytes` reports the current compiled view's panels. These describe shared capacities and must not be multiplied by request count. One CPU context shares a bounded worker pool across kernels, nested stages and concurrent requests. A session requires exclusive mutable access and retains its activation arena. Optimized and diagnostic bindings keep eight metadata-only MRU entries, with their endpoint and mode fixed by the owning session. The independent scalar reference preserves its original two-table rebinding strategy. A failed reshape preserves the prior usable binding. Borrowed outputs prevent another mutable run through ordinary Rust borrowing rules.

`OcrModelSet::from_directory(dir, preset, classification)` loads the pinned project's file layout without downloading or decoding anything. Tiny uses `ppocr_keys.txt`; small and medium use `rec_keys.txt`. `from_paths` and `from_readers` accept other sources. `.builder()` accepts compilation policy, a shared CPU context, stage options/limits and optional direction classification; `.build()` creates `Ocr`, while `.build_engine(limits)` creates a bounded pool. Explicit `None::<&std::path::Path>` / `None::<&[u8]>` can supply absent classifiers to the generic path/reader constructors.

`Ocr::set_recognition_batch_size` opts into same-target-width line batching; one remains the default. It survives `fork` and engine factories. `Recognizer::recognize_batch` returns borrowed per-input results in original order and requires one selected tensor width. Per-line resized content widths, scores, CTC boundaries and decoding are independent. The recognizer result-byte limit applies to the aggregate batch result. Synchronous tracing retains per-line diagnostic execution and the same public result. Batch buffers are included in retained request storage accounting.

Run with `capture: true` to estimate source-image character/token quads through `OcrLineView::estimate_character_boxes` or the owned line equivalent. Missing capture returns an error; captured empty alignment returns no boxes. Gap midpoints, right padding, perspective transforms and crop/classifier rotations follow the reference. `packed_text()` and `packed_text_hash()` preserve empty lines and append a NUL per line, using UTF-8 and reference FNV-1a semantics.

High-level `Detector`, `Classifier` and `Recognizer` constructors remain convenient serial request constructors. `set_execution_context(cpu, threads)` opts them into the shared pool. `Ocr::new_with_execution(extractor, recognizer, options, cpu, cpu_options)` fixes total capacity and stage budgets. `Ocr::new` uses the shared automatic context. A context of capacity one executes inline and works with `--no-default-features`.

Positive `CpuOptions` line, preprocessing and detector budgets are maxima resolved to `min(requested, cpu.capacity(), 16)`; zero keeps the original automatic selection. The public `preprocess::det_with_context` helper applies the same positive cap, while zero retains its serial behavior. OCR constructors, context setters, forks and engine factories share the stage-budget resolver.

Session constructors and execution-context setters validate the original requested thread count within `1..=cpu.capacity()` before storing an effective budget capped at 16. `Session::intra_op_threads()` reports that effective budget. Classifier requests follow the same validation rule and always execute with one intra-op thread, including diagnostic sessions. Invalid requests leave the current context and budget unchanged. The context itself may have a larger capacity to serve concurrent requests.

For simultaneous requests use separate `Ocr` values sharing compiled models/context, or the bounded `engine::OcrEngine` request pool. Its RAII lease owns the mutable request; borrowed results stay tied to that lease. Idle-request byte limits govern cached capacity, not total process peak memory. Request creation/inference does not run while holding the pool mutex.

```rust,no_run
use simd_paddle_ocr_rs::{
    engine::{EngineLimits, OcrEngine},
    image::{ImageResult, ImageView},
    ocr::{Ocr, OwnedOcr},
};

fn serve(template: Ocr, image: ImageView<'_>) -> ImageResult<OwnedOcr> {
    let engine = OcrEngine::new(template, EngineLimits {
        max_requests: 4,
        max_idle_bytes: 256 << 20,
    })?;
    // request() waits for admission; try_request() returns None when full.
    let mut request = engine.request()?;
    request.run(image, true)?.try_to_owned()
}
```

The engine consumes the template into a factory holding immutable models, dictionary, options and the CPU context; it releases the template's warm arenas. Requests allocate lazily, reuse their buffers while cached, and are discarded when they exceed the aggregate idle limit. A capacity-one context serializes simultaneous callers inline. Ordinary errors permit reuse; a panic poisons the affected request, which the lease discards even if its caller catches the unwind.

Engine requests are pinned to the engine's CPU context. A lease exposes `run`, `run_with_trace`, and, with profiling enabled, `run_profiled`, plus immutable inspection through `Deref`; it has no `DerefMut`. Configure components, stage budgets and the CPU context on the standalone `Ocr` template before creating the engine. A pooled request's components or entire `Ocr` value cannot be replaced through its lease.

Blocking acquisition is rejected from a worker executing on the same CPU context and from a caller-thread trace callback already running a request on that context. The context-specific active-request marker spans callbacks and clears on both return and unwind, preventing a callback from waiting for its own lease. Use `try_request()` for reentrant callers; it remains nonblocking. Trace observers keep their calling-thread contract and do not require `Send`.

Logical model inspection does not change when optimizations are enabled. `TensorId`, normalized node indices, shape recipes and snapshots identify the imported graph. Physical NHWC twins/conversions and fused groups belong to a separate execution representation. A prefix selects an inclusive normalized endpoint before layout/fusion lowering; the selected feature is returned canonically and excluded suffix tensors are not allocated.

Public `run`, `run_with_input` and tracing accept canonical NCHW input. Diagnostic traces emit each normalized node in order with canonical NCHW/row-major values; tracing de-fuses the execution region and may differ from production fusion within the declared tolerance. Use execution statistics to inspect production fusion/layout/packing without treating inserted physical steps as new imported nodes.

`Session::statistics()` reports selected operator/layout/packing groups and actual shape-selected families/ISAs, physical and fused step counts, conversion counts, cache activity, original/packed weights, activations and scratch. `Recognizer::projection_statistics()` separately identifies compact-head row/column scheduling and its selected ISA, including narrower AVX-512 fallbacks. `NodeTrace::elapsed_ns` measures unfused diagnostic kernel wall time and excludes callback time. It is separate from production stage profiling.

High-level stage and OCR requests expose `retained_kernel_scratch_bytes()`. These buffers are already included in `retained_workspace_bytes()`; subtract the former to report other request workspace without double counting. Scratch includes retained convolution buffers and compact-head worker winners, while fixed local stack tiles are not retained heap capacity. Process peak memory includes temporary allocations and stack usage as well as the reported retained capacities.

`max_activation_bytes` bounds committed retained F32 activation capacity, including a session's lazy diagnostic arena. Tracing reserves only the remaining capacity and can fail before filling input or invoking observers when its unfused arena would exceed the aggregate limit. A later reshape also budgets retained diagnostic storage; failure preserves the previous runnable binding. Transactional growth can briefly own old and replacement storage at the same time. Per-request planning and kernel scratch, shared packed weights and pipeline result/crop storage are reported separately; do not treat the activation limit as a process-wide memory cap.
