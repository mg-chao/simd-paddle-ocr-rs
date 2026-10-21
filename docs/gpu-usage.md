# GPU runtime

The optional `gpu` feature provides synchronous, headless inference through
wgpu 30.0.1 on native Windows, Linux, and macOS. Vulkan, D3D12, and Metal are
enabled; initialization chooses a hardware adapter and reports the selected
backend. It does not create a window or require a Vulkan SDK or shader compiler.
Browser/WebGPU execution is outside this first milestone.

CPU inference and the default `parallel` feature are unchanged. `OcrBuilder`
still creates CPU requests. The GPU runtime currently accepts small graphs
containing only the operators listed below; the complete shipped OCR models
require additional operator coverage.

## Use the library

Enable the feature in your consuming application's dependency:

```toml
[dependencies]
simd-paddle-ocr-rs = { path = "../simd-paddle-ocr-rs", features = ["gpu"] }
```

Use the project's pinned nightly toolchain. Given a supported ONNX graph with
input shape `[1, 3, 32, 32]`, create shared compiled resources and a reusable,
fixed-shape session:

```rust,no_run
use std::sync::Arc;
use simd_paddle_ocr_rs::{
    model::Model,
    runtime::{
        ExecutionLimits,
        gpu::{GpuCompiledModel, GpuContext, GpuSession},
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = GpuContext::new()?;
    println!("GPU: {:?}", gpu.adapter_info());
    let model = Arc::new(Model::load_path("conv-relu.onnx")?);
    let compiled = Arc::new(GpuCompiledModel::compile(model, gpu)?);
    println!("Uploaded constants: {} bytes", compiled.weight_bytes());
    let mut session = GpuSession::new(
        compiled,
        &[1, 3, 32, 32],
        ExecutionLimits { max_activation_bytes: 64 * 1024 * 1024 },
    )?;
    let input = vec![0.25; 3 * 32 * 32];
    let output = session.run(&input)?;
    println!("Output shape: {:?}", output.shape().dimensions());
    println!("Session memory: {:?}", session.memory_statistics());
    Ok(())
}
```

`GpuContext` shares its device and queue. An `Arc<GpuCompiledModel>` shares
uploaded constants and pipelines between sessions. Each `GpuSession` owns its
activation buffers, bound execution plan, readback buffer, and host output.
Create another session to bind a different shape. `run` borrows the output
until the next mutable use of that session, like the CPU API.

Each run uploads the input once, executes the entire supported graph on the
GPU, and reads back the final output. Intermediate activations stay on the
device. Execution is synchronous, including final readback; callers need no
async runtime.

## Operator and numerical contracts

The normalized model must have one FP32 graph input and output. Unsupported
operators are rejected before GPU resource upload; there is no implicit CPU
fallback.

| Operator | Supported behavior |
| --- | --- |
| `Conv` | Contiguous NCHW input and OIHW weights; optional `[Cout]` bias; batches, ordinary/grouped/depthwise convolution, positive strides and dilations, asymmetric nonnegative padding. |
| `Add`, `Mul` | ONNX trailing-axis broadcasting, including scalars and tensors through rank eight. |
| `Relu` | The CPU comparison rule: `x > 0 ? x : 0`. |

Per-channel NCHW broadcasting uses `[1, C, 1, 1]`; a rank-one `[C]` operand
broadcasts against the last axis, not automatically against channels.

WGSL kernels use FP32 throughout, without graph fusion, FP16 storage,
cooperative matrices, or optional GPU features. Convolution follows the scalar
CPU bias-first accumulation and channel/kernel traversal. Shader compiler
arithmetic can still round differently, so bitwise CPU equality is not promised.
The correctness fixtures require finite results and per-element
`abs_error <= 1e-4 + 1e-4 * abs(reference)` for bounded synthetic data.

## Memory limits and failures

`GpuSession::memory_statistics()` reports `device_activation_bytes`,
`readback_bytes`, and `host_output_bytes`; `total_bytes()` sums these counters.
`GpuSession::activation_bytes()` reports that same total. The session's
`ExecutionLimits.max_activation_bytes` bounds their combined retained capacity.
Lifetime planning reuses activation slots without overwriting live inputs.

Shared uploaded constants, reported by `GpuCompiledModel::weight_bytes()`, are
outside the session activation limit. This count includes the small zero
buffer used for absent operands. Host model data, command/bind-group metadata,
and per-dispatch parameter buffers are also outside that limit; it is not a
process-wide or total-VRAM cap.

Graph validation, tensor sizes, shader indexing, buffer bindings, and dispatch
dimensions are checked before execution. Initialization, allocation,
validation, submission, and readback failures return `RuntimeError`. GPU
initialization does not silently substitute a software adapter or CPU session.

## Local validation

These checks use synthetic ONNX fixtures and require no external model files.
Run on the physical machine, as required by `AGENTS.md`:

```powershell
cargo test --release --locked --lib
cargo test --release --locked --no-default-features --lib
cargo test --release --locked --features gpu --lib
cargo test --release --locked --features gpu --lib gpu_hardware -- --ignored --nocapture --test-threads=1
```

Hardware tests are ignored by default. Explicit invocation requires a real GPU
and fails when one cannot be initialized; it does not count a skipped or
software-only run as validation. The tests print the adapter and backend and
compare against the scalar NCHW runtime. They cover convolution geometry,
broadcasting, dispatch tails, branch lifetimes, repeated inputs, and sessions
sharing compiled resources.

On 2026-10-21, all six hardware tests passed locally on Windows with an AMD
Radeon RX 6750 XT using the Vulkan backend. The five device-independent GPU
validation tests also passed. D3D12, Linux, and Metal hardware validation remain
pending; Metal validation requires a physical Mac.

This milestone establishes correctness and ownership boundaries. It makes no
speedup claim.
