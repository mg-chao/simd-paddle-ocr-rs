# Local GPU validation

Run correctness checks on a physical machine before performance measurements.
No CI, virtual machine, or software Vulkan device qualifies a hardware result.

## Native Metal 4

On Apple Silicon with macOS 26+, run release-mode native foundation and kernel
checks locally:

```sh
cargo test --release --locked --features gpu,profiling runtime::gpu:: -- --ignored --test-threads=1
cargo test --release --locked --features gpu,profiling gpu_integration_tests -- --ignored --test-threads=1
```

The native suite includes allocation and transfer ordering, immutable parameter
snapshots, concurrent sessions, residency lifetime, foreign-context rejection,
quota rollback, commit completion, profiling counters and bounded retirement
faults. Numerical tests use CPU oracles with tail tiles, padding guards, offsets,
reduction boundaries, and IEEE binary16 edge cases. They do not benchmark.

See [M4 validation and measurements](metal-validation.md) for the migration's
local evidence. Other Apple GPU generations are not established by those results.

## Vulkan validation layers

Install or unpack Khronos validation layers locally. Set `VK_LAYER_PATH` to the
directory containing `VkLayer_khronos_validation.json` and its library. Then set
`SIMD_OCR_VULKAN_VALIDATION=1`. Initialization fails if the requested layer is
missing. This enables synchronization validation and records validation errors
as context failures. Do not describe a run without the layers as validated.

```powershell
$env:VK_LAYER_PATH = 'C:\path\to\validation-layers'
$env:SIMD_OCR_VULKAN_VALIDATION = '1'
cargo test --release --locked --features gpu,profiling vulkan_ -- --ignored --nocapture --test-threads=1
```

Native tests cover catalog binding contracts, actual uniform ranges, transfer
ordering, timestamps, foreign-resource rejection, initialization, abandoned
uploads, timeout retirement, and device-loss error propagation. The existing focused
hardware tests provide CPU numerical oracles for convolution, matrix tails,
reductions, attention, preprocessing, geometry, projection, and CTC. Hardware
tests are ignored by default and must be selected explicitly.

`tools/full_gpu_validation.py` compares complete OCR with local models and
representative images using the existing text, order, orientation, span, and
mode-specific geometry contracts. It hashes source files, configuration, models,
inputs, and the executable and refuses an executable older than its sources.

## Performance measurements

Disable validation layers for measurements. Rebuild after every source change.
Keep F32 and mixed-F16 results separate and preserve source/configuration identity.
Use one scheduling configuration, 1–3 representative inputs, at most one warm-up,
and 1–3 measured executions. Keep aggregate measured execution below 60 seconds
per source version by default, including harness loops. Do not repeat unchanged
code to obtain more favorable timings. Report noisy results as uncertain.

The ignored `profile_projection_variants` test compares portable F32, subgroup
F32, and cooperative F16 projection at `[40, 120, 6906]`. It uses exactly one
warm-up and three timestamped measurements per supported variant, compares
winners and probability partials, and performs no adaptive sampling. Run it
separately from correctness checks with validation disabled.

The migration baseline is preserved locally under
`reports/ash-migration/baseline`, outside production sources. It includes the
original source archive, revision and executable identity, and bounded classifier
and recognition comparisons in both storage modes.

NVIDIA and Intel Vulkan hardware require separate physical-device validation;
the Radeon RX 6750 XT and M4 results cannot establish behavior on those devices.
