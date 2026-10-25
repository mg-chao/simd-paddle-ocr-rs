# Local Metal validation

Run checks on the physical development machine. No CI, virtual GPU or software
adapter is a substitute for the hardware suite. Use release builds, focused
correctness before performance, fixed inputs and a bounded sample budget.

```sh
cargo clippy --release --locked --all-features --all-targets -- -D warnings
cargo test --release --locked --all-features --lib
cargo test --release --locked --no-default-features --lib
cargo build --release --locked --all-features --examples
```

CPU-only checks preserve CPU behavior. Explicitly invoke hardware integration
checks on macOS:

```sh
python tools/check_architecture.py --release --hardware
```

These use the mandatory Metal executor. Numeric tests retain independent CPU
oracles, input/weight storage rounding, output offset and guard checks, tail and
nonfinite cases, compile-error classification, resource lifetime and dependency
ordering. Optional arithmetic-kernel fallbacks remain on Metal. There is no
second encoder or backend comparison mode.

## Full OCR qualification

Build the release benchmark example first. The correctness harness generates
nine deterministic image fixtures from the unchanged sibling repository's
sample, runs one CPU reference and one GPU invocation for each selected case,
and performs zero benchmark repetitions:

```sh
python tools/full_gpu_validation.py \
  --presets tiny small medium --cases typical dense vertical \
  --report target/metal-strict
python tools/full_gpu_validation.py \
  --presets tiny small medium --cases typical dense vertical \
  --precision mixed-f16 --matrix f16 --wave 16 --classification-batch 16 \
  --report target/metal-fast
```

The harness calls the same Rust comparison implementation as `gpu_ocr_bench`.
The existing F32 and explicit F16 text/geometry/confidence/CTC contracts are not
reimplemented or relaxed in Python. Fast-mode confidence gates apply only to
matching geometry and tensor dimensions; changed-input lines retain raw deltas
and validity checks. JSON results include the exact accuracy contract.

For additional edge images, use `--all-cases`. All GPU dispatches must be native
Metal, with no dense diagnostic readback during production. The harness checks
those counters and actual stage placement as well as OCR results. Models remain
external; no assets are downloaded. Preserved data-only inputs and prior CPU
references are also available in `fixtures/metal-ocr`.

## Tensor diagnostics and performance

`tools/gpu_validation.py inventory` inspects model metadata without execution.
`tools/gpu_validation.py compare` uses the current Metal tensor session and a
CPU reference; select a concrete model/shape to keep work bounded. Its optional
C# developer adapter also selects Metal explicitly and never changes the sibling
C# implementation.

Use [the performance procedure](gpu-performance.md) only after qualification.
One cold invocation and at most three warm samples answer a defined performance
question. Keep device timing diagnostics separate, retain all raw samples and
source/configuration hashes, and stop when the question is answered.
