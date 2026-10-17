# macOS CPU support and validation

The Rust library and OCR example run natively on Apple Silicon macOS. Automatic
dispatch selects `NeonFma`: ARM64 NEON convolution, matrix multiplication and
compact recognition projection use ordered fused multiply/add. `Portable4`
retains its separate multiply/add arithmetic as an explicit comparison policy.
Canonical NCHW specializations and activation programs retain their established
four-lane arithmetic where selected; execution statistics report these choices.
Native NEON modules compile only for AArch64, and AVX helpers only for x86;
forcing an unsupported policy returns an error at model compilation. Inference
requires no .NET, ONNX Runtime, Metal, or other GPU runtime. The .NET SDK is needed
only for live comparison with the pinned C# CPU reference.

The [ARM64 optimization record](macos-arm64-optimization.md) reports validation
and performance measurements for the native NEON implementation. The historical
Portable4 results below remain separate.

## Setup

Use a physical Mac with Xcode Command Line Tools, Rustup, and the native .NET SDK
`10.0.303` installed. The Rust toolchain and C# SDK are pinned by the repository;
no custom `RUSTFLAGS`, Rosetta, or `target-cpu=native` setting is required.
From the Rust repository:

```bash
xcode-select -p
rustup toolchain install nightly-2026-09-08 --profile minimal --component rustfmt --component clippy
git clone https://github.com/mg-chao/SimdPaddleOCR.git ../SimdPaddleOCR
git -C ../SimdPaddleOCR checkout 33ff8a81672452041d110e8ba52dcd2639e54228
(cd tools/csharp-reference && dotnet --version)
```

Reuse an existing clean sibling reference checkout when available. Model and
dictionary hashes must match `tests/baseline/manifest.json`. The corpus is already
checked in. Verification never downloads models or changes expectations.

In the prepared `simd-paddle-ocr-workspace`, `source ../env.sh` adds its local
.NET SDK and Rustup commands to `PATH`. This workspace convenience is optional;
standalone checkouts can use their installed native toolchains directly.

## Run OCR

```bash
cargo run --release --locked --example ocr -- \
  --models-dir ../SimdPaddleOCR/models --preset tiny \
  tests/baseline/corpus/imgs_1.jpg
```

Use `--preset small` or `medium`, `--threads N` to cap CPU workers, and
`--alignment` for JSON results with token spans and character boxes.

## Reproduce correctness validation

```bash
bash tools/validate-macos.sh
```

This Bash 3.2-compatible script checks formatting, builds all release examples,
runs default-feature tests and serial library tests without Rayon, compares all
66 model/fixture combinations with both saved expectations and live C#, then
runs all 213 configuration requests. The latter covers parallel budgets,
recognition batching, alignment, rotations, padded pixel formats, classification
off, fixed widths, detector settings, buffer reuse, and recovery after rejected
inputs. Each request runs once, without performance sampling or warm-up.

For local model files without .NET or a C# checkout:

```bash
bash tools/validate-macos.sh --rust-only --models-dir /path/to/models
```

This runs the same build and test checks, then compares all 66 Rust results
with the checked-in expectations. It omits live C# and configuration
qualification. Build dependencies must be cached if building offline.

Both modes accept `--reference-root PATH` and `--models-dir PATH`, including
paths containing spaces. Explicit paths resolve relative to the invoking
directory. The default model location is the reference checkout's `models/`.
Evidence remains in `target/baseline/` and `target/qualification/`, including
per-case results, worker logs, build logs, provenance, and differences. The
[baseline](cpu-baseline.md) and [qualification](cpu-qualification.md) guides
define the unchanged exact-field checks and numerical tolerances.

For a smaller diagnostic selection:

```bash
cargo run --release --locked --example cpu_qualify -- \
  --model tiny --profile alignment --fixture imgs_11_rotate180
```

The optional local harness integration test also checks two isolated C# builds
and two simultaneously loaded reference workers:

```bash
cargo test --release --locked --test harness_contract \
  isolated_builds_and_live_workers_can_overlap -- --ignored --nocapture
```

## Recorded local validation before ARM64 kernel optimization

Validated on the physical Apple M4, macOS 27.0.1, native
`aarch64-apple-darwin`, on 2026-10-17 (Asia/Shanghai). The pinned Rust nightly,
.NET SDK 10.0.303, and .NET runtime 10.0.11 were used with standard release
settings. This historical baseline selected `Portable4`; automatic parallel
capacity was ten workers.

- Release example builds and formatting checks passed.
- Default-feature tests: 305 library, 26 integration, and five doctests passed
  (336 total). Six performance diagnostics and one explicit local integration
  test were ignored by the normal suite. The latter was run separately and
  passed, including both isolated builds and live workers.
- Without default features: 296 library tests passed; six performance
  diagnostics remained ignored.
- All targets compiled with all features, including profiling, and separately
  without default features.
- Serial baseline: all 66 model/fixture combinations passed saved → Rust,
  saved → live C#, and live C# → Rust comparisons.
- The validation script's Rust-only mode passed all 66 combinations.
- Full configuration qualification: 213/213 requests passed with zero
  differences. Each case ran once per implementation, with no warm-up or
  performance measurements.
- The native OCR example completed one tiny-model request with two workers and
  returned three lines with character boxes.

| Model | Cases | Detected lines | Largest coordinate delta (px) | Largest score delta |
| --- | ---: | ---: | ---: | ---: |
| tiny | 22 | 448 | 0.000045313 | 0.000054282 |
| small | 22 | 439 | 0.000049414 | 0.000050175 |
| medium | 22 | 452 | 0.000052735 | 0.000117999 |

Deltas compare saved C# expectations with Rust results; score acceptance uses
the existing combined absolute/relative tolerance. Goldens, model hashes, and
tolerances were unchanged. The
[machine-readable report](reports/macos-arm64-2026-10-17-correctness.json)
records the source fingerprint, commands, counts, and local artifact paths.

## Scope and limitations

The matched `cpu_benchmark` runner now supports macOS with explicit unbound
scheduling, native process CPU counters, and separately labeled resident-memory
high-water marks. See the [performance protocol](cpu-performance.md) and
[focused M4 comparison](macos-cpu-performance.md). The correctness checks above
and prior Windows performance claims do not establish M4 performance parity.
This port does not add Metal inference.
Intel macOS needs its own physical-machine validation before a qualification
claim.

The pinned C# reference's standalone unit suite has an upstream ARM64 limitation:
`Depthwise9PackedMatchesReference` unconditionally expects the AVX2-only
`TryPacked9` fast path to accept its input. It returns false on ARM64. The local
reference suite reports 118 passed and that one failure out of 119 tests.
The reference checkout is kept clean at its pinned revision for live OCR
comparisons; this standalone test is separate from the matched OCR gates above.
