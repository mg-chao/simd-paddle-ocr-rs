# simd-paddle-ocr-rs

Rust CPU implementation of SimdPaddleOCR, with portable SIMD and runtime CPU
dispatch. Detection, perspective cropping, optional orientation classification,
CTC recognition and token boxes run without a .NET or ONNX Runtime dependency.
The pinned `nightly-2026-09-08` toolchain is selected by `rust-toolchain.toml`.

For native Apple Silicon setup, OCR commands, and local correctness checks, see
[macOS CPU support](docs/macos-cpu-validation.md). Automatic dispatch uses the
ARM64 NEON/FMA backend; no Rosetta or GPU runtime is required. The portable
four-lane policy remains available for comparison.

## Run OCR

Acquire the shipped model files from the pinned reference checkout, then run
the JPEG/PNG example:

```powershell
git clone https://github.com/mg-chao/SimdPaddleOCR.git ../SimdPaddleOCR
git -C ../SimdPaddleOCR checkout 33ff8a81672452041d110e8ba52dcd2639e54228
cargo run --release --locked --example ocr -- --models-dir ../SimdPaddleOCR/models --preset tiny tests/baseline/corpus/imgs_1.jpg
```

Use `--preset small` or `medium`, `--no-cls` to omit the classifier, `--json` for
structured output, or `--alignment` for CTC spans and estimated token boxes.
Image decoding is development-only; the library accepts borrowed raw pixels.
Models are loaded from local files and are never downloaded automatically.

See [CPU usage](docs/cpu-usage.md) for presets, input formats/stride, borrowed
and owned results, concurrency, memory limits, supported model restrictions,
and using the library from a consuming project.

## Correctness baseline

See [the maintained CPU baseline guide](docs/cpu-baseline.md) for the pinned
C# reference, model/dictionary SHA-256 hashes, 22 public test fixtures, saved
tiny/small/medium predictions, and release verification commands.

```powershell
cargo build --release --locked --examples
cargo test --release --locked
cargo run --release --locked --example cpu_baseline -- verify
```

The live comparison expects the pinned C# checkout and its models in the
sibling `../SimdPaddleOCR` directory. The guide also describes offline
Rust-only verification and intentional baseline regeneration.

The initial qualification target is Windows x64 with the shipped
tiny/small/medium bundles. Correctness and performance have separate gates;
see [configuration qualification](docs/cpu-qualification.md) and the
[matched performance protocol](docs/cpu-performance.md) before relying on a
performance parity claim.

Recorded local checks and completion status are in
[Windows CPU validation](docs/windows-cpu-validation.md).
The [current Windows pipeline analysis](docs/windows-2026-10-18-pipeline-analysis.md)
records stage latency, convolution costs, recognition worker completion times,
and remaining optimization priorities on the physical Ryzen CPU.
The [parallel PadEnd2 follow-up](docs/windows-avx2-pad-end-optimization.md)
records the adopted detector optimization, exact output checks, and bounded
before/after results, including the limits of whole-page timing gains.
The [Medium slowdown investigation](docs/windows-medium-regression-investigation.md)
records an alternating-version comparison that did not reproduce the reported
10.6% slowdown and retained the Tiny improvement.

The benchmark also supports macOS with native CPU-time and peak-resident-memory
counters. See the [focused Apple M4 comparison](docs/macos-cpu-performance.md)
for the original measurements and the
[ARM64 optimization results](docs/macos-arm64-optimization.md) for NEON kernels,
Rust/C# latency, memory, and validation scope. The subsequent
[recognition scheduling results](docs/macos-arm64-recognition-optimization.md)
record the current dense-page latency improvement and its memory tradeoffs.
The [narrow convolution results](docs/macos-arm64-narrow-convolution-optimization.md)
cover the subsequent Tiny/Small latency improvements, numerical validation,
and measured Medium and memory limitations.
The [Medium mixed-width follow-up](docs/macos-arm64-medium-mixed-regression.md)
addresses the long-crop recognition bottleneck while retaining the Tiny/Small
gains, with a fresh three-version comparison and owner timing evidence.
The [pre-SME pipeline latency analysis](docs/macos-current-pipeline-analysis.md)
measures each production stage after those changes and identifies convolution
optimization opportunities using local benchmarks and operator profiles.
The [SME recognition optimization](docs/macos-arm64-sme-optimization.md)
adds guarded matrix kernels on supported macOS ARM64 hardware, with measured
Small/Medium page gains, memory costs, and the full correctness results.
The [SME pipeline analysis](docs/macos-sme-pipeline-analysis.md)
connects page and stage timings to fused-operator probes,
recognition worker completion times, and the remaining optimization priorities.
The [SME tile and crossover follow-up](docs/macos-arm64-sme-tile-optimization.md)
records further Small/Medium gains, unchanged retained memory, exact output
checks, and the measured Tiny timing limitations.
The [pipeline analysis after SME tiling](docs/macos-sme-tile-pipeline-analysis.md)
details all stages after those changes, identifies the remaining convolution
and recognition scheduling targets, and preserves the local benchmark evidence.
The [SME packing and crossover optimization](docs/macos-arm64-sme-packing-optimization.md)
extends acceleration to smaller pointwise layers, with measured page latency
reductions, exact output checks, and a documented Tiny workspace tradeoff.
The [pipeline analysis after SME packing](docs/macos-sme-packing-pipeline-analysis.md)
details the current stage latencies, fused-operator costs, recognition worker
imbalance, and remaining optimization priorities on the physical Apple M4.
The [ARM64 fused-store optimization](docs/macos-arm64-panel-store-optimization.md)
reduces repeated activation and residual work in NEON/SME convolution stores,
with measured page latency improvements and unchanged retained memory.
The [pipeline analysis after fused stores](docs/macos-panel-store-pipeline-analysis.md)
records current stage latencies, remaining convolution costs, recognition worker
imbalance, and workload-specific optimization priorities on the physical Apple M4.
The [spatial SME optimization](docs/macos-arm64-spatial-sme-optimization.md)
accelerates eligible 3×3 convolutions with bounded reusable scratch, with measured
sparse-page gains, exact output validation, and documented timing and memory limits.
The [pipeline analysis after spatial SME](docs/macos-spatial-sme-pipeline-analysis.md)
details current stage latencies, remaining operator costs, recognition worker
imbalance, and the next optimization priorities on the physical Apple M4.
The [streaming and spatial optimization](docs/macos-arm64-streaming-spatial-optimization.md)
pairs SME tiles, extends spatial kernels and adds direct transpose stores, with
measured Medium and Small dense/mixed gains and unchanged retained memory.

```powershell
cargo run --release --locked --example cpu_qualify --
cargo run --release --locked --example cpu_benchmark -- --model tiny --threads 8 --fixture imgs_1 --workload latency --budget-seconds 60
```

The qualification runner compares 213 requests over 16 configurations with the
pinned CPU reference. The benchmark defaults to a 60-second execution budget
per implementation; the full 36-comparison matrix requires an explicitly
larger bounded budget. Focused passes do not establish full qualification.

## License

Apache-2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE), and the included
[third-party license texts](licenses/README.md). External model assets retain
their upstream terms.
