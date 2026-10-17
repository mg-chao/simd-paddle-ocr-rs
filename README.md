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
