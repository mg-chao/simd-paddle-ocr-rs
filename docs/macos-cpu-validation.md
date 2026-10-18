# macOS CPU support

Build natively on Apple Silicon with the pinned `nightly-2026-09-08` toolchain.
Automatic dispatch selects the ARM64 NEON/FMA backend; supported hardware can
also use guarded SME kernels. Rosetta and a GPU runtime are unnecessary.

## Setup and OCR

Install the Xcode command-line tools and Rust through rustup, then select the
project toolchain:

```sh
xcode-select --install
rustup toolchain install nightly-2026-09-08 --profile minimal --component rustfmt --component clippy
```

From the repository root, acquire the supported external model files and run
the OCR example with your own JPEG/PNG:

```sh
git clone https://github.com/mg-chao/SimdPaddleOCR.git ../SimdPaddleOCR
git -C ../SimdPaddleOCR checkout 33ff8a81672452041d110e8ba52dcd2639e54228
cargo run --release --locked --example ocr -- \
  --models-dir ../SimdPaddleOCR/models --preset tiny input.jpg
```

Cloning the models does not require building C# or installing .NET. See
[CPU usage](cpu-usage.md) for model hashes, presets, pixel formats, and request
capacity limits. Add `--alignment` for JSON output with CTC spans and token boxes.

## Local checks

Run source unit tests on a physical Mac, with release builds:

```sh
cargo test --release --locked --lib
cargo test --release --locked --no-default-features --lib
cargo build --release --locked --example ocr
```

These checks use synthetic inputs. Validate OCR accuracy with your deployment
models and images on the target machine; results from Windows do not establish
macOS accuracy or performance. `KernelPolicy::Portable4` is available for
numerical comparisons, and forcing an unsupported backend returns an error.
