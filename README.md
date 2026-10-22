# simd-paddle-ocr-rs

Rust implementation of SimdPaddleOCR, with portable SIMD, runtime CPU dispatch,
and optional native GPU inference. Detection, perspective cropping, optional orientation classification,
CTC recognition and token boxes run without a .NET or ONNX Runtime dependency.

The project uses `nightly-2026-09-08`, selected by `rust-toolchain.toml`.
Windows x64 and native macOS Apple Silicon are supported; see
[CPU usage](docs/cpu-usage.md) and [macOS setup](docs/macos-cpu-validation.md).

## Run OCR

Acquire the supported model files from the pinned upstream checkout, then run
the JPEG/PNG example with your own image:

```powershell
git clone https://github.com/mg-chao/SimdPaddleOCR.git ../SimdPaddleOCR
git -C ../SimdPaddleOCR checkout 33ff8a81672452041d110e8ba52dcd2639e54228
cargo run --release --locked --example ocr -- --models-dir ../SimdPaddleOCR/models --preset tiny input.jpg
```

Use `--preset small` or `medium`, `--no-cls` to omit the classifier, `--json` for
structured output, or `--alignment` for CTC spans and estimated token boxes.
Run `cargo run --release --locked --example ocr -- --help` for all options.

Image decoding belongs to the example; the library accepts borrowed raw pixels.
Models are loaded from local files and are never downloaded automatically.
The [usage guide](docs/cpu-usage.md) lists model filenames, SHA-256 hashes,
input formats, concurrency controls, memory limits, and supported model shapes.
External model assets retain their upstream terms.

## Use the library

A consuming project must select the pinned nightly toolchain too. To use a
local checkout, add this dependency to the application's `Cargo.toml`:

```toml
[dependencies]
simd-paddle-ocr-rs = { path = "../simd-paddle-ocr-rs" }
```

```rust,no_run
use simd_paddle_ocr_rs::{ModelPreset, OcrModelSet};
use simd_paddle_ocr_rs::image::{ImageView, PixelFormat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = OcrModelSet::from_directory("models", ModelPreset::ChineseV6Tiny, true)?;
    let mut request = models.builder().build()?;
    let pixels = vec![255u8; 640 * 480 * 3]; // Replace with decoded RGB pixels.
    let image = ImageView::new(&pixels, 640, 480, 0, PixelFormat::Rgb24)?;
    let result = request.run(image, false)?;
    println!("{}", result.text);
    Ok(())
}
```

The default `parallel` feature enables a shared Rayon worker pool. Disable
default features for serial execution, or enable `profiling` for stage and
recognition-owner diagnostics. Refer to the [usage guide](docs/cpu-usage.md)
for reusable requests and the bounded `OcrEngine` service API.

## GPU runtime

The optional `gpu` feature runs detection, orientation classification, and
recognition on a headless wgpu device through Vulkan, D3D12, or Metal. Select
GPU execution explicitly; the default remains CPU. Image processing and CTC
text assembly run on the host. GPU intermediates stay on the device, and
recognition reads back compact class/score results.

```powershell
cargo run --release --locked --features gpu --example ocr -- --device gpu --gpu-backend vulkan --models-dir ../SimdPaddleOCR/models --preset tiny input.jpg
```

See [GPU usage](docs/gpu-usage.md) for reusable sessions, memory limits,
CPU/GPU API migration, numerical contracts, and local validation.

## Local checks

Source unit tests use synthetic inputs and do not need model downloads or .NET.
Run checks locally on the physical machine:

```powershell
cargo test --release --locked --lib
cargo test --release --locked --no-default-features --lib
cargo build --release --locked --example ocr
```

## License

Apache-2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE), and the included
[third-party license texts](licenses/README.md).
