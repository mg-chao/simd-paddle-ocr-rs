# simd-paddle-ocr-rs

A safe Rust library for PaddleOCR inference. It loads ONNX models and runs text
detection, orientation classification, and recognition directly from borrowed
image pixels. Tiny, small, and medium model families use serial F32/NCHW kernels.
Explicit SIMD, threading, and GPU execution are not implemented.

## Usage

Use the pinned Rust 1.97.1 toolchain. Add this crate as a local dependency:

```toml
[dependencies]
simd-paddle-ocr-rs = { path = "../simd-paddle-ocr-rs" }
```

Supply compatible model files and a UTF-8 recognition dictionary separately:

| Family | Detector | Recognizer | Dictionary |
| --- | --- | --- | --- |
| tiny | `det.onnx` | `rec.onnx` | `ppocr_keys.txt` |
| small | `small_det.onnx` | `small_rec.onnx` | `rec_keys.txt` |
| medium | `medium_det.onnx` | `medium_rec.onnx` | `rec_keys.txt` |

All families share `cls.onnx` for optional orientation classification. These are
the model filenames from the upstream
[SimdPaddleOCR project](https://github.com/sdcb/SimdPaddleOCR). The importer and
runtime support the operators used by these networks.

```rust,no_run
use simd_paddle_ocr_rs::{
    detector::{Detector, DetectorOptions},
    image::{ImageView, PixelFormat},
    lines::{LineOptions, OrientedLineExtractor},
    model::Model,
    ocr::{Ocr, OcrOptions},
    recognizer::{Dictionary, Recognizer, RecognizerOptions},
    runtime::{CompiledModel, ExecutionLimits},
};
use std::{error::Error, fs, sync::Arc};

fn compile(path: &str) -> Result<Arc<CompiledModel>, Box<dyn Error>> {
    let model = Arc::new(Model::load_path(path)?);
    Ok(Arc::new(CompiledModel::new(model)?))
}

fn main() -> Result<(), Box<dyn Error>> {
    let detector = Detector::new(
        compile("models/det.onnx")?,
        DetectorOptions::default(),
        ExecutionLimits::default(),
    )?;
    let extractor = OrientedLineExtractor::new(
        detector,
        Some(compile("models/cls.onnx")?),
        LineOptions::default(),
    )?;
    let dictionary = Arc::new(Dictionary::from_utf8(&fs::read("models/ppocr_keys.txt")?)?);
    let recognizer = Recognizer::new(
        compile("models/rec.onnx")?,
        dictionary,
        RecognizerOptions::default(),
        ExecutionLimits::default(),
    )?;
    let mut ocr = Ocr::new(extractor, recognizer, OcrOptions::default())?;

    // Supply decoded, tightly packed BGR pixels for a 640 x 480 image.
    let pixels = fs::read("image.bgr")?;
    let image = ImageView::new(&pixels, 640, 480, 0, PixelFormat::Bgr24)?;
    let result = ocr.run(image, true)?;
    for line in result.iter() {
        println!("{} ({:.3})", line.text, line.recognition_score);
    }
    Ok(())
}
```

Image decoding belongs to the caller. `ImageView` accepts BGR24, RGB24, BGRA32,
and RGBA32 with an explicit byte stride; zero means tightly packed rows. Pass
`None` as the classifier to skip orientation classification.

## API and memory

- `Model` imports and normalizes ONNX graphs with checked load limits.
- `CompiledModel` and `Session` execute graphs with reusable activation storage.
- `Detector`, `OrientedLineExtractor`, and `Recognizer` can run independently.
- `Ocr` returns ordered text, quadrilateral boxes, scores, rotations, and optional
  CTC spans. The second argument to `run` controls span capture.

Models, compiled models, and dictionaries can be shared through `Arc`. Mutable
engines reuse their workspaces. Results borrow engine storage; call
`try_to_owned()` when results need to survive another inference call.

Recognition uses height 48 and adaptive widths rounded up to a multiple of 32.
The default activation budget is 1 GiB per session, excluding shared weights,
retained workspaces, and owned results. Configure `ExecutionLimits` and the
engine options for application-specific limits.

Synchronous trace methods expose intermediate stages. The optional `profiling`
feature adds `Recognizer::recognize_profiled`, `Ocr::run_profiled`, and
`StageProfile`; ordinary inference does not read profiling timers.

## Development

```sh
cargo build --locked
cargo fmt --all -- --check
cargo test --locked --all-features
cargo clippy --locked --all-targets --all-features -- -D warnings
```

The Rust tests use synthetic models and inputs and require no downloaded models
or external runtimes. CI runs these checks on Windows and Linux.

## License

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [licenses](licenses/) for
upstream attribution and third-party license terms.
