# simd-paddle-ocr-rs

Rust CPU inference for SimdPaddleOCR's tiny, small and medium Chinese OCR models
and shared text-line orientation model. Development and validation target
**Windows x64**, using the pinned `nightly-2026-09-08` toolchain.

The library implements ONNX import and normalization, scalar and SIMD execution,
NHWC planning, graph fusion, compact CTC decoding, and the complete raw-pixel OCR
pipeline. Models and compiled weights are shareable; mutable sessions and OCR
requests own reusable storage. This is the operator subset needed by the shipped
models, rather than a general-purpose ONNX runtime.

## Raw pixels and execution

Image APIs accept BGR24, RGB24, BGRA32 and RGBA32, including padded row strides.
Callers decode files and retain the pixel buffer until inference returns.

Use `OcrModelSet` and its builder for complete OCR, or `Detector`, `Classifier`,
`Recognizer`, `OrientedLineExtractor` and `Ocr` for
individual stages or full OCR. Borrowed results last until the next mutable use;
`try_to_owned()` copies results when they must escape the request. `OcrEngine`
provides bounded, reusable request leases for concurrent callers.

```rust,no_run
use std::sync::Arc;
use simd_paddle_ocr_rs::{ModelPreset, OcrModelSet, cpu::CpuContext,
    image::{ImageView, PixelFormat}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = OcrModelSet::from_directory("models", ModelPreset::ChineseV6Tiny, true)?;
    let mut ocr = models.builder().cpu_context(Arc::new(CpuContext::new(4)?))
        .recognition_batch_size(1).build()?;
    let pixels = std::fs::read("image.bgr")?; // decoded 320x320 BGR24
    let image = ImageView::new(&pixels, 320, 320, 0, PixelFormat::Bgr24)?;
    let result = ocr.run(image, true)?; // capture CTC token alignment
    println!("{}", result.text);
    for line in result.iter() {
        for token in line.estimate_character_boxes()? {
            println!("{} {:?}", token.text, token.points);
        }
    }
    Ok(())
}
```

`CompileOptions::reference()` selects scalar, NCHW, unfused execution.
`CompileOptions::default()` enables optimized execution. The default `parallel`
feature uses a bounded CPU context shared by nested work; `--no-default-features`
retains serial execution. See [API and memory contracts](docs/migration.md).

`OcrModelSet::from_paths` and `from_readers` support custom sources and optional
classification. Presets pair tiny with `ppocr_keys.txt` and small/medium with
`rec_keys.txt`. `builder.build_engine(limits)` creates a bounded request pool.
Same-width recognition batching defaults to one; configure it through the
builder or `Ocr::set_recognition_batch_size`. Packed UTF-8 text and its
reference-compatible FNV-1a hash are available on borrowed and owned OCR results.

Run the raw-pixel example without an image decoder:

```powershell
cargo run --release --example ocr -- ../SimdPaddleOCR/models tiny tools/corpus-source/sample.bgr 320 320 4 1
```

## Correctness validation

Keep a clean checkout of the pinned C# reference alongside this repository and
provide all seven models and both dictionaries explicitly:

```powershell
cargo test --locked --all-features --all-targets
cargo test --locked --no-default-features
cargo clippy --locked --all-targets --all-features -- -D warnings
python -m unittest discover -s tools -p "test_cpu_*.py" -v
python tools/cpu_parity.py --reference-dir ../SimdPaddleOCR --models-dir ../SimdPaddleOCR/models --quick --threads 1 --correctness-only
```

The maintained harness pins reference revision, model bytes, dictionaries and raw
pixels. It compares independent C# scalar/optimized captures with Rust execution,
including logical node tensors, public heads, CTC spans and complete OCR. Missing
assets, omitted results, nonfinite values and mismatches fail. Large captures stay
under ignored `target/` storage. See [validation commands](docs/cpu-validation.md)
and the [CPU optimization coverage matrix](docs/cpu-coverage.md).

## Completion gates

CPU completion requires all-model parity, public feature completeness, every
active Windows x64 reference optimization family, real AVX-512 hardware checks,
and warmed OCR latency within 10% of C# at matched CPU budgets. Coverage and
measurements must be recorded separately; compile-only checks do not establish
hardware validation or performance. GPU, ARM64, WASM and image codecs are outside
the current scope. Disabled reference INT8/VNNI experiments are excluded.

The source and adapted algorithms are licensed under Apache-2.0, with MIT
components identified in [NOTICE](NOTICE) and `licenses/`.
