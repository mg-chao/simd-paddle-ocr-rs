These little-endian F32 tensors were captured by `tools/reference-exporter` from
SimdPaddleOCR revision `33ff8a81672452041d110e8ba52dcd2639e54228`, with
`PPOCR_NHWC=0`, `SIMD_OCR_NOFUSE=1`, `DOTNET_EnableHWIntrinsic=0`.

Source: 7×9 BGR24, byte `i` is `(i * 17) % 256`, no row padding. Targets:
DET `[1,3,32,32]`; CLS `[1,3,80,160]`; REC `[1,3,48,320]`. The crop quad is
`[[0,0],[6,0],[6,8],[0,8]]`; its `[8,6,3]` byte result is represented as F32
to reuse the capture/replay schema. `tests/pixel_oracle.rs` checks every bit.
`manifest.json` pins each file's SHA-256, shape, original capture case and scalar
configuration; the core test verifies these identities before replay.

Regeneration requires a successful pinned reference capture. Copy only these
four `tiny` records after reviewing a deliberate reference revision change;
do not replace them with the Rust implementation's output.
